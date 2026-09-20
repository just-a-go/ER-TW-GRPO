"""A separate auxiliary stream; the inherited TW-GRPO implementation is unchanged."""

import json
import math
from pathlib import Path

import torch.distributed as distributed
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from open_r1.cf.supervision import (
    EvidenceCollator, RepairDataset, distributed_stream_indices, weighted_sequence_nll,
)
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class CFQwen2VLGRPOTrainer(Qwen2VLGRPOTrainer):
    def __init__(self, *args, cf_local_data=None, cf_lambda=0.0, cf_local_batch_size=1, **kwargs):
        if not math.isfinite(cf_lambda) or cf_lambda < 0:
            raise ValueError("cf_lambda must be finite and nonnegative")
        if cf_local_batch_size < 1:
            raise ValueError("cf_local_batch_size must be positive")
        self.cf_lambda = cf_lambda
        self.cf_local_batch_size = cf_local_batch_size
        self.cf_microstep = 0
        self.cf_dataset = None
        if cf_lambda > 0:
            if not cf_local_data:
                raise ValueError("cf_lambda > 0 requires cf_local_data")
            self.cf_dataset = RepairDataset(cf_local_data, kind="local")
        super().__init__(*args, **kwargs)
        if self.cf_dataset and self.args.dataloader_num_workers != 0:
            raise ValueError("CF training requires dataloader_num_workers=0")
        if self.cf_dataset and self.args.gradient_checkpointing:
            checkpointing = dict(self.args.gradient_checkpointing_kwargs or {})
            if checkpointing.get("use_reentrant") is True:
                raise ValueError("CF uses two differentiable forwards; gradient checkpointing requires use_reentrant=False")
            checkpointing["use_reentrant"] = False
            self.args.gradient_checkpointing_kwargs = checkpointing
        if self.cf_dataset is not None and distributed.is_initialized():
            fingerprints = [None] * distributed.get_world_size()
            distributed.all_gather_object(fingerprints, self.cf_dataset.fingerprint)
            if len(set(fingerprints)) != 1:
                raise ValueError("all ranks must load the identical accepted-example file")
        # Preserve the parent's gradient-accumulation loss scaling.
        self.model_accepts_loss_kwargs = False
        self.cf_collator = EvidenceCollator(self.processing_class) if self.cf_dataset else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
        )
        if not self.cf_dataset or self.cf_lambda == 0 or not model.training:
            return base_loss
        indices = distributed_stream_indices(
            len(self.cf_dataset), self.cf_microstep, self.cf_local_batch_size,
            self.accelerator.process_index, self.accelerator.num_processes, self.args.seed,
        )
        self.cf_microstep += 1
        auxiliary = self.cf_collator([self.cf_dataset[index] for index in indices])
        # The parent intentionally leaves ordinary GRPO inputs unprepared.
        auxiliary = Trainer._prepare_inputs(self, auxiliary)
        labels, weights = auxiliary.pop("labels"), auxiliary.pop("weights")
        # Exactly one auxiliary model call per active microstep on every rank,
        # including when the accepted set is smaller than the global batch.
        outputs = model(**auxiliary, use_cache=False)
        local_loss = weighted_sequence_nll(outputs.logits, labels, weights)
        self._metrics["cf/local_loss"].append(
            self.accelerator.gather_for_metrics(local_loss.detach().reshape(1)).mean().item()
        )
        self._metrics["cf/mean_gain"].append(
            self.accelerator.gather_for_metrics(weights.detach()).mean().item()
        )
        return base_loss + self.cf_lambda * local_loss

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        if self.args.should_save and self.cf_dataset:
            checkpoint = Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            checkpoint.joinpath("cf_stream_state.json").write_text(
                json.dumps({"microstep": self.cf_microstep, "fingerprint": self.cf_dataset.fingerprint,
                            "world_size": self.accelerator.num_processes,
                            "batch_size": self.cf_local_batch_size, "seed": self.args.seed}),
                encoding="utf-8",
            )

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        # DeepSpeed restores its own checkpoint without Trainer._load_from_checkpoint,
        # so auxiliary stream restoration belongs at this shared entry point.
        checkpoint = resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
        if checkpoint and self.cf_dataset:
            state_path = Path(checkpoint) / "cf_stream_state.json"
            if not state_path.exists():
                raise ValueError("CF resume requires cf_stream_state.json; start a new run from a baseline checkpoint")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            expected = dict(fingerprint=self.cf_dataset.fingerprint, world_size=self.accelerator.num_processes,
                            batch_size=self.cf_local_batch_size, seed=self.args.seed)
            if any(state.get(key) != value for key, value in expected.items()):
                raise ValueError("accepted data or auxiliary sampling configuration changed while resuming")
            self.cf_microstep = int(state["microstep"])
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)
