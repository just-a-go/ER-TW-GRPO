"""A separate auxiliary stream; the inherited TW-GRPO implementation is unchanged."""

import json
import math
from pathlib import Path

import torch.distributed as distributed
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from open_r1.er.supervision import (
    EvidenceCollator, RepairDataset, distributed_stream_indices, weighted_sequence_nll,
)
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class ERQwen2VLGRPOTrainer(Qwen2VLGRPOTrainer):
    def __init__(self, *args, er_local_data=None, er_lambda=0.0, er_local_batch_size=1, **kwargs):
        if not math.isfinite(er_lambda) or er_lambda < 0:
            raise ValueError("er_lambda must be finite and nonnegative")
        if er_local_batch_size < 1:
            raise ValueError("er_local_batch_size must be positive")
        self.er_lambda = er_lambda
        self.er_local_batch_size = er_local_batch_size
        self.er_microstep = 0
        self.er_dataset = None
        if er_lambda > 0:
            if not er_local_data:
                raise ValueError("er_lambda > 0 requires er_local_data")
            self.er_dataset = RepairDataset(er_local_data, kind="local")
        super().__init__(*args, **kwargs)
        if self.er_dataset and self.args.dataloader_num_workers != 0:
            raise ValueError("ER training requires dataloader_num_workers=0")
        if self.er_dataset and self.args.gradient_checkpointing:
            checkpointing = dict(self.args.gradient_checkpointing_kwargs or {})
            if checkpointing.get("use_reentrant") is True:
                raise ValueError("ER uses two differentiable forwards; gradient checkpointing requires use_reentrant=False")
            checkpointing["use_reentrant"] = False
            self.args.gradient_checkpointing_kwargs = checkpointing
        if self.er_dataset is not None and distributed.is_initialized():
            fingerprints = [None] * distributed.get_world_size()
            distributed.all_gather_object(fingerprints, self.er_dataset.fingerprint)
            if len(set(fingerprints)) != 1:
                raise ValueError("all ranks must load the identical accepted-example file")
        # Preserve the parent's gradient-accumulation loss scaling.
        self.model_accepts_loss_kwargs = False
        self.er_collator = EvidenceCollator(self.processing_class) if self.er_dataset else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base_loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
        )
        if not self.er_dataset or self.er_lambda == 0 or not model.training:
            return base_loss
        indices = distributed_stream_indices(
            len(self.er_dataset), self.er_microstep, self.er_local_batch_size,
            self.accelerator.process_index, self.accelerator.num_processes, self.args.seed,
        )
        self.er_microstep += 1
        auxiliary = self.er_collator([self.er_dataset[index] for index in indices])
        # The parent intentionally leaves ordinary GRPO inputs unprepared.
        auxiliary = Trainer._prepare_inputs(self, auxiliary)
        labels, weights = auxiliary.pop("labels"), auxiliary.pop("weights")
        # Exactly one auxiliary model call per active microstep on every rank,
        # including when the accepted set is smaller than the global batch.
        outputs = model(**auxiliary, use_cache=False)
        local_loss = weighted_sequence_nll(outputs.logits, labels, weights)
        self._metrics["er/local_loss"].append(
            self.accelerator.gather_for_metrics(local_loss.detach().reshape(1)).mean().item()
        )
        self._metrics["er/mean_gain"].append(
            self.accelerator.gather_for_metrics(weights.detach()).mean().item()
        )
        return base_loss + self.er_lambda * local_loss

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        if self.args.should_save and self.er_dataset:
            checkpoint = Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            checkpoint.joinpath("er_stream_state.json").write_text(
                json.dumps({"microstep": self.er_microstep, "fingerprint": self.er_dataset.fingerprint,
                            "world_size": self.accelerator.num_processes,
                            "batch_size": self.er_local_batch_size, "seed": self.args.seed}),
                encoding="utf-8",
            )

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        # DeepSpeed restores its own checkpoint without Trainer._load_from_checkpoint,
        # so auxiliary stream restoration belongs at this shared entry point.
        checkpoint = resume_from_checkpoint
        if checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
        if checkpoint and self.er_dataset:
            state_path = Path(checkpoint) / "er_stream_state.json"
            if not state_path.exists():
                raise ValueError("ER resume requires er_stream_state.json; start a new run from a baseline checkpoint")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            expected = dict(fingerprint=self.er_dataset.fingerprint, world_size=self.accelerator.num_processes,
                            batch_size=self.er_local_batch_size, seed=self.args.seed)
            if any(state.get(key) != value for key, value in expected.items()):
                raise ValueError("accepted data or auxiliary sampling configuration changed while resuming")
            self.er_microstep = int(state["microstep"])
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)
