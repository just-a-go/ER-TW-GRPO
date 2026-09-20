"""Ordinary complete-response SFT after CF + TW-GRPO training.

Run with ``python -m open_r1.cf.distill`` and Transformers TrainingArguments.
Only collector ``kind=distill`` records with unit weights are accepted.
"""

from dataclasses import dataclass, field

import torch
from transformers import (
    AutoConfig, AutoProcessor, HfArgumentParser, Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments,
)

from open_r1.cf.supervision import EvidenceCollator, RepairDataset, weighted_sequence_nll


@dataclass
class DistillArguments:
    model_name_or_path: str = field(metadata={"help": "Local main-training checkpoint"})
    distill_data: str = field(metadata={"help": "Collector complete-response JSONL (kind=distill)"})
    freeze_vision_modules: bool = True
    attn_implementation: str = "flash_attention_2"
    max_pixels: int = 262144
    min_pixels: int = 3136
    max_length: int = 16384


class RepairDistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        inputs.pop("weights")
        outputs = model(**inputs, use_cache=False)
        # Unit weights are constructed here even if caller-supplied metadata differs.
        weights = torch.ones(labels.shape[0], device=labels.device)
        loss = weighted_sequence_nll(outputs.logits, labels, weights)
        return (loss, outputs) if return_outputs else loss


def main():
    model_args, training_args = HfArgumentParser((DistillArguments, TrainingArguments)).parse_args_into_dataclasses()
    if training_args.dataloader_num_workers != 0:
        raise ValueError("distillation requires dataloader_num_workers=0 for multimodal collation")
    training_args.remove_unused_columns = False
    dataset = RepairDataset(model_args.distill_data, kind="distill")
    if not dataset:
        raise ValueError("distillation set is empty; collect accepted complete repairs first")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    processor.image_processor.max_pixels = model_args.max_pixels
    processor.image_processor.min_pixels = model_args.min_pixels
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    classes = {
        "qwen2_vl": Qwen2VLForConditionalGeneration,
        "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
    }
    if config.model_type not in classes:
        raise ValueError(f"unsupported distillation model type {config.model_type!r}")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16 if training_args.fp16 else torch.float32
    model = classes[config.model_type].from_pretrained(
        model_args.model_name_or_path, local_files_only=True, config=config,
        torch_dtype=dtype, attn_implementation=model_args.attn_implementation,
    )
    model.config.use_cache = False
    if model_args.freeze_vision_modules:
        for name, parameter in model.named_parameters():
            if "visual" in name:
                parameter.requires_grad = False
    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = RepairDistillationTrainer(
        model=model, args=training_args, train_dataset=dataset, processing_class=processor,
        data_collator=EvidenceCollator(processor, max_length=model_args.max_length),
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
