#!/usr/bin/env bash
set -euo pipefail
: "${ER_MODEL:?Set ER_MODEL to the completed main-training checkpoint}"
source "$(dirname "${BASH_SOURCE[0]}")/er-env.sh"
# ER_MODEL must point to the completed main-training checkpoint for the final stage.
ER_DISTILL_OUTPUT="${ER_DISTILL_OUTPUT:-$ER_ROOT/outputs/Qwen2.5-VL-er-distilled}"
if [[ -e "$ER_DISTILL_OUTPUT" ]]; then
    echo "Output already exists; choose a new ER_DISTILL_OUTPUT: $ER_DISTILL_OUTPUT" >&2
    exit 1
fi
"${ER_LAUNCH[@]}" torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-12543}" \
    -m open_r1.er.distill --deepspeed scripts/zero3_offload.json \
    --model_name_or_path "$ER_MODEL" \
    --distill_data "${ER_DISTILL_DATA:-$ER_ROOT/artifacts/round1/distill.jsonl}" \
    --output_dir "$ER_DISTILL_OUTPUT" \
    --learning_rate 1e-6 --num_train_epochs 1 --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 --bf16 true --gradient_checkpointing true \
    --freeze_vision_modules true --attn_implementation flash_attention_2 \
    --dataloader_num_workers 0 --report_to none --logging_steps 1 --save_steps 100 \
    "$@"
