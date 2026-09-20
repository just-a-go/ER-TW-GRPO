#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cf-env.sh"
: "${CF_MODEL:?Set CF_MODEL to a local Qwen2.5-VL checkpoint directory}"
: "${CF_TRAIN_DATA:?Set CF_TRAIN_DATA to your training JSON or JSONL file}"
CF_LAMBDA="${CF_LAMBDA:-0.1}"
export PRIVATE_DATA_ROOT="$CF_ROOT/outputs"
export MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B-Instruct_clevrer_cf_twgrpo_alpha1p7_lam${CF_LAMBDA}_round1}"
export WANDB_PROJECT=Qwen2.5-VL-7B-Video-GRPO WANDB_NAME="$MODEL_NAME" DEBUG_MODE=true
mkdir -p "$PRIVATE_DATA_ROOT"
if [[ -e "$PRIVATE_DATA_ROOT/$MODEL_NAME" ]]; then
    echo "Output already exists; choose a new MODEL_NAME: $PRIVATE_DATA_ROOT/$MODEL_NAME" >&2
    exit 1
fi
mkdir "$PRIVATE_DATA_ROOT/$MODEL_NAME"
export LOG_PATH="$PRIVATE_DATA_ROOT/$MODEL_NAME/debug.log"
"${CF_LAUNCH[@]}" torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-12542}" \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir "$PRIVATE_DATA_ROOT/$MODEL_NAME" \
    --model_name_or_path "$CF_MODEL" --dataset_name xxx --jsonl_path "$CF_TRAIN_DATA" \
    --max_prompt_length 4096 --max_completion_length 4096 \
    --reward_funcs accuracy format --learning_rate 1e-6 --beta 0.00 --alpha 1.70 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --logging_steps 1 --question_type mixed --bf16 --torch_dtype bfloat16 \
    --data_seed 42 --report_to wandb --gradient_checkpointing true \
    --attn_implementation flash_attention_2 --freeze_vision_modules true \
    --loss_type tw_grpo --num_train_epochs 1 --run_name "$WANDB_NAME" \
    --save_steps 100 --max_grad_norm 20 --save_only_model true --num_generations 8 \
    --dataloader_num_workers 0 \
    --cf_local_data "${CF_LOCAL_DATA:-$CF_ROOT/artifacts/round1/local.jsonl}" \
    --cf_lambda "$CF_LAMBDA" --cf_local_batch_size "${CF_LOCAL_BATCH_SIZE:-1}" \
    "$@" 2>&1 | tee -a "$LOG_PATH"
