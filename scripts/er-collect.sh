#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/er-env.sh"
: "${ER_MODEL:?Set ER_MODEL to a local Qwen2.5-VL checkpoint directory}"
: "${ER_TRAIN_DATA:?Set ER_TRAIN_DATA to your training JSON or JSONL file}"
"${ER_LAUNCH[@]}" python -m open_r1.er.collect \
    --model "$ER_MODEL" \
    --dataset "$ER_TRAIN_DATA" \
    --output-dir "${ER_COLLECTION:-$ER_ROOT/artifacts/round1}" \
    --max-questions "${ER_MAX_QUESTIONS:-2000}" \
    --candidates "${ER_CANDIDATES:-4}" \
    --transfers-per-question "${ER_TRANSFERS:-8}" \
    --node-max-tokens "${ER_NODE_TOKENS:-256}" \
    --total-generation-tokens "${ER_GENERATION_TOKENS:-4000000}" \
    "$@"
