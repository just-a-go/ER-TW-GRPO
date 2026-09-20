#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/cf-env.sh"
: "${CF_MODEL:?Set CF_MODEL to a local Qwen2.5-VL checkpoint directory}"
: "${CF_TRAIN_DATA:?Set CF_TRAIN_DATA to your training JSON or JSONL file}"
"${CF_LAUNCH[@]}" python -m open_r1.cf.collect \
    --model "$CF_MODEL" \
    --dataset "$CF_TRAIN_DATA" \
    --output-dir "${CF_COLLECTION:-$CF_ROOT/artifacts/round1}" \
    --max-questions "${CF_MAX_QUESTIONS:-2000}" \
    --candidates "${CF_CANDIDATES:-4}" \
    --transfers-per-question "${CF_TRANSFERS:-8}" \
    --node-max-tokens "${CF_NODE_TOKENS:-256}" \
    --total-generation-tokens "${CF_GENERATION_TOKENS:-4000000}" \
    "$@"
