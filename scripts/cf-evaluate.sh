#!/usr/bin/env bash
set -euo pipefail
: "${CF_MODEL:?Set CF_MODEL to the checkpoint to evaluate}"
source "$(dirname "${BASH_SOURCE[0]}")/cf-env.sh"
"${CF_LAUNCH[@]}" python -m open_r1.cf.evaluate \
    --model "$CF_MODEL" --dataset "$CF_VAL_DATA" \
    --output "${CF_EVAL_OUTPUT:-$CF_ROOT/outputs/evaluation/$(basename "$CF_MODEL")_eval.json}" \
    --batch-size "${CF_EVAL_BATCH_SIZE:-8}" "$@"
