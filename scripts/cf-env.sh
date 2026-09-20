#!/usr/bin/env bash
# Use the caller's Python environment and CUDA installation.
CF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CF_ENV:-}" ]]; then
    export PATH="$CF_ENV/bin:$PATH"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$CF_ROOT/src:$CF_ROOT/qwen-vl-utils/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SAMPLE_MODE=true
export WANDB_MODE="${WANDB_MODE:-offline}"
export CF_MODEL="${CF_MODEL:-}"
export CF_TRAIN_DATA="${CF_TRAIN_DATA:-}"
export CF_VAL_DATA="${CF_VAL_DATA:-}"
CF_LAUNCH=()
if command -v taskset >/dev/null 2>&1; then
    CF_CPUSET="${CF_CPUSET:-$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:8])) )')}"
    CF_LAUNCH=(taskset --cpu-list "$CF_CPUSET")
fi
cd "$CF_ROOT"
