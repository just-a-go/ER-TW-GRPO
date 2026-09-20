# ER-TW-GRPO

Core implementation for **Reinforcing Video Reasoning with Focused Thinking and Evidence Repair**.

The method combines token-weighted GRPO, multi-answer soft rewards, and evidence repair. Failed candidates exchange aligned observations under a fixed receiver graph. Accepted repairs provide gain-weighted local evidence targets and complete responses for subsequent unweighted distillation. Inference uses one student response without graph replay.

## Code

| Component | Location |
|---|---|
| Token weighting and GRPO trainer | `src/open_r1/trainer/grpo_trainer.py` |
| Rewards and main training entry | `src/open_r1/grpo.py` |
| Evidence slots, graph execution and matched replay | `src/open_r1/cf/core.py` |
| Frozen collection and support screening | `src/open_r1/cf/collect.py`, `backend.py` |
| Local supervision and combined training loss | `src/open_r1/cf/supervision.py`, `src/open_r1/trainer/cf_trainer.py` |
| Repair distillation | `src/open_r1/cf/distill.py` |
| CLEVRER evaluation | `src/open_r1/cf/evaluate.py`, `src/eval/eval_clevrer.py` |
| Conference QAI conversion utilities | `data/question_answer_inverse/` |

Internal `cf` names are retained from development. They implement the evidence-repair extension.

## Installation

Use Linux, Python 3.10, and a CUDA environment compatible with PyTorch 2.5.1. The development runtime used Transformers 4.50.0, TRL 0.15.2, Accelerate 1.2.1, DeepSpeed 0.15.4, and FlashAttention 2.7.4.post1.

```bash
git clone https://github.com/just-a-go/ER-TW-GRPO.git
cd ER-TW-GRPO
# Install CUDA-compatible PyTorch 2.5.1 and torchvision 0.20.1 first.
pip install -e '.[dev]' transformers==4.50.0 accelerate==1.2.1 wandb pillow
pip install -e './qwen-vl-utils[decord]'
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

The vendored video reader is retained to keep preprocessing consistent with TW-GRPO. Launchers add both source directories to `PYTHONPATH`.

## Data and model

Download the model and datasets separately. The frozen backend loads an existing local checkpoint offline. Use a Qwen2.5-VL-7B checkpoint, with `Qwen2.5-VL` in its directory name for compatibility with the inherited trainer and evaluator.

Training annotations use JSON or JSONL records with `problem`, `video`, and `solution` fields. Use absolute video paths shared by collection and training. An illustrative record is:

```json
{"problem": "Which statements are true? A. ... B. ... C. ...", "video": "/path/to/video.mp4", "solution": "<answer>A,B</answer>"}
```

Set local paths before running:

```bash
export CF_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export CF_TRAIN_DATA=/path/to/train.json
export CF_VAL_DATA=/path/to/validation.json
export CF_COLLECTION="$PWD/artifacts/round1"
export CF_LOCAL_DATA="$CF_COLLECTION/local.jsonl"
export CF_DISTILL_DATA="$CF_COLLECTION/distill.jsonl"
```

## Run

```bash
# Collect once from the frozen initial checkpoint.
CUDA_VISIBLE_DEVICES=0 bash scripts/cf-collect.sh

# Main training: TW-GRPO plus local evidence supervision, two GPUs.
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cf-tw-grpo.sh

# Distill accepted complete responses from the completed main checkpoint.
CF_MODEL=/path/to/Qwen2.5-VL-main-checkpoint \
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/cf-distill.sh

# Evaluate the final checkpoint with one response per question.
CF_MODEL=/path/to/Qwen2.5-VL-cf-distilled \
  CUDA_VISIBLE_DEVICES=0 bash scripts/cf-evaluate.sh
```

Collection writes `local.jsonl`, `distill.jsonl`, a manifest, and audit/support files. Keep the collection fixed across component comparisons. Choose a new collection directory and output name for a new run. `CF_LAMBDA=0` disables local learning and selects the inherited TW-GRPO trainer. Launchers accept additional CLI arguments.

The launchers retain development defaults (including a 2,000-question mixed subset and epoch-based training). They are starting configurations, not a claim to reproduce the paper's 1,000-question, 500-step experiments. Select the intended training subset in advance and set collection and training sample limits consistently. No model weights, datasets, or experiment predictions are included.

The QAI utilities use heuristic text transformations. Review the resulting question semantics and labels before training; native CLEVRER multi-answer data needs no QAI conversion.

## Tests

```bash
PYTHONPATH=src:qwen-vl-utils/src python -m pytest tests -q
```

Tests cover replay equivalence, input isolation, screening and budget limits, deduplication, evidence masks, weighted losses, and trainer integration. They do not require model downloads. The latest local check passed 102 tests; three integration tests could not run because the local machine lacks Transformers and the training dependencies. Full GPU training was not rerun for this release.

## Attribution

Built on [TW-GRPO](https://github.com/longmalongma/TW-GRPO) at commit `aa529ffca7ae408e4f3df7ba71cb0dc58544c6b7`, with inherited Open R1 / Hugging Face components and Qwen video utilities. Original copyright headers and the Apache-2.0 license are retained. This release adds evidence-repair collection, local learning, distillation, tests, and portable launch configuration.
