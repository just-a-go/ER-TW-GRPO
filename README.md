<div align="center">

# ER-TW-GRPO

### Reinforcing Video Reasoning with<br>Focused Thinking and Evidence Repair

**Focus on key tokens. Learn from partial answers. Repair failed evidence.**

</div>

## Contents

1. [Method](#method)
2. [Focused thinking](#focused-thinking)
3. [Evidence repair](#evidence-repair)
4. [Results](#results)
5. [Training dynamics](#training-dynamics)
6. [Reasoning examples](#reasoning-examples)
7. [Core code](#core-code)
8. [Quick start](#quick-start)
9. [Acknowledgements](#acknowledgements)

**ECCV → TPAMI extension.** This manuscript extends our ECCV 2026 paper, *Reinforcing Video Reasoning with Focused Thinking*, with evidence repair, local evidence learning, and repair distillation.

![Focused thinking, soft rewards, and evidence repair](assets/teaser.png)

## Method

![Evidence alignment, matched replay, local learning, and distillation](assets/framework.png)

Transfer supported observations between failed responses. Replay affected reasoning. Learn from successful repairs.

## Focused thinking

<details>
<summary>ECCV foundation · TW-GRPO & token visualizations</summary>

TW-GRPO emphasizes object, event, and temporal tokens.

![Frequent tokens at high-weight positions](assets/token-focus.png)

![Token weighting at steps 0 and 500](assets/token-weights.png)

### Positional weights

![Positional weighting across 300 training samples](assets/weight-map.png)

### Score dynamics

![Content-position score dynamics with exponential smoothing](assets/score-dynamics.png)

<sub>Conference implementation scales; not final bounded weights or calibrated KL scores.</sub>

### Token examples

![Highlighted object, event, and answer tokens](assets/token-examples.png)

<sub>Darker red means greater weight, not verified visual correctness.</sub>

</details>

## Evidence repair

Replace one observation; replay only affected reasoning.

![Matched replay and evidence-only supervision](assets/matched-replay.png)

### Repair in action

One repaired observation recovers the complete answer in this schematic example.

![Single-slot evidence repair on a CLEVRER counterfactual question](assets/evidence-repair.png)

## Results

**1,000 training questions · +4.84 pp CLEVRER exact accuracy · Single-response inference**

CLEVRER: **50.35 → 55.19%**, averaged over three seeds.

<sub>All seven main-paper tables are reproduced below. Accuracy (%); pp = percentage points; n/r = unreported.</sub>

### I. Five-benchmark comparison

| Method | Training | CLEVRER | NExT-GQA | MMVU-MC | MVBench | TempCompass |
| :-- | :-- | --: | --: | --: | --: | --: |
| ER-TW-GRPO (ours) | 1K RL + auxiliary | **55.2** | **77.8** | **65.9** | **64.4** | **73.3** |

<details>
<summary>ECCV reference results · 15 settings</summary>

| Method | Training | CLEVRER | NExT-GQA | MMVU-MC | MVBench | TempCompass |
| :-- | :-- | --: | --: | --: | --: | --: |
| LLaMA-VID | n/r | n/r | n/r | n/r | 41.9 | 45.6 |
| VideoLLaMA2 | n/r | n/r | n/r | 44.8 | 54.6 | n/r |
| LongVA-7B | n/r | n/r | n/r | n/r | n/r | 56.9 |
| Video-UTR-7B | n/r | n/r | n/r | n/r | 58.8 | 59.7 |
| Kangaroo-8B | n/r | n/r | n/r | n/r | 61.1 | 62.5 |
| Qwen2.5-VL-7B, zero-shot | n/r | 30.5 | 75.9 | 65.4 | 63.3 | 72.5 |
| Qwen2.5-VL-7B, CoT | n/r | 27.7 | 73.4 | 63.0 | 57.4 | 72.2 |
| Qwen2.5-VL-7B, SFT | 165K SFT | 29.0 | 69.0 | 61.3 | 59.4 | 69.2 |
| Video-R1-Zero | 4K RL | n/r | n/r | 63.8 | 60.4 | 70.9 |
| Video-R1 | 165K SFT + 4K RL | 31.6 | 74.3 | 64.2 | 62.7 | 72.6 |
| VideoRFT | 102K SFT + 8K RL | 12.8 | 74.2 | 64.5 | 61.4 | 72.1 |
| VideoChat-R1 | 18K RL | 29.2 | 76.0 | 64.2 | 63.1 | 72.9 |
| GRPO, fixed reward | 1K RL | 41.1 | 75.2 | 65.1 | 62.8 | 71.9 |
| TW-GRPO, soft reward | 1K RL | 50.4 | 76.1 | 65.8 | 63.3 | **73.3** |
| TW-GRPO, Video-R1 data | 4K RL | 32.0 | 76.3 | 65.6 | 64.2 | **73.3** |

</details>

<sub>ER: seed 42, 1K questions, 500 base RL steps plus auxiliary learning and distillation. References retain their conference settings.</sub>

### II. FFR comparison

| Method | Exact | Soft | Local GPU-h |
| :-- | --: | --: | --: |
| TW-GRPO | 50.88 | 63.08 | **32.25** |
| Full method | **55.20** | **66.90** | 41.39 |
| FFR | 51.80 | 64.10 | 48.73 |

<sub>CLEVRER, seed 42. Adapted FFR; local GPU-h excludes external teacher computation.</sub>

<details>
<summary>III–IV. ECCV ablations · data, rewards & token weighting</summary>

### III. Data construction

#### CLEVRER counterfactual source

| Training format | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| Single-answer | **51.5** | 18.2 | 50.9 | 32.6 | 51.2 | 62.7 |
| Multi-answer | 50.4 | **32.3** | **55.4** | **41.1** | **52.7** | **65.1** |

#### NExT-GQA source

| Training format | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| Single-answer | 48.0 | **12.3** | **45.5** | 30.7 | 45.7 | 64.3 |
| Multi-answer (QAI) | **63.5** | 9.6 | 44.9 | **36.7** | **54.7** | **64.6** |

#### STAR source

| Training format | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| Single-answer | 65.1 | **2.6** | **41.4** | **29.3** | **51.5** | 64.8 |
| Multi-answer (QAI) | **65.9** | 1.5 | 40.8 | 29.1 | **51.5** | **66.2** |

### IV. Training and reward design

#### Training question type (TW-GRPO)

| Setting | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| Single-answer | 60.4 | 22.8 | 55.9 | 38.9 | 57.8 | 63.7 |
| Multi-answer | **60.9** | **42.5** | **64.4** | **50.4** | **62.9** | **65.8** |

#### Reward design (TW-GRPO, multi-answer)

| Setting | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| Fixed reward | **64.9** | 26.0 | 54.2 | 42.6 | 58.7 | 65.0 |
| Soft reward | 60.9 | **42.5** | **64.4** | **50.4** | **62.9** | **65.8** |

#### Token weighting (multi-answer, fixed reward)

| Setting | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| GRPO | 50.4 | **32.3** | **55.4** | 41.1 | 52.7 | **65.1** |
| TW-GRPO | **64.9** | 26.0 | 54.2 | **42.6** | **58.7** | 65.0 |

#### Token weighting (multi-answer, soft reward)

| Setting | Single exact | Multi exact | Multi soft | All exact | All soft | MMVU-MC |
| :-- | --: | --: | --: | --: | --: | --: |
| GRPO | 58.3 | 28.1 | 57.6 | 41.2 | 57.9 | 64.6 |
| TW-GRPO (α=0) | **64.7** | 36.6 | 60.5 | 48.6 | 62.3 | 62.1 |
| TW-GRPO | 60.9 | **42.5** | **64.4** | **50.4** | **62.9** | **65.8** |

<sub>First five metrics: CLEVRER. QAI converts NExT-GQA/STAR questions; α follows the conference parameterization.</sub>

</details>

### V. Component ablations

#### A. Primary and component comparisons: seeds 42, 123, and 2026

| Method | Local learning | SFT target source | All exact | All soft | Single exact | Multi exact | Multi soft |
| :-- | :-- | :-- | --: | --: | --: | --: | --: |
| TW-GRPO | Off | None | 50.35 ± 0.49 | 62.91 ± 0.17 | 61.15 ± 0.57 | 42.31 ± 1.23 | 64.22 ± 0.71 |
| + repair distillation | Off | Repaired | 52.01 ± 0.45 | 64.39 ± 0.21 | 61.93 ± 0.33 | 44.61 ± 0.89 | 66.21 ± 0.55 |
| + local evidence learning | On | None | 51.85 ± 0.46 | 64.27 ± 0.17 | 62.24 ± 0.41 | 44.11 ± 0.87 | 65.78 ± 0.49 |
| Full method | On | Repaired | **55.19 ± 0.44** | **67.20 ± 0.39** | **64.13 ± 0.23** | **48.53 ± 0.67** | **69.49 ± 0.51** |

#### B. Component analysis: seed 42, shared fixed repair corpus

| Method | Local learning | SFT target source | All exact | All soft | Single exact | Multi exact | Multi soft |
| :-- | :-- | :-- | --: | --: | --: | --: | --: |
| TW-GRPO | Off | None | 50.88 | 63.08 | 60.51 | 43.70 | 65.00 |
| + ordinary verified SFT | Off | Ordinary | 51.37 | 63.50 | 61.01 | 44.19 | 65.35 |
| + repair distillation | Off | Repaired | 52.49 | 64.62 | 61.70 | 45.63 | 66.80 |
| + local evidence learning | On | None | 52.32 | 64.47 | 62.00 | 45.10 | 66.30 |
| Full method | On | Repaired | **55.20** | **66.90** | **63.93** | **48.69** | **69.11** |

#### C. Paired overall exact-accuracy contrasts

| Contrast | Seed 42 | Seed 123 | Seed 2026 | Mean ± SD | Paired 95% CI |
| :-- | --: | --: | --: | --: | --: |
| Full - TW-GRPO | 4.32 | 5.35 | 4.85 | 4.84 ± 0.51 | [3.56, 6.12] |
| Full - repair distillation | 2.71 | 3.69 | 3.16 | 3.19 ± 0.49 | [2.04, 4.34] |
| Interaction I | 1.27 | 2.13 | 1.67 | 1.69 ± 0.43 | [0.28, 3.10] |

<sub>9,238 CLEVRER questions. A: mean ± sample SD. B: seed 42. C: pp gains with paired 95% video-cluster bootstrap intervals; contrasts computed before rounding.</sub>

### VI. Evidence sources

| Method | Evidence source | All exact | All soft | Single exact | Multi exact | Multi soft |
| :-- | :-- | --: | --: | --: | --: | --: |
| ER-TW-GRPO | Other failed candidates | **55.20** | **66.90** | **63.93** | **48.69** | **69.11** |
| Fresh-slot + TW-GRPO | Independent resampling | 52.16 | 65.21 | 62.43 | 44.51 | 67.28 |
| ER - fresh |  | +3.04 | +1.69 | +1.50 | +4.18 | +1.83 |

<sub>Seed 42; same local-learning and distillation recipe. Differences in pp.</sub>

### VII. Repair mechanism

#### A. Repair sources and visual screening

| Variant | Rescued/N (%) | Accepted repairs | Evidence precision (%) | Total GPU-s |
| :-- | --: | --: | --: | --: |
| Ordinary full-response resampling | 15/97 (15.46) | 18 | n/a | 4228 |
| Fresh value for the same slot | 22/97 (22.68) | 26 | 13/15 (86.67) | 3847 |
| Aligned repair, no visual screen | **25/97 (25.77)** | **32** | 12/18 (66.67) | **2881** |
| Full aligned evidence repair | 18/97 (18.56) | 21 | **15/17 (88.24)** | 3246 |

#### B. Execution under identical proposed evidence substitutions

| Execution | Original exact (%) | Replacement exact (%) | Net gain (pp) | Gain 95% CI (pp) | Evidence-use violations (%) | Total GPU-s |
| :-- | --: | --: | --: | --: | --: | --: |
| Restricted receiver graph | 0 | 13.33 | **13.33** | [8.07, 18.60] | **1/50 (2.00)** | **2885** |
| Free re-answering | 14.17 | **22.92** | 8.75 | [1.88, 15.62] | 9/50 (18.00) | 3967 |

<sub>A: N = 97 all-failed questions. B: 240 fixed substitutions. Precision = supported/audited; violations = violating/audited. GPU-s includes all attributed computation; n/a = not applicable. Within-protocol CIs do not establish a between-protocol difference.</sub>

## Training dynamics

Compared with TW-GRPO: lower late-stage reward dispersion and mostly shorter responses.

![Reward dispersion and response length before distillation](assets/training-dynamics.png)

## Reasoning examples

<details>
<summary>ECCV examples · density, counterfactual reasoning & answer consistency</summary>

TW-GRPO uses measured mass and displaced volume to infer density.

![MMVU density reasoning: Video-R1 and TW-GRPO](assets/density-reasoning.png)

### Counterfactual reasoning

![CLEVRER counterfactual reasoning: Video-R1 and TW-GRPO](assets/clevrer-case.png)

### Answer consistency

![MMVU rationale and final-answer consistency: Video-R1 and TW-GRPO](assets/mmvu-case.png)

</details>

## Core code

| Core | Source |
|:--|:--|
| Token-weighted GRPO | [Trainer](src/open_r1/trainer/grpo_trainer.py) · [Rewards](src/open_r1/grpo.py) |
| Evidence repair | [Replay](src/open_r1/er/core.py) · [Collection](src/open_r1/er/collect.py) |
| Local learning & distillation | [Supervision](src/open_r1/er/supervision.py) · [Trainer](src/open_r1/trainer/er_trainer.py) · [Distillation](src/open_r1/er/distill.py) |

## Quick start

**Hardware:** 2 × NVIDIA A100 · BF16 · DeepSpeed ZeRO-3 with CPU offload.

<details>
<summary>Install and train</summary>

Linux · Python 3.10.9+ · CUDA. Install PyTorch 2.5.1 and torchvision 0.20.1 for your CUDA environment first.

```bash
git clone https://github.com/just-a-go/ER-TW-GRPO.git
cd ER-TW-GRPO
pip install -e '.[eval]' transformers==4.50.0 accelerate==1.2.1 wandb pillow
pip install -e './qwen-vl-utils[decord]'
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Provide a local Qwen2.5-VL checkpoint and JSON/JSONL training data with absolute video paths:

```json
{"problem": "Which statements are true? A. ... B. ...", "video": "/path/to/video.mp4", "solution": "<answer>A,B</answer>"}
```

```bash
export ER_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export ER_TRAIN_DATA=/path/to/train.json

# 1. Collect evidence from the frozen checkpoint.
CUDA_VISIBLE_DEVICES=0 bash scripts/er-collect.sh

# 2. Train TW-GRPO with local evidence supervision (2 GPUs).
CUDA_VISIBLE_DEVICES=0,1 bash scripts/er-tw-grpo.sh

# 3. Distill from the completed main-training checkpoint (2 GPUs).
ER_MODEL=/path/to/Qwen2.5-VL-main-checkpoint \
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/er-distill.sh
```

Keep `Qwen2.5-VL` in checkpoint directory names. Launchers use development defaults (2K collection questions, epoch-based training); configure the subset and budgets for the paper's 1K/500-step protocol. `ER_LAMBDA=0` selects TW-GRPO without local learning. The `eval` install extra supplies `math-verify`, imported by the training entry point.

</details>

## Acknowledgements

We thank the contributors of [TW-GRPO](https://github.com/longmalongma/TW-GRPO), [Open R1](https://github.com/huggingface/open-r1), and [Qwen-VL](https://github.com/QwenLM/Qwen2-VL), and the dataset authors for their open resources.

[Apache-2.0](LICENSE).
