"""Run on the server; no model downloads or GPU are required for these checks."""

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from open_r1.cf.supervision import (
    EvidenceCollator, RepairDataset, distributed_stream_indices,
    target_token_mask, weighted_sequence_nll,
)


def record(kind="local", weight=0.25, target="red"):
    return dict(kind=kind, messages=[dict(role="user", content=[dict(type="text", text="observe")])],
                target=target, weight=weight)


def write_rows(tmp_path, rows):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_data_empty_is_valid_but_missing_is_not(tmp_path):
    assert len(RepairDataset(write_rows(tmp_path, []))) == 0
    with pytest.raises(FileNotFoundError):
        RepairDataset(tmp_path / "absent.jsonl")


@pytest.mark.parametrize("weight", [0, -1, 1.01, float("inf"), float("nan"), True])
def test_invalid_gains_fail_closed(tmp_path, weight):
    with pytest.raises(ValueError):
        RepairDataset(write_rows(tmp_path, [record(weight=weight)]))


def test_distillation_requires_explicit_stream_and_unit_weight(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        RepairDataset(write_rows(tmp_path, [record()]), kind="distill")
    with pytest.raises(ValueError, match="weight=1"):
        RepairDataset(write_rows(tmp_path, [record(kind="distill", weight=0.2)]), kind="distill")
    row = record(kind="distill", weight=1)
    row["metadata"] = {"gain": 0.125}
    assert RepairDataset(write_rows(tmp_path, [row]), kind="distill")[0]["weight"] == 1


def test_nll_is_sequence_mean_then_gain_mean_with_only_target_gradients():
    probabilities = torch.tensor([
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.25, 0.75], [0.25, 0.75], [0.5, 0.5]],
    ])
    logits = probabilities.log().requires_grad_()
    labels = torch.tensor([[-100, -100, 0, -100], [-100, -100, 0, 0]])
    weights = torch.tensor([0.25, 0.5], requires_grad=True)
    loss = weighted_sequence_nll(logits, labels, weights)
    assert loss.item() == pytest.approx((0.25 * math.log(2) + 0.5 * math.log(4)) / 2)
    loss.backward()
    assert weights.grad is None
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 2:].abs().sum() == 0
    assert logits.grad[1, 1:3].abs().sum() > 0


def test_uniformly_small_gain_remains_small():
    logits = torch.zeros(2, 3, 2, requires_grad=True)
    labels = torch.tensor([[-100, 0, 1], [-100, -100, 1]])
    base = weighted_sequence_nll(logits, labels, torch.ones(2))
    small = weighted_sequence_nll(logits, labels, torch.full((2,), 0.01))
    assert small.item() == pytest.approx(0.01 * base.item())


def test_nll_rejects_empty_target():
    with pytest.raises(ValueError, match="predictable target"):
        weighted_sequence_nll(torch.zeros(1, 3, 2), torch.full((1, 3), -100), torch.ones(1))


def test_distributed_stream_consumes_new_microsteps_without_tail_padding():
    # 5 steps * 3 ranks * 2 examples = exactly 6 complete passes over 5 rows.
    stream = []
    for step in range(5):
        for rank in range(3):
            stream.extend(distributed_stream_indices(5, step, 2, rank, 3, 42))
    for start in range(0, 30, 5):
        assert sorted(stream[start:start + 5]) == list(range(5))
    first = distributed_stream_indices(20, 0, 2, 0, 2, 42)
    second = distributed_stream_indices(20, 1, 2, 0, 2, 42)
    assert not set(first) & set(second)
    assert first == distributed_stream_indices(20, 0, 2, 0, 2, 42)
    assert distributed_stream_indices(1, 100, 2, 3, 4, 42) == [0, 0]


class CharacterTokenizer:
    is_fast = True
    all_special_ids = [0, ord("#"), ord("~")]

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(char) for char in text],
                    offset_mapping=[(i, i + 1) for i in range(len(text))])


class ExpandingProcessor:
    """Tiny processor with chat wrappers and variable-length visual expansion."""
    tokenizer = CharacterTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prompt = "user:@ " + messages[0]["content"][0]["text"] + "\nassistant:\n"
        return prompt if add_generation_prompt else prompt + messages[-1]["content"] + "#\n"

    def __call__(self, text, **kwargs):
        sequences = [[ord(char) for char in row.replace("@", "~~~~")] for row in text]
        longest = max(map(len, sequences))
        ids = [[0] * (longest - len(row)) + row for row in sequences]
        attention = [[0] * (longest - len(row)) + [1] * len(row) for row in sequences]
        return dict(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attention))


def test_collator_masks_prompt_visual_padding_and_eos_without_mutation():
    rows = [record(target="red"), record(target="blue ball")]
    before = copy.deepcopy(rows)
    batch = EvidenceCollator(ExpandingProcessor(), vision_loader=lambda _: ([], []))(rows)
    assert rows == before
    for index, expected in enumerate(["red", "blue ball"]):
        labels = batch["labels"][index]
        assert "".join(chr(token) for token in labels[labels != -100].tolist()) == expected
        assert not bool(((labels == ord("~")) | (labels == ord("#"))).any())
        assert bool((labels[batch["attention_mask"][index] == 0] == -100).all())


def test_vision_loader_preserves_explicit_support_and_handles_both_modalities(monkeypatch):
    from open_r1.cf import backend
    from open_r1.cf.supervision import _load_vision
    seen = []
    def fetch_image(part):
        seen.append(copy.deepcopy(part))
        return "image_tensor"
    def fetch_video(part):
        seen.append(copy.deepcopy(part))
        return "video_tensor"
    monkeypatch.setattr(backend, "vendored_vision", lambda: SimpleNamespace(
        fetch_image=fetch_image, fetch_video=fetch_video))
    parts = [dict(type="image", image="fixed_crop.png", max_pixels=3136),
             dict(type="video", video=["frame_0.png", "frame_1.png"], fps=2)]
    images, videos = _load_vision([dict(role="system", content="rule"), dict(role="user", content=parts)])
    assert images == ["image_tensor"] and videos == ["video_tensor"]
    assert seen == parts


def test_boundary_and_special_token_violations_are_explicit():
    with pytest.raises(ValueError, match="boundary crosses"):
        target_token_mask([1], [(0, 3)], 1, 3)
    with pytest.raises(ValueError, match="special token"):
        target_token_mask([7], [(0, 3)], 0, 3, special_ids={7})


def test_collator_refuses_truncation_and_target_changes():
    collator = EvidenceCollator(ExpandingProcessor(), max_length=2, vision_loader=lambda _: ([], []))
    with pytest.raises(ValueError, match="truncate"):
        collator([record()])
    class BrokenProcessor(ExpandingProcessor):
        def __call__(self, **kwargs):
            batch = super().__call__(**kwargs)
            batch["input_ids"][0, -3] = 1
            return batch
    with pytest.raises(ValueError, match="changed target"):
        EvidenceCollator(BrokenProcessor(), vision_loader=lambda _: ([], []))([record()])


def test_trainer_baseline_passthrough_and_one_aux_forward_per_microstep(monkeypatch, tmp_path):
    # The inherited module has baseline loggers; use explicit test-only paths.
    monkeypatch.setenv("PRIVATE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WANDB_NAME", "cf_unit")
    from open_r1.trainer.cf_trainer import CFQwen2VLGRPOTrainer
    from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
    from transformers import Trainer

    base_loss = torch.tensor(3.0, requires_grad=True)
    calls = []
    monkeypatch.setattr(Qwen2VLGRPOTrainer, "compute_loss", lambda *args, **kwargs: base_loss)
    monkeypatch.setattr(Trainer, "_prepare_inputs", lambda self, inputs: inputs)
    trainer = CFQwen2VLGRPOTrainer.__new__(CFQwen2VLGRPOTrainer)
    trainer.cf_dataset = None
    trainer.cf_lambda = 0
    model = SimpleNamespace(training=True)
    assert trainer.compute_loss(model, []) is base_loss
    trainer.cf_dataset = []
    trainer.cf_lambda = 0.5
    assert trainer.compute_loss(model, []) is base_loss
    trainer.cf_dataset = [record(weight=0.25) for _ in range(4)]
    trainer.cf_microstep, trainer.cf_local_batch_size = 0, 1
    trainer.args = SimpleNamespace(seed=42)
    trainer.accelerator = SimpleNamespace(process_index=0, num_processes=2, gather_for_metrics=lambda x: x)
    trainer._metrics = {"cf/local_loss": [], "cf/mean_gain": []}
    trainer.cf_collator = lambda rows: dict(
        input_ids=torch.tensor([[0, 1, 0]]), labels=torch.tensor([[-100, -100, 0]]),
        weights=torch.tensor([rows[0]["weight"]]),
    )
    class Model:
        training = True
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(logits=torch.zeros(1, 3, 2, requires_grad=True))
    for _ in range(2):
        result = trainer.compute_loss(Model(), [])
        assert result.item() == pytest.approx(3 + 0.5 * 0.25 * math.log(2))
    assert len(calls) == trainer.cf_microstep == 2
    assert all("labels" not in call and "weights" not in call for call in calls)


def test_cf_checkpointing_is_nonreentrant_only_for_active_auxiliary_data(monkeypatch, tmp_path):
    monkeypatch.setenv("PRIVATE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WANDB_NAME", "cf_unit")
    from open_r1.trainer.cf_trainer import CFQwen2VLGRPOTrainer
    from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer

    def fake_init(self, *args, **kwargs):
        self.args = kwargs["args"]
        self.processing_class = ExpandingProcessor()
    monkeypatch.setattr(Qwen2VLGRPOTrainer, "__init__", fake_init)
    arguments = SimpleNamespace(dataloader_num_workers=0, gradient_checkpointing=True,
                                gradient_checkpointing_kwargs=None)
    path = write_rows(tmp_path, [record()])
    trainer = CFQwen2VLGRPOTrainer(args=arguments, cf_local_data=path, cf_lambda=0.2)
    assert trainer.args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    arguments.gradient_checkpointing_kwargs = {"use_reentrant": True}
    with pytest.raises(ValueError, match="use_reentrant=False"):
        CFQwen2VLGRPOTrainer(args=arguments, cf_local_data=path, cf_lambda=0.2)
    path = write_rows(tmp_path, [])
    arguments.dataloader_num_workers = 8
    empty = CFQwen2VLGRPOTrainer(args=arguments, cf_local_data=path, cf_lambda=0.2)
    assert empty.cf_collator is None
    assert empty.args.gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_distillation_cli_help_builds_without_duplicate_fields_or_model_loading(monkeypatch, capsys):
    import sys
    from transformers import HfArgumentParser, TrainingArguments
    from open_r1.cf import distill

    parser = HfArgumentParser((distill.DistillArguments, TrainingArguments))
    assert sum(action.dest == "resume_from_checkpoint" for action in parser._actions) == 1
    def forbid_model_loading(*args, **kwargs):
        raise AssertionError("CLI help must not load any processor or model")
    monkeypatch.setattr(distill.AutoProcessor, "from_pretrained", forbid_model_loading)
    monkeypatch.setattr(sys, "argv", ["open_r1.cf.distill", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        distill.main()
    assert exit_info.value.code == 0
    assert "--distill_data" in capsys.readouterr().out
