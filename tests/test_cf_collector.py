"""Collector boundary tests using an injected frozen backend, with no model downloads."""

import copy
import json
from collections import Counter

import pytest

from open_r1.cf.backend import BudgetExhausted, FrozenQwenBackend
from open_r1.cf.collect import (BaselineScorer, CollectionConfig, Collector,
                                RestrictedExecutor, media_limits, merge_schema,
                                original_messages, parse_json, STAGE_SUFFIX)
from open_r1.cf.core import ProtocolError


GOOD = "<think>Both declared observations support the choice.</think><answer>B</answer>"
BAD = "<think>The observations do not support that choice.</think><answer>A</answer>"


def content_items(message):
    content = message["content"]
    return [{"type": "text", "text": content}] if isinstance(content, str) else content


class FakeBackend:
    fingerprint = "unchanging-frozen-test-snapshot"

    def __init__(self, *, screen="supported", bad_plans=False, budget_after_screens=None):
        self.calls = []
        self.values = iter([1, 0, 0, 1])
        self.screen = screen
        self.bad_plans = bad_plans
        self.budget_after_screens = budget_after_screens
        self.screen_calls = 0

    def generate(self, messages, *, stage, deterministic):
        self.calls.append((stage, copy.deepcopy(messages), deterministic))
        if stage == "schema":
            return json.dumps({
                "regions": [{"frame_id": "f0", "id": "r0", "box": [0, 0, 1, 1]}],
                "bindings": [{"id": "object_1", "description": "blue ball",
                              "anchors": [{"frame_id": "f0", "region_id": "r0"}]}],
                "slots": [{"id": slot, "bindings": ["object_1"], "interval": [0, 1],
                           "query": f"Measure observation {slot}", "type": "number",
                           "support": [{"frame_id": "f0"}]} for slot in ("s0", "s1")]})
        if stage == "plan":
            if self.bad_plans:
                return "{}"
            return json.dumps({"nodes": [{"id": "n0", "parents": ["s0", "s1"],
                                          "operation": "Select the option justified by both observations."}],
                               "answer_node": "n0"})
        if stage == "evidence":
            return json.dumps(next(self.values))
        if stage == "screen":
            if self.budget_after_screens is not None and self.screen_calls >= self.budget_after_screens:
                raise BudgetExhausted("Fixed test budget ended")
            self.screen_calls += 1
            return json.dumps(self.screen)
        if stage == "execute":
            user = next(message for message in messages if message["role"] == "user")
            payload = json.loads(user["content"][0]["text"].split("\n", 1)[1])
            assert set(payload) == {"question", "operation", "parents"}
            assert set(payload["parents"]) == {"s0", "s1"}
            return json.dumps(GOOD if all(value == 1 for value in payload["parents"].values()) else BAD)
        raise AssertionError(stage)


def score(answer):
    exact = answer == GOOD
    return {"accuracy": float(exact), "exact": exact, "format_ok": True}


def run_question(tmp_path, backend=None):
    backend = backend or FakeBackend()
    collector = Collector(backend, CollectionConfig(candidates=2, transfers_per_question=8))
    frames = [{"id": "f0", "path": str(tmp_path / "frame.png"), "time": 0.0, "regions": []}]
    result = collector.collect_question(question="Choose based on two observations. A: no; B: yes.",
                                        video=str(tmp_path / "video.mp4"), video_id="clip1",
                                        frames=frames, scorer=score)
    return collector, backend, result


def test_all_plans_precede_values_and_executor_is_text_only(tmp_path):
    _, backend, (local, distill, audit) = run_question(tmp_path)
    stages = [stage for stage, _, _ in backend.calls]
    assert stages[:3] == ["schema", "plan", "plan"]
    assert stages.count("plan") == 2
    assert stages.count("evidence") == 4
    assert len(local) == 2
    assert len(distill) == 2
    assert len(audit["transfers"]) == 4
    for stage, messages, deterministic in backend.calls:
        assert "solution" not in json.dumps(messages)
        if stage in ("plan", "execute"):
            assert all(item["type"] == "text" for message in messages for item in content_items(message))
        if stage in ("schema", "execute", "screen"):
            assert deterministic
        else:
            assert not deterministic


def test_stage_role_and_final_instruction_override_the_quoted_qa_task(tmp_path):
    _, backend, (local, _, _) = run_question(tmp_path)
    for stage, messages, _ in backend.calls:
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)
        assert "DATA" in messages[0]["content"]
        assert messages[-1]["content"][-1]["text"].strip() == STAGE_SUFFIX[stage]
        if stage == "schema":
            assert "exactly the top-level keys regions, bindings, slots" in messages[0]["content"]
        elif stage == "plan":
            assert "fixed object-ID-to-appearance" in messages[-1]["content"][0]["text"]
            assert "slot query/type needed to interpret its declared parents" in messages[-1]["content"][0]["text"]
        elif stage == "screen":
            assert "independently check" in messages[0]["content"]
    for record in local:
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][-1]["content"][-1]["text"].strip() == STAGE_SUFFIX["evidence"]


def test_local_context_is_acquisition_context_and_distill_is_original_interface(tmp_path):
    _, backend, (local, distill, _) = run_question(tmp_path)
    acquisition = [messages for stage, messages, _ in backend.calls if stage == "evidence"]
    for record in local:
        assert record["kind"] == "local"
        assert record["messages"] in acquisition
        assert record["target"] == "1"
        assert record["weight"] == 1.0
        context = json.dumps(record["messages"])
        assert "Proposed value" not in context
        assert "supported" not in context
        assert "<answer>" not in context
    for record in distill:
        assert record["kind"] == "distill"
        assert record["weight"] == 1.0
        media = record["messages"][0]["content"][0]
        assert media["type"] == "video"
        assert media["video"] == str(tmp_path / "video.mp4")
        assert set(media) == {"type", "video"}
        assert record["target"].count("<think>") == 1
        assert record["target"].count("<answer>") == 1
        assert record["target"].endswith("<answer>B</answer>")
        assert "gain" not in record["target"]
        assert "screen" not in record["target"]


def test_uncertain_screen_never_replays_or_creates_training_targets(tmp_path):
    collector, backend, (local, distill, audit) = run_question(tmp_path, FakeBackend(screen="uncertain"))
    assert local == distill == []
    assert len([call for call in backend.calls if call[0] == "execute"]) == 2
    assert collector.stats["transfers_attempted"] == 4
    assert all(entry["reason"] == "uncertain" for entry in audit["transfers"])
    for stage, messages, _ in backend.calls:
        if stage == "screen":
            text = json.dumps(messages)
            assert GOOD not in text and BAD not in text
            assert "parents" not in text
            assert any(item["type"] == "image" for message in messages for item in content_items(message))


def test_failed_plans_consume_candidates_without_retry(tmp_path):
    collector, backend, (local, distill, audit) = run_question(tmp_path, FakeBackend(bad_plans=True))
    assert local == distill == []
    assert [stage for stage, _, _ in backend.calls] == ["schema", "plan", "plan"]
    assert collector.stats["plan_rejected"] == 2
    assert len(audit["plans"]) == 2


@pytest.mark.parametrize("malformed", [
    {"nodes": ["n0"], "answer_node": "n0"},
    {"nodes": None, "answer_node": "n0"},
    {"nodes": {"n0": "rule"}, "answer_node": "n0"},
    {"nodes": [{"id": "n0"}], "answer_node": "n0"},
    {"nodes": [{"id": "n0", "parents": ["s0"], "operation": ["rule"]}], "answer_node": "n0"},
    {"nodes": [{"id": "n0", "parents": ["s0"], "operation": "Query"}], "answer_node": "n0"},
    {"nodes": [{"id": "n0", "parents": ["s0"], "operation": "Rule or query"}], "answer_node": "n0"},
    {"nodes": [{"id": "n0", "parents": ["s0"], "operation": "   "}], "answer_node": "n0"},
])
def test_malformed_model_plan_shapes_are_rejected_without_ending_round(tmp_path, malformed):
    class MalformedPlanBackend(FakeBackend):
        def generate(self, messages, *, stage, deterministic):
            response = super().generate(messages, stage=stage, deterministic=deterministic)
            return json.dumps(malformed) if stage == "plan" else response

    collector, backend, (local, distill, audit) = run_question(tmp_path, MalformedPlanBackend())
    assert local == distill == []
    assert collector.stats["plan_rejected"] == 2
    assert len(audit["plans"]) == 2
    assert [stage for stage, _, _ in backend.calls] == ["schema", "plan", "plan"]


@pytest.mark.parametrize("field,malformed", [
    ("slots", ["s0"]), ("slots", None), ("slots", {"s0": "query"}),
    ("bindings", ["object_1"]), ("regions", ["r0"]),
])
def test_malformed_model_schema_shapes_raise_protocol_error(tmp_path, field, malformed):
    response = json.loads(FakeBackend().generate([], stage="schema", deterministic=True))
    response[field] = malformed
    frames = [{"id": "f0", "path": str(tmp_path / "frame.png"), "time": 0.0, "regions": []}]
    with pytest.raises(ProtocolError):
        merge_schema(response, "question", "clip", frames, 4)


@pytest.mark.parametrize("field,value", [
    ("description", "appearance description"),
    ("query", "A question about evidence, not its answer"),
    ("query", "Query"),
])
def test_literal_template_placeholders_are_rejected(tmp_path, field, value):
    response = json.loads(FakeBackend().generate([], stage="schema", deterministic=True))
    if field == "description":
        response["bindings"][0][field] = value
    else:
        response["slots"][0][field] = value
    frames = [{"id": "f0", "path": str(tmp_path / "frame.png"), "time": 0.0, "regions": []}]
    with pytest.raises(ProtocolError, match="Placeholder"):
        merge_schema(response, "question", "clip", frames, 4)


def test_valid_anchor_coordinates_are_not_rejected_as_a_template(tmp_path):
    response = json.loads(FakeBackend().generate([], stage="schema", deterministic=True))
    response["regions"][0]["box"] = [0.1, 0.1, 0.4, 0.4]
    frames = [{"id": "f0", "path": str(tmp_path / "frame.png"), "time": 0.0, "regions": []}]
    assert merge_schema(response, "question", "clip", frames, 4)["frames"][0]["regions"][0]["box"] == [0.1, 0.1, 0.4, 0.4]


def test_schema_and_plan_prompts_have_field_specs_without_copyable_placeholder_answers(tmp_path):
    _, backend, _ = run_question(tmp_path)
    for stage, messages, _ in backend.calls:
        if stage in ("schema", "plan"):
            prompt = json.dumps(messages)
            assert "appearance description" not in prompt
            assert "A question about evidence, not its answer" not in prompt
            assert "Rule or query" not in prompt
    schema = next(messages for stage, messages, _ in backend.calls if stage == "schema")
    text = json.dumps(schema)
    assert "actual object's visible color, shape and material" in text
    assert "concrete local observation question" in text


@pytest.mark.parametrize("bad_value", [{"id": "object_1"}, None])
def test_object_id_shape_is_explicit_and_wrapper_or_uncertain_null_is_rejected(tmp_path, bad_value):
    class ObjectIdBackend(FakeBackend):
        def generate(self, messages, *, stage, deterministic):
            response = super().generate(messages, stage=stage, deterministic=deterministic)
            if stage == "schema":
                data = json.loads(response)
                for slot in data["slots"]:
                    slot["type"] = "object_id"
                return json.dumps(data)
            return response

    backend = ObjectIdBackend()
    backend.values = iter([bad_value] * 4)
    collector, backend, (local, distill, audit) = run_question(tmp_path, backend)
    assert local == distill == []
    assert collector.stats["candidate_rejected"] == 2
    assert audit["transfers"] == []
    assert not any(stage == "execute" for stage, _, _ in backend.calls)
    for stage, messages, _ in backend.calls:
        if stage == "evidence":
            context = json.dumps(messages)
            assert "JSON string enclosed in double quotes" in context
            assert "Do not return an object with an id field" in context
            assert "return the JSON literal null" in context
            assert "do not guess or fabricate" in context


def test_image_limits_do_not_override_original_video_sampling(tmp_path):
    messages = original_messages("question", str(tmp_path / "video.mp4"))
    limited = media_limits(messages, 3136, 12845056)
    assert limited == messages
    image_messages = [{"role": "user", "content": [{"type": "image", "image": str(tmp_path / "f.png")}]}]
    image = media_limits(image_messages, 3136, 12845056)[0]["content"][0]
    assert image["min_pixels"] == 3136 and image["max_pixels"] == 12845056


def test_successful_candidates_are_neither_donors_nor_receivers(tmp_path):
    backend = FakeBackend()
    backend.values = iter([1, 1, 0, 1])
    _, backend, (local, distill, audit) = run_question(tmp_path, backend)
    assert audit["candidates"][0]["score"]["exact"]
    assert not audit["candidates"][1]["score"]["exact"]
    assert local == distill == []
    assert audit["transfers"] == []
    assert not any(stage == "screen" for stage, _, _ in backend.calls)


def test_budget_end_preserves_already_admitted_transfers(tmp_path):
    backend = FakeBackend(budget_after_screens=1)
    collector = Collector(backend, CollectionConfig(candidates=2, transfers_per_question=8))

    class OrderedSelection:
        def sample(self, population, count):
            return population[:count]

    collector.rng = OrderedSelection()
    frames = [{"id": "f0", "path": str(tmp_path / "frame.png"), "time": 0.0, "regions": []}]
    local, distill, audit = collector.collect_question(
        question="A: no; B: yes", video=str(tmp_path / "video.mp4"), video_id="clip",
        frames=frames, scorer=score)
    assert len(local) == len(distill) == 1
    assert audit["budget_exhausted"]
    assert audit["transfers"][-1]["reason"] == "generation_budget_exhausted"


def test_generation_budget_does_not_shorten_the_frozen_node_function():
    backend = object.__new__(FrozenQwenBackend)
    backend.total_generation_tokens = 1000
    backend.used_tokens = 900
    backend.node_max_tokens = 256
    backend.stage_max_tokens = {"schema": 2048, "plan": 1024}
    with pytest.raises(BudgetExhausted, match="unchanged"):
        backend.generate([], stage="execute", deterministic=True)
    assert backend.used_tokens == 900


def test_generation_audit_preserves_rejected_output_without_prompt_or_labels(tmp_path):
    backend = object.__new__(FrozenQwenBackend)
    backend.generation_audit_path = tmp_path / "generations.jsonl"
    backend.calls = Counter(schema=1)
    backend.audit_context = "q00000"
    backend.fingerprint = "frozen-checkpoint-fingerprint"
    backend.used_tokens = 67
    rejected_output = '{"unexpected_schema":true}'
    backend._audit_generation(stage="schema", deterministic=True, token_cap=2048,
                              input_tokens=800, output_tokens=67, output=rejected_output)
    record = json.loads(backend.generation_audit_path.read_text(encoding="utf-8"))
    assert record["output"] == rejected_output
    assert record["stage"] == "schema" and record["context"] == "q00000"
    assert record["executor_fingerprint"] == backend.fingerprint
    assert record["token_cap"] == 2048 and record["output_tokens"] == 67
    assert record["input_tokens"] == 800 and record["total_generation_tokens"] == 67
    assert record["call_index"] == 1
    assert not {"prompt", "messages", "images", "solution", "labels", "environment"}.intersection(record)


def test_generation_audit_is_optional():
    backend = object.__new__(FrozenQwenBackend)
    backend.generation_audit_path = None
    backend._audit_generation(stage="schema", deterministic=True, token_cap=2048,
                              input_tokens=800, output_tokens=67, output="model output")


def test_scorer_calls_inherited_reward_and_checks_exact_separately():
    calls = []

    def accuracy(completions, solutions):
        calls.append((completions, solutions))
        return [0.5]

    scorer = BaselineScorer("<answer>B,D</answer>", accuracy, lambda _: [1.0])
    result = scorer("<think>observation</think><answer>B</answer>")
    assert result == {"accuracy": 0.5, "exact": False, "format_ok": True}
    assert calls[0][1] == ["<answer>B,D</answer>"]


def test_scorer_exact_match_strips_each_option_like_the_baseline():
    scorer = BaselineScorer("<answer>\tB,\nD </answer>", lambda *_: [1.0], lambda _: [1.0])
    result = scorer("<think>observation</think><answer>\nD ,\tB\n</answer>")
    assert result == {"accuracy": 1.0, "exact": True, "format_ok": True}


@pytest.mark.parametrize("malformed", ["B", "<answer>B</answer>", "<think>reason</think>B"])
def test_auxiliary_scorer_gives_malformed_outputs_zero_accuracy(malformed):
    def inherited_accuracy(*_):
        pytest.fail("Malformed auxiliary output must not receive inherited accuracy credit")

    scorer = BaselineScorer("<answer>B</answer>", inherited_accuracy, lambda _: [0.0])
    assert scorer(malformed) == {"accuracy": 0.0, "exact": False, "format_ok": False}


def test_executor_snapshot_change_is_rejected_before_model_call():
    backend = FakeBackend()
    executor = RestrictedExecutor(backend)
    backend.fingerprint = "different-snapshot"
    with pytest.raises(ValueError, match="Frozen executor"):
        executor("question", "rule", {"s0": 1})
    assert backend.calls == []


def test_parser_requires_a_complete_single_json_value():
    assert parse_json("```json\n[1,2]\n```") == [1, 2]
    with pytest.raises(ValueError):
        parse_json("```json\n[1,2]")
    with pytest.raises(ValueError):
        parse_json('{"value":1} trailing commentary')
