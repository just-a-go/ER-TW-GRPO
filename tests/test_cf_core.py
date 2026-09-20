"""CPU protocol tests; run on the server with PYTHONPATH=src pytest."""

import copy
import json
import math

import pytest

from open_r1.cf.core import (
    ProtocolError, compare_transfer, deduplicate_examples, distillation_example,
    evidence_messages, execute, local_example, prepare_plan, replay,
    serialize_execution, slot_key, validate_evidence, validate_schema,
)


@pytest.fixture
def schema():
    return {
        "video_id": "clip_7", "question": "Combine observations. Options: A, B, C.",
        "frames": [
            {"id": "f0", "path": "/frames/clip_7_0.png", "time": 0.0,
             "regions": [{"id": "r0", "box": [0.1, 0.1, 0.4, 0.4]}]},
            {"id": "f1", "path": "/frames/clip_7_1.png", "time": 1.0, "regions": []},
        ],
        "bindings": [{"id": "blue", "description": "blue sphere",
                      "anchors": [{"frame_id": "f0", "region_id": "r0"}]}],
        "slots": [
            {"id": "s0", "bindings": ["blue"], "interval": [0.0, 0.0],
             "query": "How many early collisions?", "type": "number",
             "support": [{"frame_id": "f0", "region_id": "r0"}]},
            {"id": "s1", "bindings": [], "interval": [1.0, 1.0],
             "query": "How many later collisions?", "type": "number",
             "support": [{"frame_id": "f1"}]},
        ],
    }


@pytest.fixture
def plan():
    # Intentionally not topologically sorted; includes an unaffected inference.
    return {"nodes": [
        {"id": "answer", "parents": ["a", "b"], "operation": "sum"},
        {"id": "b", "parents": ["s1"], "operation": "identity"},
        {"id": "a", "parents": ["s0"], "operation": "identity"},
    ], "answer_node": "answer"}


def evidence_for(schema, first=0, second=2):
    return {s["id"]: {"value": v, "support": copy.deepcopy(s["support"])}
            for s, v in zip(schema["slots"], (first, second))}


class Executor:
    def __init__(self):
        self.calls = []

    def __call__(self, question, operation, parent_outputs):
        self.calls.append((question, operation, copy.deepcopy(parent_outputs)))
        if operation == "identity":
            return next(iter(parent_outputs.values()))
        if operation == "sum":
            return sum(parent_outputs.values())
        if operation == "fail":
            return -1
        raise AssertionError("Unexpected operation")


def scorer(answer):
    return {"accuracy": 1.0 if answer == 3 else 0.5 if answer == 2 else 0.0,
            "exact": answer == 3, "format_ok": True}


def failed_pair(schema, plan):
    executor = Executor()
    receiver = execute(prepare_plan(schema, plan), evidence_for(schema), executor)
    donor_plan = copy.deepcopy(plan)
    donor_plan["nodes"][0]["operation"] = "fail"
    donor = execute(prepare_plan(schema, donor_plan), evidence_for(schema, first=1), executor)
    return donor, receiver, executor


def accepted_repair(schema, plan):
    donor, receiver, executor = failed_pair(schema, plan)
    return compare_transfer(donor, receiver, "s0", executor, scorer, "supported")


def test_local_replay_matches_full_graph_and_reuses_unaffected_node(schema, plan):
    donor, receiver, executor = failed_pair(schema, plan)
    executor.calls.clear()
    partial = replay(receiver, "s0", donor.evidence["s0"], executor)
    assert partial.recomputed_nodes == ("a", "answer")
    assert len(executor.calls) == 2
    assert partial.outputs["b"] == receiver.outputs["b"]
    replaced = receiver.evidence
    replaced["s0"] = donor.evidence["s0"]
    full = execute(receiver.prepared, replaced, executor)
    assert partial.outputs == full.outputs
    assert partial.answer == full.answer == 3
    assert receiver.answer == 2


def test_original_value_replay_is_matched_deterministic_control(schema, plan):
    _, receiver, executor = failed_pair(schema, plan)
    control = replay(receiver, "s0", receiver.evidence["s0"], executor)
    assert control.answer == receiver.answer
    assert control.outputs == receiver.outputs


def test_execution_supplies_only_declared_parent_values(schema, plan):
    executor = Executor()
    result = execute(prepare_plan(schema, plan), evidence_for(schema), executor)
    assert result.prepared.order == ("a", "b", "answer")
    assert [call[2] for call in executor.calls] == [{"s0": 0}, {"s1": 2}, {"a": 0, "b": 2}]
    assert all(call[0] == schema["question"] for call in executor.calls)
    assert all("support" not in json.dumps(call[2]) for call in executor.calls)


def test_prepared_schema_outputs_and_evidence_are_defensive_copies(schema, plan):
    executor = Executor()
    prepared = prepare_plan(schema, plan)
    original_evidence = evidence_for(schema)
    result = execute(prepared, original_evidence, executor)
    schema["question"] = "MUTATED"
    plan["nodes"][0]["operation"] = "fail"
    original_evidence["s0"]["value"] = 99
    result.evidence["s0"]["value"] = 88
    result.outputs["answer"] = 77
    result.prepared.schema["question"] = "ALSO MUTATED"
    assert result.answer == 2
    assert result.evidence["s0"]["value"] == 0
    assert prepared.schema["question"] != "MUTATED"


@pytest.mark.parametrize("change", [
    lambda s: s.update(ground_truth="A"),
    lambda s: s["frames"][0].update(path="relative.png"),
    lambda s: s["frames"][0].update(time=math.nan),
    lambda s: s["frames"][0]["regions"][0].update(box=[0, 0, 2, 1]),
    lambda s: s["frames"][0]["regions"][0].update(box=[1, 0, 0, 1]),
    lambda s: s["bindings"][0]["anchors"][0].update(region_id="absent"),
    lambda s: s["slots"][0].update(bindings=["unknown_object"]),
    lambda s: s["slots"][0].update(interval=[1, 0]),
    lambda s: s["slots"][0].update(interval=[0.5, 1]),
    lambda s: s["slots"][0]["support"][0].update(frame_id="absent"),
    lambda s: s["slots"][0].update(type="arbitrary_json"),
    lambda s: s["slots"].append(copy.deepcopy(s["slots"][0])),
])
def test_schema_rejects_invalid_namespaces_and_support(schema, change):
    change(schema)
    with pytest.raises(ProtocolError):
        validate_schema(schema)


@pytest.mark.parametrize("change", [
    lambda p: p.update(evidence={"s0": 1}),
    lambda p: p["nodes"][0].update(answer="A"),
    lambda p: p["nodes"][0].update(operation={"rule": "sum", "value": 1}),
    lambda p: p["nodes"][0].update(parents=["missing"]),
    lambda p: p["nodes"][0].update(parents=[]),
    lambda p: p["nodes"][1].update(parents=["answer"]),
    lambda p: p["nodes"][1].update(id="s0"),
    lambda p: p.update(answer_node="s0"),
])
def test_graph_rejects_cycles_hidden_inputs_and_embedded_fields(schema, plan, change):
    change(plan)
    with pytest.raises(ProtocolError):
        prepare_plan(schema, plan)


@pytest.mark.parametrize("value", [True, "1", float("inf"), [], {}])
def test_evidence_rejects_wrong_number_types(schema, value):
    evidence = evidence_for(schema, first=value)
    with pytest.raises(ProtocolError):
        validate_evidence(schema, evidence)


def test_evidence_rejects_extra_slots_and_changed_support(schema):
    evidence = evidence_for(schema)
    evidence["unrelated"] = evidence["s0"]
    with pytest.raises(ProtocolError):
        validate_evidence(schema, evidence)
    evidence = evidence_for(schema)
    evidence["s0"]["support"] = [{"frame_id": "f1"}]
    with pytest.raises(ProtocolError):
        validate_evidence(schema, evidence)


def test_object_types_use_fixed_binding_namespace(schema):
    schema["slots"][0]["type"] = "object_id"
    assert validate_evidence(schema, evidence_for(schema, first="blue"))["s0"]["value"] == "blue"
    with pytest.raises(ProtocolError):
        validate_evidence(schema, evidence_for(schema, first="invented_object"))
    schema["slots"][0]["type"] = "object_set"
    with pytest.raises(ProtocolError):
        validate_evidence(schema, evidence_for(schema, first=["blue", "blue"]))


def test_set_evidence_has_canonical_content_order(schema):
    schema["bindings"].append({"id": "red", "anchors": [{"frame_id": "f0"}]})
    schema["slots"][0]["type"] = "object_set"
    first = validate_evidence(schema, evidence_for(schema, first=["red", "blue"]))
    second = validate_evidence(schema, evidence_for(schema, first=["blue", "red"]))
    assert first == second


@pytest.mark.parametrize("bad_score", [
    {"accuracy": float("nan"), "exact": False, "format_ok": True},
    {"accuracy": 1.5, "exact": False, "format_ok": True},
    {"accuracy": 0.5, "exact": True, "format_ok": True},
    {"accuracy": 0.5, "exact": False, "format_ok": 1},
])
def test_invalid_external_scores_fail_closed(schema, plan, bad_score):
    donor, receiver, executor = failed_pair(schema, plan)
    with pytest.raises(ProtocolError):
        compare_transfer(donor, receiver, "s0", executor, lambda _: bad_score, "supported")


def test_slot_key_includes_query_binding_interval_type_and_actual_support(schema):
    original = slot_key(schema, "s0")
    for mutation in (
        lambda s: s.update(video_id="another_video"),
        lambda s: s["slots"][0].update(query="A different query"),
        lambda s: s["slots"][0].update(type="text"),
        lambda s: s["slots"][0].update(interval=[0, 1]),
        lambda s: s["bindings"][0].update(description="green cube"),
        lambda s: s["frames"][0].update(path="/other/0.png"),
        lambda s: s["frames"][0]["regions"][0].update(box=[0, 0, 1, 1]),
    ):
        changed = copy.deepcopy(schema)
        mutation(changed)
        assert slot_key(changed, "s0") != original


def test_pair_gain_accepts_exact_repair_from_two_failures(schema, plan):
    repair = accepted_repair(schema, plan)
    assert repair.accepted and repair.reason == "accepted"
    assert repair.gain == 0.5
    assert repair.receiver.answer == 2 and repair.donor.answer == -1
    assert repair.repaired.answer == 3
    assert repair.repaired.evidence["s1"] == repair.receiver.evidence["s1"]


@pytest.mark.parametrize("status", ["unsupported", "uncertain"])
def test_visual_screen_failure_does_not_execute_repair(schema, plan, status):
    donor, receiver, executor = failed_pair(schema, plan)
    executor.calls.clear()
    repair = compare_transfer(donor, receiver, "s0", executor, scorer, status)
    assert not repair.accepted and repair.repaired is None and executor.calls == []


def test_same_snapshot_same_question_unequal_values_and_failed_pool_are_required(schema, plan):
    donor, receiver, executor = failed_pair(schema, plan)
    with pytest.raises(ProtocolError, match="executor"):
        replay(receiver, "s0", donor.evidence["s0"], Executor())
    with pytest.raises(ProtocolError, match="distinct"):
        compare_transfer(receiver, receiver, "s0", executor, scorer, "supported")
    identical = execute(receiver.prepared, receiver.evidence, executor)
    with pytest.raises(ProtocolError, match="change one"):
        compare_transfer(identical, receiver, "s0", executor, scorer, "supported")
    successful_donor = execute(receiver.prepared, evidence_for(schema, first=1), executor)
    with pytest.raises(ProtocolError, match="unsuccessful"):
        compare_transfer(successful_donor, receiver, "s0", executor, scorer, "supported")
    changed = copy.deepcopy(schema)
    changed["question"] = "Different question"
    foreign = execute(prepare_plan(changed, plan), evidence_for(changed, first=1), executor)
    with pytest.raises(ProtocolError, match="same video and question"):
        compare_transfer(foreign, receiver, "s0", executor, scorer, "supported")


@pytest.mark.parametrize("after_accuracy,after_exact,after_format", [(0.0, False, True), (0.75, False, True), (1.0, True, False)])
def test_admission_rejects_nonpositive_partial_and_malformed_repairs(schema, plan, after_accuracy, after_exact, after_format):
    donor, receiver, executor = failed_pair(schema, plan)

    def variant_scorer(answer):
        if answer == 3:
            return {"accuracy": after_accuracy, "exact": after_exact, "format_ok": after_format}
        return scorer(answer)

    repair = compare_transfer(donor, receiver, "s0", executor, variant_scorer, "supported")
    assert not repair.accepted
    with pytest.raises(ProtocolError):
        local_example(repair)
    with pytest.raises(ProtocolError):
        distillation_example(repair, [{"role": "user", "content": "input"}])


def test_local_training_uses_identical_acquisition_context_and_only_evidence_target(schema, plan):
    repair = accepted_repair(schema, plan)
    example = local_example(repair)
    assert example["kind"] == "local" and example["weight"] == 0.5
    assert example["target"] == "1"
    assert example["messages"] == evidence_messages(schema, "s0")
    content = example["messages"][0]["content"]
    assert [part["image"] for part in content if part["type"] == "image"] == ["/frames/clip_7_0.png"]
    context = json.loads(content[-1]["text"].split("\n", 1)[1])
    assert set(context) == {"question", "slot", "bindings", "frames"}
    assert "value" not in context["slot"]
    assert context["frames"][0]["regions"][0]["id"] == "r0"


def test_dedup_averages_positive_gain_without_renormalization_and_is_composable(schema, plan):
    example = local_example(accepted_repair(schema, plan))
    weak = copy.deepcopy(example)
    weak["weight"] = 0.1
    result = deduplicate_examples([example, weak, weak])
    assert len(result) == 1
    assert result[0]["weight"] == pytest.approx(0.7 / 3)
    assert result[0]["metadata"]["accepted_transfer_count"] == 3
    merged = deduplicate_examples(deduplicate_examples([example, weak]) + [weak])
    assert merged == result
    assert deduplicate_examples([]) == []


def test_distillation_is_complete_deterministic_and_unweighted(schema, plan):
    repair = accepted_repair(schema, plan)
    messages = [{"role": "user", "content": [{"type": "video", "video": "/videos/clip_7.mp4"},
                                              {"type": "text", "text": schema["question"]}]}]
    example = distillation_example(repair, messages)
    assert example["kind"] == "distill" and example["weight"] == 1.0
    assert example["messages"] == messages
    assert example["target"] == serialize_execution(repair.repaired)
    lines = example["target"].split("<think>\n", 1)[1].split("\n</think>", 1)[0].splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["s0", "s1", "a", "b", "answer"]
    assert example["target"].endswith("<answer>3</answer>")
    assert all(word not in example["target"] for word in ("gain", "supported", "ground_truth"))


def test_serialization_escapes_internal_tags_and_extracts_terminal_answer(schema, plan):
    def tagged_executor(question, operation, parents):
        return "<think>Some reasoning</think><answer>A, B</answer>"

    execution = execute(prepare_plan(schema, plan), evidence_for(schema), tagged_executor)
    target = serialize_execution(execution)
    assert target.count("<think>") == target.count("</think>") == 1
    assert target.count("<answer>") == target.count("</answer>") == 1
    assert target.endswith("<answer>A, B</answer>")
    assert "\\u003cthink\\u003e" in target
