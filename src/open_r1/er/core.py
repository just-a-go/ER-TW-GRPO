"""Validated evidence exchange and deterministic dependency-constrained replay."""

import copy
import json
import math
import ntpath
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class ProtocolError(ValueError):
    """An input violates the evidence-repair collection protocol."""


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Values must be finite JSON data") from exc


def _copy(value):
    _json(value)
    return copy.deepcopy(value)


def _require(condition, message):
    if not condition:
        raise ProtocolError(message)


def _keys(value, required, optional=()):
    _require(isinstance(value, dict), "Expected an object")
    _require(all(isinstance(k, str) for k in value), "Object keys must be strings")
    _require(set(required) <= set(value) <= set(required) | set(optional),
             "Missing or unknown fields: " + ", ".join(sorted(value)))


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _indexed(items, name):
    _require(isinstance(items, list), name + " must be a list")
    _require(all(isinstance(item, dict) and _text(item.get("id")) for item in items),
             name + " require nonempty string IDs")
    result = {item["id"]: item for item in items}
    _require(len(result) == len(items), "Duplicate " + name + " IDs")
    return result


def _references(refs, frames):
    _require(isinstance(refs, list) and refs, "Visual support must be nonempty")
    for ref in refs:
        _keys(ref, ("frame_id",), ("region_id",))
        _require(isinstance(ref["frame_id"], str) and ref["frame_id"] in frames,
                 "Unknown support frame")
        if "region_id" in ref:
            regions = {r["id"] for r in frames[ref["frame_id"]]["regions"]}
            _require(isinstance(ref["region_id"], str) and ref["region_id"] in regions,
                     "Unknown support region")
    _require(len({_json(r) for r in refs}) == len(refs), "Duplicate support references")


def validate_schema(schema: dict) -> dict:
    """Validate the frozen shared namespace and visual support; return a copy."""
    _keys(schema, ("video_id", "question", "frames", "bindings", "slots"))
    _require(_text(schema["video_id"]) and _text(schema["question"]), "Missing video/question")
    frames = _indexed(schema["frames"], "frame")
    _require(bool(frames), "At least one frame is required")
    for frame in frames.values():
        _keys(frame, ("id", "path", "time", "regions"))
        _require(_text(frame["path"]) and (posixpath.isabs(frame["path"]) or ntpath.isabs(frame["path"])),
                 "Frame paths must be absolute")
        _require(_number(frame["time"]) and frame["time"] >= 0, "Invalid frame time")
        for region in _indexed(frame["regions"], "region").values():
            _keys(region, ("id", "box"))
            box = region["box"]
            _require(isinstance(box, list) and len(box) == 4 and all(_number(x) for x in box),
                     "Region box must contain four finite numbers")
            _require(0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1,
                     "Region box must be normalized xyxy")
    bindings = _indexed(schema["bindings"], "binding")
    for binding in bindings.values():
        _keys(binding, ("id", "anchors"), ("description",))
        if "description" in binding:
            _require(_text(binding["description"]), "Binding description must be text")
        _references(binding["anchors"], frames)
    slots = _indexed(schema["slots"], "slot")
    _require(bool(slots), "At least one evidence slot is required")
    for slot in slots.values():
        _keys(slot, ("id", "bindings", "interval", "query", "type", "support"))
        ids = slot["bindings"]
        _require(isinstance(ids, list) and all(isinstance(x, str) and x in bindings for x in ids)
                 and len(ids) == len(set(ids)), "Invalid slot bindings")
        interval = slot["interval"]
        _require(isinstance(interval, list) and len(interval) == 2 and all(_number(x) for x in interval)
                 and 0 <= interval[0] <= interval[1], "Invalid temporal interval")
        _require(_text(slot["query"]), "Empty slot query")
        _require(slot["type"] in ("text", "number", "boolean", "object_id", "object_set", "answer_set"),
                 "Unsupported slot type")
        _references(slot["support"], frames)
        _require(all(interval[0] <= frames[r["frame_id"]]["time"] <= interval[1]
                     for r in slot["support"]), "Support frame outside slot interval")
    return _copy(schema)


def _slot(schema, slot_id):
    slots = {s["id"]: s for s in schema["slots"]}
    _require(isinstance(slot_id, str) and slot_id in slots, "Unknown slot ID")
    return slots[slot_id]


def slot_key(schema: dict, slot_id: str) -> str:
    schema = validate_schema(schema)
    slot = _slot(schema, slot_id)
    frame_ids = {r["frame_id"] for r in slot["support"]}
    frame_ids.update(r["frame_id"] for binding in schema["bindings"] for r in binding["anchors"])
    key_slot = {k: slot[k] for k in ("bindings", "interval", "query", "type", "support")}
    key_slot["bindings"] = sorted(key_slot["bindings"])
    key_slot["support"] = sorted(key_slot["support"], key=_json)
    for binding in schema["bindings"]:
        binding["anchors"] = sorted(binding["anchors"], key=_json)
    for frame in schema["frames"]:
        frame["regions"] = sorted(frame["regions"], key=lambda r: r["id"])
    return _json({"video_id": schema["video_id"], "slot": key_slot,
                  "bindings": sorted(schema["bindings"], key=lambda x: x["id"]),
                  "frames": sorted((f for f in schema["frames"] if f["id"] in frame_ids), key=lambda x: x["id"])})


@dataclass(frozen=True)
class PreparedPlan:
    _schema: dict = field(repr=False)
    _plan: dict = field(repr=False)
    order: tuple

    @property
    def schema(self):
        return _copy(self._schema)

    @property
    def plan(self):
        return _copy(self._plan)


def prepare_plan(schema: dict, plan: dict) -> PreparedPlan:
    """Prepare before evidence sampling; reject embedded values and extra fields."""
    schema = validate_schema(schema)
    _keys(plan, ("nodes", "answer_node"))
    nodes = _indexed(plan["nodes"], "node")
    slots = {s["id"] for s in schema["slots"]}
    _require(not slots.intersection(nodes), "Node and slot IDs must be disjoint")
    _require(isinstance(plan["answer_node"], str) and plan["answer_node"] in nodes,
             "Answer node must be an inference node")
    for node in nodes.values():
        _keys(node, ("id", "parents", "operation"))
        parents = node["parents"]
        _require(isinstance(parents, list) and parents and
                 all(isinstance(p, str) and p in slots | set(nodes) for p in parents)
                 and len(parents) == len(set(parents)), "Invalid declared parents")
        _require(_text(node["operation"]), "Operation must be a nonempty rule/query")
    ready, order = set(slots), []
    while len(order) < len(nodes):
        next_ids = sorted(n for n in nodes if n not in ready and set(nodes[n]["parents"]) <= ready)
        _require(bool(next_ids), "Plan contains a cycle")
        order.extend(next_ids)
        ready.update(next_ids)
    return PreparedPlan(schema, _copy(plan), tuple(order))


def validate_evidence(schema: dict, evidence: dict) -> dict:
    schema = validate_schema(schema)
    _require(isinstance(evidence, dict) and set(evidence) == {s["id"] for s in schema["slots"]},
             "Evidence must contain exactly the shared slots")
    evidence = _copy(evidence)
    bindings = {b["id"] for b in schema["bindings"]}
    for slot in schema["slots"]:
        record = evidence[slot["id"]]
        _keys(record, ("value", "support"))
        _require(isinstance(record["support"], list) and
                 sorted(_json(r) for r in record["support"]) == sorted(_json(r) for r in slot["support"]),
                 "Evidence must retain exactly the fixed slot support")
        value, kind = record["value"], slot["type"]
        valid = {"text": lambda: _text(value), "number": lambda: _number(value),
                 "boolean": lambda: isinstance(value, bool),
                 "object_id": lambda: isinstance(value, str) and value in bindings,
                 "object_set": lambda: isinstance(value, list) and all(isinstance(x, str) and x in bindings for x in value)
                 and len(value) == len(set(value)),
                 "answer_set": lambda: isinstance(value, list) and all(_text(x) for x in value)
                 and len(value) == len(set(value))}[kind]()
        _require(valid, "Evidence value violates slot type/binding constraints")
        if kind in ("object_set", "answer_set"):
            record["value"] = sorted(value)
    return evidence


@dataclass(frozen=True)
class Execution:
    prepared: PreparedPlan
    _evidence: dict = field(repr=False)
    _outputs: dict = field(repr=False)
    _executor: Any = field(repr=False, compare=False)
    recomputed_nodes: tuple = ()

    @property
    def evidence(self):
        return _copy(self._evidence)

    @property
    def outputs(self):
        return _copy(self._outputs)

    @property
    def answer(self):
        return _copy(self._outputs[self.prepared._plan["answer_node"]])


def _run(prepared, evidence, executor, cached=None, affected=None):
    nodes = {n["id"]: n for n in prepared._plan["nodes"]}
    outputs, recomputed = _copy(cached or {}), []
    values = {s: r["value"] for s, r in evidence.items()}
    for node_id in prepared.order:
        node = nodes[node_id]
        if affected is None or node_id in affected:
            parents = {p: _copy(values[p]) for p in node["parents"]}
            outputs[node_id] = _copy(executor(prepared._schema["question"], node["operation"], parents))
            recomputed.append(node_id)
        values[node_id] = outputs[node_id]
    return Execution(prepared, _copy(evidence), outputs, executor, tuple(recomputed))


def execute(prepared: PreparedPlan, evidence: dict, executor: Callable) -> Execution:
    _require(isinstance(prepared, PreparedPlan) and callable(executor), "Prepared plan/executor required")
    return _run(prepared, validate_evidence(prepared._schema, evidence), executor)


def replay(original: Execution, slot_id: str, replacement: dict, executor: Callable) -> Execution:
    _require(executor is original._executor, "Replay must use the same frozen executor object")
    _slot(original.prepared._schema, slot_id)
    evidence = original.evidence
    evidence[slot_id] = _copy(replacement)
    evidence = validate_evidence(original.prepared._schema, evidence)
    affected = {slot_id}
    nodes = {n["id"]: n for n in original.prepared._plan["nodes"]}
    for node_id in original.prepared.order:
        if affected.intersection(nodes[node_id]["parents"]):
            affected.add(node_id)
    return _run(original.prepared, evidence, executor, original._outputs, affected)


def _score(scorer, answer):
    score = scorer(_copy(answer))
    _keys(score, ("accuracy", "exact", "format_ok"))
    _require(_number(score["accuracy"]) and 0 <= score["accuracy"] <= 1
             and isinstance(score["exact"], bool) and isinstance(score["format_ok"], bool), "Invalid answer score")
    _require(not score["exact"] or score["accuracy"] == 1, "Exact answer must have accuracy 1")
    return score


@dataclass(frozen=True)
class Repair:
    donor: Execution
    receiver: Execution
    repaired: Optional[Execution]
    slot_id: str
    gain: float
    accepted: bool
    reason: str


def compare_transfer(donor: Execution, receiver: Execution, slot_id: str,
                     executor: Callable, scorer: Callable, support_status: str) -> Repair:
    _require(donor is not receiver, "Donor and receiver must be distinct executions")
    _require(donor._executor is executor and receiver._executor is executor, "Executor snapshot mismatch")
    ds, rs = donor.prepared._schema, receiver.prepared._schema
    _require(ds["video_id"] == rs["video_id"] and ds["question"] == rs["question"],
             "Transfer must use the same video and question")
    _require(slot_key(ds, slot_id) == slot_key(rs, slot_id), "Incompatible shared slot key")
    old, new = receiver._evidence[slot_id], donor._evidence[slot_id]
    _require(_json(old["value"]) != _json(new["value"]), "A transfer must change one evidence value")
    before = _score(scorer, receiver.answer)
    _require(not before["exact"] and not _score(scorer, donor.answer)["exact"],
             "Primary repair pool requires two unsuccessful executions")
    _require(support_status in ("supported", "unsupported", "uncertain"), "Invalid support screening result")
    if support_status != "supported":
        return Repair(donor, receiver, None, slot_id, 0.0, False, support_status)
    repaired = replay(receiver, slot_id, new, executor)
    after = _score(scorer, repaired.answer)
    gain = float(after["accuracy"] - before["accuracy"])
    accepted = gain > 0 and after["exact"] and after["format_ok"]
    reason = "accepted" if accepted else "nonpositive_gain" if gain <= 0 else "not_exact_or_malformed"
    return Repair(donor, receiver, repaired, slot_id, gain, accepted, reason)


def evidence_messages(schema: dict, slot_id: str) -> list:
    """Use identically for acquisition and evidence-only supervised training."""
    schema = validate_schema(schema)
    slot = _slot(schema, slot_id)
    frame_ids = {r["frame_id"] for r in slot["support"]}
    frames = sorted((f for f in schema["frames"] if f["id"] in frame_ids), key=lambda f: (f["time"], f["id"]))
    context = {"question": schema["question"], "slot": slot, "bindings": schema["bindings"], "frames": frames}
    content = [{"type": "image", "image": f["path"]} for f in frames]
    content.append({"type": "text", "text": "Observe only the fixed visual support. Return only a JSON value matching the slot type.\n" + _json(context)})
    return [{"role": "user", "content": content}]


def local_example(repair: Repair) -> dict:
    _require(repair.accepted and repair.repaired is not None, "Only accepted repairs are training examples")
    return {"kind": "local", "messages": evidence_messages(repair.receiver.prepared._schema, repair.slot_id),
            "target": _json(repair.repaired._evidence[repair.slot_id]["value"]), "weight": repair.gain,
            "metadata": {"video_id": repair.receiver.prepared._schema["video_id"], "slot_id": repair.slot_id}}


def deduplicate_examples(examples: list) -> list:
    groups = {}
    for example in examples:
        _require(example.get("kind") == "local" and _number(example.get("weight"))
                 and 0 < example["weight"] <= 1, "Only positive local examples can be deduplicated")
        _require(isinstance(example.get("messages"), list) and example["messages"] and
                 isinstance(example.get("target"), str), "Invalid local messages/target")
        key = _json([example["messages"], example["target"]])
        groups.setdefault(key, []).append(example)
    result = []
    for key in sorted(groups):
        group = groups[key]
        item = _copy(group[0])
        counts = [e.get("metadata", {}).get("accepted_transfer_count", 1) for e in group]
        _require(all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in counts), "Invalid transfer count")
        item["weight"] = math.fsum(e["weight"] * c for e, c in zip(group, counts)) / sum(counts)
        item.setdefault("metadata", {})["accepted_transfer_count"] = sum(counts)
        result.append(item)
    return result


def serialize_execution(execution: Execution) -> str:
    records = [{"id": s, "evidence": execution._evidence[s]["value"]} for s in sorted(execution._evidence)]
    records.extend({"id": n, "inference": execution._outputs[n]} for n in execution.prepared.order)
    answer = execution.answer
    if isinstance(answer, dict) and "answer" in answer:
        answer = answer["answer"]
    answer_text = ", ".join(sorted(answer)) if isinstance(answer, list) and all(isinstance(x, str) for x in answer) else str(answer)
    match = re.search(r"<answer>(.*?)</answer>", answer_text, flags=re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
    _require("<" not in answer_text and ">" not in answer_text, "Terminal answer contains malformed markup")
    body = "\n".join(_json(r).replace("<", "\\u003c").replace(">", "\\u003e") for r in records)
    return "<think>\n" + body + "\n</think>\n<answer>" + answer_text + "</answer>"


def distillation_example(repair: Repair, messages: list) -> dict:
    _require(repair.accepted and repair.repaired is not None, "Only accepted repairs can be distilled")
    _require(isinstance(messages, list) and messages, "Ordinary input messages are required")
    return {"kind": "distill", "messages": _copy(messages), "target": serialize_execution(repair.repaired),
            "weight": 1.0, "metadata": {"video_id": repair.receiver.prepared._schema["video_id"]}}
