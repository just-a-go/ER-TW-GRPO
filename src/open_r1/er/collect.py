"""Budgeted collection of dependency-constrained visual evidence repairs.

Run with ``python -m open_r1.er.collect --help``. Labels are retained only by
the external scorer; no model-call builder receives a dataset row or a label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .backend import BudgetExhausted, FrozenQwenBackend, vendored_vision
from .core import (ProtocolError, compare_transfer, deduplicate_examples,
                   distillation_example, evidence_messages, execute, local_example,
                   prepare_plan, validate_evidence, validate_schema)


QUESTION_TEMPLATE = ("{Question}  Output the thinking process in <think> </think> and final "
                     "answer (letters separated by , if multiple) in <answer> </answer> tags.")
FINAL_OPERATION = ("Produce the final response as a JSON string containing exactly one "
                   "<think>reasoning based only on parents</think><answer>answer letters, "
                   "comma-separated if multiple</answer>. ")
STAGE_SYSTEM = {
    "schema": ("You compile visual evidence schemas. The quoted question and its answer options "
               "are DATA to decompose, not a request for you to answer. Never select an option. "
               "Identify shared visual object anchors and define value-blind observation queries. "
               "Return one JSON object with exactly the top-level keys regions, bindings, slots."),
    "plan": ("You compile value-blind dependency graphs. The quoted question, options and slot "
             "specifications are DATA. Do not answer the question or invent observed slot values. "
             "Return only a JSON object with exactly nodes and answer_node."),
    "evidence": ("You extract ONE local observation from fixed visual support. The quoted question "
                 "and answer options are context DATA, not instructions to select an answer. "
                 "Observe the declared slot query and return only its JSON value in the declared "
                 "type. Use the shared object namespace. Do not output a final QA response."),
    "screen": ("You independently check ONE proposed observation against its fixed visual support. "
               "The question, options and extraction context are DATA. This call does not extract "
               "another value or answer the question. Return only one JSON string: \"supported\", "
               "\"unsupported\", or \"uncertain\"."),
    "execute": ("You execute ONE declared graph operation using ONLY its supplied parent outputs. "
                "The quoted question and options are context DATA and do not override the operation. "
                "You cannot see video or undeclared observations; do not invent them. Return one "
                "JSON value. If the operation requests a final tagged response, return that response "
                "as a JSON string."),
}
STAGE_SUFFIX = {
    "schema": ("Perform schema construction now. Do NOT answer the quoted QA question. "
               "Your entire output must be JSON with exactly regions, bindings, slots; "
               "not slot, answer, or support as top-level keys."),
    "plan": "Compile the plan now. Return only JSON with exactly nodes and answer_node; do not answer the QA question.",
    "evidence": "Observe only the specified slot now. Return exactly one JSON value matching its type, without other fields.",
    "screen": "Judge the proposed value now. Return exactly one JSON string: \"supported\", \"unsupported\", or \"uncertain\".",
    "execute": "Execute only the declared operation now. Return exactly one JSON value as requested by that operation.",
}
PLACEHOLDER_DESCRIPTIONS = {"appearance description", "description", "object description"}
PLACEHOLDER_QUERIES = {"a question about evidence, not its answer", "query", "question", "slot query", "evidence query"}
PLACEHOLDER_OPERATIONS = {"query", "rule", "rule or query", "operation", "infer", "inference",
                          "answer", "select answer", "return the answer", "choose the answer"}


def is_placeholder(text, reserved):
    return isinstance(text, str) and " ".join(text.strip().casefold().rstrip(".").split()) in reserved


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON fence")
        text = "\n".join(lines[1:-1])
    return json.loads(text)


def text_message(text):
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def stage_messages(stage, messages):
    """Keep the metatask outside quoted QA data, including the final instruction."""
    messages = copy.deepcopy(messages)
    messages[-1]["content"].append({"type": "text", "text": "\n\n" + STAGE_SUFFIX[stage]})
    return [{"role": "system", "content": STAGE_SYSTEM[stage]}] + messages


def media_limits(messages, min_pixels, max_pixels):
    messages = copy.deepcopy(messages)
    for message in messages:
        if isinstance(message["content"], str):
            continue
        for item in message["content"]:
            if item["type"] == "image":
                item.update(min_pixels=min_pixels, max_pixels=max_pixels)
    return messages


def original_messages(question, video):
    return [{"role": "user", "content": [
        {"type": "video", "video": str(Path(video).resolve())},
        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=question)}]}]


def schema_messages(question, frames, max_slots):
    """The only schema inputs are the original question and frozen policy frames."""
    inventory = [{"id": f["id"], "time": f["time"]} for f in frames]
    prompt = ("Construct a shared visual evidence interface, without answering the question. "
              "Images are supplied in the listed order. Establish stable object IDs through "
              "visible anchor regions. A binding description may identify appearance only; "
              "it must not assert task-specific events or a final answer. Create between 1 and "
              f"{max_slots} evidence slots, each querying a local observation needed by the question. "
              "Slot keys contain queries and search intervals, never proposed observations, inferred "
              "event times, or precomputed answers. Every slot uses a fixed nonempty subset of the "
              "supplied whole frames; do not put region_id in slot support. Regions are used only "
              "for shared object anchors. All times are seconds from the clip. Output JSON only. "
              "The required top-level fields and their types are:\n"
              "regions: an array of anchor-region objects. Each has frame_id (one listed frame ID), "
              "id (a region ID unique within that frame), and box (four normalized numbers in "
              "x_min,y_min,x_max,y_max order enclosing an actual visible object).\n"
              "bindings: an array of object-binding objects. Each has id (a shared object ID), "
              "description (the actual object's visible color, shape and material as identifiable), "
              "and anchors (a nonempty array of objects with frame_id and region_id).\n"
              "slots: an array of evidence-slot objects. Each has id (a slot ID), bindings (an array "
              "of involved shared object IDs), interval (two seconds enclosing its search support), "
              "query (a concrete local observation question about those objects relevant to this "
              "QA task), type (an allowed type name), and support (a nonempty array of objects "
              "with only frame_id).\n"
              "Fill these fields from this video's objects and this question. Do not copy generic "
              "placeholders, invent appearance, or write a generic instruction in place of a query. "
              "Allowed types: text, number, boolean, object_id, object_set. Object values use only "
              "the shared bindings. Include every frame needed to observe the queried event; "
              "interval endpoints must contain all support frame times. Do not include other keys.\n"
              + json.dumps({"question": question, "frames": inventory}, ensure_ascii=False))
    return stage_messages("schema", [{"role": "user", "content":
             [{"type": "image", "image": f["path"]} for f in frames]
             + [{"type": "text", "text": prompt}]}])


def merge_schema(response, question, video_id, frames, max_slots):
    if not isinstance(response, dict) or set(response) != {"regions", "bindings", "slots"}:
        raise ProtocolError("Schema response must contain exactly regions, bindings, slots")
    if any(not isinstance(response[key], list) for key in ("regions", "bindings", "slots")):
        raise ProtocolError("Schema regions, bindings and slots must be lists")
    if not 1 <= len(response["slots"]) <= max_slots or not 1 <= len(response["bindings"]) <= 32:
        raise ProtocolError("Schema size exceeds fixed bounds")
    for binding in response["bindings"]:
        if not isinstance(binding, dict) or not isinstance(binding.get("description"), str) or not binding["description"].strip():
            raise ProtocolError("Collected object bindings require a concrete visual appearance description")
        if is_placeholder(binding["description"], PLACEHOLDER_DESCRIPTIONS):
            raise ProtocolError("Placeholder object appearance is not a grounded binding")
    frames = copy.deepcopy(frames)
    by_id = {frame["id"]: frame for frame in frames}
    if len(response["regions"]) > 128:
        raise ProtocolError("Too many anchor regions")
    for region in response["regions"]:
        if not isinstance(region, dict) or set(region) != {"frame_id", "id", "box"}:
            raise ProtocolError("Invalid anchor region fields")
        if not isinstance(region["frame_id"], str) or region["frame_id"] not in by_id:
            raise ProtocolError("Anchor frame does not exist")
        by_id[region["frame_id"]]["regions"].append({"id": region["id"], "box": region["box"]})
    for slot in response["slots"]:
        if not isinstance(slot, dict) or set(slot) != {"id", "bindings", "interval", "query", "type", "support"}:
            raise ProtocolError("Invalid evidence slot fields")
        if slot["type"] == "answer_set":
            raise ProtocolError("Evidence acquisition must not directly target answer options")
        if is_placeholder(slot["query"], PLACEHOLDER_QUERIES):
            raise ProtocolError("Placeholder evidence query is not specific to the video question")
        if not isinstance(slot["support"], list) or any(
                not isinstance(ref, dict) or set(ref) != {"frame_id"} for ref in slot["support"]):
            raise ProtocolError("Collector uses fixed whole-frame slot support")
    return validate_schema({"video_id": video_id, "question": question, "frames": frames,
                            "bindings": response["bindings"], "slots": response["slots"]})


def plan_messages(schema, max_nodes):
    # Frame paths and images are deliberately absent from every plan call.
    interface = {"question": schema["question"], "bindings": schema["bindings"],
                 "slots": schema["slots"]}
    return stage_messages("plan", text_message(
        "Create a value-blind directed acyclic inference plan. You have slot specifications but "
        "no observed values. Write queries or inference rules only: never assert a candidate "
        "observation, invent an event, hardcode an option as the answer, or embed an answer in an "
        "operation. Internal calls receive only the question, their operation and declared parent "
        "values. Each operation must be self-contained: include the fixed object-ID-to-appearance "
        "mapping and slot query/type needed to interpret its declared parents, including IDs that "
        "can occur in object-valued parents. Copy these meanings only from the supplied shared "
        "bindings and slot specifications. Do not embed observed events, candidate values, "
        "unrelated evidence or a precomputed answer. List all needed dependencies; raw video will "
        "not be available. Use at least one "
        f"and at most {max_nodes} internal nodes. Return JSON with exactly two top-level fields: "
        "nodes (a nonempty array of objects, each with id: a unique node ID string, parents: a "
        "nonempty array of declared slot or node ID strings, and operation: a concrete inference "
        "rule string specific to the question and those parents); answer_node (the ID string of "
        "the terminal node). Do not emit generic operations such as Query or Rule. "
        "Node/slot IDs must be disjoint. The answer node must depend on "
        "evidence. The final operation must derive the selected answer options from its parents.\n"
        + json.dumps(interface, ensure_ascii=False)))


class RestrictedExecutor:
    """A single immutable model snapshot, with no reference to schemas or dataset labels."""
    def __init__(self, backend):
        self.backend = backend
        self.fingerprint = backend.fingerprint

    def __call__(self, question, operation, parent_outputs):
        if self.backend.fingerprint != self.fingerprint:
            raise ProtocolError("Frozen executor changed during collection")
        prompt = ("Execute this one inference operation using only its declared parent outputs. "
                  "No raw visual input or other observations are available. Do not invent missing "
                  "evidence. Return a JSON value only; final answers must use the requested tagged "
                  "response inside a JSON string.\n" + json.dumps(
                      {"question": question, "operation": operation, "parents": parent_outputs},
                      ensure_ascii=False))
        response = self.backend.generate(stage_messages("execute", text_message(prompt)),
                                         stage="execute", deterministic=True)
        if response.startswith("<think>"):
            return response
        return parse_json(response)


class BaselineScorer:
    """Inherited rewards with the method's auxiliary-only malformed-output gate."""
    def __init__(self, solution, accuracy_fn, format_fn):
        self._solution = solution
        self._accuracy = accuracy_fn
        self._format = format_fn

    def __call__(self, answer):
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        completions = [[{"role": "assistant", "content": text}]]
        format_ok = bool(self._format(completions)[0] == 1.0)
        if not format_ok:
            return {"accuracy": 0.0, "exact": False, "format_ok": False}
        accuracy = float(self._accuracy(completions, [self._solution])[0])
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        truth = re.search(r"<answer>(.*?)</answer>", self._solution, re.DOTALL)
        target = truth.group(1) if truth else self._solution
        predicted = {option.strip() for option in match.group(1).replace(" ", "").split(",")} if match else set()
        expected = {option.strip() for option in target.replace(" ", "").split(",")}
        exact = bool(predicted and "" not in predicted and predicted == expected)
        return {"accuracy": accuracy, "exact": exact, "format_ok": format_ok}


@dataclass(frozen=True)
class CollectionConfig:
    candidates: int = 8
    transfers_per_question: int = 8
    max_slots: int = 4
    max_nodes: int = 8
    seed: int = 42
    min_pixels: int = 3136
    max_pixels: int = 12845056


class Collector:
    def __init__(self, backend, config):
        self.backend = backend
        self.config = config
        self.executor = RestrictedExecutor(backend)
        self.rng = random.Random(config.seed)
        self.stats = Counter()

    def limited(self, messages):
        return media_limits(messages, self.config.min_pixels, self.config.max_pixels)

    def evidence_input(self, schema, slot_id):
        # Apply exactly the same stage framing to acquisition and local supervision.
        slot = next(slot for slot in schema["slots"] if slot["id"] == slot_id)
        allowed_ids = json.dumps([binding["id"] for binding in schema["bindings"]], ensure_ascii=False)
        shapes = {
            "text": "Return one JSON string containing the local observation, not a wrapper object.",
            "number": "Return one finite JSON number, not a string or wrapper object.",
            "boolean": "Return the JSON literal true or false, without quotation marks or a wrapper object.",
            "object_id": ("Return exactly one shared object ID as a JSON string enclosed in double quotes. "
                          "Do not return an object with an id field. Allowed shared IDs: " + allowed_ids),
            "object_set": ("Return a JSON array of distinct shared object ID strings, without a wrapper object. "
                           "Allowed shared IDs: " + allowed_ids),
        }
        messages = self.limited(evidence_messages(schema, slot_id))
        messages[0]["content"].append({"type": "text", "text": "\nRequired value shape: " + shapes[slot["type"]]
            + " If the fixed support does not establish the observation, return the JSON literal null. "
              "Null is rejected as unusable evidence; do not guess or fabricate a value."})
        return stage_messages("evidence", messages)

    def collect_question(self, *, question, video, video_id, frames, scorer):
        """No dataset row or label is accepted; scorer is an opaque external function."""
        cfg = self.config
        audit = {"video_id": video_id, "plans": [], "candidates": [], "transfers": []}
        schema_raw = parse_json(self.backend.generate(
            self.limited(schema_messages(question, frames, cfg.max_slots)),
            stage="schema", deterministic=True))
        schema = merge_schema(schema_raw, question, video_id, frames, cfg.max_slots)
        audit["schema"] = schema
        prepared = []
        # Complete EVERY plan-generation attempt before any evidence is generated.
        for candidate in range(cfg.candidates):
            try:
                plan = parse_json(self.backend.generate(plan_messages(schema, cfg.max_nodes),
                                                        stage="plan", deterministic=False))
                if not isinstance(plan, dict) or set(plan) != {"nodes", "answer_node"}:
                    raise ProtocolError("Plan must contain exactly nodes and answer_node")
                if not isinstance(plan["nodes"], list) or not 1 <= len(plan["nodes"]) <= cfg.max_nodes:
                    raise ProtocolError("Plan size exceeds fixed node limit")
                for node in plan["nodes"]:
                    if not isinstance(node, dict) or set(node) != {"id", "parents", "operation"}:
                        raise ProtocolError("Invalid plan node fields")
                    if not isinstance(node["operation"], str) or not node["operation"].strip():
                        raise ProtocolError("Node operation must be a nonempty rule/query")
                    if is_placeholder(node["operation"], PLACEHOLDER_OPERATIONS):
                        raise ProtocolError("Placeholder node operation does not specify executable inference")
                    if node["id"] == plan["answer_node"]:
                        node["operation"] = FINAL_OPERATION + node["operation"]
                prepared.append((candidate, prepare_plan(schema, plan)))
                audit["plans"].append({"candidate": candidate, "plan": plan})
            except (ValueError, TypeError, KeyError, ProtocolError) as error:
                self.stats["plan_rejected"] += 1
                audit["plans"].append({"candidate": candidate, "rejected": str(error)})
        executions = []
        for candidate, plan in prepared:
            try:
                evidence = {}
                for slot in schema["slots"]:
                    value = parse_json(self.backend.generate(
                        self.evidence_input(schema, slot["id"]),
                        stage="evidence", deterministic=False))
                    evidence[slot["id"]] = {"value": value, "support": copy.deepcopy(slot["support"])}
                evidence = validate_evidence(schema, evidence)
                execution = execute(plan, evidence, self.executor)
                score = scorer(execution.answer)
                executions.append((candidate, execution, score))
                audit["candidates"].append({"candidate": candidate, "evidence": evidence,
                                            "outputs": execution.outputs, "score": score})
                self.stats["candidates_executed"] += 1
            except (ValueError, TypeError, KeyError, ProtocolError) as error:
                self.stats["candidate_rejected"] += 1
                audit["candidates"].append({"candidate": candidate, "rejected": str(error)})
        failures = [(index, execution) for index, execution, score in executions if not score["exact"]]
        pairs = []
        for donor_index, donor in failures:
            for receiver_index, receiver in failures:
                if donor_index == receiver_index:
                    continue
                for slot in schema["slots"]:
                    slot_id = slot["id"]
                    if donor.evidence[slot_id]["value"] != receiver.evidence[slot_id]["value"]:
                        pairs.append((donor_index, donor, receiver_index, receiver, slot_id))
        # Eligibility uses labels only for the failure gate. No gain/answer ranking.
        chosen = self.rng.sample(pairs, min(cfg.transfers_per_question, len(pairs)))
        local, distill = [], []
        for donor_index, donor, receiver_index, receiver, slot_id in chosen:
            entry = {"donor": donor_index, "receiver": receiver_index, "slot_id": slot_id}
            self.stats["transfers_attempted"] += 1
            try:
                screen_messages = self.limited(evidence_messages(schema, slot_id))
                # Independent call: only the proposed observation, its fixed support and schema.
                screen_messages[0]["content"].append({"type": "text", "text":
                    "For this call, do not generate another value. Independently check the proposed "
                    "value against the same fixed visual support. Reply with JSON string \"supported\", "
                    "\"unsupported\", or \"uncertain\" only. Supported requires clear visual support; "
                    "uncertain applies to occlusion, ambiguous identities or unseen events. Proposed value: "
                    + json.dumps(donor.evidence[slot_id]["value"], ensure_ascii=False)})
                status = parse_json(self.backend.generate(stage_messages("screen", screen_messages),
                                                          stage="screen", deterministic=True))
                if status not in ("supported", "unsupported", "uncertain"):
                    raise ProtocolError("Invalid support-screen result")
                repair = compare_transfer(donor, receiver, slot_id, self.executor, scorer, status)
                entry.update(screen=status, accepted=repair.accepted, reason=repair.reason, gain=repair.gain)
                self.stats[f"transfer_{repair.reason}"] += 1
                if repair.accepted:
                    example = local_example(repair)
                    example["messages"] = self.evidence_input(repair.receiver.prepared.schema, repair.slot_id)
                    local.append(example)
                    distill.append(distillation_example(repair, original_messages(question, video)))
                    entry["repaired_outputs"] = repair.repaired.outputs
                    self.stats["repairs_accepted"] += 1
            except BudgetExhausted:
                # Preserve earlier admitted transfers when the fixed budget ends mid-question.
                audit["budget_exhausted"] = True
                entry.update(accepted=False, reason="generation_budget_exhausted")
                audit["transfers"].append(entry)
                break
            except (ValueError, TypeError, KeyError, ProtocolError) as error:
                entry.update(accepted=False, reason=str(error))
                self.stats["transfer_protocol_rejected"] += 1
            audit["transfers"].append(entry)
        self.stats["questions_completed"] += 1
        return local, distill, audit


def load_training_rows(dataset_path, train_samples, max_questions):
    from datasets import Dataset

    # Identical mixed-question selection to baseline create_dataset_from_jsonl_simple.
    dataset = Dataset.from_json(str(Path(dataset_path).resolve(strict=True)))
    if len(dataset) > train_samples:
        dataset = dataset.shuffle(seed=42).select(range(train_samples))
    return dataset.select(range(min(max_questions, len(dataset))))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--dataset", required=True, help="Existing local baseline training JSON/JSONL")
    result.add_argument("--output-dir", required=True)
    result.add_argument("--video-root", help="Resolve relative dataset video paths under this directory")
    result.add_argument("--train-samples", type=int, default=2000)
    result.add_argument("--max-questions", type=int, default=16)
    result.add_argument("--candidates", type=int, default=8)
    result.add_argument("--transfers-per-question", type=int, default=8)
    result.add_argument("--max-slots", type=int, default=4)
    result.add_argument("--max-nodes", type=int, default=8)
    result.add_argument("--node-max-tokens", type=int, default=512)
    result.add_argument("--schema-max-tokens", type=int, default=2048)
    result.add_argument("--plan-max-tokens", type=int, default=1024)
    result.add_argument("--total-generation-tokens", type=int, default=100000)
    result.add_argument("--max-prompt-tokens", type=int, default=16384)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    result.add_argument("--temperature", type=float, default=0.8)
    result.add_argument("--min-pixels", type=int, default=3136)
    result.add_argument("--max-pixels", type=int, default=12845056)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("train_samples", "max_questions", "candidates", "max_slots", "max_nodes",
                 "node_max_tokens", "schema_max_tokens", "plan_max_tokens", "total_generation_tokens",
                 "max_prompt_tokens", "min_pixels", "max_pixels"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.transfers_per_question < 0 or args.temperature <= 0 or args.min_pixels > args.max_pixels:
        raise ValueError("Invalid transfer budget, temperature or pixel limits")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit").mkdir()
    # The inherited reward module initializes loggers on import. Keep this round separate.
    os.environ.setdefault("PRIVATE_DATA_ROOT", str(output))
    os.environ.setdefault("WANDB_NAME", "er-collector")
    os.environ["DEBUG_MODE"] = "false"
    os.environ.setdefault("SAMPLE_MODE", "true")
    vendored_vision()  # Establish vendor precedence before grpo imports its trainer.
    from open_r1.grpo import accuracy_reward, format_reward

    rows = load_training_rows(args.dataset, args.train_samples, args.max_questions)
    config = CollectionConfig(**{name: getattr(args, name) for name in CollectionConfig.__dataclass_fields__})
    manifest = {"status": "initializing", "arguments": vars(args), "protocol": asdict(config),
                "selection": {"baseline_shuffle_seed": 42, "train_samples": args.train_samples,
                              "max_questions": args.max_questions, "selected_count": len(rows)},
                "budget_policy": "Fixed maxima, uniform eligible single-slot transfers, no outcome-based retries",
                "source_dataset": str(Path(args.dataset).resolve()),
                "source_dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                "generation_audit": "generations.jsonl",
                "labels": "External scorer only; absent from all prompts and serialized targets"}
    write_json(output / "manifest.json", manifest)
    backend = FrozenQwenBackend(args.model, node_max_tokens=args.node_max_tokens,
                               schema_max_tokens=args.schema_max_tokens, plan_max_tokens=args.plan_max_tokens,
                               total_generation_tokens=args.total_generation_tokens,
                               max_prompt_tokens=args.max_prompt_tokens, seed=args.seed,
                               device=args.device, dtype=args.dtype, temperature=args.temperature,
                               max_pixels=args.max_pixels, min_pixels=args.min_pixels,
                               generation_audit_path=output / "generations.jsonl")
    collector = Collector(backend, config)
    manifest.update(status="collecting", model=backend.identity, executor_fingerprint=backend.fingerprint)
    write_json(output / "manifest.json", manifest)
    local, distill = [], []
    stop_reason = "question_budget_completed"
    try:
        for index, row in enumerate(rows):
            if not all(key in row for key in ("problem", "video", "solution")):
                raise ValueError("Training rows must contain problem, video and solution")
            video = Path(row["video"])
            if not video.is_absolute():
                video = Path(args.video_root or Path(args.dataset).resolve().parent) / video
            key = f"q{index:05d}"
            backend.audit_context = key
            try:
                frames, support_audit = backend.export_support(video, output / "support" / key)
                examples, targets, audit = collector.collect_question(
                    question=row["problem"], video=str(video), video_id=str(video.resolve()), frames=frames,
                    scorer=BaselineScorer(row["solution"], accuracy_reward, format_reward))
                local.extend(examples)
                distill.extend(targets)
                audit["support"] = support_audit
                write_json(output / "audit" / f"{key}.json", audit)
                if audit.get("budget_exhausted") or backend.used_tokens >= args.total_generation_tokens:
                    stop_reason = "generation_budget_exhausted"
                    break
            except BudgetExhausted:
                stop_reason = "generation_budget_exhausted"
                write_json(output / "audit" / f"{key}.json", {"status": stop_reason})
                break
            except (OSError, ValueError, TypeError, KeyError, ProtocolError) as error:
                collector.stats["questions_rejected"] += 1
                write_json(output / "audit" / f"{key}.json", {"status": "rejected", "reason": str(error)})
            print(f"ER question {index + 1}/{len(rows)}: accepted={collector.stats['repairs_accepted']}", flush=True)
        manifest["status"] = "complete"
    except Exception:
        manifest["status"] = "failed"
        stop_reason = "unhandled_error"
        raise
    finally:
        local = deduplicate_examples(local)
        # Every complete execution has unit weight; identical targets need no duplicate copies.
        distinct_distill = {}
        for record in distill:
            identity = json.dumps([record["messages"], record["target"]], sort_keys=True)
            distinct_distill.setdefault(identity, record)
        distill = list(distinct_distill.values())
        write_jsonl(output / "local.jsonl", local)
        write_jsonl(output / "distill.jsonl", distill)
        manifest.update(stop_reason=stop_reason, statistics=dict(collector.stats),
                        compute=backend.statistics(), local_examples=len(local), distill_examples=len(distill))
        write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
