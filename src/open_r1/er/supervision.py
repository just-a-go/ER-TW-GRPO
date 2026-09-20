"""Evidence-only collation and sequence-normalized supervised objectives.

This module does not import the GRPO trainer or alter the baseline vision helper.
"""

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RepairDataset(Dataset):
    """Read typed collector JSONL, failing closed on invalid or mixed streams."""

    def __init__(self, path, kind="local"):
        if kind not in {"local", "distill"}:
            raise ValueError("kind must be local or distill")
        self.records = []
        self.fingerprint = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    self._validate(row, kind)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                record = copy.deepcopy(row)
                # A response must never acquire the local gain during distillation.
                record["weight"] = 1.0 if kind == "distill" else float(row["weight"])
                self.records.append(record)

    @staticmethod
    def _validate(row, kind):
        if not isinstance(row, dict) or row.get("kind") != kind:
            raise ValueError(f"expected an explicitly tagged kind={kind!r} record")
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty chat-message list")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("invalid message role")
            content = message.get("content")
            if not isinstance(content, (str, list)):
                raise ValueError("message content must be text or a multimodal content list")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict) or part.get("type") not in {"text", "image", "video"}:
                        raise ValueError("content parts must be text, image, or video")
                    value = part.get(part["type"])
                    if value is None or (isinstance(value, (str, list)) and not value):
                        raise ValueError(f"missing {part['type']} content")
        if messages[-1]["role"] != "user":
            raise ValueError("the target context must end with a user message")
        if not isinstance(row.get("target"), str) or not row["target"].strip():
            raise ValueError("target must be nonempty text")
        weight = row.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise ValueError("weight must be a finite number")
        if kind == "local" and not 0 < weight <= 1:
            raise ValueError("accepted local gains must satisfy 0 < weight <= 1")
        if kind == "distill" and weight != 1:
            raise ValueError("distillation requires weight=1; do not pass a local-gain stream")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def distributed_stream_indices(size, microstep, batch_size, rank, world_size, seed):
    """Slice a common, endlessly shuffled stream without biased tail padding.

    A new microstep consumes a new global batch even during gradient accumulation.
    Every complete pass over the stream visits each accepted example exactly once.
    """
    if size <= 0 or batch_size <= 0 or microstep < 0 or not 0 <= rank < world_size:
        raise ValueError("invalid auxiliary sampling dimensions")
    offset = (microstep * world_size + rank) * batch_size
    permutations = {}
    result = []
    for position in range(offset, offset + batch_size):
        cycle, index = divmod(position, size)
        if cycle not in permutations:
            permutation = list(range(size))
            random.Random(seed + cycle).shuffle(permutation)
            permutations[cycle] = permutation
        result.append(permutations[cycle][index])
    return result


def target_token_mask(token_ids, offsets, start, end, special_ids=()):
    """Map an assistant-content character span to whole, non-special tokens."""
    mask = []
    for token, (left, right) in zip(token_ids, offsets):
        overlaps = right > start and left < end
        if overlaps and not (start <= left < right <= end):
            raise ValueError("chat-template boundary crosses a token; cannot isolate evidence-only labels")
        selected = overlaps and right > left
        if selected and token in special_ids:
            raise ValueError("target contains a special token instead of ordinary evidence text")
        mask.append(selected)
    if not any(mask):
        raise ValueError("target contains no supervised tokens")
    return mask


def _load_vision(messages):
    # The pinned baseline process_vision_info is video-only and returns a list.
    # Use its underlying loaders directly for a standard image/video chat schema.
    from open_r1.er.backend import vendored_vision

    vision = vendored_vision()

    images, videos = [], []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for part in message["content"]:
            if part["type"] == "image":
                images.append(vision.fetch_image(part))
            elif part["type"] == "video":
                videos.append(vision.fetch_video(part))
    return images, videos


class EvidenceCollator:
    """Mask assistant content after verifying tokenizer and visual expansion.

    Offsets come from the complete rendered chat, never separately tokenized
    prompt/target concatenation. The multimodal processor expands only prompt
    vision placeholders; its unchanged target-and-wrapper suffix is checked.
    No truncation is permitted, because it could alter the fixed visual support.
    """

    def __init__(self, processor, max_length=None, vision_loader=None):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.max_length = max_length
        self.vision_loader = vision_loader or _load_vision
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("evidence masking requires a fast tokenizer with offsets")

    def __call__(self, records):
        if not records:
            raise ValueError("cannot collate an empty auxiliary batch")
        texts, tokenizations, masks, all_images, all_videos = [], [], [], [], []
        for row in records:
            messages, target = row["messages"], row["target"]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = self.processor.apply_chat_template(
                messages + [{"role": "assistant", "content": target}],
                tokenize=False, add_generation_prompt=False,
            )
            if not text.startswith(prompt + target):
                raise ValueError("chat template changes the generation prefix; cannot isolate the target")
            encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            token_ids = encoding["input_ids"]
            masks.append(target_token_mask(
                token_ids, encoding["offset_mapping"], len(prompt), len(prompt) + len(target),
                set(self.tokenizer.all_special_ids),
            ))
            tokenizations.append(token_ids)
            texts.append(text)
            images, videos = self.vision_loader(messages)
            all_images.extend(images)
            all_videos.extend(videos)
        processor_kwargs = dict(
            text=texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False,
        )
        if all_images:
            processor_kwargs["images"] = all_images
        if all_videos:
            processor_kwargs["videos"] = all_videos
        batch = self.processor(**processor_kwargs)
        if self.max_length is not None and batch["input_ids"].shape[1] > self.max_length:
            raise ValueError("auxiliary example exceeds max_length; refusing to truncate its context or support")
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, (original_ids, mask) in enumerate(zip(tokenizations, masks)):
            active = batch["attention_mask"][row_index].bool()
            processed_ids = batch["input_ids"][row_index][active]
            first_target = mask.index(True)
            suffix = original_ids[first_target:]
            if len(processed_ids) < len(suffix) or processed_ids[-len(suffix):].tolist() != suffix:
                raise ValueError("multimodal processor changed target tokens; refusing misaligned labels")
            # Left padding and visual-token expansion both occur before this suffix.
            position = batch["input_ids"].shape[1] - len(suffix)
            for offset, selected in enumerate(mask[first_target:]):
                if selected:
                    labels[row_index, position + offset] = batch["input_ids"][row_index, position + offset]
        batch["labels"] = labels
        batch["weights"] = torch.tensor([row["weight"] for row in records], dtype=torch.float32)
        return batch


def weighted_sequence_nll(logits, labels, weights):
    """Mean_i weight_i * mean_{target tokens i} NLL, never divide by sum(weights)."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or weights.shape != (logits.shape[0],):
        raise ValueError("incompatible logits, labels, or sequence weights")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("sequence weights must be finite and nonnegative")
    shifted_labels = labels[:, 1:]
    target_mask = shifted_labels != -100
    token_counts = target_mask.sum(dim=1)
    if bool((token_counts == 0).any()):
        raise ValueError("every accepted example must have at least one predictable target token")
    # Select before converting to fp32: video/prompt vocabulary logits are not
    # duplicated, and rows can have different evidence lengths without bias.
    row_losses = []
    for row in range(logits.shape[0]):
        selected_logits = logits[row, :-1][target_mask[row]].float()
        selected_labels = shifted_labels[row][target_mask[row]]
        row_losses.append(F.cross_entropy(selected_logits, selected_labels, reduction="mean"))
    return (torch.stack(row_losses) * weights.detach().to(logits.device, dtype=torch.float32)).mean()
