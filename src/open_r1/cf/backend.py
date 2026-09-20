"""Local frozen Qwen-VL backend; imports heavyweight dependencies only on use."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


def vendored_vision():
    """Use the very same reader/resizer as the ordinary TW-GRPO stream."""
    vendor = Path(__file__).resolve().parents[3] / "qwen-vl-utils" / "src"
    if not vendor.is_dir():
        raise FileNotFoundError(f"Missing baseline qwen-vl-utils: {vendor}")
    sys.path.insert(0, str(vendor))
    import qwen_vl_utils.vision_process as vision

    if vendor not in Path(vision.__file__).resolve().parents:
        raise RuntimeError("A different qwen_vl_utils was imported before the baseline vendor")
    return vision


class FrozenQwenBackend:
    def __init__(self, model_path, *, node_max_tokens=512, schema_max_tokens=2048,
                 plan_max_tokens=1024, total_generation_tokens=100000,
                 max_prompt_tokens=16384, seed=42, device="cuda:0", dtype="bfloat16",
                 temperature=0.8, max_pixels=12845056, min_pixels=3136,
                 generation_audit_path=None):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import AutoConfig, AutoProcessor, set_seed
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

        checkpoint = Path(model_path).resolve(strict=True)
        if not checkpoint.is_dir():
            raise ValueError("--model must name a local checkpoint directory")
        self.torch = torch
        self.vision = vendored_vision()
        self.node_max_tokens = node_max_tokens
        self.stage_max_tokens = {"schema": schema_max_tokens, "plan": plan_max_tokens}
        self.total_generation_tokens = total_generation_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.temperature = temperature
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.used_tokens = 0
        self.input_tokens = 0
        self.calls = Counter()
        self.audit_context = None
        self.generation_audit_path = Path(generation_audit_path).resolve() if generation_audit_path else None
        if self.generation_audit_path is not None:
            self.generation_audit_path.parent.mkdir(parents=True, exist_ok=True)
            # Each run owns a new trace, so diagnostics cannot silently mix snapshots.
            with self.generation_audit_path.open("x", encoding="utf-8"):
                pass
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        config = AutoConfig.from_pretrained(str(checkpoint), local_files_only=True)
        classes = {"qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
                   "qwen2_vl": Qwen2VLForConditionalGeneration}
        if config.model_type not in classes:
            raise ValueError(f"Unsupported local model type: {config.model_type}")
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint), local_files_only=True, max_pixels=max_pixels, min_pixels=min_pixels)
        self.model = classes[config.model_type].from_pretrained(
            str(checkpoint), local_files_only=True, torch_dtype=getattr(torch, dtype),
            device_map={"": device}, attn_implementation="sdpa")
        self.model.eval()
        self.model.requires_grad_(False)
        files = [{"path": str(p.relative_to(checkpoint)), "size": p.stat().st_size,
                  "mtime_ns": p.stat().st_mtime_ns}
                 for p in sorted(checkpoint.iterdir()) if p.is_file()]
        config_hash = hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()
        self.identity = {"path": str(checkpoint), "model_type": config.model_type,
                         "config_sha256": config_hash, "checkpoint_file_inventory": files,
                         "weights_content_hashed": False, "seed": seed, "dtype": dtype,
                         "device": device, "attention": "sdpa", "frozen": True,
                         "local_files_only": True, "deterministic_algorithms": True,
                         "torch_version": torch.__version__,
                         "vision_source": str(Path(self.vision.__file__).resolve()),
                         "vision_sha256": hashlib.sha256(Path(self.vision.__file__).read_bytes()).hexdigest()}
        self.fingerprint = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def generate(self, messages, *, stage, deterministic):
        remaining = self.total_generation_tokens - self.used_tokens
        call_cap = self.stage_max_tokens.get(stage, self.node_max_tokens)
        if remaining < call_cap:
            raise BudgetExhausted("Remaining budget cannot fit the unchanged stage decoding limit")
        images, videos = [], []
        for message in messages:
            if isinstance(message["content"], str):
                continue
            for item in message["content"]:
                if item["type"] == "image":
                    images.append(self.vision.fetch_image(item))
                elif item["type"] == "video":
                    videos.append(self.vision.fetch_video(item))
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, videos=videos or None,
                                return_tensors="pt", padding=True)
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens > self.max_prompt_tokens:
            raise ValueError(f"Prompt token cap exceeded: {prompt_tokens} > {self.max_prompt_tokens}")
        inputs = inputs.to(self.model.device)
        kwargs = {"max_new_tokens": call_cap,
                  "do_sample": not deterministic, "use_cache": True,
                  "pad_token_id": self.processor.tokenizer.pad_token_id,
                  "eos_token_id": self.processor.tokenizer.eos_token_id}
        if not deterministic:
            kwargs.update(temperature=self.temperature, top_p=1.0, top_k=0)
        self.calls[stage] += 1
        self.input_tokens += prompt_tokens
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        completion = generated[0, prompt_tokens:]
        self.used_tokens += completion.numel()
        output = self.processor.tokenizer.decode(completion, skip_special_tokens=True).strip()
        self._audit_generation(stage=stage, deterministic=deterministic, token_cap=call_cap,
                               input_tokens=prompt_tokens, output_tokens=completion.numel(), output=output)
        return output

    def _audit_generation(self, *, stage, deterministic, token_cap, input_tokens, output_tokens, output):
        """Record decoded model outputs before any downstream parser can reject them."""
        if self.generation_audit_path is None:
            return
        record = {"call_index": sum(self.calls.values()), "context": self.audit_context,
                  "executor_fingerprint": self.fingerprint, "stage": stage,
                  "deterministic": deterministic, "token_cap": token_cap,
                  "input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_generation_tokens": self.used_tokens, "output": output}
        # Intentionally no prompts, visual payloads, labels, environment or exception dumps.
        with self.generation_audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def statistics(self):
        return {"generation_tokens": self.used_tokens, "input_tokens": self.input_tokens,
                "model_calls_by_stage": dict(self.calls)}

    def export_support(self, video_path, output_dir):
        """Save exactly fetch_video's policy frames, with original sampled timestamps."""
        from PIL import Image

        source = Path(video_path).resolve(strict=True)
        if not source.is_file():
            raise ValueError("Dataset video must be a local file")
        # The base trainer supplies only the path, so its vendored defaults apply.
        element = {"video": str(source)}
        actual_reader = []
        original_readers = dict(self.vision.VIDEO_READER_BACKENDS)
        for name, reader in original_readers.items():
            def capture(ele, _name=name, _reader=reader):
                result = _reader(ele)
                actual_reader[:] = [_name]
                return result
            self.vision.VIDEO_READER_BACKENDS[name] = capture
        try:
            frames, sample_fps = self.vision.fetch_video(element, return_video_sample_fps=True)
        finally:
            self.vision.VIDEO_READER_BACKENDS.update(original_readers)
        if actual_reader == ["decord"]:
            import decord
            video = decord.VideoReader(str(source), num_threads=1)
            total_frames, fps = len(video), video.get_avg_fps()
            count = self.vision.smart_nframes(element, total_frames, fps)
            indices = self.torch.linspace(0, total_frames - 1, count).round().long()
            if self.vision.SAMPLE_MODE != "true":
                indices = indices[self.torch.linspace(0, count - 1, 16).long()]
            times = [float(video.get_frame_timestamp(int(index))[0]) for index in indices]
        else:
            from torchvision.io import read_video_timestamps
            timestamps, fps = read_video_timestamps(str(source), pts_unit="sec")
            count = self.vision.smart_nframes(element, len(timestamps), fps)
            indices = self.torch.linspace(0, len(timestamps) - 1, count).round().long()
            if self.vision.SAMPLE_MODE != "true":
                indices = indices[self.torch.linspace(0, count - 1, 16).long()]
            times = [float(timestamps[int(index)]) for index in indices]
        if len(times) != frames.shape[0]:
            raise ValueError("Frame timestamp reconstruction disagrees with baseline sampling")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        records, frame_hashes = [], {}
        for index, (frame, time) in enumerate(zip(frames, times)):
            path = (directory / f"f{index:03d}.png").resolve()
            array = frame.round().clamp(0, 255).to(self.torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(array).save(path)
            frame_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append({"id": f"f{index}", "path": str(path), "time": time, "regions": []})
        return records, {"source": str(source), "reader": actual_reader[0],
                         "sample_fps": sample_fps, "source_indices": indices.tolist(),
                         "sample_mode": self.vision.SAMPLE_MODE,
                         "fps_max_frames": self.vision.FPS_MAX_FRAMES,
                         "video_max_pixels": self.vision.VIDEO_MAX_PIXELS,
                         "frame_file_sha256": frame_hashes,
                         "frame_shape": list(frames.shape), "png_quantization": "round-clamp-uint8"}
