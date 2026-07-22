from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from PIL import Image


DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCALIZATION_CACHE_SCHEMA_VERSION = 1
DEFAULT_LOCALIZATION_DECODING = {
    "do_sample": False,
    "max_new_tokens": 128,
    "num_beams": 1,
}


@dataclass(frozen=True)
class QwenAvailability:
    model_id: str
    cache_dir: str
    cached_locally: bool
    qwen_vl_utils_available: bool
    transformers_available: bool
    free_gib: float
    min_free_gib: float
    local_files_only: bool

    @property
    def ready(self) -> bool:
        if not self.qwen_vl_utils_available or not self.transformers_available:
            return False
        if self.local_files_only and not self.cached_locally:
            return False
        if not self.cached_locally and self.free_gib < self.min_free_gib:
            return False
        return True

    def failure_reason(self) -> str | None:
        if self.ready:
            return None
        reasons: list[str] = []
        if not self.qwen_vl_utils_available:
            reasons.append("qwen-vl-utils is not installed")
        if not self.transformers_available:
            reasons.append("transformers is not installed")
        if self.local_files_only and not self.cached_locally:
            reasons.append(f"{self.model_id} is not cached locally and local_files_only=true")
        if not self.cached_locally and self.free_gib < self.min_free_gib:
            reasons.append(f"only {self.free_gib:.1f} GiB free below required {self.min_free_gib:.1f} GiB")
        return "; ".join(reasons)


class QwenFeatureExtractor:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_QWEN_MODEL,
        cache_dir: str | None = None,
        local_files_only: bool = True,
        device: str = "auto",
        torch_dtype: str = "auto",
        token_count: int = 16,
        min_free_gib: float = 30.0,
    ) -> None:
        availability = qwen_availability(
            model_id=model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            min_free_gib=min_free_gib,
        )
        if not availability.ready:
            raise RuntimeError(f"Qwen provider is not ready: {availability.failure_reason()}")
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.device = _device(device)
        self.token_count = token_count
        self.torch_dtype = _dtype(torch_dtype)
        self.generation_settings = dict(DEFAULT_LOCALIZATION_DECODING)
        self._processor: Any | None = None
        self._model: Any | None = None

    def close(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def analyze(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        processor, model = self._load()
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **self.generation_settings)
            generated_text = processor.batch_decode(
                generated_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1].detach().float().cpu()
        tokens, selected = _select_tokens(hidden, inputs["input_ids"].detach().cpu(), model.config, self.token_count)
        return {
            "text": generated_text,
            "tokens": tokens,
            "selected_token_indices": selected,
            "hidden_width": int(tokens.shape[-1]),
            "input_token_count": int(inputs["input_ids"].shape[1]),
        }

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {"cache_dir": self.cache_dir, "local_files_only": self.local_files_only}
        self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self._model.eval()
        return self._processor, self._model


class CachedQwenLocalizationExtractor:
    """Content-addressed Qwen text cache for deterministic localization replay."""

    def __init__(
        self,
        extractor: QwenFeatureExtractor,
        *,
        cache_dir: Path,
        model_id: str,
        model_revision: str,
        torch_dtype: str,
        decoding_settings: dict[str, Any] | None = None,
        enabled: bool = True,
        refresh: bool = False,
    ) -> None:
        self.extractor = extractor
        self.cache_dir = cache_dir
        self.enabled = enabled
        self.refresh = refresh
        self.model_id = model_id
        self.model_revision = model_revision
        self.torch_dtype = torch_dtype
        self.decoding_settings = dict(decoding_settings or DEFAULT_LOCALIZATION_DECODING)

    def analyze(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        identity = self._identity(image, prompt)
        cache_key = _json_sha256(identity)
        cache_path = self.cache_dir / cache_key[:2] / f"{cache_key}.json"
        invalid_reason: str | None = None
        if self.enabled and cache_path.exists() and not self.refresh:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                text = _validate_localization_cache_payload(payload, cache_key, identity)
                return {
                    "text": text,
                    "localization_cache": self._metadata(cache_key, cache_path, text=text, cache_hit=True),
                }
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                invalid_reason = str(exc)

        result = self.extractor.analyze(image, prompt)
        text = str(result["text"])
        if self.enabled:
            payload = {
                "schema_version": LOCALIZATION_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "identity": identity,
                "response": {"text": text},
                "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(f".{os.getpid()}.{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(cache_path)
        output = dict(result)
        output["localization_cache"] = self._metadata(
            cache_key,
            cache_path,
            text=text,
            cache_hit=False,
            invalid_reason=invalid_reason,
        )
        return output

    def close(self) -> None:
        self.extractor.close()

    def _identity(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        rgb = image.convert("RGB")
        image_digest = hashlib.sha256()
        image_digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
        image_digest.update(rgb.tobytes())
        try:
            transformers_version = importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            transformers_version = "unavailable"
        return {
            "image_sha256": image_digest.hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "torch_dtype": self.torch_dtype,
            "transformers_version": transformers_version,
            "decoding_settings": self.decoding_settings,
        }

    def _metadata(
        self,
        cache_key: str,
        cache_path: Path,
        *,
        text: str,
        cache_hit: bool,
        invalid_reason: str | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "enabled": self.enabled,
            "cache_hit": cache_hit,
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "schema_version": LOCALIZATION_CACHE_SCHEMA_VERSION,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "torch_dtype": self.torch_dtype,
            "decoding_settings": self.decoding_settings,
        }
        if invalid_reason is not None:
            metadata["invalid_reason"] = invalid_reason
        return metadata


def resolve_qwen_model_revision(model_id: str, cache_dir: str | Path | None) -> str:
    root = Path(cache_dir or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    hub = root / "hub" if root.name != "hub" else root
    model_dir = hub / f"models--{model_id.replace('/', '--')}"
    main_ref = model_dir / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if revision:
            return revision
    snapshots = sorted(path.name for path in (model_dir / "snapshots").glob("*") if path.is_dir())
    return snapshots[-1] if snapshots else "unresolved"


def _validate_localization_cache_payload(payload: object, cache_key: str, identity: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Qwen localization cache payload must be an object")
    if int(payload.get("schema_version", -1)) != LOCALIZATION_CACHE_SCHEMA_VERSION:
        raise ValueError("Qwen localization cache schema mismatch")
    if str(payload.get("cache_key", "")) != cache_key or payload.get("identity") != identity:
        raise ValueError("Qwen localization cache identity mismatch")
    response = payload.get("response")
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("Qwen localization cache response is missing text")
    text = str(response["text"])
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if str(payload.get("response_sha256", "")) != expected:
        raise ValueError("Qwen localization cache response hash mismatch")
    return text


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qwen_availability(
    *,
    model_id: str = DEFAULT_QWEN_MODEL,
    cache_dir: str | None = None,
    local_files_only: bool = True,
    min_free_gib: float = 30.0,
) -> QwenAvailability:
    root = Path(cache_dir or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    hub = root / "hub" if root.name != "hub" else root
    model_dir = hub / f"models--{model_id.replace('/', '--')}"
    free_gib = _free_gib(root if root.exists() else root.parent)
    return QwenAvailability(
        model_id=model_id,
        cache_dir=str(root),
        cached_locally=model_dir.exists(),
        qwen_vl_utils_available=importlib.util.find_spec("qwen_vl_utils") is not None,
        transformers_available=importlib.util.find_spec("transformers") is not None,
        free_gib=free_gib,
        min_free_gib=min_free_gib,
        local_files_only=local_files_only,
    )


def parse_normalized_box(text: str) -> tuple[float, float, float, float] | None:
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            box = payload.get("bbox_xyxy") or payload.get("box") or payload.get("bbox") or payload.get("region")
            if isinstance(box, list | tuple) and len(box) == 4:
                return _normalize_box(tuple(float(value) for value in box))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    match = re.search(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]", text)
    if not match:
        match = re.search(
            r"x1\s*[:=]\s*([0-9.]+).*?y1\s*[:=]\s*([0-9.]+).*?x2\s*[:=]\s*([0-9.]+).*?y2\s*[:=]\s*([0-9.]+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
    if not match:
        return None
    return _normalize_box(tuple(float(match.group(index)) for index in range(1, 5)))


def _normalize_box(values: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    if max(values) > 1.0:
        values = tuple(value / 1000.0 if value > 1.0 else value for value in values)
    x1, y1, x2, y2 = (max(0.0, min(1.0, value)) for value in values)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def save_qwen_cache(
    path: Path,
    *,
    tokens: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens.cpu(), "metadata": metadata}, path)


def write_availability(path: Path, availability: QwenAvailability) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(availability.__dict__, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _select_tokens(hidden: torch.Tensor, input_ids: torch.Tensor, config: Any, token_count: int) -> tuple[torch.Tensor, list[int]]:
    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is not None:
        positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=False).flatten()
    else:
        positions = torch.arange(input_ids.shape[1])
    if len(positions) == 0:
        positions = torch.arange(input_ids.shape[1])
    if len(positions) >= token_count:
        chosen = positions[torch.linspace(0, len(positions) - 1, token_count).round().long()]
    else:
        repeats = positions.repeat((token_count + len(positions) - 1) // len(positions))
        chosen = repeats[:token_count]
    return hidden[:, chosen, :].contiguous(), [int(value) for value in chosen.tolist()]


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Qwen provider requested CUDA but torch.cuda.is_available() is false")
    return value


def _dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    return getattr(torch, value)


def _free_gib(path: Path) -> float:
    target = path
    while not target.exists() and target.parent != target:
        target = target.parent
    usage = shutil.disk_usage(target)
    return usage.free / (1024**3)
