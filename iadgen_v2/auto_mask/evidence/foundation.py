from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.contracts import AutoMaskContext, EvidenceMap, RegistrationResult
from iadgen_v2.auto_mask.evidence.base import apply_soft_spatial_prior, robust_probability


_MODEL_CACHE: dict[tuple[str, str | None, str], tuple[Any, Any, Any]] = {}


def _load_model(model_id: str, cache_dir: str | None, device: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    key = (model_id, cache_dir, resolved)
    if key not in _MODEL_CACHE:
        processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True).to(resolved).eval()
        _MODEL_CACHE[key] = (processor, model, torch.device(resolved))
    return _MODEL_CACHE[key]


def _tokens(processor: Any, model: Any, device: Any, image: Image.Image, scale: int, layers: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, int]]:
    import torch

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt",
        do_resize=True,
        size={"height": int(scale), "width": int(scale)},
        do_center_crop=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
    selected = []
    for layer in layers:
        values = output.hidden_states[layer][:, 1:, :].squeeze(0).float()
        selected.append(torch.nn.functional.normalize(values, dim=-1))
    values = torch.cat(selected, dim=-1)
    values = torch.nn.functional.normalize(values, dim=-1).cpu().numpy().astype(np.float32)
    count = values.shape[0]
    side = int(round(count**0.5))
    return values, (side, side) if side * side == count else (count, 1)


def _nearest_distance(target: np.ndarray, memory: np.ndarray, chunk: int = 4096) -> np.ndarray:
    best = np.full(target.shape[0], np.inf, dtype=np.float32)
    for start in range(0, memory.shape[0], chunk):
        distance = 1.0 - target @ memory[start : start + chunk].T
        best = np.minimum(best, distance.min(axis=1))
    return best


def _resize_map(values: np.ndarray, grid: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    small = values.reshape(grid)
    return cv2.resize(small, size, interpolation=cv2.INTER_CUBIC).astype(np.float32)


@dataclass
class MultiScaleDinoProvider:
    name: str = "dinov2_multiscale"
    model_id: str = "facebook/dinov2-small"
    cache_dir: str | None = None
    device: str = "auto"
    scales: tuple[int, ...] = (448, 672)
    layers: tuple[int, ...] = (-4, -1)
    max_normals: int = 16
    memory_stride: int = 1
    artifact_dir: Path | None = None

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        cache_path = self._cache_path(context)
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            return EvidenceMap(
                source=self.name,
                values=cached["values"].astype(np.float32),
                reliability=float(cached["reliability"]),
                calibration={"median": float(cached["median"]), "mad": float(cached["mad"])},
                augmentation_consistency=float(cached["consistency"]),
                artifact_path=cache_path,
                metadata={"cache_hit": True, "scales": list(self.scales), "layers": list(self.layers)},
            )
        processor, model, device = _load_model(self.model_id, self.cache_dir, self.device)
        image = Image.open(context.image_path).convert("RGB")
        scale_maps: list[np.ndarray] = []
        for scale in self.scales:
            target, grid = _tokens(processor, model, device, image, scale, self.layers)
            memory_parts = []
            for path in context.normal_paths[: self.max_normals]:
                normal = Image.open(path).convert("RGB")
                values, _ = _tokens(processor, model, device, normal, scale, self.layers)
                memory_parts.append(values[:: max(1, self.memory_stride)])
            if not memory_parts:
                raise ValueError("DINOv2 evidence requires normal references")
            distance = _nearest_distance(target, np.concatenate(memory_parts, axis=0))
            scale_maps.append(_resize_map(distance, grid, context.image_size))
        raw = np.mean(np.stack(scale_maps), axis=0)
        probability, calibration = robust_probability(raw)
        probability = apply_soft_spatial_prior(probability, context.semantic_regions)
        normalized = [(item - item.mean()) / max(1e-6, item.std()) for item in scale_maps]
        consistency = float(np.clip(np.mean(np.corrcoef(item.reshape(-1), normalized[0].reshape(-1))[0, 1] for item in normalized), 0.0, 1.0))
        reliability = float(np.clip(0.45 + 0.35 * consistency + 0.20 * min(1.0, len(context.normal_paths) / 8.0), 0.05, 1.0))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, values=probability, reliability=reliability, median=calibration["median"], mad=calibration["mad"], consistency=consistency)
        return EvidenceMap(
            source=self.name,
            values=probability,
            reliability=reliability,
            calibration=calibration,
            augmentation_consistency=consistency,
            artifact_path=cache_path,
            metadata={"cache_hit": False, "scales": list(self.scales), "layers": list(self.layers), "normals_used": min(len(context.normal_paths), self.max_normals)},
        )

    def _cache_path(self, context: AutoMaskContext) -> Path | None:
        if self.artifact_dir is None:
            return None
        identity = f"{context.cache_key}|{self.model_id}|{self.scales}|{self.layers}|{self.max_normals}|{self.memory_stride}"
        return self.artifact_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.npz"


@dataclass
class RegisteredDinoResidualProvider:
    name: str = "registered_normal_residual"
    model_id: str = "facebook/dinov2-small"
    cache_dir: str | None = None
    device: str = "auto"
    scale: int = 448
    layer: int = -1
    max_normals: int = 16
    min_inlier_ratio: float = 0.25
    max_reprojection_error: float = 8.0
    last_registration: RegistrationResult | None = field(default=None, init=False)

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        processor, model, device = _load_model(self.model_id, self.cache_dir, self.device)
        image = Image.open(context.image_path).convert("RGB")
        target_tokens, grid = _tokens(processor, model, device, image, self.scale, (self.layer,))
        target_xy = _grid_coordinates(grid, context.image_size)
        best: tuple[float, RegistrationResult, np.ndarray] | None = None
        target_array = np.asarray(image, dtype=np.float32)
        for path in context.normal_paths[: self.max_normals]:
            normal_image = Image.open(path).convert("RGB").resize(context.image_size, Image.Resampling.BILINEAR)
            normal_tokens, normal_grid = _tokens(processor, model, device, normal_image, self.scale, (self.layer,))
            if normal_grid != grid:
                continue
            similarity = target_tokens @ normal_tokens.T
            target_to_normal = similarity.argmax(axis=1)
            normal_to_target = similarity.argmax(axis=0)
            target_ids = np.arange(target_tokens.shape[0])
            mutual = normal_to_target[target_to_normal] == target_ids
            src = _grid_coordinates(normal_grid, context.image_size)[target_to_normal[mutual]].astype(np.float32)
            dst = target_xy[mutual].astype(np.float32)
            if len(src) < 6:
                continue
            matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
            if matrix is None or inliers is None:
                continue
            projected = cv2.transform(src[None], matrix)[0]
            inlier_mask = inliers.reshape(-1).astype(bool)
            error = float(np.linalg.norm(projected[inlier_mask] - dst[inlier_mask], axis=1).mean()) if inlier_mask.any() else float("inf")
            ratio = float(inlier_mask.mean())
            applied = ratio >= self.min_inlier_ratio and error <= self.max_reprojection_error
            confidence = float(np.clip(ratio * np.exp(-error / max(1e-6, self.max_reprojection_error)), 0.0, 1.0))
            registration = RegistrationResult(matrix=matrix, inlier_ratio=ratio, reprojection_error=error, confidence=confidence, applied=applied, reference_path=path)
            normal_array = np.asarray(normal_image, dtype=np.float32)
            if applied:
                normal_array = cv2.warpAffine(normal_array, matrix, context.image_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            residual = _appearance_residual(target_array, normal_array)
            score = confidence - float(np.mean(residual)) * 0.05
            if best is None or score > best[0]:
                best = (score, registration, residual)
        if best is None:
            raise ValueError("Registered residual could not match a normal reference")
        _, registration, raw = best
        self.last_registration = registration
        probability, calibration = robust_probability(raw)
        probability = apply_soft_spatial_prior(probability, context.semantic_regions)
        reliability = float(np.clip(0.25 + 0.75 * registration.confidence, 0.05, 1.0))
        return EvidenceMap(
            source=self.name,
            values=probability,
            reliability=reliability,
            calibration=calibration,
            metadata={
                "registration_applied": registration.applied,
                "registration_inlier_ratio": registration.inlier_ratio,
                "registration_reprojection_error": registration.reprojection_error,
                "registration_reference_path": str(registration.reference_path),
            },
        )


def _grid_coordinates(grid: tuple[int, int], image_size: tuple[int, int]) -> np.ndarray:
    height, width = grid
    image_width, image_height = image_size
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack(((xs.reshape(-1) + 0.5) * image_width / width, (ys.reshape(-1) + 0.5) * image_height / height), axis=1)


def _appearance_residual(target: np.ndarray, normal: np.ndarray) -> np.ndarray:
    target_gray = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    normal_gray = cv2.cvtColor(normal.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    intensity = np.abs(target_gray - normal_gray) / 255.0
    target_gradient = cv2.magnitude(cv2.Sobel(target_gray, cv2.CV_32F, 1, 0), cv2.Sobel(target_gray, cv2.CV_32F, 0, 1))
    normal_gradient = cv2.magnitude(cv2.Sobel(normal_gray, cv2.CV_32F, 1, 0), cv2.Sobel(normal_gray, cv2.CV_32F, 0, 1))
    gradient = np.abs(target_gradient - normal_gradient)
    gradient /= max(1e-6, float(np.percentile(gradient, 99.0)))
    return np.clip(0.65 * intensity + 0.35 * gradient, 0.0, 1.0).astype(np.float32)
