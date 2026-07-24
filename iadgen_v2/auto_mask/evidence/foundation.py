from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.contracts import AutoMaskContext, EvidenceMap, RegistrationResult
from iadgen_v2.auto_mask.evidence.base import apply_soft_spatial_prior, robust_probability


_MODEL_CACHE: dict[tuple[str, str | None, str], tuple[Any, Any, Any]] = {}
# Normal-reference DINOv2 tokens are identical across every defect image in a
# category, but were previously re-encoded per image. Cache them in-process
# (and optionally on disk) so each normal is encoded once per (content, scale).
_NORMAL_TOKEN_CACHE: dict[tuple[str, str, tuple[int, ...], int], np.ndarray] = {}


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


def _nearest_distance(target: np.ndarray, memory: np.ndarray, chunk: int = 8192, use_gpu: bool | None = None) -> np.ndarray:
    """Nearest cosine distance of each target token to the memory bank.

    Tokens are L2-normalized upstream, so cosine distance is ``1 - target @ memory.T``.
    Runs on the GPU only when ``use_gpu`` is true (i.e. the provider resolved to
    a CUDA device); an explicit CPU provider stays on NumPy even when a GPU is
    present. ``use_gpu=None`` falls back to auto-detection for standalone callers.
    Note: this is cosine distance, not the Euclidean distance ``torch.cdist`` gives.
    """

    if use_gpu is None:
        use_gpu = _gpu_knn_available()
    if use_gpu and _gpu_knn_available():
        return _nearest_distance_gpu(target, memory, chunk)
    return _nearest_distance_numpy(target, memory, chunk)


def _nearest_distance_numpy(target: np.ndarray, memory: np.ndarray, chunk: int = 8192) -> np.ndarray:
    best = np.full(target.shape[0], np.inf, dtype=np.float32)
    for start in range(0, memory.shape[0], chunk):
        distance = 1.0 - target @ memory[start : start + chunk].T
        best = np.minimum(best, distance.min(axis=1))
    return best.astype(np.float32)


def _gpu_knn_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _nearest_distance_gpu(target: np.ndarray, memory: np.ndarray, chunk: int = 8192) -> np.ndarray:
    import torch

    device = torch.device("cuda")
    with torch.no_grad():
        target_t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32)).to(device)
        best = torch.full((target_t.shape[0],), float("inf"), device=device)
        for start in range(0, memory.shape[0], chunk):
            block = torch.from_numpy(np.ascontiguousarray(memory[start : start + chunk], dtype=np.float32)).to(device)
            distance = 1.0 - target_t @ block.T
            best = torch.minimum(best, distance.min(dim=1).values)
            del block
        return best.detach().cpu().numpy().astype(np.float32)


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
                metadata={
                    "cache_hit": True,
                    "scales": list(self.scales),
                    "layers": list(self.layers),
                    "calibration_source": "leave_one_normal_out",
                },
            )
        processor, model, device = _load_model(self.model_id, self.cache_dir, self.device)
        use_gpu = getattr(device, "type", str(device)) == "cuda"
        image = Image.open(context.image_path).convert("RGB")
        self._token_cache_hits = 0
        self._token_cache_misses = 0
        scale_maps: list[np.ndarray] = []
        scale_calibrations: list[dict[str, float]] = []
        normal_medians: list[float] = []
        for scale in self.scales:
            target, grid = _tokens(processor, model, device, image, scale, self.layers)
            memory_parts: list[np.ndarray] = []
            for path in context.normal_paths[: self.max_normals]:
                values = self._normal_tokens(processor, model, device, Path(path), scale)
                memory_parts.append(values[:: max(1, self.memory_stride)])
            if not memory_parts:
                raise ValueError("DINOv2 evidence requires normal references")
            distance = _nearest_distance(target, np.concatenate(memory_parts, axis=0), use_gpu=use_gpu)
            calibration_values: list[np.ndarray] = []
            if len(memory_parts) >= 2:
                for index, held_out in enumerate(memory_parts):
                    other_memory = np.concatenate([part for part_index, part in enumerate(memory_parts) if part_index != index], axis=0)
                    held_distance = _nearest_distance(held_out, other_memory, use_gpu=use_gpu)
                    calibration_values.append(held_distance)
                    normal_medians.append(float(np.median(held_distance)))
            if calibration_values:
                values = np.concatenate(calibration_values)
                median = float(np.median(values))
                calibration = {"median": median, "mad": max(1e-6, float(np.median(np.abs(values - median))))}
            else:
                calibration = None
            scale_probability, used_calibration = robust_probability(
                _resize_map(distance, grid, context.image_size),
                calibration,
            )
            scale_maps.append(scale_probability)
            scale_calibrations.append(used_calibration)
        probability = np.mean(np.stack(scale_maps), axis=0)
        probability = apply_soft_spatial_prior(probability, context.semantic_regions)
        normalized = [(item - item.mean()) / max(1e-6, item.std()) for item in scale_maps]
        correlations = [
            float(np.nan_to_num(np.corrcoef(item.reshape(-1), normalized[0].reshape(-1))[0, 1], nan=0.0))
            for item in normalized
        ]
        consistency = float(np.clip(np.mean(correlations), 0.0, 1.0))
        normal_stability = float(1.0 / (1.0 + 10.0 * np.std(normal_medians))) if normal_medians else 0.5
        reliability = float(
            np.clip(
                0.30 + 0.30 * consistency + 0.20 * normal_stability + 0.20 * min(1.0, len(context.normal_paths) / 8.0),
                0.05,
                1.0,
            )
        )
        calibration = {
            "median": float(np.mean([item["median"] for item in scale_calibrations])),
            "mad": float(np.mean([item["mad"] for item in scale_calibrations])),
        }
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
            metadata={
                "cache_hit": False,
                "scales": list(self.scales),
                "layers": list(self.layers),
                "normals_used": min(len(context.normal_paths), self.max_normals),
                "normal_stability": normal_stability,
                "calibration_source": "leave_one_normal_out" if len(memory_parts) >= 2 else "target_fallback",
                "normal_token_cache_hits": int(getattr(self, "_token_cache_hits", 0)),
                "normal_token_cache_misses": int(getattr(self, "_token_cache_misses", 0)),
                "knn_backend": "gpu" if use_gpu else "numpy",
            },
        )

    def _normal_tokens(self, processor: Any, model: Any, device: Any, path: Path, scale: int) -> np.ndarray:
        # Content signature (size + mtime) so a same-path content change is not
        # served stale from the in-process cache.
        stat = path.stat()
        mem_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, self.model_id, tuple(self.layers), int(scale))
        cached = _NORMAL_TOKEN_CACHE.get(mem_key)
        if cached is not None:
            self._token_cache_hits += 1
            return cached
        disk_path = self._normal_token_disk_path(path, scale, processor, model)
        if disk_path is not None and disk_path.exists():
            try:
                values = np.load(disk_path).astype(np.float32)
                _NORMAL_TOKEN_CACHE[mem_key] = values
                self._token_cache_hits += 1
                return values
            except Exception:
                # Corrupted / partial cache entry: drop it and recompute.
                disk_path.unlink(missing_ok=True)
        self._token_cache_misses += 1
        normal = Image.open(path).convert("RGB")
        values, _ = _tokens(processor, model, device, normal, scale, self.layers)
        _NORMAL_TOKEN_CACHE[mem_key] = values
        if disk_path is not None:
            self._atomic_save(disk_path, values)
        return values

    @staticmethod
    def _atomic_save(path: Path, values: np.ndarray) -> None:
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, values)  # file object -> no .npy suffix rewriting
            os.replace(tmp, path)  # atomic within the same directory
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _normal_token_disk_path(self, path: Path, scale: int, processor: Any = None, model: Any = None) -> Path | None:
        if self.artifact_dir is None:
            return None
        try:
            import transformers

            transformers_version = transformers.__version__
        except Exception:
            transformers_version = "unknown"
        config = getattr(model, "config", None)
        model_identity = {
            "model_type": getattr(config, "model_type", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "name_or_path": getattr(config, "_name_or_path", None),
        }
        identity = json.dumps(
            {
                "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_id": self.model_id,
                "transformers_version": transformers_version,
                "processor_class": type(processor).__name__ if processor is not None else None,
                "model_identity": model_identity,
                "scale": int(scale),
                "layers": list(self.layers),
                "dtype": "float32",
            },
            sort_keys=True,
        )
        return self.artifact_dir / "dino_normal_tokens" / f"{hashlib.sha256(identity.encode()).hexdigest()}.npy"

    def _cache_path(self, context: AutoMaskContext) -> Path | None:
        if self.artifact_dir is None:
            return None
        identity = f"v2-loo|{context.cache_key}|{self.model_id}|{self.scales}|{self.layers}|{self.max_normals}|{self.memory_stride}"
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
    artifact_dir: Path | None = None
    last_registration: RegistrationResult | None = field(default=None, init=False)

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        cache_path = self._cache_path(context)
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            return EvidenceMap(
                source=self.name,
                values=cached["values"].astype(np.float32),
                reliability=float(cached["reliability"]),
                calibration={"median": float(cached["median"]), "mad": float(cached["mad"])},
                artifact_path=cache_path,
                metadata={
                    "cache_hit": True,
                    "registration_applied": bool(cached["registration_applied"]),
                    "registration_inlier_ratio": float(cached["registration_inlier_ratio"]),
                    "registration_reprojection_error": float(cached["registration_reprojection_error"]),
                    "calibration_source": str(cached["calibration_source"]),
                },
            )
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
        normal_samples: list[np.ndarray] = []
        calibration_paths = list(context.normal_paths[: min(self.max_normals, 5)])
        for index in range(len(calibration_paths) - 1):
            first = np.asarray(
                Image.open(calibration_paths[index]).convert("RGB").resize(context.image_size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            second = np.asarray(
                Image.open(calibration_paths[index + 1]).convert("RGB").resize(context.image_size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            normal_samples.append(_appearance_residual(first, second)[::8, ::8].reshape(-1))
        normal_calibration = None
        if normal_samples:
            normal_values = np.concatenate(normal_samples)
            normal_median = float(np.median(normal_values))
            normal_calibration = {
                "median": normal_median,
                "mad": max(1e-6, float(np.median(np.abs(normal_values - normal_median)))),
            }
        probability, calibration = robust_probability(raw, normal_calibration)
        probability = apply_soft_spatial_prior(probability, context.semantic_regions)
        reliability = float(np.clip(0.25 + 0.75 * registration.confidence, 0.05, 1.0))
        result = EvidenceMap(
            source=self.name,
            values=probability,
            reliability=reliability,
            calibration=calibration,
            artifact_path=cache_path,
            metadata={
                "cache_hit": False,
                "registration_applied": registration.applied,
                "registration_inlier_ratio": registration.inlier_ratio,
                "registration_reprojection_error": registration.reprojection_error,
                "registration_reference_path": str(registration.reference_path),
                "calibration_source": "leave_one_normal_out" if normal_calibration else "target_fallback",
            },
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                values=result.values,
                reliability=result.reliability,
                median=result.calibration["median"],
                mad=result.calibration["mad"],
                registration_applied=registration.applied,
                registration_inlier_ratio=registration.inlier_ratio,
                registration_reprojection_error=registration.reprojection_error,
                calibration_source=result.metadata["calibration_source"],
            )
        return result

    def _cache_path(self, context: AutoMaskContext) -> Path | None:
        if self.artifact_dir is None:
            return None
        identity = (
            f"v1-loo|{context.cache_key}|{self.model_id}|{self.scale}|{self.layer}|{self.max_normals}|"
            f"{self.min_inlier_ratio}|{self.max_reprojection_error}"
        )
        return self.artifact_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.npz"


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


@dataclass
class SubspacePcaDinoProvider:
    """SubspaceAD-style evidence (arXiv 2602.23013): fit a PCA subspace to normal
    DINOv2 patch features and score test patches by reconstruction residual.

    Training-free, no memory bank; reuses the shared in-process normal-token
    cache. Complements the nearest-neighbour DINO provider — anomalies show up
    as variance orthogonal to the normal subspace rather than distance to a
    stored patch.
    """

    name: str = "dinov2_subspace"
    model_id: str = "facebook/dinov2-small"
    cache_dir: str | None = None
    device: str = "auto"
    scale: int = 448
    layers: tuple[int, ...] = (-4, -1)
    max_normals: int = 16
    variance: float = 0.9
    calibration_normals: int = 3
    artifact_dir: Path | None = None

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        processor, model, device = _load_model(self.model_id, self.cache_dir, self.device)
        image = Image.open(context.image_path).convert("RGB")
        target, grid = _tokens(processor, model, device, image, self.scale, self.layers)
        normal_tokens = [
            self._normal_tokens(processor, model, device, Path(path), self.scale)
            for path in context.normal_paths[: self.max_normals]
        ]
        if not normal_tokens:
            raise ValueError("PCA subspace evidence requires normal references")
        residual = self._reconstruction_residual(np.concatenate(normal_tokens, axis=0), target)
        raw = _resize_map(residual.reshape(grid), grid, context.image_size)
        calibration = self._loo_calibration(normal_tokens)
        probability, used = robust_probability(raw, calibration)
        probability = apply_soft_spatial_prior(probability, context.semantic_regions)
        normal_factor = min(1.0, len(context.normal_paths) / 4.0)
        nondegenerate = float(np.std(probability) >= 0.01)
        reliability = float(np.clip(0.40 + 0.35 * normal_factor + 0.25 * nondegenerate, 0.05, 1.0))
        return EvidenceMap(
            source=self.name,
            values=probability,
            reliability=reliability,
            calibration=used,
            augmentation_consistency=None,  # not measured; do not fabricate a value
            metadata={
                "scale": int(self.scale),
                "layers": list(self.layers),
                "pca_variance": self.variance,
                "normals_used": len(normal_tokens),
                "calibration_source": "leave_one_normal_out" if len(normal_tokens) >= 2 else "target_fallback",
            },
        )

    def _fit(self, feats: np.ndarray):
        from sklearn.decomposition import PCA

        variance = self.variance
        if not (0.0 < variance < 1.0):
            variance = min(int(variance), min(feats.shape) - 1) if variance >= 1 else 0.9
        pca = PCA(n_components=variance, svd_solver="full")
        pca.fit(np.asarray(feats, dtype=np.float32))
        return pca

    def _reconstruction_residual(self, normal_feats: np.ndarray, target: np.ndarray) -> np.ndarray:
        pca = self._fit(normal_feats)
        recon = pca.inverse_transform(pca.transform(np.asarray(target, dtype=np.float32)))
        return np.linalg.norm(target - recon, axis=1).astype(np.float32)

    def _loo_calibration(self, normal_tokens: list[np.ndarray]) -> dict[str, float] | None:
        if len(normal_tokens) < 2:
            return None
        samples: list[np.ndarray] = []
        for index in range(min(self.calibration_normals, len(normal_tokens))):
            others = np.concatenate([t for j, t in enumerate(normal_tokens) if j != index], axis=0)
            pca = self._fit(others)
            held = normal_tokens[index]
            recon = pca.inverse_transform(pca.transform(held))
            samples.append(np.linalg.norm(held - recon, axis=1))
        values = np.concatenate(samples)
        median = float(np.median(values))
        return {"median": median, "mad": max(1e-6, float(np.median(np.abs(values - median))))}

    def _normal_tokens(self, processor: Any, model: Any, device: Any, path: Path, scale: int) -> np.ndarray:
        # Share MultiScaleDinoProvider's in-process token cache (identical key).
        stat = path.stat()
        mem_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, self.model_id, tuple(self.layers), int(scale))
        cached = _NORMAL_TOKEN_CACHE.get(mem_key)
        if cached is not None:
            return cached
        normal = Image.open(path).convert("RGB")
        values, _ = _tokens(processor, model, device, normal, scale, self.layers)
        _NORMAL_TOKEN_CACHE[mem_key] = values
        return values
