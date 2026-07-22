from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.contracts import AutoMaskContext, EvidenceMap


HeatmapBuilder = Callable[[Image.Image, list, tuple[int, int, int, int]], tuple[np.ndarray, dict[str, Any]]]


def robust_probability(values: np.ndarray, calibration: dict[str, float] | None = None) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32), {"median": 0.0, "mad": 1.0}
    supplied = calibration or {}
    median = float(supplied.get("median", np.median(finite)))
    mad = float(supplied.get("mad", np.median(np.abs(finite - median))))
    scale = max(1e-6, 1.4826 * mad)
    z = np.clip((array - median) / scale, -8.0, 8.0)
    probability = 1.0 / (1.0 + np.exp(-(z - 1.5)))
    probability[~np.isfinite(probability)] = 0.0
    return probability.astype(np.float32), {"median": median, "mad": mad, "robust_scale": scale}


def apply_soft_spatial_prior(
    values: np.ndarray,
    regions: tuple[tuple[int, int, int, int], ...],
    *,
    margin_fraction: float = 0.10,
    inside_weight: float = 1.0,
    margin_weight: float = 0.5,
    outside_weight: float = 0.2,
) -> np.ndarray:
    height, width = values.shape
    weights = np.full((height, width), float(outside_weight), dtype=np.float32)
    for left, top, right, bottom in regions:
        pad_x = int(round(max(1, right - left) * margin_fraction))
        pad_y = int(round(max(1, bottom - top) * margin_fraction))
        x1, y1 = max(0, left - pad_x), max(0, top - pad_y)
        x2, y2 = min(width, right + pad_x), min(height, bottom + pad_y)
        weights[y1:y2, x1:x2] = np.maximum(weights[y1:y2, x1:x2], margin_weight)
        weights[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = inside_weight
    return np.clip(values * weights, 0.0, 1.0).astype(np.float32)


def fuse_evidence_maps(evidence: list[EvidenceMap]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not evidence:
        raise ValueError("At least one evidence map is required")
    shape = evidence[0].values.shape
    if any(item.values.shape != shape for item in evidence):
        raise ValueError("All evidence maps must share one spatial shape")
    probabilities = np.stack([np.clip(item.values, 1e-5, 1.0 - 1e-5) for item in evidence], axis=0)
    reliability = np.asarray([max(0.05, item.reliability) for item in evidence], dtype=np.float32)
    reliability /= max(1e-6, float(reliability.sum()))
    logits = np.log(probabilities / (1.0 - probabilities))
    fused = 1.0 / (1.0 + np.exp(-np.sum(logits * reliability[:, None, None], axis=0)))
    disagreement = np.sqrt(np.sum(((probabilities - fused[None]) ** 2) * reliability[:, None, None], axis=0))
    return fused.astype(np.float32), np.clip(disagreement * 2.0, 0.0, 1.0).astype(np.float32), {
        "sources": [item.source for item in evidence],
        "normalized_weights": {item.source: float(weight) for item, weight in zip(evidence, reliability, strict=True)},
        "mean_disagreement": float(disagreement.mean()),
    }


@dataclass
class FunctionalEvidenceProvider:
    name: str
    builder: HeatmapBuilder
    base_reliability: float = 0.7
    calibration_normals: int = 3
    calibration_stride: int = 8
    artifact_dir: Path | None = None
    cache_identity: str = "v1"

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        cache_path = self._evidence_cache_path(context)
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path)
            return EvidenceMap(
                source=self.name,
                values=cached["values"].astype(np.float32),
                reliability=float(cached["reliability"]),
                calibration={"median": float(cached["median"]), "mad": float(cached["mad"])},
                augmentation_consistency=float(cached["consistency"]),
                artifact_path=cache_path,
                metadata={"cache_hit": True, "calibration_source": str(cached["calibration_source"])},
            )
        image = Image.open(context.image_path).convert("RGB")
        # Providers produce an unfenced anomaly field. Qwen is applied only
        # afterward as a soft prior, so evidence outside its region survives.
        full_region = (0, 0, context.image_size[0], context.image_size[1])
        raw, metadata = self.builder(image, list(context.normal_paths), full_region)
        normal_calibration = metadata.get("normal_calibration")
        calibration_source = "provider"
        if not normal_calibration:
            normal_calibration = self._leave_one_normal_out_calibration(context)
            calibration_source = "leave_one_normal_out" if normal_calibration else "target_fallback"
        probability, calibration = robust_probability(raw, normal_calibration)
        probability = apply_soft_spatial_prior(probability, context.semantic_regions)
        nondegenerate = float(np.std(probability) >= 0.01)
        normal_factor = min(1.0, len(context.normal_paths) / 4.0)
        consistency = float(metadata.get("augmentation_consistency", 0.75))
        reliability = np.clip(self.base_reliability * (0.35 + 0.35 * normal_factor + 0.30 * consistency) * nondegenerate, 0.05, 1.0)
        result = EvidenceMap(
            source=self.name,
            values=probability,
            reliability=float(reliability),
            calibration=calibration,
            augmentation_consistency=consistency,
            artifact_path=cache_path,
            metadata={**metadata, "cache_hit": False, "calibration_source": calibration_source},
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                values=result.values,
                reliability=result.reliability,
                median=result.calibration["median"],
                mad=result.calibration["mad"],
                consistency=result.augmentation_consistency,
                calibration_source=calibration_source,
            )
        return result

    def _leave_one_normal_out_calibration(self, context: AutoMaskContext) -> dict[str, float] | None:
        paths = list(context.normal_paths)
        if len(paths) < 2:
            return None
        cache_path = self._calibration_cache_path(context)
        if cache_path is not None and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return {"median": float(cached["median"]), "mad": float(cached["mad"])}
        samples: list[np.ndarray] = []
        for index, path in enumerate(paths[: max(1, self.calibration_normals)]):
            references = [item for item in paths if item != path]
            if not references:
                continue
            try:
                normal = Image.open(path).convert("RGB")
                raw, _ = self.builder(normal, references, (0, 0, normal.width, normal.height))
            except Exception:
                continue
            finite = np.asarray(raw, dtype=np.float32)[:: max(1, self.calibration_stride), :: max(1, self.calibration_stride)]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                samples.append(finite)
        if not samples:
            return None
        values = np.concatenate(samples)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        calibration = {"median": median, "mad": max(1e-6, mad)}
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(calibration, sort_keys=True) + "\n", encoding="utf-8")
        return calibration

    def _evidence_cache_path(self, context: AutoMaskContext) -> Path | None:
        if self.artifact_dir is None:
            return None
        identity = f"{self.cache_identity}|{self.name}|{context.cache_key}"
        return self.artifact_dir / "evidence" / f"{hashlib.sha256(identity.encode()).hexdigest()}.npz"

    def _calibration_cache_path(self, context: AutoMaskContext) -> Path | None:
        if self.artifact_dir is None:
            return None
        normal_rows = []
        for path in context.normal_paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            normal_rows.append((str(path.resolve()), path.stat().st_size, digest))
        identity = json.dumps(
            {
                "version": self.cache_identity,
                "provider": self.name,
                "image_size": context.image_size,
                "normals": normal_rows,
                "calibration_normals": self.calibration_normals,
                "calibration_stride": self.calibration_stride,
            },
            sort_keys=True,
        )
        return self.artifact_dir / "normal_calibration" / f"{hashlib.sha256(identity.encode()).hexdigest()}.json"
