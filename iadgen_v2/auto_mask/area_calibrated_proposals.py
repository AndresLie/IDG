from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CandidateAreaCalibrationContract:
    """Category-free contract for precision-oriented additive candidates."""

    enabled: bool = False
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.975)
    component_counts: tuple[int, ...] = (2, 4, 8)
    seed_quantile: float = 0.99
    support_quantiles: tuple[float, ...] = (0.90, 0.95)
    max_components: int = 32
    max_area_fraction: float = 0.10

    def __post_init__(self) -> None:
        if any(not 0.0 < value < 1.0 for value in (*self.quantiles, *self.support_quantiles)):
            raise ValueError("Area-calibrated proposal quantiles must be in (0, 1)")
        if not 0.0 < self.seed_quantile < 1.0:
            raise ValueError("Area-calibrated seed_quantile must be in (0, 1)")
        if tuple(sorted(set(self.quantiles))) != self.quantiles:
            raise ValueError("Area-calibrated quantiles must be unique and sorted")
        if tuple(sorted(set(self.support_quantiles))) != self.support_quantiles:
            raise ValueError("Area-calibrated support_quantiles must be unique and sorted")
        if not self.component_counts or any(value < 2 for value in self.component_counts):
            raise ValueError("Area-calibrated component_counts must contain integers >= 2")
        if tuple(sorted(set(self.component_counts))) != self.component_counts:
            raise ValueError("Area-calibrated component_counts must be unique and sorted")
        if self.max_components < max(self.component_counts):
            raise ValueError("Area-calibrated max_components must cover every component count")
        if not 0.0 < self.max_area_fraction <= 1.0:
            raise ValueError("Area-calibrated max_area_fraction must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Any) -> CandidateAreaCalibrationContract:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("candidate_calibration.area_calibrated_proposals must be a mapping")
        return cls(
            enabled=bool(value.get("enabled", False)),
            quantiles=tuple(float(item) for item in value.get("quantiles", cls.quantiles)),
            component_counts=tuple(int(item) for item in value.get("component_counts", cls.component_counts)),
            seed_quantile=float(value.get("seed_quantile", cls.seed_quantile)),
            support_quantiles=tuple(
                float(item) for item in value.get("support_quantiles", cls.support_quantiles)
            ),
            max_components=int(value.get("max_components", cls.max_components)),
            max_area_fraction=float(value.get("max_area_fraction", cls.max_area_fraction)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AreaCalibratedMaskProposal:
    mode: str
    mask: np.ndarray
    quantile: float

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_ or not self.mask.any():
            raise ValueError("Area-calibrated proposals require a non-empty 2D boolean mask")


def generate_area_calibrated_candidates(
    fused: np.ndarray,
    *,
    contract: CandidateAreaCalibrationContract,
    min_component_area: int,
) -> list[AreaCalibratedMaskProposal]:
    """Build strict-additive candidates with bounded area and component count.

    Legacy pools expose the complete threshold mask and isolated components. This
    family fills the missing middle: cumulative unions of the strongest evidence
    components, plus low-threshold components anchored by high-confidence seeds.
    All thresholds are within-image ranks, so no dataset/category scale is used.
    """

    if not contract.enabled:
        return []
    values = _probability(fused)
    minimum_area = max(1, int(min_component_area))
    proposals: list[AreaCalibratedMaskProposal] = []
    seen: set[bytes] = set()
    for quantile in contract.quantiles:
        threshold = max(0.05, float(np.quantile(values, quantile)))
        components = _ranked_components(
            values,
            threshold=threshold,
            min_area=minimum_area,
            max_components=contract.max_components,
        )
        _append_component_unions(
            proposals,
            seen,
            components,
            component_counts=contract.component_counts,
            max_area_fraction=contract.max_area_fraction,
            mode_prefix=f"v4_evidence_union_q{int(round(quantile * 1000)):03d}",
            quantile=quantile,
        )

    seed_threshold = max(0.05, float(np.quantile(values, contract.seed_quantile)))
    seeds = _remove_small(values >= seed_threshold, max(1, minimum_area // 2))
    if seeds.any():
        seed_neighborhood = cv2.dilate(seeds.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for quantile in contract.support_quantiles:
            support_threshold = max(0.05, float(np.quantile(values, quantile)))
            supported = [
                component
                for component in _ranked_components(
                    values,
                    threshold=support_threshold,
                    min_area=minimum_area,
                    max_components=contract.max_components,
                )
                if bool((component & seed_neighborhood).any())
            ]
            _append_component_unions(
                proposals,
                seen,
                supported,
                component_counts=contract.component_counts,
                max_area_fraction=contract.max_area_fraction,
                mode_prefix=(
                    f"v4_seeded_q{int(round(contract.seed_quantile * 1000)):03d}"
                    f"_support_q{int(round(quantile * 1000)):03d}"
                ),
                quantile=quantile,
            )
    return proposals


def _append_component_unions(
    output: list[AreaCalibratedMaskProposal],
    seen: set[bytes],
    components: list[np.ndarray],
    *,
    component_counts: tuple[int, ...],
    max_area_fraction: float,
    mode_prefix: str,
    quantile: float,
) -> None:
    if len(components) < 2:
        return
    for requested_count in component_counts:
        count = min(requested_count, len(components))
        union = np.logical_or.reduce(components[:count])
        if float(union.mean()) > max_area_fraction:
            continue
        identity = np.packbits(union).tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        output.append(
            AreaCalibratedMaskProposal(
                mode=f"{mode_prefix}_top{count}",
                mask=union,
                quantile=float(quantile),
            )
        )


def _ranked_components(
    fused: np.ndarray,
    *,
    threshold: float,
    min_area: int,
    max_components: int,
) -> list[np.ndarray]:
    raw = _remove_small(fused >= threshold, min_area)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
    ranked: list[tuple[float, int, np.ndarray]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == index
        component_values = fused[component]
        mean_excess = float(np.maximum(component_values - threshold, 0.0).mean())
        # Log-area support suppresses isolated hot pixels without allowing large,
        # weak texture regions to dominate as they do under area-only ordering.
        priority = mean_excess * float(np.log1p(area)) + 0.05 * float(component_values.mean())
        ranked.append((priority, area, component))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [component for _, _, component in ranked[:max_components]]


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == index] = True
    return output


def _probability(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    output[~np.isfinite(output)] = 0.0
    return np.clip(output, 0.0, 1.0)
