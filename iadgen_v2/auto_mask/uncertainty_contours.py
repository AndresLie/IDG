from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class UncertaintyContourContract:
    """Category-free contract for bounded, precision-oriented contours."""

    enabled: bool = False
    base_support_quantile: float = 0.80
    seed_quantile: float = 0.95
    contour_quantiles: tuple[float, ...] = (0.75, 0.85, 0.95)
    uncertainty_penalty: float = 0.75
    max_area_fraction: float = 0.10
    max_candidates_per_image: int = 3
    seed_dilation_iterations: int = 1

    def __post_init__(self) -> None:
        quantiles = (self.base_support_quantile, self.seed_quantile, *self.contour_quantiles)
        if any(not 0.0 < value < 1.0 for value in quantiles):
            raise ValueError("Uncertainty-contour quantiles must be in (0, 1)")
        if tuple(sorted(set(self.contour_quantiles))) != self.contour_quantiles:
            raise ValueError("Uncertainty-contour contour_quantiles must be unique and sorted")
        if not 0.0 <= self.uncertainty_penalty <= 1.0:
            raise ValueError("Uncertainty-contour uncertainty_penalty must be in [0, 1]")
        if not 0.0 < self.max_area_fraction <= 1.0:
            raise ValueError("Uncertainty-contour max_area_fraction must be in (0, 1]")
        if self.max_candidates_per_image < 1:
            raise ValueError("Uncertainty-contour max_candidates_per_image must be positive")
        if self.seed_dilation_iterations < 0:
            raise ValueError("Uncertainty-contour seed_dilation_iterations must be non-negative")

    @classmethod
    def from_mapping(cls, value: Any) -> UncertaintyContourContract:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("candidate_calibration.uncertainty_contours must be a mapping")
        return cls(
            enabled=bool(value.get("enabled", False)),
            base_support_quantile=float(value.get("base_support_quantile", cls.base_support_quantile)),
            seed_quantile=float(value.get("seed_quantile", cls.seed_quantile)),
            contour_quantiles=tuple(
                float(item) for item in value.get("contour_quantiles", cls.contour_quantiles)
            ),
            uncertainty_penalty=float(value.get("uncertainty_penalty", cls.uncertainty_penalty)),
            max_area_fraction=float(value.get("max_area_fraction", cls.max_area_fraction)),
            max_candidates_per_image=int(
                value.get("max_candidates_per_image", cls.max_candidates_per_image)
            ),
            seed_dilation_iterations=int(
                value.get("seed_dilation_iterations", cls.seed_dilation_iterations)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UncertaintyContourProposal:
    mode: str
    mask: np.ndarray
    quantile: float

    def __post_init__(self) -> None:
        if self.mask.ndim != 2 or self.mask.dtype != np.bool_ or not self.mask.any():
            raise ValueError("Uncertainty contours require a non-empty 2D boolean mask")


def generate_uncertainty_contour_candidates(
    fused: np.ndarray,
    disagreement: np.ndarray,
    posterior: np.ndarray,
    *,
    contract: UncertaintyContourContract,
    min_component_area: int,
) -> list[UncertaintyContourProposal]:
    """Extract seeded geodesic contours from category-free confidence ranks.

    A contour must be supported by high fused evidence, posterior confidence,
    and low source disagreement. Connected support is retained only when it
    touches a high-confidence seed, which trims uncertain component boundaries
    without introducing a category-specific size or intensity threshold.
    """

    if not contract.enabled:
        return []
    fused = _probability(fused)
    disagreement = _probability(disagreement)
    posterior = _probability(posterior)
    if fused.shape != disagreement.shape or fused.shape != posterior.shape:
        raise ValueError("Uncertainty-contour arrays must share one shape")

    fused_rank = _rank_probability(fused)
    posterior_rank = _rank_probability(posterior)
    certainty = 1.0 - contract.uncertainty_penalty * disagreement
    consensus = np.sqrt(fused_rank * posterior_rank) * np.clip(certainty, 0.0, 1.0)
    base_support = fused_rank >= contract.base_support_quantile
    support_values = consensus[base_support]
    if not support_values.size:
        return []

    seed_threshold = float(np.quantile(support_values, contract.seed_quantile))
    seeds = base_support & (consensus >= seed_threshold)
    seeds = _remove_small(seeds, max(1, int(min_component_area) // 2))
    if not seeds.any():
        return []
    if contract.seed_dilation_iterations:
        seed_contact = cv2.dilate(
            seeds.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=contract.seed_dilation_iterations,
        ) > 0
    else:
        seed_contact = seeds

    proposals: list[UncertaintyContourProposal] = []
    seen: set[bytes] = set()
    for quantile in contract.contour_quantiles:
        threshold = float(np.quantile(support_values, quantile))
        eligible = base_support & (consensus >= threshold)
        contour = _seeded_components(eligible, seed_contact, max(1, int(min_component_area)))
        if not contour.any() or float(contour.mean()) > contract.max_area_fraction:
            continue
        identity = np.packbits(contour).tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        proposals.append(
            UncertaintyContourProposal(
                mode=f"v4_uncertainty_contour_q{int(round(quantile * 1000)):03d}",
                mask=contour,
                quantile=float(quantile),
            )
        )
    return _diverse_budget(proposals, contract.max_candidates_per_image)


def _seeded_components(eligible: np.ndarray, seed_contact: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(eligible.astype(np.uint8), connectivity=8)
    output = np.zeros_like(eligible, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
            continue
        component = labels == index
        if bool((component & seed_contact).any()):
            output |= component
    return output


def _diverse_budget(
    proposals: list[UncertaintyContourProposal],
    maximum: int,
) -> list[UncertaintyContourProposal]:
    if len(proposals) <= maximum:
        return proposals
    indices = np.linspace(0, len(proposals) - 1, num=maximum)
    selected = sorted({int(round(index)) for index in indices})
    return [proposals[index] for index in selected[:maximum]]


def _rank_probability(values: np.ndarray) -> np.ndarray:
    if float(np.ptp(values)) <= 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    ordered = np.sort(values.reshape(-1))
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return ((left + right).astype(np.float32) / (2.0 * float(len(ordered)))).astype(np.float32)


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
