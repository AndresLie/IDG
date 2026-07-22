from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


PatchFeatureCache = dict[Any, Any]


@dataclass(frozen=True)
class RegistrationResult:
    matrix: np.ndarray | None = None
    inlier_ratio: float = 0.0
    reprojection_error: float | None = None
    confidence: float = 0.0
    applied: bool = False
    reference_path: Path | None = None


@dataclass(frozen=True)
class AutoMaskContext:
    image_path: Path
    normal_paths: tuple[Path, ...]
    image_size: tuple[int, int]
    semantic_regions: tuple[tuple[int, int, int, int], ...]
    semantic_attributes: dict[str, Any] = field(default_factory=dict)
    foreground: np.ndarray | None = None
    registration: RegistrationResult | None = None
    cache_key: str = ""

    @property
    def primary_region(self) -> tuple[int, int, int, int]:
        if self.semantic_regions:
            return self.semantic_regions[0]
        return (0, 0, self.image_size[0], self.image_size[1])

DEFAULT_AUTO_CANDIDATE_MODES = [
    "normal_anomaly",
    "foundation_anomaly_field",
    "nearest_normal_residual",
    "patchcore_guided",
    "musc_mutual_score",
    "sam_prompt_regularized",
    "soft_patch",
    "scuff_cluster",
    "multi_scuff_fusion",
    "support_constrained_fusion",
    "normal_residual_fusion",
    "repeated_chain_refiner",
    "polar_rim_residual",
    "zipper_fabric_border_layout",
    "fft_texture_suppression",
    "sam2_heatmap",
    "scratch_band_clean",
    "structure_tensor_ridge",
    "residual",
    "multi_linear",
    "pixel",
    "procedural",
]


@dataclass(frozen=True)
class AutoMaskRecord:
    category: str
    defect_type: str
    image_path: str
    mask_path: str
    provider: str
    prompt: str
    description: str
    qwen_text: str | None
    qwen_defect_type: str | None
    confidence: float | None
    evidence: str | None
    region_xyxy: tuple[int, int, int, int]
    box_mask_path: str
    refined_mask_path: str
    inpaint_mask_path: str
    eval_mask_path: str | None
    training_mask_path: str | None
    uncertainty_mask_path: str | None
    mask_variant_paths: dict[str, str]
    mask_variant_overlay_paths: dict[str, str]
    overlay_path: str | None
    seed: int
    settings: dict[str, Any]


@dataclass(frozen=True)
class EvidenceMap:
    source: str
    values: np.ndarray
    reliability: float
    calibration: dict[str, float] = field(default_factory=dict)
    augmentation_consistency: float | None = None
    artifact_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("EvidenceMap.values must be a 2D array")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("EvidenceMap.reliability must be in [0, 1]")


@dataclass(frozen=True)
class CandidateProposal:
    mode: str
    mask: np.ndarray
    score: float
    measurements: dict[str, float] = field(default_factory=dict)
    evidence_sources: tuple[str, ...] = ()
    probability: np.ndarray | None = None
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("CandidateProposal.mask must be a 2D array")
        if self.probability is not None and self.probability.shape != self.mask.shape:
            raise ValueError("CandidateProposal.probability must match mask shape")


@dataclass(frozen=True)
class SelectionDecision:
    selected_mode: str | None
    expected_iou: float | None
    expected_precision: float | None
    expected_recall: float | None
    confidence: float
    disposition: str
    reasons: tuple[str, ...] = ()
    conformal_iou_lower_bound: float | None = None
    source_disagreement: float | None = None

    def __post_init__(self) -> None:
        if self.disposition not in {"hard_mask_ok", "soft_mask_only", "needs_review"}:
            raise ValueError(f"Unsupported selection disposition: {self.disposition}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SelectionDecision.confidence must be in [0, 1]")


class EvidenceProvider(Protocol):
    name: str

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        ...
