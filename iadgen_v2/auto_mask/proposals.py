from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from iadgen_v2.auto_mask.contracts import CandidateProposal, EvidenceMap


SamRefiner = Callable[[np.ndarray, np.ndarray, tuple[int, int, int, int]], list[np.ndarray]]


MEASUREMENT_NAMES = (
    "evidence_coverage",
    "normal_contrast",
    "source_agreement",
    "augmentation_stability",
    "component_count",
    "area_fraction",
    "compactness",
    "elongation",
    "boundary_contact",
    "foreground_containment",
    "periodicity_disruption",
    "sam_boundary_agreement",
    "source_disagreement",
)


def generate_generic_proposals(
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    region: tuple[int, int, int, int],
    *,
    foreground: np.ndarray | None = None,
    quantiles: tuple[float, ...] = (0.85, 0.90, 0.95, 0.975),
    min_area: int = 8,
    sam_refiner: SamRefiner | None = None,
    edge_refiner: SamRefiner | None = None,
) -> list[CandidateProposal]:
    region_values = _region_values(fused, region)
    proposals: list[CandidateProposal] = []
    seen: set[bytes] = set()
    for quantile in quantiles:
        threshold = float(np.quantile(region_values, quantile)) if region_values.size else 1.0
        threshold = max(0.05, threshold)
        raw = fused >= threshold
        raw = _remove_small(raw, min_area)
        _append_unique(
            proposals,
            seen,
            raw,
            mode=f"fused_q{int(round(quantile * 1000)):03d}",
            fused=fused,
            disagreement=disagreement,
            evidence=evidence,
            foreground=foreground,
            region=region,
            sam_boundary_agreement=0.0,
        )
        count, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
        components = [(labels == index) for index in range(1, count) if int((labels == index).sum()) >= min_area]
        components.sort(key=lambda mask: int(mask.sum()), reverse=True)
        for index, component in enumerate(components[:4]):
            _append_unique(
                proposals,
                seen,
                component,
                mode=f"fused_q{int(round(quantile * 1000)):03d}_component_{index + 1}",
                fused=fused,
                disagreement=disagreement,
                evidence=evidence,
                foreground=foreground,
                region=region,
                sam_boundary_agreement=0.0,
            )
    # SAM refines the raw fused blobs first; edge refinement then snaps BOTH the
    # fused blobs and the SAM masks to real image edges, so the whole boundary
    # family is edge-aware rather than only the threshold candidates.
    if sam_refiner is not None:
        sam_base = [proposal for proposal in proposals if proposal.mode.startswith("fused_")][:6]
        for refined in refine_proposals(
            sam_refiner, sam_base, fused, disagreement, evidence, foreground=foreground, region=region, prefix="sam2"
        ):
            _append_proposal(proposals, seen, refined)
    if edge_refiner is not None:
        edge_base = [proposal for proposal in proposals if proposal.mode.startswith(("fused_", "sam2_"))][:8]
        for refined in refine_proposals(
            edge_refiner, edge_base, fused, disagreement, evidence, foreground=foreground, region=region, prefix="edge"
        ):
            _append_proposal(proposals, seen, refined)
    return sorted(proposals, key=lambda proposal: proposal.score, reverse=True)


def refine_proposals(
    refiner: SamRefiner,
    seeds: list[CandidateProposal],
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    *,
    foreground: np.ndarray | None,
    region: tuple[int, int, int, int],
    prefix: str,
    min_support: float = 0.35,
) -> list[CandidateProposal]:
    """Apply a boundary refiner to seed proposals and return gated new proposals.

    Shared by SAM, edge, and specialist refinement so every refined candidate is
    scored identically and rejected when it is not supported by the fused field.
    """

    output: list[CandidateProposal] = []
    seen: set[bytes] = set()
    for seed in seeds:
        for index, refined in enumerate(refiner(fused, seed.mask, region)):
            refined = np.asarray(refined, dtype=bool)
            if not refined.any():
                continue
            if float(fused[refined].mean()) < min_support:
                continue
            identity = np.packbits(refined).tobytes()
            if identity in seen:
                continue
            seen.add(identity)
            measurements = proposal_measurements(
                refined,
                fused,
                disagreement,
                evidence,
                foreground=foreground,
                region=region,
                sam_boundary_agreement=_boundary_iou(refined, seed.mask),
            )
            output.append(
                CandidateProposal(
                    mode=f"{prefix}_{seed.mode}_{index + 1}",
                    mask=refined,
                    score=_score_from_measurements(measurements),
                    measurements=measurements,
                    evidence_sources=tuple(item.source for item in evidence),
                    probability=fused,
                    parent_mode=seed.mode,
                )
            )
    return output


def edge_family_code(mode: str) -> float:
    """Coarse provenance of a refined candidate, as a numeric feature.

    1 = edge-refined fused-threshold blob, 2 = edge-refined SAM mask,
    3 = edge-refined specialist. Lets the reliability model condition on where a
    refinement came from without depending on category names.
    """

    if mode.startswith("edge_sam2_"):
        return 2.0
    if mode.startswith("edge_fused_"):
        return 1.0
    return 3.0


def paired_feature_vector(
    edge_measurements: dict[str, float],
    baseline_measurements: dict[str, float],
    family_code: float = 0.0,
) -> list[float]:
    """Features for the edge-vs-baseline reliability model.

    Absolute edge measurements, then edge-minus-baseline deltas, then a
    provenance code. The baseline is the candidate the edge would displace at
    selection time (the best non-edge candidate), so the model predicts the gain
    of the actual swap decision, not merely edge-vs-parent.
    """

    edge = [float(edge_measurements.get(name, 0.0)) for name in MEASUREMENT_NAMES]
    delta = [
        float(edge_measurements.get(name, 0.0)) - float(baseline_measurements.get(name, 0.0))
        for name in MEASUREMENT_NAMES
    ]
    return edge + delta + [float(family_code)]


def proposal_from_mask(
    mode: str,
    mask: np.ndarray,
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    *,
    foreground: np.ndarray | None,
    region: tuple[int, int, int, int],
) -> CandidateProposal:
    proposals: list[CandidateProposal] = []
    _append_unique(
        proposals,
        set(),
        mask,
        mode=mode,
        fused=fused,
        disagreement=disagreement,
        evidence=evidence,
        foreground=foreground,
        region=region,
        sam_boundary_agreement=0.0,
    )
    if not proposals:
        raise ValueError(f"Specialist proposal {mode!r} is empty")
    return proposals[0]


def additive_map_proposals(
    source: str,
    evidence_map: np.ndarray,
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    region: tuple[int, int, int, int],
    *,
    foreground: np.ndarray | None = None,
    quantiles: tuple[float, ...] = (0.90, 0.95),
    min_area: int = 8,
    max_components: int = 2,
) -> list[CandidateProposal]:
    """Threshold an out-of-fusion evidence map into candidate masks and score them
    against the BASELINE ``fused`` map / ``evidence`` list, so they enter the pool
    as strictly-additive candidates: every baseline candidate is preserved and
    these are appended. Used for default-off PCA-residual candidates."""
    region_values = _region_values(evidence_map, region)
    proposals: list[CandidateProposal] = []
    seen: set[bytes] = set()
    for quantile in quantiles:
        threshold = max(0.05, float(np.quantile(region_values, quantile)) if region_values.size else 1.0)
        raw = _remove_small(evidence_map >= threshold, min_area)
        tag = f"{source}_q{int(round(quantile * 1000)):03d}"
        if raw.any():
            _append_unique(proposals, seen, raw, mode=tag, fused=fused, disagreement=disagreement,
                           evidence=evidence, foreground=foreground, region=region, sam_boundary_agreement=0.0)
        count, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
        components = sorted(
            ((labels == index) for index in range(1, count) if int((labels == index).sum()) >= min_area),
            key=lambda mask: int(mask.sum()), reverse=True,
        )
        for index, component in enumerate(components[:max_components]):
            _append_unique(proposals, seen, component, mode=f"{tag}_component_{index + 1}", fused=fused,
                           disagreement=disagreement, evidence=evidence, foreground=foreground,
                           region=region, sam_boundary_agreement=0.0)
    return proposals


def proposal_measurements(
    mask: np.ndarray,
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    *,
    foreground: np.ndarray | None,
    region: tuple[int, int, int, int],
    sam_boundary_agreement: float,
) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    height, width = mask.shape
    if area == 0:
        return {name: 0.0 for name in MEASUREMENT_NAMES}
    outside = ~mask
    evidence_coverage = float(fused[mask].mean())
    normal_contrast = evidence_coverage - (float(fused[outside].mean()) if outside.any() else 0.0)
    source_support = np.stack([item.values >= 0.5 for item in evidence], axis=0)
    source_agreement = float(source_support[:, mask].mean())
    augmentation = [item.augmentation_consistency for item in evidence if item.augmentation_consistency is not None]
    augmentation_stability = float(np.mean(augmentation)) if augmentation else 0.5
    component_count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    component_count = max(0, component_count - 1)
    ys, xs = np.where(mask)
    box_width, box_height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    compactness = float(area / max(1, box_width * box_height))
    elongation = float(max(box_width / max(1, box_height), box_height / max(1, box_width)))
    boundary_contact = float(
        np.mean(
            [
                bool(mask[0].any()),
                bool(mask[-1].any()),
                bool(mask[:, 0].any()),
                bool(mask[:, -1].any()),
            ]
        )
    )
    foreground_containment = float(foreground[mask].mean()) if foreground is not None else 1.0
    source_disagreement = float(disagreement[mask].mean())
    return {
        "evidence_coverage": evidence_coverage,
        "normal_contrast": normal_contrast,
        "source_agreement": source_agreement,
        "augmentation_stability": augmentation_stability,
        "component_count": float(component_count),
        "area_fraction": float(area / max(1, height * width)),
        "compactness": compactness,
        "elongation": elongation,
        "boundary_contact": boundary_contact,
        "foreground_containment": foreground_containment,
        "periodicity_disruption": _periodicity_disruption(mask, region),
        "sam_boundary_agreement": float(sam_boundary_agreement),
        "source_disagreement": source_disagreement,
    }


def _append_unique(
    proposals: list[CandidateProposal],
    seen: set[bytes],
    mask: np.ndarray,
    *,
    mode: str,
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    foreground: np.ndarray | None,
    region: tuple[int, int, int, int],
    sam_boundary_agreement: float,
) -> None:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return
    identity = np.packbits(mask).tobytes()
    if identity in seen:
        return
    seen.add(identity)
    measurements = proposal_measurements(
        mask,
        fused,
        disagreement,
        evidence,
        foreground=foreground,
        region=region,
        sam_boundary_agreement=sam_boundary_agreement,
    )
    proposals.append(
        CandidateProposal(
            mode=mode,
            mask=mask,
            score=_score_from_measurements(measurements),
            measurements=measurements,
            evidence_sources=tuple(item.source for item in evidence),
            probability=fused,
        )
    )


def _score_from_measurements(measurements: dict[str, float]) -> float:
    return float(
        0.40 * measurements["evidence_coverage"]
        + 0.22 * max(0.0, measurements["normal_contrast"])
        + 0.18 * measurements["source_agreement"]
        + 0.12 * measurements["augmentation_stability"]
        + 0.08 * measurements["foreground_containment"]
        - 0.20 * measurements["source_disagreement"]
        - 0.10 * max(0.0, measurements["area_fraction"] - 0.20)
    )


def _append_proposal(proposals: list[CandidateProposal], seen: set[bytes], proposal: CandidateProposal) -> None:
    if not proposal.mask.any():
        return
    identity = np.packbits(np.asarray(proposal.mask, dtype=bool)).tobytes()
    if identity in seen:
        return
    seen.add(identity)
    proposals.append(proposal)


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == index] = True
    return output


def _region_values(values: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = region
    crop = values[max(0, top) : min(values.shape[0], bottom), max(0, left) : min(values.shape[1], right)]
    return crop[np.isfinite(crop)]


def _boundary_iou(first: np.ndarray, second: np.ndarray) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    first_boundary = first ^ (cv2.erode(first.astype(np.uint8), kernel) > 0)
    second_boundary = second ^ (cv2.erode(second.astype(np.uint8), kernel) > 0)
    union = int((first_boundary | second_boundary).sum())
    return float((first_boundary & second_boundary).sum() / union) if union else 1.0


def _periodicity_disruption(mask: np.ndarray, region: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = region
    crop = mask[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    projection = crop.mean(axis=1 if crop.shape[0] >= crop.shape[1] else 0)
    if projection.size < 8 or float(projection.std()) < 1e-6:
        return 0.0
    spectrum = np.abs(np.fft.rfft(projection - projection.mean()))
    if spectrum.size <= 2:
        return 0.0
    return float(np.clip(spectrum[1:].max() / max(1e-6, spectrum[1:].sum()), 0.0, 1.0))
