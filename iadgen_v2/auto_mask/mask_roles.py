from __future__ import annotations

import cv2
import numpy as np

from iadgen_v2.auto_mask.contracts import CandidateProposal, EvidenceMap, SelectionDecision


def posterior_mask_roles(
    fused: np.ndarray,
    disagreement: np.ndarray,
    evidence: list[EvidenceMap],
    selected: CandidateProposal,
    decision: SelectionDecision,
) -> dict[str, np.ndarray]:
    support = np.mean(np.stack([item.values >= 0.5 for item in evidence], axis=0), axis=0)
    registration_uncertainty = max(
        [1.0 - float(item.metadata.get("registration_inlier_ratio", 1.0)) for item in evidence if "registration_inlier_ratio" in item.metadata]
        or [0.0]
    )
    uncertainty = np.clip(disagreement + 0.25 * registration_uncertainty, 0.0, 1.0)
    positive_core = (fused >= 0.80) & (support >= 0.5) & selected.mask
    possible_region = (fused >= 0.35) | selected.mask
    if not positive_core.any():
        positive_core = selected.mask & (fused >= float(np.quantile(fused[selected.mask], 0.65)))
    training_soft = np.clip(fused * (1.0 - 0.75 * uncertainty) * possible_region, 0.0, 1.0)
    if decision.disposition == "hard_mask_ok":
        eval_tight = selected.mask
    else:
        eval_tight = positive_core
    kernel = np.ones((5, 5), dtype=np.uint8)
    medium = cv2.dilate(eval_tight.astype(np.uint8), kernel, iterations=1) > 0
    wide = cv2.dilate(eval_tight.astype(np.uint8), kernel, iterations=2) > 0
    inpaint = cv2.GaussianBlur((possible_region.astype(np.uint8) * 255), (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
    return {
        "generation_core": positive_core.astype(np.float32),
        "eval_tight": eval_tight.astype(np.float32),
        "eval_mvtec": eval_tight.astype(np.float32),
        "training_medium": medium.astype(np.float32),
        "training_wide": wide.astype(np.float32),
        "training_soft": training_soft.astype(np.float32),
        "positive_core": positive_core.astype(np.float32),
        "possible_region": possible_region.astype(np.float32),
        "uncertainty_map": uncertainty.astype(np.float32),
        "inpaint_soft": inpaint.astype(np.float32),
    }
