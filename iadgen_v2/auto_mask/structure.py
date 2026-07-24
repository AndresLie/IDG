from __future__ import annotations

from typing import Any, Mapping


def infer_structure_profile_from_measurements(
    *,
    repeated_texture_score: float,
    rim_geometry_score: float,
    boundary_contact: float,
    elongation: float,
    repeated_texture_threshold: float = 0.55,
    rim_geometry_threshold: float = 1.15,
) -> str:
    """Infer optional specialist topology without product or defect names."""

    if repeated_texture_score >= repeated_texture_threshold:
        return "repeated_chain"
    if rim_geometry_score >= rim_geometry_threshold:
        return "ring_sector"
    if elongation >= 3.0:
        return "thin_linear"
    if boundary_contact >= 0.5:
        return "edge_border"
    return "unknown"


def evaluate_evidence_gate(
    mode: str,
    attributes: Mapping[str, Any],
    *,
    repeated_texture_threshold: float = 0.20,
) -> tuple[bool, dict[str, Any]]:
    """Gate optional evidence using measured topology, never category names."""

    normalized = str(mode).strip().lower()
    if normalized == "always":
        return True, {"mode": normalized, "passed": True, "reason": "unconditional"}
    if normalized == "repeated_texture":
        score = float(attributes.get("repeated_texture_score", 0.0))
        threshold = float(repeated_texture_threshold)
        passed = score >= threshold
        return passed, {
            "mode": normalized,
            "passed": passed,
            "reason": "score_meets_threshold" if passed else "score_below_threshold",
            "repeated_texture_score": score,
            "repeated_texture_threshold": threshold,
        }
    raise ValueError(f"Unsupported evidence gate mode: {mode}")
