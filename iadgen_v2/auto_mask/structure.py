from __future__ import annotations


def infer_structure_profile_from_measurements(
    *,
    repeated_texture_score: float,
    rim_geometry_score: float,
    boundary_contact: float,
    elongation: float,
) -> str:
    """Infer optional specialist topology without product or defect names."""

    if repeated_texture_score >= 0.55:
        return "repeated_chain"
    if rim_geometry_score >= 0.65:
        return "ring_sector"
    if boundary_contact >= 0.5:
        return "edge_border"
    if elongation >= 3.0:
        return "thin_linear"
    return "unknown"
