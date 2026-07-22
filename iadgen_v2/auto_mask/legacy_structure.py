from __future__ import annotations


def infer_structure_profile(category: str, defect_type: str, description: str = "") -> str:
    """Legacy text router retained only for specialist-baseline parity."""

    text = f"{category} {defect_type} {description}".lower().replace("-", "_")
    edge_terms = ("fabric_border", "border", "edge", "fray", "frayed", "side_band", "seam")
    chain_terms = ("zipper", "tooth", "teeth", "chain", "periodic", "repeated", "split_teeth", "broken_teeth")
    ring_terms = ("bottle", "rim", "annular", "circular_rim", "chip", "chipped", "broken_large", "broken_small", "glass")

    if any(term in text for term in edge_terms) and any(
        term in text for term in ("fabric", "fray", "border", "side_band", "seam", "zipper")
    ):
        return "edge_border"
    if any(term in text for term in chain_terms):
        return "repeated_chain"
    if any(term in text for term in ring_terms):
        if any(term in text for term in ("contamination", "stain", "foreign", "spot", "surface", "dirty")):
            return "flat_surface_patch"
        return "ring_sector"
    if any(term in text for term in ("scratch", "crack", "split", "line", "linear", "stroke")):
        return "thin_linear"
    if any(term in text for term in ("scuff", "smudge", "rub", "abrasion", "fuzzy")):
        return "multi_scuff"
    if any(term in text for term in ("contamination", "stain", "foreign", "spot", "surface", "patch")):
        return "flat_surface_patch"
    return "unknown"


def infer_structure_attributes(defect_type: str, description: str = "") -> dict[str, str]:
    text = f"{defect_type} {description}".lower().replace("-", "_")
    if any(term in text for term in ("micro", "tiny", "small", "pinpoint", "broken_small")):
        scale = "micro" if any(term in text for term in ("micro", "tiny", "pinpoint")) else "small"
    elif any(term in text for term in ("large", "broad", "wide", "broken_large")):
        scale = "large"
    else:
        scale = "unknown"
    return {"scale": scale}
