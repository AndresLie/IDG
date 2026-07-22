"""Edge-aware boundary refinement for generic anomaly candidates.

The generic proposal generator thresholds the fused anomaly field directly.
That field is low-resolution and blurry (patch-grid features, upsampled
residuals), so a hard threshold produces blobby masks whose boundaries follow
the evidence field rather than the true defect edges in the image. This is the
dominant reason localization recall is high while mask Dice saturates and the
oracle candidate is barely above the selected candidate: no candidate in the
pool has image-accurate boundaries, so the selector cannot recover them.

This module snaps the fused field to real image structure with an edge-aware
guided filter, then grows edge-hugging masks by hysteresis seeded from an
existing proposal. The result is a set of *additional* candidates that raise the
achievable ceiling; the selector is free to keep the original proposal when a
refined variant is not better supported, so the change cannot silently degrade a
sample below the current pool.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _box(values: np.ndarray, radius: int) -> np.ndarray:
    ksize = (2 * radius + 1, 2 * radius + 1)
    return cv2.boxFilter(values.astype(np.float32), ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """Single-channel guided filter (He et al., 2010) in pure OpenCV/NumPy.

    Transfers the edge structure of ``guide`` onto ``src`` so that transitions
    in ``src`` align with real image edges instead of the smooth evidence field.
    """

    guide = np.clip(np.asarray(guide, dtype=np.float32), 0.0, 1.0)
    src = np.asarray(src, dtype=np.float32)
    mean_guide = _box(guide, radius)
    mean_src = _box(src, radius)
    corr_guide = _box(guide * guide, radius)
    corr_cross = _box(guide * src, radius)
    var_guide = corr_guide - mean_guide * mean_guide
    cov_cross = corr_cross - mean_guide * mean_src
    a = cov_cross / (var_guide + eps)
    b = mean_src - a * mean_guide
    return _box(a, radius) * guide + _box(b, radius)


def edge_align_field(
    image: Image.Image,
    fused: np.ndarray,
    *,
    radius: int | None = None,
    eps: float = 1e-3,
) -> np.ndarray:
    """Return the fused anomaly field re-aligned to the image's luminance edges."""

    fused = np.clip(np.asarray(fused, dtype=np.float32), 0.0, 1.0)
    guide = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if guide.shape != fused.shape:
        guide = cv2.resize(guide, (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_AREA)
    if radius is None:
        radius = max(4, int(round(min(fused.shape) * 0.015)))
    aligned = guided_filter(guide, fused, radius=radius, eps=eps)
    return np.clip(aligned, 0.0, 1.0).astype(np.float32)


def _hysteresis(field: np.ndarray, roi: np.ndarray, seed: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = (field >= high) & roi & seed
    if not strong.any():
        strong = (field >= high) & roi
    permissive = (field >= low) & roi
    if not strong.any() or not permissive.any():
        return np.zeros(field.shape, dtype=bool)
    count, labels = cv2.connectedComponents(permissive.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[strong])
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros(field.shape, dtype=bool)
    return np.isin(labels, keep)


class EdgeAwareRefiner:
    """Callable matching the proposal refiner signature.

    Constructed with the pre-computed edge-aligned field so it can be applied to
    each base proposal without recomputing the guided filter. It produces a
    precision-leaning and a recall-leaning edge-hugging variant per proposal.
    """

    def __init__(
        self,
        aligned_field: np.ndarray,
        *,
        search_radius: int | None = None,
        low_quantiles: tuple[float, ...] = (0.60, 0.40),
        high_quantile: float = 0.80,
        min_area: int = 8,
    ) -> None:
        self.aligned_field = np.clip(np.asarray(aligned_field, dtype=np.float32), 0.0, 1.0)
        if search_radius is None:
            search_radius = max(3, int(round(min(self.aligned_field.shape) * 0.02)))
        self.search_radius = int(search_radius)
        self.low_quantiles = low_quantiles
        self.high_quantile = float(high_quantile)
        self.min_area = int(min_area)

    def __call__(
        self,
        fused: np.ndarray,  # noqa: ARG002 - interface parity with SamRefiner
        proposal: np.ndarray,
        region: tuple[int, int, int, int],  # noqa: ARG002 - unused; ROI comes from the proposal
    ) -> list[np.ndarray]:
        proposal = np.asarray(proposal, dtype=bool)
        if not proposal.any():
            return []
        field = self.aligned_field
        kernel = np.ones((3, 3), dtype=np.uint8)
        iterations = max(1, self.search_radius // 2)
        roi = cv2.dilate(proposal.astype(np.uint8), kernel, iterations=iterations) > 0
        roi_values = field[roi]
        if roi_values.size == 0:
            return []
        high = float(np.quantile(roi_values, self.high_quantile))
        results: list[np.ndarray] = []
        seen: set[bytes] = set()
        for low_q in self.low_quantiles:
            low = float(np.quantile(roi_values, low_q))
            if high <= low:
                high = low + 1e-3
            refined = _hysteresis(field, roi, proposal, low, high)
            if int(refined.sum()) < self.min_area:
                continue
            identity = np.packbits(refined).tobytes()
            if identity in seen:
                continue
            seen.add(identity)
            results.append(refined)
        return results
