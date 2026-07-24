from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GenerationCriticScores:
    score: float
    feature_alignment_score: float
    adaptive_mask_coverage_score: float
    adaptive_mask_coverage_fraction: float
    texture_preservation_score: float
    leakage_score: float
    morphology_fit_score: float
    outside_change_fraction: float
    shell_change_fraction: float
    reject_reasons: list[str]
    # Salience of the edit inside the mask (magnitude/structure/contrast), so a
    # low-contrast smudge that satisfies coverage still scores low. Coverage
    # counts *whether* pixels changed; visibility measures *how visibly*.
    defect_visibility_score: float = 0.0
    defect_visibility_magnitude: float = 0.0

    @property
    def accepted(self) -> bool:
        return not self.reject_reasons


def score_generation(
    generated_full: Image.Image,
    background_full: Image.Image,
    refined_mask: Image.Image | np.ndarray | Path,
    inpaint_mask: Image.Image | np.ndarray | Path | None,
    settings: dict[str, Any],
    *,
    morphology: str = "unknown",
    reference_image: Image.Image | None = None,
    reference_mask: Image.Image | np.ndarray | Path | None = None,
    require_reference: bool = True,
) -> GenerationCriticScores:
    """Score generated defects with the shared TF-IDG-lite critic.

    This is intentionally model-free and deterministic. Phase 4 uses it during
    retry selection; Phase 11 uses the same implementation for final reporting.
    """

    critic_size = int(settings.get("critic_image_size", 160))
    generated_source = generated_full.convert("RGB")
    background_source = background_full.convert("RGB").resize(generated_source.size, Image.Resampling.BILINEAR)
    generated = resize_for_critic(generated_source, critic_size)
    background = resize_for_critic(background_source, critic_size)
    refined = load_mask_array(refined_mask, generated.size)
    inpaint = load_mask_array(inpaint_mask, generated.size) if inpaint_mask is not None else refined
    effective_mask = np.maximum(refined, inpaint)

    if reference_image is not None and reference_mask is not None:
        reference_resized = resize_for_critic(reference_image.convert("RGB"), critic_size)
        reference_mask_arr = load_mask_array(reference_mask, reference_resized.size)
        alignment = _feature_alignment_score(generated, background, effective_mask, reference_resized, reference_mask_arr)
        has_reference = True
    else:
        alignment = 0.0 if require_reference else 1.0
        has_reference = False

    coverage, coverage_fraction = _adaptive_mask_coverage_score(generated, background, effective_mask, settings)
    texture = _texture_preservation_score(generated, background, effective_mask)
    leakage, outside_changed, shell_changed = _leakage_score(generated, background, refined, inpaint, settings)
    morphology_fit = _morphology_fit_score(generated, background, refined, morphology)
    visibility, visibility_magnitude = _defect_visibility_score(generated, background, refined, settings)

    weights = critic_weights(settings)
    score = (
        weights["feature_alignment"] * alignment
        + weights["mask_coverage"] * coverage
        + weights["texture_preservation"] * texture
        + weights["leakage"] * leakage
        + weights["morphology_fit"] * morphology_fit
        + weights["defect_visibility"] * visibility
    )
    reasons = _reject_reasons(
        alignment,
        coverage,
        texture,
        leakage,
        morphology_fit,
        visibility,
        settings,
        has_reference=has_reference,
        require_reference=require_reference,
    )
    return GenerationCriticScores(
        score=float(max(0.0, min(1.0, score))),
        feature_alignment_score=float(alignment),
        adaptive_mask_coverage_score=float(coverage),
        adaptive_mask_coverage_fraction=float(coverage_fraction),
        texture_preservation_score=float(texture),
        leakage_score=float(leakage),
        morphology_fit_score=float(morphology_fit),
        outside_change_fraction=float(outside_changed),
        shell_change_fraction=float(shell_changed),
        reject_reasons=reasons,
        defect_visibility_score=float(visibility),
        defect_visibility_magnitude=float(visibility_magnitude),
    )


def critic_weights(settings: dict[str, Any]) -> dict[str, float]:
    configured = settings.get("weights", {}) if isinstance(settings.get("weights"), dict) else {}
    # Coverage demoted (it is gameable by low-contrast edits); defect_visibility
    # added so the objective rewards edits that are actually visible as defects.
    weights = {
        "feature_alignment": float(configured.get("feature_alignment", 0.32)),
        "mask_coverage": float(configured.get("mask_coverage", 0.12)),
        "texture_preservation": float(configured.get("texture_preservation", 0.22)),
        "leakage": float(configured.get("leakage", 0.14)),
        "morphology_fit": float(configured.get("morphology_fit", 0.08)),
        "defect_visibility": float(configured.get("defect_visibility", 0.12)),
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else weights


def load_mask_array(mask: Image.Image | np.ndarray | Path | None, size: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros((size[1], size[0]), dtype=np.float32)
    if isinstance(mask, np.ndarray):
        arr = mask.astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr = arr / 255.0
        if arr.shape == (size[1], size[0]):
            return np.clip(arr, 0.0, 1.0)
        image = Image.fromarray(np.uint8(np.clip(arr, 0.0, 1.0) * 255.0), mode="L")
    elif isinstance(mask, Path):
        if not mask.exists():
            return np.zeros((size[1], size[0]), dtype=np.float32)
        image = Image.open(mask).convert("L")
    else:
        image = mask.convert("L")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def resize_for_critic(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0 or max(image.size) <= max_size:
        return image
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return resized


def _feature_alignment_score(
    generated: Image.Image,
    background: Image.Image,
    mask: np.ndarray,
    reference: Image.Image,
    reference_mask: np.ndarray,
) -> float:
    gen_vec = _defect_feature_vector(generated, mask, background)
    ref_vec = _defect_feature_vector(reference, reference_mask, None)
    if gen_vec is None or ref_vec is None:
        return 0.0
    distance = float(np.linalg.norm(gen_vec - ref_vec) / math.sqrt(len(gen_vec)))
    return float(math.exp(-2.4 * distance))


def _defect_feature_vector(image: Image.Image, mask: np.ndarray, background: Image.Image | None) -> np.ndarray | None:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    weights = np.clip(mask.astype(np.float32), 0.0, 1.0)
    active = weights > 0.03
    if int(active.sum()) < 8:
        return None
    weights = weights / max(float(weights.sum()), 1e-6)
    gray = _gray(rgb)
    grad = _gradient_magnitude(gray)
    detail = np.abs(gray - _box_blur(gray, radius=3))
    features: list[float] = []
    for channel in range(3):
        features.extend(_weighted_mean_std(rgb[:, :, channel], weights))
    features.extend(_weighted_mean_std(gray, weights))
    features.extend(_weighted_mean_std(grad, weights))
    features.extend(_weighted_mean_std(detail, weights))
    if background is not None:
        bg = np.asarray(background, dtype=np.float32) / 255.0
        diff = np.abs(rgb - bg).mean(axis=2)
        features.extend(_weighted_mean_std(diff, weights))
    else:
        features.extend([0.0, 0.0])
    return np.asarray(features, dtype=np.float32)


def _adaptive_mask_coverage_score(
    generated: Image.Image,
    background: Image.Image,
    mask: np.ndarray,
    settings: dict[str, Any],
) -> tuple[float, float]:
    active = mask > 0.03
    if int(active.sum()) < 8:
        return 0.0, 0.0
    diff = _image_diff(generated, background)
    threshold = float(settings.get("change_threshold", 0.045))
    changed = diff > threshold
    soft_weights = np.clip(mask, 0.0, 1.0)
    weighted_changed = float((changed.astype(np.float32) * soft_weights).sum() / max(float(soft_weights.sum()), 1e-6))
    target = float(settings.get("target_mask_coverage", 0.38))
    if weighted_changed <= target:
        return max(0.0, weighted_changed / max(target, 1e-6)), weighted_changed
    over = max(0.0, weighted_changed - float(settings.get("max_mask_coverage", 0.92)))
    return max(0.0, 1.0 - over / 0.25), weighted_changed


def _texture_preservation_score(generated: Image.Image, background: Image.Image, mask: np.ndarray) -> float:
    outside = mask <= 0.02
    if int(outside.sum()) < 8:
        return 0.0
    gen = np.asarray(generated, dtype=np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32) / 255.0
    rgb_l1 = float(np.abs(gen - bg).mean(axis=2)[outside].mean())
    gen_grad = _gradient_magnitude(_gray(gen))
    bg_grad = _gradient_magnitude(_gray(bg))
    grad_l1 = float(np.abs(gen_grad - bg_grad)[outside].mean())
    return float(max(0.0, min(1.0, 1.0 - 8.0 * rgb_l1 - 2.5 * grad_l1)))


def _leakage_score(
    generated: Image.Image,
    background: Image.Image,
    refined: np.ndarray,
    inpaint: np.ndarray,
    settings: dict[str, Any],
) -> tuple[float, float, float]:
    diff = _image_diff(generated, background)
    threshold = float(settings.get("change_threshold", 0.045))
    changed = diff > threshold
    refined_active = refined > 0.03
    inpaint_active = inpaint > 0.03
    outside = ~refined_active
    shell = inpaint_active & ~refined_active
    outside_changed = float(changed[outside].mean()) if outside.any() else 0.0
    shell_changed = float(changed[shell].mean()) if shell.any() else 0.0
    max_outside = float(settings.get("max_outside_change_fraction", 0.18))
    outside_score = 1.0 - outside_changed / max(max_outside, 1e-6)
    shell_score = 1.0 - max(0.0, shell_changed - 0.35) / 0.45
    score = float(max(0.0, min(1.0, 0.75 * outside_score + 0.25 * shell_score)))
    return score, outside_changed, shell_changed


def _morphology_fit_score(generated: Image.Image, background: Image.Image, refined: np.ndarray, morphology: str) -> float:
    active = refined > 0.05
    if int(active.sum()) < 8:
        return 0.0
    diff = _image_diff(generated, background)
    strong = active & (diff > max(0.045, float(np.percentile(diff[active], 60.0))))
    if int(strong.sum()) < 8:
        strong = active
    ys, xs = np.where(strong)
    aspect = _aspect(xs, ys)
    components = _connected_component_count(strong)
    if morphology in {"scratch_band", "single_stroke", "crack_band"}:
        aspect_score = min(1.0, max(0.0, (aspect - 1.5) / 6.0))
        frag_score = max(0.0, 1.0 - max(0, components - 8) / 20.0)
        return float(0.7 * aspect_score + 0.3 * frag_score)
    if morphology == "multi_scuff":
        area = float(strong.sum() / max(1, refined.size))
        area_score = max(0.0, min(1.0, area / 0.025))
        aspect_score = max(0.0, 1.0 - max(0.0, aspect - 8.0) / 12.0)
        return float(0.55 * area_score + 0.45 * aspect_score)
    return 0.6


def _reject_reasons(
    alignment: float,
    coverage: float,
    texture: float,
    leakage: float,
    morphology_fit: float,
    visibility: float,
    settings: dict[str, Any],
    *,
    has_reference: bool,
    require_reference: bool,
) -> list[str]:
    reasons: list[str] = []
    if require_reference and not has_reference:
        reasons.append("missing_reference")
    if alignment < float(settings.get("min_feature_alignment_score", 0.28 if require_reference else 0.0)):
        reasons.append("low_feature_alignment")
    min_coverage = settings.get("min_mask_coverage_score", settings.get("min_adaptive_mask_coverage_score", 0.35))
    if coverage < float(min_coverage):
        reasons.append("low_mask_coverage")
    if texture < float(settings.get("min_texture_preservation_score", 0.55)):
        reasons.append("poor_texture_preservation")
    if leakage < float(settings.get("min_leakage_score", 0.45)):
        reasons.append("high_leakage")
    if morphology_fit < float(settings.get("min_morphology_fit_score", 0.20)):
        reasons.append("poor_morphology_fit")
    # Defaults to 0.0 (off) until a generation re-audit tunes the threshold, so
    # this term does not silently reject the current corpus. Turn it on via
    # settings.min_defect_visibility_score to enforce visible defects.
    if visibility < float(settings.get("min_defect_visibility_score", 0.0)):
        reasons.append("low_defect_visibility")
    return reasons


def _defect_visibility_score(
    generated: Image.Image,
    background: Image.Image,
    refined: np.ndarray,
    settings: dict[str, Any],
) -> tuple[float, float]:
    """How visibly the in-mask edit reads as a defect (not just whether it changed).

    Every term is defined on the generated-minus-background CHANGE (never on
    absolute generated intensity), so an unchanged high-contrast structure scores
    ~0 and edits confined to the surrounding ring cannot raise the score. Terms:
    in-mask change magnitude, added edge/gradient structure, and change contrast
    of the mask against its ring. A faint smudge scores low even at high coverage.
    """

    active = refined > 0.05
    if int(active.sum()) < 8:
        return 0.0, 0.0
    gen = np.asarray(generated, dtype=np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32) / 255.0
    gray_gen = _gray(gen)
    gray_bg = _gray(bg)
    diff = np.abs(gen - bg).mean(axis=2)
    deviation = float(diff[active].mean())
    grad_gain = float(np.clip((_gradient_magnitude(gray_gen) - _gradient_magnitude(gray_bg))[active], 0.0, None).mean())
    # Change-based contrast: how much MORE the mask changed than its ring. Using
    # the change map (|gen - bg|) rather than absolute intensity means a
    # structured but unchanged region scores 0, and a ring-only edit (change
    # concentrated outside the mask) yields a negative difference -> clamped to 0.
    change = np.abs(gray_gen - gray_bg)
    ring = (_box_blur(active.astype(np.float32), radius=3) > 0.0) & (~active)
    reference_region = ring if ring.any() else (~active)
    if reference_region.any():
        contrast = max(0.0, float(change[active].mean()) - float(change[reference_region].mean()))
    else:
        contrast = 0.0
    raw = (
        float(settings.get("visibility_deviation_gain", 6.0)) * deviation
        + float(settings.get("visibility_gradient_gain", 3.0)) * grad_gain
        + float(settings.get("visibility_contrast_gain", 4.0)) * contrast
    )
    return float(1.0 - math.exp(-raw)), deviation


def _image_diff(a: Image.Image, b: Image.Image) -> np.ndarray:
    arr_a = np.asarray(a, dtype=np.float32) / 255.0
    arr_b = np.asarray(b, dtype=np.float32) / 255.0
    return np.abs(arr_a - arr_b).mean(axis=2)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.hypot(gx, gy).astype(np.float32)


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    pad = radius
    padded = np.pad(values, pad, mode="reflect")
    out = np.zeros_like(values, dtype=np.float32)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return out / float(size * size)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> list[float]:
    mean_value = float((values * weights).sum())
    var = float((((values - mean_value) ** 2) * weights).sum())
    return [mean_value, math.sqrt(max(0.0, var))]


def _aspect(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size == 0 or ys.size == 0:
        return 0.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return max(width / max(1.0, height), height / max(1.0, width))


def _connected_component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return count
