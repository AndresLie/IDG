from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from iadgen_v2.generation_critic import score_generation


def _mask_32() -> Image.Image:
    mask = Image.new("L", (32, 32), 0)
    ImageDraw.Draw(mask).rectangle((10, 10, 21, 21), fill=255)
    return mask


def test_visibility_separates_faint_smudge_from_real_defect() -> None:
    background = Image.new("RGB", (32, 32), (128, 128, 128))
    mask = _mask_32()
    settings = {"critic_image_size": 160}

    # Faint smudge: a barely-different fill inside the mask (games coverage).
    faint = background.copy()
    faint.paste(Image.new("RGB", background.size, (134, 134, 134)), mask=mask)
    faint_scores = score_generation(faint, background, mask, mask, settings, require_reference=False)

    # Real defect: strong, structured contrast inside the mask.
    strong = background.copy()
    strong.paste(Image.new("RGB", background.size, (190, 30, 30)), mask=mask)
    strong_scores = score_generation(strong, background, mask, mask, settings, require_reference=False)

    assert strong_scores.defect_visibility_score > faint_scores.defect_visibility_score + 0.3
    assert faint_scores.defect_visibility_score < 0.35
    assert strong_scores.defect_visibility_score > 0.6


def test_visibility_gate_rejects_faint_when_enabled() -> None:
    background = Image.new("RGB", (32, 32), (128, 128, 128))
    mask = _mask_32()
    faint = background.copy()
    faint.paste(Image.new("RGB", background.size, (134, 134, 134)), mask=mask)
    # Opt-in gate: enforce a minimum visibility.
    settings = {
        "critic_image_size": 160,
        "min_mask_coverage_score": 0.0,
        "min_texture_preservation_score": 0.0,
        "min_leakage_score": 0.0,
        "min_morphology_fit_score": 0.0,
        "min_defect_visibility_score": 0.35,
    }
    scored = score_generation(faint, background, mask, mask, settings, require_reference=False)
    assert "low_defect_visibility" in scored.reject_reasons


def test_visibility_gate_rejects_excessive_edit_when_upper_band_enabled() -> None:
    background = Image.new("RGB", (32, 32), (128, 128, 128))
    mask = _mask_32()
    excessive = background.copy()
    excessive.paste(Image.new("RGB", background.size, (250, 5, 5)), mask=mask)
    settings = {
        "critic_image_size": 160,
        "min_mask_coverage_score": 0.0,
        "min_texture_preservation_score": 0.0,
        "min_leakage_score": 0.0,
        "min_morphology_fit_score": 0.0,
        "max_defect_visibility_score": 0.55,
    }

    scored = score_generation(excessive, background, mask, mask, settings, require_reference=False)

    assert scored.defect_visibility_score > 0.55
    assert "excessive_defect_visibility" in scored.reject_reasons


def test_generation_critic_rejects_clean_output_and_accepts_visible_mask_edit() -> None:
    background = Image.new("RGB", (32, 32), (96, 96, 96))
    mask = Image.new("L", background.size, 0)
    ImageDraw.Draw(mask).rectangle((10, 10, 21, 21), fill=255)
    settings = {
        "min_mask_coverage_score": 0.35,
        "min_leakage_score": 0.45,
        "min_texture_preservation_score": 0.0,
        "min_morphology_fit_score": 0.0,
        "critic_image_size": 160,
    }

    clean = score_generation(
        background.copy(),
        background,
        mask,
        mask,
        settings,
        morphology="scratch_band",
        require_reference=False,
    )
    assert not clean.accepted
    assert "low_mask_coverage" in clean.reject_reasons

    edited = background.copy()
    edited.paste(Image.new("RGB", background.size, (180, 40, 40)), mask=mask)
    visible = score_generation(
        edited,
        background,
        mask,
        mask,
        settings,
        morphology="scratch_band",
        require_reference=False,
    )
    assert visible.accepted
    assert visible.adaptive_mask_coverage_score >= 0.35
    assert visible.leakage_score >= 0.45


def test_visibility_is_zero_on_structured_unchanged_background() -> None:
    # High-contrast structure (dark|light split) with the mask straddling the
    # edge, and the generated image identical to the background (no edit).
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :16] = 40
    arr[:, 16:] = 210
    bg = Image.fromarray(arr, "RGB")
    mask = Image.new("L", bg.size, 0)
    ImageDraw.Draw(mask).rectangle((8, 8, 23, 23), fill=255)  # spans the structure edge
    scores = score_generation(bg.copy(), bg, mask, mask, {"critic_image_size": 160}, require_reference=False)
    assert scores.defect_visibility_score < 0.05  # unchanged structure must not read as visible


def test_visibility_not_increased_by_ring_only_edit() -> None:
    background = Image.new("RGB", (48, 48), (128, 128, 128))
    mask = Image.new("L", background.size, 0)
    ImageDraw.Draw(mask).rectangle((18, 18, 29, 29), fill=255)  # central mask
    settings = {"critic_image_size": 160}

    # Edit only OUTSIDE the mask (a bright frame in the ring); mask region untouched.
    ring_only = background.copy()
    d = ImageDraw.Draw(ring_only)
    d.rectangle((10, 10, 37, 37), outline=(230, 30, 30), width=3)  # sits in the ring, not the mask
    ring_scores = score_generation(ring_only, background, mask, mask, settings, require_reference=False)

    # A genuine in-mask edit for comparison.
    in_mask = background.copy()
    in_mask.paste(Image.new("RGB", background.size, (210, 30, 30)), mask=mask)
    in_mask_scores = score_generation(in_mask, background, mask, mask, settings, require_reference=False)

    assert ring_scores.defect_visibility_score < 0.15                       # ring-only barely registers
    assert in_mask_scores.defect_visibility_score > ring_scores.defect_visibility_score + 0.3
