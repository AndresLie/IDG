from __future__ import annotations

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
