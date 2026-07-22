from __future__ import annotations

from PIL import Image, ImageDraw

from iadgen_v2.generation_critic import score_generation


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
