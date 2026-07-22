from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest
from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.phase2 import run_phase2_proposals
from iadgen_v2.phase3 import build_phase3_adaptation_cache, train_phase3_adapter
from iadgen_v2.phase3 import GatedProjectionAdapter
import iadgen_v2.phase4 as phase4_mod
from iadgen_v2.phase4 import _qwen_sd15_render, _qwen_sd15_text_render, run_phase4_generation
from iadgen_v2.phase4 import (
    _adapter_token_noise_std,
    _critic_guided_attempt_settings,
    _critic_guided_final_coverage_repair,
    _critic_guided_settings,
    _critic_guided_visibility_boost,
    _final_coverage_repair_profile,
    _phase4_generation_critic,
    _generation_seed,
    _morphology_for_row,
    _phase4_variants,
    _quality_diagnostics,
    _quality_profiles_for_row,
    _qwen_ip_adapter_hybrid_render,
    _qwen_latent_blend_harmonized_render,
)


TARGETS = {"metal_nut": "scratch", "tile": "crack", "wood": "scratch"}


def test_phase4_integrated_generation_uses_phase2_and_phase3_artifacts(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    run_phase2_proposals(config, "heuristic")
    build_phase3_adaptation_cache(config, "heuristic")
    train_phase3_adapter(config, "heuristic")

    metadata_path = run_phase4_generation(config, "heuristic")

    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    for record in records:
        assert Path(record["output_path"]).exists()
        assert record["conditioning_shape"] == [1, 81, 768]
        assert record["projected_tokens_shape"] == [1, 4, 768]
        assert record["latency_sec"] >= 0.0
        assert record["background_preservation_l1"] == 0.0
        assert record["mask_changed_pixel_fraction"] > 0.0
        assert record["settings"]["generator"] == "heuristic_adapter_contract"
        assert "not real SD1.5" in record["settings"]["note"]
    assert (config.report_dir / "phase4" / "heuristic" / "summary.md").exists()


def test_phase4_rejects_stale_phase3_checkpoint(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    run_phase2_proposals(config, "heuristic")
    build_phase3_adaptation_cache(config, "heuristic")
    train_phase3_adapter(config, "heuristic")

    config.data["phase3"]["adapter_hidden_dim"] = 20
    with pytest.raises(ValueError, match="Phase 3 checkpoint"):
        run_phase4_generation(config, "heuristic")


def test_phase4_rejects_feature_width_mismatch(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    metadata_path = run_phase2_proposals(config, "heuristic")
    build_phase3_adaptation_cache(config, "heuristic")
    train_phase3_adapter(config, "heuristic")

    record = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])
    feature_path = Path(record["feature_cache_path"])
    feature = torch.load(feature_path, map_location="cpu", weights_only=False)
    feature["tokens"] = feature["tokens"][..., :5]
    torch.save(feature, feature_path)

    with pytest.raises(ValueError, match="does not match adapter input width"):
        run_phase4_generation(config, "heuristic")


def test_phase4_qwen_provider_fails_until_real_inference_exists(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    with pytest.raises(FileNotFoundError, match="Missing Phase 2 metadata"):
        run_phase4_generation(config, "qwen")


def test_phase4_rejects_unknown_qwen_variant(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase4"]["provider"] = "qwen"

    with pytest.raises(ValueError, match="Unsupported Phase 4 variant"):
        run_phase4_generation(config, "qwen", "not_a_variant")


def test_phase4_all_qwen_variants_include_clone_harmonized() -> None:
    variants = _phase4_variants({"variants": "all"}, "qwen", None)
    assert "clone_harmonized" in variants
    assert "ip_adapter_hybrid" in variants
    assert "latent_blend_harmonized" in variants


def test_phase4_rejects_unknown_quality_profile() -> None:
    with pytest.raises(ValueError, match="Unsupported Phase 4 quality_profile"):
        _quality_profiles_for_row({"quality_profiles": ["not_a_profile"]}, {"defect_type": "scratch", "prompt": "scratch"})


def test_phase4_morphology_default_quality_profile() -> None:
    scuff = _quality_profiles_for_row({"quality_profiles": "auto"}, {"defect_type": "scratch", "prompt": "many rubbed scuffed scratches"})
    single = _quality_profiles_for_row({"quality_profiles": "auto"}, {"defect_type": "scratch", "prompt": "one long narrow scratch"})
    band = _quality_profiles_for_row({"quality_profiles": "auto"}, {"defect_type": "scratch", "prompt": "diagonal scratch"})

    assert scuff == ["scuff_soft_low_strength"]
    assert single == ["single_stroke_clean"]
    assert band == ["scratch_ridge_balanced"]


def test_phase4_quality_profile_filter_keeps_compatible_profiles() -> None:
    profiles = _quality_profiles_for_row(
        {
            "quality_profiles": ["scuff_soft_low_strength", "scratch_ridge_balanced", "scratch_thin_detail"],
            "quality_profile_filter_by_morphology": True,
        },
        {"defect_type": "scratch", "prompt": "diagonal scratch"},
    )

    assert profiles == ["scratch_ridge_balanced", "scratch_thin_detail"]


def test_phase4_qwen_mask_only_conditioning_shape() -> None:
    pipe = _FakePipe()
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    mask = Image.new("L", image.size, 255)
    config = _minimal_render_config()

    _, combined, projected = _qwen_sd15_text_render(
        config,
        {"image_size": 16, "num_inference_steps": 1, "strength": 0.33, "guidance_scale": 2.5},
        pipe,
        "scratch",
        {"seed": 1},
        image,
        mask,
        "cpu",
    )

    assert list(combined.shape) == [1, 77, 768]
    assert list(projected.shape) == [1, 0, 768]
    assert pipe.last_kwargs["strength"] == 0.33
    assert pipe.last_kwargs["guidance_scale"] == 2.5


def test_phase4_full_hybrid_conditioning_shape() -> None:
    pipe = _FakePipe()
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    mask = Image.new("L", image.size, 255)
    config = _minimal_render_config()
    adapter = GatedProjectionAdapter(input_dim=6, output_dim=768, hidden_dim=8)
    tokens = torch.randn(1, 16, 6)

    _, combined, projected = _qwen_sd15_render(
        config,
        {"image_size": 16, "num_inference_steps": 1, "strength": 0.33, "guidance_scale": 2.5},
        pipe,
        adapter,
        {"clip_token_count": 77, "output_dim": 768},
        {"seed": 1},
        "scratch",
        tokens,
        image,
        mask,
        "cpu",
    )

    assert list(combined.shape) == [1, 93, 768]
    assert list(projected.shape) == [1, 16, 768]


def test_phase4_adapter_token_noise_keeps_conditioning_shape() -> None:
    pipe = _FakePipe()
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    mask = Image.new("L", image.size, 255)
    config = _minimal_render_config()
    adapter = GatedProjectionAdapter(input_dim=6, output_dim=768, hidden_dim=8)
    tokens = torch.randn(1, 16, 6)
    phase4 = {
        "image_size": 16,
        "num_inference_steps": 1,
        "strength": 0.33,
        "guidance_scale": 2.5,
        "diversity": {"enabled": True, "adapter_token_noise_std": [0.01]},
    }

    _, combined, projected = _qwen_sd15_render(
        config,
        phase4,
        pipe,
        adapter,
        {"clip_token_count": 77, "output_dim": 768},
        {"seed": 1},
        "scratch",
        tokens,
        image,
        mask,
        "cpu",
        123,
    )

    assert _adapter_token_noise_std(phase4, 123) == 0.01
    assert list(combined.shape) == [1, 93, 768]
    assert list(projected.shape) == [1, 16, 768]


def test_phase4_ip_adapter_hybrid_uses_visual_reference(tmp_path: Path) -> None:
    pipe = _FakePipe()
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((5, 5, 10, 10), fill=255)
    source_image, source_mask = _write_source_pair(tmp_path)
    config = _minimal_render_config()
    row = {"category": "wood", "defect_type": "scratch", "seed": 3}
    source_index = {("wood", "scratch"): [{"image_path": str(source_image), "mask_path": str(source_mask)}]}

    _, combined, projected, metadata = _qwen_ip_adapter_hybrid_render(
        config,
        {
            "image_size": 16,
            "ip_adapter": {
                "enabled": True,
                "model_id": "local/ip-adapter",
                "weight_name": "ip-adapter_sd15.bin",
                "local_files_only": True,
            },
        },
        pipe,
        "scratch",
        row,
        source_index,
        image,
        mask,
        mask,
        "cpu",
        3,
    )

    assert list(combined.shape) == [1, 77, 768]
    assert list(projected.shape) == [1, 0, 768]
    assert metadata["status"] == "ip_adapter_loaded"
    assert pipe.ip_adapter_loaded
    assert "ip_adapter_image" in pipe.last_kwargs


def test_phase4_latent_blend_harmonized_conditioning_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _FakePipe()
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((5, 5, 10, 10), fill=255)
    source_image, source_mask = _write_source_pair(tmp_path)
    config = _minimal_render_config()
    row = {"category": "wood", "defect_type": "scratch", "seed": 3}
    source_index = {("wood", "scratch"): [{"image_path": str(source_image), "mask_path": str(source_mask)}]}

    monkeypatch.setattr(phase4_mod, "_normal_clone_to_target", lambda **kwargs: kwargs["target_image"])
    _, combined, projected, metadata = _qwen_latent_blend_harmonized_render(
        config,
        {"image_size": 16, "latent_blend_strength": 0.8},
        pipe,
        "scratch",
        row,
        source_index,
        image,
        mask,
        mask,
        "cpu",
        3,
    )

    assert list(combined.shape) == [1, 77, 768]
    assert list(projected.shape) == [1, 0, 768]
    assert metadata["status"] == "pixel_alpha_blend_fallback_no_vae"


def test_phase4_seed_reuse_is_variant_independent() -> None:
    row = {"seed": 123}
    assert _generation_seed(row, 0) == 123
    assert _generation_seed(row, 1) == 1_000_126


def test_phase4_label_policy_selects_scuff_soft_profile() -> None:
    row = {
        "category": "wood",
        "defect_type": "scratch",
        "settings": {
            "label_policy": {
                "label_policy": "soft_mask_only",
                "quality_morphology": "multi_scuff",
            }
        },
    }

    assert _morphology_for_row(row) == "multi_scuff"
    assert _quality_profiles_for_row({"quality_profile": "auto"}, row) == ["scuff_soft_low_strength"]


def test_phase4_quality_diagnostics_flags_weak_and_broad_masks() -> None:
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    output = Image.new("RGB", (16, 16), (82, 82, 82))
    refined = Image.new("L", (16, 16), 0)
    ImageDraw.Draw(refined).rectangle((7, 7, 8, 8), fill=255)
    inpaint = Image.new("L", (16, 16), 255)

    diagnostics = _quality_diagnostics(
        image,
        output,
        refined,
        inpaint,
        {
            "min_defect_visibility_score": 0.02,
            "max_background_l1": 0.05,
            "max_outside_refined_change_fraction": 0.01,
            "max_inpaint_area_fraction": 0.5,
            "min_mask_changed_fraction": 0.3,
        },
        background_preservation_l1=0.0,
        mask_changed_fraction=0.0,
    )

    assert "weak_visible_defect" in diagnostics["quality_flags"]
    assert "overbroad_inpaint_mask" in diagnostics["quality_flags"]
    assert "low_mask_edit_fraction" in diagnostics["quality_flags"]


def test_phase4_critic_guided_attempt_boosts_strength_prompt_and_seed() -> None:
    phase4 = {
        "strength": 0.55,
        "guidance_scale": 7.2,
        "num_inference_steps": 18,
        "clone_harmonize_strength": 0.20,
        "critic_guided_regeneration": {
            "enabled": True,
            "max_attempts": 3,
            "strength_step": 0.08,
            "guidance_step": 0.35,
            "step_increment": 2,
            "retry_seed_stride": 100,
            "prompt_suffix": "force visible defect",
        },
    }

    settings = _critic_guided_settings(phase4, "qwen")
    attempt, prompt, seed = _critic_guided_attempt_settings(
        phase4,
        "scratch",
        {"defect_type": "scratch"},
        11,
        2,
        settings,
        previous_attempt={"reject_reasons": ["high_leakage"]},
    )

    assert seed == 211
    assert prompt.endswith("force visible defect")
    assert attempt["strength"] == pytest.approx(0.71)
    assert attempt["clone_harmonize_strength"] == pytest.approx(0.36)
    assert attempt["guidance_scale"] == pytest.approx(7.9)
    assert attempt["num_inference_steps"] == 22


def test_phase4_low_coverage_retry_gets_reason_aware_boosts() -> None:
    phase4 = {
        "strength": 0.55,
        "guidance_scale": 7.2,
        "num_inference_steps": 18,
        "critic_guided_regeneration": {
            "enabled": True,
            "max_attempts": 3,
            "strength_step": 0.08,
            "guidance_step": 0.35,
            "step_increment": 2,
            "retry_seed_stride": 100,
            "low_coverage_extra_strength": 0.10,
            "low_coverage_extra_guidance": 0.45,
            "low_coverage_extra_steps": 2,
            "local_mask_noise_step": 0.04,
            "postprocess_mask_contrast_step": 0.09,
            "prompt_suffix": "force visible defect",
            "low_coverage_prompt_suffix": "fill the selected mask",
        },
    }

    settings = _critic_guided_settings(phase4, "qwen")
    attempt, prompt, seed = _critic_guided_attempt_settings(
        phase4,
        "scratch",
        {"defect_type": "scratch", "settings": {"quality_morphology": "scratch_band"}},
        11,
        1,
        settings,
        previous_attempt={"reject_reasons": ["low_mask_coverage"]},
    )

    assert seed == 111
    assert "force visible defect" in prompt
    assert "fill the selected mask" in prompt
    assert attempt["critic_retry_reason"] == "low_mask_coverage"
    assert attempt["strength"] == pytest.approx(0.73)
    assert attempt["guidance_scale"] == pytest.approx(8.0)
    assert attempt["num_inference_steps"] == 22
    assert attempt["local_mask_noise_boost"] == pytest.approx(0.04)
    assert attempt["postprocess_mask_contrast_boost"] == pytest.approx(0.09)


def test_phase4_generation_critic_rejects_underedited_and_accepts_visible_mask() -> None:
    image = Image.new("RGB", (16, 16), (80, 80, 80))
    refined = Image.new("L", (16, 16), 0)
    ImageDraw.Draw(refined).rectangle((4, 4, 11, 11), fill=255)
    settings = _critic_guided_settings(
        {
            "critic_guided_regeneration": {
                "enabled": True,
                "min_adaptive_mask_coverage_score": 0.35,
                "min_leakage_score": 0.45,
            }
        },
        "qwen",
    )

    weak = _phase4_generation_critic(image, image.copy(), refined, refined, settings)
    assert not weak["accepted"]
    assert "low_mask_coverage" in weak["reject_reasons"]

    strong = image.copy()
    strong.paste(Image.new("RGB", image.size, (160, 40, 40)), mask=refined)
    accepted = _phase4_generation_critic(image, strong, refined, refined, settings)
    assert accepted["accepted"]
    assert accepted["adaptive_mask_coverage_score"] >= 0.35
    assert accepted["leakage_score"] >= 0.45


def test_phase4_visibility_boost_increases_mask_coverage() -> None:
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    output = image.copy()
    refined = Image.new("L", image.size, 0)
    ImageDraw.Draw(refined).line((8, 16, 24, 16), fill=255, width=3)
    settings = _critic_guided_settings(
        {
            "critic_guided_regeneration": {
                "enabled": True,
                "min_adaptive_mask_coverage_score": 0.35,
                "min_leakage_score": 0.45,
                "min_texture_preservation_score": 0.0,
                "min_morphology_fit_score": 0.0,
            }
        },
        "qwen",
    )

    before = _phase4_generation_critic(image, output, refined, refined, settings, morphology="scratch_band")
    boosted, metadata = _critic_guided_visibility_boost(
        image,
        output,
        refined,
        refined,
        {"postprocess_mask_contrast_boost": 0.35},
        123,
        morphology="scratch_band",
    )
    after = _phase4_generation_critic(image, boosted, refined, refined, settings, morphology="scratch_band")

    assert metadata["enabled"]
    assert after["adaptive_mask_coverage_score"] > before["adaptive_mask_coverage_score"]


def test_phase4_final_coverage_repair_can_accept_underedited_output() -> None:
    image = Image.new("RGB", (32, 32), (120, 120, 120))
    output = image.copy()
    refined = Image.new("L", image.size, 0)
    ImageDraw.Draw(refined).rectangle((9, 9, 22, 22), fill=255)
    settings = _critic_guided_settings(
        {
            "critic_guided_regeneration": {
                "enabled": True,
                "min_adaptive_mask_coverage_score": 0.35,
                "min_leakage_score": 0.45,
                "min_texture_preservation_score": 0.0,
                "min_morphology_fit_score": 0.0,
                "final_coverage_repair_enabled": True,
                "final_coverage_repair_max_attempts": 3,
                "final_coverage_repair_step": 0.25,
                "final_coverage_repair_max_boost": 0.75,
            }
        },
        "qwen",
    )
    base = _phase4_generation_critic(image, output, refined, refined, settings, morphology="scratch_band")
    base_record = {
        **base,
        "attempt_index": 0,
        "generation_seed": 7,
        "strength": 0.55,
        "guidance_scale": 7.5,
        "num_inference_steps": 20,
        "prompt": "scratch",
    }

    repaired, repaired_record, metadata = _critic_guided_final_coverage_repair(
        image,
        output,
        refined,
        refined,
        settings,
        7,
        base_record,
        morphology="scratch_band",
    )

    assert metadata["enabled"]
    assert metadata["attempts"]
    assert repaired_record["adaptive_mask_coverage_score"] > base_record["adaptive_mask_coverage_score"]
    assert repaired_record["accepted"]
    assert repaired.tobytes() != output.tobytes()


def test_phase4_final_coverage_repair_profiles_are_morphology_specific_and_configurable() -> None:
    settings = _critic_guided_settings(
        {
            "critic_guided_regeneration": {
                "enabled": True,
                "final_coverage_repair_profiles": {
                    "scratch_band": {"step": 0.21, "max_boost": 0.61},
                    "default": {"blend_cap": 0.33},
                },
            }
        },
        "qwen",
    )

    scratch = _final_coverage_repair_profile(settings, "scratch_band")
    scuff = _final_coverage_repair_profile(settings, "multi_scuff")
    chip = _final_coverage_repair_profile(settings, "micro_chip")
    unknown = _final_coverage_repair_profile(settings, "not_a_known_morphology")

    assert scratch["mode"] == "dark_line"
    assert scratch["step"] == pytest.approx(0.21)
    assert scratch["max_boost"] == pytest.approx(0.61)
    assert scuff["mode"] == "contrast_scuff"
    assert scuff["max_boost"] < chip["max_boost"]
    assert chip["mode"] == "bright_chip"
    assert unknown["blend_cap"] == pytest.approx(0.33)


class _FakePipe:
    def __init__(self):
        self.last_kwargs = {}
        self.ip_adapter_loaded = False

    def encode_prompt(self, **kwargs):
        return torch.zeros(1, 77, 768), torch.zeros(1, 77, 768)

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        return type("Result", (), {"images": [kwargs["image"]]})()

    def load_ip_adapter(self, *args, **kwargs):
        self.ip_adapter_loaded = True
        self.ip_adapter_args = args
        self.ip_adapter_kwargs = kwargs

    def set_ip_adapter_scale(self, scale):
        self.ip_adapter_scale = scale


def _minimal_render_config():
    config = type("Config", (), {})()
    config.data = {
        "generation": {"negative_prompt": ""},
        "models": {"sd15": {"guidance_scale": 1.0, "num_inference_steps": 1, "strength": 0.5}},
    }
    return config


def _write_source_pair(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "source.png"
    mask_path = tmp_path / "source_mask.png"
    image = Image.new("RGB", (16, 16), (90, 80, 70))
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).line((4, 8, 12, 8), fill=255, width=2)
    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def _fixture_config(tmp_path: Path):
    dataset_root = tmp_path / "data" / "mvtec_ad"
    for category, defect_type in TARGETS.items():
        clean_dir = dataset_root / category / "train" / "good"
        test_dir = dataset_root / category / "test" / defect_type
        mask_dir = dataset_root / category / "ground_truth" / defect_type
        clean_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        clean = Image.new("RGB", (48, 48), (15, 15, 15) if category == "metal_nut" else (80, 80, 80))
        if category == "metal_nut":
            ImageDraw.Draw(clean).ellipse((8, 8, 40, 40), fill=(150, 150, 150))
        clean.save(clean_dir / "000.png")
        for index in range(4):
            image = clean.copy()
            mask = Image.new("L", (48, 48), 0)
            draw = ImageDraw.Draw(mask)
            draw.line((8, 18 + index, 38, 20 + index), fill=255, width=2)
            image.paste(Image.new("RGB", image.size, (180, 40, 40)), mask=mask)
            image.save(test_dir / f"{index:03d}.png")
            mask.save(mask_dir / f"{index:03d}_mask.png")
    config_path = tmp_path / "phase4.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {dataset_root}",
                "  download: false",
                "  seed: 11",
                "  adaptation_per_defect: 2",
                "  targets:",
                "    metal_nut: [scratch]",
                "    tile: [crack]",
                "    wood: [scratch]",
                "generation:",
                "  samples_per_category: 1",
                "  seed: 11",
                "  device: cpu",
                "  prompt_by_defect: {scratch: scratch, crack: crack}",
                "models:",
                "  mock: {enabled: true}",
                "  sd15:",
                "    enabled: false",
                "    base_model: example/base",
                "    controlnet_model: example/control",
                "evaluation: {}",
                "phase2:",
                "  provider: heuristic",
                "  qwen_model: Qwen/Qwen2.5-VL-3B-Instruct",
                "  samples_per_category: 1",
                "  token_count: 4",
                "  feature_dim: 10",
                "  box_area_fraction: {scratch: 0.06, crack: 0.06}",
                "  min_surface_coverage: 0.90",
                "phase3:",
                "  provider: heuristic",
                "  device: cpu",
                "  clip_token_count: 77",
                "  cross_attention_dim: 768",
                "  adapter_hidden_dim: 16",
                "  adapter_epochs: 2",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  initial_gate_logit: -2.0",
                "phase4:",
                "  provider: heuristic",
                "  device: cpu",
                "  max_records: 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
