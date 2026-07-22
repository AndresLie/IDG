from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.phase2 import run_phase2_proposals
from iadgen_v2.phase3 import build_phase3_adaptation_cache, train_phase3_adapter
from iadgen_v2.phase4 import run_phase4_generation
from iadgen_v2.phase4 import PHASE4_SCHEMA_VERSION, _phase4_fingerprint
from iadgen_v2.phase5 import (
    SegmentationSample,
    _clean_inpaint_negative_samples,
    _phase5_evaluators,
    _load_loss_weight,
    _load_tensor_pair,
    _mix_samples,
    _passes_phase5_quality_filter,
    _real_split_samples,
    _select_threshold,
    _soft_dice_loss,
    _selected_candidate_keys,
    _supervised_normal_negative_samples,
    _synthetic_selector_decision,
    _write_mask_policy_validation_report,
    _weighted_bce_with_logits,
    _weighted_focal_loss,
    _write_synthetic_selector_reports,
    run_phase5_evaluation,
)
import torch
from iadgen_v2.phase9_visual import write_phase9_visual_report


TARGETS = {"metal_nut": "scratch", "tile": "crack", "wood": "scratch"}


def test_phase5_runs_tiny_unet_ratios_on_held_out_real_samples(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["synthetic_ratio"] == "0.0"
    assert rows[1]["synthetic_ratio"] == "0.5"
    assert int(rows[0]["train_synthetic_count"]) == 0
    assert int(rows[1]["train_synthetic_count"]) > 0
    assert all(int(row["held_out_count"]) == 6 for row in rows)
    assert all(row["note"] == "tiny_unet_smoke" for row in rows)
    assert all("predicted_positive_rate" in row for row in rows)
    assert all("truth_positive_rate" in row for row in rows)
    assert all("threshold_policy" in row for row in rows)
    assert all("fixed_predicted_positive_rate" in row for row in rows)
    assert all("score_p95" in row for row in rows)
    assert all(Path(row["prediction_contact_sheet"]).exists() for row in rows)
    assert (config.report_dir / "phase5" / "heuristic" / "summary.md").exists()
    status = json.loads((config.output_dir / "phase5" / "heuristic" / "run_status.json").read_text(encoding="utf-8"))
    assert status["held_out_by_category"] == {"metal_nut": 2, "tile": 2, "wood": 2}


def test_phase5_rejects_stale_phase4_metadata(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _prepare_all_artifacts(config)

    config.data["phase4"]["max_records"] = 1
    with pytest.raises(ValueError, match="Phase 4 metadata"):
        run_phase5_evaluation(config, "heuristic")


def test_phase5_qwen_provider_fails_until_real_outputs_exist(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    with pytest.raises(FileNotFoundError, match="Missing Phase 4 metadata"):
        run_phase5_evaluation(config, "qwen")


def test_phase5_research_grade_patchcore_lite_and_audit_manifest(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["tiny_unet", "patchcore_lite"]
    config.data["phase5"]["patchcore_max_memory_patches"] = 256
    config.data["phase5"]["patchcore_distance_chunk_size"] = 64
    config.data["phase5"]["human_audit"] = {"enabled": True, "max_samples": 4}
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["evaluator"] for row in rows} == {"tiny_unet", "patchcore_lite"}
    assert any(row["evaluator"] == "patchcore_lite" and row["variant"] == "normal_only" for row in rows)
    assert all("morphology_metrics" in row for row in rows)
    audit_path = config.report_dir / "phase5" / "heuristic" / "human_audit_manifest.jsonl"
    assert audit_path.exists()
    assert len([line for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]) == 4
    summary = (config.report_dir / "phase5" / "heuristic" / "summary.md").read_text(encoding="utf-8")
    assert "patchcore_lite" in summary
    assert "Morphology Summary" in summary
    assert "Synthetic selector report" in summary
    assert (config.report_dir / "phase5" / "heuristic" / "synthetic_selector_report.md").exists()
    status = json.loads((config.output_dir / "phase5" / "heuristic" / "run_status.json").read_text(encoding="utf-8"))
    assert status["evaluators"] == ["tiny_unet", "patchcore_lite"]
    assert Path(status["human_audit_manifest"]).exists()
    assert Path(status["synthetic_selector_report"]).exists()


def test_phase9_visual_report_combines_generation_masks_and_predictions(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _prepare_all_artifacts(config)
    run_phase5_evaluation(config, "heuristic")

    contact_sheet = write_phase9_visual_report(config, "heuristic")

    assert contact_sheet.exists()
    assert (config.report_dir / "phase9_visual" / "heuristic" / "summary.md").exists()
    assert Image.open(contact_sheet).size[0] > 0


def test_phase5_patchcore_resnet_smoke_with_untrained_weights(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["patchcore_resnet"]
    config.data["phase5"]["repeated_seeds"] = [11]
    config.data["phase5"]["patchcore_resnet_weights"] = "none"
    config.data["phase5"]["patchcore_resnet_image_size"] = 64
    config.data["phase5"]["patchcore_resnet_max_memory_patches"] = 64
    config.data["phase5"]["patchcore_resnet_distance_chunk_size"] = 16
    prepare_splits(config, download=False)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["evaluator"] == "patchcore_resnet"
    assert rows[0]["variant"] == "normal_only"
    assert "patchcore_resnet" in rows[0]["note"]


def test_phase5_real_split_uses_soft_training_masks_and_eval_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "000.png"
    hard_mask_path = tmp_path / "000_mask.png"
    train_mask_path = tmp_path / "000_inpaint_soft.png"
    eval_mask_path = tmp_path / "000_eval_tight.png"
    uncertainty_mask_path = tmp_path / "000_uncertainty.png"
    Image.new("RGB", (8, 8), (80, 80, 80)).save(image_path)
    Image.new("L", (8, 8), 0).save(hard_mask_path)
    train_mask = Image.new("L", (8, 8), 0)
    train_mask.putpixel((3, 3), 128)
    train_mask.save(train_mask_path)
    eval_mask = Image.new("L", (8, 8), 0)
    eval_mask.putpixel((3, 3), 255)
    eval_mask.save(eval_mask_path)
    uncertainty = Image.new("L", (8, 8), 0)
    uncertainty.putpixel((3, 3), 255)
    uncertainty.save(uncertainty_mask_path)
    manifest = {
        "targets": {
            "custom_part/scratch": {
                "category": "custom_part",
                "defect_type": "scratch",
                "adaptation": [
                    {
                        "image_path": str(image_path),
                        "mask_path": str(hard_mask_path),
                        "training_mask_path": str(train_mask_path),
                        "eval_mask_path": str(eval_mask_path),
                        "uncertainty_mask_path": str(uncertainty_mask_path),
                        "label_policy": {"label_policy": "soft_mask_only"},
                    }
                ],
                "held_out": [
                    {
                        "image_path": str(image_path),
                        "mask_path": str(hard_mask_path),
                        "training_mask_path": str(train_mask_path),
                        "eval_mask_path": str(eval_mask_path),
                        "uncertainty_mask_path": str(uncertainty_mask_path),
                    }
                ],
            }
        }
    }

    train, held_out = _real_split_samples(manifest, held_out_per_category=1)
    _, train_mask_tensor = _load_tensor_pair(train[0], 8, "cpu")

    assert train[0].mask_path == str(train_mask_path)
    assert train[0].mask_mode == "soft"
    assert train[0].uncertainty_mask_path == str(uncertainty_mask_path)
    assert float(train_mask_tensor[0, 3, 3]) == pytest.approx(128 / 255, abs=1e-4)
    assert held_out[0].mask_path == str(eval_mask_path)
    assert held_out[0].mask_mode == "binary"


def test_phase5_uncertainty_loss_weight_downweights_ambiguous_pixels(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    uncertainty_path = tmp_path / "uncertainty.png"
    Image.new("RGB", (4, 4), (100, 100, 100)).save(image_path)
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    mask.save(mask_path)
    uncertainty = Image.new("L", (4, 4), 0)
    uncertainty.putpixel((1, 1), 255)
    uncertainty.save(uncertainty_path)
    sample = SegmentationSample(
        image_path=str(image_path),
        mask_path=str(mask_path),
        category="wood",
        defect_type="scratch",
        source="real_adaptation",
        uncertainty_mask_path=str(uncertainty_path),
    )

    weight = _load_loss_weight(sample, 4, "cpu", {"uncertainty_loss_weight": 0.25})

    assert float(weight[0, 1, 1]) == pytest.approx(0.25, abs=1e-4)
    assert float(weight[0, 0, 0]) == pytest.approx(1.0, abs=1e-4)

    logits = torch.zeros((1, 1, 4, 4))
    logits[0, 0, 1, 1] = -4.0
    target = torch.zeros((1, 1, 4, 4))
    target[0, 0, 1, 1] = 1.0
    full = torch.ones_like(target)
    down = weight.unsqueeze(0)
    pos_weight = torch.ones((1, 1, 1, 1))

    assert _weighted_bce_with_logits(logits, target, down, pos_weight) < _weighted_bce_with_logits(logits, target, full, pos_weight)
    assert torch.isfinite(_soft_dice_loss(logits, target, weight=down))


def test_phase5_positive_core_boosts_target_and_loss_weight(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "soft.png"
    core_path = tmp_path / "core.png"
    uncertainty_path = tmp_path / "uncertainty.png"
    Image.new("RGB", (4, 4), (100, 100, 100)).save(image_path)
    Image.new("L", (4, 4), 0).save(mask_path)
    core = Image.new("L", (4, 4), 0)
    core.putpixel((2, 2), 255)
    core.save(core_path)
    Image.new("L", (4, 4), 0).save(uncertainty_path)
    sample = SegmentationSample(
        image_path=str(image_path),
        mask_path=str(mask_path),
        category="wood",
        defect_type="scratch",
        source="real_adaptation",
        mask_mode="soft",
        uncertainty_mask_path=str(uncertainty_path),
        positive_core_path=str(core_path),
    )

    _, target = _load_tensor_pair(sample, 4, "cpu")
    weight = _load_loss_weight(sample, 4, "cpu", {"uncertainty_loss_weight": 1.0, "positive_core_loss_weight": 3.0})

    assert float(target[0, 2, 2]) == pytest.approx(1.0)
    assert float(weight[0, 2, 2]) == pytest.approx(3.0)


def test_phase5_supervised_normal_negatives_load_zero_masks(tmp_path: Path) -> None:
    image_path = tmp_path / "normal.png"
    Image.new("RGB", (8, 8), (90, 90, 90)).save(image_path)
    normal = [
        SegmentationSample(
            image_path=str(image_path),
            mask_path="",
            category="wood",
            defect_type="scratch",
            source="normal_train_good",
        )
    ]

    selected = _supervised_normal_negative_samples(normal, {"supervised_normal_negatives": {"enabled": True, "max_samples": 1}}, 7)
    _, mask = _load_tensor_pair(selected[0], 8, "cpu")

    assert selected[0].source == "supervised_normal_negative"
    assert float(mask.sum()) == 0.0


def test_phase5_constrained_threshold_avoids_all_positive_choice() -> None:
    truth = np.zeros((10, 10), dtype=np.float32)
    truth[0, 0] = 1.0
    score = np.full((10, 10), 0.5, dtype=np.float32)
    score[0, 0] = 0.6
    rows = [{"score": score, "truth": truth}]

    constrained = _select_threshold(
        rows,
        [0.05, 0.55],
        {"min_predicted_positive_rate": 0.001, "max_predicted_positive_rate": 0.2, "target_predicted_positive_rate": 0.01},
    )

    assert constrained["selected_threshold"] > 0.5
    assert constrained["predicted_positive_rate"] <= 0.2
    assert constrained["threshold_constraint_satisfied"] == 1.0


def test_phase5_weighted_focal_loss_is_finite() -> None:
    logits = torch.zeros((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    target[0, 0, 1, 1] = 1.0
    weight = torch.ones_like(target)

    loss = _weighted_focal_loss(logits, target, weight, gamma=2.0)

    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_phase5_evaluator_alias_and_unknown_rejection() -> None:
    assert _phase5_evaluators({"evaluators": ["tiny_unet", "patchcore"]}) == ["tiny_unet", "patchcore_resnet"]
    assert _phase5_evaluators({"evaluators": ["patchcore_lite", "patchcore_full"]}) == [
        "patchcore_lite",
        "patchcore_resnet",
    ]
    with pytest.raises(ValueError, match="Unsupported Phase 5 evaluator"):
        _phase5_evaluators({"evaluators": ["tiny_unet", "mystery"]})


def test_phase5_qwen_grouped_comparison_csv_includes_expected_variants(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    variants = ["qwen_mask_only", "full_qwen_hybrid", "fixed_mask_adapter"]
    config.data["phase4"]["provider"] = "qwen"
    config.data["phase4"]["variants"] = variants
    config.data["phase5"]["provider"] = "qwen"
    config.data["phase5"]["variants"] = variants
    expected = _phase4_fingerprint(config, "qwen")
    mask_path = _first_adaptation_mask(config)
    image_path = _first_clean_image(config)
    for variant in variants:
        output_path = config.output_dir / "phase4" / "qwen" / variant / f"{variant}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(image_path).save(output_path)
        metadata_path = config.output_dir / "phase4" / "qwen" / variant / "metadata.jsonl"
        metadata_path.write_text(
            json.dumps(
                {
                    "category": "metal_nut",
                    "defect_type": "scratch",
                    "provider": "qwen",
                    "variant": variant,
                    "background_path": str(image_path),
                    "prompt": "scratch",
                    "refined_mask_path": str(mask_path),
                    "inpaint_mask_path": str(mask_path),
                    "phase2_feature_cache_path": "unused.pt",
                    "adapter_checkpoint_path": None,
                    "output_path": str(output_path),
                    "conditioning_shape": [1, 77, 768] if variant == "qwen_mask_only" else [1, 93, 768],
                    "projected_tokens_shape": [1, 0, 768] if variant == "qwen_mask_only" else [1, 16, 768],
                    "gate_value": 0.0,
                    "latency_sec": 0.0,
                    "peak_cuda_memory_bytes": None,
                    "max_rss_kb": 0,
                    "background_preservation_l1": 0.0,
                    "mask_changed_pixel_fraction": 1.0,
                    "settings": {"phase4_schema_version": PHASE4_SCHEMA_VERSION, "phase4_fingerprint": expected},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    csv_path = run_phase5_evaluation(config, "qwen")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rows} == {"real_only", *variants}
    assert any(row["variant"] == "real_only" and row["synthetic_ratio"] == "0.0" for row in rows)
    for variant in variants:
        assert any(row["variant"] == variant and row["synthetic_ratio"] == "0.5" for row in rows)
    assert all("quality_profile" in row for row in rows)


def test_phase5_quality_profile_grouped_csv_filters_synthetic_rows(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    variants = ["qwen_mask_only", "full_qwen_hybrid"]
    profile = "scratch_ridge_balanced"
    config.data["phase4"]["provider"] = "qwen"
    config.data["phase4"]["variants"] = variants
    config.data["phase4"]["quality_profiles"] = [profile]
    config.data["phase5"]["provider"] = "qwen"
    config.data["phase5"]["variants"] = variants
    config.data["phase5"]["group_by_quality_profile"] = True
    config.data["phase5"]["quality_profiles"] = [profile]
    expected = _phase4_fingerprint(config, "qwen")
    mask_path = _first_adaptation_mask(config)
    image_path = _first_clean_image(config)
    for variant in variants:
        output_path = config.output_dir / "phase4" / "qwen" / variant / f"{variant}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(image_path).save(output_path)
        metadata_path = config.output_dir / "phase4" / "qwen" / variant / "metadata.jsonl"
        metadata_path.write_text(
            json.dumps(
                {
                    "category": "metal_nut",
                    "defect_type": "scratch",
                    "provider": "qwen",
                    "variant": variant,
                    "quality_profile": profile,
                    "background_path": str(image_path),
                    "prompt": "scratch",
                    "refined_mask_path": str(mask_path),
                    "inpaint_mask_path": str(mask_path),
                    "phase2_feature_cache_path": "unused.pt",
                    "adapter_checkpoint_path": None,
                    "output_path": str(output_path),
                    "conditioning_shape": [1, 77, 768] if variant == "qwen_mask_only" else [1, 93, 768],
                    "projected_tokens_shape": [1, 0, 768] if variant == "qwen_mask_only" else [1, 16, 768],
                    "gate_value": 0.0,
                    "latency_sec": 0.0,
                    "peak_cuda_memory_bytes": None,
                    "max_rss_kb": 0,
                    "background_preservation_l1": 0.0,
                    "mask_changed_pixel_fraction": 1.0,
                    "settings": {"phase4_schema_version": PHASE4_SCHEMA_VERSION, "phase4_fingerprint": expected},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    csv_path = run_phase5_evaluation(config, "qwen")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in rows} == {"real_only", f"qwen_mask_only:{profile}", f"full_qwen_hybrid:{profile}"}
    assert all(row["quality_profile"] in {"none", profile} for row in rows)


def test_phase5_class_balanced_mix_preserves_categories() -> None:
    real = [
        SegmentationSample("r1.png", "m1.png", "wood", "scratch", "real_adaptation"),
        SegmentationSample("r2.png", "m2.png", "tile", "crack", "real_adaptation"),
    ]
    synthetic = [
        SegmentationSample("s1.png", "m1.png", "wood", "scratch", "qwen"),
        SegmentationSample("s2.png", "m2.png", "tile", "crack", "qwen"),
    ]

    mixed = _mix_samples(real, synthetic, 0.5, 7, class_balanced=True)
    synthetic_keys = {(sample.category, sample.defect_type) for sample in mixed if sample.source != "real_adaptation"}

    assert synthetic_keys == {("wood", "scratch"), ("tile", "crack")}


def test_phase5_clean_inpaint_negatives_are_zero_mask_samples(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    manifest = json.loads((config.output_dir / "prepared" / "split_manifest.json").read_text(encoding="utf-8"))
    phase5 = {
        "image_size": 32,
        "sd_clean_inpaint_negatives": {"enabled": True, "samples_per_clean": 1},
    }

    samples = _clean_inpaint_negative_samples(config, manifest, phase5, "heuristic", 11)

    assert samples
    assert all(sample.source == "sd_clean_inpaint_negative" for sample in samples)
    assert all(Path(sample.image_path).exists() for sample in samples)
    for sample in samples:
        mask = Image.open(sample.mask_path).convert("L")
        assert mask.getbbox() is None


def test_phase5_quality_filter_rejects_flagged_or_weak_rows() -> None:
    phase5 = {"quality_filtered_training": {"enabled": True, "min_defect_visibility_score": 0.02}}
    good = {
        "quality_flags": [],
        "defect_visibility_score": 0.03,
        "background_preservation_l1": 0.01,
        "outside_refined_change_fraction": 0.02,
    }
    flagged = {**good, "quality_flags": ["weak_visible_defect"]}
    weak = {**good, "defect_visibility_score": 0.001}

    assert _passes_phase5_quality_filter(good, phase5)
    assert not _passes_phase5_quality_filter(flagged, phase5)
    assert not _passes_phase5_quality_filter(weak, phase5)


def test_phase5_synthetic_selector_scores_rejects_and_caps_groups(tmp_path: Path) -> None:
    phase5 = {
        "quality_filtered_training": {"enabled": True, "min_defect_visibility_score": 0.02},
        "synthetic_selector": {
            "enabled": True,
            "max_per_group": 1,
            "group_by": ["category", "defect_type", "morphology"],
            "preferred_profiles": {"scratch_band": ["scratch_thin_detail"]},
            "tfidg_lite": {
                "enabled": True,
                "min_adaptive_mask_coverage_score": 0.35,
                "selector_score_weight": 0.35,
            },
        },
    }
    good = {
        "category": "wood",
        "defect_type": "scratch",
        "variant": "fixed_mask_adapter",
        "quality_profile": "scratch_thin_detail",
        "output_path": "a.png",
        "generation_quality_score": 0.2,
        "defect_visibility_score": 0.05,
        "background_preservation_l1": 0.01,
        "outside_refined_change_fraction": 0.01,
        "inpaint_mask_area_fraction": 0.03,
        "quality_flags": [],
        "settings": {"quality_morphology": "scratch_band"},
        "tfidg_lite": {
            "tfidg_lite_score": 0.8,
            "adaptive_mask_coverage_score": 0.6,
            "feature_alignment_score": 0.9,
            "texture_preservation_score": 0.9,
            "leakage_score": 0.9,
        },
    }
    better = {**good, "output_path": "b.png", "generation_quality_score": 0.3}
    flagged = {**good, "output_path": "c.png", "quality_flags": ["weak_visible_defect"]}
    low_coverage = {
        **good,
        "output_path": "d.png",
        "tfidg_lite": {**good["tfidg_lite"], "adaptive_mask_coverage_score": 0.1},
    }

    assert _synthetic_selector_decision(good, phase5)["accepted"]
    assert not _synthetic_selector_decision(flagged, phase5)["accepted"]
    low_coverage_decision = _synthetic_selector_decision(low_coverage, phase5)
    assert not low_coverage_decision["accepted"]
    assert "low_tfidg_mask_coverage" in low_coverage_decision["reject_reasons"]

    candidates = [(row, _synthetic_selector_decision(row, phase5)) for row in [good, better, flagged, low_coverage]]
    selected = _selected_candidate_keys(candidates, phase5)

    assert selected == {"b.png"}

    logs = [
        {"accepted": True, "reject_reasons": [], "morphology": "scratch_band", "quality_profile": "scratch_thin_detail", "selector_score": 0.3, "variant": "fixed_mask_adapter", "output_path": "b.png"},
        {"accepted": False, "reject_reasons": ["quality_flags"], "morphology": "scratch_band", "quality_profile": "scratch_thin_detail", "selector_score": 0.0, "variant": "fixed_mask_adapter", "output_path": "c.png"},
    ]
    report = _write_synthetic_selector_reports(tmp_path, logs, phase5)
    text = report.read_text(encoding="utf-8")
    assert "Accepted rows: `1`" in text
    assert "`quality_flags`: `1`" in text


def test_phase5_mask_policy_validation_reports_expected_training_roles(tmp_path: Path) -> None:
    soft_train = tmp_path / "000_training_soft.png"
    hard_train = tmp_path / "001_training_medium.png"
    eval_mask = tmp_path / "001_eval_tight.png"
    for path in (soft_train, hard_train, eval_mask):
        Image.new("L", (8, 8), 0).save(path)
    manifest = {
        "targets": {
            "wood/scratch": {
                "category": "wood",
                "defect_type": "scratch",
                "adaptation": [
                    {
                        "image_path": str(tmp_path / "000.png"),
                        "mask_path": str(tmp_path / "000_mask.png"),
                        "training_mask_path": str(soft_train),
                        "eval_mask_path": str(eval_mask),
                        "label_policy": {"label_policy": "soft_mask_only", "quality_morphology": "multi_scuff"},
                    },
                    {
                        "image_path": str(tmp_path / "001.png"),
                        "mask_path": str(tmp_path / "001_mask.png"),
                        "training_mask_path": str(hard_train),
                        "eval_mask_path": str(eval_mask),
                        "label_policy": {"label_policy": "hard_mask_ok", "quality_morphology": "scratch_band"},
                    },
                ],
                "held_out": [
                    {
                        "image_path": str(tmp_path / "002.png"),
                        "mask_path": str(tmp_path / "002_mask.png"),
                        "training_mask_path": str(hard_train),
                        "eval_mask_path": str(eval_mask),
                        "label_policy": {"label_policy": "hard_mask_ok", "quality_morphology": "scratch_band"},
                    }
                ],
            }
        }
    }

    report = _write_mask_policy_validation_report(tmp_path, manifest, "qwen")

    text = report.read_text(encoding="utf-8")
    rows = list(csv.DictReader((tmp_path / "mask_policy_validation.csv").open()))
    assert "Mask Policy Validation" in text
    assert all(row["status"] == "pass" for row in rows)
    assert any(row["policy"] == "soft_mask_only" and row["observed_training_role"] == "training_soft" for row in rows)
    assert any(row["policy"] == "hard_mask_ok" and row["observed_training_role"] == "training_medium" for row in rows)


def _prepare_all_artifacts(config) -> None:
    prepare_splits(config, download=False)
    run_phase2_proposals(config, "heuristic")
    build_phase3_adaptation_cache(config, "heuristic")
    train_phase3_adapter(config, "heuristic")
    run_phase4_generation(config, "heuristic")


def _first_adaptation_mask(config) -> Path:
    manifest = json.loads((config.output_dir / "prepared" / "split_manifest.json").read_text(encoding="utf-8"))
    first_target = next(iter(manifest["targets"].values()))
    return Path(first_target["adaptation"][0]["mask_path"])


def _first_clean_image(config) -> Path:
    manifest = json.loads((config.output_dir / "prepared" / "split_manifest.json").read_text(encoding="utf-8"))
    first_target = next(iter(manifest["targets"].values()))
    return Path(first_target["clean_targets"][0])


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
    config_path = tmp_path / "phase5.yaml"
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
                "  adapter_epochs: 1",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  initial_gate_logit: -2.0",
                "phase4:",
                "  provider: heuristic",
                "  device: cpu",
                "  max_records: 3",
                "phase5:",
                "  provider: heuristic",
                "  device: cpu",
                "  image_size: 32",
                "  base_channels: 4",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  threshold: 0.5",
                "  held_out_per_category: 2",
                "  synthetic_ratios: [0.0, 0.5]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
