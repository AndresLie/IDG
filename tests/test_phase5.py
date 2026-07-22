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
    _calibrate_fused_score_rows,
    _clean_inpaint_negative_samples,
    _outside_possible_region_loss,
    _fuse_score_rows,
    _phase5_evaluators,
    _load_loss_weight,
    _load_tensor_pair,
    _mix_samples,
    _passes_phase5_quality_filter,
    _real_split_samples,
    _select_threshold,
    _soft_boundary_loss,
    _soft_dice_loss,
    _selected_candidate_keys,
    _supervised_normal_negative_samples,
    _synthetic_selector_decision,
    _teacher_distillation_tensors_for_sample,
    _teacher_refined_target_and_weight,
    _variant_ratio_arbitration,
    _write_mask_policy_validation_report,
    _weighted_bce_with_logits,
    _weighted_area_prior_loss,
    _weighted_focal_loss,
    _weighted_teacher_mse_loss,
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


def test_phase5_patchcore_guided_tiny_unet_smoke(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["patchcore_guided_tiny_unet"]
    config.data["phase5"]["epochs"] = 1
    config.data["phase5"]["patchcore_max_memory_patches"] = 128
    config.data["phase5"]["patchcore_distance_chunk_size"] = 32
    config.data["phase5"]["patchcore_guided_prior_weight"] = 0.4
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["evaluator"] for row in rows} == {"patchcore_guided_tiny_unet"}
    assert all(row["note"] == "tiny_unet_patchcore_lite_score_fusion" for row in rows)
    assert all(float(row["patchcore_guided_prior_weight"]) == pytest.approx(0.4) for row in rows)
    assert all(Path(row["prediction_contact_sheet"]).exists() for row in rows)


def test_phase5_patchcore_distilled_tiny_unet_smoke(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["patchcore_distilled_tiny_unet"]
    config.data["phase5"]["epochs"] = 1
    config.data["phase5"]["patchcore_distillation"] = {
        "teacher": "patchcore_lite",
        "image_size": 32,
        "max_memory_patches": 128,
        "distance_chunk_size": 32,
        "teacher_loss_weight": 0.2,
    }
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["evaluator"] for row in rows} == {"patchcore_distilled_tiny_unet"}
    assert all(row["distillation_teacher"] == "patchcore_lite" for row in rows)
    assert all(float(row["distillation_teacher_loss_weight"]) == pytest.approx(0.2) for row in rows)
    assert all(int(row["distillation_weighted_samples"]) > 0 for row in rows)
    assert all(row["note"] == "tiny_unet_distilled_from_patchcore_lite" for row in rows)
    assert all(Path(row["prediction_contact_sheet"]).exists() for row in rows)


def test_phase5_teacher_refined_tiny_unet_smoke(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["teacher_refined_tiny_unet"]
    config.data["phase5"]["epochs"] = 1
    config.data["phase5"]["patchcore_distillation"] = {
        "teacher": "patchcore_lite",
        "image_size": 32,
        "max_memory_patches": 128,
        "distance_chunk_size": 32,
    }
    config.data["phase5"]["teacher_refinement"] = {
        "positive_quantile": 0.75,
        "negative_quantile": 0.25,
        "teacher_positive_loss_weight": 0.75,
        "teacher_negative_loss_weight": 0.75,
        "disagreement_loss_weight": 0.25,
    }
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["evaluator"] for row in rows} == {"teacher_refined_tiny_unet"}
    assert all(row["distillation_teacher"] == "patchcore_lite" for row in rows)
    assert all(float(row["teacher_refined_positive_rate"]) >= 0.0 for row in rows)
    assert all(float(row["teacher_refined_weight_mean"]) > 0.0 for row in rows)
    assert all(row["note"] == "tiny_unet_teacher_refined_by_patchcore_lite" for row in rows)
    assert all(Path(row["prediction_contact_sheet"]).exists() for row in rows)


def test_phase5_teacher_refined_resnet18_unet_smoke(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["teacher_refined_resnet18_unet"]
    config.data["phase5"]["epochs"] = 1
    config.data["phase5"]["image_size"] = 64
    config.data["phase5"]["patchcore_distillation"] = {
        "teacher": "patchcore_lite",
        "image_size": 64,
        "max_memory_patches": 128,
        "distance_chunk_size": 32,
    }
    config.data["phase5"]["teacher_refinement"] = {
        "positive_quantile": 0.75,
        "negative_quantile": 0.25,
        "teacher_positive_loss_weight": 0.75,
        "teacher_negative_loss_weight": 0.75,
        "disagreement_loss_weight": 0.25,
        "max_target_area_multiplier": 1.75,
        "max_teacher_positive_fraction": 0.20,
    }
    config.data["phase5"]["pretrained_student"] = {
        "architecture": "resnet18_unet",
        "weights": "none",
        "decoder_channels": 16,
    }
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["evaluator"] for row in rows} == {"teacher_refined_resnet18_unet"}
    assert all(row["distillation_teacher"] == "patchcore_lite" for row in rows)
    assert all(float(row["teacher_refined_positive_rate"]) >= 0.0 for row in rows)
    assert all(float(row["teacher_refined_weight_mean"]) > 0.0 for row in rows)
    assert all(row["note"] == "resnet18_unet_teacher_refined_by_patchcore_lite" for row in rows)
    assert all(Path(row["prediction_contact_sheet"]).exists() for row in rows)


def test_phase5_patchcore_guided_teacher_refined_resnet18_unet_smoke(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config.data["phase5"]["evaluators"] = ["patchcore_guided_teacher_refined_resnet18_unet"]
    config.data["phase5"]["epochs"] = 1
    config.data["phase5"]["image_size"] = 64
    config.data["phase5"]["patchcore_guided_resnet18_prior_weight"] = 0.35
    config.data["phase5"]["patchcore_distillation"] = {
        "teacher": "patchcore_lite",
        "image_size": 64,
        "max_memory_patches": 128,
        "distance_chunk_size": 32,
    }
    config.data["phase5"]["teacher_refinement"] = {
        "positive_quantile": 0.75,
        "negative_quantile": 0.25,
        "teacher_positive_loss_weight": 0.75,
        "teacher_negative_loss_weight": 0.75,
        "disagreement_loss_weight": 0.25,
        "max_target_area_multiplier": 1.75,
        "max_teacher_positive_fraction": 0.20,
    }
    config.data["phase5"]["pretrained_student"] = {
        "architecture": "resnet18_unet",
        "weights": "none",
        "decoder_channels": 16,
    }
    _prepare_all_artifacts(config)

    csv_path = run_phase5_evaluation(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["evaluator"] for row in rows} == {"patchcore_guided_teacher_refined_resnet18_unet"}
    assert all(row["distillation_teacher"] == "patchcore_lite" for row in rows)
    assert all(float(row["patchcore_guided_prior_weight"]) == pytest.approx(0.35) for row in rows)
    assert all(float(row["teacher_refined_positive_rate"]) >= 0.0 for row in rows)
    assert all(float(row["teacher_refined_weight_mean"]) > 0.0 for row in rows)
    assert all(row["note"] == "resnet18_unet_teacher_refined_patchcore_fused_by_patchcore_lite" for row in rows)
    assert all(Path(row["prediction_contact_sheet"]).exists() for row in rows)


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


def test_phase5_boundary_area_and_outside_possible_losses() -> None:
    target = torch.zeros((1, 1, 8, 8))
    target[:, :, 2:6, 2:6] = 1.0
    good_logits = torch.full_like(target, -5.0)
    good_logits[:, :, 2:6, 2:6] = 5.0
    bad_logits = torch.zeros_like(target)
    weight = torch.ones_like(target)

    assert _soft_boundary_loss(good_logits, target, weight, width=3) < _soft_boundary_loss(bad_logits, target, weight, width=3)
    assert _weighted_area_prior_loss(good_logits, target, weight) < _weighted_area_prior_loss(bad_logits, target, weight)

    possible = torch.zeros((1, 8, 8))
    possible[:, 2:6, 2:6] = 1.0
    outside_bad = torch.full_like(target, 4.0)
    outside_good = torch.full_like(target, -4.0)

    assert _outside_possible_region_loss(outside_good, possible) < _outside_possible_region_loss(outside_bad, possible)


def test_phase5_evaluator_alias_and_unknown_rejection() -> None:
    assert _phase5_evaluators({"evaluators": ["tiny_unet", "patchcore"]}) == ["tiny_unet", "patchcore_resnet"]
    assert _phase5_evaluators({"evaluators": ["patchcore_lite", "patchcore_full"]}) == [
        "patchcore_lite",
        "patchcore_resnet",
    ]
    assert _phase5_evaluators({"evaluators": ["tiny_unet_patchcore", "patchcore_guided"]}) == [
        "patchcore_guided_tiny_unet"
    ]
    assert _phase5_evaluators({"evaluators": ["patchcore_distilled", "resnet_distilled_tiny_unet"]}) == [
        "patchcore_distilled_tiny_unet"
    ]
    assert _phase5_evaluators({"evaluators": ["teacher_refined", "patchcore_refined_tiny_unet"]}) == [
        "teacher_refined_tiny_unet"
    ]
    assert _phase5_evaluators({"evaluators": ["resnet18_unet", "patchcore_refined_resnet18_unet"]}) == [
        "teacher_refined_resnet18_unet"
    ]
    assert _phase5_evaluators({"evaluators": ["teacher_refined_resnet18_patchcore", "patchcore_guided_resnet18_unet"]}) == [
        "patchcore_guided_teacher_refined_resnet18_unet"
    ]
    with pytest.raises(ValueError, match="Unsupported Phase 5 evaluator"):
        _phase5_evaluators({"evaluators": ["tiny_unet", "mystery"]})


def test_phase5_patchcore_guided_fuses_student_and_prior_scores() -> None:
    student = [
        {
            "score": np.asarray([[0.0, 0.2], [0.4, 1.0]], dtype=np.float32),
            "truth": np.zeros((2, 2), dtype=np.float32),
            "morphology": "scratch_band",
        }
    ]
    prior = [
        {
            "score": np.asarray([[1.0, 0.8], [0.6, 0.0]], dtype=np.float32),
            "truth": np.zeros((2, 2), dtype=np.float32),
            "morphology": "scratch_band",
        }
    ]

    fused = _fuse_score_rows(student, prior, prior_weight=0.25)

    expected = np.asarray([[0.25, 0.35], [0.45, 0.75]], dtype=np.float32)
    assert np.allclose(fused[0]["score"], expected)
    assert np.allclose(fused[0]["student_score"], student[0]["score"])
    assert np.allclose(fused[0]["patchcore_prior_score"], prior[0]["score"])
    assert float(fused[0]["patchcore_guided_prior_weight"]) == pytest.approx(0.25)


def test_phase5_patchcore_guided_fusion_rejects_mismatched_shapes() -> None:
    student = [{"score": np.zeros((2, 2), dtype=np.float32), "truth": np.zeros((2, 2), dtype=np.float32)}]
    prior = [{"score": np.zeros((3, 3), dtype=np.float32), "truth": np.zeros((3, 3), dtype=np.float32)}]

    with pytest.raises(ValueError, match="different score shapes"):
        _fuse_score_rows(student, prior, prior_weight=0.5)


def test_phase5_fused_score_calibration_suppresses_low_score_halo() -> None:
    rows = [
        {
            "score": np.asarray([[0.2, 0.4], [0.6, 1.0]], dtype=np.float32),
            "truth": np.zeros((2, 2), dtype=np.float32),
        }
    ]

    calibrated, metadata = _calibrate_fused_score_rows(
        rows,
        {
            "fused_score_calibration": {
                "enabled": True,
                "low_quantile": 0.50,
                "high_quantile": 1.0,
                "gamma": 2.0,
                "threshold_area_penalty_weight": 0.35,
            }
        },
    )

    assert metadata["score_calibration_enabled"] == 1.0
    assert metadata["score_calibration_gamma"] == pytest.approx(2.0)
    assert metadata["score_calibration_threshold_area_penalty_weight"] == pytest.approx(0.35)
    assert np.allclose(calibrated[0]["raw_fused_score"], rows[0]["score"])
    assert float(calibrated[0]["score"][0, 0]) == pytest.approx(0.0)
    assert float(calibrated[0]["score"][1, 1]) == pytest.approx(1.0)
    assert float(calibrated[0]["score"][1, 0]) < float(rows[0]["score"][1, 0])


def test_phase5_variant_ratio_arbitration_recommends_and_demotes() -> None:
    rows = [
        {
            "evaluator": "patchcore_guided_teacher_refined_resnet18_unet",
            "variant": "real_only",
            "quality_profile": "none",
            "synthetic_ratio": 0.0,
            "pixel_auroc": 0.80,
            "aupro": 0.45,
            "dice": 0.24,
            "predicted_positive_rate": 0.06,
            "truth_positive_rate": 0.05,
        },
        {
            "evaluator": "patchcore_guided_teacher_refined_resnet18_unet",
            "variant": "qwen_mask_only:scratch_thin_detail",
            "quality_profile": "scratch_thin_detail",
            "synthetic_ratio": 0.25,
            "pixel_auroc": 0.83,
            "aupro": 0.50,
            "dice": 0.25,
            "predicted_positive_rate": 0.065,
            "truth_positive_rate": 0.05,
        },
        {
            "evaluator": "patchcore_guided_teacher_refined_resnet18_unet",
            "variant": "qwen_mask_only:scratch_thin_detail",
            "quality_profile": "scratch_thin_detail",
            "synthetic_ratio": 0.50,
            "pixel_auroc": 0.78,
            "aupro": 0.40,
            "dice": 0.20,
            "predicted_positive_rate": 0.12,
            "truth_positive_rate": 0.05,
        },
    ]

    decisions = _variant_ratio_arbitration(
        rows,
        {"variant_ratio_arbitration": {"max_predicted_positive_multiplier": 1.5, "high_ratio_threshold": 0.5}},
    )

    by_ratio = {float(row["synthetic_ratio"]): row for row in decisions}
    assert by_ratio[0.25]["decision"] == "recommended"
    assert "improves Dice and ranking" in by_ratio[0.25]["reason"]
    assert by_ratio[0.50]["decision"] == "demote_ratio"
    assert "inflates predicted area" in by_ratio[0.50]["reason"]


def test_phase5_weighted_teacher_mse_loss_tracks_teacher_map() -> None:
    logits_good = torch.full((1, 1, 2, 2), -4.0)
    logits_good[0, 0, 0, 0] = 4.0
    logits_bad = torch.zeros((1, 1, 2, 2))
    teacher = torch.zeros((1, 1, 2, 2))
    teacher[0, 0, 0, 0] = 1.0
    weight = torch.ones((1, 1, 2, 2))

    assert _weighted_teacher_mse_loss(logits_good, teacher, weight) < _weighted_teacher_mse_loss(logits_bad, teacher, weight)


def test_phase5_teacher_distillation_uses_confident_extremes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (2, 2), (100, 100, 100)).save(image_path)
    Image.new("L", (2, 2), 0).save(mask_path)
    sample = SegmentationSample(str(image_path), str(mask_path), "wood", "scratch", "real_adaptation")
    teacher_maps = {str(image_path): np.asarray([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)}

    target, confidence = _teacher_distillation_tensors_for_sample(sample, teacher_maps, 2, "cpu", 0.75, 0.25)

    assert target.shape == (1, 2, 2)
    assert confidence.shape == (1, 2, 2)
    assert float(target[0, 1, 1]) == 1.0
    assert float(confidence[0, 1, 1]) == 1.0
    assert float(confidence[0, 0, 0]) == 1.0
    assert float(confidence[0, 0, 1]) == 0.0


def test_phase5_teacher_refined_target_respects_possible_region_and_core(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    core_path = tmp_path / "core.png"
    possible_path = tmp_path / "possible.png"
    Image.new("RGB", (2, 2), (100, 100, 100)).save(image_path)
    Image.new("L", (2, 2), 0).save(mask_path)
    core = Image.new("L", (2, 2), 0)
    core.putpixel((0, 0), 255)
    core.save(core_path)
    possible = Image.new("L", (2, 2), 0)
    possible.putpixel((1, 1), 255)
    possible.save(possible_path)
    sample = SegmentationSample(
        str(image_path),
        str(mask_path),
        "wood",
        "scratch",
        "real_adaptation",
        positive_core_path=str(core_path),
        possible_region_path=str(possible_path),
    )
    teacher_maps = {str(image_path): np.asarray([[0.1, 0.2], [0.3, 1.0]], dtype=np.float32)}

    target, weight = _teacher_refined_target_and_weight(
        sample,
        teacher_maps,
        2,
        "cpu",
        {"uncertainty_loss_weight": 0.25, "positive_core_loss_weight": 2.0},
        0.75,
        0.25,
        0.75,
        0.75,
        0.25,
        2.0,
        1.75,
        0.50,
    )

    assert float(target[0, 0, 0]) == 1.0
    assert float(weight[0, 0, 0]) == pytest.approx(2.0)
    assert float(target[0, 1, 1]) == 1.0
    assert float(weight[0, 1, 1]) >= 0.75
    assert float(target[0, 0, 1]) == 0.0


def test_phase5_teacher_refined_target_caps_teacher_added_area(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    possible_path = tmp_path / "possible.png"
    Image.new("RGB", (4, 4), (100, 100, 100)).save(image_path)
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((0, 0), 255)
    mask.save(mask_path)
    Image.new("L", (4, 4), 255).save(possible_path)
    sample = SegmentationSample(
        str(image_path),
        str(mask_path),
        "wood",
        "scratch",
        "real_adaptation",
        possible_region_path=str(possible_path),
    )
    teacher_maps = {
        str(image_path): np.asarray(
            [
                [0.99, 0.98, 0.97, 0.96],
                [0.95, 0.94, 0.93, 0.92],
                [0.10, 0.10, 0.10, 0.10],
                [0.10, 0.10, 0.10, 0.10],
            ],
            dtype=np.float32,
        )
    }

    target, _ = _teacher_refined_target_and_weight(
        sample,
        teacher_maps,
        4,
        "cpu",
        {"uncertainty_loss_weight": 0.25},
        0.50,
        0.25,
        0.75,
        0.75,
        0.25,
        2.0,
        1.75,
        0.50,
    )

    assert int(target.sum()) == 2


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
        "sd_clean_inpaint_negatives": {"enabled": True, "samples_per_clean": 2, "max_total_samples": 2},
    }

    samples = _clean_inpaint_negative_samples(config, manifest, phase5, "heuristic", 11)

    assert samples
    assert len(samples) == 2
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


def test_phase5_synthetic_selector_penalizes_unbalanced_tfidg_coverage() -> None:
    phase5 = {
        "synthetic_selector": {
            "enabled": True,
            "tfidg_lite": {
                "enabled": True,
                "min_adaptive_mask_coverage_score": 0.35,
                "max_adaptive_mask_coverage_fraction": 0.58,
                "target_adaptive_mask_coverage_fraction": 0.38,
                "coverage_fraction_tolerance": 0.20,
                "coverage_balance_penalty_weight": 0.12,
                "selector_score_weight": 0.35,
            },
        },
    }
    base = {
        "category": "wood",
        "defect_type": "scratch",
        "variant": "qwen_mask_only",
        "quality_profile": "scratch_thin_detail",
        "output_path": "balanced.png",
        "generation_quality_score": 0.2,
        "defect_visibility_score": 0.05,
        "background_preservation_l1": 0.01,
        "outside_refined_change_fraction": 0.01,
        "inpaint_mask_area_fraction": 0.03,
        "quality_flags": [],
        "settings": {"quality_morphology": "scratch_band"},
        "tfidg_lite": {
            "tfidg_lite_score": 0.8,
            "adaptive_mask_coverage_score": 0.8,
            "adaptive_mask_coverage_fraction": 0.38,
            "feature_alignment_score": 0.9,
            "texture_preservation_score": 0.9,
            "leakage_score": 0.9,
        },
    }
    high_fraction = {
        **base,
        "output_path": "too_much.png",
        "tfidg_lite": {**base["tfidg_lite"], "adaptive_mask_coverage_fraction": 0.62},
    }
    off_target = {
        **base,
        "output_path": "off_target.png",
        "tfidg_lite": {**base["tfidg_lite"], "adaptive_mask_coverage_fraction": 0.54},
    }

    balanced_decision = _synthetic_selector_decision(base, phase5)
    off_target_decision = _synthetic_selector_decision(off_target, phase5)
    high_decision = _synthetic_selector_decision(high_fraction, phase5)

    assert balanced_decision["accepted"]
    assert off_target_decision["accepted"]
    assert off_target_decision["selector_score"] < balanced_decision["selector_score"]
    assert not high_decision["accepted"]
    assert "high_tfidg_mask_coverage_fraction" in high_decision["reject_reasons"]


def test_phase5_selected_candidate_keys_caps_repaired_samples_per_group() -> None:
    phase5 = {
        "synthetic_selector": {
            "enabled": True,
            "max_per_group": 4,
            "max_repaired_fraction_per_group": 0.5,
            "group_by": ["category", "defect_type", "morphology"],
        }
    }

    def row(name: str, repaired: bool) -> dict[str, object]:
        return {
            "category": "wood",
            "defect_type": "scratch",
            "quality_profile": "scratch_thin_detail",
            "settings": {"quality_morphology": "scratch_band"},
            "output_path": name,
            "critic_guided_generation": {"selected_repair_index": 0 if repaired else -1},
        }

    candidates = [
        (row("repaired_a", True), {"accepted": True, "selector_score": 10.0}),
        (row("repaired_b", True), {"accepted": True, "selector_score": 9.0}),
        (row("repaired_c", True), {"accepted": True, "selector_score": 8.0}),
        (row("natural_a", False), {"accepted": True, "selector_score": 7.0}),
        (row("natural_b", False), {"accepted": True, "selector_score": 6.0}),
    ]

    selected = _selected_candidate_keys(candidates, phase5)

    assert selected == {"repaired_a", "repaired_b", "natural_a", "natural_b"}


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
