from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.area_calibrated_proposals import (
    CandidateAreaCalibrationContract,
    generate_area_calibrated_candidates,
)
from iadgen_v2.auto_mask.candidate_calibration import (
    CANDIDATE_FEATURE_NAMES,
    LIST_RANKING_CONTRACT,
    CandidateDecisionCalibrationContract,
    CrossDatasetCandidateCalibrator,
    _list_pair_feature_matrix,
    _list_rank_representation,
    _select_prediction_row,
    _select_source_decision_policy,
    _source_balanced_quantile,
    build_arbitration_candidates,
    evidence_augmented_regions,
    regions_to_mask,
    train_cross_dataset_candidate_calibrator,
)
from iadgen_v2.auto_mask.contracts import PixelPosteriorResult
from iadgen_v2.auto_mask.uncertainty_contours import (
    UncertaintyContourContract,
    generate_uncertainty_contour_candidates,
)
from iadgen_v2.config import AppConfig


def test_candidate_features_are_complete_and_category_free() -> None:
    fused = np.full((64, 64), 0.05, dtype=np.float32)
    fused[28:38, 44:54] = 0.95
    baseline = np.zeros((64, 64), dtype=bool)
    baseline[24:44, 40:58] = True
    compact = np.zeros_like(baseline)
    compact[29:37, 45:53] = True
    posterior = np.full((64, 64), 0.02, dtype=np.float32)
    posterior[compact] = 0.95

    candidates, regions = build_arbitration_candidates(
        baseline_mask=baseline,
        fused=fused,
        disagreement=np.zeros_like(fused),
        posterior_result=PixelPosteriorResult(posterior, compact, 0.5),
        semantic_regions=((0, 0, 24, 24),),
        min_component_area=4,
    )

    assert candidates[0].mode == "v4_baseline_selected"
    assert any(candidate.mode == "v4_posterior_compact" for candidate in candidates)
    assert all(tuple(candidate.features) == CANDIDATE_FEATURE_NAMES for candidate in candidates)
    assert all("category" not in name and "defect" not in name for name in CANDIDATE_FEATURE_NAMES)
    assert regions_to_mask(fused.shape, regions)[28:38, 44:54].all()


def test_evidence_augmented_regions_recover_signal_outside_semantic_box() -> None:
    fused = np.zeros((80, 80), dtype=np.float32)
    fused[60:70, 52:66] = 1.0

    regions = evidence_augmented_regions(
        fused,
        ((3, 4, 22, 25),),
        quantile=0.90,
        max_components=4,
        min_component_area=4,
    )
    mask = regions_to_mask(fused.shape, regions)

    assert regions[0] == (3, 4, 22, 25)
    assert mask[60:70, 52:66].all()
    assert mask.mean() < 0.25


def test_area_calibrated_candidates_add_bounded_evidence_ranked_unions() -> None:
    fused = np.zeros((64, 64), dtype=np.float32)
    for index, value in enumerate(np.linspace(0.99, 0.72, 10)):
        top = 2 + (index // 5) * 24
        left = 2 + (index % 5) * 12
        fused[top : top + 4, left : left + 4] = value
    contract = CandidateAreaCalibrationContract(
        enabled=True,
        quantiles=(0.90,),
        component_counts=(2, 4, 8),
        seed_quantile=0.99,
        support_quantiles=(0.90,),
        max_components=10,
        max_area_fraction=0.03,
    )

    proposals = generate_area_calibrated_candidates(
        fused,
        contract=contract,
        min_component_area=4,
    )
    by_mode = {proposal.mode: proposal for proposal in proposals}

    assert "v4_evidence_union_q900_top2" in by_mode
    assert "v4_evidence_union_q900_top4" in by_mode
    assert "v4_evidence_union_q900_top8" not in by_mode
    assert int(by_mode["v4_evidence_union_q900_top2"].mask.sum()) == 32
    assert all(float(proposal.mask.mean()) <= 0.03 for proposal in proposals)


def test_area_calibrated_pool_is_strictly_additive_to_legacy_candidates() -> None:
    fused = np.full((64, 64), 0.01, dtype=np.float32)
    fused[8:16, 8:16] = 0.99
    fused[40:48, 40:48] = 0.95
    fused[8:16, 40:48] = 0.90
    baseline = fused > 0.98
    posterior = PixelPosteriorResult(fused, baseline, 0.5)
    common = {
        "baseline_mask": baseline,
        "fused": fused,
        "disagreement": np.zeros_like(fused),
        "posterior_result": posterior,
        "min_component_area": 4,
        "quantiles": (0.90,),
        "max_components": 4,
    }

    legacy, _ = build_arbitration_candidates(**common)
    expanded, _ = build_arbitration_candidates(
        **common,
        area_calibration=CandidateAreaCalibrationContract(
            enabled=True,
            quantiles=(0.90,),
            component_counts=(2,),
            seed_quantile=0.99,
            support_quantiles=(0.90,),
            max_components=4,
            max_area_fraction=0.10,
        ),
    )

    assert {candidate.mode for candidate in legacy} <= {candidate.mode for candidate in expanded}
    assert any(candidate.mode.startswith("v4_evidence_union_") for candidate in expanded)


def test_uncertainty_contours_trim_unstable_component_boundaries() -> None:
    fused = np.zeros((64, 64), dtype=np.float32)
    fused[18:46, 18:46] = 0.75
    fused[27:37, 27:37] = 1.0
    posterior = np.full_like(fused, 0.05)
    posterior[18:46, 18:46] = 0.60
    posterior[27:37, 27:37] = 1.0
    disagreement = np.zeros_like(fused)
    disagreement[18:46, 18:46] = 0.95
    disagreement[27:37, 27:37] = 0.0
    contract = UncertaintyContourContract(
        enabled=True,
        base_support_quantile=0.80,
        seed_quantile=0.95,
        contour_quantiles=(0.75, 0.85, 0.95),
        uncertainty_penalty=0.75,
        max_area_fraction=0.10,
        max_candidates_per_image=3,
    )

    proposals = generate_uncertainty_contour_candidates(
        fused,
        disagreement,
        posterior,
        contract=contract,
        min_component_area=4,
    )

    assert 1 <= len(proposals) <= 3
    assert all(proposal.mask[31, 31] for proposal in proposals)
    assert any(not proposal.mask[19, 19] for proposal in proposals)
    assert all(float(proposal.mask.mean()) <= 0.10 for proposal in proposals)


def test_uncertainty_contour_pool_is_bounded_and_strictly_additive() -> None:
    fused = np.zeros((64, 64), dtype=np.float32)
    fused[8:24, 8:24] = 0.85
    fused[13:19, 13:19] = 1.0
    baseline = fused > 0.90
    posterior = np.clip(fused + 0.05, 0.0, 1.0)
    disagreement = np.zeros_like(fused)
    disagreement[8:24, 8:24] = 0.90
    disagreement[11:21, 11:21] = 0.40
    disagreement[13:19, 13:19] = 0.0
    common = {
        "baseline_mask": baseline,
        "fused": fused,
        "disagreement": disagreement,
        "posterior_result": PixelPosteriorResult(posterior, baseline, 0.5),
        "min_component_area": 4,
        "quantiles": (0.85, 0.95),
        "max_components": 4,
    }

    legacy, _ = build_arbitration_candidates(**common)
    expanded, _ = build_arbitration_candidates(
        **common,
        uncertainty_contours=UncertaintyContourContract(
            enabled=True,
            contour_quantiles=(0.70, 0.80, 0.90, 0.95),
            max_candidates_per_image=2,
        ),
    )
    legacy_modes = {candidate.mode for candidate in legacy}
    expanded_modes = {candidate.mode for candidate in expanded}
    contour_modes = {mode for mode in expanded_modes if mode.startswith("v4_uncertainty_contour_")}

    assert legacy_modes <= expanded_modes
    assert 1 <= len(contour_modes) <= 2


def test_list_ranking_contract_is_category_free_and_permutation_equivariant() -> None:
    assert LIST_RANKING_CONTRACT.strategy == "all_pairs_list_rank"
    assert all(
        "category" not in name and "defect" not in name
        for name in LIST_RANKING_CONTRACT.pair_feature_names
    )
    matrix = np.asarray(
        [
            np.linspace(0.0, 0.5, len(CANDIDATE_FEATURE_NAMES)),
            np.linspace(0.2, 0.7, len(CANDIDATE_FEATURE_NAMES)),
            np.linspace(0.1, 0.6, len(CANDIDATE_FEATURE_NAMES)),
        ],
        dtype=np.float32,
    )
    permutation = np.asarray([2, 0, 1])

    original = _list_rank_representation(matrix)
    permuted = _list_rank_representation(matrix[permutation])

    assert np.allclose(permuted, original[permutation])

    forward = _list_pair_feature_matrix(matrix, np.asarray([0]), np.asarray([1]))
    reverse = _list_pair_feature_matrix(matrix, np.asarray([1]), np.asarray([0]))
    directional = 2 * len(CANDIDATE_FEATURE_NAMES)
    assert np.allclose(forward[:, :directional], -reverse[:, :directional])
    assert np.allclose(forward[:, directional:], reverse[:, directional:])


def test_list_rank_selection_requires_positive_lower_bound_over_v3() -> None:
    bundle = {
        "selection_strategy": "all_pairs_list_rank",
        "minimum_gain_lower_bound": 0.0,
        "minimum_expected_iou_gain": 0.01,
    }
    baseline = {
        "mode": "v4_baseline_selected",
        "is_baseline": True,
        "list_score": 0.0,
        "gain_lower_bound": 0.0,
        "expected_iou": 0.20,
    }
    unsafe = {
        "mode": "compact",
        "is_baseline": False,
        "list_score": 0.4,
        "gain_lower_bound": -0.01,
        "expected_iou": 0.24,
    }
    safe = {**unsafe, "gain_lower_bound": 0.02}

    selected, override = _select_prediction_row(bundle, [baseline, unsafe])
    assert selected["mode"] == "v4_baseline_selected"
    assert not override

    selected, override = _select_prediction_row(bundle, [baseline, safe])
    assert selected["mode"] == "compact"
    assert override

    disagreement = {**safe, "expected_iou": 0.20}
    selected, override = _select_prediction_row(bundle, [baseline, disagreement])
    assert selected["mode"] == "v4_baseline_selected"
    assert not override


def test_source_calibrated_list_rank_uses_margin_and_fails_closed() -> None:
    baseline = {
        "mode": "v4_baseline_selected",
        "is_baseline": True,
        "list_score": 0.10,
    }
    candidate = {
        "mode": "compact",
        "is_baseline": False,
        "list_score": 0.12,
        "gain_lower_bound": -1.0,
    }
    bundle = {
        "selection_strategy": "source_calibrated_list_rank",
        "decision_calibration_enabled": True,
        "decision_margin_threshold": 0.01,
    }

    selected, override = _select_prediction_row(bundle, [baseline, candidate])
    assert selected["mode"] == "compact"
    assert override

    selected, override = _select_prediction_row(
        {**bundle, "decision_calibration_enabled": False},
        [baseline, candidate],
    )
    assert selected["mode"] == "v4_baseline_selected"
    assert not override


def test_source_decision_policy_selects_only_eligible_threshold() -> None:
    records = [
        {"dataset_id": "a", "candidate_available": True, "decision_margin": 0.006, "dice_gain": 0.04},
        {"dataset_id": "a", "candidate_available": True, "decision_margin": 0.020, "dice_gain": -0.01},
        {"dataset_id": "b", "candidate_available": True, "decision_margin": 0.006, "dice_gain": 0.03},
        {"dataset_id": "b", "candidate_available": True, "decision_margin": 0.020, "dice_gain": -0.03},
    ]
    contract = CandidateDecisionCalibrationContract(
        threshold_grid=(0.005, 0.01),
        minimum_macro_dice_gain=0.005,
        max_source_dice_regression=0.02,
    )

    selected = _select_source_decision_policy(records, contract)
    rejected = _select_source_decision_policy(
        records,
        CandidateDecisionCalibrationContract(
            threshold_grid=(0.005, 0.01),
            minimum_macro_dice_gain=0.05,
            max_source_dice_regression=0.02,
        ),
    )

    assert selected["decision_calibration_enabled"]
    assert selected["decision_margin_threshold"] == 0.005
    assert selected["decision_calibration_worst_source_dice_gain"] >= 0.0
    assert not rejected["decision_calibration_enabled"]


def test_source_balanced_quantile_prevents_large_dataset_tail_dominance() -> None:
    values = np.asarray([0.0] * 100 + [1.0] * 10, dtype=np.float32)
    sources = np.asarray(["large"] * 100 + ["small"] * 10)

    assert float(np.quantile(values, 0.90)) < 1.0
    assert _source_balanced_quantile(values, sources, 0.90) == 1.0


def test_cross_dataset_candidate_training_writes_loadable_guarded_bundle(tmp_path: Path) -> None:
    posterior_path = _write_fixed_posterior_bundle(tmp_path)
    sources = []
    for dataset_index, dataset_id in enumerate(("dataset_a", "dataset_b")):
        dataset_root = tmp_path / dataset_id
        metadata_path = dataset_root / "metadata.jsonl"
        reference_path = dataset_root / "references.jsonl"
        metadata_path.parent.mkdir(parents=True)
        metadata_rows = []
        reference_rows = []
        for category_index in range(3):
            category = f"part_{category_index}"
            normal_dir = dataset_root / category / "train" / "good"
            normal_dir.mkdir(parents=True)
            Image.new("RGB", (48, 48), (100, 100, 100)).save(normal_dir / "normal.png")
            for image_index in range(3):
                stem = f"{category_index}_{image_index}"
                image_path = dataset_root / category / f"{stem}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48), (100, 100, 100)).save(image_path)
                top = 8 + category_index * 3
                left = 24 + image_index * 2
                fused = np.full((48, 48), 0.04, dtype=np.float32)
                fused[top : top + 8, left : left + 7] = 0.96
                truth = np.zeros((48, 48), dtype=np.uint8)
                truth[top : top + 8, left : left + 7] = 255
                baseline = np.zeros((48, 48), dtype=np.uint8)
                baseline[max(0, top - 4) : top + 12, max(0, left - 5) : left + 12] = 255
                uncertainty = np.zeros((48, 48), dtype=np.uint8)
                fused_path = dataset_root / category / f"{stem}_fused.png"
                truth_path = dataset_root / category / f"{stem}_mask.png"
                baseline_path = dataset_root / category / f"{stem}_baseline.png"
                uncertainty_path = dataset_root / category / f"{stem}_uncertainty.png"
                Image.fromarray(np.uint8(fused * 255)).save(fused_path)
                Image.fromarray(truth).save(truth_path)
                Image.fromarray(baseline).save(baseline_path)
                Image.fromarray(uncertainty).save(uncertainty_path)
                metadata_rows.append(
                    {
                        "category": category,
                        "defect_type": "bad",
                        "image_path": str(image_path),
                        "region_xyxy": [0, 0, 20, 20],
                        "eval_mask_path": str(baseline_path),
                        "uncertainty_mask_path": str(uncertainty_path),
                        "settings": {"mask_parameters": {"fused_evidence_path": str(fused_path)}},
                    }
                )
                reference_rows.append(
                    {
                        "category": category,
                        "defect_type": "bad",
                        "image_path": str(image_path),
                        "official_mask_path": str(truth_path),
                    }
                )
        metadata_path.write_text("".join(json.dumps(row) + "\n" for row in metadata_rows), encoding="utf-8")
        reference_path.write_text("".join(json.dumps(row) + "\n" for row in reference_rows), encoding="utf-8")
        sources.append(
            {
                "dataset_id": dataset_id,
                "categories": ["part_0", "part_1", "part_2"],
                "metadata_path": str(metadata_path),
                "runtime_root": str(dataset_root),
                "reference": {"kind": "manifest", "manifest_path": str(reference_path)},
            }
        )

    model_path = tmp_path / "candidate.joblib"
    report_dir = tmp_path / "reports"
    config = AppConfig(
        path=tmp_path / "config.yaml",
        data={
            "candidate_calibration": {
                "selection_strategy": "source_calibrated_list_rank",
                "risk_calibration": "source_jackknife_equal_weight",
                "sources": sources,
                "posterior_model_path": str(posterior_path),
                "model_output_path": str(model_path),
                "report_dir": str(report_dir),
                "validation_size": 48,
                "max_iter": 12,
                "candidate_quantiles": [0.85, 0.95],
                "max_components_per_quantile": 2,
                "min_component_area": 4,
                "candidate_row_cache_path": str(tmp_path / "candidate_rows.joblib"),
                "area_calibrated_proposals": {
                    "enabled": True,
                    "quantiles": [0.90],
                    "component_counts": [2],
                    "seed_quantile": 0.99,
                    "support_quantiles": [0.90],
                    "max_components": 4,
                    "max_area_fraction": 0.10,
                },
                "uncertainty_contours": {
                    "enabled": True,
                    "base_support_quantile": 0.80,
                    "seed_quantile": 0.95,
                    "contour_quantiles": [0.75, 0.90],
                    "uncertainty_penalty": 0.75,
                    "max_area_fraction": 0.10,
                    "max_candidates_per_image": 2,
                    "seed_dilation_iterations": 1,
                },
            }
        },
    )

    manifest_path = train_cross_dataset_candidate_calibrator(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidate_row_cache"]["enabled"]
    assert not manifest["candidate_row_cache"]["cache_hit"]
    train_cross_dataset_candidate_calibrator(config)
    cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert cached_manifest["candidate_row_cache"]["cache_hit"]
    assert cached_manifest["candidate_row_cache"]["fingerprint"] == manifest["candidate_row_cache"]["fingerprint"]
    calibrator = CrossDatasetCandidateCalibrator(model_path)
    fused = np.full((48, 48), 0.04, dtype=np.float32)
    fused[10:18, 25:32] = 0.96
    baseline = np.zeros((48, 48), dtype=bool)
    baseline[6:22, 20:37] = True
    compact = fused > 0.9
    result = calibrator.select(
        baseline_mask=baseline,
        fused=fused,
        disagreement=np.zeros_like(fused),
        posterior_result=PixelPosteriorResult(fused, compact, 0.5),
        semantic_regions=((0, 0, 20, 20),),
        min_component_area=4,
    )

    assert manifest["model_sha256"]
    assert set(calibrator.bundle["development_datasets"]) == {"dataset_a", "dataset_b"}
    assert calibrator.bundle["selection_strategy"] == "source_calibrated_list_rank"
    assert calibrator.bundle["risk_calibration"] == "source_jackknife_equal_weight"
    assert calibrator.bundle["uncertainty_contours"]["enabled"]
    assert calibrator.bundle["uncertainty_contours"]["max_candidates_per_image"] == 2
    assert calibrator.bundle["risk_calibration_effective"] == "source_jackknife_equal_weight"
    assert set(calibrator.bundle["gain_residual_q90_by_source"]) == {"dataset_a", "dataset_b"}
    assert all(source["categories"] == ["part_0", "part_1", "part_2"] for source in manifest["sources"])
    assert tuple(calibrator.bundle["pair_feature_names"]) == LIST_RANKING_CONTRACT.pair_feature_names
    assert calibrator.bundle["decision_calibration"] == "nested_source_jackknife_list_margin"
    assert result.baseline.mode == "v4_baseline_selected"
    assert result.predictions
    assert (report_dir / "leave_dataset_out_candidate_metrics.json").exists()


def _write_fixed_posterior_bundle(tmp_path: Path) -> Path:
    import joblib
    from sklearn.dummy import DummyClassifier

    from iadgen_v2.auto_mask.posterior_calibration import POSTERIOR_FEATURE_NAMES

    features = np.asarray([[0.0] * len(POSTERIOR_FEATURE_NAMES), [1.0] * len(POSTERIOR_FEATURE_NAMES)], dtype=np.float32)
    labels = np.asarray([0, 1], dtype=np.uint8)
    model = DummyClassifier(strategy="prior").fit(features, labels)
    path = tmp_path / "posterior.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "feature_names": list(POSTERIOR_FEATURE_NAMES),
            "model": model,
            "calibrator": None,
            "threshold": 0.2,
            "override_area_ratio": 4.0,
            "compact": {
                "threshold_mode": "fixed",
                "seed_threshold": 0.4,
                "seed_quantile": 0.99,
                "min_component_area": 4,
                "max_components": 4,
                "max_area_fraction": 0.2,
                "growth_factor": 4.0,
                "minimum_output_area_fraction": 0.0,
                "fallback_rank_threshold": 0.95,
                "fallback_seed_quantile": 0.999,
                "fallback_growth_factor": 4.0,
                "fallback_max_components": 2,
            },
        },
        path,
    )
    return path
