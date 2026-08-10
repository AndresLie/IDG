from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.candidate_calibration import (
    CANDIDATE_FEATURE_NAMES,
    CrossDatasetCandidateCalibrator,
    build_arbitration_candidates,
    evidence_augmented_regions,
    regions_to_mask,
    train_cross_dataset_candidate_calibrator,
)
from iadgen_v2.auto_mask.contracts import PixelPosteriorResult
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
                "sources": sources,
                "posterior_model_path": str(posterior_path),
                "model_output_path": str(model_path),
                "report_dir": str(report_dir),
                "validation_size": 48,
                "max_iter": 12,
                "candidate_quantiles": [0.85, 0.95],
                "max_components_per_quantile": 2,
                "min_component_area": 4,
            }
        },
    )

    manifest_path = train_cross_dataset_candidate_calibrator(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
