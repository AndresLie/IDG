from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.contracts import AutoMaskContext, EvidenceMap, PixelPosteriorResult
from iadgen_v2.auto_mask.pipeline import run_generic_evidence_pipeline
from iadgen_v2.auto_mask.posterior_calibration import (
    POSTERIOR_FEATURE_NAMES,
    ScoreToMaskCalibrator,
    extract_calibrated_components,
    extract_compact_components,
    posterior_feature_image,
    train_score_to_mask_calibrator,
)
from iadgen_v2.auto_mask.selection import GenericCandidateSelector
from iadgen_v2.config import AppConfig


class _FixedEvidenceProvider:
    name = "fixed"

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        return EvidenceMap(self.name, self.values, 1.0, augmentation_consistency=1.0)


class _FixedPosteriorCalibrator:
    def __init__(self, posterior: np.ndarray, compact_mask: np.ndarray) -> None:
        self.posterior = posterior
        self.compact_mask = compact_mask

    def predict(
        self,
        fused: np.ndarray,
        *,
        min_component_area: int = 8,
        normal_paths: tuple[Path, ...] = (),
        semantic_regions: tuple[tuple[int, int, int, int], ...] = (),
    ) -> PixelPosteriorResult:
        return PixelPosteriorResult(
            posterior=self.posterior,
            compact_mask=self.compact_mask,
            threshold=0.6,
            diagnostics={"component_count": 1},
        )


def test_posterior_feature_image_is_finite_and_category_agnostic() -> None:
    fused = np.linspace(0.0, 1.0, 24 * 20, dtype=np.float32).reshape(24, 20)
    original = fused.copy()

    features = posterior_feature_image(fused)

    assert features.shape == (24, 20, len(POSTERIOR_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.array_equal(fused, original)
    assert np.all((features >= 0.0) & (features <= 1.0))


def test_compact_extractor_removes_low_confidence_halo_with_exact_budget() -> None:
    posterior = np.full((64, 64), 0.05, dtype=np.float32)
    posterior[8:56, 8:56] = 0.25
    posterior[26:34, 27:35] = 0.95

    mask, diagnostics = extract_compact_components(
        posterior,
        threshold=0.20,
        seed_threshold=0.80,
        seed_quantile=0.995,
        min_component_area=8,
        max_components=4,
        max_area_fraction=0.20,
        growth_factor=2.0,
    )

    assert mask[26:34, 27:35].all()
    assert int(mask.sum()) <= 128
    assert diagnostics["compact_area_fraction"] < diagnostics["raw_area_fraction"]


def test_pipeline_can_opt_in_to_calibrated_compact_eval_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "defect.png"
    normal_path = tmp_path / "normal.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(image_path)
    Image.new("RGB", (64, 64), (120, 120, 120)).save(normal_path)
    fused = np.full((64, 64), 0.05, dtype=np.float32)
    fused[10:54, 10:54] = 0.70
    compact = np.zeros((64, 64), dtype=bool)
    compact[26:36, 27:37] = True
    posterior = np.full((64, 64), 0.02, dtype=np.float32)
    posterior[compact] = 0.95
    context = AutoMaskContext(
        image_path=image_path,
        normal_paths=(normal_path,),
        image_size=(64, 64),
        semantic_regions=((0, 0, 64, 64),),
        foreground=np.ones((64, 64), dtype=bool),
        cache_key="posterior-test",
    )

    artifacts = run_generic_evidence_pipeline(
        context,
        [_FixedEvidenceProvider(fused)],
        output_dir=tmp_path / "masks",
        variant_dir=tmp_path / "variants",
        artifact_stem="sample",
        selector=GenericCandidateSelector(),
        min_component_area=4,
        edge_refine=False,
        posterior_calibrator=_FixedPosteriorCalibrator(posterior, compact),  # type: ignore[arg-type]
        posterior_override_eval=True,
    )

    written = np.asarray(Image.open(artifacts["eval_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert np.array_equal(written, compact)
    assert artifacts["selected_refinement"] == "calibrated_posterior_compact"
    assert artifacts["selection_arbitration"]["reason"] == "calibrated_posterior_override"
    assert artifacts["parameters"]["architecture"] == "v4-generic-evidence"
    assert Path(artifacts["mask_variant_paths"]["calibrated_posterior"]).exists()


def test_calibrated_extractor_falls_back_when_model_output_collapses() -> None:
    fused = np.full((64, 64), 0.05, dtype=np.float32)
    fused[20:40, 22:42] = 0.9
    features = posterior_feature_image(fused)
    posterior = np.zeros((64, 64), dtype=np.float32)

    mask, diagnostics = extract_calibrated_components(
        posterior,
        feature_image=features,
        threshold=0.4,
        compact={
            "threshold_mode": "otsu_floor",
            "minimum_output_area_fraction": 0.0,
            "fallback_rank_threshold": 0.95,
            "fallback_seed_quantile": 0.999,
            "fallback_growth_factor": 4.0,
            "fallback_max_components": 3,
        },
        min_component_area=4,
    )

    assert mask.any()
    assert diagnostics["fallback"] == "invariant_rank_compact"


def test_cross_dataset_training_writes_loadable_bundle_and_lodo_report(tmp_path: Path) -> None:
    sources = []
    for dataset_index, dataset_id in enumerate(("dataset_a", "dataset_b")):
        metadata_path = tmp_path / dataset_id / "metadata.jsonl"
        reference_path = tmp_path / dataset_id / "references.jsonl"
        metadata_path.parent.mkdir(parents=True)
        metadata_rows = []
        reference_rows = []
        for image_index in range(3):
            fused = np.full((32, 32), 0.05 + 0.01 * dataset_index, dtype=np.float32)
            truth = np.zeros((32, 32), dtype=np.uint8)
            top = 7 + image_index
            fused[top : top + 8, 11:19] = 0.92
            truth[top : top + 8, 11:19] = 255
            fused_path = tmp_path / dataset_id / f"{image_index}_fused.png"
            truth_path = tmp_path / dataset_id / f"{image_index}_mask.png"
            Image.fromarray(np.uint8(fused * 255)).save(fused_path)
            Image.fromarray(truth).save(truth_path)
            image_path = tmp_path / dataset_id / f"{image_index}.png"
            Image.new("RGB", (32, 32)).save(image_path)
            metadata_rows.append(
                {
                    "category": dataset_id,
                    "defect_type": "bad",
                    "image_path": str(image_path),
                    "eval_mask_path": str(truth_path),
                    "settings": {"mask_parameters": {"fused_evidence_path": str(fused_path)}},
                }
            )
            reference_rows.append(
                {
                    "category": dataset_id,
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
                "runtime_root": str(tmp_path / dataset_id),
                "reference": {"kind": "manifest", "manifest_path": str(reference_path)},
            }
        )

    model_path = tmp_path / "posterior.joblib"
    report_dir = tmp_path / "reports"
    config = AppConfig(
        path=tmp_path / "config.yaml",
        data={
            "posterior_calibration": {
                "sources": sources,
                "model_output_path": str(model_path),
                "report_dir": str(report_dir),
                "training_size": 32,
                "validation_size": 32,
                "max_pixels_per_class_per_image": 32,
                "max_iter": 12,
                "thresholds": [0.3, 0.5, 0.7],
                "min_component_area": 4,
            }
        },
    )

    manifest_path = train_score_to_mask_calibrator(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibrator = ScoreToMaskCalibrator(model_path)
    result = calibrator.predict(np.asarray(Image.open(tmp_path / "dataset_a" / "0_fused.png"), dtype=np.float32) / 255.0)

    assert manifest["model_sha256"]
    assert set(calibrator.bundle["development_datasets"]) == {"dataset_a", "dataset_b"}
    assert result.compact_mask.any()
    assert (report_dir / "leave_dataset_out_metrics.json").exists()
    assert "excluded its entire dataset" in (report_dir / "score_to_mask_calibration_report.md").read_text(encoding="utf-8")
