from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from iadgen_v2.auto_mask.contracts import PixelPosteriorResult
from iadgen_v2.auto_mask.posterior_calibration import (
    ScoreToMaskCalibrator,
    posterior_feature_image,
)
from iadgen_v2.config import AppConfig
from iadgen_v2.records import write_json


CANDIDATE_CALIBRATION_SCHEMA_VERSION = 1
CANDIDATE_FEATURE_NAMES = (
    "fused_mean",
    "fused_contrast",
    "high_evidence_fraction",
    "uncertainty_mean",
    "area_fraction",
    "component_count_log",
    "compactness",
    "elongation_log",
    "boundary_contact",
    "posterior_mean",
    "posterior_compact_precision",
    "posterior_compact_recall",
    "normal_boundary_burden",
    "semantic_prior_containment",
    "augmented_region_containment",
    "baseline_overlap_iou",
    "consensus_mean_iou",
    "consensus_max_iou",
    "is_baseline",
    "is_posterior",
    "is_component",
    "quantile_level",
)


@dataclass(frozen=True)
class CandidateCalibrationSource:
    dataset_id: str
    metadata_path: Path
    reference_kind: str
    reference_path: Path
    runtime_root: Path

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("candidate calibration dataset_id must be non-empty")
        if self.reference_kind not in {"mvtec", "manifest"}:
            raise ValueError("candidate calibration reference kind must be mvtec or manifest")


@dataclass(frozen=True)
class ArbitrationCandidate:
    mode: str
    mask: np.ndarray
    features: dict[str, float]

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("ArbitrationCandidate.mask must be 2D")
        if tuple(self.features) != CANDIDATE_FEATURE_NAMES:
            raise ValueError("ArbitrationCandidate feature contract mismatch")
        if not all(np.isfinite(value) for value in self.features.values()):
            raise ValueError("ArbitrationCandidate features must be finite")


@dataclass(frozen=True)
class CandidateArbitrationResult:
    selected: ArbitrationCandidate
    baseline: ArbitrationCandidate
    predictions: tuple[dict[str, float | str | bool], ...]
    override_applied: bool
    augmented_regions: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class _CalibrationSample:
    dataset_id: str
    category: str
    sample_id: str
    image_path: Path
    fused_path: Path
    uncertainty_path: Path | None
    baseline_mask_path: Path
    truth_path: Path
    normal_paths: tuple[Path, ...]
    semantic_regions: tuple[tuple[int, int, int, int], ...]


class CrossDatasetCandidateCalibrator:
    """Guarded candidate-level arbitrator with a mandatory legacy fallback."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        loaded = joblib.load(self.model_path)
        if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != CANDIDATE_CALIBRATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported candidate calibration bundle: {model_path}")
        if tuple(loaded.get("feature_names", ())) != CANDIDATE_FEATURE_NAMES:
            raise ValueError(f"Candidate calibration feature contract mismatch: {model_path}")
        self.bundle: dict[str, Any] = loaded

    def select(
        self,
        *,
        baseline_mask: np.ndarray,
        fused: np.ndarray,
        disagreement: np.ndarray,
        posterior_result: PixelPosteriorResult,
        normal_paths: tuple[Path, ...] = (),
        semantic_regions: tuple[tuple[int, int, int, int], ...] = (),
        min_component_area: int = 8,
    ) -> CandidateArbitrationResult:
        candidates, augmented_regions = build_arbitration_candidates(
            baseline_mask=baseline_mask,
            fused=fused,
            disagreement=disagreement,
            posterior_result=posterior_result,
            normal_paths=normal_paths,
            semantic_regions=semantic_regions,
            min_component_area=min_component_area,
            quantiles=tuple(float(value) for value in self.bundle.get("candidate_quantiles", (0.85, 0.90, 0.95, 0.975))),
            max_components=int(self.bundle.get("max_components_per_quantile", 4)),
            localization_envelope=bool(self.bundle.get("localization_envelope", False)),
        )
        prediction_rows = _predict_candidates(self.bundle, candidates)
        baseline = candidates[0]
        alternatives = [row for row in prediction_rows if not bool(row["is_baseline"])]
        best = max(alternatives, key=lambda row: float(row["gain_lower_bound"])) if alternatives else prediction_rows[0]
        override = bool(float(best["gain_lower_bound"]) > float(self.bundle.get("minimum_gain_lower_bound", 0.0)))
        selected_mode = str(best["mode"]) if override else baseline.mode
        selected = next(candidate for candidate in candidates if candidate.mode == selected_mode)
        return CandidateArbitrationResult(
            selected=selected,
            baseline=baseline,
            predictions=tuple(prediction_rows),
            override_applied=override,
            augmented_regions=augmented_regions,
        )


def build_arbitration_candidates(
    *,
    baseline_mask: np.ndarray,
    fused: np.ndarray,
    disagreement: np.ndarray,
    posterior_result: PixelPosteriorResult,
    normal_paths: tuple[Path, ...] = (),
    semantic_regions: tuple[tuple[int, int, int, int], ...] = (),
    min_component_area: int = 8,
    quantiles: tuple[float, ...] = (0.85, 0.90, 0.95, 0.975),
    max_components: int = 4,
    localization_envelope: bool = False,
) -> tuple[list[ArbitrationCandidate], tuple[tuple[int, int, int, int], ...]]:
    fused = _probability(fused)
    disagreement = _probability(disagreement)
    baseline_mask = np.asarray(baseline_mask, dtype=bool)
    if baseline_mask.shape != fused.shape or disagreement.shape != fused.shape:
        raise ValueError("Candidate calibration arrays must share one shape")
    if posterior_result.posterior.shape != fused.shape:
        raise ValueError("Candidate posterior must match fused evidence")

    region = semantic_regions[0] if semantic_regions else (0, 0, fused.shape[1], fused.shape[0])
    region_values = _region_values(fused, region)
    masks: list[tuple[str, np.ndarray, float, bool, bool]] = [
        ("v4_baseline_selected", baseline_mask, 0.0, False, False),
    ]
    if posterior_result.compact_mask.any():
        masks.append(("v4_posterior_compact", posterior_result.compact_mask, 0.0, True, False))
    for quantile in quantiles:
        threshold = max(0.05, float(np.quantile(region_values, quantile)) if region_values.size else 1.0)
        raw = _remove_small(fused >= threshold, min_component_area)
        masks.append((f"v4_fused_q{int(round(quantile * 1000)):03d}", raw, quantile, False, False))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), connectivity=8)
        indices = sorted(
            (index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_component_area),
            key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
            reverse=True,
        )[:max_components]
        for component_index, label_index in enumerate(indices, start=1):
            masks.append(
                (
                    f"v4_fused_q{int(round(quantile * 1000)):03d}_component_{component_index}",
                    labels == label_index,
                    quantile,
                    False,
                    True,
                )
            )
    unique: list[tuple[str, np.ndarray, float, bool, bool]] = []
    seen: set[bytes] = set()
    for mode, mask, quantile, is_posterior, is_component in masks:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        identity = np.packbits(mask).tobytes()
        if identity in seen:
            continue
        seen.add(identity)
        unique.append((mode, mask, quantile, is_posterior, is_component))
    if not unique or unique[0][0] != "v4_baseline_selected":
        raise ValueError("Candidate calibration requires a non-empty baseline mask")

    augmented_regions = evidence_augmented_regions(
        fused,
        semantic_regions,
        quantile=0.90,
        max_components=8,
        min_component_area=min_component_area,
        envelope=localization_envelope,
    )
    augmented_mask = regions_to_mask(fused.shape, augmented_regions)
    feature_image = posterior_feature_image(
        fused,
        semantic_regions=semantic_regions,
        normal_paths=normal_paths,
    )
    semantic_prior = feature_image[..., 5]
    normal_boundary = feature_image[..., 6]
    masks_only = [item[1] for item in unique]
    baseline = masks_only[0]
    output = []
    for (mode, mask, quantile, is_posterior, is_component), candidate_mask in zip(unique, masks_only):
        features = _candidate_features(
            mask=candidate_mask,
            masks=masks_only,
            baseline=baseline,
            fused=fused,
            disagreement=disagreement,
            posterior=posterior_result.posterior,
            posterior_compact=posterior_result.compact_mask,
            semantic_prior=semantic_prior,
            normal_boundary=normal_boundary,
            augmented_region=augmented_mask,
            is_baseline=mode == "v4_baseline_selected",
            is_posterior=is_posterior,
            is_component=is_component,
            quantile=quantile,
        )
        output.append(ArbitrationCandidate(mode=mode, mask=candidate_mask, features=features))
    return output, augmented_regions


def evidence_augmented_regions(
    fused: np.ndarray,
    semantic_regions: tuple[tuple[int, int, int, int], ...],
    *,
    quantile: float = 0.90,
    max_components: int = 8,
    min_component_area: int = 8,
    padding_fraction: float = 0.012,
    envelope: bool = False,
) -> tuple[tuple[int, int, int, int], ...]:
    fused = _probability(fused)
    height, width = fused.shape
    base_regions = semantic_regions or ((0, 0, width, height),)
    primary = base_regions[0]
    values = _region_values(fused, primary)
    threshold = max(0.05, float(np.quantile(values, quantile)) if values.size else 1.0)
    high = _remove_small(fused >= threshold, min_component_area)
    count, _, stats, _ = cv2.connectedComponentsWithStats(high.astype(np.uint8), connectivity=8)
    indices = sorted(
        (index for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_component_area),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:max_components]
    pad = max(1, int(round(max(height, width) * padding_fraction)))
    regions = [tuple(int(value) for value in region) for region in base_regions]
    for index in indices:
        left, top, box_width, box_height = (int(value) for value in stats[index, :4])
        regions.append(
            (
                max(0, left - pad),
                max(0, top - pad),
                min(width, left + box_width + pad),
                min(height, top + box_height + pad),
            )
        )
    unique = tuple(dict.fromkeys(regions))
    if not envelope or not unique:
        return unique
    return (
        (
            min(region[0] for region in unique),
            min(region[1] for region in unique),
            max(region[2] for region in unique),
            max(region[3] for region in unique),
        ),
    )


def regions_to_mask(shape: tuple[int, int], regions: tuple[tuple[int, int, int, int], ...]) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    height, width = shape
    for left, top, right, bottom in regions:
        output[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = True
    return output


def train_cross_dataset_candidate_calibrator(config: AppConfig) -> Path:
    settings = config.data.get("candidate_calibration", {})
    if not isinstance(settings, dict):
        raise ValueError("candidate_calibration must be a mapping")
    sources = _parse_sources(config, settings)
    if len(sources) < 2:
        raise ValueError("Cross-dataset candidate calibration requires at least two datasets")
    posterior_value = settings.get("posterior_model_path")
    if not posterior_value:
        raise ValueError("candidate_calibration.posterior_model_path is required")
    posterior = ScoreToMaskCalibrator(config.resolve_path(str(posterior_value)))
    size = int(settings.get("validation_size", 256))
    quantiles = tuple(float(value) for value in settings.get("candidate_quantiles", (0.85, 0.90, 0.95, 0.975)))
    max_components = int(settings.get("max_components_per_quantile", 4))
    min_component_area = int(settings.get("min_component_area", 8))
    localization_envelope = bool(settings.get("localization_envelope", False))
    rows, sample_rows = _build_training_rows(
        sources,
        posterior,
        size=size,
        quantiles=quantiles,
        max_components=max_components,
        min_component_area=min_component_area,
        localization_envelope=localization_envelope,
    )
    datasets = sorted({str(row["dataset_id"]) for row in rows})
    if len(datasets) < 2:
        raise ValueError("Candidate calibration found fewer than two usable datasets")

    model_settings = {
        "max_iter": int(settings.get("max_iter", 160)),
        "max_leaf_nodes": int(settings.get("max_leaf_nodes", 15)),
        "learning_rate": float(settings.get("learning_rate", 0.06)),
        "l2_regularization": float(settings.get("l2_regularization", 0.2)),
        "random_state": int(settings.get("seed", 20260810)),
    }
    held_out_rows = []
    fold_summaries = []
    for dataset_id in datasets:
        train_rows = [row for row in rows if row["dataset_id"] != dataset_id]
        test_rows = [row for row in rows if row["dataset_id"] == dataset_id]
        bundle = _fit_candidate_bundle(train_rows, model_settings)
        predicted = _predict_label_rows(bundle, test_rows)
        held_out_rows.extend(predicted)
        fold_summaries.append(_fold_summary(dataset_id, predicted, sample_rows))

    overall = _overall_summary(fold_summaries, held_out_rows, settings)
    final_bundle = _fit_candidate_bundle(rows, model_settings)
    final_bundle.update(
        {
            "schema_version": CANDIDATE_CALIBRATION_SCHEMA_VERSION,
            "feature_names": list(CANDIDATE_FEATURE_NAMES),
            "selection_strategy": "baseline_guarded_candidate_gain",
            "candidate_quantiles": list(quantiles),
            "max_components_per_quantile": max_components,
            "localization_envelope": localization_envelope,
            "minimum_gain_lower_bound": float(settings.get("minimum_gain_lower_bound", 0.0)),
            "development_datasets": datasets,
            "training_candidate_count": len(rows),
            "training_image_count": len(sample_rows),
            "leave_dataset_out_summary": overall,
        }
    )
    output_value = settings.get("model_output_path") or (
        config.output_dir / "auto_masks" / "selector" / "cross_dataset_candidate_calibrator.joblib"
    )
    output_path = config.resolve_path(str(output_value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_bundle, output_path)

    report_value = settings.get("report_dir") or (config.report_dir / "candidate_calibration")
    report_dir = config.resolve_path(str(report_value))
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "leave_dataset_out_candidate_metrics.json"
    write_json(
        metrics_path,
        {
            "overall": overall,
            "folds": fold_summaries,
            "candidate_predictions": held_out_rows,
            "samples": sample_rows,
        },
    )
    report_path = report_dir / "candidate_calibration_report.md"
    report_path.write_text(_candidate_report(output_path, overall, fold_summaries), encoding="utf-8")
    manifest_path = report_dir / "candidate_calibration_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "model_path": str(output_path),
            "model_sha256": _sha256_file(output_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256_file(metrics_path),
            "report_path": str(report_path),
            "report_sha256": _sha256_file(report_path),
            "promotion_passed": bool(overall["promotion_passed"]),
            "sources": [
                {
                    "dataset_id": source.dataset_id,
                    "metadata_path": str(source.metadata_path),
                    "metadata_sha256": _sha256_file(source.metadata_path),
                    "reference_kind": source.reference_kind,
                    "reference_path": str(source.reference_path),
                    "reference_sha256": _sha256_file(source.reference_path) if source.reference_path.is_file() else None,
                    "runtime_root": str(source.runtime_root),
                }
                for source in sources
            ],
        },
    )
    return manifest_path


def _candidate_features(
    *,
    mask: np.ndarray,
    masks: list[np.ndarray],
    baseline: np.ndarray,
    fused: np.ndarray,
    disagreement: np.ndarray,
    posterior: np.ndarray,
    posterior_compact: np.ndarray,
    semantic_prior: np.ndarray,
    normal_boundary: np.ndarray,
    augmented_region: np.ndarray,
    is_baseline: bool,
    is_posterior: bool,
    is_component: bool,
    quantile: float,
) -> dict[str, float]:
    area = int(mask.sum())
    outside = ~mask
    fused_mean = float(fused[mask].mean())
    high_threshold = float(np.quantile(fused, 0.90))
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    ys, xs = np.where(mask)
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    pair_ious = [_mask_iou(mask, other) for other in masks if other is not mask]
    compact_overlap = int((mask & posterior_compact).sum())
    features = {
        "fused_mean": fused_mean,
        "fused_contrast": fused_mean - (float(fused[outside].mean()) if outside.any() else 0.0),
        "high_evidence_fraction": float((fused[mask] >= high_threshold).mean()),
        "uncertainty_mean": float(disagreement[mask].mean()),
        "area_fraction": float(area / mask.size),
        "component_count_log": float(np.log1p(max(0, count - 1))),
        "compactness": float(area / max(1, box_width * box_height)),
        "elongation_log": float(np.log1p(max(box_width / max(1, box_height), box_height / max(1, box_width)))),
        "boundary_contact": float(np.mean([mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()])),
        "posterior_mean": float(posterior[mask].mean()),
        "posterior_compact_precision": float(compact_overlap / max(1, area)),
        "posterior_compact_recall": float(compact_overlap / max(1, int(posterior_compact.sum()))),
        "normal_boundary_burden": float(normal_boundary[mask].mean()),
        "semantic_prior_containment": float(semantic_prior[mask].mean()),
        "augmented_region_containment": float(augmented_region[mask].mean()),
        "baseline_overlap_iou": _mask_iou(mask, baseline),
        "consensus_mean_iou": float(np.mean(pair_ious)) if pair_ious else 1.0,
        "consensus_max_iou": float(np.max(pair_ious)) if pair_ious else 1.0,
        "is_baseline": float(is_baseline),
        "is_posterior": float(is_posterior),
        "is_component": float(is_component),
        "quantile_level": float(quantile),
    }
    return {name: float(features[name]) for name in CANDIDATE_FEATURE_NAMES}


def _build_training_rows(
    sources: list[CandidateCalibrationSource],
    posterior: ScoreToMaskCalibrator,
    *,
    size: int,
    quantiles: tuple[float, ...],
    max_components: int,
    min_component_area: int,
    localization_envelope: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    sample_rows = []
    for source in sources:
        for sample in _source_samples(source):
            fused = _load_probability(sample.fused_path, size)
            baseline = _load_mask(sample.baseline_mask_path, fused.shape)
            truth = _load_mask(sample.truth_path, fused.shape)
            disagreement = (
                _load_probability(sample.uncertainty_path, size)
                if sample.uncertainty_path is not None and sample.uncertainty_path.is_file()
                else np.zeros_like(fused)
            )
            regions = _resize_regions(sample.semantic_regions, sample.image_path, fused.shape)
            posterior_result = posterior.predict(
                fused,
                min_component_area=min_component_area,
                normal_paths=sample.normal_paths,
                semantic_regions=regions,
            )
            candidates, augmented_regions = build_arbitration_candidates(
                baseline_mask=baseline,
                fused=fused,
                disagreement=disagreement,
                posterior_result=posterior_result,
                normal_paths=sample.normal_paths,
                semantic_regions=regions,
                min_component_area=min_component_area,
                quantiles=quantiles,
                max_components=max_components,
                localization_envelope=localization_envelope,
            )
            region_mask = regions_to_mask(fused.shape, augmented_regions)
            sample_rows.append(
                {
                    "dataset_id": sample.dataset_id,
                    "category": sample.category,
                    "sample_id": sample.sample_id,
                    "search_region_recall": float((region_mask & truth).sum() / max(1, int(truth.sum()))),
                    "search_region_area_fraction": float(region_mask.mean()),
                }
            )
            actual = int(truth.sum())
            for candidate in candidates:
                predicted = int(candidate.mask.sum())
                intersection = int((candidate.mask & truth).sum())
                union = predicted + actual - intersection
                rows.append(
                    {
                        "dataset_id": sample.dataset_id,
                        "category": sample.category,
                        "sample_id": sample.sample_id,
                        "mode": candidate.mode,
                        "features": candidate.features,
                        "iou": intersection / max(1, union),
                        "dice": 2 * intersection / max(1, predicted + actual),
                        "precision": intersection / max(1, predicted),
                        "recall": intersection / max(1, actual),
                        "predicted_pixels": predicted,
                        "actual_pixels": actual,
                        "is_baseline": candidate.mode == "v4_baseline_selected",
                    }
                )
    if not rows:
        raise ValueError("Candidate calibration found no usable candidate rows")
    return rows, sample_rows


def _fit_candidate_bundle(rows: list[dict[str, Any]], model_settings: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < 20:
        raise ValueError("Candidate calibration requires at least 20 rows")
    features = np.asarray(
        [[float(row["features"][name]) for name in CANDIDATE_FEATURE_NAMES] for row in rows],
        dtype=np.float32,
    )
    groups = np.asarray([str(row["category"]) for row in rows])
    weights = _balanced_row_weights(rows)
    bundle: dict[str, Any] = {
        "schema_version": CANDIDATE_CALIBRATION_SCHEMA_VERSION,
        "feature_names": list(CANDIDATE_FEATURE_NAMES),
    }
    for target_name in ("iou", "precision", "recall"):
        target = np.asarray([float(row[target_name]) for row in rows], dtype=np.float32)
        oof = _group_oof_regression(features, target, groups, weights, model_settings)
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof, target, sample_weight=weights)
        calibrated = np.asarray(calibrator.predict(oof), dtype=np.float32)
        residual = float(np.quantile(np.maximum(0.0, calibrated - target), 0.90))
        model = HistGradientBoostingRegressor(**model_settings).fit(features, target, sample_weight=weights)
        bundle[f"{target_name}_model"] = model
        bundle[f"{target_name}_calibrator"] = calibrator
        bundle[f"{target_name}_residual_q90"] = residual

    pair_features, pair_targets, pair_groups, pair_weights = _gain_training_arrays(rows)
    pair_oof = _group_oof_regression(pair_features, pair_targets, pair_groups, pair_weights, model_settings)
    gain_model = HistGradientBoostingRegressor(**model_settings).fit(
        pair_features,
        pair_targets,
        sample_weight=pair_weights,
    )
    bundle["gain_model"] = gain_model
    bundle["gain_residual_q90"] = float(np.quantile(np.maximum(0.0, pair_oof - pair_targets), 0.90))
    return bundle


def _predict_candidates(bundle: dict[str, Any], candidates: list[ArbitrationCandidate]) -> list[dict[str, float | str | bool]]:
    feature_matrix = np.asarray(
        [[candidate.features[name] for name in CANDIDATE_FEATURE_NAMES] for candidate in candidates],
        dtype=np.float32,
    )
    predicted: dict[str, np.ndarray] = {}
    for target_name in ("iou", "precision", "recall"):
        raw = bundle[f"{target_name}_model"].predict(feature_matrix)
        predicted[target_name] = np.clip(bundle[f"{target_name}_calibrator"].predict(raw), 0.0, 1.0)
    baseline_features = feature_matrix[0]
    pair_features = np.concatenate(
        (
            feature_matrix,
            np.repeat(baseline_features[None, :], len(candidates), axis=0),
            feature_matrix - baseline_features[None, :],
        ),
        axis=1,
    )
    gain = bundle["gain_model"].predict(pair_features)
    gain_lower = gain - float(bundle["gain_residual_q90"])
    iou_lower = predicted["iou"] - float(bundle["iou_residual_q90"])
    return [
        {
            "mode": candidate.mode,
            "expected_iou": float(predicted["iou"][index]),
            "expected_precision": float(predicted["precision"][index]),
            "expected_recall": float(predicted["recall"][index]),
            "conformal_iou_lower_bound": float(max(0.0, iou_lower[index])),
            "expected_gain": float(gain[index]),
            "gain_lower_bound": float(gain_lower[index]),
            "is_baseline": candidate.mode == "v4_baseline_selected",
        }
        for index, candidate in enumerate(candidates)
    ]


def _predict_label_rows(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for sample_id in sorted({str(row["sample_id"]) for row in rows}):
        group = [row for row in rows if row["sample_id"] == sample_id]
        candidates = [
            ArbitrationCandidate(
                mode=str(row["mode"]),
                mask=np.ones((1, 1), dtype=bool),
                features={name: float(row["features"][name]) for name in CANDIDATE_FEATURE_NAMES},
            )
            for row in group
        ]
        predictions = _predict_candidates(bundle, candidates)
        for row, prediction in zip(group, predictions):
            output.append({**row, **prediction})
    return output


def _fold_summary(dataset_id: str, rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = []
    baseline = []
    oracle = []
    for sample_id in sorted({str(row["sample_id"]) for row in rows}):
        group = [row for row in rows if row["sample_id"] == sample_id]
        base = next(row for row in group if bool(row["is_baseline"]))
        alternatives = [row for row in group if not bool(row["is_baseline"])]
        best = max(alternatives, key=lambda row: float(row["gain_lower_bound"])) if alternatives else base
        chosen = best if float(best["gain_lower_bound"]) > 0.0 else base
        selected.append(chosen)
        baseline.append(base)
        oracle.append(max(group, key=lambda row: float(row["iou"])))
    actual_iou = np.asarray([float(row["iou"]) for row in rows])
    expected_iou = np.asarray([float(row["expected_iou"]) for row in rows])
    if float(expected_iou.std()) < 1e-12 or float(actual_iou.std()) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(spearmanr(expected_iou, actual_iou).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0
    source_samples = [row for row in sample_rows if row["dataset_id"] == dataset_id]
    return {
        "dataset_id": dataset_id,
        "images": len(selected),
        "candidate_count": len(rows),
        "baseline_dice": float(np.mean([float(row["dice"]) for row in baseline])),
        "selected_dice": float(np.mean([float(row["dice"]) for row in selected])),
        "oracle_dice": float(np.mean([float(row["dice"]) for row in oracle])),
        "selected_precision": float(np.mean([float(row["precision"]) for row in selected])),
        "selected_recall": float(np.mean([float(row["recall"]) for row in selected])),
        "selected_area_ratio": float(
            sum(int(row["predicted_pixels"]) for row in selected)
            / max(1, sum(int(row["actual_pixels"]) for row in selected))
        ),
        "override_rate": float(np.mean([not bool(row["is_baseline"]) for row in selected])),
        "expected_iou_spearman": correlation,
        "expected_iou_mae": float(np.mean(np.abs(expected_iou - actual_iou))),
        "conformal_coverage": float(
            np.mean([float(row["conformal_iou_lower_bound"]) <= float(row["iou"]) for row in rows])
        ),
        "search_region_recall": float(np.mean([float(row["search_region_recall"]) for row in source_samples])),
        "search_region_area_fraction": float(np.mean([float(row["search_region_area_fraction"]) for row in source_samples])),
    }


def _overall_summary(
    folds: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    mean = lambda key: float(np.mean([float(fold[key]) for fold in folds]))
    max_regression = float(settings.get("max_dataset_dice_regression", 0.02))
    coverage_min = float(settings.get("target_coverage_min", 0.85))
    coverage_max = float(settings.get("target_coverage_max", 0.95))
    gates = {
        "expected_iou_spearman": all(
            float(fold["expected_iou_spearman"]) >= float(settings.get("target_spearman", 0.40)) for fold in folds
        ),
        "expected_iou_mae": all(
            float(fold["expected_iou_mae"]) <= float(settings.get("target_mae", 0.12)) for fold in folds
        ),
        "conformal_coverage": all(
            coverage_min <= float(fold["conformal_coverage"]) <= coverage_max for fold in folds
        ),
        "search_region_recall": all(
            float(fold["search_region_recall"]) >= float(settings.get("target_search_recall", 0.85)) for fold in folds
        ),
        "dataset_non_regression": all(
            float(fold["selected_dice"]) >= float(fold["baseline_dice"]) - max_regression for fold in folds
        ),
        "utility_gain": (
            mean("selected_dice") - mean("baseline_dice")
            >= float(settings.get("minimum_macro_dice_gain", 0.01))
        ),
    }
    return {
        "dataset_macro_baseline_dice": mean("baseline_dice"),
        "dataset_macro_selected_dice": mean("selected_dice"),
        "dataset_macro_oracle_dice": mean("oracle_dice"),
        "dataset_macro_precision": mean("selected_precision"),
        "dataset_macro_recall": mean("selected_recall"),
        "dataset_macro_area_ratio": mean("selected_area_ratio"),
        "dataset_macro_expected_iou_spearman": mean("expected_iou_spearman"),
        "dataset_macro_expected_iou_mae": mean("expected_iou_mae"),
        "dataset_macro_conformal_coverage": mean("conformal_coverage"),
        "dataset_macro_search_region_recall": mean("search_region_recall"),
        "dataset_macro_search_region_area_fraction": mean("search_region_area_fraction"),
        "gates": gates,
        "promotion_passed": all(gates.values()),
        "held_out_candidate_count": len(rows),
    }


def _parse_sources(config: AppConfig, settings: dict[str, Any]) -> list[CandidateCalibrationSource]:
    raw_sources = settings.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("candidate_calibration.sources must be a list")
    output = []
    for item in raw_sources:
        if not isinstance(item, dict) or not isinstance(item.get("reference"), dict):
            raise ValueError("Each candidate calibration source requires a reference mapping")
        reference = item["reference"]
        output.append(
            CandidateCalibrationSource(
                dataset_id=str(item.get("dataset_id", "")),
                metadata_path=config.resolve_path(str(item["metadata_path"])),
                reference_kind=str(reference.get("kind", "")),
                reference_path=config.resolve_path(str(reference.get("root") or reference.get("manifest_path"))),
                runtime_root=config.resolve_path(str(item.get("runtime_root", ""))),
            )
        )
    if len({source.dataset_id for source in output}) != len(output):
        raise ValueError("candidate calibration dataset_id values must be unique")
    for source in output:
        if not source.metadata_path.is_file():
            raise FileNotFoundError(f"Missing candidate metadata: {source.metadata_path}")
        if not source.reference_path.exists():
            raise FileNotFoundError(f"Missing candidate reference: {source.reference_path}")
        if not source.runtime_root.is_dir():
            raise FileNotFoundError(f"Missing candidate runtime root: {source.runtime_root}")
    return output


def _source_samples(source: CandidateCalibrationSource) -> list[_CalibrationSample]:
    reference_index: dict[tuple[str, str, str], Path] = {}
    if source.reference_kind == "manifest":
        for row in _read_jsonl(source.reference_path):
            key = (str(row["category"]), str(row["defect_type"]), Path(str(row["image_path"])).stem)
            reference_index[key] = Path(str(row["official_mask_path"]))
    normal_cache: dict[str, tuple[Path, ...]] = {}
    samples = []
    for row in _read_jsonl(source.metadata_path):
        category = str(row["category"])
        defect_type = str(row["defect_type"])
        image_path = Path(str(row["image_path"]))
        stem = image_path.stem
        settings = row.get("settings", {})
        parameters = settings.get("mask_parameters", {})
        fused_path = Path(str(parameters.get("fused_evidence_path", "")))
        uncertainty_value = row.get("uncertainty_mask_path") or row.get("mask_variant_paths", {}).get("uncertainty_map")
        uncertainty_path = Path(str(uncertainty_value)) if uncertainty_value else None
        baseline_path = Path(str(row.get("eval_mask_path") or row.get("refined_mask_path") or ""))
        truth_path = (
            source.reference_path / category / "ground_truth" / defect_type / f"{stem}_mask.png"
            if source.reference_kind == "mvtec"
            else reference_index.get((category, defect_type, stem), Path())
        )
        if category not in normal_cache:
            normal_cache[category] = tuple(
                sorted(
                    path
                    for path in (source.runtime_root / category / "train" / "good").rglob("*")
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                )[:6]
            )
        region = row.get("region_xyxy", [0, 0, 0, 0])
        regions = (tuple(int(value) for value in region),) if isinstance(region, (list, tuple)) and len(region) == 4 else ()
        if fused_path.is_file() and baseline_path.is_file() and truth_path.is_file() and image_path.is_file():
            samples.append(
                _CalibrationSample(
                    dataset_id=source.dataset_id,
                    category=category,
                    sample_id=f"{source.dataset_id}:{category}/{defect_type}/{stem}",
                    image_path=image_path,
                    fused_path=fused_path,
                    uncertainty_path=uncertainty_path,
                    baseline_mask_path=baseline_path,
                    truth_path=truth_path,
                    normal_paths=normal_cache[category],
                    semantic_regions=regions,
                )
            )
    return samples


def _group_oof_regression(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    model_settings: dict[str, Any],
) -> np.ndarray:
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        model = HistGradientBoostingRegressor(**model_settings).fit(features, target, sample_weight=weights)
        return np.asarray(model.predict(features), dtype=np.float32)
    output = np.zeros(len(target), dtype=np.float32)
    for group in unique_groups:
        held = groups == group
        train = ~held
        model = HistGradientBoostingRegressor(**model_settings).fit(
            features[train],
            target[train],
            sample_weight=weights[train],
        )
        output[held] = model.predict(features[held])
    return output


def _gain_training_arrays(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    groups = []
    weights = []
    sample_counts: dict[str, int] = {}
    for sample_id in {str(row["sample_id"]) for row in rows}:
        sample_counts[sample_id] = sum(str(row["sample_id"]) == sample_id for row in rows) - 1
    dataset_counts = {dataset: len({str(row["sample_id"]) for row in rows if row["dataset_id"] == dataset}) for dataset in {str(row["dataset_id"]) for row in rows}}
    for sample_id in sorted(sample_counts):
        group_rows = [row for row in rows if row["sample_id"] == sample_id]
        baseline = next(row for row in group_rows if bool(row["is_baseline"]))
        baseline_features = np.asarray([baseline["features"][name] for name in CANDIDATE_FEATURE_NAMES], dtype=np.float32)
        for row in group_rows:
            if bool(row["is_baseline"]):
                continue
            candidate_features = np.asarray([row["features"][name] for name in CANDIDATE_FEATURE_NAMES], dtype=np.float32)
            features.append(np.concatenate((candidate_features, baseline_features, candidate_features - baseline_features)))
            targets.append(float(row["iou"]) - float(baseline["iou"]))
            groups.append(str(row["category"]))
            weights.append(1.0 / max(1, dataset_counts[str(row["dataset_id"])]) / max(1, sample_counts[sample_id]))
    weights_array = np.asarray(weights, dtype=np.float32)
    weights_array /= max(1e-8, float(weights_array.mean()))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(groups),
        weights_array,
    )


def _balanced_row_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    dataset_samples = {
        dataset: len({str(row["sample_id"]) for row in rows if row["dataset_id"] == dataset})
        for dataset in {str(row["dataset_id"]) for row in rows}
    }
    sample_counts = {
        sample_id: sum(str(row["sample_id"]) == sample_id for row in rows)
        for sample_id in {str(row["sample_id"]) for row in rows}
    }
    weights = np.asarray(
        [
            1.0 / max(1, dataset_samples[str(row["dataset_id"])]) / max(1, sample_counts[str(row["sample_id"])])
            for row in rows
        ],
        dtype=np.float32,
    )
    weights /= max(1e-8, float(weights.mean()))
    return weights


def _candidate_report(model_path: Path, overall: dict[str, Any], folds: list[dict[str, Any]]) -> str:
    lines = [
        "# V4.2 Cross-Dataset Candidate Calibration Report",
        "",
        "Every metric row is predicted by a model that excluded the complete source dataset. Official masks label retained candidates only after generation.",
        "",
        f"- Development candidate model: `{model_path}`",
        f"- Promotion passed: `{bool(overall['promotion_passed'])}`",
        "",
        "| Dataset | Images | Baseline Dice | Selected Dice | Delta | Oracle Dice | Precision | Recall | Area Ratio | Override Rate | Spearman | MAE | Coverage | Search Recall | Search Area |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in folds:
        lines.append(
            f"| {fold['dataset_id']} | {fold['images']} | `{fold['baseline_dice']:.4f}` | "
            f"`{fold['selected_dice']:.4f}` | `{fold['selected_dice'] - fold['baseline_dice']:+.4f}` | "
            f"`{fold['oracle_dice']:.4f}` | `{fold['selected_precision']:.4f}` | `{fold['selected_recall']:.4f}` | "
            f"`{fold['selected_area_ratio']:.4f}` | `{fold['override_rate']:.4f}` | "
            f"`{fold['expected_iou_spearman']:.4f}` | `{fold['expected_iou_mae']:.4f}` | "
            f"`{fold['conformal_coverage']:.4f}` | `{fold['search_region_recall']:.4f}` | "
            f"`{fold['search_region_area_fraction']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Promotion Gates",
            "",
            *[f"- {name}: `{passed}`" for name, passed in overall["gates"].items()],
            "",
            "This is exposed multi-dataset development evidence, not a locked-generalization claim.",
            "",
        ]
    )
    return "\n".join(lines)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int((first | second).sum())
    return float((first & second).sum() / union) if union else 1.0


def _remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    output = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
            output[labels == index] = True
    return output


def _region_values(values: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = region
    crop = values[max(0, top) : min(values.shape[0], bottom), max(0, left) : min(values.shape[1], right)]
    return crop[np.isfinite(crop)]


def _resize_regions(
    regions: tuple[tuple[int, int, int, int], ...],
    image_path: Path,
    shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    if not regions:
        return ()
    with Image.open(image_path) as image:
        source_width, source_height = image.size
    height, width = shape
    scale_x = width / max(1, source_width)
    scale_y = height / max(1, source_height)
    return tuple(
        (
            int(round(left * scale_x)),
            int(round(top * scale_y)),
            int(round(right * scale_x)),
            int(round(bottom * scale_y)),
        )
        for left, top, right, bottom in regions
    )


def _load_probability(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size > 0:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 0


def _probability(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    output[~np.isfinite(output)] = 0.0
    return np.clip(output, 0.0, 1.0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
