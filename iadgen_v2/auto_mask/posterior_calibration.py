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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from iadgen_v2.auto_mask.contracts import PixelPosteriorResult
from iadgen_v2.config import AppConfig
from iadgen_v2.records import write_json


POSTERIOR_BUNDLE_SCHEMA_VERSION = 1
POSTERIOR_FEATURE_NAMES = (
    "robust_fused_score",
    "within_image_rank",
    "local_rank_mean",
    "positive_local_rank_contrast",
    "rank_gradient_magnitude",
    "semantic_spatial_prior",
    "stable_normal_boundary",
)
_NORMAL_BOUNDARY_CACHE: dict[tuple[tuple[str, ...], tuple[int, int]], np.ndarray] = {}


@dataclass(frozen=True)
class PosteriorCalibrationSource:
    dataset_id: str
    metadata_path: Path
    reference_kind: str
    reference_path: Path
    runtime_root: Path

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("posterior calibration dataset_id must be non-empty")
        if self.reference_kind not in {"mvtec", "manifest"}:
            raise ValueError("posterior calibration reference kind must be mvtec or manifest")


@dataclass(frozen=True)
class _CalibrationImage:
    dataset_id: str
    sample_id: str
    fused_path: Path
    truth_path: Path
    baseline_mask_path: Path
    image_path: Path
    normal_paths: tuple[Path, ...]
    semantic_regions: tuple[tuple[int, int, int, int], ...]


class ScoreToMaskCalibrator:
    """Frozen category-agnostic mapping from fused evidence to a compact mask."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path.resolve()
        loaded = joblib.load(self.model_path)
        if not isinstance(loaded, dict) or int(loaded.get("schema_version", 0)) != POSTERIOR_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported score-to-mask posterior bundle: {model_path}")
        if tuple(loaded.get("feature_names", ())) != POSTERIOR_FEATURE_NAMES:
            raise ValueError(f"Posterior feature contract mismatch: {model_path}")
        self.bundle: dict[str, Any] = loaded

    def should_override(self, baseline_mask: np.ndarray, result: PixelPosteriorResult) -> bool:
        baseline_area = int(np.asarray(baseline_mask, dtype=bool).sum())
        compact_area = int(result.compact_mask.sum())
        ratio = float(self.bundle.get("override_area_ratio", 0.0))
        return compact_area > 0 and ratio > 0.0 and baseline_area > ratio * compact_area

    def predict(
        self,
        fused: np.ndarray,
        *,
        min_component_area: int = 8,
        normal_paths: tuple[Path, ...] = (),
        semantic_regions: tuple[tuple[int, int, int, int], ...] = (),
    ) -> PixelPosteriorResult:
        fused = _as_probability(fused)
        features = posterior_feature_image(
            fused,
            semantic_regions=semantic_regions,
            normal_paths=normal_paths,
        )
        raw = self.bundle["model"].predict_proba(features.reshape(-1, features.shape[-1]))[:, 1]
        calibrator = self.bundle.get("calibrator")
        posterior = calibrator.predict(raw) if calibrator is not None else raw
        posterior = np.asarray(posterior, dtype=np.float32).reshape(fused.shape)
        compact = dict(self.bundle.get("compact", {}))
        threshold = float(self.bundle["threshold"])
        mask, diagnostics = extract_calibrated_components(
            posterior,
            feature_image=features,
            threshold=threshold,
            compact=compact,
            min_component_area=min_component_area,
        )
        return PixelPosteriorResult(
            posterior=posterior,
            compact_mask=mask,
            threshold=threshold,
            diagnostics={
                **diagnostics,
                "model_path": str(self.model_path),
                "model_sha256": _sha256_file(self.model_path),
            },
        )


def posterior_feature_image(
    fused: np.ndarray,
    *,
    semantic_regions: tuple[tuple[int, int, int, int], ...] = (),
    normal_paths: tuple[Path, ...] = (),
) -> np.ndarray:
    fused = _as_probability(fused)
    median = float(np.median(fused))
    mad = max(1e-6, float(np.median(np.abs(fused - median))))
    robust_z = np.clip((fused - median) / (1.4826 * mad), -8.0, 8.0)
    robust = 1.0 / (1.0 + np.exp(-(robust_z - 1.5)))
    quantiles = np.quantile(fused, np.linspace(0.0, 1.0, 257))
    rank = np.searchsorted(quantiles, fused, side="right").astype(np.float32) / float(len(quantiles))
    local_mean = cv2.GaussianBlur(rank, (0, 0), sigmaX=3.0, sigmaY=3.0)
    positive_contrast = np.clip(rank - local_mean, 0.0, 1.0)
    gradient_x = cv2.Sobel(rank, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(rank, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    gradient /= max(1e-6, float(np.quantile(gradient, 0.99)))
    return np.stack(
        (
            robust,
            rank,
            local_mean,
            positive_contrast,
            np.clip(gradient, 0.0, 1.0),
            _semantic_prior(fused.shape, semantic_regions),
            stable_normal_boundary(fused.shape, normal_paths),
        ),
        axis=-1,
    ).astype(np.float32)


def stable_normal_boundary(shape: tuple[int, int], normal_paths: tuple[Path, ...]) -> np.ndarray:
    selected = tuple(str(path.resolve()) for path in normal_paths[:6] if path.is_file())
    if not selected:
        return np.zeros(shape, dtype=np.float32)
    key = (selected, shape)
    cached = _NORMAL_BOUNDARY_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    height, width = shape
    gradients = []
    for path_value in selected:
        gray = np.asarray(
            Image.open(path_value).convert("L").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(np.sqrt(gx * gx + gy * gy))
    boundary = cv2.GaussianBlur(np.median(np.stack(gradients), axis=0), (0, 0), sigmaX=1.2)
    boundary /= max(1e-6, float(np.quantile(boundary, 0.99)))
    output = np.clip(boundary, 0.0, 1.0).astype(np.float32)
    _NORMAL_BOUNDARY_CACHE[key] = output
    return output.copy()


def extract_compact_components(
    posterior: np.ndarray,
    *,
    threshold: float,
    seed_threshold: float,
    seed_quantile: float = 0.995,
    min_component_area: int,
    max_components: int,
    max_area_fraction: float,
    growth_factor: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    posterior = _as_probability(posterior)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    seed_quantile = float(np.clip(seed_quantile, 0.50, 1.0))
    relative_seed = float(np.quantile(posterior, seed_quantile))
    seed_threshold = float(np.clip(max(threshold + 1e-6, min(seed_threshold, relative_seed)), 0.0, 1.0))
    raw = posterior >= threshold
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    seeds = posterior >= seed_threshold
    count, labels = cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)
    components: list[tuple[float, np.ndarray]] = []
    for index in range(1, count):
        component = labels == index
        area = int(component.sum())
        if area < max(1, min_component_area) or not bool((component & seeds).any()):
            continue
        values = posterior[component]
        score = float(values.mean() * values.max() * np.sqrt(area))
        components.append((score, component))
    components.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(raw, dtype=bool)
    for _, component in components[: max(1, max_components)]:
        selected |= component

    image_area = posterior.size
    seed_area = int((selected & seeds).sum())
    hard_cap = max(min_component_area, int(round(max_area_fraction * image_area)))
    evidence_budget = max(min_component_area, int(round(max(1, seed_area) * max(1.0, growth_factor))))
    budget = min(hard_cap, evidence_budget)
    if int(selected.sum()) > budget:
        selected_indices = np.flatnonzero(selected)
        order = np.argsort(-posterior.reshape(-1)[selected_indices], kind="stable")
        limited = np.zeros_like(selected, dtype=bool)
        limited.reshape(-1)[selected_indices[order[:budget]]] = True
        selected = limited
        selected = _keep_seeded_components(selected, seeds, min_component_area, max_components)

    return selected, {
        "raw_area_fraction": float(raw.mean()),
        "seed_area_fraction": float(seeds.mean()),
        "compact_area_fraction": float(selected.mean()),
        "component_count": int(cv2.connectedComponents(selected.astype(np.uint8), connectivity=8)[0] - 1),
        "area_budget_pixels": int(budget),
        "effective_seed_threshold": seed_threshold,
        "seed_quantile": seed_quantile,
    }


def extract_calibrated_components(
    posterior: np.ndarray,
    *,
    feature_image: np.ndarray,
    threshold: float,
    compact: dict[str, Any],
    min_component_area: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    effective_min_area = max(min_component_area, int(compact.get("min_component_area", 8)))
    effective_threshold = float(threshold)
    if str(compact.get("threshold_mode", "fixed")) == "otsu_floor":
        otsu_threshold, _ = cv2.threshold(
            np.uint8(np.clip(posterior, 0.0, 1.0) * 255),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        effective_threshold = max(effective_threshold, float(otsu_threshold) / 255.0)
    mask, diagnostics = extract_compact_components(
        posterior,
        threshold=effective_threshold,
        seed_threshold=float(compact.get("seed_threshold", max(0.65, effective_threshold + 0.15))),
        seed_quantile=float(compact.get("seed_quantile", 0.995)),
        min_component_area=effective_min_area,
        max_components=int(compact.get("max_components", 8)),
        max_area_fraction=float(compact.get("max_area_fraction", 0.20)),
        growth_factor=float(compact.get("growth_factor", 4.0)),
    )
    minimum_fraction = float(compact.get("minimum_output_area_fraction", 0.001))
    if mask.any() and float(mask.mean()) >= minimum_fraction:
        return mask, {
            **diagnostics,
            "fallback": "none",
            "effective_threshold": effective_threshold,
        }

    rank_index = POSTERIOR_FEATURE_NAMES.index("within_image_rank")
    rank = feature_image[..., rank_index]
    fallback, fallback_diagnostics = extract_compact_components(
        rank,
        threshold=float(compact.get("fallback_rank_threshold", 0.95)),
        seed_threshold=1.0,
        seed_quantile=float(compact.get("fallback_seed_quantile", 0.999)),
        min_component_area=effective_min_area,
        max_components=int(compact.get("fallback_max_components", 3)),
        max_area_fraction=float(compact.get("max_area_fraction", 0.20)),
        growth_factor=float(compact.get("fallback_growth_factor", 4.0)),
    )
    if int(fallback.sum()) <= int(mask.sum()):
        return mask, {**diagnostics, "fallback": "rank_candidate_not_larger"}
    return fallback, {
        **fallback_diagnostics,
        "fallback": "invariant_rank_compact",
        "learned_compact_area_fraction": float(mask.mean()),
        "effective_threshold": effective_threshold,
    }


def train_score_to_mask_calibrator(config: AppConfig) -> Path:
    settings = config.data.get("posterior_calibration", {})
    if not isinstance(settings, dict):
        raise ValueError("posterior_calibration must be a mapping")
    sources = _parse_sources(config, settings)
    if len(sources) < 2:
        raise ValueError("Cross-dataset posterior calibration requires at least two datasets")
    images = [image for source in sources for image in _source_images(source)]
    if not images:
        raise ValueError("Posterior calibration found no valid fused-evidence images")

    seed = int(settings.get("seed", 20260810))
    max_per_class = int(settings.get("max_pixels_per_class_per_image", 256))
    train_size = int(settings.get("training_size", 256))
    features, labels, weights, dataset_ids = _sample_training_pixels(
        images,
        seed=seed,
        max_per_class=max_per_class,
        size=train_size,
    )
    unique_datasets = sorted(set(dataset_ids.tolist()))
    if len(unique_datasets) < 2:
        raise ValueError("Posterior calibration requires samples from at least two datasets")

    model_settings = {
        "max_iter": int(settings.get("max_iter", 120)),
        "max_leaf_nodes": int(settings.get("max_leaf_nodes", 15)),
        "learning_rate": float(settings.get("learning_rate", 0.06)),
        "l2_regularization": float(settings.get("l2_regularization", 0.2)),
        "random_state": seed,
    }
    oof_raw = np.zeros(len(labels), dtype=np.float32)
    fold_models: dict[str, HistGradientBoostingClassifier] = {}
    for dataset_id in unique_datasets:
        train = dataset_ids != dataset_id
        held_out = ~train
        model = HistGradientBoostingClassifier(**model_settings)
        model.fit(features[train], labels[train], sample_weight=weights[train])
        oof_raw[held_out] = model.predict_proba(features[held_out])[:, 1]
        fold_models[dataset_id] = model
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_raw, labels, sample_weight=weights)

    compact = {
        "seed_threshold": float(settings.get("seed_threshold", 0.65)),
        "seed_quantile": float(settings.get("seed_quantile", 0.995)),
        "min_component_area": int(settings.get("min_component_area", 8)),
        "max_components": int(settings.get("max_components", 8)),
        "max_area_fraction": float(settings.get("max_area_fraction", 0.20)),
        "growth_factor": float(settings.get("growth_factor", 4.0)),
        "threshold_mode": str(settings.get("threshold_mode", "fixed")),
        "minimum_output_area_fraction": float(settings.get("minimum_output_area_fraction", 0.001)),
        "fallback_rank_threshold": float(settings.get("fallback_rank_threshold", 0.95)),
        "fallback_seed_quantile": float(settings.get("fallback_seed_quantile", 0.999)),
        "fallback_growth_factor": float(settings.get("fallback_growth_factor", 4.0)),
        "fallback_max_components": int(settings.get("fallback_max_components", 3)),
    }
    thresholds = tuple(float(value) for value in settings.get("thresholds", [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]))
    validation_size = int(settings.get("validation_size", train_size))
    override_ratios = tuple(float(value) for value in settings.get("override_area_ratios", [1.5, 2.0, 3.0, 4.0, 6.0, 8.0]))
    baseline_summary = _baseline_summary(images, validation_size)
    fold_rows = _evaluate_lodo_thresholds(
        images,
        fold_models,
        calibrator,
        thresholds,
        override_ratios,
        compact,
        validation_size,
    )
    threshold, override_ratio, threshold_summary = _choose_threshold(
        fold_rows,
        thresholds,
        baseline_summary=baseline_summary,
        max_area_ratio=float(settings.get("target_max_area_ratio", 2.0)),
        min_precision=float(settings.get("target_min_precision", 0.30)),
        max_dice_regression=float(settings.get("max_dataset_dice_regression", 0.02)),
    )

    final_model = HistGradientBoostingClassifier(**model_settings)
    final_model.fit(features, labels, sample_weight=weights)
    output_value = settings.get("model_output_path")
    if output_value is None:
        output_value = config.output_dir / "auto_masks" / "posterior" / "score_to_mask.joblib"
    output_path = config.resolve_path(str(output_value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": POSTERIOR_BUNDLE_SCHEMA_VERSION,
        "feature_names": list(POSTERIOR_FEATURE_NAMES),
        "model": final_model,
        "calibrator": calibrator,
        "threshold": threshold,
        "override_area_ratio": override_ratio,
        "compact": compact,
        "development_datasets": unique_datasets,
        "training_image_count": len(images),
        "training_pixel_count": len(labels),
        "threshold_summary": threshold_summary,
    }
    joblib.dump(bundle, output_path)

    report_value = settings.get("report_dir")
    if report_value is None:
        report_value = config.report_dir / "posterior_calibration"
    report_dir = config.resolve_path(str(report_value))
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "leave_dataset_out_metrics.json"
    write_json(
        metrics_path,
        {
            "selected_threshold": threshold,
            "selected_override_area_ratio": override_ratio,
            "baseline_summary": baseline_summary,
            "threshold_summary": threshold_summary,
            "rows": fold_rows,
        },
    )
    report_path = report_dir / "score_to_mask_calibration_report.md"
    report_path.write_text(
        _calibration_report(
            output_path,
            images,
            threshold,
            override_ratio,
            baseline_summary,
            threshold_summary,
            fold_rows,
        ),
        encoding="utf-8",
    )
    manifest_path = report_dir / "posterior_calibration_manifest.json"
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


def _parse_sources(config: AppConfig, settings: dict[str, Any]) -> list[PosteriorCalibrationSource]:
    raw_sources = settings.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("posterior_calibration.sources must be a list")
    output = []
    for item in raw_sources:
        if not isinstance(item, dict) or not isinstance(item.get("reference"), dict):
            raise ValueError("Each posterior calibration source requires a reference mapping")
        reference = item["reference"]
        output.append(
            PosteriorCalibrationSource(
                dataset_id=str(item.get("dataset_id", "")),
                metadata_path=config.resolve_path(str(item["metadata_path"])),
                reference_kind=str(reference.get("kind", "")),
                reference_path=config.resolve_path(str(reference.get("root") or reference.get("manifest_path"))),
                runtime_root=config.resolve_path(str(item.get("runtime_root", ""))),
            )
        )
    if len({source.dataset_id for source in output}) != len(output):
        raise ValueError("posterior calibration dataset_id values must be unique")
    for source in output:
        if not source.metadata_path.is_file():
            raise FileNotFoundError(f"Missing posterior metadata: {source.metadata_path}")
        if not source.reference_path.exists():
            raise FileNotFoundError(f"Missing posterior reference: {source.reference_path}")
        if not source.runtime_root.is_dir():
            raise FileNotFoundError(f"Missing posterior runtime root: {source.runtime_root}")
    return output


def _source_images(source: PosteriorCalibrationSource) -> list[_CalibrationImage]:
    reference_index: dict[tuple[str, str, str], Path] = {}
    if source.reference_kind == "manifest":
        for row in _read_jsonl(source.reference_path):
            key = (str(row["category"]), str(row["defect_type"]), Path(str(row["image_path"])).stem)
            reference_index[key] = Path(str(row["official_mask_path"]))
    images = []
    normal_cache: dict[str, tuple[Path, ...]] = {}
    for row in _read_jsonl(source.metadata_path):
        category = str(row["category"])
        defect_type = str(row["defect_type"])
        image_stem = Path(str(row["image_path"])).stem
        image_path = Path(str(row["image_path"]))
        parameters = row.get("settings", {}).get("mask_parameters", {})
        fused_path = Path(str(parameters.get("fused_evidence_path", "")))
        baseline_mask_path = Path(str(row.get("eval_mask_path") or row.get("refined_mask_path") or ""))
        if source.reference_kind == "mvtec":
            truth_path = source.reference_path / category / "ground_truth" / defect_type / f"{image_stem}_mask.png"
        else:
            truth_path = reference_index.get((category, defect_type, image_stem), Path())
        if category not in normal_cache:
            normal_cache[category] = tuple(
                sorted(
                    path
                    for path in (source.runtime_root / category / "train" / "good").rglob("*")
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                )[:6]
            )
        region = row.get("region_xyxy", [0, 0, 0, 0])
        semantic_regions = (
            (tuple(int(value) for value in region),)
            if isinstance(region, (list, tuple)) and len(region) == 4
            else ()
        )
        if fused_path.is_file() and truth_path.is_file() and baseline_mask_path.is_file() and image_path.is_file():
            images.append(
                _CalibrationImage(
                    dataset_id=source.dataset_id,
                    sample_id=f"{category}/{defect_type}/{image_stem}",
                    fused_path=fused_path,
                    truth_path=truth_path,
                    baseline_mask_path=baseline_mask_path,
                    image_path=image_path,
                    normal_paths=normal_cache[category],
                    semantic_regions=semantic_regions,
                )
            )
    return images


def _sample_training_pixels(
    images: list[_CalibrationImage],
    *,
    seed: int,
    max_per_class: int,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = []
    label_rows = []
    weight_rows = []
    dataset_rows = []
    for image in images:
        fused, truth = _load_pair(image, size)
        features = posterior_feature_image(
            fused,
            semantic_regions=_resize_regions(image.semantic_regions, image.image_path, fused.shape),
            normal_paths=image.normal_paths,
        ).reshape(-1, len(POSTERIOR_FEATURE_NAMES))
        flat_truth = truth.reshape(-1)
        rng = np.random.default_rng(_sample_seed(seed, image.dataset_id, image.sample_id))
        selected_parts = []
        selected_weights = []
        for label in (False, True):
            indices = np.flatnonzero(flat_truth == label)
            if indices.size == 0:
                continue
            count = min(max_per_class, indices.size)
            chosen = rng.choice(indices, size=count, replace=False)
            selected_parts.append(chosen)
            selected_weights.append(np.full(count, indices.size / count, dtype=np.float32))
        if not selected_parts:
            continue
        selected = np.concatenate(selected_parts)
        sample_weights = np.concatenate(selected_weights)
        sample_weights /= max(1e-6, float(sample_weights.mean()))
        feature_rows.append(features[selected])
        label_rows.append(flat_truth[selected])
        weight_rows.append(sample_weights)
        dataset_rows.append(np.full(len(selected), image.dataset_id, dtype=object))
    if not feature_rows:
        raise ValueError("Posterior calibration could not sample any pixels")
    return (
        np.concatenate(feature_rows).astype(np.float32),
        np.concatenate(label_rows).astype(np.uint8),
        np.concatenate(weight_rows).astype(np.float32),
        np.concatenate(dataset_rows),
    )


def _evaluate_lodo_thresholds(
    images: list[_CalibrationImage],
    fold_models: dict[str, HistGradientBoostingClassifier],
    calibrator: IsotonicRegression,
    thresholds: tuple[float, ...],
    override_ratios: tuple[float, ...],
    compact: dict[str, Any],
    size: int,
) -> list[dict[str, Any]]:
    rows = []
    for image in images:
        fused, truth = _load_pair(image, size)
        feature_image = posterior_feature_image(
            fused,
            semantic_regions=_resize_regions(image.semantic_regions, image.image_path, fused.shape),
            normal_paths=image.normal_paths,
        )
        raw = fold_models[image.dataset_id].predict_proba(feature_image.reshape(-1, feature_image.shape[-1]))[:, 1]
        posterior = np.asarray(calibrator.predict(raw), dtype=np.float32).reshape(fused.shape)
        actual = int(truth.sum())
        baseline = _load_binary_mask(image.baseline_mask_path, fused.shape)
        for threshold in thresholds:
            compact_mask, diagnostics = extract_calibrated_components(
                posterior,
                feature_image=feature_image,
                threshold=threshold,
                compact=compact,
                min_component_area=int(compact["min_component_area"]),
            )
            compact_area = int(compact_mask.sum())
            baseline_area = int(baseline.sum())
            for override_ratio in override_ratios:
                overridden = compact_area > 0 and baseline_area > override_ratio * compact_area
                mask = compact_mask if overridden else baseline
                intersection = int((mask & truth).sum())
                predicted = int(mask.sum())
                rows.append(
                    {
                        "dataset_id": image.dataset_id,
                        "sample_id": image.sample_id,
                        "threshold": threshold,
                        "override_area_ratio": override_ratio,
                        "overridden": overridden,
                        "dice": 2 * intersection / max(1, predicted + actual),
                        "precision": intersection / max(1, predicted),
                        "recall": intersection / max(1, actual),
                        "area_ratio": predicted / max(1, actual),
                        "predicted_pixels": predicted,
                        "actual_pixels": actual,
                        "predicted_positive_rate": predicted / mask.size,
                        "fallback": str(diagnostics.get("fallback", "none")),
                    }
                )
    return rows


def _choose_threshold(
    rows: list[dict[str, Any]],
    thresholds: tuple[float, ...],
    *,
    baseline_summary: dict[str, dict[str, float]],
    max_area_ratio: float,
    min_precision: float,
    max_dice_regression: float,
) -> tuple[float, float, dict[str, Any]]:
    summaries = []
    pairs = sorted({(float(row["threshold"]), float(row["override_area_ratio"])) for row in rows})
    for threshold, override_ratio in pairs:
        subset = [
            row
            for row in rows
            if float(row["threshold"]) == threshold and float(row["override_area_ratio"]) == override_ratio
        ]
        datasets = sorted({str(row["dataset_id"]) for row in subset})
        dataset_dice = {
            dataset: float(np.mean([float(row["dice"]) for row in subset if row["dataset_id"] == dataset]))
            for dataset in datasets
        }
        macro = {
            metric: float(
                np.mean(
                    [
                        np.mean([float(row[metric]) for row in subset if row["dataset_id"] == dataset])
                        for dataset in datasets
                    ]
                )
            )
            for metric in ("dice", "precision", "recall")
        }
        dataset_area_ratios = [
            sum(int(row["predicted_pixels"]) for row in subset if row["dataset_id"] == dataset)
            / max(1, sum(int(row["actual_pixels"]) for row in subset if row["dataset_id"] == dataset))
            for dataset in datasets
        ]
        macro["area_ratio"] = float(np.mean(dataset_area_ratios))
        non_regression = all(
            dataset_dice[dataset] >= float(baseline_summary[dataset]["dice"]) - max_dice_regression
            for dataset in datasets
        )
        feasible = macro["area_ratio"] <= max_area_ratio and macro["precision"] >= min_precision and non_regression
        score = macro["dice"] if feasible else macro["dice"] - 0.10 * max(0.0, macro["area_ratio"] - max_area_ratio) - 0.20 * max(0.0, min_precision - macro["precision"])
        summaries.append(
            {
                "threshold": threshold,
                "override_area_ratio": override_ratio,
                **macro,
                "dataset_dice": dataset_dice,
                "non_regression": non_regression,
                "feasible": feasible,
                "selection_score": score,
            }
        )
    feasible_rows = [row for row in summaries if row["feasible"]]
    selected = max(
        feasible_rows or summaries,
        key=lambda row: (row["selection_score"], row["override_area_ratio"], row["threshold"]),
    )
    return (
        float(selected["threshold"]),
        float(selected["override_area_ratio"]),
        {"selected": selected, "candidates": summaries},
    )


def _calibration_report(
    model_path: Path,
    images: list[_CalibrationImage],
    threshold: float,
    override_ratio: float,
    baseline_summary: dict[str, dict[str, float]],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    selected_rows = [
        row
        for row in rows
        if float(row["threshold"]) == threshold and float(row["override_area_ratio"]) == override_ratio
    ]
    lines = [
        "# V4.1 Score-To-Mask Calibration Report",
        "",
        "The posterior is trained only on exposed development labels. Every reported row is predicted by a model that excluded its entire dataset.",
        "",
        f"- Frozen model: `{model_path}`",
        f"- Development images: `{len(images)}`",
        f"- Selected threshold: `{threshold:.3f}`",
        f"- Broad-mask override ratio: `{override_ratio:.3f}`",
        f"- Selected row passes all promotion constraints: `{bool(summary['selected']['feasible'])}`",
        "",
        "| Dataset | Images | V3 Dice | V4 Dice | Delta | Precision | Recall | Predicted/True Area | Override Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset_id in sorted({str(row["dataset_id"]) for row in selected_rows}):
        group = [row for row in selected_rows if row["dataset_id"] == dataset_id]
        mean = lambda key: float(np.mean([float(row[key]) for row in group]))
        aggregate_area_ratio = sum(int(row["predicted_pixels"]) for row in group) / max(
            1, sum(int(row["actual_pixels"]) for row in group)
        )
        baseline_dice = float(baseline_summary[dataset_id]["dice"])
        v4_dice = mean("dice")
        lines.append(
            f"| {dataset_id} | {len(group)} | `{baseline_dice:.4f}` | `{v4_dice:.4f}` | "
            f"`{v4_dice - baseline_dice:+.4f}` | `{mean('precision'):.4f}` | `{mean('recall'):.4f}` | "
            f"`{aggregate_area_ratio:.4f}` | `{np.mean([bool(row['overridden']) for row in group]):.4f}` |"
        )
    lines.extend(["", "This is development evidence, not a new locked-generalization claim.", ""])
    return "\n".join(lines)


def _baseline_summary(images: list[_CalibrationImage], size: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, int | float]]] = {}
    for image in images:
        _, truth = _load_pair(image, size)
        prediction = _load_binary_mask(image.baseline_mask_path, truth.shape)
        intersection = int((prediction & truth).sum())
        predicted = int(prediction.sum())
        actual = int(truth.sum())
        grouped.setdefault(image.dataset_id, []).append(
            {
                "dice": 2 * intersection / max(1, predicted + actual),
                "precision": intersection / max(1, predicted),
                "recall": intersection / max(1, actual),
                "predicted_pixels": predicted,
                "actual_pixels": actual,
            }
        )
    output = {}
    for dataset_id, rows in grouped.items():
        output[dataset_id] = {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in ("dice", "precision", "recall")
        }
        output[dataset_id]["area_ratio"] = sum(int(row["predicted_pixels"]) for row in rows) / max(
            1, sum(int(row["actual_pixels"]) for row in rows)
        )
    return output


def _load_pair(image: _CalibrationImage, size: int) -> tuple[np.ndarray, np.ndarray]:
    fused_image = Image.open(image.fused_path).convert("L")
    if size > 0:
        fused_image = fused_image.resize((size, size), Image.Resampling.BILINEAR)
    fused = np.asarray(fused_image, dtype=np.float32) / 255.0
    truth_image = Image.open(image.truth_path).convert("L").resize(fused_image.size, Image.Resampling.NEAREST)
    truth = np.asarray(truth_image, dtype=np.uint8) > 0
    return fused, truth


def _load_binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST),
        dtype=np.uint8,
    ) > 0


def _keep_seeded_components(
    mask: np.ndarray,
    seeds: np.ndarray,
    min_component_area: int,
    max_components: int,
) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    components = []
    for index in range(1, count):
        component = labels == index
        if int(component.sum()) >= min_component_area and bool((component & seeds).any()):
            components.append(component)
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    output = np.zeros_like(mask, dtype=bool)
    for component in components[: max(1, max_components)]:
        output |= component
    return output


def _semantic_prior(
    shape: tuple[int, int],
    regions: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    height, width = shape
    if not regions:
        return np.ones(shape, dtype=np.float32)
    prior = np.full(shape, 0.2, dtype=np.float32)
    for left, top, right, bottom in regions:
        pad_x = max(1, int(round((right - left) * 0.10)))
        pad_y = max(1, int(round((bottom - top) * 0.10)))
        x1, y1 = max(0, left - pad_x), max(0, top - pad_y)
        x2, y2 = min(width, right + pad_x), min(height, bottom + pad_y)
        prior[y1:y2, x1:x2] = np.maximum(prior[y1:y2, x1:x2], 0.5)
        prior[max(0, top) : min(height, bottom), max(0, left) : min(width, right)] = 1.0
    return prior


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


def _as_probability(values: np.ndarray) -> np.ndarray:
    array = np.array(values, dtype=np.float32, copy=True)
    array[~np.isfinite(array)] = 0.0
    return np.clip(array, 0.0, 1.0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_seed(seed: int, dataset_id: str, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_id}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
