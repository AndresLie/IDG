from __future__ import annotations

import json
import re
import importlib.util
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
import skimage.filters
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageMath

from iadgen_v2.auto_mask.contracts import AutoMaskContext, DEFAULT_AUTO_CANDIDATE_MODES, AutoMaskRecord, PatchFeatureCache
from iadgen_v2.auto_mask.evidence import FunctionalEvidenceProvider, MultiScaleDinoProvider, RegisteredDinoResidualProvider
from iadgen_v2.auto_mask.pipeline import run_generic_evidence_pipeline
from iadgen_v2.auto_mask.selection import GenericCandidateSelector
from iadgen_v2.auto_mask.specialists import specialist_applicability
from iadgen_v2.auto_mask.legacy_structure import infer_structure_attributes, infer_structure_profile
from iadgen_v2.auto_mask.structure import infer_structure_profile_from_measurements
from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.dataset import IMAGE_EXTENSIONS
from iadgen_v2.masks import write_mask_overlay, write_pixel_refined_bbox_masks, write_refined_bbox_masks
from iadgen_v2.qwen_provider import (
    DEFAULT_LOCALIZATION_DECODING,
    CachedQwenLocalizationExtractor,
    QwenFeatureExtractor,
    qwen_availability,
    resolve_qwen_model_revision,
    write_availability,
)
from iadgen_v2.records import write_json


_DINO_MODEL_CACHE: dict[tuple[str, str | None, bool, str], tuple[Any, Any]] = {}
_SAM2_PREDICTOR_CACHE: dict[tuple[str, str, str], Any] = {}
_SAM1_PREDICTOR_CACHE: dict[tuple[str, str, str], Any] = {}
def _auto_mask_sample_key(category: str, defect_type: str, image_path: str) -> tuple[str, str, str]:
    return category, defect_type, str(Path(image_path).resolve())


def _load_auto_mask_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _auto_mask_record_from_row(row: dict[str, Any]) -> AutoMaskRecord:
    optional_paths = ("eval_mask_path", "training_mask_path", "uncertainty_mask_path", "overlay_path")
    values: dict[str, Any] = {}
    for field in fields(AutoMaskRecord):
        name = field.name
        if name in row:
            values[name] = row[name]
        elif name in optional_paths:
            values[name] = None
        elif name in {"mask_variant_paths", "mask_variant_overlay_paths", "settings"}:
            values[name] = {}
        elif name == "seed":
            values[name] = 0
        elif name == "region_xyxy":
            values[name] = (0, 0, 0, 0)
        elif name in {"confidence"}:
            values[name] = None
        elif name in {"qwen_text", "qwen_defect_type", "evidence"}:
            values[name] = None
        else:
            values[name] = ""
    values["region_xyxy"] = tuple(int(value) for value in values["region_xyxy"])
    values["mask_variant_paths"] = dict(values["mask_variant_paths"])
    values["mask_variant_overlay_paths"] = dict(values["mask_variant_overlay_paths"])
    values["settings"] = dict(values["settings"])
    return AutoMaskRecord(**values)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _publish_auto_mask_checkpoint(
    pending_masks: list[tuple[Path, Path]],
    *,
    run_metadata_path: Path,
    stable_metadata_path: Path,
    run_id: str,
) -> None:
    staged: list[tuple[Path, Path, Path | None]] = []
    published: list[tuple[Path, Path | None]] = []
    try:
        for source, destination in pending_masks:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{run_id}.tmp")
            backup = destination.with_name(f".{destination.name}.{run_id}.bak") if destination.exists() else None
            Image.open(source).convert("L").save(temporary, format="PNG")
            if backup is not None:
                backup.write_bytes(destination.read_bytes())
            staged.append((temporary, destination, backup))

        stable_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_temporary = stable_metadata_path.with_name(f".{stable_metadata_path.name}.{run_id}.tmp")
        metadata_backup = (
            stable_metadata_path.with_name(f".{stable_metadata_path.name}.{run_id}.bak")
            if stable_metadata_path.exists()
            else None
        )
        metadata_temporary.write_bytes(run_metadata_path.read_bytes())
        if metadata_backup is not None:
            metadata_backup.write_bytes(stable_metadata_path.read_bytes())
        staged.append((metadata_temporary, stable_metadata_path, metadata_backup))

        for temporary, destination, backup in staged:
            temporary.replace(destination)
            published.append((destination, backup))
    except BaseException:
        for destination, backup in reversed(published):
            if backup is not None and backup.exists():
                backup.replace(destination)
            else:
                destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _, backup in staged:
            temporary.unlink(missing_ok=True)
            if backup is not None:
                backup.unlink(missing_ok=True)


def run_auto_masks(config: AppConfig) -> Path:
    auto = _auto_config(config)
    _resolve_auto_paths(config, auto)
    architecture = _auto_architecture(auto)
    provider = str(auto.get("provider", "qwen"))
    if provider != "qwen":
        raise ValueError(f"Unsupported auto mask provider: {provider}")

    output_dir = config.output_dir / "auto_masks" / provider
    report_dir = config.report_dir / "auto_masks" / provider
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_output_dir = output_dir / "runs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    run_metadata_path = run_output_dir / "metadata.jsonl"
    pending_metadata_path = run_output_dir / "metadata.jsonl.tmp"
    run_status_path = output_dir / "last_run_status.json"
    current_run_path = output_dir / "current_run.json"
    descriptions = _load_descriptions(config, auto)
    normal_index = _normal_index(config)
    existing_rows = _load_auto_mask_rows(metadata_path)
    existing_by_sample = {
        _auto_mask_sample_key(
            str(row.get("category", "")),
            str(row.get("defect_type", "")),
            str(row.get("image_path", "")),
        ): row
        for row in existing_rows
    }

    availability = _qwen_availability(config, auto)
    write_availability(output_dir / "qwen_availability.json", availability)
    if not availability.ready:
        raise RuntimeError(f"Qwen auto-mask provider is not ready: {availability.failure_reason()}")
    pending_metadata_path.write_text("", encoding="utf-8")

    qwen_model_id = str(auto.get("qwen_model", _fallback_qwen(config, "qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")))
    qwen_cache_dir = auto.get("qwen_cache_dir", _fallback_qwen(config, "qwen_cache_dir", None))
    qwen_dtype = str(auto.get("qwen_dtype", _fallback_qwen(config, "qwen_dtype", "auto")))
    base_extractor = QwenFeatureExtractor(
        model_id=qwen_model_id,
        cache_dir=qwen_cache_dir,
        local_files_only=bool(auto.get("qwen_local_files_only", _fallback_qwen(config, "qwen_local_files_only", True))),
        device=str(auto.get("qwen_device", _fallback_qwen(config, "qwen_device", "auto"))),
        torch_dtype=qwen_dtype,
        token_count=int(auto.get("token_count", _fallback_qwen(config, "token_count", 16))),
        min_free_gib=float(auto.get("qwen_min_free_gib", _fallback_qwen(config, "qwen_min_free_gib", 30.0))),
    )
    localization_cache_value = auto.get("qwen_localization_cache_dir")
    localization_cache_dir = (
        Path(str(localization_cache_value))
        if localization_cache_value
        else output_dir / "localization_cache"
    )
    extractor = CachedQwenLocalizationExtractor(
        base_extractor,
        cache_dir=localization_cache_dir,
        model_id=qwen_model_id,
        model_revision=resolve_qwen_model_revision(qwen_model_id, qwen_cache_dir),
        torch_dtype=qwen_dtype,
        decoding_settings=DEFAULT_LOCALIZATION_DECODING,
        enabled=bool(auto.get("qwen_localization_cache_enabled", True)),
        refresh=bool(auto.get("qwen_localization_cache_refresh", False)),
    )

    rows: list[AutoMaskRecord] = []
    pending_ground_truth: list[tuple[Path, Path]] = []
    try:
        for category, defect_type, image_path in _defect_images(config):
            image = Image.open(image_path).convert("RGB")
            publish_dataset_ground_truth = bool(
                auto.get("publish_dataset_ground_truth", architecture != "generic_evidence")
            )
            if publish_dataset_ground_truth:
                mask_dir = config.dataset_root / category / "ground_truth" / defect_type
            else:
                mask_dir = output_dir / "pseudo_ground_truth" / category / defect_type
            mask_path = mask_dir / f"{image_path.stem}_mask.png"
            if mask_path.exists() and not bool(auto.get("overwrite", False)):
                existing = existing_by_sample.get(_auto_mask_sample_key(category, defect_type, str(image_path)))
                if existing is not None:
                    record = _auto_mask_record_from_row(existing)
                    rows.append(record)
                    with pending_metadata_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                        handle.flush()
                continue

            description = _description_for(config, descriptions, category, defect_type, image_path)
            prompt = _prompt(category, defect_type, image.size, auto, description)
            result = extractor.analyze(image, prompt)
            parsed = parse_qwen_bbox_payload(str(result["text"]), image.size)
            region, localization = _validated_or_fallback_region(
                parsed["bbox_xyxy"],
                image.size,
                auto=auto,
            )
            valid_sub_boxes = _validated_sub_boxes(
                parsed.get("sub_boxes_xyxy", []),
                image.size,
                min_area_ratio=float(auto.get("min_sub_box_area_ratio", auto.get("min_box_area_ratio", 0.001) / 4.0)),
                max_area_ratio=float(auto.get("max_sub_box_area_ratio", auto.get("max_box_area_ratio", 0.5))),
            )
            region, sub_box_selection = _select_qwen_region_from_sub_boxes(region, valid_sub_boxes, image.size, auto)
            localization.update(sub_box_selection)
            localization["qwen_cache"] = result.get(
                "localization_cache",
                {"enabled": False, "cache_hit": False},
            )
            seed = int(auto.get("mask_seed", 17)) + _stable_seed(f"{category}:{defect_type}:{image_path.name}")
            artifact_stem = f"{category}_{defect_type}_{image_path.stem}"
            if architecture == "generic_evidence":
                localization_policy = "generic_soft_spatial_prior"
                policy_diagnostics = {
                    "reason": "Qwen region is a soft prior; generic evidence remains active outside it",
                    "specialists": str(auto.get("specialists", "disabled")),
                }
            else:
                localization_policy, policy_diagnostics = resolve_auto_localization_policy(
                    auto=auto,
                    category=category,
                    defect_type=defect_type,
                    description=description,
                    image=image,
                    qwen_region=region,
                    localization=localization,
                )
            localization["policy"] = localization_policy
            localization["policy_diagnostics"] = policy_diagnostics
            if architecture != "generic_evidence" and localization_policy == "normal_guided_qwen_verify":
                try:
                    guided_region, guided_localization = normal_guided_qwen_verify_region(
                        image=image,
                        normal_paths=normal_index.get(category, []),
                        category=category,
                        defect_type=defect_type,
                        description=description,
                        extractor=extractor,
                        auto=auto,
                        output_dir=run_output_dir / "localization" / category / defect_type,
                        stem=artifact_stem,
                    )
                    region = guided_region
                    localization["normal_guided_qwen_verify"] = guided_localization
                except Exception as exc:
                    localization["normal_guided_qwen_verify"] = {
                        "status": "fallback_to_qwen_bbox",
                        "failure_reason": str(exc),
                    }
            elif architecture != "generic_evidence" and localization_policy == "bottle_rim_guided_qwen":
                region, bottle_localization = bottle_rim_guided_qwen_region(
                    image_size=image.size,
                    qwen_region=region,
                    defect_type=defect_type,
                    auto=auto,
                )
                localization["bottle_rim_guided_qwen"] = bottle_localization
            elif architecture != "generic_evidence" and localization_policy == "bottle_surface_guided":
                region, bottle_localization = bottle_surface_guided_region(
                    image_size=image.size,
                    qwen_region=region,
                    auto=auto,
                )
                localization["bottle_surface_guided"] = bottle_localization
            requested_refinement = str(auto.get("mask_refinement", "auto"))
            if architecture == "generic_evidence":
                generic_ev = auto.get("generic_evidence", {}) if isinstance(auto.get("generic_evidence"), dict) else {}
                generic_region = region
                generic_semantic = [region, *valid_sub_boxes]
                # A-S2 localization reliability: on strongly repeated structures
                # (measured, category-agnostic) Qwen tends to localize the wrong
                # repeat with high confidence, and the soft spatial prior then
                # suppresses the true defect. When repetition is high, widen to
                # the full image so evidence is not fenced to a wrong box.
                # Verified: recovers oracle 0.0 -> 0.43-0.66 on the dead
                # split/broken-teeth samples without touching the correctly
                # localized ones. Low-repetition objects (e.g. bottle) keep the
                # narrow box, where fencing aids precision.
                # Default OFF: validated to recover the oracle ceiling on
                # repeated structures (zipper oracle 0.4565->0.5750; dead
                # split/broken-teeth 0.0->0.43-0.66) but it REGRESSES selected
                # Dice (zipper 0.2614->0.1906) because the miscalibrated selector
                # mis-picks among the widened pool (fabric_border collapses).
                # So it is gated behind A-S1b selector recalibration: enable
                # widen_on_repeated_texture once the selector can rank the
                # enriched pool. Prefer widening to the periodic extent rather
                # than the full image when re-enabling.
                widen_enabled = bool(generic_ev.get("widen_on_repeated_texture", False))
                widen_threshold = float(generic_ev.get("localization_widen_repeated_texture", 0.17))
                repeated_texture = _repeated_texture_score(image)
                if widen_enabled and float(repeated_texture) >= widen_threshold:
                    full_region = (0, 0, image.width, image.height)
                    generic_region = full_region
                    generic_semantic = [full_region]
                    localization["repeated_structure_widening"] = {
                        "applied": True,
                        "repeated_texture_score": round(float(repeated_texture), 4),
                        "threshold": widen_threshold,
                    }
                mask_artifacts = _run_generic_mask_artifacts(
                    config=config,
                    auto=auto,
                    image=image,
                    image_path=image_path,
                    normal_paths=normal_index.get(category, []),
                    region=generic_region,
                    semantic_regions=generic_semantic,
                    output_dir=run_output_dir / "masks" / category / defect_type,
                    variant_dir=run_output_dir / "mask_variants" / category / defect_type,
                    artifact_stem=artifact_stem,
                    seed=seed,
                )
            else:
                mask_artifacts = _select_mask_artifacts(
                    image=image,
                    category=category,
                    defect_type=defect_type,
                    region=region,
                    output_dir=run_output_dir / "masks" / category / defect_type,
                    artifact_stem=artifact_stem,
                    seed=seed,
                    auto=auto,
                    description=description,
                    normal_paths=normal_index.get(category, []),
                    requested=requested_refinement,
                    extractor=extractor,
                    localization_status=str(localization.get("status", "unknown")),
                )
                mask_artifacts = _attach_mask_variants(
                    image=image,
                    category=category,
                    defect_type=defect_type,
                    region=region,
                    output_dir=run_output_dir / "mask_variants" / category / defect_type,
                    artifact_stem=artifact_stem,
                    artifacts=mask_artifacts,
                    auto=auto,
                )
            ground_truth_source = str(auto.get("ground_truth_mask_variant", "eval_tight"))
            pending_ground_truth.append(
                (Path(_variant_or_default(mask_artifacts, ground_truth_source, "refined_mask_path")), mask_path)
            )

            overlay_path: Path | None = None
            if bool(auto.get("write_overlays", True)):
                overlay_path = run_output_dir / "overlays" / category / defect_type / f"{image_path.stem}.png"
                write_mask_overlay(image, region, Path(mask_artifacts["refined_mask_path"]), overlay_path)

            record = AutoMaskRecord(
                category=category,
                defect_type=defect_type,
                image_path=str(image_path),
                mask_path=str(mask_path),
                provider=provider,
                prompt=prompt,
                description=description,
                qwen_text=str(result["text"]),
                qwen_defect_type=parsed.get("defect_type"),
                confidence=parsed.get("confidence"),
                evidence=parsed.get("evidence"),
                region_xyxy=region,
                box_mask_path=str(mask_artifacts["box_mask_path"]),
                refined_mask_path=str(mask_artifacts["refined_mask_path"]),
                inpaint_mask_path=str(mask_artifacts["inpaint_mask_path"]),
                eval_mask_path=str(mask_artifacts.get("eval_mask_path")) if mask_artifacts.get("eval_mask_path") else None,
                training_mask_path=str(mask_artifacts.get("training_mask_path")) if mask_artifacts.get("training_mask_path") else None,
                uncertainty_mask_path=(
                    str(mask_artifacts.get("uncertainty_mask_path")) if mask_artifacts.get("uncertainty_mask_path") else None
                ),
                mask_variant_paths={str(key): str(value) for key, value in mask_artifacts.get("mask_variant_paths", {}).items()},
                mask_variant_overlay_paths={
                    str(key): str(value) for key, value in mask_artifacts.get("mask_variant_overlay_paths", {}).items()
                },
                overlay_path=str(overlay_path) if overlay_path else None,
                seed=seed,
                settings={
                    "auto_mask_schema_version": 2,
                    "auto_mask_architecture": architecture,
                    "auto_mask_run_id": run_id,
                    "label_source": "path_defect_folder",
                    "mask_truth_source": "qwen_auto_refined_masks",
                    "qwen_reported_defect_type": parsed.get("defect_type"),
                    "qwen_localization": localization,
                    "qwen_sub_boxes_xyxy": valid_sub_boxes,
                    "qwen_sub_box_count": len(valid_sub_boxes),
                    "path_defect_type_used": defect_type,
                    "mask_refinement": requested_refinement,
                    "selected_refinement": mask_artifacts["selected_refinement"],
                    "candidate_scores": mask_artifacts.get("candidate_scores", {}),
                    "candidate_qc": mask_artifacts.get("candidate_qc", {}),
                    "candidate_measurements": mask_artifacts.get("candidate_measurements", {}),
                    "candidate_predictions": mask_artifacts.get("candidate_predictions", []),
                    "candidate_failures": mask_artifacts.get("candidate_failures", {}),
                    "candidate_modes": mask_artifacts.get("candidate_modes", []),
                    "policy_scores": mask_artifacts.get("policy_scores", {}),
                    "scratch_morphology_class": mask_artifacts.get("scratch_morphology_class"),
                    "quality_morphology": mask_artifacts.get("label_policy", {}).get("quality_morphology"),
                    "label_policy": mask_artifacts.get("label_policy", {}),
                    "selection_policy": mask_artifacts.get("selection_policy"),
                    "selection_arbitration": mask_artifacts.get("selection_arbitration", {}),
                    "selection_decision": mask_artifacts.get("selection_decision", {}),
                    "structure_profile": mask_artifacts.get("structure_profile", "unknown"),
                    "candidate_rejections": mask_artifacts.get("candidate_rejections", {}),
                    "candidate_refined_paths": mask_artifacts.get("candidate_refined_paths", {}),
                    "candidate_heatmap_paths": mask_artifacts.get("candidate_heatmap_paths", {}),
                    "mask_parameters": mask_artifacts["parameters"],
                    "ground_truth_mask_variant": ground_truth_source,
                    "generation_core_mask_path": str(mask_artifacts.get("generation_core_mask_path")),
                    "benchmark_eval_mask_path": str(mask_artifacts.get("benchmark_eval_mask_path")),
                    "eval_mask_path": str(mask_artifacts.get("eval_mask_path")) if mask_artifacts.get("eval_mask_path") else None,
                    "training_mask_path": str(mask_artifacts.get("training_mask_path")) if mask_artifacts.get("training_mask_path") else None,
                    "uncertainty_mask_path": (
                        str(mask_artifacts.get("uncertainty_mask_path")) if mask_artifacts.get("uncertainty_mask_path") else None
                    ),
                    "mask_variant_paths": mask_artifacts.get("mask_variant_paths", {}),
                    "mask_variant_overlay_paths": mask_artifacts.get("mask_variant_overlay_paths", {}),
                    "qc": mask_artifacts["qc"],
                    "auto_mask_fingerprint": _auto_fingerprint(config),
                },
            )
            rows.append(record)
            with pending_metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                handle.flush()
    except BaseException as exc:
        partial_path = run_output_dir / "metadata.partial.jsonl"
        if pending_metadata_path.exists():
            pending_metadata_path.replace(partial_path)
        _write_json_atomic(
            run_status_path,
            {
                "run_id": run_id,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "records_completed": len(rows),
                "partial_metadata_path": str(partial_path),
                "preserved_metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "run_output_dir": str(run_output_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "auto_mask_fingerprint": _auto_fingerprint(config),
            },
        )
        raise
    finally:
        extractor.close()

    pending_metadata_path.replace(run_metadata_path)
    try:
        _publish_auto_mask_checkpoint(
            pending_ground_truth,
            run_metadata_path=run_metadata_path,
            stable_metadata_path=metadata_path,
            run_id=run_id,
        )
    except BaseException as exc:
        _write_json_atomic(
            run_status_path,
            {
                "run_id": run_id,
                "status": "publish_failed",
                "records_completed": len(rows),
                "run_metadata_path": str(run_metadata_path),
                "preserved_metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "run_output_dir": str(run_output_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "auto_mask_fingerprint": _auto_fingerprint(config),
            },
        )
        raise
    _write_json_atomic(
        current_run_path,
        {
            "run_id": run_id,
            "status": "complete",
            "run_output_dir": str(run_output_dir),
            "run_metadata_path": str(run_metadata_path),
            "stable_metadata_path": str(metadata_path),
            "records": len(rows),
            "auto_mask_fingerprint": _auto_fingerprint(config),
        },
    )
    _write_json_atomic(
        run_status_path,
        {
            "run_id": run_id,
            "status": "complete",
            "records_completed": len(rows),
            "metadata_path": str(metadata_path),
            "run_metadata_path": str(run_metadata_path),
            "run_output_dir": str(run_output_dir),
            "auto_mask_fingerprint": _auto_fingerprint(config),
        },
    )
    _write_summary(report_dir / "summary.md", rows, metadata_path, auto)
    _write_contact_sheet(report_dir / "contact_sheet_mask_variants_current.png", rows)
    _write_candidate_comparison_sheet(report_dir / "contact_sheet_candidate_comparison.png", rows)
    write_json(
        report_dir / "summary.json",
        {
            "provider": provider,
            "records": len(rows),
            "qc": _qc_counts(rows),
            "metadata_path": str(metadata_path),
            "mask_truth_source": "qwen_auto_refined_masks",
            "auto_mask_fingerprint": _auto_fingerprint(config),
        },
    )
    return metadata_path


def reselect_auto_masks(config: AppConfig) -> Path:
    auto = _auto_config(config)
    _resolve_auto_paths(config, auto)
    if _auto_architecture(auto) == "generic_evidence":
        raise ValueError(
            "auto-masks-reselect is restricted to the frozen legacy specialist path; "
            "generic_evidence must rerun auto-masks so cached evidence is fused by the calibrated selector"
        )
    provider = str(auto.get("provider", "qwen"))
    output_dir = config.output_dir / "auto_masks" / provider
    report_dir = config.report_dir / "auto_masks" / provider
    metadata_path = output_dir / "metadata.jsonl"
    source_rows = _load_auto_mask_rows(metadata_path)
    if not source_rows:
        raise FileNotFoundError(f"No cached auto-mask metadata found: {metadata_path}")
    source_cache_audit = _candidate_cache_summary(source_rows)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}-reselect"
    run_output_dir = output_dir / "runs" / run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    pending_metadata_path = run_output_dir / "metadata.jsonl.tmp"
    run_metadata_path = run_output_dir / "metadata.jsonl"
    run_status_path = output_dir / "last_run_status.json"
    current_run_path = output_dir / "current_run.json"
    pending_metadata_path.write_text("", encoding="utf-8")
    pending_ground_truth: list[tuple[Path, Path]] = []
    rows: list[dict[str, Any]] = []

    try:
        for source_row in source_rows:
            row = json.loads(json.dumps(source_row))
            category = str(row["category"])
            defect_type = str(row["defect_type"])
            image_path = Path(str(row["image_path"]))
            image = Image.open(image_path).convert("RGB")
            region = tuple(int(value) for value in row["region_xyxy"])
            settings = dict(row.get("settings", {}))
            candidates = _cached_candidates_from_row(row)
            cache_audit = _candidate_cache_audit(row)
            selected_mask_fallback_used = False
            valid_candidates = {
                mode: candidate
                for mode, candidate in candidates.items()
                if _candidate_mode_applicable(
                    mode,
                    category=category,
                    defect_type=defect_type,
                    description=str(row.get("description", "")),
                )
                and _candidate_is_valid(candidate)
            }
            if not valid_candidates:
                fallback = _selected_mask_compatibility_candidate(row, image=image, region=region)
                if fallback is not None and bool(auto.get("allow_selected_mask_reselect_fallback", True)):
                    fallback_mode = str(fallback["selected_refinement"])
                    candidates[fallback_mode] = fallback
                    valid_candidates[fallback_mode] = fallback
                    selected_mask_fallback_used = True
                else:
                    raise ValueError(
                        "No reusable candidates for "
                        f"{category}/{defect_type}/{image_path.name}; "
                        f"recorded={cache_audit['recorded_candidate_paths']}, "
                        f"existing={cache_audit['existing_candidate_paths']}, "
                        f"selected_eval_exists={cache_audit['selected_eval_mask_exists']}, "
                        f"selected_refined_exists={cache_audit['selected_refined_mask_exists']}. "
                        "Regenerate full auto-mask candidates or enable "
                        "auto_masks.allow_selected_mask_reselect_fallback for a selected-mask-only "
                        "compatibility run."
                    )

            hierarchy = [
                str(mode)
                for mode in settings.get("candidate_modes", [])
                if str(mode) in valid_candidates
            ]
            hierarchy.extend(mode for mode in valid_candidates if mode not in hierarchy)
            policy = _selection_policy(
                category=category,
                defect_type=defect_type,
                region=region,
                image_size=image.size,
                description=str(row.get("description", "")),
                candidates=valid_candidates,
                localization_status=str(settings.get("qwen_localization", {}).get("status", "unknown")),
                auto=auto,
            )
            adjusted_scores = {
                mode: _policy_adjusted_score(mode, candidate, policy)
                for mode, candidate in valid_candidates.items()
            }
            _add_derived_recall_safe_candidates(
                candidates=candidates,
                valid_candidates=valid_candidates,
                image=image,
                category=category,
                defect_type=defect_type,
                region=region,
                output_dir=run_output_dir / "masks" / category / defect_type,
                artifact_stem=f"{category}_{defect_type}_{image_path.stem}",
                auto=auto,
                description=str(row.get("description", "")),
            )
            for mode, candidate in valid_candidates.items():
                if mode not in adjusted_scores:
                    adjusted_scores[mode] = _policy_adjusted_score(mode, candidate, policy)
                if mode not in hierarchy:
                    hierarchy.append(mode)
            selected_mode, selected, arbitration = _choose_candidate_with_arbitration(
                valid_candidates,
                adjusted_scores=adjusted_scores,
                hierarchy=hierarchy,
                policy=policy,
            )
            if selected_mask_fallback_used:
                arbitration = {
                    **arbitration,
                    "applied": False,
                    "initial_selected_refinement": selected_mode,
                    "final_selected_refinement": selected_mode,
                    "reason": "selected_mask_compatibility_fallback",
                    "candidate_pool_available": False,
                    "oracle_regret_available": False,
                }
            artifact_stem = f"{category}_{defect_type}_{image_path.stem}"
            artifacts = _copy_candidate_as_final(
                selected,
                run_output_dir / "masks" / category / defect_type,
                artifact_stem,
            )
            artifacts.update(
                {
                    "selected_refinement": selected_mode,
                    "candidate_scores": {
                        mode: float(candidate.get("score", -1.0))
                        for mode, candidate in candidates.items()
                    },
                    "candidate_qc": {
                        mode: candidate.get("qc", {})
                        for mode, candidate in candidates.items()
                    },
                    "candidate_refined_paths": {
                        mode: str(candidate["refined_mask_path"])
                        for mode, candidate in candidates.items()
                    },
                    "candidate_heatmap_paths": {
                        **{
                            mode: str(path)
                            for mode, path in settings.get("candidate_heatmap_paths", {}).items()
                            if path and Path(str(path)).exists()
                        },
                        **{
                            mode: str(candidate.get("parameters", {}).get("heatmap_path"))
                            for mode, candidate in candidates.items()
                            if candidate.get("parameters", {}).get("heatmap_path")
                            and Path(str(candidate.get("parameters", {}).get("heatmap_path"))).exists()
                        },
                    },
                    "candidate_failures": dict(settings.get("candidate_failures", {})),
                    "candidate_modes": hierarchy,
                    "policy_scores": adjusted_scores,
                    "scratch_morphology_class": policy["scratch_morphology_class"],
                    "structure_profile": policy.get("structure_profile", "unknown"),
                    "selection_policy": policy["selection_policy"],
                    "candidate_rejections": policy["candidate_rejections"],
                    "selection_arbitration": arbitration,
                    "description": str(row.get("description", "")),
                }
            )
            artifacts = _attach_mask_variants(
                image=image,
                category=category,
                defect_type=defect_type,
                region=region,
                output_dir=run_output_dir / "mask_variants" / category / defect_type,
                artifact_stem=artifact_stem,
                artifacts=artifacts,
                auto=auto,
            )

            row.update(
                {
                    "box_mask_path": str(artifacts["box_mask_path"]),
                    "refined_mask_path": str(artifacts["refined_mask_path"]),
                    "inpaint_mask_path": str(artifacts["inpaint_mask_path"]),
                    "eval_mask_path": str(artifacts["eval_mask_path"]),
                    "training_mask_path": str(artifacts["training_mask_path"]),
                    "uncertainty_mask_path": str(artifacts["uncertainty_mask_path"]),
                    "mask_variant_paths": dict(artifacts["mask_variant_paths"]),
                    "mask_variant_overlay_paths": dict(artifacts["mask_variant_overlay_paths"]),
                }
            )
            settings.update(
                {
                    "auto_mask_run_id": run_id,
                    "selected_refinement": selected_mode,
                    "candidate_scores": artifacts["candidate_scores"],
                    "candidate_qc": artifacts["candidate_qc"],
                    "candidate_refined_paths": artifacts["candidate_refined_paths"],
                    "candidate_heatmap_paths": artifacts["candidate_heatmap_paths"],
                    "candidate_modes": hierarchy,
                    "policy_scores": adjusted_scores,
                    "scratch_morphology_class": artifacts["scratch_morphology_class"],
                    "label_policy": artifacts["label_policy"],
                    "selection_policy": artifacts["selection_policy"],
                    "selection_arbitration": arbitration,
                    "structure_profile": artifacts["structure_profile"],
                    "candidate_rejections": artifacts["candidate_rejections"],
                    "candidate_cache_audit": cache_audit,
                    "candidate_cache_status": (
                        "selected_mask_compatibility_fallback"
                        if selected_mask_fallback_used
                        else "full_candidate_pool_available"
                    ),
                    "oracle_regret_available": not selected_mask_fallback_used,
                    "mask_parameters": artifacts["parameters"],
                    "generation_core_mask_path": str(artifacts["generation_core_mask_path"]),
                    "benchmark_eval_mask_path": str(artifacts["benchmark_eval_mask_path"]),
                    "eval_mask_path": str(artifacts["eval_mask_path"]),
                    "training_mask_path": str(artifacts["training_mask_path"]),
                    "uncertainty_mask_path": str(artifacts["uncertainty_mask_path"]),
                    "mask_variant_paths": dict(artifacts["mask_variant_paths"]),
                    "mask_variant_overlay_paths": dict(artifacts["mask_variant_overlay_paths"]),
                    "qc": artifacts["qc"],
                    "reselected_from_cached_candidates": True,
                    "auto_mask_fingerprint": _auto_fingerprint(config),
                }
            )
            row["settings"] = settings
            ground_truth_source = str(auto.get("ground_truth_mask_variant", "eval_tight"))
            pending_ground_truth.append(
                (
                    Path(_variant_or_default(artifacts, ground_truth_source, "refined_mask_path")),
                    Path(str(row["mask_path"])),
                )
            )
            rows.append(row)
            with pending_metadata_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    except BaseException as exc:
        partial_path = run_output_dir / "metadata.partial.jsonl"
        if pending_metadata_path.exists():
            pending_metadata_path.replace(partial_path)
        _write_json_atomic(
            run_status_path,
            {
                "run_id": run_id,
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "run_kind": "cached_candidate_reselection",
                "records_completed": len(rows),
                "partial_metadata_path": str(partial_path),
                "preserved_metadata_path": str(metadata_path),
                "run_output_dir": str(run_output_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    pending_metadata_path.replace(run_metadata_path)
    _publish_auto_mask_checkpoint(
        pending_ground_truth,
        run_metadata_path=run_metadata_path,
        stable_metadata_path=metadata_path,
        run_id=run_id,
    )
    pointer = {
        "run_id": run_id,
        "run_kind": "cached_candidate_reselection",
        "status": "complete",
        "run_output_dir": str(run_output_dir),
        "run_metadata_path": str(run_metadata_path),
        "stable_metadata_path": str(metadata_path),
        "records": len(rows),
        "auto_mask_fingerprint": _auto_fingerprint(config),
        "source_cache_audit": source_cache_audit,
        "output_cache_audit": _candidate_cache_summary(rows),
    }
    _write_json_atomic(current_run_path, pointer)
    _write_json_atomic(
        run_status_path,
        {
            **pointer,
            "records_completed": len(rows),
            "metadata_path": str(metadata_path),
        },
    )
    record_rows = [_auto_mask_record_from_row(row) for row in rows]
    _write_summary(report_dir / "summary.md", record_rows, metadata_path, auto)
    _write_contact_sheet(report_dir / "contact_sheet_mask_variants_current.png", record_rows)
    _write_candidate_comparison_sheet(report_dir / "contact_sheet_candidate_comparison.png", record_rows)
    return metadata_path


def _cached_candidates_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = dict(row.get("settings", {}))
    selected_mode = str(settings.get("selected_refinement", ""))
    candidates: dict[str, dict[str, Any]] = {}
    for mode, path_value in settings.get("candidate_refined_paths", {}).items():
        mode = str(mode)
        refined_path = Path(str(path_value))
        if not refined_path.exists():
            continue
        name = refined_path.name
        prefix = name[: -len("_refined.png")] if name.endswith("_refined.png") else refined_path.stem
        box_path = refined_path.with_name(f"{prefix}_box.png")
        inpaint_path = refined_path.with_name(f"{prefix}_inpaint.png")
        parameters: dict[str, Any] = {"kind": mode}
        if mode == selected_mode:
            parameters.update(dict(settings.get("mask_parameters", {})))
        heatmap_path = settings.get("candidate_heatmap_paths", {}).get(mode)
        if heatmap_path:
            parameters["heatmap_path"] = str(heatmap_path)
        candidates[mode] = {
            "box_mask_path": str(box_path if box_path.exists() else refined_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path if inpaint_path.exists() else refined_path),
            "parameters": parameters,
            "qc": dict(settings.get("candidate_qc", {}).get(mode, {})),
            "score": float(settings.get("candidate_scores", {}).get(mode, -1.0)),
            "selected_refinement": mode,
        }
    return candidates


def _candidate_cache_audit(row: dict[str, Any]) -> dict[str, Any]:
    settings = dict(row.get("settings", {}))
    refined_paths = {
        str(mode): str(path)
        for mode, path in settings.get("candidate_refined_paths", {}).items()
        if str(path)
    }
    heatmap_paths = {
        str(mode): str(path)
        for mode, path in settings.get("candidate_heatmap_paths", {}).items()
        if str(path)
    }
    existing_candidate_modes = [mode for mode, path in refined_paths.items() if Path(path).exists()]
    missing_candidate_modes = [mode for mode, path in refined_paths.items() if not Path(path).exists()]
    existing_heatmap_modes = [mode for mode, path in heatmap_paths.items() if Path(path).exists()]
    missing_heatmap_modes = [mode for mode, path in heatmap_paths.items() if not Path(path).exists()]
    selected_eval_path = str(row.get("eval_mask_path") or settings.get("eval_mask_path") or "")
    selected_refined_path = str(row.get("refined_mask_path") or "")
    return {
        "recorded_candidate_paths": len(refined_paths),
        "existing_candidate_paths": len(existing_candidate_modes),
        "missing_candidate_paths": len(missing_candidate_modes),
        "existing_candidate_modes": existing_candidate_modes,
        "missing_candidate_modes": missing_candidate_modes,
        "recorded_heatmap_paths": len(heatmap_paths),
        "existing_heatmap_paths": len(existing_heatmap_modes),
        "missing_heatmap_paths": len(missing_heatmap_modes),
        "selected_eval_mask_exists": bool(selected_eval_path and Path(selected_eval_path).exists()),
        "selected_refined_mask_exists": bool(selected_refined_path and Path(selected_refined_path).exists()),
    }


def _candidate_cache_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    audits = [_candidate_cache_audit(row) for row in rows]
    return {
        "rows": len(rows),
        "recorded_candidate_paths": sum(int(audit["recorded_candidate_paths"]) for audit in audits),
        "existing_candidate_paths": sum(int(audit["existing_candidate_paths"]) for audit in audits),
        "missing_candidate_paths": sum(int(audit["missing_candidate_paths"]) for audit in audits),
        "rows_with_existing_candidates": sum(1 for audit in audits if int(audit["existing_candidate_paths"]) > 0),
        "rows_with_selected_eval_mask": sum(1 for audit in audits if bool(audit["selected_eval_mask_exists"])),
        "rows_with_selected_refined_mask": sum(1 for audit in audits if bool(audit["selected_refined_mask_exists"])),
    }


def _selected_mask_compatibility_candidate(
    row: dict[str, Any],
    *,
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> dict[str, Any] | None:
    settings = dict(row.get("settings", {}))
    selected_mode = str(settings.get("selected_refinement") or "selected_mask")
    path_candidates = [
        row.get("eval_mask_path"),
        settings.get("benchmark_eval_mask_path"),
        row.get("refined_mask_path"),
        settings.get("generation_core_mask_path"),
    ]
    refined_path = next((Path(str(path)) for path in path_candidates if path and Path(str(path)).exists()), None)
    if refined_path is None:
        return None

    box_path = Path(str(row.get("box_mask_path") or refined_path))
    inpaint_path = Path(str(row.get("inpaint_mask_path") or refined_path))
    mask = Image.open(refined_path).convert("L").point(lambda value: 255 if value > 0 else 0)
    qc = dict(settings.get("qc", {}))
    if not qc:
        qc = _qc_mask(
            mask,
            region,
            _padded_bbox(mask, image.size, 0) or region,
            image.size,
            str(row.get("defect_type", "")),
        )
    return {
        "box_mask_path": str(box_path if box_path.exists() else refined_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path if inpaint_path.exists() else refined_path),
        "parameters": {
            **dict(settings.get("mask_parameters", {})),
            "kind": selected_mode,
            "cache_repair": "selected_mask_compatibility_fallback",
            "oracle_regret_available": False,
        },
        "qc": qc,
        "score": float(settings.get("candidate_scores", {}).get(selected_mode, 1.0)),
        "selected_refinement": selected_mode,
    }


def write_residual_refined_bbox_masks(
    image: Image.Image,
    normal_path: Path | None,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    clip_to_surface: bool,
    text_hint: str,
    percentile: float,
    min_component_area: int,
    padding: int,
) -> dict[str, Any]:
    pixel_artifacts = write_pixel_refined_bbox_masks(
        image,
        category,
        defect_type,
        region,
        output_dir,
        stem,
        clip_to_surface=clip_to_surface,
        text_hint=text_hint,
        percentile=percentile,
    )
    pixel_mask = Image.open(pixel_artifacts["refined_mask_path"]).convert("L")
    residual_mask, residual_params = residual_refined_mask(
        image,
        normal_path,
        region,
        defect_type,
        text_hint=text_hint,
        percentile=percentile,
        min_component_area=min_component_area,
    )
    refined = _combine_and_filter_masks(pixel_mask, residual_mask, defect_type, min_component_area)
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))

    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    parameters = {
        **pixel_artifacts["parameters"],
        **residual_params,
        "kind": "normal_residual",
        "normal_path": str(normal_path) if normal_path else None,
        "shrunk_region_xyxy": shrunk_region,
        "surface_clipped": bool(clip_to_surface),
    }
    return {
        **pixel_artifacts,
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": parameters,
        "qc": qc,
    }


def write_linear_refined_bbox_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    clip_to_surface: bool,
    text_hint: str,
    percentile: float,
    padding: int,
) -> dict[str, Any]:
    pixel_artifacts = write_pixel_refined_bbox_masks(
        image,
        category,
        defect_type,
        region,
        output_dir,
        stem,
        clip_to_surface=clip_to_surface,
        text_hint=text_hint,
        percentile=percentile,
    )
    pixel_mask = Image.open(pixel_artifacts["refined_mask_path"]).convert("L")
    refined, linear_params = linearized_scratch_mask(pixel_mask, image.size, region, defect_type)
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))

    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    parameters = {
        **pixel_artifacts["parameters"],
        **linear_params,
        "surface_clipped": bool(clip_to_surface),
        "shrunk_region_xyxy": shrunk_region,
    }
    return {
        **pixel_artifacts,
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": parameters,
        "qc": qc,
    }


def write_multi_linear_refined_bbox_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    clip_to_surface: bool,
    text_hint: str,
    percentile: float,
    padding: int,
    max_lines: int,
) -> dict[str, Any]:
    pixel_artifacts = write_pixel_refined_bbox_masks(
        image,
        category,
        defect_type,
        region,
        output_dir,
        stem,
        clip_to_surface=clip_to_surface,
        text_hint=text_hint,
        percentile=percentile,
    )
    pixel_mask = Image.open(pixel_artifacts["refined_mask_path"]).convert("L")
    refined, params = multi_linearized_scratch_mask(pixel_mask, image.size, region, defect_type, max_lines=max_lines)
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))

    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    return {
        **pixel_artifacts,
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            **pixel_artifacts["parameters"],
            **params,
            "surface_clipped": bool(clip_to_surface),
            "shrunk_region_xyxy": shrunk_region,
        },
        "qc": qc,
    }


def write_sam_heatmap_bbox_masks(
    image: Image.Image,
    normal_path: Path | None,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = anomaly_heatmap(image, normal_path, region, text_hint)
    point_coords, point_labels = heatmap_prompt_points(
        heatmap,
        region,
        positive_count=int(auto.get("sam_positive_points", 3)),
        negative_count=int(auto.get("sam_negative_points", 2)),
    )
    sam_mask, sam_params = _predict_sam_mask(image, region, point_coords, point_labels, auto)
    if sam_mask is None:
        if not bool(auto.get("allow_heatmap_fallback", True)):
            raise RuntimeError("SAM2/SAM is unavailable and auto_masks.allow_heatmap_fallback=false")
        refined = heatmap_fallback_mask(heatmap, region, image.size, percentile=float(auto.get("heatmap_mask_percentile", 96.0)))
        sam_params = {"sam_provider": "heatmap_fallback", "sam_available": False}
    else:
        refined = sam_mask
    refined = _filter_heatmap_mask(refined, defect_type, int(auto.get("min_component_area", 12)))
    if refined.getbbox() is None:
        refined = heatmap_fallback_mask(heatmap, region, image.size, percentile=92.0)
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "sam2_heatmap",
            "normal_path": str(normal_path) if normal_path else None,
            "heatmap_path": str(heatmap_path),
            "prompt_points": point_coords,
            "prompt_labels": point_labels,
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **sam_params,
        },
        "qc": qc,
    }


def write_sam_prompt_regularized_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)

    heatmaps: list[np.ndarray] = []
    heat_sources: list[str] = []
    heat_params: dict[str, Any] = {}
    try:
        patchcore_heat, params = patchcore_guided_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            patch_size=int(auto.get("sam_regularized_patchcore_patch_size", auto.get("patchcore_guided_patch_size", 17))),
            stride=int(auto.get("sam_regularized_patchcore_stride", auto.get("patchcore_guided_stride", 6))),
            max_normals=int(auto.get("sam_regularized_max_normals", auto.get("patchcore_guided_max_normals", 16))),
            max_memory_patches=int(auto.get("sam_regularized_max_memory_patches", auto.get("patchcore_guided_max_memory_patches", 4096))),
            feature_cache=feature_cache,
        )
        heatmaps.append(patchcore_heat)
        heat_sources.append("patchcore_guided")
        heat_params.update({f"sam_regularized_patchcore_{key}": value for key, value in params.items()})
    except Exception as exc:
        heat_params["sam_regularized_patchcore_error"] = str(exc)
    try:
        residual_heat, params = nearest_normal_patch_residual_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            max_normals=int(auto.get("sam_regularized_max_normals", auto.get("nearest_normal_residual_max_normals", 20))),
            search_radius=int(auto.get("sam_regularized_search_radius", auto.get("nearest_normal_residual_search_radius", 18))),
            search_step=int(auto.get("sam_regularized_search_step", auto.get("nearest_normal_residual_search_step", 6))),
        )
        heatmaps.append(residual_heat)
        heat_sources.append("nearest_normal_residual")
        heat_params.update({f"sam_regularized_residual_{key}": value for key, value in params.items()})
    except Exception as exc:
        heat_params["sam_regularized_residual_error"] = str(exc)
    try:
        normal_path = _nearest_normal(image, normal_paths, image.size)
        anomaly, params = anomaly_heatmap(image, normal_path, region, text_hint)
        heatmaps.append(anomaly)
        heat_sources.append("anomaly_heatmap")
        heat_params.update({f"sam_regularized_anomaly_{key}": value for key, value in params.items()})
    except Exception as exc:
        heat_params["sam_regularized_anomaly_error"] = str(exc)

    if not heatmaps:
        raise ValueError("sam_prompt_regularized could not build any prompt heatmap")
    normalized = [_normalize_float(heat) for heat in heatmaps]
    if len(normalized) == 1:
        fused = normalized[0]
    else:
        stack = np.stack(normalized, axis=0)
        fused = _normalize_float(0.62 * np.max(stack, axis=0) + 0.38 * np.mean(stack, axis=0))

    point_coords, point_labels = heatmap_prompt_points(
        fused,
        region,
        positive_count=int(auto.get("sam_regularized_positive_points", 5)),
        negative_count=int(auto.get("sam_regularized_negative_points", 4)),
    )
    sam_mask, sam_params = _predict_sam_mask(image, region, point_coords, point_labels, auto)
    if sam_mask is None:
        if not bool(auto.get("allow_heatmap_fallback", True)):
            raise RuntimeError("SAM2/SAM is unavailable and auto_masks.allow_heatmap_fallback=false")
        refined = heatmap_fallback_mask(
            fused,
            region,
            image.size,
            percentile=float(auto.get("sam_regularized_fallback_percentile", auto.get("heatmap_mask_percentile", 94.0))),
        )
        sam_params = {"sam_provider": "heatmap_fallback", "sam_available": False}
    else:
        refined = sam_mask

    refined = _filter_heatmap_mask(refined, defect_type, int(auto.get("min_component_area", 12)))
    if refined.getbbox() is None:
        refined = heatmap_fallback_mask(fused, region, image.size, percentile=92.0)

    arr = np.asarray(refined.convert("L"), dtype=np.uint8) > 0
    left, top, right, bottom = region
    crop = arr[top:bottom, left:right]
    heat_crop = fused[top:bottom, left:right]
    if crop.size and crop.any():
        crop = _limit_mask_area(crop, heat_crop, max_area_fraction=float(auto.get("sam_regularized_max_box_fraction", 0.12)))
        arr = np.zeros_like(arr, dtype=bool)
        arr[top:bottom, left:right] = crop
        refined = Image.fromarray(arr.astype(np.uint8) * 255, mode="L").convert("L")

    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(fused, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "sam_prompt_regularized",
            "heatmap_path": str(heatmap_path),
            "heatmap_sources": heat_sources,
            "prompt_points": point_coords,
            "prompt_labels": point_labels,
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **sam_params,
        },
        "qc": qc,
    }


def write_vlm_point_sam2_bbox_masks(
    image: Image.Image,
    vlm_point: tuple[int, int],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    
    x1, y1, x2, y2 = region
    pad_w = int((x2 - x1) * 0.25)
    pad_h = int((y2 - y1) * 0.25)
    
    w, h = image.size
    px1, py1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    px2, py2 = min(w - 1, x2 + pad_w), min(h - 1, y2 + pad_h)
    
    point_coords = [
        [vlm_point[0], vlm_point[1]], # center (positive)
        [px1, py1], # top-left (negative)
        [px2, py1], # top-right (negative)
        [px1, py2], # bottom-left (negative)
        [px2, py2], # bottom-right (negative)
    ]
    point_labels = [1, 0, 0, 0, 0]
    
    sam_mask, sam_params = _predict_sam_mask(image, region, point_coords, point_labels, auto)
    if sam_mask is None:
        raise RuntimeError("SAM2 unavailable for VLM Point")
        
    refined = sam_mask
    refined = _filter_heatmap_mask(refined, defect_type, int(auto.get("min_component_area", 12)))
    if refined.getbbox() is None:
        raise RuntimeError("SAM2 returned empty mask for VLM Point")
        
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "vlm_point_sam2",
            "prompt_points": point_coords,
            "prompt_labels": point_labels,
            "shrunk_region_xyxy": shrunk_region,
            **sam_params,
        },
        "qc": qc,
    }


def write_soft_patch_bbox_masks(
    image: Image.Image,
    normal_path: Path | None,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = anomaly_heatmap(image, normal_path, region, text_hint)
    refined, patch_params = soft_patch_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("soft_patch_percentile", 93.0)),
        close_radius=int(auto.get("soft_patch_close_radius", 2)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_area_fraction=float(auto.get("soft_patch_max_box_fraction", 0.22)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "soft_patch",
            "normal_path": str(normal_path) if normal_path else None,
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **patch_params,
        },
        "qc": qc,
    }


def write_normal_anomaly_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    
    suppress_grain = bool(auto.get("normal_anomaly_suppress_parallel_texture", True))
    working_image = image
    working_normals = normal_paths
    if suppress_grain:
        dom_angle = _dominant_texture_angle_degrees(image, normal_paths, region)
        if dom_angle is not None:
            working_image = _apply_gabor_suppression(image, dom_angle)
            
    heatmap, heat_params = normal_texture_anomaly_heatmap(
        working_image,
        working_normals,
        region,
        text_hint,
        max_normals=int(auto.get("normal_anomaly_max_normals", 32)),
    )
    refined, anomaly_params = normal_anomaly_mask(
        heatmap,
        image,
        normal_paths,
        region,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("normal_anomaly_percentile", 96.0)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("normal_anomaly_max_components", 18)),
        dilate_radius=int(auto.get("normal_anomaly_dilate_radius", 1)),
        envelope=bool(auto.get("normal_anomaly_envelope", True)),
        envelope_mode=str(auto.get("normal_anomaly_envelope_mode", "adaptive")),
        envelope_radius=int(auto.get("normal_anomaly_envelope_radius", 5)),
        envelope_percentile=float(auto.get("normal_anomaly_envelope_percentile", 82.0)),
        max_box_fraction=float(auto.get("normal_anomaly_max_box_fraction", 0.18)),
        suppress_parallel_texture=bool(auto.get("normal_anomaly_suppress_parallel_texture", True)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.3))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "normal_texture_anomaly",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **anomaly_params,
        },
        "qc": qc,
    }


def write_nearest_normal_residual_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = nearest_normal_patch_residual_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        max_normals=int(auto.get("nearest_normal_residual_max_normals", 32)),
        search_radius=int(auto.get("nearest_normal_residual_search_radius", 18)),
        search_step=int(auto.get("nearest_normal_residual_search_step", 6)),
    )
    refined, residual_params = nearest_normal_patch_residual_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("nearest_normal_residual_percentile", 88.0)),
        close_radius=int(auto.get("nearest_normal_residual_close_radius", 2)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("nearest_normal_residual_max_components", 14)),
        max_box_fraction=float(auto.get("nearest_normal_residual_max_box_fraction", 0.12)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "nearest_normal_patch_residual",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **residual_params,
        },
        "qc": qc,
    }


def write_patchcore_guided_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = patchcore_guided_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        patch_size=int(auto.get("patchcore_guided_patch_size", 17)),
        stride=int(auto.get("patchcore_guided_stride", 6)),
        max_normals=int(auto.get("patchcore_guided_max_normals", 16)),
        max_memory_patches=int(auto.get("patchcore_guided_max_memory_patches", 4096)),
        feature_cache=feature_cache,
    )
    refined, mask_params = patchcore_guided_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("patchcore_guided_percentile", 88.0)),
        close_radius=int(auto.get("patchcore_guided_close_radius", 2)),
        dilate_radius=int(auto.get("patchcore_guided_dilate_radius", 1)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("patchcore_guided_max_components", 14)),
        max_box_fraction=float(auto.get("patchcore_guided_max_box_fraction", 0.12)),
        min_contrast=float(auto.get("patchcore_guided_min_contrast", 0.025)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "patchcore_guided",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **mask_params,
        },
        "qc": qc,
    }


def foundation_anomaly_field_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    text_hint: str,
    *,
    auto: dict[str, Any],
    feature_cache: PatchFeatureCache | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fuse zero-shot normal-memory and texture residuals into one field."""

    heatmaps: list[np.ndarray] = []
    weights: list[float] = []
    sources: list[str] = []
    failures: dict[str, str] = {}

    def add_source(name: str, weight: float, builder: Any) -> None:
        if weight <= 0.0:
            return
        try:
            heat, params = builder()
            heatmaps.append(_robust_region_calibrated_heatmap(heat, region))
            weights.append(float(weight))
            sources.append(name)
            for key, value in params.items():
                if key not in failures:
                    failures[f"{name}_{key}"] = value
        except Exception as exc:
            failures[f"{name}_error"] = str(exc)

    add_source(
        "patchcore_guided",
        float(auto.get("foundation_patchcore_weight", 0.34)),
        lambda: patchcore_guided_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            patch_size=int(auto.get("foundation_patchcore_patch_size", auto.get("patchcore_guided_patch_size", 17))),
            stride=int(auto.get("foundation_patchcore_stride", auto.get("patchcore_guided_stride", 6))),
            max_normals=int(auto.get("foundation_max_normals", auto.get("patchcore_guided_max_normals", 16))),
            max_memory_patches=int(auto.get("foundation_max_memory_patches", auto.get("patchcore_guided_max_memory_patches", 4096))),
            feature_cache=feature_cache,
        ),
    )
    add_source(
        "musc_mutual_score",
        float(auto.get("foundation_musc_weight", 0.24)),
        lambda: musc_mutual_score_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            patch_size=int(auto.get("foundation_musc_patch_size", auto.get("musc_patch_size", auto.get("patchcore_guided_patch_size", 17)))),
            stride=int(auto.get("foundation_musc_stride", auto.get("musc_stride", auto.get("patchcore_guided_stride", 6)))),
            max_normals=int(auto.get("foundation_max_normals", auto.get("musc_max_normals", 16))),
            max_memory_patches=int(auto.get("foundation_max_memory_patches", auto.get("musc_max_memory_patches", 4096))),
            knn_k=int(auto.get("foundation_musc_knn_k", auto.get("musc_knn_k", 5))),
            feature_cache=feature_cache,
        ),
    )
    add_source(
        "nearest_normal_residual",
        float(auto.get("foundation_nearest_residual_weight", 0.20)),
        lambda: nearest_normal_patch_residual_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            max_normals=int(auto.get("foundation_max_normals", auto.get("nearest_normal_residual_max_normals", 20))),
            search_radius=int(auto.get("foundation_search_radius", auto.get("nearest_normal_residual_search_radius", 18))),
            search_step=int(auto.get("foundation_search_step", auto.get("nearest_normal_residual_search_step", 6))),
        ),
    )
    add_source(
        "normal_texture_anomaly",
        float(auto.get("foundation_normal_texture_weight", 0.14)),
        lambda: normal_texture_anomaly_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            max_normals=int(auto.get("foundation_max_normals", auto.get("normal_anomaly_max_normals", 16))),
        ),
    )
    add_source(
        "fft_texture_suppression",
        float(auto.get("foundation_fft_weight", 0.08)),
        lambda: fft_texture_suppression_heatmap(
            image,
            region,
            peak_count=int(auto.get("foundation_fft_peak_count", auto.get("fft_peak_count", 10))),
            notch_radius=int(auto.get("foundation_fft_notch_radius", auto.get("fft_notch_radius", 3))),
            low_frequency_radius=int(auto.get("foundation_fft_low_frequency_radius", auto.get("fft_low_frequency_radius", 5))),
        ),
    )

    if not heatmaps:
        raise ValueError(f"foundation_anomaly_field could not build any heatmap: {failures}")
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_arr = weight_arr / max(1e-6, float(weight_arr.sum()))
    stack = np.stack(heatmaps, axis=0)
    weighted = np.sum(stack * weight_arr[:, None, None], axis=0)
    maximum = np.max(stack, axis=0)
    agreement = np.mean(stack >= 0.62, axis=0).astype(np.float32)
    fused = _normalize_float(0.58 * weighted + 0.27 * maximum + 0.15 * agreement)

    left, top, right, bottom = region
    fenced = np.zeros_like(fused, dtype=np.float32)
    fenced[top:bottom, left:right] = fused[top:bottom, left:right]
    return fenced, {
        "foundation_model": "zero_shot_normal_memory_texture_field",
        "foundation_sources": sources,
        "foundation_source_count": len(sources),
        "foundation_weights": {source: float(weight) for source, weight in zip(sources, weight_arr.tolist(), strict=False)},
        "foundation_failures": {key: value for key, value in failures.items() if str(key).endswith("_error")},
        "foundation_region_contrast": _heatmap_region_contrast(fenced, region),
    }


def write_foundation_anomaly_field_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = foundation_anomaly_field_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        auto=auto,
        feature_cache=feature_cache,
    )
    refined, mask_params = patchcore_guided_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("foundation_percentile", auto.get("patchcore_guided_percentile", 88.0))),
        close_radius=int(auto.get("foundation_close_radius", auto.get("patchcore_guided_close_radius", 2))),
        dilate_radius=int(auto.get("foundation_dilate_radius", auto.get("patchcore_guided_dilate_radius", 1))),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("foundation_max_components", auto.get("patchcore_guided_max_components", 14))),
        max_box_fraction=float(auto.get("foundation_max_box_fraction", auto.get("patchcore_guided_max_box_fraction", 0.12))),
        min_contrast=float(auto.get("foundation_min_contrast", 0.015)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_foundation_anomaly_heatmap.png"
    candidate_path = output_dir / f"{stem}_foundation_anomaly_candidate.png"
    overlay_path = output_dir / f"{stem}_foundation_anomaly_overlay.png"
    box.save(box_path)
    refined.save(refined_path)
    refined.save(candidate_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    _write_heatmap_overlay(image, heatmap, region, refined, overlay_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "foundation_anomaly_field",
            "heatmap_path": str(heatmap_path),
            "foundation_candidate_path": str(candidate_path),
            "foundation_overlay_path": str(overlay_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **{f"foundation_{key}": value for key, value in mask_params.items()},
        },
        "qc": qc,
    }


def write_repeated_chain_refiner_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = patchcore_guided_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        patch_size=int(auto.get("repeated_chain_patch_size", auto.get("patchcore_guided_patch_size", 17))),
        stride=int(auto.get("repeated_chain_stride", auto.get("patchcore_guided_stride", 6))),
        max_normals=int(auto.get("repeated_chain_max_normals", auto.get("patchcore_guided_max_normals", 16))),
        max_memory_patches=int(
            auto.get("repeated_chain_max_memory_patches", auto.get("patchcore_guided_max_memory_patches", 4096))
        ),
        feature_cache=feature_cache,
    )
    refined, mask_params = repeated_chain_refiner_mask(
        heatmap,
        region,
        image.size,
        orientation_degrees=_estimate_repeated_chain_orientation(image, region),
        percentile=float(auto.get("repeated_chain_percentile", 86.0)),
        corridor_width_fraction=float(auto.get("repeated_chain_corridor_width_fraction", 0.34)),
        max_window_height=int(auto.get("repeated_chain_max_window_height_px", 132)),
        min_window_height=int(auto.get("repeated_chain_min_window_height_px", 52)),
        close_radius=int(auto.get("repeated_chain_close_radius", 2)),
        dilate_radius=int(auto.get("repeated_chain_dilate_radius", 1)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("repeated_chain_max_components", 8)),
        max_box_fraction=float(auto.get("repeated_chain_max_box_fraction", 0.085)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.2))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "repeated_chain_refiner",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **mask_params,
        },
        "qc": qc,
    }


def write_polar_rim_residual_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = nearest_normal_patch_residual_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        max_normals=int(auto.get("polar_rim_max_normals", 16)),
        search_radius=int(auto.get("polar_rim_search_radius", 8)),
        search_step=int(auto.get("polar_rim_search_step", 4)),
    )
    refined, mask_params = polar_rim_residual_mask(
        heatmap,
        region,
        image.size,
        angle_bins=int(auto.get("polar_rim_angle_bins", 180)),
        inner_radius_fraction=float(auto.get("polar_rim_inner_radius_fraction", 0.52)),
        outer_radius_fraction=float(auto.get("polar_rim_outer_radius_fraction", 1.02)),
        profile_percentile=float(auto.get("polar_rim_profile_percentile", 72.0)),
        support_percentile=float(auto.get("polar_rim_support_percentile", 55.0)),
        min_sector_degrees=float(auto.get("polar_rim_min_sector_degrees", 10.0)),
        max_sector_degrees=float(auto.get("polar_rim_max_sector_degrees", 70.0)),
        close_radius=int(auto.get("polar_rim_close_radius", 2)),
        dilate_radius=int(auto.get("polar_rim_dilate_radius", 2)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_box_fraction=float(auto.get("polar_rim_max_box_fraction", 0.16)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.2))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "polar_rim_residual",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **mask_params,
        },
        "qc": qc,
    }


def _estimate_repeated_chain_orientation(
    image: Image.Image,
    region: tuple[int, int, int, int],
) -> float:
    import scipy.ndimage as ndimage

    left, top, right, bottom = region
    crop = np.asarray(image.convert("L"), dtype=np.float32)[top:bottom, left:right]
    fallback = 90.0 if crop.shape[0] >= crop.shape[1] else 0.0
    if min(crop.shape, default=0) < 8:
        return fallback

    gradient_y = ndimage.sobel(crop, axis=0, mode="reflect")
    gradient_x = ndimage.sobel(crop, axis=1, mode="reflect")
    strength = np.hypot(gradient_x, gradient_y)
    threshold = float(np.percentile(strength, 78.0))
    ys, xs = np.where(strength >= threshold)
    if len(xs) < 24:
        return fallback

    weights = strength[ys, xs].astype(np.float64)
    coordinates = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    center = np.average(coordinates, axis=0, weights=weights)
    centered = coordinates - center
    covariance = (centered * weights[:, None]).T @ centered / max(1e-6, float(weights.sum()))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= 1e-6 or float(eigenvalues[-1] / max(1e-6, eigenvalues[-2])) < 1.15:
        return fallback
    vector = eigenvectors[:, -1]
    angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
    return angle


def write_musc_mutual_score_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = musc_mutual_score_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        patch_size=int(auto.get("musc_patch_size", auto.get("patchcore_guided_patch_size", 17))),
        stride=int(auto.get("musc_stride", auto.get("patchcore_guided_stride", 6))),
        max_normals=int(auto.get("musc_max_normals", auto.get("patchcore_guided_max_normals", 16))),
        max_memory_patches=int(auto.get("musc_max_memory_patches", auto.get("patchcore_guided_max_memory_patches", 4096))),
        knn_k=int(auto.get("musc_knn_k", 5)),
        feature_cache=feature_cache,
    )
    refined, mask_params = patchcore_guided_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("musc_percentile", auto.get("patchcore_guided_percentile", 88.0))),
        close_radius=int(auto.get("musc_close_radius", auto.get("patchcore_guided_close_radius", 2))),
        dilate_radius=int(auto.get("musc_dilate_radius", auto.get("patchcore_guided_dilate_radius", 1))),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("musc_max_components", auto.get("patchcore_guided_max_components", 14))),
        max_box_fraction=float(auto.get("musc_max_box_fraction", auto.get("patchcore_guided_max_box_fraction", 0.12))),
        min_contrast=float(auto.get("musc_min_contrast", auto.get("patchcore_guided_min_contrast", 0.025))),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "musc_mutual_score",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **{f"musc_{key}": value for key, value in mask_params.items()},
        },
        "qc": qc,
    }


def write_scuff_cluster_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_artifacts = write_normal_anomaly_bbox_masks(
        image,
        normal_paths,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_normal_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    soft_artifacts = write_soft_patch_bbox_masks(
        image,
        _nearest_normal(image, normal_paths, image.size),
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_soft_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    normal_mask = Image.open(normal_artifacts["refined_mask_path"]).convert("L")
    soft_mask = Image.open(soft_artifacts["refined_mask_path"]).convert("L")
    refined, params = scuff_cluster_mask(
        normal_mask,
        soft_mask,
        region,
        image.size,
        min_component_area=int(auto.get("min_component_area", 12)),
        close_radius=int(auto.get("scuff_cluster_close_radius", 3)),
        max_components=int(auto.get("scuff_cluster_max_components", 14)),
        max_box_fraction=float(auto.get("scuff_cluster_max_box_fraction", 0.14)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    Image.open(normal_artifacts["box_mask_path"]).save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "scuff_cluster",
            "normal_seed_path": normal_artifacts["refined_mask_path"],
            "soft_seed_path": soft_artifacts["refined_mask_path"],
            "shrunk_region_xyxy": shrunk_region,
            **params,
        },
        "qc": qc,
    }


def write_fft_texture_suppression_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = fft_texture_suppression_heatmap(
        image,
        region,
        peak_count=int(auto.get("fft_peak_count", 10)),
        notch_radius=int(auto.get("fft_notch_radius", 3)),
        low_frequency_radius=int(auto.get("fft_low_frequency_radius", 5)),
    )
    fft_mask, fft_params = fft_texture_suppression_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("fft_texture_percentile", 88.0)),
        close_radius=int(auto.get("fft_texture_close_radius", 3)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_area_fraction=float(auto.get("fft_texture_max_box_fraction", 0.18)),
        min_contrast=float(auto.get("fft_texture_min_contrast", 0.08)),
    )

    if bool(auto.get("fft_texture_blend_with_scuff_seeds", True)):
        try:
            scuff_artifacts = write_scuff_cluster_bbox_masks(
                image,
                normal_paths,
                category,
                defect_type,
                region,
                output_dir,
                f"{stem}_scuff_seed",
                auto=auto,
                text_hint=text_hint,
                padding=padding,
            )
            scuff_mask = Image.open(scuff_artifacts["refined_mask_path"]).convert("L")
            fft_arr = np.asarray(fft_mask, dtype=np.uint8) > 0
            scuff_arr = np.asarray(scuff_mask, dtype=np.uint8) > 0
            crop = (fft_arr | (scuff_arr & _dilate_bool(fft_arr, radius=4)))[region[1] : region[3], region[0] : region[2]]
            heat_crop = heatmap[region[1] : region[3], region[0] : region[2]]
            crop = _morph_close(crop, radius=max(1, int(auto.get("fft_texture_close_radius", 3))))
            crop = _limit_mask_area(crop, heat_crop, max_area_fraction=float(auto.get("fft_texture_max_box_fraction", 0.18)))
            blended = Image.new("L", image.size, 0)
            blended.paste(Image.fromarray(crop.astype(np.uint8) * 255, mode="L"), (region[0], region[1]))
            if blended.getbbox() is not None:
                fft_mask = blended
                fft_params["fft_texture_scuff_seed_path"] = scuff_artifacts["refined_mask_path"]
                fft_params["fft_texture_blended_with_scuff_seed"] = True
        except Exception as exc:
            fft_params["fft_texture_scuff_seed_error"] = str(exc)
            fft_params["fft_texture_blended_with_scuff_seed"] = False

    shrunk_region = _padded_bbox(fft_mask, image.size, padding) or region
    qc = _qc_mask(fft_mask, region, shrunk_region, image.size, defect_type)
    inpaint = fft_mask.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    fft_mask.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "fft_texture_suppression",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **fft_params,
        },
        "qc": qc,
    }


def write_structure_tensor_ridge_bbox_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = structure_tensor_ridge_heatmap(
        image,
        region,
        sigma=float(auto.get("structure_tensor_sigma", 1.2)),
        residual_blur=float(auto.get("structure_tensor_residual_blur", 5.0)),
    )
    refined, ridge_params = structure_tensor_ridge_mask(
        heatmap,
        region,
        image.size,
        defect_type,
        percentile=float(auto.get("structure_tensor_percentile", 91.0)),
        close_radius=int(auto.get("structure_tensor_close_radius", 1)),
        dilate_radius=int(auto.get("structure_tensor_dilate_radius", 1)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("structure_tensor_max_components", 10)),
        max_area_fraction=float(auto.get("structure_tensor_max_box_fraction", 0.12)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.3))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "structure_tensor_ridge",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **ridge_params,
        },
        "qc": qc,
    }


def write_multi_scuff_fusion_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_artifacts = write_normal_anomaly_bbox_masks(
        image,
        normal_paths,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_normal_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    soft_artifacts = write_soft_patch_bbox_masks(
        image,
        _nearest_normal(image, normal_paths, image.size),
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_soft_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    fft_heatmap, fft_heat_params = fft_texture_suppression_heatmap(
        image,
        region,
        peak_count=int(auto.get("fft_peak_count", 10)),
        notch_radius=int(auto.get("fft_notch_radius", 3)),
        low_frequency_radius=int(auto.get("fft_low_frequency_radius", 5)),
    )
    normal_mask = Image.open(normal_artifacts["refined_mask_path"]).convert("L")
    soft_mask = Image.open(soft_artifacts["refined_mask_path"]).convert("L")
    refined, params = multi_scuff_fusion_mask(
        normal_mask,
        soft_mask,
        fft_heatmap,
        image,
        normal_paths,
        region,
        image.size,
        text_hint=text_hint,
        min_component_area=int(auto.get("min_component_area", 12)),
        support_percentile=float(auto.get("multi_scuff_fusion_support_percentile", 76.0)),
        fine_dilate_radius=int(auto.get("multi_scuff_fusion_fine_dilate_radius", 5)),
        close_radius=int(auto.get("multi_scuff_fusion_close_radius", 2)),
        max_components=int(auto.get("multi_scuff_fusion_max_components", 10)),
        max_box_fraction=float(auto.get("multi_scuff_fusion_max_box_fraction", 0.11)),
        grain_reject_angle_degrees=float(auto.get("multi_scuff_fusion_grain_reject_angle_degrees", 14.0)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_fft_support_heatmap.png"
    Image.open(normal_artifacts["box_mask_path"]).save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(fft_heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "multi_scuff_fusion",
            "normal_seed_path": normal_artifacts["refined_mask_path"],
            "soft_seed_path": soft_artifacts["refined_mask_path"],
            "fft_support_heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **fft_heat_params,
            **params,
        },
        "qc": qc,
    }


def write_normal_residual_fusion_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_artifacts = write_multi_scuff_fusion_bbox_masks(
        image,
        normal_paths,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_fusion_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    residual_artifacts = write_nearest_normal_residual_bbox_masks(
        image,
        normal_paths,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_nearest_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    fusion_mask = Image.open(fusion_artifacts["refined_mask_path"]).convert("L")
    residual_mask = Image.open(residual_artifacts["refined_mask_path"]).convert("L")
    residual_heatmap_path = residual_artifacts.get("parameters", {}).get("heatmap_path")
    if residual_heatmap_path and Path(str(residual_heatmap_path)).exists():
        residual_heat = np.asarray(Image.open(str(residual_heatmap_path)).convert("L"), dtype=np.float32) / 255.0
    else:
        residual_heat = np.asarray(residual_mask, dtype=np.float32) / 255.0
    refined, params = normal_residual_fusion_mask(
        fusion_mask,
        residual_mask,
        residual_heat,
        image,
        normal_paths,
        region,
        image.size,
        min_component_area=int(auto.get("min_component_area", 12)),
        support_dilate_radius=int(auto.get("normal_residual_fusion_support_dilate_radius", 4)),
        close_radius=int(auto.get("normal_residual_fusion_close_radius", 1)),
        max_components=int(auto.get("normal_residual_fusion_max_components", 12)),
        max_box_fraction=float(auto.get("normal_residual_fusion_max_box_fraction", 0.12)),
        grain_reject_angle_degrees=float(auto.get("normal_residual_fusion_grain_reject_angle_degrees", 16.0)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_nearest_support_heatmap.png"
    Image.open(fusion_artifacts["box_mask_path"]).save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(residual_heat, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "normal_residual_fusion",
            "fusion_seed_path": fusion_artifacts["refined_mask_path"],
            "nearest_seed_path": residual_artifacts["refined_mask_path"],
            "nearest_support_heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **params,
        },
        "qc": qc,
    }


def write_zipper_fabric_border_layout_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    heatmap, heat_params = patchcore_guided_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        patch_size=int(auto.get("zipper_fabric_layout_patch_size", auto.get("patchcore_guided_patch_size", 17))),
        stride=int(auto.get("zipper_fabric_layout_stride", auto.get("patchcore_guided_stride", 6))),
        max_normals=int(auto.get("zipper_fabric_layout_max_normals", auto.get("patchcore_guided_max_normals", 16))),
        max_memory_patches=int(
            auto.get("zipper_fabric_layout_max_memory_patches", auto.get("patchcore_guided_max_memory_patches", 4096))
        ),
        feature_cache=feature_cache,
    )
    refined, mask_params = zipper_fabric_border_layout_mask(
        heatmap,
        region,
        image.size,
        percentile=float(auto.get("zipper_fabric_layout_percentile", 90.0)),
        close_radius=int(auto.get("zipper_fabric_layout_close_radius", 2)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("zipper_fabric_layout_max_components", 8)),
        max_image_fraction=float(auto.get("zipper_fabric_layout_max_image_fraction", 0.014)),
        side_fractions=auto.get("zipper_fabric_layout_side_fractions", [0.08, 0.34, 0.66, 0.94]),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "zipper_fabric_border_layout",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **mask_params,
        },
        "qc": qc,
    }


def write_support_constrained_fusion_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_artifacts = write_multi_scuff_fusion_bbox_masks(
        image,
        normal_paths,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_base",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    heatmap, heat_params = support_anomaly_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        auto=auto,
    )
    base_mask = Image.open(base_artifacts["refined_mask_path"]).convert("L")
    refined, params = support_constrained_fusion_mask(
        base_mask,
        heatmap,
        image,
        normal_paths,
        region,
        image.size,
        min_component_area=int(auto.get("min_component_area", 12)),
        support_percentile=float(auto.get("support_constrained_percentile", 78.0)),
        support_dilate_radius=int(auto.get("support_constrained_dilate_radius", 5)),
        min_support_overlap=float(auto.get("support_constrained_min_overlap", 0.10)),
        min_component_heat=float(auto.get("support_constrained_min_heat", 0.34)),
        min_keep_fraction=float(auto.get("support_constrained_min_keep_fraction", 0.55)),
        grain_reject_angle_degrees=float(auto.get("support_constrained_grain_reject_angle_degrees", 14.0)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.3))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_support_heatmap.png"
    Image.open(base_artifacts["box_mask_path"]).save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "support_constrained_fusion",
            "base_mask_path": base_artifacts["refined_mask_path"],
            "support_heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **params,
        },
        "qc": qc,
    }


def write_scratch_band_clean_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normal_artifacts = write_normal_anomaly_bbox_masks(
        image,
        normal_paths,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_normal_seed",
        auto=auto,
        text_hint=text_hint,
        padding=padding,
    )
    multi_artifacts = write_multi_linear_refined_bbox_masks(
        image,
        category,
        defect_type,
        region,
        output_dir,
        f"{stem}_multi_seed",
        clip_to_surface=bool(auto.get("clip_to_surface", False)),
        text_hint=text_hint,
        percentile=float(auto.get("pixel_refine_percentile", 93.0)),
        padding=padding,
        max_lines=int(auto.get("multi_linear_max_lines", 6)),
    )
    normal_mask = Image.open(normal_artifacts["refined_mask_path"]).convert("L")
    multi_mask = Image.open(multi_artifacts["refined_mask_path"]).convert("L")
    refined, params = scratch_band_clean_mask(
        normal_mask,
        multi_mask,
        region,
        image.size,
        radius=int(auto.get("scratch_band_clean_radius", 7)),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_area_fraction=float(auto.get("scratch_band_clean_max_box_fraction", 0.09)),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    Image.open(normal_artifacts["box_mask_path"]).save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "scratch_band_clean",
            "normal_seed_path": normal_artifacts["refined_mask_path"],
            "multi_seed_path": multi_artifacts["refined_mask_path"],
            "shrunk_region_xyxy": shrunk_region,
            **params,
        },
        "qc": qc,
    }


def write_dinov2_memory_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    try:
        heatmap, heat_params = dinov2_memory_heatmap(
            image,
            normal_paths,
            region,
            model_id=str(auto.get("dinov2_model", "facebook/dinov2-small")),
            cache_dir=auto.get("dinov2_cache_dir", auto.get("qwen_cache_dir")),
            local_files_only=bool(auto.get("dinov2_local_files_only", False)),
            device=str(auto.get("dinov2_device", auto.get("qwen_device", "auto"))),
            max_normals=int(auto.get("dinov2_max_normals", 16)),
            memory_stride=int(auto.get("dinov2_memory_stride", 1)),
        )
    except Exception as exc:
        if not bool(auto.get("dinov2_allow_fallback", True)):
            raise
        heatmap, fallback_params = normal_texture_anomaly_heatmap(
            image,
            normal_paths,
            region,
            text_hint,
            max_normals=int(auto.get("normal_anomaly_max_normals", 32)),
        )
        heat_params = {
            **fallback_params,
            "dinov2_provider": "fallback_normal_texture",
            "dinov2_available": False,
            "dinov2_error": str(exc),
        }
    refined, anomaly_params = normal_anomaly_mask(
        heatmap,
        image,
        normal_paths,
        region,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("dinov2_percentile", auto.get("normal_anomaly_percentile", 96.0))),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("dinov2_max_components", auto.get("normal_anomaly_max_components", 18))),
        dilate_radius=int(auto.get("dinov2_dilate_radius", auto.get("normal_anomaly_dilate_radius", 1))),
        envelope=bool(auto.get("dinov2_envelope", auto.get("normal_anomaly_envelope", True))),
        envelope_mode=str(auto.get("dinov2_envelope_mode", auto.get("normal_anomaly_envelope_mode", "adaptive"))),
        envelope_radius=int(auto.get("dinov2_envelope_radius", auto.get("normal_anomaly_envelope_radius", 5))),
        envelope_percentile=float(auto.get("dinov2_envelope_percentile", auto.get("normal_anomaly_envelope_percentile", 82.0))),
        max_box_fraction=float(auto.get("dinov2_max_box_fraction", auto.get("normal_anomaly_max_box_fraction", 0.18))),
        suppress_parallel_texture=bool(auto.get("dinov2_suppress_parallel_texture", auto.get("normal_anomaly_suppress_parallel_texture", True))),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.3))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "dinov2_memory",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **{f"dinov2_{key.removeprefix('normal_anomaly_')}": value for key, value in anomaly_params.items() if key.startswith("normal_anomaly_")},
        },
        "qc": qc,
    }


def write_dinov2_fusion_bbox_masks(
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    dino_heat, dino_params = dinov2_memory_heatmap(
        image,
        normal_paths,
        region,
        model_id=str(auto.get("dinov2_model", "facebook/dinov2-small")),
        cache_dir=auto.get("dinov2_cache_dir", auto.get("qwen_cache_dir")),
        local_files_only=bool(auto.get("dinov2_local_files_only", False)),
        device=str(auto.get("dinov2_device", auto.get("qwen_device", "auto"))),
        max_normals=int(auto.get("dinov2_max_normals", 16)),
        memory_stride=int(auto.get("dinov2_memory_stride", 1)),
    )
    texture_heat, texture_params = normal_texture_anomaly_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        max_normals=int(auto.get("normal_anomaly_max_normals", 32)),
    )
    dino_weight = float(auto.get("dinov2_fusion_dino_weight", 0.55))
    texture_weight = 1.0 - dino_weight
    heatmap = _normalize_float(np.maximum(dino_heat * dino_weight, texture_heat * texture_weight))
    refined, anomaly_params = normal_anomaly_mask(
        heatmap,
        image,
        normal_paths,
        region,
        defect_type,
        text_hint=text_hint,
        percentile=float(auto.get("dinov2_fusion_percentile", auto.get("dinov2_percentile", 95.0))),
        min_component_area=int(auto.get("min_component_area", 12)),
        max_components=int(auto.get("dinov2_fusion_max_components", auto.get("dinov2_max_components", 24))),
        dilate_radius=int(auto.get("dinov2_fusion_dilate_radius", auto.get("dinov2_dilate_radius", 1))),
        envelope=bool(auto.get("dinov2_fusion_envelope", auto.get("dinov2_envelope", True))),
        envelope_mode=str(auto.get("dinov2_fusion_envelope_mode", auto.get("dinov2_envelope_mode", auto.get("normal_anomaly_envelope_mode", "adaptive")))),
        envelope_radius=int(auto.get("dinov2_fusion_envelope_radius", auto.get("dinov2_envelope_radius", 5))),
        envelope_percentile=float(auto.get("dinov2_fusion_envelope_percentile", auto.get("dinov2_envelope_percentile", 80.0))),
        max_box_fraction=float(auto.get("dinov2_fusion_max_box_fraction", auto.get("dinov2_max_box_fraction", 0.22))),
        suppress_parallel_texture=bool(auto.get("dinov2_fusion_suppress_parallel_texture", auto.get("dinov2_suppress_parallel_texture", True))),
    )
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.3))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "dinov2_texture_fusion",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            "dinov2_fusion_dino_weight": dino_weight,
            "dinov2_fusion_texture_weight": texture_weight,
            **dino_params,
            **{f"fusion_texture_{key}": value for key, value in texture_params.items()},
            **{f"dinov2_fusion_{key.removeprefix('normal_anomaly_')}": value for key, value in anomaly_params.items() if key.startswith("normal_anomaly_")},
        },
        "qc": qc,
    }


_SD_PIPELINE_CACHE: dict[tuple[str, str | None, bool, str, str], Any] = {}


def write_delta_deno_bbox_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    auto: dict[str, Any],
    text_hint: str,
    padding: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    suppress_grain = bool(auto.get("normal_anomaly_suppress_parallel_texture", True))
    working_image = image
    if suppress_grain:
        dom_angle = _dominant_texture_angle_degrees(image, [], region)
        if dom_angle is not None:
            working_image = _apply_gabor_suppression(image, dom_angle)

    heatmap, heat_params = delta_deno_heatmap(
        working_image, region, category, defect_type, auto, text_hint
    )
    _validate_delta_deno_heatmap(heatmap, region, auto)
    
    # Extract peak points from the semantic cross-attention heatmap
    point_coords, point_labels = heatmap_prompt_points(
        heatmap,
        region,
        positive_count=int(auto.get("sam_positive_points", 3)),
        negative_count=int(auto.get("sam_negative_points", 2)),
    )
    
    # Query SAM2 with the semantic points
    sam_mask, patch_params = _predict_sam_mask(image, region, point_coords, point_labels, auto)
    
    if sam_mask is None:
        if not bool(auto.get("allow_heatmap_fallback", True)):
            raise RuntimeError("SAM2/SAM is unavailable and auto_masks.allow_heatmap_fallback=false")
        refined = heatmap_fallback_mask(heatmap, region, image.size, percentile=float(auto.get("heatmap_mask_percentile", 96.0)))
        patch_params = {"sam_provider": "heatmap_fallback", "sam_available": False}
    else:
        # Phase 7: Context-Aware Fusion
        # Extract the softer semantic context (halo) from the heatmap
        context_mask = heatmap_fallback_mask(heatmap, region, image.size, percentile=float(auto.get("delta_deno_percentile", 90.0)))
        # Fuse the razor-sharp SAM2 core with the softer contextual halo
        from PIL import ImageMath
        refined = ImageMath.eval("convert(max(a, b), 'L')", a=sam_mask, b=context_mask)
        
    refined = _filter_heatmap_mask(refined, defect_type, int(auto.get("min_component_area", 12)))
    if refined.getbbox() is None:
        refined = heatmap_fallback_mask(heatmap, region, image.size, percentile=92.0)
    
    shrunk_region = _padded_bbox(refined, image.size, padding) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L").save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "delta_deno",
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
            **heat_params,
            **patch_params,
        },
        "qc": qc,
    }


class StoreAttnProcessor(object):
    def __init__(self, target_indices):
        self.target_indices = target_indices
        self.attn_maps = []
        
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        import torch
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
            
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        if encoder_hidden_states.shape[1] == 77 and len(self.target_indices) > 0:
            maps = attention_probs[:, :, self.target_indices].mean(dim=-1)
            import numpy as np
            seq_len = maps.shape[1]
            size = int(np.sqrt(seq_len))
            if size * size == seq_len and size >= 32:
                spatial_map = maps.view(-1, size, size).mean(dim=0).detach().cpu()
                self.attn_maps.append(spatial_map)
            
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            
        if attn.residual_connection:
            hidden_states = hidden_states + residual
            
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def delta_deno_heatmap(
    image: Image.Image,
    region: tuple[int, int, int, int],
    category: str,
    defect_type: str,
    auto: dict[str, Any],
    text_hint: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    from diffusers.models.attention_processor import AttnProcessor
    import scipy.ndimage as ndimage
    
    left, top, right, bottom = region
    cropped = image.crop((left, top, right, bottom))
    original_size = cropped.size
    cropped_512 = cropped.resize((512, 512), Image.Resampling.LANCZOS)
    
    model_id = str(auto.get("delta_deno_model", auto.get("sd15_base_model", "runwayml/stable-diffusion-v1-5")))
    cache_dir = auto.get("delta_deno_cache_dir", auto.get("sd15_cache_dir", auto.get("qwen_cache_dir")))
    local_files_only = bool(auto.get("delta_deno_local_files_only", auto.get("sd15_local_files_only", True)))
    device = str(auto.get("sd_device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_name = str(auto.get("delta_deno_dtype", "float16" if device == "cuda" else "float32"))
    dtype = torch.float16 if dtype_name in {"auto", "float16", "fp16"} and device == "cuda" else torch.float32
    cache_key = (model_id, str(cache_dir) if cache_dir else None, local_files_only, device, str(dtype))
    
    if cache_key not in _SD_PIPELINE_CACHE:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
            pipe.safety_checker = None
        _SD_PIPELINE_CACHE[cache_key] = pipe
    
    pipe = _SD_PIPELINE_CACHE[cache_key]
    
    anomaly_prompt = text_hint if text_hint and len(text_hint) > 3 else f"a {category} with {defect_type}"
    target_word = defect_type.split("_")[0] if "_" in defect_type else defect_type
    
    inputs = pipe.tokenizer(anomaly_prompt)
    target_ids = pipe.tokenizer(target_word, add_special_tokens=False).input_ids
    indices = [i for i, tid in enumerate(inputs.input_ids) if tid in target_ids]
    if not indices:
        raise ValueError(f"delta_deno rejected: target token {target_word!r} was not found in prompt {anomaly_prompt!r}")
    
    original_processors = dict(pipe.unet.attn_processors)
    attn_processors = {}
    stored_processors = []
    for name in pipe.unet.attn_processors.keys():
        if "attn2" in name:
            proc = StoreAttnProcessor(indices)
            attn_processors[name] = proc
            stored_processors.append(proc)
        else:
            attn_processors[name] = AttnProcessor()
            
    try:
        pipe.unet.set_attn_processor(attn_processors)
        with torch.no_grad():
            pipe(
                prompt=anomaly_prompt,
                image=cropped_512,
                strength=float(auto.get("delta_deno_strength", 0.3)),
                num_inference_steps=int(auto.get("delta_deno_num_inference_steps", 20)),
                guidance_scale=float(auto.get("delta_deno_guidance_scale", 7.5)),
            )
    finally:
        pipe.unet.set_attn_processor(original_processors)
        
    all_maps = []
    for p in stored_processors:
        for m in p.attn_maps:
            m = m.unsqueeze(0).unsqueeze(0).float()
            m = torch.nn.functional.interpolate(m, size=(512, 512), mode="bilinear").squeeze()
            all_maps.append(m)
            
    if all_maps:
        diff_np = torch.stack(all_maps).mean(dim=0).numpy()
    else:
        raise ValueError("delta_deno rejected: no cross-attention maps were captured")
        
    diff_np = ndimage.gaussian_filter(diff_np, sigma=1.5)
    
    diff_img = Image.fromarray(diff_np).resize(original_size, Image.Resampling.BILINEAR)
    diff_np = np.asarray(diff_img)
    diff_np = _normalize_float(diff_np)
    
    heatmap = np.zeros((image.height, image.width), dtype=np.float32)
    heatmap[top:bottom, left:right] = diff_np
    
    return heatmap, {
        "delta_deno_anomaly_prompt": anomaly_prompt,
        "delta_deno_target_word": target_word,
        "delta_deno_target_indices": indices,
        "delta_deno_model": model_id,
        "delta_deno_cache_dir": str(cache_dir) if cache_dir else None,
        "delta_deno_local_files_only": local_files_only,
        "delta_deno_num_inference_steps": int(auto.get("delta_deno_num_inference_steps", 20)),
    }


def _validate_delta_deno_heatmap(heatmap: np.ndarray, region: tuple[int, int, int, int], auto: dict[str, Any]) -> None:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("delta_deno rejected: empty heatmap crop")
    if not np.isfinite(crop).all():
        raise ValueError("delta_deno rejected: non-finite heatmap values")
    maximum = float(crop.max())
    minimum = float(crop.min())
    if maximum <= 0.0:
        raise ValueError("delta_deno rejected: all-zero heatmap")
    min_contrast = float(auto.get("delta_deno_min_heatmap_contrast", 0.05))
    if maximum - minimum < min_contrast:
        raise ValueError(
            f"delta_deno rejected: heatmap contrast {maximum - minimum:.4f} is below {min_contrast:.4f}"
        )


def anomaly_heatmap(
    image: Image.Image,
    normal_path: Path | None,
    region: tuple[int, int, int, int],
    text_hint: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = image.convert("L")
    if normal_path is not None:
        normal = Image.open(normal_path).convert("L").resize(image.size, Image.Resampling.BILINEAR)
        normal_source = "nearest_normal"
    else:
        normal = gray.filter(ImageFilter.GaussianBlur(radius=9.0))
        normal_source = "blurred_defect_image"
    arr = np.asarray(gray, dtype=np.float32)
    normal_arr = np.asarray(normal, dtype=np.float32)
    residual = np.abs(arr - normal_arr)
    blurred = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=5.0)), dtype=np.float32)
    dark = np.clip(blurred - arr, 0, None)
    bright = np.clip(arr - blurred, 0, None)
    polarity = _polarity_from_hint(text_hint)
    if polarity == "dark":
        local = dark
    elif polarity == "bright":
        local = bright
    else:
        local = np.maximum(dark, bright)
    heat = _normalize_float(residual) * 0.55 + _normalize_float(local) * 0.45
    mask = np.zeros_like(heat, dtype=np.float32)
    left, top, right, bottom = region
    mask[top:bottom, left:right] = heat[top:bottom, left:right]
    return mask, {"heatmap_normal_source": normal_source, "heatmap_polarity": polarity}


def normal_texture_anomaly_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    text_hint: str,
    *,
    max_normals: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = image.convert("L")
    defect_features = _texture_features(gray, text_hint)
    normal_feature_values: dict[str, list[np.ndarray]] = {key: [] for key in defect_features}
    used_normals: list[str] = []
    for path in normal_paths[: max(0, max_normals)]:
        normal = Image.open(path).convert("L").resize(image.size, Image.Resampling.BILINEAR)
        for key, values in _texture_features(normal, text_hint).items():
            normal_feature_values[key].append(values.reshape(-1))
        used_normals.append(str(path))

    heat_parts: list[np.ndarray] = []
    for key, values in defect_features.items():
        normal_values = normal_feature_values.get(key, [])
        if normal_values:
            reference = np.concatenate(normal_values)
            median = float(np.median(reference))
            mad = float(np.median(np.abs(reference - median))) + 1e-6
            z = np.clip((values - median) / (1.4826 * mad), 0, None)
            heat_parts.append(_normalize_float(z))
        else:
            heat_parts.append(_normalize_float(values))

    weights = np.asarray([0.40, 0.35, 0.25], dtype=np.float32)[: len(heat_parts)]
    weights = weights / max(1e-6, float(weights.sum()))
    heat = sum(part * float(weight) for part, weight in zip(heat_parts, weights, strict=False))
    mask = np.zeros_like(heat, dtype=np.float32)
    left, top, right, bottom = region
    mask[top:bottom, left:right] = heat[top:bottom, left:right]
    return mask, {
        "normal_anomaly_model": "robust_texture_stats",
        "normal_anomaly_normals_used": len(used_normals),
        "normal_anomaly_polarity": _polarity_from_hint(text_hint),
    }


def nearest_normal_patch_residual_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    text_hint: str,
    *,
    max_normals: int,
    search_radius: int,
    search_step: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import scipy.ndimage as ndimage

    if not normal_paths:
        raise ValueError("nearest_normal_residual requires at least one train/good normal image")
    left, top, right, bottom = region
    width, height = image.size
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    defect_full = np.asarray(image.convert("L"), dtype=np.float32)
    defect_crop = defect_full[top:bottom, left:right]
    if defect_crop.size == 0:
        raise ValueError("nearest_normal_residual received an empty bbox crop")

    defect_low = ndimage.gaussian_filter(defect_crop, sigma=3.0, mode="reflect")
    defect_high = defect_crop - defect_low
    best: tuple[float, Path, tuple[int, int, int, int], np.ndarray] | None = None
    step = max(1, int(search_step))
    radius = max(0, int(search_radius))
    offsets = list(range(-radius, radius + 1, step)) or [0]
    if 0 not in offsets:
        offsets.append(0)
    for normal_path in normal_paths[: max(1, max_normals)]:
        normal = np.asarray(
            Image.open(normal_path).convert("L").resize(image.size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        for dy in offsets:
            yy1 = min(max(0, top + dy), max(0, height - crop_height))
            yy2 = yy1 + crop_height
            for dx in offsets:
                xx1 = min(max(0, left + dx), max(0, width - crop_width))
                xx2 = xx1 + crop_width
                normal_crop = normal[yy1:yy2, xx1:xx2]
                if normal_crop.shape != defect_crop.shape:
                    continue
                normal_low = ndimage.gaussian_filter(normal_crop, sigma=3.0, mode="reflect")
                low_score = float(np.mean(np.abs(defect_low - normal_low)))
                texture_score = abs(float(defect_high.std()) - float((normal_crop - normal_low).std()))
                score = low_score + 0.20 * texture_score
                if best is None or score < best[0]:
                    best = (score, normal_path, (xx1, yy1, xx2, yy2), normal_crop.copy())
    if best is None:
        raise ValueError("nearest_normal_residual could not find a comparable normal patch")

    _, best_path, best_region, normal_crop = best
    normal_low = ndimage.gaussian_filter(normal_crop, sigma=3.0, mode="reflect")
    normal_high = normal_crop - normal_low
    high_residual = np.abs(defect_high - normal_high)
    local_residual = np.abs(defect_crop - normal_crop)
    hint = text_hint.lower()
    polarity = "absolute"
    if any(word in hint for word in ("dark", "black", "brown", "shadow", "burn")):
        signed = np.clip(normal_crop - defect_crop, 0, None)
        local_residual = np.maximum(local_residual * 0.35, signed)
        polarity = "dark"
    elif any(word in hint for word in ("bright", "white", "light", "silver", "pale")):
        signed = np.clip(defect_crop - normal_crop, 0, None)
        local_residual = np.maximum(local_residual * 0.35, signed)
        polarity = "bright"

    residual = _normalize_float(_normalize_float(high_residual) * 0.62 + _normalize_float(local_residual) * 0.38)
    residual = ndimage.gaussian_filter(residual, sigma=0.75, mode="reflect")
    heat = np.zeros((height, width), dtype=np.float32)
    heat[top:bottom, left:right] = _normalize_float(residual)
    contrast = float(np.percentile(residual, 95.0) - np.percentile(residual, 50.0))
    return heat, {
        "nearest_normal_residual_model": "patch_nearest_neighbor_residual",
        "nearest_normal_residual_normal_path": str(best_path),
        "nearest_normal_residual_normal_region_xyxy": best_region,
        "nearest_normal_residual_match_score": float(best[0]),
        "nearest_normal_residual_normals_seen": min(len(normal_paths), max_normals),
        "nearest_normal_residual_search_radius": int(search_radius),
        "nearest_normal_residual_search_step": int(search_step),
        "nearest_normal_residual_polarity": polarity,
        "nearest_normal_residual_contrast": contrast,
    }


def patchcore_guided_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    text_hint: str,
    *,
    patch_size: int,
    stride: int,
    max_normals: int,
    max_memory_patches: int,
    feature_cache: PatchFeatureCache | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not normal_paths:
        raise ValueError("patchcore_guided requires at least one train/good normal image")
    patch_size = max(5, int(patch_size))
    if patch_size % 2 == 0:
        patch_size += 1
    stride = max(1, int(stride))
    width, height = image.size

    memory_parts: list[np.ndarray] = []
    used_normals: list[str] = []
    for normal_path in normal_paths[: max(1, int(max_normals))]:
        features, _ = _cached_patchcore_patch_features_for_path(
            normal_path,
            image.size,
            patch_size=patch_size,
            stride=stride,
            text_hint=text_hint,
            feature_cache=feature_cache,
        )
        if features.size:
            memory_parts.append(features)
            used_normals.append(str(normal_path))
    if not memory_parts:
        raise ValueError("patchcore_guided could not build a normal patch memory")

    memory = np.concatenate(memory_parts, axis=0).astype(np.float32)
    if memory.shape[0] > max(1, int(max_memory_patches)):
        indices = np.linspace(0, memory.shape[0] - 1, max(1, int(max_memory_patches)), dtype=np.int64)
        memory = memory[indices]

    target_features, centers = _cached_patchcore_patch_features_for_image(
        image,
        ("target", id(image)),
        patch_size=patch_size,
        stride=stride,
        text_hint=text_hint,
        feature_cache=feature_cache,
    )
    if target_features.size == 0 or not centers:
        raise ValueError("patchcore_guided received an empty feature grid")

    mean = memory.mean(axis=0, keepdims=True)
    std = memory.std(axis=0, keepdims=True) + 1e-6
    memory_z = (memory - mean) / std
    target_z = (target_features.astype(np.float32) - mean) / std
    distances = _nearest_feature_distances(target_z, memory_z, chunk_size=512)

    left, top, right, bottom = region
    heat = np.zeros((height, width), dtype=np.float32)
    half = patch_size // 2
    for (x, y), distance in zip(centers, distances, strict=False):
        if x < left or x >= right or y < top or y >= bottom:
            continue
        x1 = max(left, x - half)
        y1 = max(top, y - half)
        x2 = min(right, x + half + 1)
        y2 = min(bottom, y + half + 1)
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(distance))

    heat = np.asarray(
        Image.fromarray(np.uint8(_normalize_float(heat) * 255), mode="L").filter(
            ImageFilter.GaussianBlur(radius=max(0.5, stride / 3.0))
        ),
        dtype=np.float32,
    ) / 255.0
    fenced = np.zeros_like(heat, dtype=np.float32)
    fenced[top:bottom, left:right] = heat[top:bottom, left:right]
    return fenced, {
        "patchcore_guided_model": "handcrafted_patch_memory",
        "patchcore_guided_normals_used": len(used_normals),
        "patchcore_guided_memory_patches": int(memory.shape[0]),
        "patchcore_guided_target_patches": int(target_features.shape[0]),
        "patchcore_guided_patch_size": int(patch_size),
        "patchcore_guided_stride": int(stride),
        **_patch_feature_cache_stats(feature_cache, prefix="patchcore_guided"),
    }


def musc_mutual_score_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    text_hint: str,
    *,
    patch_size: int,
    stride: int,
    max_normals: int,
    max_memory_patches: int,
    knn_k: int,
    feature_cache: PatchFeatureCache | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not normal_paths:
        raise ValueError("musc_mutual_score requires at least one train/good normal image")
    patch_size = max(5, int(patch_size))
    if patch_size % 2 == 0:
        patch_size += 1
    stride = max(1, int(stride))
    knn_k = max(1, int(knn_k))
    width, height = image.size

    memory_parts: list[np.ndarray] = []
    used_normals: list[str] = []
    for normal_path in normal_paths[: max(1, int(max_normals))]:
        features, _ = _cached_patchcore_patch_features_for_path(
            normal_path,
            image.size,
            patch_size=patch_size,
            stride=stride,
            text_hint=text_hint,
            feature_cache=feature_cache,
        )
        if features.size:
            memory_parts.append(features)
            used_normals.append(str(normal_path))
    if not memory_parts:
        raise ValueError("musc_mutual_score could not build a normal patch memory")

    memory = np.concatenate(memory_parts, axis=0).astype(np.float32)
    if memory.shape[0] > max(1, int(max_memory_patches)):
        indices = np.linspace(0, memory.shape[0] - 1, max(1, int(max_memory_patches)), dtype=np.int64)
        memory = memory[indices]

    target_features, centers = _cached_patchcore_patch_features_for_image(
        image,
        ("target", id(image)),
        patch_size=patch_size,
        stride=stride,
        text_hint=text_hint,
        feature_cache=feature_cache,
    )
    if target_features.size == 0 or not centers:
        raise ValueError("musc_mutual_score received an empty feature grid")

    mean = memory.mean(axis=0, keepdims=True)
    std = memory.std(axis=0, keepdims=True) + 1e-6
    memory_z = (memory - mean) / std
    target_z = (target_features.astype(np.float32) - mean) / std
    normal_knn = _knn_feature_distance_mean(target_z, memory_z, k=min(knn_k, memory_z.shape[0]), chunk_size=512)
    target_density = _target_feature_density(target_z, k=min(max(2, knn_k + 1), target_z.shape[0]), chunk_size=512)
    density_norm = _normalize_float(target_density)
    anomaly = _normalize_float(normal_knn) * (0.78 + 0.22 * (1.0 - density_norm))

    left, top, right, bottom = region
    heat = np.zeros((height, width), dtype=np.float32)
    half = patch_size // 2
    for (x, y), value in zip(centers, anomaly, strict=False):
        if x < left or x >= right or y < top or y >= bottom:
            continue
        x1 = max(left, x - half)
        y1 = max(top, y - half)
        x2 = min(right, x + half + 1)
        y2 = min(bottom, y + half + 1)
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(value))

    heat = np.asarray(
        Image.fromarray(np.uint8(_normalize_float(heat) * 255), mode="L").filter(
            ImageFilter.GaussianBlur(radius=max(0.5, stride / 3.0))
        ),
        dtype=np.float32,
    ) / 255.0
    fenced = np.zeros_like(heat, dtype=np.float32)
    fenced[top:bottom, left:right] = heat[top:bottom, left:right]
    return fenced, {
        "musc_model": "normal_memory_mutual_patch_scoring",
        "musc_normals_used": len(used_normals),
        "musc_memory_patches": int(memory.shape[0]),
        "musc_target_patches": int(target_features.shape[0]),
        "musc_patch_size": int(patch_size),
        "musc_stride": int(stride),
        "musc_knn_k": int(knn_k),
        "musc_mean_normal_knn": float(np.mean(normal_knn)),
        "musc_mean_target_density": float(np.mean(target_density)),
        **_patch_feature_cache_stats(feature_cache, prefix="musc"),
    }


def _patch_feature_cache_key(
    source: Any,
    image_size: tuple[int, int],
    *,
    patch_size: int,
    stride: int,
    text_hint: str,
) -> tuple[Any, ...]:
    return (
        "patchcore_patch_features",
        source,
        tuple(int(value) for value in image_size),
        int(patch_size),
        int(stride),
        _polarity_from_hint(text_hint),
    )


def _patch_feature_cache_stats_key() -> tuple[str]:
    return ("patch_feature_cache_stats",)


def _patch_feature_cache_stats(
    feature_cache: PatchFeatureCache | None,
    *,
    prefix: str,
) -> dict[str, int]:
    if feature_cache is None:
        return {
            f"{prefix}_feature_cache_enabled": 0,
            f"{prefix}_feature_cache_hits": 0,
            f"{prefix}_feature_cache_misses": 0,
        }
    stats = feature_cache.get(_patch_feature_cache_stats_key(), {})
    return {
        f"{prefix}_feature_cache_enabled": 1,
        f"{prefix}_feature_cache_hits": int(stats.get("hits", 0)),
        f"{prefix}_feature_cache_misses": int(stats.get("misses", 0)),
    }


def _increment_patch_feature_cache_stat(feature_cache: PatchFeatureCache, key: str) -> None:
    stats = feature_cache.setdefault(_patch_feature_cache_stats_key(), {"hits": 0, "misses": 0})
    stats[key] = int(stats.get(key, 0)) + 1


def _cached_patchcore_patch_features_for_image(
    image: Image.Image,
    source: Any,
    *,
    patch_size: int,
    stride: int,
    text_hint: str,
    feature_cache: PatchFeatureCache | None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if feature_cache is None:
        return _patchcore_patch_features(image, patch_size=patch_size, stride=stride, text_hint=text_hint)
    key = _patch_feature_cache_key(
        source,
        image.size,
        patch_size=patch_size,
        stride=stride,
        text_hint=text_hint,
    )
    cached = feature_cache.get(key)
    if cached is not None:
        _increment_patch_feature_cache_stat(feature_cache, "hits")
        features, centers = cached
        return features, list(centers)
    _increment_patch_feature_cache_stat(feature_cache, "misses")
    features, centers = _patchcore_patch_features(image, patch_size=patch_size, stride=stride, text_hint=text_hint)
    feature_cache[key] = (features, list(centers))
    return features, list(centers)


def _cached_patchcore_patch_features_for_path(
    path: Path,
    image_size: tuple[int, int],
    *,
    patch_size: int,
    stride: int,
    text_hint: str,
    feature_cache: PatchFeatureCache | None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if feature_cache is None:
        normal = Image.open(path).convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
        return _patchcore_patch_features(normal, patch_size=patch_size, stride=stride, text_hint=text_hint)
    source = ("normal_path", str(Path(path).resolve()))
    key = _patch_feature_cache_key(
        source,
        image_size,
        patch_size=patch_size,
        stride=stride,
        text_hint=text_hint,
    )
    cached = feature_cache.get(key)
    if cached is not None:
        _increment_patch_feature_cache_stat(feature_cache, "hits")
        features, centers = cached
        return features, list(centers)
    _increment_patch_feature_cache_stat(feature_cache, "misses")
    normal = Image.open(path).convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    features, centers = _patchcore_patch_features(normal, patch_size=patch_size, stride=stride, text_hint=text_hint)
    feature_cache[key] = (features, list(centers))
    return features, list(centers)


def _patchcore_patch_features(
    image: Image.Image,
    *,
    patch_size: int,
    stride: int,
    text_hint: str,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    blur_small = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32) / 255.0
    blur_large = np.asarray(gray_image.filter(ImageFilter.GaussianBlur(radius=8.0)), dtype=np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    edge = np.hypot(gx, gy)
    detail = np.abs(gray - blur_small)
    dark = np.clip(blur_large - gray, 0, None)
    bright = np.clip(gray - blur_large, 0, None)
    polarity = _polarity_from_hint(text_hint)
    contrast = dark if polarity == "dark" else bright if polarity == "bright" else np.maximum(dark, bright)

    height, width = gray.shape
    half = patch_size // 2
    centers = [
        (x, y)
        for y in range(half, max(half + 1, height - half), stride)
        for x in range(half, max(half + 1, width - half), stride)
    ]
    features: list[list[float]] = []
    for x, y in centers:
        x1, x2 = max(0, x - half), min(width, x + half + 1)
        y1, y2 = max(0, y - half), min(height, y + half + 1)
        rgb_patch = rgb[y1:y2, x1:x2]
        gray_patch = gray[y1:y2, x1:x2]
        edge_patch = edge[y1:y2, x1:x2]
        detail_patch = detail[y1:y2, x1:x2]
        contrast_patch = contrast[y1:y2, x1:x2]
        features.append(
            [
                *rgb_patch.reshape(-1, 3).mean(axis=0).tolist(),
                *rgb_patch.reshape(-1, 3).std(axis=0).tolist(),
                float(gray_patch.mean()),
                float(gray_patch.std()),
                float(edge_patch.mean()),
                float(edge_patch.std()),
                float(detail_patch.mean()),
                float(detail_patch.std()),
                float(contrast_patch.mean()),
                float(contrast_patch.std()),
            ]
        )
    if not features:
        return np.zeros((0, 14), dtype=np.float32), []
    return np.asarray(features, dtype=np.float32), centers


def _nearest_feature_distances(target: np.ndarray, memory: np.ndarray, *, chunk_size: int) -> np.ndarray:
    output = np.zeros(target.shape[0], dtype=np.float32)
    memory_sq = np.sum(memory * memory, axis=1, keepdims=True).T
    for start in range(0, target.shape[0], max(1, chunk_size)):
        chunk = target[start : start + chunk_size]
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances_sq = np.maximum(chunk_sq + memory_sq - 2.0 * (chunk @ memory.T), 0.0)
        output[start : start + chunk.shape[0]] = np.sqrt(np.min(distances_sq, axis=1))
    return output


def _knn_feature_distance_mean(target: np.ndarray, memory: np.ndarray, *, k: int, chunk_size: int) -> np.ndarray:
    output = np.zeros(target.shape[0], dtype=np.float32)
    memory_sq = np.sum(memory * memory, axis=1, keepdims=True).T
    k = max(1, min(int(k), memory.shape[0]))
    for start in range(0, target.shape[0], max(1, chunk_size)):
        chunk = target[start : start + chunk_size]
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances_sq = np.maximum(chunk_sq + memory_sq - 2.0 * (chunk @ memory.T), 0.0)
        nearest = np.partition(distances_sq, kth=k - 1, axis=1)[:, :k]
        output[start : start + chunk.shape[0]] = np.sqrt(np.maximum(nearest, 0.0)).mean(axis=1)
    return output


def _target_feature_density(target: np.ndarray, *, k: int, chunk_size: int) -> np.ndarray:
    if target.shape[0] <= 1:
        return np.ones(target.shape[0], dtype=np.float32)
    output = np.zeros(target.shape[0], dtype=np.float32)
    target_sq_all = np.sum(target * target, axis=1, keepdims=True).T
    k = max(2, min(int(k), target.shape[0]))
    for start in range(0, target.shape[0], max(1, chunk_size)):
        chunk = target[start : start + chunk_size]
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances_sq = np.maximum(chunk_sq + target_sq_all - 2.0 * (chunk @ target.T), 0.0)
        for row_index, global_index in enumerate(range(start, start + chunk.shape[0])):
            distances_sq[row_index, global_index] = np.inf
        nearest = np.partition(distances_sq, kth=k - 1, axis=1)[:, : k - 1]
        output[start : start + chunk.shape[0]] = np.sqrt(np.maximum(nearest, 0.0)).mean(axis=1)
    return output


def dinov2_memory_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    *,
    model_id: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
    max_normals: int,
    memory_stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not normal_paths:
        raise ValueError("DINOv2 memory anomaly requires at least one train/good normal image")
    import torch

    processor, model = _load_dinov2(model_id, cache_dir, local_files_only, device)
    model_device = next(model.parameters()).device
    defect_tokens, grid = _dinov2_patch_tokens(processor, model, image, model_device)
    memory_tokens: list[torch.Tensor] = []
    used = 0
    for path in normal_paths[: max(1, max_normals)]:
        normal = Image.open(path).convert("RGB").resize(image.size, Image.Resampling.BILINEAR)
        tokens, _ = _dinov2_patch_tokens(processor, model, normal, model_device)
        if memory_stride > 1:
            tokens = tokens[::memory_stride]
        memory_tokens.append(tokens)
        used += 1
    memory = torch.cat(memory_tokens, dim=0)
    distances = _nearest_neighbor_distances(defect_tokens, memory)
    patch_map = distances.reshape(grid)
    heat_small = _normalize_float(patch_map.detach().cpu().numpy())
    heat_image = Image.fromarray(np.uint8(heat_small * 255), mode="L").resize(image.size, Image.Resampling.BILINEAR)
    heat = np.asarray(heat_image, dtype=np.float32) / 255.0
    masked = np.zeros_like(heat, dtype=np.float32)
    left, top, right, bottom = region
    masked[top:bottom, left:right] = heat[top:bottom, left:right]
    return masked, {
        "dinov2_provider": "dinov2_memory_bank",
        "dinov2_available": True,
        "dinov2_model": model_id,
        "dinov2_normals_used": used,
        "dinov2_patch_grid": [int(grid[0]), int(grid[1])],
        "dinov2_memory_patches": int(memory.shape[0]),
        "dinov2_feature_dim": int(memory.shape[1]),
    }


def _load_dinov2(
    model_id: str,
    cache_dir: str | None,
    local_files_only: bool,
    device: str,
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    key = (model_id, cache_dir, local_files_only, resolved_device)
    if key in _DINO_MODEL_CACHE:
        return _DINO_MODEL_CACHE[key]
    processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=local_files_only)
    model = model.to(resolved_device)
    model.eval()
    _DINO_MODEL_CACHE[key] = (processor, model)
    return processor, model


def _dinov2_patch_tokens(processor: Any, model: Any, image: Image.Image, device: Any) -> tuple[Any, tuple[int, int]]:
    import torch

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    tokens = output.last_hidden_state[:, 1:, :].squeeze(0).float()
    tokens = torch.nn.functional.normalize(tokens, dim=-1)
    count = int(tokens.shape[0])
    side = int(round(count**0.5))
    if side * side == count:
        grid = (side, side)
    else:
        grid = (count, 1)
    return tokens, grid


def _nearest_neighbor_distances(tokens: Any, memory: Any) -> Any:
    import torch

    best: torch.Tensor | None = None
    for chunk in memory.split(4096, dim=0):
        # Features are L2-normalized, so cosine distance is stable and cheap.
        distance = 1.0 - tokens @ chunk.T
        chunk_best = distance.min(dim=1).values
        best = chunk_best if best is None else torch.minimum(best, chunk_best)
    if best is None:
        raise ValueError("Empty DINOv2 memory bank")
    return best


def support_anomaly_heatmap(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    text_hint: str,
    *,
    auto: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    backend = str(auto.get("support_heatmap_backend", "normal_texture"))
    if backend not in {"normal_texture", "dinov2"}:
        raise ValueError(f"Unsupported support_heatmap_backend: {backend}")
    if backend == "dinov2":
        try:
            heatmap, params = dinov2_memory_heatmap(
                image,
                normal_paths,
                region,
                model_id=str(auto.get("dinov2_model", "facebook/dinov2-small")),
                cache_dir=auto.get("dinov2_cache_dir", auto.get("qwen_cache_dir")),
                local_files_only=bool(auto.get("dinov2_local_files_only", False)),
                device=str(auto.get("dinov2_device", auto.get("qwen_device", "auto"))),
                max_normals=int(auto.get("dinov2_max_normals", 16)),
                memory_stride=int(auto.get("dinov2_memory_stride", 1)),
            )
            return heatmap, {"support_heatmap_backend": "dinov2", **params}
        except Exception as exc:
            if not bool(auto.get("support_heatmap_allow_fallback", True)):
                raise
            heatmap, params = normal_texture_anomaly_heatmap(
                image,
                normal_paths,
                region,
                text_hint,
                max_normals=int(auto.get("normal_anomaly_max_normals", 32)),
            )
            return heatmap, {
                "support_heatmap_backend": "normal_texture_fallback",
                "support_heatmap_requested_backend": "dinov2",
                "support_heatmap_error": str(exc),
                **params,
            }
    heatmap, params = normal_texture_anomaly_heatmap(
        image,
        normal_paths,
        region,
        text_hint,
        max_normals=int(auto.get("normal_anomaly_max_normals", 32)),
    )
    return heatmap, {"support_heatmap_backend": "normal_texture", **params}


def _texture_features(gray: Image.Image, text_hint: str) -> dict[str, np.ndarray]:
    arr = np.asarray(gray, dtype=np.float32)
    blur_small = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32)
    blur_large = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=8.0)), dtype=np.float32)
    dark = np.clip(blur_large - arr, 0, None)
    bright = np.clip(arr - blur_large, 0, None)
    polarity = _polarity_from_hint(text_hint)
    if polarity == "dark":
        contrast = dark
    elif polarity == "bright":
        contrast = bright
    else:
        contrast = np.maximum(dark, bright)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:] = arr[:, 1:] - arr[:, :-1]
    gy[1:, :] = arr[1:, :] - arr[:-1, :]
    edge = np.hypot(gx, gy)
    detail = np.abs(arr - blur_small)
    return {"contrast": contrast, "edge": edge, "detail": detail}


def _polarity_from_hint(text_hint: str) -> str:
    hint = text_hint.lower()
    dark = any(word in hint for word in ("dark", "black", "brown", "shadow", "burn"))
    bright = any(word in hint for word in ("bright", "white", "light", "silver", "pale"))
    if dark and not bright:
        return "dark"
    if bright and not dark:
        return "bright"
    return "absolute"


def heatmap_prompt_points(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    *,
    positive_count: int,
    negative_count: int,
) -> tuple[list[list[int]], list[int]]:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0:
        return [[(left + right) // 2, (top + bottom) // 2]], [1]
    threshold = np.percentile(crop, 97.0)
    active = crop >= threshold
    components = _connected_components(active)
    scored: list[tuple[float, int, int]] = []
    for component in components:
        if len(component) < 2:
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        score = float(crop[ys, xs].mean())
        scored.append((score, left + int(np.median(xs)), top + int(np.median(ys))))
    scored.sort(reverse=True)
    points = [[x, y] for _, x, y in scored[: max(1, positive_count)]]
    labels = [1] * len(points)
    negative_candidates = [
        [left + max(1, (right - left) // 10), top + max(1, (bottom - top) // 10)],
        [right - max(1, (right - left) // 10), top + max(1, (bottom - top) // 10)],
        [left + max(1, (right - left) // 10), bottom - max(1, (bottom - top) // 10)],
        [right - max(1, (right - left) // 10), bottom - max(1, (bottom - top) // 10)],
    ]
    points.extend(negative_candidates[: max(0, negative_count)])
    labels.extend([0] * min(len(negative_candidates), max(0, negative_count)))
    if not points:
        points = [[(left + right) // 2, (top + bottom) // 2]]
        labels = [1]
    return points, labels


def heatmap_fallback_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    percentile: float,
) -> Image.Image:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    threshold = float(np.percentile(crop, max(50.0, min(99.5, percentile)))) if crop.size else 1.0
    active = crop >= threshold if crop.size else np.zeros((1, 1), dtype=bool)
    active = _filter_components(active, "scratch", min_area=4)
    if active.sum() == 0 and crop.size:
        active = crop >= float(np.percentile(crop, 92.0))
    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(active.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask


def soft_patch_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    defect_type: str,
    *,
    text_hint: str,
    percentile: float,
    close_radius: int,
    min_component_area: int,
    max_area_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0:
        return Image.new("L", image_size, 0), {"kind": "soft_patch", "reason": "empty_crop"}

    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(crop, threshold_percentile))
    active = crop >= threshold
    active = _filter_components(active, defect_type, max(2, min_component_area // 3), require_elongated=False)

    patch_like = _description_implies_patch(text_hint)
    if patch_like or _description_implies_multiple(text_hint):
        active = _morph_close(active, radius=max(1, close_radius))
        active = _filter_components(active, defect_type, max(2, min_component_area // 2), require_elongated=False)
    else:
        active = _filter_components(active, defect_type, max(2, min_component_area), require_elongated=False)

    if int(active.sum()) < max(4, min_component_area // 2):
        threshold_percentile = min(threshold_percentile, 90.0)
        threshold = float(np.percentile(crop, threshold_percentile))
        active = crop >= threshold
        active = _morph_close(active, radius=max(1, close_radius))
        active = _filter_components(active, defect_type, max(2, min_component_area // 2), require_elongated=False)

    max_pixels = int(max(1.0, crop.size * max(0.01, min(0.45, max_area_fraction))))
    while int(active.sum()) > max_pixels and threshold_percentile < 99.0:
        threshold_percentile += 1.0
        threshold = float(np.percentile(crop, threshold_percentile))
        active = crop >= threshold
        active = _morph_close(active, radius=max(1, close_radius))
        active = _filter_components(active, defect_type, max(2, min_component_area // 2), require_elongated=False)

    if active.sum() == 0:
        active = crop >= float(np.percentile(crop, 97.0))

    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(active.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "kind": "soft_patch",
        "soft_patch_percentile": float(threshold_percentile),
        "soft_patch_threshold": float(threshold),
        "soft_patch_close_radius": int(close_radius),
        "soft_patch_active_pixels": int(active.sum()),
        "soft_patch_patch_like_hint": bool(patch_like),
    }


def patchcore_guided_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    defect_type: str,
    *,
    text_hint: str,
    percentile: float,
    close_radius: int,
    dilate_radius: int,
    min_component_area: int,
    max_components: int,
    max_box_fraction: float,
    min_contrast: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    heat_crop = heatmap[top:bottom, left:right]
    if heat_crop.size == 0:
        return Image.new("L", image_size, 0), {"patchcore_guided_reason": "empty_crop"}
    contrast = float(np.percentile(heat_crop, 95.0) - np.percentile(heat_crop, 50.0))
    if contrast < min_contrast:
        return Image.new("L", image_size, 0), {
            "patchcore_guided_reason": "low_contrast",
            "patchcore_guided_contrast": contrast,
            "patchcore_guided_min_contrast": float(min_contrast),
            "patchcore_guided_active_pixels": 0,
        }

    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(heat_crop, threshold_percentile))
    active = heat_crop >= threshold
    patch_like = _description_implies_patch(text_hint) or _description_implies_multiple(text_hint)
    if patch_like:
        active = _morph_close(active, radius=max(1, close_radius))
        active = _dilate_bool(active, radius=max(0, dilate_radius))
        require_elongated = False
    else:
        active = _morph_close(active, radius=max(0, close_radius // 2))
        require_elongated = defect_type == "scratch"

    filtered = _filter_components(active, defect_type, min_component_area, require_elongated=require_elongated)
    if not filtered.any():
        threshold_percentile = max(50.0, threshold_percentile - 8.0)
        threshold = float(np.percentile(heat_crop, threshold_percentile))
        active = heat_crop >= threshold
        if patch_like:
            active = _morph_close(active, radius=max(1, close_radius))
        filtered = _filter_components(active, defect_type, max(4, min_component_area // 2), require_elongated=False)

    components = _connected_components(filtered)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    for component in components:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        area = len(component)
        fill = area / max(1, width * height)
        aspect = max(width / max(1, height), height / max(1, width))
        if fill > 0.88 and area > min_component_area * 5:
            continue
        score = float(heat_crop[ys, xs].mean()) + min(0.18, area / max(1, heat_crop.size) * 3.0)
        if defect_type == "scratch":
            score += min(0.10, aspect / 40.0)
        scored.append((score, component))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = np.zeros_like(filtered, dtype=bool)
    for _, component in scored[: max(1, max_components)]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
    if not selected.any():
        selected = filtered
    selected = _limit_mask_area(selected, heat_crop, max_area_fraction=max_box_fraction)
    selected = _cap_mask_by_heat(selected, heat_crop, max_area_fraction=max_box_fraction)

    output = Image.new("L", image_size, 0)
    output.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return output, {
        "patchcore_guided_percentile": float(threshold_percentile),
        "patchcore_guided_threshold": float(threshold),
        "patchcore_guided_close_radius": int(close_radius),
        "patchcore_guided_dilate_radius": int(dilate_radius),
        "patchcore_guided_max_components": int(max_components),
        "patchcore_guided_components_seen": len(components),
        "patchcore_guided_components_kept": min(len(scored), max_components),
        "patchcore_guided_max_box_fraction": float(max_box_fraction),
        "patchcore_guided_active_pixels": int(selected.sum()),
        "patchcore_guided_contrast": contrast,
        "patchcore_guided_patch_like_hint": bool(patch_like),
    }


def nearest_normal_patch_residual_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    defect_type: str,
    *,
    text_hint: str,
    percentile: float,
    close_radius: int,
    min_component_area: int,
    max_components: int,
    max_box_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    heat_crop = heatmap[top:bottom, left:right]
    if heat_crop.size == 0:
        return Image.new("L", image_size, 0), {"nearest_normal_residual_reason": "empty_crop"}
    contrast = float(np.percentile(heat_crop, 95.0) - np.percentile(heat_crop, 50.0))
    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(heat_crop, threshold_percentile))
    active = heat_crop >= threshold
    patch_like = _description_implies_patch(text_hint) or _description_implies_multiple(text_hint)
    if patch_like:
        active = _morph_close(active, radius=max(1, close_radius))
        require_elongated = False
    else:
        active = _morph_close(active, radius=max(0, close_radius // 2))
        require_elongated = defect_type == "scratch"
    filtered = _filter_components(active, defect_type, min_component_area, require_elongated=require_elongated)
    if int(filtered.sum()) == 0:
        threshold_percentile = max(50.0, threshold_percentile - 8.0)
        threshold = float(np.percentile(heat_crop, threshold_percentile))
        active = heat_crop >= threshold
        if patch_like:
            active = _morph_close(active, radius=max(1, close_radius))
        filtered = _filter_components(active, defect_type, max(4, min_component_area // 2), require_elongated=False)
    components = _connected_components(filtered)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    for component in components:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        area = len(component)
        aspect = max(width / max(1, height), height / max(1, width))
        fill = area / max(1, width * height)
        if fill > 0.86 and area > min_component_area * 5:
            continue
        score = float(heat_crop[ys, xs].mean()) + min(0.20, area / max(1, heat_crop.size) * 3.0)
        if defect_type == "scratch":
            score += min(0.12, aspect / 35.0)
        scored.append((score, component))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(filtered, dtype=bool)
    for _, component in scored[: max(1, max_components)]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
    if not selected.any():
        selected = filtered
    selected = _limit_mask_area(selected, heat_crop, max_area_fraction=max_box_fraction)
    max_pixels = int(max(1, selected.size * max(0.01, min(0.60, max_box_fraction))))
    if int(selected.sum()) > max_pixels:
        values = heat_crop[selected]
        cutoff = float(np.partition(values, max(0, values.size - max_pixels))[max(0, values.size - max_pixels)])
        trimmed = selected & (heat_crop >= cutoff)
        if trimmed.any():
            selected = trimmed
    output = Image.new("L", image_size, 0)
    output.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return output, {
        "nearest_normal_residual_percentile": float(threshold_percentile),
        "nearest_normal_residual_threshold": float(threshold),
        "nearest_normal_residual_close_radius": int(close_radius),
        "nearest_normal_residual_max_components": int(max_components),
        "nearest_normal_residual_components_seen": len(components),
        "nearest_normal_residual_components_kept": min(len(scored), max_components),
        "nearest_normal_residual_max_box_fraction": float(max_box_fraction),
        "nearest_normal_residual_active_pixels": int(selected.sum()),
        "nearest_normal_residual_contrast": contrast,
        "nearest_normal_residual_patch_like_hint": bool(patch_like),
    }


def normal_anomaly_mask(
    heatmap: np.ndarray,
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    defect_type: str,
    *,
    text_hint: str,
    percentile: float,
    min_component_area: int,
    max_components: int,
    dilate_radius: int,
    envelope: bool,
    envelope_mode: str,
    envelope_radius: int,
    envelope_percentile: float,
    max_box_fraction: float,
    suppress_parallel_texture: bool,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0:
        return Image.new("L", image.size, 0), {"kind": "normal_texture_anomaly", "reason": "empty_crop"}
    threshold_values = crop[crop > 0]
    if threshold_values.size == 0:
        threshold_values = crop.reshape(-1)
    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(threshold_values, threshold_percentile))
    active = crop >= threshold
    active = _morph_close(active, radius=1)
    dominant_angle = _dominant_texture_angle_degrees(image, normal_paths, region)
    filtered, component_params = _filter_anomaly_components(
        active,
        crop,
        defect_type,
        text_hint,
        min_component_area=max(2, min_component_area),
        max_components=max(1, max_components),
        dominant_texture_angle=dominant_angle,
        suppress_parallel_texture=suppress_parallel_texture,
    )
    if filtered.sum() == 0:
        threshold_percentile = min(92.0, threshold_percentile)
        threshold = float(np.percentile(threshold_values, threshold_percentile))
        active = _morph_close(crop >= threshold, radius=1)
        filtered, component_params = _filter_anomaly_components(
            active,
            crop,
            defect_type,
            text_hint,
            min_component_area=max(2, min_component_area // 2),
            max_components=max(1, max_components),
            dominant_texture_angle=dominant_angle,
            suppress_parallel_texture=False,
        )
    if dilate_radius > 0 and filtered.any():
        filtered = _dilate_bool(filtered, radius=dilate_radius)
    if envelope and filtered.any():
        mode = envelope_mode.lower()
        if mode not in {"adaptive", "patch", "scratch_band"}:
            raise ValueError(f"Unsupported normal anomaly envelope mode: {envelope_mode}")
        use_scratch_band = mode == "scratch_band" or (
            mode == "adaptive" and defect_type in {"scratch", "crack"} and _mask_crop_aspect(filtered) >= 1.8
        )
        if use_scratch_band:
            filtered, envelope_kind = _scratch_band_envelope(
                filtered,
                crop,
                text_hint=text_hint,
                radius=max(1, envelope_radius),
                percentile=envelope_percentile,
                max_area_fraction=max_box_fraction,
            )
        else:
            filtered = _damage_envelope(
                filtered,
                crop,
                radius=max(1, envelope_radius),
                percentile=envelope_percentile,
                max_area_fraction=max_box_fraction,
            )
            envelope_kind = "patch"
    mask = Image.new("L", image.size, 0)
    mask.paste(Image.fromarray(filtered.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "kind": "normal_texture_anomaly",
        "normal_anomaly_percentile": float(threshold_percentile),
        "normal_anomaly_threshold": float(threshold),
        "normal_anomaly_dominant_texture_angle_degrees": dominant_angle,
        "normal_anomaly_suppress_parallel_texture": bool(suppress_parallel_texture),
        "normal_anomaly_dilate_radius": int(dilate_radius),
        "normal_anomaly_envelope": bool(envelope),
        "normal_anomaly_envelope_mode": envelope_mode,
        "normal_anomaly_envelope_kind": envelope_kind if envelope and filtered.any() else "none",
        "normal_anomaly_envelope_radius": int(envelope_radius),
        "normal_anomaly_envelope_percentile": float(envelope_percentile),
        "normal_anomaly_max_box_fraction": float(max_box_fraction),
        "normal_anomaly_active_pixels": int(filtered.sum()),
        **component_params,
    }


def fft_texture_suppression_heatmap(
    image: Image.Image,
    region: tuple[int, int, int, int],
    *,
    peak_count: int,
    notch_radius: int,
    low_frequency_radius: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    left, top, right, bottom = region
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    crop = gray[top:bottom, left:right]
    full = np.zeros_like(gray, dtype=np.float32)
    if crop.size == 0 or min(crop.shape) < 8:
        return full, {
            "fft_texture_reason": "empty_or_tiny_crop",
            "fft_texture_peak_count": 0,
            "fft_texture_contrast": 0.0,
        }

    centered = crop - float(crop.mean())
    window_y = np.hanning(crop.shape[0]).astype(np.float32)
    window_x = np.hanning(crop.shape[1]).astype(np.float32)
    window = np.outer(window_y, window_x)
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    magnitude = np.log1p(np.abs(spectrum))
    cy, cx = crop.shape[0] // 2, crop.shape[1] // 2
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    eligible = radius >= max(1, low_frequency_radius)
    peak_scores = magnitude.copy()
    peak_scores[~eligible] = -np.inf
    peaks: list[tuple[int, int, float]] = []
    filtered_spectrum = spectrum.copy()
    for _ in range(max(0, peak_count)):
        if not np.isfinite(peak_scores).any():
            break
        y, x = np.unravel_index(int(np.argmax(peak_scores)), peak_scores.shape)
        value = float(peak_scores[y, x])
        if not np.isfinite(value) or value <= 0:
            break
        peaks.append((int(y), int(x), value))
        for py, px in ((y, x), (2 * cy - y, 2 * cx - x)):
            y1 = max(0, py - notch_radius)
            y2 = min(crop.shape[0], py + notch_radius + 1)
            x1 = max(0, px - notch_radius)
            x2 = min(crop.shape[1], px + notch_radius + 1)
            filtered_spectrum[y1:y2, x1:x2] = 0
            peak_scores[y1:y2, x1:x2] = -np.inf

    suppressed = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)))
    local_residual = np.abs(suppressed)
    local_contrast = float(np.percentile(local_residual, 98) - np.percentile(local_residual, 50))
    local_heat = _normalize_float(local_residual)
    if local_heat.size:
        # Favor actual disrupted patches over the padded fence edge.
        margin = max(1, min(crop.shape) // 40)
        local_heat[:margin, :] *= 0.5
        local_heat[-margin:, :] *= 0.5
        local_heat[:, :margin] *= 0.5
        local_heat[:, -margin:] *= 0.5
    full[top:bottom, left:right] = local_heat
    return full, {
        "fft_texture_model": "dominant_frequency_notch_residual",
        "fft_texture_peak_count": len(peaks),
        "fft_texture_notch_radius": int(notch_radius),
        "fft_texture_low_frequency_radius": int(low_frequency_radius),
        "fft_texture_contrast": local_contrast,
        "fft_texture_peaks_yx": [[y, x] for y, x, _ in peaks[:8]],
    }


def fft_texture_suppression_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    defect_type: str,
    *,
    text_hint: str,
    percentile: float,
    close_radius: int,
    min_component_area: int,
    max_area_fraction: float,
    min_contrast: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0:
        return Image.new("L", image_size, 0), {"kind": "fft_texture_suppression", "fft_texture_reason": "empty_crop"}
    contrast = float(np.percentile(crop, 98) - np.percentile(crop, 50))
    if not np.isfinite(crop).all() or contrast < min_contrast:
        return Image.new("L", image_size, 0), {
            "kind": "fft_texture_suppression",
            "fft_texture_reason": "low_contrast_or_non_finite",
            "fft_texture_mask_contrast": contrast,
            "fft_texture_min_contrast": float(min_contrast),
        }
    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(crop, threshold_percentile))
    active = crop >= threshold
    active = _morph_close(active, radius=max(1, close_radius))
    active = _filter_components(active, defect_type, max(2, min_component_area // 2), require_elongated=False)
    if _description_implies_patch(text_hint) or _description_implies_multiple(text_hint):
        active = _damage_envelope(
            active,
            crop,
            radius=max(2, close_radius + 1),
            percentile=max(70.0, threshold_percentile - 6.0),
            max_area_fraction=max_area_fraction,
        )
    else:
        active = _limit_mask_area(active, crop, max_area_fraction=max_area_fraction)
    if int(active.sum()) < max(4, min_component_area // 2):
        threshold_percentile = min(92.0, threshold_percentile)
        threshold = float(np.percentile(crop, threshold_percentile))
        active = _morph_close(crop >= threshold, radius=max(1, close_radius))
        active = _filter_components(active, defect_type, max(2, min_component_area // 3), require_elongated=False)
        active = _limit_mask_area(active, crop, max_area_fraction=max_area_fraction)
    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(active.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "kind": "fft_texture_suppression",
        "fft_texture_percentile": float(threshold_percentile),
        "fft_texture_threshold": float(threshold),
        "fft_texture_close_radius": int(close_radius),
        "fft_texture_max_box_fraction": float(max_area_fraction),
        "fft_texture_mask_contrast": contrast,
        "fft_texture_active_pixels": int(active.sum()),
        "fft_texture_patch_like_hint": bool(_description_implies_patch(text_hint) or _description_implies_multiple(text_hint)),
    }


def structure_tensor_ridge_heatmap(
    image: Image.Image,
    region: tuple[int, int, int, int],
    *,
    sigma: float,
    residual_blur: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import scipy.ndimage as ndimage

    left, top, right, bottom = region
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    crop = gray[top:bottom, left:right]
    full = np.zeros_like(gray, dtype=np.float32)
    if crop.size == 0 or min(crop.shape) < 5:
        return full, {
            "structure_tensor_reason": "empty_or_tiny_crop",
            "structure_tensor_contrast": 0.0,
            "structure_tensor_sigma": float(sigma),
        }

    smoothed = ndimage.gaussian_filter(crop, sigma=max(0.1, sigma))
    gy, gx = np.gradient(smoothed)
    jxx = ndimage.gaussian_filter(gx * gx, sigma=max(0.1, sigma))
    jyy = ndimage.gaussian_filter(gy * gy, sigma=max(0.1, sigma))
    jxy = ndimage.gaussian_filter(gx * gy, sigma=max(0.1, sigma))
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1e-6)
    gradient = _normalize_float(np.sqrt(gx * gx + gy * gy))
    local_bg = ndimage.gaussian_filter(crop, sigma=max(1.0, residual_blur))
    residual = _normalize_float(np.abs(crop - local_bg))
    ridge = _normalize_float(coherence * (0.58 * gradient + 0.42 * residual))

    margin = max(1, min(crop.shape) // 48)
    ridge[:margin, :] *= 0.55
    ridge[-margin:, :] *= 0.55
    ridge[:, :margin] *= 0.55
    ridge[:, -margin:] *= 0.55
    full[top:bottom, left:right] = ridge
    return full, {
        "structure_tensor_model": "coherence_gradient_residual",
        "structure_tensor_sigma": float(sigma),
        "structure_tensor_residual_blur": float(residual_blur),
        "structure_tensor_contrast": float(np.percentile(ridge, 98) - np.percentile(ridge, 50)),
        "structure_tensor_mean_coherence": float(np.mean(coherence)),
    }


def structure_tensor_ridge_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    defect_type: str,
    *,
    percentile: float,
    close_radius: int,
    dilate_radius: int,
    min_component_area: int,
    max_components: int,
    max_area_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0 or not np.isfinite(crop).all():
        return Image.new("L", image_size, 0), {
            "kind": "structure_tensor_ridge",
            "structure_tensor_reason": "empty_or_non_finite_crop",
        }
    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(crop, threshold_percentile))
    active = crop >= threshold
    active = _morph_close(active, radius=max(0, close_radius))
    if dilate_radius > 0:
        active = _dilate_bool(active, radius=dilate_radius)
    active = _filter_components(active, defect_type, min_component_area, require_elongated=defect_type in {"scratch", "crack"})
    active = _keep_largest_components(active, max_components=max_components)
    active = _limit_mask_area(active, crop, max_area_fraction=max_area_fraction)
    if int(active.sum()) < max(4, min_component_area // 2):
        threshold_percentile = min(88.0, threshold_percentile)
        threshold = float(np.percentile(crop, threshold_percentile))
        active = crop >= threshold
        active = _morph_close(active, radius=max(0, close_radius))
        active = _filter_components(active, defect_type, max(2, min_component_area // 2), require_elongated=False)
        active = _keep_largest_components(active, max_components=max_components)
        active = _limit_mask_area(active, crop, max_area_fraction=max_area_fraction)
    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(active.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "kind": "structure_tensor_ridge",
        "structure_tensor_percentile": float(threshold_percentile),
        "structure_tensor_threshold": threshold,
        "structure_tensor_close_radius": int(close_radius),
        "structure_tensor_dilate_radius": int(dilate_radius),
        "structure_tensor_max_components": int(max_components),
        "structure_tensor_max_box_fraction": float(max_area_fraction),
        "structure_tensor_active_pixels": int(active.sum()),
    }


def scuff_cluster_mask(
    normal_mask: Image.Image,
    soft_mask: Image.Image,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_component_area: int,
    close_radius: int,
    max_components: int,
    max_box_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    normal = np.asarray(normal_mask.convert("L"), dtype=np.uint8) > 0
    soft = np.asarray(soft_mask.convert("L"), dtype=np.uint8) > 0
    combined = (normal | soft)[top:bottom, left:right]
    if combined.size == 0:
        return Image.new("L", image_size, 0), {"scuff_cluster_reason": "empty_crop"}
    combined = _morph_close(combined, radius=max(1, close_radius))
    components = _connected_components(combined)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    for component in components:
        if len(component) < max(2, min_component_area // 2):
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        aspect = max(width / max(1, height), height / max(1, width))
        fill = len(component) / max(1, width * height)
        score = len(component) + min(200.0, aspect * 8.0) - max(0.0, fill - 0.55) * 80.0
        scored.append((score, component))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(combined, dtype=bool)
    for _, component in scored[: max(1, max_components)]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
    selected = _limit_mask_area(selected, combined.astype(np.float32), max_area_fraction=max_box_fraction)
    selected = _filter_components(selected, "scratch", max(2, min_component_area // 3), require_elongated=False)
    if not selected.any():
        selected = _filter_components(combined, "scratch", max(2, min_component_area // 3), require_elongated=False)
        selected = _limit_mask_area(selected, combined.astype(np.float32), max_area_fraction=max_box_fraction)
    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "scuff_cluster_close_radius": int(close_radius),
        "scuff_cluster_components_seen": len(components),
        "scuff_cluster_components_kept": min(len(scored), max_components),
        "scuff_cluster_max_box_fraction": float(max_box_fraction),
        "scuff_cluster_pixels": int(selected.sum()),
    }


def multi_scuff_fusion_mask(
    normal_mask: Image.Image,
    soft_mask: Image.Image,
    fft_heatmap: np.ndarray,
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    text_hint: str,
    min_component_area: int,
    support_percentile: float,
    fine_dilate_radius: int,
    close_radius: int,
    max_components: int,
    max_box_fraction: float,
    grain_reject_angle_degrees: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    normal = np.asarray(normal_mask.convert("L"), dtype=np.uint8) > 0
    soft = np.asarray(soft_mask.convert("L"), dtype=np.uint8) > 0
    fine = (normal | soft)[top:bottom, left:right]
    support_crop = fft_heatmap[top:bottom, left:right]
    if fine.size == 0 or support_crop.size == 0:
        return Image.new("L", image_size, 0), {"multi_scuff_fusion_reason": "empty_crop"}
    if not np.isfinite(support_crop).all() or float(support_crop.max()) <= 0.0:
        support = np.ones_like(fine, dtype=bool)
        support_reason = "fft_support_unavailable"
    else:
        support_threshold = float(np.percentile(support_crop, max(50.0, min(99.0, support_percentile))))
        support = support_crop >= support_threshold
        support = _morph_close(support, radius=max(1, close_radius))
        support_reason = "fft_support"
    if not fine.any():
        fine = support & (support_crop >= float(np.percentile(support_crop, 92.0))) if support_crop.size else support
    fine_support = _dilate_bool(fine, radius=max(1, fine_dilate_radius))
    fused = (fine | (support & fine_support))
    fused = _morph_close(fused, radius=max(1, close_radius))

    dominant_angle = _dominant_texture_angle_degrees(image, normal_paths, region)
    components = _connected_components(fused)
    scored: list[tuple[float, list[tuple[int, int]], dict[str, Any]]] = []
    for component in components:
        if len(component) < max(2, min_component_area // 3):
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        area = len(component)
        fill = area / max(1, width * height)
        aspect = max(width / max(1, height), height / max(1, width))
        angle = _mask_angle_degrees(xs, ys)
        grain_penalty = 0.0
        if dominant_angle is not None and aspect >= 1.6:
            diff = _angle_difference_degrees(angle, dominant_angle)
            if diff < grain_reject_angle_degrees:
                grain_penalty = 0.35
        blob_penalty = max(0.0, fill - 0.55) * 0.35
        fine_overlap = float(fine[ys, xs].mean())
        support_mean = float(support_crop[ys, xs].mean()) if support_crop.size else 0.0
        score = support_mean + 0.35 * fine_overlap + min(0.18, area / max(1, fused.size) * 3.0) - grain_penalty - blob_penalty
        if fill > 0.82 and area > min_component_area * 3:
            score -= 0.30
        scored.append(
            (
                score,
                component,
                {
                    "area": int(area),
                    "angle": float(angle),
                    "fill": float(fill),
                    "aspect": float(aspect),
                    "grain_penalty": float(grain_penalty),
                    "fine_overlap": float(fine_overlap),
                },
            )
        )
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(fused, dtype=bool)
    kept = 0
    grain_rejected = 0
    for score, component, params in scored:
        if kept >= max(1, max_components):
            break
        if float(params["grain_penalty"]) > 0.0 and score < 0.35:
            grain_rejected += 1
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
        kept += 1
    if not selected.any():
        selected = _filter_components(fine, "scratch", max(2, min_component_area // 3), require_elongated=False)
    selected = _limit_mask_area(selected, support_crop, max_area_fraction=max_box_fraction)
    selected = _cap_mask_by_heat(selected, support_crop, max_area_fraction=max_box_fraction)
    selected = _filter_components(selected, "scratch", max(2, min_component_area // 4), require_elongated=False)
    if not selected.any() and fine.any():
        selected = _cap_mask_by_heat(fine, support_crop, max_area_fraction=max_box_fraction)

    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "multi_scuff_fusion_support_reason": support_reason,
        "multi_scuff_fusion_support_percentile": float(support_percentile),
        "multi_scuff_fusion_fine_dilate_radius": int(fine_dilate_radius),
        "multi_scuff_fusion_close_radius": int(close_radius),
        "multi_scuff_fusion_components_seen": len(components),
        "multi_scuff_fusion_components_kept": int(kept),
        "multi_scuff_fusion_grain_rejected": int(grain_rejected),
        "multi_scuff_fusion_dominant_texture_angle_degrees": dominant_angle,
        "multi_scuff_fusion_max_box_fraction": float(max_box_fraction),
        "multi_scuff_fusion_pixels": int(selected.sum()),
        "multi_scuff_fusion_patch_like_hint": bool(_description_implies_patch(text_hint) or _description_implies_multiple(text_hint)),
    }


def normal_residual_fusion_mask(
    fusion_mask: Image.Image,
    residual_mask: Image.Image,
    residual_heatmap: np.ndarray,
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_component_area: int,
    support_dilate_radius: int,
    close_radius: int,
    max_components: int,
    max_box_fraction: float,
    grain_reject_angle_degrees: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    fusion = np.asarray(fusion_mask.convert("L"), dtype=np.uint8) > 0
    residual = np.asarray(residual_mask.convert("L"), dtype=np.uint8) > 0
    support = _dilate_bool(fusion, radius=max(1, support_dilate_radius))
    residual_crop = residual[top:bottom, left:right]
    support_crop = support[top:bottom, left:right]
    heat_crop = residual_heatmap[top:bottom, left:right]
    if residual_crop.size == 0:
        return fusion_mask.convert("L"), {"normal_residual_fusion_reason": "empty_crop"}
    dominant_angle = _dominant_texture_angle_degrees(image, normal_paths, region)
    components = _connected_components(residual_crop)
    selected = np.zeros_like(residual_crop, dtype=bool)
    scored: list[tuple[float, list[tuple[int, int]], bool]] = []
    grain_rejected = 0
    support_kept = 0
    independent_kept = 0
    for component in components:
        area = len(component)
        if area < min_component_area:
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        aspect = max(width / max(1, height), height / max(1, width))
        fill = area / max(1, width * height)
        overlap = float(support_crop[ys, xs].mean()) if support_crop.size else 0.0
        angle = _mask_angle_degrees(xs, ys)
        grain_like = False
        if dominant_angle is not None and aspect >= 1.8:
            grain_like = _angle_difference_degrees(angle, dominant_angle) < grain_reject_angle_degrees
        if grain_like and overlap < 0.25:
            grain_rejected += 1
            continue
        if fill > 0.88 and area > min_component_area * 5 and overlap < 0.35:
            grain_rejected += 1
            continue
        mean_heat = float(heat_crop[ys, xs].mean()) if heat_crop.size else 0.0
        score = mean_heat + min(0.22, area / max(1, residual_crop.size) * 4.0) + min(0.20, overlap * 0.35)
        scored.append((score, component, overlap >= 0.25))
    scored.sort(key=lambda item: item[0], reverse=True)
    for _, component, overlaps_support in scored[: max(1, max_components)]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
        if overlaps_support:
            support_kept += 1
        else:
            independent_kept += 1
    selected_full = np.zeros_like(fusion, dtype=bool)
    selected_full[top:bottom, left:right] = selected
    combined = fusion | selected_full
    combined_crop = combined[top:bottom, left:right]
    if close_radius > 0:
        combined_crop = _morph_close(combined_crop, radius=close_radius)
    combined_crop = _limit_mask_area(combined_crop, heat_crop, max_area_fraction=max_box_fraction)
    combined_crop = _cap_mask_by_heat(combined_crop, heat_crop, max_area_fraction=max_box_fraction)
    output = Image.new("L", image_size, 0)
    output.paste(Image.fromarray(combined_crop.astype(np.uint8) * 255, mode="L"), (left, top))
    return output, {
        "normal_residual_fusion_components_seen": len(components),
        "normal_residual_fusion_components_scored": len(scored),
        "normal_residual_fusion_components_kept": min(len(scored), max_components),
        "normal_residual_fusion_grain_rejected": int(grain_rejected),
        "normal_residual_fusion_support_kept": int(support_kept),
        "normal_residual_fusion_independent_kept": int(independent_kept),
        "normal_residual_fusion_dominant_texture_angle_degrees": dominant_angle,
        "normal_residual_fusion_grain_reject_angle_degrees": float(grain_reject_angle_degrees),
        "normal_residual_fusion_max_box_fraction": float(max_box_fraction),
        "normal_residual_fusion_pixels": int((np.asarray(output, dtype=np.uint8) > 0).sum()),
    }


def support_constrained_fusion_mask(
    base_mask: Image.Image,
    support_heatmap: np.ndarray,
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_component_area: int,
    support_percentile: float,
    support_dilate_radius: int,
    min_support_overlap: float,
    min_component_heat: float,
    min_keep_fraction: float,
    grain_reject_angle_degrees: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    base = np.asarray(base_mask.convert("L"), dtype=np.uint8) > 0
    base_crop = base[top:bottom, left:right]
    heat_crop = support_heatmap[top:bottom, left:right]
    if base_crop.size == 0:
        return Image.new("L", image_size, 0), {"support_constrained_reason": "empty_crop"}
    if not base_crop.any() or heat_crop.size == 0 or not np.isfinite(heat_crop).all() or float(heat_crop.max()) <= 0.0:
        return base_mask.convert("L"), {
            "support_constrained_reason": "support_unavailable",
            "support_constrained_reverted": True,
            "support_constrained_base_pixels": int(base_crop.sum()),
        }

    heat_crop = _normalize_float(heat_crop)
    threshold_values = heat_crop[heat_crop > 0]
    if threshold_values.size == 0:
        threshold_values = heat_crop.reshape(-1)
    threshold = float(np.percentile(threshold_values, max(50.0, min(99.0, support_percentile))))
    support = heat_crop >= threshold
    support = _dilate_bool(support, radius=max(1, support_dilate_radius))
    dominant_angle = _dominant_texture_angle_degrees(image, normal_paths, region)
    selected = np.zeros_like(base_crop, dtype=bool)
    components = _connected_components(base_crop)
    kept = 0
    rejected_low_support = 0
    rejected_grain = 0
    for component in components:
        area = len(component)
        if area < max(2, min_component_area // 4):
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        aspect = max(width / max(1, height), height / max(1, width))
        mean_heat = float(heat_crop[ys, xs].mean())
        overlap = float(support[ys, xs].mean())
        angle = _mask_angle_degrees(xs, ys)
        grain_like = False
        if dominant_angle is not None and aspect >= 1.7:
            grain_like = _angle_difference_degrees(angle, dominant_angle) < grain_reject_angle_degrees
        if grain_like and overlap < max(0.18, min_support_overlap * 1.5) and mean_heat < min_component_heat + 0.08:
            rejected_grain += 1
            continue
        if overlap < min_support_overlap and mean_heat < min_component_heat:
            rejected_low_support += 1
            continue
        selected[ys, xs] = True
        kept += 1

    base_pixels = int(base_crop.sum())
    selected_pixels = int(selected.sum())
    min_pixels = int(base_pixels * max(0.05, min(0.95, min_keep_fraction)))
    reverted = False
    if selected_pixels < min_pixels:
        selected = base_crop
        selected_pixels = base_pixels
        reverted = True
    selected_pixels = int(selected.sum())
    if selected_pixels <= 0:
        selected = base_crop
        selected_pixels = base_pixels
        reverted = True

    output = Image.new("L", image_size, 0)
    output.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return output, {
        "support_constrained_percentile": float(support_percentile),
        "support_constrained_threshold": float(threshold),
        "support_constrained_dilate_radius": int(support_dilate_radius),
        "support_constrained_min_overlap": float(min_support_overlap),
        "support_constrained_min_heat": float(min_component_heat),
        "support_constrained_min_keep_fraction": float(min_keep_fraction),
        "support_constrained_components_seen": len(components),
        "support_constrained_components_kept": int(kept),
        "support_constrained_rejected_low_support": int(rejected_low_support),
        "support_constrained_rejected_grain": int(rejected_grain),
        "support_constrained_dominant_texture_angle_degrees": dominant_angle,
        "support_constrained_base_pixels": int(base_pixels),
        "support_constrained_pixels": int(selected_pixels),
        "support_constrained_reverted": bool(reverted),
    }


def _cap_mask_by_heat(mask: np.ndarray, heat_crop: np.ndarray, *, max_area_fraction: float) -> np.ndarray:
    max_pixels = int(max(1, mask.size * max(0.01, min(0.60, max_area_fraction))))
    if int(mask.sum()) <= max_pixels:
        return mask
    capped = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return capped
    values = heat_crop[ys, xs] if heat_crop.shape == mask.shape else np.ones(len(xs), dtype=np.float32)
    order = np.argsort(values)[::-1][:max_pixels]
    capped[ys[order], xs[order]] = True
    capped = _morph_close(capped, radius=1)
    if int(capped.sum()) > max_pixels:
        ys2, xs2 = np.where(capped)
        values2 = heat_crop[ys2, xs2] if heat_crop.shape == mask.shape else np.ones(len(xs2), dtype=np.float32)
        order2 = np.argsort(values2)[::-1][:max_pixels]
        strict = np.zeros_like(mask, dtype=bool)
        strict[ys2[order2], xs2[order2]] = True
        capped = strict
    return capped


def scratch_band_clean_mask(
    normal_mask: Image.Image,
    multi_mask: Image.Image,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    radius: int,
    min_component_area: int,
    max_area_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    normal = np.asarray(normal_mask.convert("L"), dtype=np.uint8) > 0
    multi = np.asarray(multi_mask.convert("L"), dtype=np.uint8) > 0
    crop = (normal | multi)[top:bottom, left:right]
    if crop.size == 0:
        return Image.new("L", image_size, 0), {"scratch_band_clean_reason": "empty_crop"}
    ys, xs = np.where(crop)
    if len(xs) < max(6, min_component_area):
        mask = Image.new("L", image_size, 0)
        mask.paste(Image.fromarray(crop.astype(np.uint8) * 255, mode="L"), (left, top))
        return mask, {"scratch_band_clean_reason": "too_few_pixels", "scratch_band_clean_pixels": int(crop.sum())}
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    normal_vec = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    along = centered @ direction
    across = centered @ normal_vec
    t_min = float(np.percentile(along, 1.0))
    t_max = float(np.percentile(along, 99.0))
    median_across = float(np.median(across))
    keep_width = float(max(2, radius))
    in_band = (along >= t_min) & (along <= t_max) & (np.abs(across - median_across) <= keep_width)
    cleaned = np.zeros_like(crop, dtype=bool)
    cleaned[ys[in_band], xs[in_band]] = True
    cleaned = _dilate_bool(cleaned, radius=max(1, radius // 3))
    cleaned &= crop | _dilate_bool(crop, radius=max(1, radius // 2))
    cleaned = _filter_components(cleaned, "scratch", max(2, min_component_area // 2), require_elongated=False)
    cleaned = _limit_mask_area(cleaned, crop.astype(np.float32), max_area_fraction=max_area_fraction)
    if not cleaned.any():
        cleaned = _limit_mask_area(crop, crop.astype(np.float32), max_area_fraction=max_area_fraction)
    mask = Image.new("L", image_size, 0)
    mask.paste(Image.fromarray(cleaned.astype(np.uint8) * 255, mode="L"), (left, top))
    return mask, {
        "scratch_band_clean_radius": int(radius),
        "scratch_band_clean_angle_degrees": float(np.degrees(np.arctan2(direction[1], direction[0]))),
        "scratch_band_clean_input_pixels": int(crop.sum()),
        "scratch_band_clean_pixels": int(cleaned.sum()),
        "scratch_band_clean_max_box_fraction": float(max_area_fraction),
    }


def _damage_envelope(
    evidence: np.ndarray,
    heat_crop: np.ndarray,
    *,
    radius: int,
    percentile: float,
    max_area_fraction: float,
) -> np.ndarray:
    support_threshold = float(np.percentile(heat_crop, max(50.0, min(99.0, percentile))))
    support = heat_crop >= support_threshold
    envelope = _dilate_bool(evidence, radius=radius)
    envelope = _morph_close(envelope | (support & _dilate_bool(evidence, radius=max(1, radius // 2))), radius=max(1, radius // 2))
    max_pixels = int(max(1, evidence.size * max(0.01, min(0.60, max_area_fraction))))
    if int(envelope.sum()) > max_pixels:
        components = _connected_components(envelope)
        scored: list[tuple[float, list[tuple[int, int]]]] = []
        for component in components:
            ys = np.asarray([point[0] for point in component])
            xs = np.asarray([point[1] for point in component])
            score = float(heat_crop[ys, xs].mean()) + min(0.2, len(component) / max(1, envelope.size))
            scored.append((score, component))
        scored.sort(key=lambda item: item[0], reverse=True)
        limited = np.zeros_like(envelope, dtype=bool)
        used = 0
        for _, component in scored:
            if used >= max_pixels:
                break
            ys = np.asarray([point[0] for point in component])
            xs = np.asarray([point[1] for point in component])
            limited[ys, xs] = True
            used += len(component)
        envelope = limited
    return envelope


def _scratch_band_envelope(
    evidence: np.ndarray,
    heat_crop: np.ndarray,
    *,
    text_hint: str,
    radius: int,
    percentile: float,
    max_area_fraction: float,
) -> tuple[np.ndarray, str]:
    ys, xs = np.where(evidence)
    if len(xs) < 4:
        return _damage_envelope(evidence, heat_crop, radius=radius, percentile=percentile, max_area_fraction=max_area_fraction), "patch_fallback"

    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = np.median(points, axis=0)
    centered = points - center
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(values.argmax())]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return _damage_envelope(evidence, heat_crop, radius=radius, percentile=percentile, max_area_fraction=max_area_fraction), "patch_fallback"
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    t = centered @ direction
    d = centered @ normal
    t_min = float(np.percentile(t, 2))
    t_max = float(np.percentile(t, 98))
    if t_max - t_min < 4:
        t_min = float(t.min())
        t_max = float(t.max())
    distance_width = float(np.percentile(np.abs(d - np.median(d)), 80)) * 2.0
    width = int(max(3, min(18, round(distance_width + radius * 1.5))))
    line_center = center + normal * float(np.median(d))
    p1 = line_center + direction * t_min
    p2 = line_center + direction * t_max

    band_image = Image.new("L", (evidence.shape[1], evidence.shape[0]), 0)
    draw = ImageDraw.Draw(band_image)
    draw.line((int(round(p1[0])), int(round(p1[1])), int(round(p2[0])), int(round(p2[1]))), fill=255, width=width)
    band = np.asarray(band_image, dtype=np.uint8) > 0

    support_threshold = float(np.percentile(heat_crop[heat_crop > 0], max(50.0, min(99.0, percentile)))) if np.any(heat_crop > 0) else 1.0
    support = heat_crop >= support_threshold
    support = _dilate_bool(support | evidence, radius=max(1, radius))
    envelope = (band & support) | _dilate_bool(evidence, radius=max(1, radius // 2))
    envelope = _morph_close(envelope, radius=max(1, radius // 3))
    envelope = _limit_mask_area(envelope, heat_crop, max_area_fraction=max_area_fraction)
    if int(envelope.sum()) < max(8, int(evidence.sum())):
        envelope = band | _dilate_bool(evidence, radius=max(1, radius // 2))
        envelope = _limit_mask_area(envelope, heat_crop, max_area_fraction=max_area_fraction)
    return envelope, "scratch_band"


def _limit_mask_area(mask: np.ndarray, heat_crop: np.ndarray, *, max_area_fraction: float) -> np.ndarray:
    max_pixels = int(max(1, mask.size * max(0.01, min(0.60, max_area_fraction))))
    if int(mask.sum()) <= max_pixels:
        return mask
    components = _connected_components(mask)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    for component in components:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        score = float(heat_crop[ys, xs].mean()) + min(0.2, len(component) / max(1, mask.size))
        scored.append((score, component))
    scored.sort(key=lambda item: item[0], reverse=True)
    limited = np.zeros_like(mask, dtype=bool)
    used = 0
    for _, component in scored:
        if used >= max_pixels:
            break
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        limited[ys, xs] = True
        used += len(component)
    return limited


def _mask_crop_aspect(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return max(width / max(1, height), height / max(1, width))


def _filter_anomaly_components(
    active: np.ndarray,
    heat_crop: np.ndarray,
    defect_type: str,
    text_hint: str,
    *,
    min_component_area: int,
    max_components: int,
    dominant_texture_angle: float | None,
    suppress_parallel_texture: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    components = _connected_components(active)
    scored: list[tuple[float, list[tuple[int, int]], dict[str, float]]] = []
    wants_vertical = "vertical" in text_hint.lower()
    for component in components:
        area = len(component)
        if area < min_component_area:
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        fill = area / max(1, width * height)
        aspect = max(width / max(1, height), height / max(1, width))
        if fill > 0.80 and area > min_component_area * 4:
            continue
        angle = _mask_angle_degrees(xs, ys)
        mean_heat = float(heat_crop[ys, xs].mean())
        parallel_penalty = 0.0
        if (
            suppress_parallel_texture
            and defect_type == "scratch"
            and dominant_texture_angle is not None
            and not wants_vertical
            and aspect >= 2.0
        ):
            difference = _angle_difference_degrees(angle, dominant_texture_angle)
            if difference < 18.0:
                parallel_penalty = 0.40
        score = mean_heat + min(0.20, area / max(1, active.size) * 4.0) + min(0.12, aspect / 40.0) - parallel_penalty
        scored.append((score, component, {"angle": angle, "area": float(area), "parallel_penalty": parallel_penalty}))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:max_components]
    output = np.zeros_like(active, dtype=bool)
    for _, component, _ in selected:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        output[ys, xs] = True
    return output, {
        "normal_anomaly_components_seen": len(components),
        "normal_anomaly_components_kept": len(selected),
        "normal_anomaly_parallel_components_penalized": sum(
            1 for _, _, params in scored if params["parallel_penalty"] > 0
        ),
    }


def _filter_heatmap_mask(mask: Image.Image, defect_type: str, min_component_area: int) -> Image.Image:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    filtered = _filter_components(arr, defect_type, min_component_area)
    if filtered.sum() == 0:
        filtered = arr
    return Image.fromarray(filtered.astype(np.uint8) * 255).convert("L")


def _predict_sam_mask(
    image: Image.Image,
    region: tuple[int, int, int, int],
    point_coords: list[list[int]],
    point_labels: list[int],
    auto: dict[str, Any],
) -> tuple[Image.Image | None, dict[str, Any]]:
    provider = str(auto.get("sam_provider", "auto"))
    if provider in {"auto", "sam2"} and importlib.util.find_spec("sam2") is not None:
        return _predict_sam2_mask(image, region, point_coords, point_labels, auto)
    if provider in {"auto", "sam"} and importlib.util.find_spec("segment_anything") is not None:
        return _predict_sam1_mask(image, region, point_coords, point_labels, auto)
    return None, {"sam_provider": provider, "sam_available": False, "sam_reason": "sam2 and segment_anything are not installed"}


def _predict_sam2_mask(
    image: Image.Image,
    region: tuple[int, int, int, int],
    point_coords: list[list[int]],
    point_labels: list[int],
    auto: dict[str, Any],
) -> tuple[Image.Image | None, dict[str, Any]]:
    checkpoint = auto.get("sam2_checkpoint")
    model_cfg = auto.get("sam2_model_cfg")
    if not checkpoint or not model_cfg:
        return None, {"sam_provider": "sam2", "sam_available": False, "sam_reason": "sam2_checkpoint or sam2_model_cfg missing"}
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = str(auto.get("sam_device", "cuda" if torch.cuda.is_available() else "cpu"))
    key = (str(model_cfg), str(checkpoint), device)
    predictor = _SAM2_PREDICTOR_CACHE.get(key)
    if predictor is None:
        model = build_sam2(str(model_cfg), str(checkpoint), device=device)
        predictor = SAM2ImagePredictor(model)
        _SAM2_PREDICTOR_CACHE[key] = predictor
    predictor.set_image(np.asarray(image.convert("RGB")))
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(point_coords, dtype=np.float32),
        point_labels=np.asarray(point_labels, dtype=np.int32),
        box=np.asarray(region, dtype=np.float32),
        multimask_output=True,
    )
    index = int(np.argmax(scores))
    mask = Image.fromarray(masks[index].astype(np.uint8) * 255).convert("L")
    return mask, {"sam_provider": "sam2", "sam_available": True, "sam_score": float(scores[index])}


def _predict_sam1_mask(
    image: Image.Image,
    region: tuple[int, int, int, int],
    point_coords: list[list[int]],
    point_labels: list[int],
    auto: dict[str, Any],
) -> tuple[Image.Image | None, dict[str, Any]]:
    checkpoint = auto.get("sam_checkpoint")
    if not checkpoint:
        return None, {"sam_provider": "sam", "sam_available": False, "sam_reason": "sam_checkpoint missing"}
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    model_type = str(auto.get("sam_model_type", "vit_b"))
    device = str(auto.get("sam_device", "cuda" if torch.cuda.is_available() else "cpu"))
    key = (model_type, str(checkpoint), device)
    predictor = _SAM1_PREDICTOR_CACHE.get(key)
    if predictor is None:
        model = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device)
        predictor = SamPredictor(model)
        _SAM1_PREDICTOR_CACHE[key] = predictor
    predictor.set_image(np.asarray(image.convert("RGB")))
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray(point_coords, dtype=np.float32),
        point_labels=np.asarray(point_labels, dtype=np.int32),
        box=np.asarray(region, dtype=np.float32),
        multimask_output=True,
    )
    index = int(np.argmax(scores))
    mask = Image.fromarray(masks[index].astype(np.uint8) * 255).convert("L")
    return mask, {"sam_provider": "sam", "sam_available": True, "sam_score": float(scores[index])}


def _normalize_float(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _robust_region_calibrated_heatmap(heatmap: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    heat = np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    left, top, right, bottom = region
    crop = heat[top:bottom, left:right]
    if crop.size == 0:
        return np.zeros_like(heat, dtype=np.float32)
    median = float(np.median(crop))
    mad = float(np.median(np.abs(crop - median))) + 1e-6
    z = np.clip((heat - median) / (1.4826 * mad), 0.0, 8.0)
    calibrated = _normalize_float(z)
    fenced = np.zeros_like(calibrated, dtype=np.float32)
    fenced[top:bottom, left:right] = calibrated[top:bottom, left:right]
    return fenced


def _heatmap_region_contrast(heatmap: np.ndarray, region: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = region
    crop = heatmap[top:bottom, left:right]
    if crop.size == 0:
        return 0.0
    return float(np.percentile(crop, 95.0) - np.percentile(crop, 50.0))


def _write_heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    mask: Image.Image,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = image.convert("RGB")
    heat = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(Image.fromarray(heat, mode="L").point(lambda value: int(value * 0.55)))
    overlay = Image.alpha_composite(base.convert("RGBA"), red)
    outline = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(outline)
    draw.rectangle((region[0], region[1], region[2] - 1, region[3] - 1), outline=(255, 230, 0, 210), width=2)
    mask_edges = mask.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_overlay = Image.new("RGBA", base.size, (0, 255, 0, 0))
    edge_overlay.putalpha(mask_edges.point(lambda value: 220 if value > 0 else 0))
    overlay = Image.alpha_composite(overlay, outline)
    overlay = Image.alpha_composite(overlay, edge_overlay)
    overlay.convert("RGB").save(path)


def linearized_scratch_mask(
    pixel_mask: Image.Image,
    image_size: tuple[int, int],
    region: tuple[int, int, int, int],
    defect_type: str,
) -> tuple[Image.Image, dict[str, Any]]:
    if defect_type not in {"scratch", "crack"}:
        return pixel_mask, {"kind": "linear_passthrough", "linearized": False}
    left, top, right, bottom = region
    arr = np.asarray(pixel_mask.convert("L"), dtype=np.uint8) > 0
    crop = arr[top:bottom, left:right]
    ys, xs = np.where(crop)
    if len(xs) < 6:
        return pixel_mask, {"kind": "linear_passthrough", "linearized": False, "reason": "too_few_pixels"}

    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = np.median(points, axis=0)
    centered = points - center
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(values.argmax())]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return pixel_mask, {"kind": "linear_passthrough", "linearized": False, "reason": "degenerate_direction"}
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    t = centered @ direction
    d = centered @ normal
    offset = float(np.median(d))
    spread = float(np.percentile(np.abs(d - offset), 60))
    inliers = np.abs(d - offset) <= max(2.5, spread)
    if int(inliers.sum()) < 4:
        inliers = np.ones_like(t, dtype=bool)
    t_min = float(np.percentile(t[inliers], 5))
    t_max = float(np.percentile(t[inliers], 95))
    if t_max - t_min < 4:
        t_min = float(t.min())
        t_max = float(t.max())
    line_center = center + normal * offset
    p1 = line_center + direction * t_min
    p2 = line_center + direction * t_max
    width = max(2, min(10, round(min(right - left, bottom - top) / 18)))

    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    draw.line(
        (
            int(round(left + p1[0])),
            int(round(top + p1[1])),
            int(round(left + p2[0])),
            int(round(top + p2[1])),
        ),
        fill=255,
        width=width,
    )
    return mask, {
        "kind": "linearized_scratch",
        "linearized": True,
        "source_pixels": int(len(xs)),
        "inlier_pixels": int(inliers.sum()),
        "line_width": int(width),
        "line_angle_degrees": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }


def multi_linearized_scratch_mask(
    pixel_mask: Image.Image,
    image_size: tuple[int, int],
    region: tuple[int, int, int, int],
    defect_type: str,
    *,
    max_lines: int,
) -> tuple[Image.Image, dict[str, Any]]:
    if defect_type not in {"scratch", "crack"}:
        return pixel_mask, {"kind": "multi_linear_passthrough", "linearized": False}
    left, top, right, bottom = region
    arr = np.asarray(pixel_mask.convert("L"), dtype=np.uint8) > 0
    crop = arr[top:bottom, left:right]
    ys, xs = np.where(crop)
    if len(xs) < 8:
        return linearized_scratch_mask(pixel_mask, image_size, region, defect_type)

    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = np.median(points, axis=0)
    centered = points - center
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    direction = vectors[:, int(values.argmax())]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return linearized_scratch_mask(pixel_mask, image_size, region, defect_type)
    direction = direction / norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    offsets = centered @ normal
    bins = _offset_bins(offsets, max_lines=max(1, max_lines))
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    line_count = 0
    width = max(2, min(8, round(min(right - left, bottom - top) / 24)))
    for start, end in bins:
        selected = (offsets >= start) & (offsets <= end)
        if int(selected.sum()) < 5:
            continue
        sub_points = points[selected]
        sub_center = np.median(sub_points, axis=0)
        sub_centered = sub_points - sub_center
        t = sub_centered @ direction
        if float(np.percentile(t, 95) - np.percentile(t, 5)) < 5:
            continue
        t_min = float(np.percentile(t, 5))
        t_max = float(np.percentile(t, 95))
        p1 = sub_center + direction * t_min
        p2 = sub_center + direction * t_max
        draw.line(
            (
                int(round(left + p1[0])),
                int(round(top + p1[1])),
                int(round(left + p2[0])),
                int(round(top + p2[1])),
            ),
            fill=255,
            width=width,
        )
        line_count += 1
    if line_count == 0:
        return linearized_scratch_mask(pixel_mask, image_size, region, defect_type)
    return mask, {
        "kind": "multi_linearized_scratch",
        "linearized": True,
        "source_pixels": int(len(xs)),
        "line_count": line_count,
        "line_width": int(width),
        "line_angle_degrees": float(np.degrees(np.arctan2(direction[1], direction[0]))),
    }


def _offset_bins(offsets: np.ndarray, max_lines: int) -> list[tuple[float, float]]:
    if len(offsets) == 0:
        return []
    hist, edges = np.histogram(offsets, bins=min(24, max(6, len(offsets) // 8)))
    peaks = sorted(range(len(hist)), key=lambda index: int(hist[index]), reverse=True)
    bins: list[tuple[float, float]] = []
    for peak in peaks:
        if int(hist[peak]) < 4:
            continue
        start = max(0, peak - 1)
        end = min(len(edges) - 2, peak + 1)
        candidate = (float(edges[start]), float(edges[end + 1]))
        center = (candidate[0] + candidate[1]) / 2
        if any(existing[0] <= center <= existing[1] for existing in bins):
            continue
        bins.append(candidate)
        if len(bins) >= max_lines:
            break
    return sorted(bins)


def _generate_ensemble_consensus(
    candidates: list[dict[str, Any]],
    image: Image.Image,
    region: tuple[int, int, int, int],
    defect_type: str,
    output_dir: Path,
    stem: str,
    auto: dict[str, Any],
    text_hint: str,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No candidates provided for ensemble")
        
    import cv2
    mask_arrays = []
    for cand in candidates:
        refined_path = cand.get("refined_mask_path")
        if refined_path and Path(refined_path).exists():
            mask_im = Image.open(refined_path).convert("L")
            arr = np.asarray(mask_im)
            bin_arr = (arr > 127).astype(np.uint8)
            
            # Topological Pruning: Prune detached islands < 5% of total mask area
            total_area = bin_arr.sum()
            if total_area > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_arr, connectivity=8)
                cleaned_arr = np.zeros_like(bin_arr)
                for label in range(1, num_labels):
                    comp_area = stats[label, cv2.CC_STAT_AREA]
                    if comp_area >= 0.05 * total_area:
                        cleaned_arr[labels == label] = 1
                mask_arrays.append(cleaned_arr)
            else:
                mask_arrays.append(bin_arr)
            
    if not mask_arrays:
        raise ValueError("Could not load refined masks for ensemble")
        
    sum_arr = np.sum(mask_arrays, axis=0)
    consensus_arr = (sum_arr >= 2).astype(np.uint8) * 255
    consensus_im = Image.fromarray(consensus_arr, mode="L")
    consensus_im = _filter_heatmap_mask(consensus_im, defect_type, int(auto.get("min_component_area", 12)))
    
    padding = int(auto.get("shrink_padding_px", 6))
    shrunk_region = _padded_bbox(consensus_im, image.size, padding) or region
    qc = _qc_mask(consensus_im, region, shrunk_region, image.size, defect_type)
    inpaint = consensus_im.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    
    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    
    box.save(box_path)
    consensus_im.save(refined_path)
    inpaint.save(inpaint_path)
    
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": "ensemble_consensus",
            "shrunk_region_xyxy": shrunk_region,
            "voted_from": len(mask_arrays),
        },
        "qc": qc,
        "score": _score_candidate(consensus_im, image, qc, "ensemble_consensus", defect_type, text_hint),
    }


def _add_derived_recall_safe_candidates(
    *,
    candidates: dict[str, dict[str, Any]],
    valid_candidates: dict[str, dict[str, Any]],
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    artifact_stem: str,
    auto: dict[str, Any],
    description: str,
) -> None:
    if category == "zipper" and defect_type == "broken_teeth":
        if "zipper_tooth_agreement" not in candidates:
            derived = _combine_candidate_masks(
                candidates,
                required_modes=("patchcore_guided", "musc_mutual_score"),
                output_mode="zipper_tooth_agreement",
                operation="intersection",
                image=image,
                region=region,
                output_dir=output_dir,
                artifact_stem=artifact_stem,
                defect_type=defect_type,
                description=description,
                min_component_area=max(2, int(auto.get("min_component_area", 12)) // 3),
            )
            if derived is not None:
                candidates["zipper_tooth_agreement"] = derived
                if _candidate_is_valid(derived):
                    valid_candidates["zipper_tooth_agreement"] = derived
    if category == "bottle" and defect_type == "contamination":
        if "bottle_contamination_recall_fusion" not in candidates:
            fft_qc = candidates.get("fft_texture_suppression", {}).get("qc", {})
            sam_qc = candidates.get("sam2_heatmap", {}).get("qc", {})
            fft_reasons = {str(reason) for reason in fft_qc.get("reasons", [])}
            sam_reasons = {str(reason) for reason in sam_qc.get("reasons", [])}
            # Strictly target the case where FFT is broad but coherent, and SAM adds
            # a single broader contamination support. Avoid the many-component SAM
            # case, which was empirically a false-expansion pattern.
            if (
                "large_image_area" in fft_reasons
                and "large_fraction_of_qwen_box" in sam_reasons
                and "many_components" not in sam_reasons
                and int(sam_qc.get("component_count", 999)) <= 20
            ):
                derived = _combine_candidate_masks(
                    candidates,
                    required_modes=("fft_texture_suppression", "sam2_heatmap"),
                    output_mode="bottle_contamination_recall_fusion",
                    operation="union",
                    image=image,
                    region=region,
                    output_dir=output_dir,
                    artifact_stem=artifact_stem,
                    defect_type=defect_type,
                    description=description,
                    min_component_area=max(2, int(auto.get("min_component_area", 12)) // 3),
                )
                if derived is not None:
                    candidates["bottle_contamination_recall_fusion"] = derived
                    if _candidate_is_valid(derived):
                        valid_candidates["bottle_contamination_recall_fusion"] = derived


def _combine_candidate_masks(
    candidates: dict[str, dict[str, Any]],
    *,
    required_modes: tuple[str, ...],
    output_mode: str,
    operation: str,
    image: Image.Image,
    region: tuple[int, int, int, int],
    output_dir: Path,
    artifact_stem: str,
    defect_type: str,
    description: str,
    min_component_area: int,
) -> dict[str, Any] | None:
    arrays: list[np.ndarray] = []
    source_paths: dict[str, str] = {}
    for mode in required_modes:
        candidate = candidates.get(mode)
        if not candidate or not _candidate_is_valid(candidate):
            return None
        path_value = candidate.get("refined_mask_path")
        if not path_value or not Path(str(path_value)).exists():
            return None
        arrays.append(np.asarray(Image.open(str(path_value)).convert("L"), dtype=np.uint8) > 0)
        source_paths[mode] = str(path_value)
    if not arrays:
        return None
    if operation == "intersection":
        combined = np.logical_and.reduce(arrays)
    elif operation == "union":
        combined = np.logical_or.reduce(arrays)
    else:
        raise ValueError(f"Unsupported derived candidate operation: {operation}")
    combined = _filter_components_by_area(combined, min_area=max(1, int(min_component_area)))
    if int(combined.sum()) <= 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{artifact_stem}_{output_mode}"
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    refined = Image.fromarray(combined.astype(np.uint8) * 255, mode="L")
    inpaint = refined.filter(ImageFilter.MaxFilter(size=5)).filter(ImageFilter.GaussianBlur(radius=1.4))
    heat = refined.filter(ImageFilter.GaussianBlur(radius=1.0))
    shrunk_region = _padded_bbox(refined, image.size, 0) or region
    qc = _qc_mask(refined, region, shrunk_region, image.size, defect_type)

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    heat.save(heatmap_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": {
            "kind": output_mode,
            "operation": operation,
            "source_modes": list(required_modes),
            "source_paths": source_paths,
            "heatmap_path": str(heatmap_path),
            "shrunk_region_xyxy": shrunk_region,
        },
        "qc": qc,
        "score": _score_candidate(refined, image, qc, output_mode, defect_type, description),
        "selected_refinement": output_mode,
    }


def _select_mask_artifacts(
    *,
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    artifact_stem: str,
    seed: int,
    auto: dict[str, Any],
    description: str,
    normal_paths: list[Path],
    requested: str,
    extractor: Any = None,
    localization_status: str = "unknown",
) -> dict[str, Any]:
    feature_cache: PatchFeatureCache | None = {} if bool(auto.get("enable_patch_feature_cache", True)) else None
    if requested != "auto":
        selected = _make_mask_candidate(
            mode=requested,
            image=image,
            category=category,
            defect_type=defect_type,
            region=region,
            output_dir=output_dir,
            artifact_stem=artifact_stem,
            seed=seed,
            auto=auto,
            description=description,
            normal_paths=normal_paths,
            feature_cache=feature_cache,
        )
        selected["selected_refinement"] = requested
        selected["candidate_scores"] = {requested: selected.get("score", 1.0)}
        return selected

    hierarchy = [
        mode
        for mode in _auto_candidate_modes(auto)
        if _candidate_mode_applicable(
            mode,
            category=category,
            defect_type=defect_type,
            description=description,
        )
    ]
    if not hierarchy:
        hierarchy = ["procedural"]
    candidates: dict[str, dict[str, Any]] = {}
    candidate_failures: dict[str, str] = {}
    
    fast_neural_modes = [m for m in hierarchy if m in ("normal_anomaly", "delta_deno")]
    other_modes = [m for m in hierarchy if m not in fast_neural_modes]
    
    early_exit = False
    allow_early_exit = bool(auto.get("allow_neural_early_exit", False))
    
    for mode in fast_neural_modes:
        try:
            candidate = _make_mask_candidate(
                mode=mode,
                image=image,
                category=category,
                defect_type=defect_type,
                region=region,
                output_dir=output_dir,
                artifact_stem=f"{artifact_stem}_{mode}",
                seed=seed,
                auto=auto,
                description=description,
                normal_paths=normal_paths,
                feature_cache=feature_cache,
            )
            candidates[mode] = candidate
            
            if allow_early_exit and _candidate_is_valid(candidate):
                refined_path = candidate.get("refined_mask_path")
                if refined_path and Path(refined_path).exists():
                    mask_im = Image.open(refined_path).convert("L")
                    mask_arr = np.asarray(mask_im)
                    image_arr = np.asarray(image.convert("L"))
                    
                    defect_pixels = image_arr[mask_arr > 127]
                    bg_pixels = image_arr[mask_arr <= 127]
                    
                    if len(defect_pixels) > 0 and len(bg_pixels) > 0:
                        snr = abs(np.mean(defect_pixels) - np.mean(bg_pixels)) / (np.std(bg_pixels) + 1e-6)
                        if snr > 5.0:
                            early_exit = True
                            print(f"Early exit triggered by {mode} with SNR {snr:.2f}")
                            break
        except Exception as exc:
            candidate_failures[mode] = str(exc)

    if not early_exit:
        for mode in other_modes:
            try:
                candidate = _make_mask_candidate(
                    mode=mode,
                    image=image,
                    category=category,
                    defect_type=defect_type,
                    region=region,
                    output_dir=output_dir,
                    artifact_stem=f"{artifact_stem}_{mode}",
                    seed=seed,
                    auto=auto,
                    description=description,
                    normal_paths=normal_paths,
                    feature_cache=feature_cache,
                )
                candidates[mode] = candidate
            except Exception as exc:
                candidate_failures[mode] = str(exc)

    valid_candidates = {
        mode: candidate
        for mode, candidate in candidates.items()
        if _candidate_is_valid(candidate)
    }
    policy = _selection_policy(
        category=category,
        defect_type=defect_type,
        region=region,
        image_size=image.size,
        description=description,
        candidates=valid_candidates,
        localization_status=localization_status,
        auto=auto,
    )
    adjusted_scores = {
        mode: _policy_adjusted_score(mode, candidate, policy)
        for mode, candidate in valid_candidates.items()
    }
    
    if len(valid_candidates) >= 2 and bool(auto.get("use_ensemble_consensus", True)):
        sorted_modes = sorted(valid_candidates.keys(), key=lambda m: (adjusted_scores.get(m, -1.0), -hierarchy.index(m)), reverse=True)
        top_modes = sorted_modes[:3]
        top_cands = [valid_candidates[m] for m in top_modes]
        try:
            ensemble = _generate_ensemble_consensus(
                top_cands, image, region, defect_type, output_dir, f"{artifact_stem}_ensemble_consensus", auto, description
            )
            if _candidate_is_valid(ensemble):
                candidates["ensemble_consensus"] = ensemble
                valid_candidates["ensemble_consensus"] = ensemble
                ensemble_bonus = float(
                    policy.get(
                        "ensemble_bonus",
                        -0.03 if policy.get("scratch_morphology_class") == "multi_scuff" else 0.2,
                    )
                )
                if bool(policy.get("score_generated_ensemble_independently", False)):
                    adjusted_scores["ensemble_consensus"] = round(
                        float(max(-1.0, min(1.50, _policy_adjusted_score("ensemble_consensus", ensemble, policy) + ensemble_bonus))),
                        4,
                    )
                else:
                    adjusted_scores["ensemble_consensus"] = adjusted_scores[top_modes[0]] + ensemble_bonus
                if "ensemble_consensus" not in hierarchy:
                    hierarchy.append("ensemble_consensus")
        except Exception as e:
            candidate_failures["ensemble_consensus"] = str(e)

    _add_derived_recall_safe_candidates(
        candidates=candidates,
        valid_candidates=valid_candidates,
        image=image,
        category=category,
        defect_type=defect_type,
        region=region,
        output_dir=output_dir,
        artifact_stem=artifact_stem,
        auto=auto,
        description=description,
    )
    for mode, candidate in valid_candidates.items():
        if mode not in adjusted_scores:
            adjusted_scores[mode] = _policy_adjusted_score(mode, candidate, policy)
        if mode not in hierarchy:
            hierarchy.append(mode)
            
    vlm_judgment = None
    if valid_candidates and bool(auto.get("use_vlm_critic", True)) and extractor is not None:
        sorted_modes = sorted(valid_candidates.keys(), key=lambda m: (adjusted_scores.get(m, -1.0), -hierarchy.index(m)), reverse=True)
        top_score = adjusted_scores[sorted_modes[0]]
        if len(sorted_modes) >= 2:
            second_score = adjusted_scores[sorted_modes[1]]
            if top_score - second_score < 0.15:
                critic_winner = _invoke_vlm_critic(
                    image, region, description,
                    sorted_modes[:3], valid_candidates, extractor
                )
                if critic_winner and critic_winner in adjusted_scores:
                    vlm_judgment = f"Critic chose {critic_winner}"
                    adjusted_scores[critic_winner] += 0.5
                    top_score = adjusted_scores[critic_winner]
        
        if top_score < 0.4:
            pad = 50
            cx1 = max(0, region[0] - pad)
            cy1 = max(0, region[1] - pad)
            cx2 = min(image.width, region[2] + pad)
            cy2 = min(image.height, region[3] + pad)
            crop = image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
            
            prompt = (
                f"The system failed to find the '{description}' anomaly mask. "
                "Look closely at the image inside the bounding box area. "
                "Output the EXACT [X, Y] pixel coordinate of the thickest or most visible part of the defect. "
                "Output ONLY a JSON list, e.g. [123, 456]."
            )
            
            try:
                result = extractor.analyze(crop, prompt)
                text = result.get("text", "").strip()
                import ast
                start = text.find('[')
                end = text.rfind(']')
                if start != -1 and end != -1:
                    coords = ast.literal_eval(text[start:end+1])
                    if len(coords) == 2:
                        point_x = int(coords[0]) + cx1
                        point_y = int(coords[1]) + cy1
                        
                        vlm_candidate = _make_mask_candidate(
                            mode="vlm_point_sam2",
                            image=image,
                            category=category,
                            defect_type=defect_type,
                            region=region,
                            output_dir=output_dir,
                            artifact_stem=f"{artifact_stem}_vlm_point_sam2",
                            seed=seed,
                            auto=auto,
                            description=description,
                            normal_paths=normal_paths,
                            vlm_point=(point_x, point_y),
                        )
                        
                        if _candidate_is_valid(vlm_candidate):
                            candidates["vlm_point_sam2"] = vlm_candidate
                            valid_candidates["vlm_point_sam2"] = vlm_candidate
                            adjusted_scores["vlm_point_sam2"] = 1.0
                            hierarchy.append("vlm_point_sam2")
                            vlm_judgment = "VLM Auto-Correction invoked"
            except Exception as e:
                candidate_failures["vlm_point_sam2"] = str(e)

    if valid_candidates:
        selected_mode, selected, selection_arbitration = _choose_candidate_with_arbitration(
            valid_candidates,
            adjusted_scores=adjusted_scores,
            hierarchy=hierarchy,
            policy=policy,
        )
    else:
        selected_mode = "procedural"
        if "procedural" in candidates:
            selected = candidates["procedural"]
        else:
            selected = _make_mask_candidate(
                mode="procedural",
                image=image,
                category=category,
                defect_type=defect_type,
                region=region,
                output_dir=output_dir,
                artifact_stem=f"{artifact_stem}_procedural",
                seed=seed,
                auto=auto,
                description=description,
                normal_paths=normal_paths,
            )
            candidates["procedural"] = selected
        selection_arbitration = {
            "applied": False,
            "initial_selected_refinement": selected_mode,
            "final_selected_refinement": selected_mode,
            "reason": "fallback_procedural",
        }

    final_stem = artifact_stem
    copied = _copy_candidate_as_final(selected, output_dir, final_stem)
    copied["selected_refinement"] = selected_mode
    copied["candidate_scores"] = {mode: candidate.get("score", 1.0) for mode, candidate in candidates.items()}
    copied["candidate_qc"] = {mode: candidate.get("qc", {}) for mode, candidate in candidates.items()}
    copied["candidate_refined_paths"] = {
        mode: str(candidate.get("refined_mask_path"))
        for mode, candidate in candidates.items()
        if candidate.get("refined_mask_path") and Path(str(candidate.get("refined_mask_path"))).exists()
    }
    copied["candidate_heatmap_paths"] = {
        mode: str(candidate.get("parameters", {}).get("heatmap_path"))
        for mode, candidate in candidates.items()
        if candidate.get("parameters", {}).get("heatmap_path")
        and Path(str(candidate.get("parameters", {}).get("heatmap_path"))).exists()
    }
    copied["candidate_failures"] = candidate_failures
    copied["candidate_modes"] = hierarchy
    copied["policy_scores"] = adjusted_scores
    copied["vlm_judgment"] = vlm_judgment
    copied["scratch_morphology_class"] = policy["scratch_morphology_class"]
    copied["structure_profile"] = policy.get("structure_profile", "unknown")
    copied["selection_policy"] = policy["selection_policy"]
    copied["candidate_rejections"] = policy["candidate_rejections"]
    copied["selection_arbitration"] = selection_arbitration
    return copied


def _choose_candidate_with_arbitration(
    valid_candidates: dict[str, dict[str, Any]],
    *,
    adjusted_scores: dict[str, float],
    hierarchy: list[str],
    policy: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selected_mode, selected = max(
        valid_candidates.items(),
        key=lambda item: (
            adjusted_scores.get(item[0], -1.0),
            -hierarchy.index(item[0]) if item[0] in hierarchy else -len(hierarchy),
        ),
    )
    arbitration: dict[str, Any] = {
        "applied": False,
        "initial_selected_refinement": selected_mode,
        "final_selected_refinement": selected_mode,
        "reason": "score_selection",
    }
    repeated_candidate = valid_candidates.get("repeated_chain_refiner")
    if bool(policy.get("repeated_chain_gate_enabled", False)) and repeated_candidate is not None:
        arbitration = {
            "applied": True,
            "initial_selected_refinement": selected_mode,
            "final_selected_refinement": "repeated_chain_refiner",
            "reason": str(policy.get("repeated_chain_gate_reason", "repeated_chain_gate")),
            "gate": {
                key: value
                for key, value in policy.items()
                if str(key).startswith("repeated_chain_gate_")
            },
        }
        selected_mode = "repeated_chain_refiner"
        selected = repeated_candidate
        return selected_mode, selected, arbitration
    recall_safe_mode, recall_safe_reason, recall_safe_details = _recall_safe_selector_override(
        selected_mode=selected_mode,
        valid_candidates=valid_candidates,
        policy=policy,
    )
    if recall_safe_mode and recall_safe_mode in valid_candidates:
        arbitration = {
            "applied": True,
            "initial_selected_refinement": selected_mode,
            "final_selected_refinement": recall_safe_mode,
            "reason": recall_safe_reason,
            "gate": recall_safe_details,
        }
        selected_mode = recall_safe_mode
        selected = valid_candidates[recall_safe_mode]
    return selected_mode, selected, arbitration


def _recall_safe_selector_override(
    *,
    selected_mode: str,
    valid_candidates: dict[str, dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[str | None, str, dict[str, Any]]:
    selection_policy = str(policy.get("selection_policy", ""))
    if selection_policy == "bottle_contamination_prefer_texture_or_sam":
        selected = valid_candidates.get(selected_mode)
        fusion = valid_candidates.get("bottle_contamination_recall_fusion")
        if selected_mode == "fft_texture_suppression" and selected is not None and fusion is not None:
            selected_qc = selected.get("qc", {})
            fusion_qc = fusion.get("qc", {})
            selected_reasons = {str(reason) for reason in selected_qc.get("reasons", [])}
            fusion_reasons = {str(reason) for reason in fusion_qc.get("reasons", [])}
            selected_area = max(1, int(selected_qc.get("mask_area", 0)))
            fusion_area = max(1, int(fusion_qc.get("mask_area", 0)))
            area_ratio = fusion_area / selected_area
            if (
                "large_image_area" in selected_reasons
                and {"large_image_area", "large_fraction_of_qwen_box"}.issubset(fusion_reasons)
                and 1.25 <= area_ratio <= 1.85
                and float(fusion_qc.get("mask_to_qwen_box_fraction", 1.0)) <= 0.65
                and int(fusion_qc.get("component_count", 999)) <= 12
            ):
                return "bottle_contamination_recall_fusion", "broad_fft_uses_coherent_sam_recall_fusion", {
                    "selected_area": selected_area,
                    "fusion_area": fusion_area,
                    "area_ratio": float(area_ratio),
                    "selected_reasons": sorted(selected_reasons),
                    "fusion_reasons": sorted(fusion_reasons),
                    "initial_selected_refinement": selected_mode,
                }
        return None, "", {}

    if selection_policy == "bottle_small_chip_sam_gate":
        polar = valid_candidates.get("polar_rim_residual")
        selected = valid_candidates.get(selected_mode)
        if polar is None or selected is None or selected_mode not in {"sam2_heatmap", "sam_prompt_regularized"}:
            return None, "", {}
        polar_qc = polar.get("qc", {})
        selected_qc = selected.get("qc", {})
        polar_box_fraction = float(polar_qc.get("mask_to_qwen_box_fraction", 1.0))
        selected_box_fraction = float(selected_qc.get("mask_to_qwen_box_fraction", 0.0))
        polar_components = int(polar_qc.get("component_count", 999))
        if (
            0.020 <= polar_box_fraction <= 0.045
            and selected_box_fraction >= 0.10
            and polar_components <= 3
            and str(polar_qc.get("status", "pass")) == "pass"
        ):
            return "polar_rim_residual", "compact_polar_rim_chip_over_broad_sam", {
                "polar_box_fraction": polar_box_fraction,
                "selected_box_fraction": selected_box_fraction,
                "polar_components": polar_components,
                "initial_selected_refinement": selected_mode,
            }
        return None, "", {}

    if selection_policy == "zipper_tooth_chain_prefer_patchcore_normal_residual":
        region = policy.get("qwen_region_xyxy")
        image_size = policy.get("image_size")
        if not (isinstance(region, (list, tuple)) and len(region) == 4 and isinstance(image_size, (list, tuple)) and len(image_size) == 2):
            return None, "", {}
        left, top, right, bottom = [int(value) for value in region]
        width, height = [max(1, int(value)) for value in image_size]
        region_height_fraction = (bottom - top) / max(1, height)
        touches_y_border = top <= 0 or bottom >= height

        if selected_mode == "patchcore_guided":
            selected = valid_candidates.get("patchcore_guided")
            agreement = valid_candidates.get("zipper_tooth_agreement")
            if agreement is not None and selected is not None:
                selected_qc = selected.get("qc", {})
                agreement_qc = agreement.get("qc", {})
                selected_area = max(1, int(selected_qc.get("mask_area", 0)))
                agreement_area = max(1, int(agreement_qc.get("mask_area", 0)))
                area_ratio = agreement_area / selected_area
                if (
                    region_height_fraction >= 0.65
                    and 0.65 <= area_ratio <= 0.92
                    and str(agreement_qc.get("status", "pass")) == "pass"
                    and int(agreement_qc.get("component_count", 999)) <= int(selected_qc.get("component_count", 999))
                ):
                    return "zipper_tooth_agreement", "long_tooth_region_patchcore_musc_agreement", {
                        "region_height_fraction": float(region_height_fraction),
                        "selected_area": selected_area,
                        "agreement_area": agreement_area,
                        "area_ratio": float(area_ratio),
                        "initial_selected_refinement": selected_mode,
                    }
            ensemble = valid_candidates.get("ensemble_consensus")
            if ensemble is not None and selected is not None:
                selected_qc = selected.get("qc", {})
                ensemble_qc = ensemble.get("qc", {})
                selected_area = max(1, int(selected_qc.get("mask_area", 0)))
                ensemble_area = max(1, int(ensemble_qc.get("mask_area", 0)))
                area_ratio = ensemble_area / selected_area
                if (
                    region_height_fraction <= 0.45
                    and int(selected_qc.get("component_count", 999)) <= 2
                    and int(ensemble_qc.get("component_count", 999)) <= 2
                    and 0.85 <= area_ratio <= 1.20
                    and str(ensemble_qc.get("status", "pass")) == "pass"
                ):
                    return "ensemble_consensus", "local_tooth_consensus_matches_patchcore_extent", {
                        "region_height_fraction": float(region_height_fraction),
                        "selected_area": selected_area,
                        "ensemble_area": ensemble_area,
                        "area_ratio": float(area_ratio),
                        "initial_selected_refinement": selected_mode,
                    }

            normal_fusion = valid_candidates.get("normal_residual_fusion")
            if normal_fusion is not None and selected is not None and touches_y_border:
                selected_qc = selected.get("qc", {})
                normal_qc = normal_fusion.get("qc", {})
                selected_area = max(1, int(selected_qc.get("mask_area", 0)))
                normal_area = max(1, int(normal_qc.get("mask_area", 0)))
                normal_box_fraction = float(normal_qc.get("mask_to_qwen_box_fraction", 1.0))
                reasons = {str(reason) for reason in normal_qc.get("reasons", [])}
                if (
                    normal_area >= int(selected_area * 1.10)
                    and normal_box_fraction <= 0.14
                    and "touches_image_border" in reasons
                ):
                    return "normal_residual_fusion", "edge_split_teeth_preserve_normal_residual_recall", {
                        "touches_y_border": bool(touches_y_border),
                        "selected_area": selected_area,
                        "normal_residual_fusion_area": normal_area,
                        "normal_residual_fusion_box_fraction": normal_box_fraction,
                        "normal_residual_fusion_reasons": sorted(reasons),
                        "initial_selected_refinement": selected_mode,
                    }
        return None, "", {}

    if selection_policy == "zipper_fabric_border_prefer_sam_patchcore_border_support":
        if selected_mode == "sam2_heatmap":
            selected = valid_candidates.get("sam2_heatmap")
            ensemble = valid_candidates.get("ensemble_consensus")
            if selected is not None and ensemble is not None:
                selected_qc = selected.get("qc", {})
                ensemble_qc = ensemble.get("qc", {})
                selected_area = max(1, int(selected_qc.get("mask_area", 0)))
                ensemble_area = max(1, int(ensemble_qc.get("mask_area", 0)))
                selected_reasons = {str(reason) for reason in selected_qc.get("reasons", [])}
                area_ratio = ensemble_area / selected_area
                if (
                    "touches_image_border" in selected_reasons
                    and 0.65 <= area_ratio <= 0.92
                    and float(ensemble_qc.get("mask_to_qwen_box_fraction", 1.0)) <= 0.30
                    and int(ensemble_qc.get("component_count", 999)) <= int(selected_qc.get("component_count", 999))
                ):
                    return "ensemble_consensus", "border_touching_sam_prefers_compact_consensus", {
                        "selected_area": selected_area,
                        "ensemble_area": ensemble_area,
                        "area_ratio": float(area_ratio),
                        "sam_reasons": sorted(selected_reasons),
                        "initial_selected_refinement": selected_mode,
                    }
        return None, "", {}

    return None, "", {}


def _auto_candidate_modes(auto: dict[str, Any]) -> list[str]:
    raw = auto.get("auto_candidate_modes", DEFAULT_AUTO_CANDIDATE_MODES)
    if isinstance(raw, str):
        modes = [mode.strip() for mode in raw.split(",") if mode.strip()]
    else:
        modes = [str(mode).strip() for mode in raw if str(mode).strip()]
    if not modes:
        raise ValueError("auto_masks.auto_candidate_modes must contain at least one refinement mode")
    return modes


def _candidate_mode_applicable(
    mode: str,
    *,
    category: str,
    defect_type: str,
    description: str = "",
) -> bool:
    structure_profile = _structure_profile(category, defect_type, description)
    applicability = specialist_applicability(
        mode,
        structure_profile=structure_profile,
        attributes=infer_structure_attributes(defect_type, description),
    )
    if applicability is not None:
        return applicability
    return True


def _structure_attributes(defect_type: str, description: str = "") -> dict[str, str]:
    return infer_structure_attributes(defect_type, description)


def _candidate_is_valid(candidate: dict[str, Any]) -> bool:
    qc = candidate.get("qc", {})
    return (
        int(qc.get("mask_area", 0)) > 0
        and str(qc.get("status", "pass")) != "reject"
        and float(candidate.get("score", -1.0)) >= 0.0
    )


def _structure_profile(category: str, defect_type: str, description: str = "") -> str:
    return infer_structure_profile(category, defect_type, description)


def _repeated_chain_correction_gate(
    candidates: dict[str, dict[str, Any]],
    *,
    region: tuple[int, int, int, int],
    localization_status: str,
) -> dict[str, Any]:
    baseline_modes = ("patchcore_guided", "normal_residual_fusion", "nearest_normal_residual")
    available = [
        (mode, candidates[mode])
        for mode in baseline_modes
        if mode in candidates and _candidate_is_valid(candidates[mode])
    ]
    if not available:
        return {
            "repeated_chain_gate_enabled": False,
            "repeated_chain_gate_reason": "missing_baseline_candidate",
        }
    baseline_mode, baseline = max(available, key=lambda item: float(item[1].get("score", -1.0)))
    path_value = baseline.get("refined_mask_path")
    if not path_value or not Path(str(path_value)).exists():
        return {
            "repeated_chain_gate_enabled": False,
            "repeated_chain_gate_reason": "missing_baseline_mask",
            "repeated_chain_gate_baseline_mode": baseline_mode,
        }

    arr = np.asarray(Image.open(str(path_value)).convert("L"), dtype=np.uint8) > 0
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return {
            "repeated_chain_gate_enabled": False,
            "repeated_chain_gate_reason": "empty_baseline_mask",
            "repeated_chain_gate_baseline_mode": baseline_mode,
        }
    coordinates = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    covariance = centered.T @ centered / max(1, len(coordinates))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    axial_coordinates = centered @ axis
    perpendicular_coordinates = centered @ perpendicular
    axial_span = float(axial_coordinates.max() - axial_coordinates.min() + 1.0)
    perpendicular_span = float(perpendicular_coordinates.max() - perpendicular_coordinates.min() + 1.0)

    left, top, right, bottom = region
    corners = np.asarray(
        [[left, top], [right, top], [left, bottom], [right, bottom]],
        dtype=np.float64,
    )
    corner_axis = corners @ axis
    region_axis_span = float(corner_axis.max() - corner_axis.min())
    axis_span_fraction = axial_span / max(1.0, region_axis_span)
    oriented_fill = len(xs) / max(1.0, axial_span * perpendicular_span)
    region_scale_fraction = region_axis_span / max(1.0, float(max(arr.shape)))
    orientation_degrees = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    enabled = (
        str(localization_status) == "valid"
        and region_scale_fraction >= 0.32
        and axis_span_fraction >= 0.88
        and oriented_fill >= 0.25
    )
    if str(localization_status) != "valid":
        reason = "localization_not_valid"
    elif region_scale_fraction < 0.32:
        reason = "region_already_local"
    elif axis_span_fraction < 0.88:
        reason = "baseline_not_chain_spanning"
    elif oriented_fill < 0.25:
        reason = "baseline_sparse_or_multi_local"
    else:
        reason = "dense_full_chain_overcoverage"
    return {
        "repeated_chain_gate_enabled": bool(enabled),
        "repeated_chain_gate_reason": reason,
        "repeated_chain_gate_baseline_mode": baseline_mode,
        "repeated_chain_gate_orientation_degrees": orientation_degrees,
        "repeated_chain_gate_axis_span_fraction": float(axis_span_fraction),
        "repeated_chain_gate_vertical_span_fraction": float(axis_span_fraction),
        "repeated_chain_gate_oriented_fill": float(oriented_fill),
        "repeated_chain_gate_bbox_fill": float(oriented_fill),
        "repeated_chain_gate_region_scale_fraction": float(region_scale_fraction),
        "repeated_chain_gate_region_height": int(bottom - top),
    }


def _foundation_selector_bonus_scale(auto: dict[str, Any] | None) -> float:
    if not isinstance(auto, dict):
        return 1.0
    if "foundation_selector_bonus_scale" in auto:
        return float(auto.get("foundation_selector_bonus_scale", 1.0))
    if bool(auto.get("foundation_selector_diagnostic_only", False)):
        return 0.0
    return 1.0


def _scaled_foundation_bonus(value: float, scale: float) -> float:
    return round(float(value) * float(scale), 4)


def _selection_policy(
    *,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    description: str,
    candidates: dict[str, dict[str, Any]],
    localization_status: str = "unknown",
    auto: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = f"{category} {defect_type} {description}".lower().replace("-", "_")
    structure_profile = _structure_profile(category, defect_type, description)
    foundation_scale = _foundation_selector_bonus_scale(auto)
    if defect_type != "scratch":
        if category == "bottle" and any(
            term in text for term in ("contamination", "stain", "foreign", "spot", "surface")
        ):
            return {
                "scratch_morphology_class": "not_scratch",
                "structure_profile": structure_profile,
                "selection_policy": "bottle_contamination_prefer_texture_or_sam",
                "candidate_rejections": {},
                "mode_score_adjustments": {
                    "fft_texture_suppression": 0.70,
                    "sam2_heatmap": 0.65,
                    "sam_prompt_regularized": 0.25,
                    "foundation_anomaly_field": _scaled_foundation_bonus(0.22, foundation_scale),
                    "bottle_contamination_recall_fusion": 0.76,
                    "musc_mutual_score": 0.15,
                    "scuff_cluster": 0.20,
                    "soft_patch": 0.15,
                    "ensemble_consensus": -0.40,
                    "scratch_band_clean": -0.30,
                    "procedural": -0.30,
                },
                "ensemble_bonus": -0.40,
                "score_generated_ensemble_independently": True,
                "foundation_selector_bonus_scale": foundation_scale,
                "qwen_region_xyxy": region,
                "image_size": image_size,
            }
        if category == "bottle" and defect_type == "broken_small":
            return {
                "scratch_morphology_class": "not_scratch",
                "structure_profile": structure_profile,
                "selection_policy": "bottle_small_chip_sam_gate",
                "candidate_rejections": {},
                "mode_score_adjustments": {
                    "polar_rim_residual": 0.30,
                    "sam2_heatmap": 0.95,
                    "sam_prompt_regularized": 0.65,
                    "foundation_anomaly_field": _scaled_foundation_bonus(0.18, foundation_scale),
                    "musc_mutual_score": 0.35,
                    "nearest_normal_residual": 0.20,
                    "patchcore_guided": 0.15,
                    "normal_residual_fusion": 0.10,
                    "ensemble_consensus": -0.10,
                },
                "ensemble_bonus": -0.10,
                "score_generated_ensemble_independently": True,
                "foundation_selector_bonus_scale": foundation_scale,
                "qwen_region_xyxy": region,
                "image_size": image_size,
            }
        if category == "bottle" and any(
            term in text
            for term in ("broken", "chip", "chipped", "missing", "rim", "glass", "broken_large", "broken_small")
        ):
            return {
                "scratch_morphology_class": "not_scratch",
                "structure_profile": structure_profile,
                "selection_policy": "bottle_rim_chip_prefer_normal_residual_sam_patchcore",
                "candidate_rejections": {},
                "mode_score_adjustments": {
                    "nearest_normal_residual": 0.35,
                    "sam2_heatmap": 0.35,
                    "sam_prompt_regularized": 0.25,
                    "foundation_anomaly_field": _scaled_foundation_bonus(0.22, foundation_scale),
                    "musc_mutual_score": 0.20,
                    "patchcore_guided": 0.20,
                    "normal_residual_fusion": 0.15,
                    "ensemble_consensus": -0.05,
                },
                "ensemble_bonus": -0.05,
                "score_generated_ensemble_independently": True,
                "foundation_selector_bonus_scale": foundation_scale,
                "qwen_region_xyxy": region,
                "image_size": image_size,
            }
        if category == "zipper" and defect_type in {"broken_teeth", "split_teeth"}:
            chain_gate = _repeated_chain_correction_gate(
                candidates,
                region=region,
                localization_status=localization_status,
            )
            return {
                "scratch_morphology_class": "not_scratch",
                "structure_profile": structure_profile,
                "selection_policy": "zipper_tooth_chain_prefer_patchcore_normal_residual",
                "candidate_rejections": {},
                "mode_score_adjustments": {
                    "repeated_chain_refiner": 0.48 if chain_gate["repeated_chain_gate_enabled"] else -0.35,
                    "foundation_anomaly_field": _scaled_foundation_bonus(0.26, foundation_scale),
                    "zipper_tooth_agreement": 0.12,
                    "patchcore_guided": 0.45,
                    "musc_mutual_score": 0.25,
                    "normal_residual_fusion": 0.30,
                    "nearest_normal_residual": 0.20,
                    "sam_prompt_regularized": 0.18,
                    "scratch_band_clean": 0.12,
                    "ensemble_consensus": -0.05,
                    "fft_texture_suppression": -0.20,
                },
                "ensemble_bonus": -0.05,
                "score_generated_ensemble_independently": True,
                "foundation_selector_bonus_scale": foundation_scale,
                "qwen_region_xyxy": region,
                "image_size": image_size,
                "localization_status": localization_status,
                **chain_gate,
            }
        if category == "zipper" and defect_type == "fabric_border":
            left, top, right, bottom = region
            region_area_fraction = ((right - left) * (bottom - top)) / max(1, image_size[0] * image_size[1])
            full_region = region_area_fraction >= 0.50
            return {
                "scratch_morphology_class": "not_scratch",
                "structure_profile": structure_profile,
                "selection_policy": "zipper_fabric_border_prefer_sam_patchcore_border_support",
                "candidate_rejections": {},
                "mode_score_adjustments": {
                    "sam2_heatmap": 0.65,
                    "sam_prompt_regularized": 0.35,
                    "zipper_fabric_border_layout": 0.80 if full_region else 0.10,
                    "foundation_anomaly_field": _scaled_foundation_bonus(0.18, foundation_scale),
                    "musc_mutual_score": 0.05,
                    "patchcore_guided": 0.10,
                    "nearest_normal_residual": 0.08,
                    "normal_residual_fusion": -0.15 if full_region else 0.0,
                    "ensemble_consensus": 0.05,
                },
                "ensemble_bonus": 0.05,
                "score_generated_ensemble_independently": True,
                "foundation_selector_bonus_scale": foundation_scale,
                "qwen_region_xyxy": region,
                "image_size": image_size,
                "qwen_region_area_fraction": float(region_area_fraction),
            }
        return {
            "scratch_morphology_class": "not_scratch",
            "structure_profile": structure_profile,
            "selection_policy": "score_only",
            "candidate_rejections": {},
            "mode_score_adjustments": {},
            "ensemble_bonus": 0.20,
        }
    left, top, right, bottom = region
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    box_aspect = max(box_width / box_height, box_height / box_width)
    box_area_fraction = (box_width * box_height) / max(1, image_size[0] * image_size[1])
    patch_hint = _description_implies_patch(description) or _description_implies_multiple(description)
    linear = candidates.get("linear")
    normal = candidates.get("normal_anomaly")
    linear_fraction = float(linear.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) if linear else 0.0
    normal_components = int(normal.get("qc", {}).get("component_count", 0)) if normal else 0

    if box_aspect >= 2.2 or (normal_components >= 3 and not patch_hint):
        morphology = "scratch_band"
        selection = "prefer_scratch_band_clean_normal_multi; penalize_detached_chunks"
        rejections = {}
    elif patch_hint:
        morphology = "multi_scuff"
        selection = "prefer_multi_scuff_fusion_scuff_cluster_soft_patch; block_undercovered_linear"
        rejections = {"linear": "scuffed or multi-scratch box needs cluster support, not one tidy line"}
    else:
        morphology = "single_stroke"
        selection = "allow_line_like_candidates"
        rejections = {}

    if morphology == "multi_scuff" and linear_fraction >= 0.08:
        rejections.pop("linear", None)
    return {
        "scratch_morphology_class": morphology,
        "structure_profile": structure_profile,
        "selection_policy": selection,
        "candidate_rejections": rejections,
        "mode_score_adjustments": {},
        "ensemble_bonus": -0.03 if morphology == "multi_scuff" else 0.20,
        "qwen_box_aspect": float(box_aspect),
        "qwen_box_area_fraction": float(box_area_fraction),
        "linear_mask_to_qwen_box_fraction": float(linear_fraction),
        "normal_component_count": int(normal_components),
    }


def _policy_adjusted_score(mode: str, candidate: dict[str, Any], policy: dict[str, Any]) -> float:
    score = float(candidate.get("score", -1.0))
    morphology = str(policy.get("scratch_morphology_class", "score_only"))
    selection_policy = str(policy.get("selection_policy", ""))
    if mode in policy.get("candidate_rejections", {}):
        return -1.0
    adjustments = policy.get("mode_score_adjustments", {})
    if isinstance(adjustments, dict):
        score += float(adjustments.get(mode, 0.0))
    qc = candidate.get("qc", {})
    qc_reasons = {str(reason) for reason in qc.get("reasons", [])}
    mask_fraction = float(qc.get("mask_to_qwen_box_fraction", 0.0))
    image_fraction = float(qc.get("mask_area_fraction", 0.0))
    component_count = int(qc.get("component_count", 0))
    if selection_policy == "bottle_small_chip_sam_gate" and mode in {"sam2_heatmap", "sam_prompt_regularized"}:
        if mask_fraction > 0.45 or image_fraction > 0.25 or "large_fraction_of_qwen_box" in qc_reasons:
            score -= 0.80
        elif component_count > 24:
            score -= 0.20
    elif mode == "foundation_anomaly_field":
        if mask_fraction > 0.32 or image_fraction > 0.10:
            score -= 0.30
        if component_count > 32:
            score -= 0.18
        contrast = float(candidate.get("parameters", {}).get("foundation_region_contrast", 0.0))
        if contrast >= 0.25:
            score += min(0.10, contrast * 0.18)
    elif selection_policy == "bottle_small_chip_sam_gate" and mode == "polar_rim_residual":
        params = candidate.get("parameters", {})
        sector_degrees = float(params.get("polar_rim_sector_degrees", 360.0))
        if mask_fraction > 0.24 or image_fraction > 0.10:
            score -= 0.75
        if sector_degrees > 80.0:
            score -= 0.30
        elif 8.0 <= sector_degrees <= 65.0:
            score += 0.10
    elif selection_policy == "zipper_fabric_border_prefer_sam_patchcore_border_support":
        if mode in {"sam2_heatmap", "sam_prompt_regularized"}:
            region = policy.get("qwen_region_xyxy")
            image_size = policy.get("image_size")
            region_area_fraction = 0.0
            if isinstance(region, (list, tuple)) and len(region) == 4 and isinstance(image_size, (list, tuple)) and len(image_size) == 2:
                left, top, right, bottom = [int(value) for value in region]
                width, height = [max(1, int(value)) for value in image_size]
                region_area_fraction = ((right - left) * (bottom - top)) / max(1, width * height)
            if "tiny_fraction_of_qwen_box" in qc_reasons:
                score -= 0.35
            if mask_fraction < 0.004 and region_area_fraction > 0.50:
                score -= 0.55
            if 0.16 <= mask_fraction <= 0.36:
                score += 0.18
        elif mode == "patchcore_guided" and ("large_image_area" in qc_reasons or "touches_image_border" in qc_reasons):
            score -= 0.15
        elif mode == "musc_mutual_score" and mask_fraction < 0.16:
            score -= 0.40
    elif selection_policy == "zipper_tooth_chain_prefer_patchcore_normal_residual" and mode == "repeated_chain_refiner":
        params = candidate.get("parameters", {})
        explanation = float(params.get("repeated_chain_explanation_score", 0.0))
        structure = float(params.get("repeated_chain_structure_score", 0.0))
        vertical_span = float(params.get("repeated_chain_vertical_span_fraction", 1.0))
        score += 0.18 * explanation + 0.18 * structure
        if vertical_span > 0.72:
            score -= 0.35
        elif vertical_span <= 0.45:
            score += 0.08
    if morphology == "multi_scuff":
        if mode == "foundation_anomaly_field":
            score += 0.30
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) > 0.22:
                score -= 0.12
            if 1 <= int(candidate.get("qc", {}).get("component_count", 0)) <= 18:
                score += 0.05
        if mode == "nearest_normal_residual":
            score += 0.34
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) > 0.20:
                score -= 0.18
            if int(candidate.get("qc", {}).get("component_count", 0)) > 12:
                score -= 0.12
        elif mode == "patchcore_guided":
            score += 0.42
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) > 0.18:
                score -= 0.16
            if int(candidate.get("qc", {}).get("component_count", 0)) > 18:
                score -= 0.12
        elif mode == "normal_residual_fusion":
            score += 0.46
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) > 0.20:
                score -= 0.16
            if int(candidate.get("qc", {}).get("component_count", 0)) > 36:
                score -= 0.18
        elif mode == "multi_scuff_fusion":
            score += 0.44
        elif mode == "support_constrained_fusion":
            score += 0.40
            params = candidate.get("parameters", {})
            if bool(params.get("support_constrained_reverted", False)):
                score -= 0.10
            if int(params.get("support_constrained_rejected_grain", 0)) > 0:
                score += 0.04
        elif mode == "scuff_cluster":
            score += 0.34
        elif mode == "fft_texture_suppression":
            score += 0.32
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) > 0.22:
                score -= 0.45
        elif mode == "structure_tensor_ridge":
            score -= 0.10
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) < 0.05:
                score -= 0.16
        elif mode == "soft_patch":
            score += 0.22
        elif mode == "normal_anomaly":
            score += 0.12
        elif mode in {"linear", "multi_linear"}:
            score -= 0.35
    elif morphology == "scratch_band":
        if mode == "foundation_anomaly_field":
            score += 0.18
            if int(candidate.get("qc", {}).get("component_count", 0)) <= 12:
                score += 0.04
            if float(candidate.get("qc", {}).get("mask_to_qwen_box_fraction", 0.0)) > 0.20:
                score -= 0.10
        if mode == "scratch_band_clean":
            score += 0.28
        elif mode == "structure_tensor_ridge":
            score += 0.24
        elif mode == "normal_residual_fusion":
            score += 0.10
        elif mode == "nearest_normal_residual":
            score += 0.10
        elif mode == "patchcore_guided":
            score += 0.06
        elif mode == "normal_anomaly":
            score += 0.08
        elif mode == "multi_linear":
            score += 0.04
        qc = candidate.get("qc", {})
        if int(qc.get("component_count", 0)) > 8:
            score -= 0.18
    elif morphology == "single_stroke":
        if mode in {
            "linear",
            "multi_linear",
            "structure_tensor_ridge",
            "normal_anomaly",
            "nearest_normal_residual",
            "patchcore_guided",
            "normal_residual_fusion",
        }:
            score += 0.08
    if mode == "procedural":
        score -= 0.30
    return round(float(max(-1.0, min(1.50, score))), 4)



def _make_mask_candidate(
    *,
    mode: str,
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    artifact_stem: str,
    seed: int,
    auto: dict[str, Any],
    description: str,
    normal_paths: list[Path],
    vlm_point: tuple[int, int] | None = None,
    feature_cache: PatchFeatureCache | None = None,
) -> dict[str, Any]:
    if mode == "delta_deno":
        artifacts = write_delta_deno_bbox_masks(
            image,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "pixel":
        artifacts = write_pixel_refined_bbox_masks(
            image,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            clip_to_surface=bool(auto.get("clip_to_surface", False)),
            text_hint=description,
            percentile=float(auto.get("pixel_refine_percentile", 93.0)),
        )
    elif mode == "sam2_heatmap":
        normal_path = _nearest_normal(image, normal_paths, image.size)
        artifacts = write_sam_heatmap_bbox_masks(
            image,
            normal_path,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "sam_prompt_regularized":
        artifacts = write_sam_prompt_regularized_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
            feature_cache=feature_cache,
        )
    elif mode == "soft_patch":
        normal_path = _nearest_normal(image, normal_paths, image.size)
        artifacts = write_soft_patch_bbox_masks(
            image,
            normal_path,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "normal_anomaly":
        artifacts = write_normal_anomaly_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "foundation_anomaly_field":
        artifacts = write_foundation_anomaly_field_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
            feature_cache=feature_cache,
        )
    elif mode == "nearest_normal_residual":
        artifacts = write_nearest_normal_residual_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "patchcore_guided":
        artifacts = write_patchcore_guided_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
            feature_cache=feature_cache,
        )
    elif mode == "musc_mutual_score":
        artifacts = write_musc_mutual_score_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
            feature_cache=feature_cache,
        )
    elif mode == "scuff_cluster":
        artifacts = write_scuff_cluster_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "multi_scuff_fusion":
        artifacts = write_multi_scuff_fusion_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "support_constrained_fusion":
        artifacts = write_support_constrained_fusion_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "normal_residual_fusion":
        artifacts = write_normal_residual_fusion_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "repeated_chain_refiner":
        artifacts = write_repeated_chain_refiner_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
            feature_cache=feature_cache,
        )
    elif mode == "polar_rim_residual":
        artifacts = write_polar_rim_residual_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "zipper_fabric_border_layout":
        artifacts = write_zipper_fabric_border_layout_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
            feature_cache=feature_cache,
        )
    elif mode == "fft_texture_suppression":
        artifacts = write_fft_texture_suppression_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "scratch_band_clean":
        artifacts = write_scratch_band_clean_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "structure_tensor_ridge":
        artifacts = write_structure_tensor_ridge_bbox_masks(
            image,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "dinov2_memory":
        artifacts = write_dinov2_memory_bbox_masks(
            image,
            normal_paths,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            text_hint=description,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "dinov2_fusion":
        try:
            artifacts = write_dinov2_fusion_bbox_masks(
                image,
                normal_paths,
                category,
                defect_type,
                region,
                output_dir,
                artifact_stem,
                auto=auto,
                text_hint=description,
                padding=int(auto.get("shrink_padding_px", 6)),
            )
        except Exception:
            if not bool(auto.get("dinov2_allow_fallback", True)):
                raise
            artifacts = write_dinov2_memory_bbox_masks(
                image,
                normal_paths,
                category,
                defect_type,
                region,
                output_dir,
                artifact_stem,
                auto={**auto, "dinov2_allow_fallback": True},
                text_hint=description,
                padding=int(auto.get("shrink_padding_px", 6)),
            )
    elif mode == "vlm_point_sam2":
        if vlm_point is None:
            raise ValueError("vlm_point must be provided for vlm_point_sam2 mode")
        artifacts = write_vlm_point_sam2_bbox_masks(
            image,
            vlm_point,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            auto=auto,
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "residual":
        normal_path = _nearest_normal(image, normal_paths, image.size)
        artifacts = write_residual_refined_bbox_masks(
            image,
            normal_path,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            clip_to_surface=bool(auto.get("clip_to_surface", False)),
            text_hint=description,
            percentile=float(auto.get("residual_refine_percentile", 95.0)),
            min_component_area=int(auto.get("min_component_area", 12)),
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "linear":
        artifacts = write_linear_refined_bbox_masks(
            image,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            clip_to_surface=bool(auto.get("clip_to_surface", False)),
            text_hint=description,
            percentile=float(auto.get("pixel_refine_percentile", 93.0)),
            padding=int(auto.get("shrink_padding_px", 6)),
        )
    elif mode == "multi_linear":
        artifacts = write_multi_linear_refined_bbox_masks(
            image,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            clip_to_surface=bool(auto.get("clip_to_surface", False)),
            text_hint=description,
            percentile=float(auto.get("pixel_refine_percentile", 93.0)),
            padding=int(auto.get("shrink_padding_px", 6)),
            max_lines=int(auto.get("multi_linear_max_lines", 6)),
        )
    elif mode == "procedural":
        artifacts = write_refined_bbox_masks(
            image,
            category,
            defect_type,
            region,
            output_dir,
            artifact_stem,
            seed,
            clip_to_surface=bool(auto.get("clip_to_surface", False)),
        )
    else:
        raise ValueError(f"Unsupported auto mask refinement mode: {mode}")
    if "qc" not in artifacts:
        refined_for_qc = Image.open(artifacts["refined_mask_path"]).convert("L")
        artifacts["qc"] = _qc_mask(
            refined_for_qc,
            region,
            _padded_bbox(refined_for_qc, image.size, 0) or region,
            image.size,
            defect_type,
        )
    artifacts["score"] = _score_candidate(
        Image.open(artifacts["refined_mask_path"]).convert("L"),
        image,
        artifacts["qc"],
        mode,
        defect_type,
        description,
    )
    artifacts["selected_refinement"] = mode
    return artifacts


def _copy_candidate_as_final(candidate: dict[str, Any], output_dir: Path, stem: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "box_mask_path": output_dir / f"{stem}_box.png",
        "refined_mask_path": output_dir / f"{stem}_refined.png",
        "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
    }
    for key, destination in paths.items():
        Image.open(candidate[key]).save(destination)
    return {
        **candidate,
        "box_mask_path": str(paths["box_mask_path"]),
        "refined_mask_path": str(paths["refined_mask_path"]),
        "inpaint_mask_path": str(paths["inpaint_mask_path"]),
    }


def _invoke_vlm_critic(
    image: Image.Image,
    region: tuple[int, int, int, int],
    description: str,
    top_modes: list[str],
    candidates: dict[str, dict[str, Any]],
    extractor: Any,
) -> str | None:
    if extractor is None:
        return None
        
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_names = ["Red", "Green", "Blue"]
    
    left, top, right, bottom = region
    pad = 30
    cx1 = max(0, left - pad)
    cy1 = max(0, top - pad)
    cx2 = min(image.width, right + pad)
    cy2 = min(image.height, bottom + pad)
    crop = image.crop((cx1, cy1, cx2, cy2)).convert("RGBA")
    
    composite = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    for i, mode in enumerate(top_modes):
        mask = Image.open(candidates[mode]["refined_mask_path"]).convert("L")
        mask_crop = mask.crop((cx1, cy1, cx2, cy2))
        
        edges = mask_crop.filter(ImageFilter.FIND_EDGES)
        edges = edges.point(lambda p: 255 if p > 0 else 0)
        
        color_layer = Image.new("RGBA", crop.size, (*colors[i], 255))
        composite = Image.composite(color_layer, composite, edges)
        
    final_img = Image.alpha_composite(crop, composite).convert("RGB")
    
    prompt = (
        f"You are evaluating anomaly masks for a defect described as: '{description}'. "
        f"The image shows {len(top_modes)} candidate mask boundaries in {', '.join(color_names[:len(top_modes)])}. "
        "Which color mask best covers the actual defect without spilling into healthy areas? "
        f"Answer ONLY with the color name: {' or '.join(color_names[:len(top_modes)])}."
    )
    
    try:
        result = extractor.analyze(final_img, prompt)
        text = result.get("text", "").strip().lower()
        for i, color_name in enumerate(color_names[:len(top_modes)]):
            if color_name.lower() in text:
                return top_modes[i]
    except Exception:
        pass
        
    return None


def _attach_mask_variants(
    *,
    image: Image.Image,
    category: str = "",
    defect_type: str = "",
    region: tuple[int, int, int, int],
    output_dir: Path,
    artifact_stem: str,
    artifacts: dict[str, Any],
    auto: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Image.open(artifacts["refined_mask_path"]).convert("L").point(lambda value: 255 if value > 0 else 0)
    morphology = str(artifacts.get("scratch_morphology_class", ""))
    selected = str(artifacts.get("selected_refinement", ""))
    if morphology == "multi_scuff" or selected == "scuff_cluster":
        tight = _scuff_eval_tight_mask(
            base,
            region,
            radius=int(auto.get("scuff_eval_tight_radius", 1)),
            max_components=int(auto.get("scuff_eval_tight_max_components", 10)),
            max_box_fraction=float(auto.get("scuff_eval_tight_max_box_fraction", 0.09)),
        )
    else:
        tight = _variant_mask(base, radius=int(auto.get("eval_tight_radius", 0)), blur=0.0)
    tight, recall_params = _recall_expand_eval_mask(
        tight,
        artifacts=artifacts,
        category=category,
        defect_type=defect_type,
        region=region,
        image_size=image.size,
        auto=auto,
    )
    tight, zipper_structure_params = _zipper_structure_cleanup_eval_mask(
        tight,
        artifacts=artifacts,
        category=category,
        defect_type=defect_type,
        region=region,
        image_size=image.size,
        auto=auto,
    )
    tight, tight_params = _precision_tight_eval_mask(
        tight,
        artifacts=artifacts,
        category=category,
        defect_type=defect_type,
        region=region,
        image_size=image.size,
        auto=auto,
    )
        
    import math
    mask_area = np.count_nonzero(np.asarray(base))
    scale_factor = math.sqrt(max(1, mask_area) / 1000.0)
    medium_radius = max(2, min(30, int(int(auto.get("training_medium_radius", 4)) * scale_factor)))
    wide_radius = max(4, min(60, int(int(auto.get("training_wide_radius", 9)) * scale_factor)))
    
    medium = _anisotropic_variant_mask(base, radius=medium_radius, blur=0.0)
    wide = _anisotropic_variant_mask(base, radius=wide_radius, blur=0.0)
    inpaint, inpaint_params = _adaptive_inpaint_soft_mask(base, image, morphology, selected, auto)
    uncertainty_variants, uncertainty_params = _uncertainty_mask_variants(
        base,
        artifacts.get("candidate_refined_paths", {}),
        artifacts.get("candidate_heatmap_paths", {}),
        morphology,
        region,
        image.size,
        auto,
    )
    generation_core = base
    eval_mvtec, eval_mvtec_params = _mvtec_eval_mask_variant(
        eval_tight=tight,
        generation_core=generation_core,
        artifacts=artifacts,
        category=category,
        defect_type=defect_type,
        region=region,
        image_size=image.size,
        auto=auto,
    )
    variants = {
        "generation_core": generation_core,
        "eval_tight": tight,
        "eval_mvtec": eval_mvtec,
        "training_medium": medium,
        "training_wide": wide,
        "training_soft": uncertainty_variants["training_soft"],
        "positive_core": uncertainty_variants["positive_core"],
        "possible_region": uncertainty_variants["possible_region"],
        "uncertainty_map": uncertainty_variants["uncertainty_map"],
        "inpaint_soft": inpaint,
    }
    variant_paths: dict[str, str] = {}
    overlay_paths: dict[str, str] = {}
    for name, mask in variants.items():
        path = output_dir / f"{artifact_stem}_{name}.png"
        mask.save(path)
        variant_paths[name] = str(path)
        if bool(auto.get("write_variant_overlays", True)):
            overlay_path = output_dir / f"{artifact_stem}_{name}_overlay.png"
            write_mask_overlay(image, region, path, overlay_path)
            overlay_paths[name] = str(overlay_path)
    eval_variant = str(auto.get("eval_mask_variant", "eval_tight"))
    if eval_variant not in variant_paths:
        raise ValueError(f"Unsupported eval_mask_variant: {eval_variant}")
    eval_path = variant_paths[eval_variant]
    label_policy = _label_policy_for_variants(morphology, selected, auto)
    if label_policy["label_policy"] == "soft_mask_only":
        training_variant = str(auto.get("scuff_training_mask_variant", "training_soft"))
    else:
        training_variant = str(auto.get("training_mask_variant", "training_medium"))
    if training_variant not in variant_paths:
        raise ValueError(f"Unsupported training_mask_variant: {training_variant}")
    training_path = variant_paths[training_variant]
    return {
        **artifacts,
        "generation_core_mask_path": variant_paths["generation_core"],
        "benchmark_eval_mask_path": variant_paths["eval_mvtec"],
        "inpaint_mask_path": variant_paths["inpaint_soft"],
        "eval_mask_path": eval_path,
        "training_mask_path": training_path,
        "uncertainty_mask_path": variant_paths["uncertainty_map"],
        "mask_variant_paths": variant_paths,
        "mask_variant_overlay_paths": overlay_paths,
        "label_policy": label_policy,
        "parameters": {
            **artifacts["parameters"],
            "dual_mask_outputs": True,
            "eval_mask_variant": eval_variant,
            "training_mask_variant": training_variant,
            "label_policy": label_policy,
            "mask_variants": {
                "generation_core": {
                    "purpose": "visible defect core for synthetic generation/reference conditioning",
                },
                "eval_tight": {
                    "radius": int(auto.get("eval_tight_radius", 0)),
                    "purpose": "pseudo segmentation/evaluation mask",
                    **recall_params,
                    **zipper_structure_params,
                    **tight_params,
                },
                "eval_mvtec": {
                    "purpose": "benchmark-oriented broad pseudo-mask for MVTec-style evaluation",
                    **eval_mvtec_params,
                },
                "training_medium": {"radius": medium_radius, "purpose": "default adapter/inpaint training region"},
                "training_wide": {"radius": wide_radius, "purpose": "wide adapter/inpaint training region"},
                "training_soft": {
                    "purpose": "soft pseudo-label for fuzzy/scuffed defects and uncertainty-aware training",
                    **uncertainty_params,
                },
                "positive_core": {"purpose": "high-confidence pixels agreed on by candidate masks"},
                "possible_region": {"purpose": "broader possible defect support from candidate union/soft expansion"},
                "uncertainty_map": {"purpose": "uncertain pseudo-label boundary or candidate-disagreement region"},
                "inpaint_soft": {
                    **inpaint_params,
                    "purpose": "soft SD inpainting mask candidate",
                },
            },
        },
    }


def _label_policy_for_variants(morphology: str, selected_refinement: str, auto: dict[str, Any]) -> dict[str, Any]:
    soft_scuff_enabled = bool(auto.get("scuff_soft_label_training", True))
    if soft_scuff_enabled and morphology == "multi_scuff":
        return {
            "label_policy": "soft_mask_only",
            "quality_morphology": "multi_scuff",
            "hard_mask_usage": "pseudo_eval_only",
            "adapter_training_mask_variant": str(auto.get("scuff_training_mask_variant", "training_soft")),
            "generation_mask_variant": "inpaint_soft",
            "reason": "low-contrast scuffed/rubbed defects have fuzzy boundaries; use soft masks for training/generation",
            "selected_refinement": selected_refinement,
        }
    return {
        "label_policy": "hard_mask_ok",
        "quality_morphology": morphology or "unknown",
        "hard_mask_usage": "pseudo_eval_and_training",
        "adapter_training_mask_variant": str(auto.get("training_mask_variant", "training_medium")),
        "generation_mask_variant": "inpaint_soft",
        "reason": "line-like defect geometry is suitable for hard pseudo-label training",
        "selected_refinement": selected_refinement,
    }


def _mvtec_eval_mask_variant(
    *,
    eval_tight: Image.Image,
    generation_core: Image.Image,
    artifacts: dict[str, Any],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not bool(auto.get("eval_mvtec_enabled", False)):
        return eval_tight, {"eval_mvtec_enabled": False, "eval_mvtec_reason": "disabled"}

    selected = str(artifacts.get("selected_refinement", ""))
    structure_profile = str(
        artifacts.get("structure_profile")
        or _structure_profile(category, defect_type, str(artifacts.get("description", "")))
    )
    params: dict[str, Any] = {
        "eval_mvtec_enabled": True,
        "eval_mvtec_structure_profile": structure_profile,
        "eval_mvtec_source": "eval_tight",
        "eval_mvtec_reason": "default_tight",
    }

    if category == "bottle" and defect_type == "broken_small" and selected in {"sam_prompt_regularized", "sam2_heatmap"}:
        broad = _candidate_mask_for_variant(artifacts, "ensemble_consensus", image_size)
        if broad is not None:
            broad_arr = np.asarray(broad.convert("L"), dtype=np.uint8) > 0
            core_arr = np.asarray(eval_tight.convert("L"), dtype=np.uint8) > 0
            left, top, right, bottom = region
            fence = np.zeros(core_arr.shape, dtype=bool)
            fence[top:bottom, left:right] = True
            broad_arr &= fence
            core_arr &= fence
            broad_pixels = int(broad_arr.sum())
            core_pixels = int(core_arr.sum())
            overlap = int((broad_arr & core_arr).sum())
            overlap_ratio = overlap / max(1, min(broad_pixels, core_pixels))
            max_image_fraction = float(auto.get("bottle_small_eval_mvtec_max_image_fraction", 0.080))
            min_overlap = float(auto.get("bottle_small_eval_mvtec_min_core_overlap", 0.25))
            if (
                broad_pixels > core_pixels
                and broad_pixels <= int(image_size[0] * image_size[1] * max_image_fraction)
                and overlap_ratio >= min_overlap
            ):
                return Image.fromarray(broad_arr.astype(np.uint8) * 255, mode="L"), {
                    **params,
                    "eval_mvtec_source": "ensemble_consensus",
                    "eval_mvtec_reason": "bottle_small_chip_broad_consensus",
                    "eval_mvtec_core_pixels": core_pixels,
                    "eval_mvtec_broad_pixels": broad_pixels,
                    "eval_mvtec_core_overlap_ratio": float(overlap_ratio),
                }
            params.update(
                {
                    "eval_mvtec_reason": "bottle_small_consensus_rejected",
                    "eval_mvtec_core_pixels": core_pixels,
                    "eval_mvtec_broad_pixels": broad_pixels,
                    "eval_mvtec_core_overlap_ratio": float(overlap_ratio),
                }
            )

    if structure_profile == "ring_sector" and category == "bottle" and defect_type == "broken_large":
        expanded, expand_params = _ring_sector_eval_prior(
            eval_tight,
            artifacts=artifacts,
            region=region,
            image_size=image_size,
            auto=auto,
        )
        return expanded, {**params, **expand_params}

    if category == "zipper" and defect_type == "fabric_border":
        expanded, expand_params = _zipper_fabric_border_eval_mvtec_prior(
            eval_tight,
            artifacts=artifacts,
            image_size=image_size,
            auto=auto,
        )
        return expanded, {**params, **expand_params}

    return eval_tight, params


def _candidate_mask_for_variant(artifacts: dict[str, Any], mode: str, image_size: tuple[int, int]) -> Image.Image | None:
    path_value = artifacts.get("candidate_refined_paths", {}).get(mode)
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.exists():
        return None
    return Image.open(path).convert("L").resize(image_size, Image.Resampling.NEAREST).point(lambda value: 255 if value > 0 else 0)


def _zipper_fabric_border_eval_mvtec_prior(
    eval_tight: Image.Image,
    *,
    artifacts: dict[str, Any],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not bool(auto.get("zipper_fabric_eval_mvtec_vertical_expand_enabled", True)):
        return eval_tight, {"eval_mvtec_reason": "zipper_fabric_vertical_expand_disabled"}
    selected = str(artifacts.get("selected_refinement", ""))
    if selected not in {"sam2_heatmap", "ensemble_consensus"}:
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "zipper_fabric_selected_mode_not_expandable",
            "eval_mvtec_selected_refinement": selected,
        }

    qc = artifacts.get("qc", {})
    reasons = {str(reason) for reason in qc.get("reasons", [])}
    if selected == "ensemble_consensus" and "touches_image_border" not in reasons:
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "zipper_fabric_consensus_not_border_touching",
            "eval_mvtec_selected_refinement": selected,
        }

    base_arr = np.asarray(eval_tight.convert("L"), dtype=np.uint8) > 0
    original_pixels = int(base_arr.sum())
    if original_pixels <= 0:
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "zipper_fabric_empty_eval_tight",
            "eval_mvtec_selected_refinement": selected,
        }

    if selected == "sam2_heatmap":
        radius = int(auto.get("zipper_fabric_eval_mvtec_sam_vertical_radius", 8))
    else:
        radius = int(auto.get("zipper_fabric_eval_mvtec_consensus_vertical_radius", 6))
    radius = max(0, min(32, radius))
    if radius <= 0:
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "zipper_fabric_vertical_radius_zero",
            "eval_mvtec_selected_refinement": selected,
        }

    expanded = _dilate_bool_axis(base_arr, radius=radius, axis=0)
    expanded_pixels = int(expanded.sum())
    max_image_fraction = float(auto.get("zipper_fabric_eval_mvtec_max_image_fraction", 0.035))
    max_pixels = int(max(1, image_size[0] * image_size[1] * max_image_fraction))
    if expanded_pixels <= int(original_pixels * 1.02):
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "zipper_fabric_no_effective_vertical_expansion",
            "eval_mvtec_original_pixels": original_pixels,
            "eval_mvtec_pixels": expanded_pixels,
            "eval_mvtec_selected_refinement": selected,
        }
    if expanded_pixels > max_pixels:
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "zipper_fabric_vertical_expansion_exceeds_cap",
            "eval_mvtec_original_pixels": original_pixels,
            "eval_mvtec_pixels": expanded_pixels,
            "eval_mvtec_max_pixels": max_pixels,
            "eval_mvtec_selected_refinement": selected,
        }

    return Image.fromarray(expanded.astype(np.uint8) * 255, mode="L"), {
        "eval_mvtec_source": "vertical_expanded_eval_tight",
        "eval_mvtec_reason": "zipper_fabric_border_vertical_support",
        "eval_mvtec_original_pixels": original_pixels,
        "eval_mvtec_pixels": expanded_pixels,
        "eval_mvtec_vertical_radius": radius,
        "eval_mvtec_selected_refinement": selected,
        "eval_mvtec_qc_reasons": sorted(reasons),
    }


def _dilate_bool_axis(mask: np.ndarray, *, radius: int, axis: int) -> np.ndarray:
    radius = max(0, int(radius))
    output = mask.astype(bool, copy=True)
    source = mask.astype(bool, copy=False)
    for offset in range(1, radius + 1):
        if axis == 0:
            output[offset:, :] |= source[:-offset, :]
            output[:-offset, :] |= source[offset:, :]
        elif axis == 1:
            output[:, offset:] |= source[:, :-offset]
            output[:, :-offset] |= source[:, offset:]
        else:
            raise ValueError(f"Unsupported dilation axis: {axis}")
    return output


def _ring_sector_eval_prior(
    eval_tight: Image.Image,
    *,
    artifacts: dict[str, Any],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not bool(auto.get("ring_sector_eval_mvtec_prior_enabled", auto.get("bottle_large_eval_mvtec_prior_enabled", True))):
        return eval_tight, {"eval_mvtec_reason": "ring_sector_prior_disabled"}
    heatmap_path = _selected_candidate_heatmap_path(artifacts)
    if not heatmap_path:
        return eval_tight, {"eval_mvtec_reason": "ring_sector_missing_heatmap"}
    left, top, right, bottom = region
    if right <= left or bottom <= top:
        return eval_tight, {"eval_mvtec_reason": "ring_sector_empty_region"}

    base_arr = np.asarray(eval_tight.convert("L"), dtype=np.uint8) > 0
    heat = Image.open(heatmap_path).convert("L").resize(image_size, Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat, dtype=np.float32) / 255.0
    crop = base_arr[top:bottom, left:right]
    heat_crop = heat_arr[top:bottom, left:right]
    if crop.size == 0 or not np.any(heat_crop > 0):
        return eval_tight, {"eval_mvtec_reason": "ring_sector_unusable_heatmap"}

    if bool(auto.get("ring_sector_eval_mvtec_sector_enabled", auto.get("bottle_large_eval_mvtec_sector_enabled", True))):
        sector_width_degrees = float(auto.get("ring_sector_width_degrees", auto.get("bottle_large_eval_mvtec_sector_width_degrees", 160.0)))
        sector = _ring_sector_mask(
            eval_tight,
            heat_arr,
            region,
            image_size,
            width_degrees=sector_width_degrees,
            inner_radius_fraction=float(auto.get("ring_sector_inner_radius_fraction", auto.get("bottle_large_eval_mvtec_sector_inner_radius_fraction", 0.28))),
            outer_radius_fraction=float(auto.get("ring_sector_outer_radius_fraction", auto.get("bottle_large_eval_mvtec_sector_outer_radius_fraction", 1.02))),
            heat_support_percentile=float(auto.get("ring_sector_heat_support_percentile", auto.get("bottle_large_eval_mvtec_sector_heat_support_percentile", 55.0))),
            close_size=int(auto.get("ring_sector_close_size", auto.get("bottle_large_eval_mvtec_sector_close_size", 9))),
            erode_size=int(auto.get("ring_sector_erode_size", auto.get("bottle_large_eval_mvtec_sector_erode_size", 5))),
        )
        sector_arr = np.asarray(sector.convert("L"), dtype=np.uint8) > 0
        if int(sector_arr.sum()) > int(base_arr.sum() * 0.75):
            return sector, {
                "eval_mvtec_source": "polar_heat_rim_sector",
                "eval_mvtec_reason": "ring_sector_heat_selected_sector",
                "eval_mvtec_original_pixels": int(base_arr.sum()),
                "eval_mvtec_pixels": int(sector_arr.sum()),
                "eval_mvtec_heatmap_path": heatmap_path,
                "eval_mvtec_sector_width_degrees": sector_width_degrees,
            }

    values = heat_crop[heat_crop > 0]
    threshold = float(np.percentile(values, float(auto.get("ring_sector_heat_percentile", auto.get("bottle_large_eval_mvtec_heat_percentile", 70.0))))) if values.size else 1.0
    support = heat_crop >= threshold
    support = _morph_close(
        support | _dilate_bool(crop, radius=int(auto.get("ring_sector_dilate_radius", auto.get("bottle_large_eval_mvtec_dilate_radius", 7)))),
        radius=int(auto.get("ring_sector_close_radius", auto.get("bottle_large_eval_mvtec_close_radius", 4))),
    )
    support = _filter_components_by_area(support, min_area=max(4, int(auto.get("min_component_area", 12))))
    max_fraction = float(auto.get("ring_sector_max_box_fraction", auto.get("bottle_large_eval_mvtec_max_box_fraction", 0.24)))
    support = _cap_mask_by_heat(support, heat_crop, max_area_fraction=max_fraction)
    if int(support.sum()) <= int(crop.sum() * 1.02):
        return eval_tight, {
            "eval_mvtec_source": "eval_tight",
            "eval_mvtec_reason": "ring_sector_no_effective_expansion",
            "eval_mvtec_original_pixels": int(crop.sum()),
            "eval_mvtec_pixels": int(support.sum()),
        }
    output = np.zeros_like(base_arr, dtype=bool)
    output[top:bottom, left:right] = support
    return Image.fromarray(output.astype(np.uint8) * 255, mode="L"), {
        "eval_mvtec_source": "heat_expanded_eval_tight",
        "eval_mvtec_reason": "ring_sector_heat_supported_broad_region",
        "eval_mvtec_original_pixels": int(crop.sum()),
        "eval_mvtec_pixels": int(support.sum()),
        "eval_mvtec_heatmap_path": heatmap_path,
        "eval_mvtec_threshold": float(threshold),
    }


def _bottle_large_mvtec_eval_prior(
    eval_tight: Image.Image,
    *,
    artifacts: dict[str, Any],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    return _ring_sector_eval_prior(
        eval_tight,
        artifacts=artifacts,
        region=region,
        image_size=image_size,
        auto=auto,
    )


def _ring_sector_mask(
    eval_tight: Image.Image,
    heat_arr: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    width_degrees: float,
    inner_radius_fraction: float,
    outer_radius_fraction: float,
    heat_support_percentile: float,
    close_size: int,
    erode_size: int,
) -> Image.Image:
    width, height = image_size
    left, top, right, bottom = region
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    radius = max(1.0, min(max(1, right - left), max(1, bottom - top)) / 2.0)
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx.astype(np.float32) - float(cx)
    dy = yy.astype(np.float32) - float(cy)
    rr = np.sqrt(dx * dx + dy * dy)
    angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    annulus = (
        (rr >= radius * max(0.05, inner_radius_fraction))
        & (rr <= radius * max(inner_radius_fraction + 0.05, outer_radius_fraction))
        & (xx >= left)
        & (xx < right)
        & (yy >= top)
        & (yy < bottom)
    )
    if not np.any(annulus):
        return eval_tight

    core = np.asarray(eval_tight.convert("L").resize(image_size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
    signal = heat_arr.astype(np.float32) * (0.35 + 0.65 * core.astype(np.float32))
    bin_count = 72
    bin_width = 360.0 / bin_count
    scores = np.zeros(bin_count, dtype=np.float32)
    for index in range(bin_count):
        lo = index * bin_width
        hi = (index + 1) * bin_width
        selected = annulus & (angle >= lo) & (angle < hi)
        if np.any(selected):
            scores[index] = float(signal[selected].mean())
    if not np.any(scores > 0):
        return eval_tight

    smooth = np.zeros_like(scores)
    for index in range(bin_count):
        smooth[index] = (
            scores[(index - 2) % bin_count] * 0.12
            + scores[(index - 1) % bin_count] * 0.22
            + scores[index] * 0.32
            + scores[(index + 1) % bin_count] * 0.22
            + scores[(index + 2) % bin_count] * 0.12
        )
    bins_to_keep = max(1, int(round(max(10.0, min(240.0, width_degrees)) / bin_width)))
    best_start = 0
    best_score = -1.0
    for start in range(bin_count):
        score = float(sum(smooth[(start + offset) % bin_count] for offset in range(bins_to_keep)))
        if score > best_score:
            best_score = score
            best_start = start

    sector = np.zeros((height, width), dtype=bool)
    for offset in range(bins_to_keep):
        index = (best_start + offset) % bin_count
        lo = index * bin_width
        hi = (index + 1) * bin_width
        sector |= (angle >= lo) & (angle < hi)

    heat_values = heat_arr[annulus]
    threshold = float(np.percentile(heat_values, max(1.0, min(99.0, heat_support_percentile)))) if heat_values.size else 1.0
    mask = annulus & sector & (core | (heat_arr >= threshold))
    close_size = max(1, int(close_size))
    if close_size % 2 == 0:
        close_size += 1
    erode_size = max(1, int(erode_size))
    if erode_size % 2 == 0:
        erode_size += 1
    output = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    output = output.filter(ImageFilter.MaxFilter(size=close_size)).filter(ImageFilter.MinFilter(size=erode_size))
    return output.convert("L")


def _bottle_large_rim_sector_mask(
    eval_tight: Image.Image,
    heat_arr: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    width_degrees: float,
    inner_radius_fraction: float,
    outer_radius_fraction: float,
    heat_support_percentile: float,
    close_size: int,
    erode_size: int,
) -> Image.Image:
    return _ring_sector_mask(
        eval_tight,
        heat_arr,
        region,
        image_size,
        width_degrees=width_degrees,
        inner_radius_fraction=inner_radius_fraction,
        outer_radius_fraction=outer_radius_fraction,
        heat_support_percentile=heat_support_percentile,
        close_size=close_size,
        erode_size=erode_size,
    )


def _selected_candidate_heatmap_path(artifacts: dict[str, Any]) -> str:
    selected = str(artifacts.get("selected_refinement", ""))
    heatmap_path = str(artifacts.get("candidate_heatmap_paths", {}).get(selected, ""))
    if heatmap_path and Path(heatmap_path).exists():
        return heatmap_path
    params = artifacts.get("parameters", {})
    for key in (
        "heatmap_path",
        "nearest_support_heatmap_path",
        "support_heatmap_path",
        "normal_residual_fusion_heatmap_path",
    ):
        heatmap_path = str(params.get(key, ""))
        if heatmap_path and Path(heatmap_path).exists():
            return heatmap_path
    return ""


def _recall_expand_eval_mask(
    mask: Image.Image,
    *,
    artifacts: dict[str, Any],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not bool(auto.get("bottle_large_recall_expand_enabled", True)):
        return mask, {"recall_expanded": False, "recall_expand_reason": "disabled"}
    if category != "bottle" or defect_type != "broken_large":
        return mask, {"recall_expanded": False, "recall_expand_reason": "no_profile"}

    heatmap_path = _selected_candidate_heatmap_path(artifacts)
    if not heatmap_path:
        return mask, {"recall_expanded": False, "recall_expand_reason": "missing_selected_heatmap"}

    left, top, right, bottom = region
    if right <= left or bottom <= top:
        return mask, {"recall_expanded": False, "recall_expand_reason": "empty_region"}
    mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    crop = mask_arr[top:bottom, left:right]
    original_pixels = int(crop.sum())
    if original_pixels <= 0:
        return mask, {"recall_expanded": False, "recall_expand_reason": "empty_mask"}

    region_pixels = max(1, crop.size)
    current_fraction = original_pixels / region_pixels
    min_trigger_fraction = float(auto.get("bottle_large_recall_expand_min_box_fraction", 0.15))
    if current_fraction >= min_trigger_fraction:
        return mask, {
            "recall_expanded": False,
            "recall_expand_reason": "already_large_enough",
            "recall_expand_original_pixels": original_pixels,
            "recall_expand_box_fraction": float(current_fraction),
        }

    heat = Image.open(heatmap_path).convert("L").resize(image_size, Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat, dtype=np.float32) / 255.0
    heat_crop = heat_arr[top:bottom, left:right]
    if heat_crop.shape != crop.shape or not np.any(heat_crop > 0):
        return mask, {"recall_expanded": False, "recall_expand_reason": "unusable_selected_heatmap"}

    percentile = float(auto.get("bottle_large_recall_expand_percentile", 82.0))
    values = heat_crop[heat_crop > 0]
    threshold = float(np.percentile(values, max(50.0, min(99.0, percentile)))) if values.size else 1.0
    support = heat_crop >= threshold
    support = support | (_dilate_bool(crop, radius=int(auto.get("bottle_large_recall_expand_dilate_radius", 5))) & (heat_crop >= threshold * 0.75))
    expanded = _morph_close(crop | support, radius=int(auto.get("bottle_large_recall_expand_close_radius", 2)))
    expanded = _filter_components_by_area(expanded, min_area=max(2, int(auto.get("min_component_area", 12)) // 2))
    max_box_fraction = float(auto.get("bottle_large_recall_expand_max_box_fraction", 0.20))
    expanded = _cap_mask_by_heat(expanded, heat_crop, max_area_fraction=max_box_fraction)
    if int(expanded.sum()) <= int(original_pixels * 1.05):
        return mask, {
            "recall_expanded": False,
            "recall_expand_reason": "no_effective_expansion",
            "recall_expand_original_pixels": original_pixels,
        }

    output = np.zeros_like(mask_arr, dtype=bool)
    output[top:bottom, left:right] = expanded
    return Image.fromarray(output.astype(np.uint8) * 255, mode="L"), {
        "recall_expanded": True,
        "recall_expand_profile": "bottle_large_heat_support",
        "recall_expand_original_pixels": original_pixels,
        "recall_expand_pixels": int(expanded.sum()),
        "recall_expand_threshold": float(threshold),
        "recall_expand_heatmap_path": heatmap_path,
    }


def _zipper_structure_cleanup_eval_mask(
    mask: Image.Image,
    *,
    artifacts: dict[str, Any],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not bool(auto.get("zipper_structure_cleanup_enabled", True)):
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "disabled"}
    if category != "zipper":
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "no_profile"}
    if defect_type == "split_teeth":
        return mask, {
            "zipper_structure_cleaned": False,
            "zipper_structure_reason": "split_teeth_preserve_recall",
        }
    if defect_type == "broken_teeth":
        return _zipper_tooth_axis_cleanup_eval_mask(
            mask,
            artifacts=artifacts,
            region=region,
            image_size=image_size,
            auto=auto,
        )
    if defect_type == "fabric_border":
        return _zipper_fabric_border_cleanup_eval_mask(
            mask,
            region=region,
            image_size=image_size,
            auto=auto,
        )
    return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "no_profile"}


def _zipper_tooth_axis_cleanup_eval_mask(
    mask: Image.Image,
    *,
    artifacts: dict[str, Any],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    if right <= left or bottom <= top:
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "empty_region"}
    mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    crop = mask_arr[top:bottom, left:right]
    original_pixels = int(crop.sum())
    if original_pixels <= 0:
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "empty_mask"}

    region_pixels = max(1, crop.size)
    trigger_fraction = float(auto.get("zipper_tooth_structure_min_box_fraction", 0.035))
    if original_pixels / region_pixels < trigger_fraction:
        return mask, {
            "zipper_structure_cleaned": False,
            "zipper_structure_reason": "already_sparse",
            "zipper_structure_original_pixels": original_pixels,
            "zipper_structure_box_fraction": float(original_pixels / region_pixels),
        }

    ys, xs = np.where(crop)
    if len(xs) < max(8, int(auto.get("min_component_area", 16)) // 2):
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "too_few_pixels"}

    coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    center = coords.mean(axis=0)
    centered = coords - center
    covariance = centered.T @ centered / max(1, len(coords))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    yy, xx = np.indices(crop.shape, dtype=np.float64)
    grid_coordinates = np.stack((xx, yy), axis=-1)
    selected_perp_coordinate = coords @ perpendicular
    if selected_perp_coordinate.size == 0:
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "empty_axis"}

    hist, edges = np.histogram(selected_perp_coordinate, bins=min(32, max(8, int(np.sqrt(len(coords))))))
    peak_index = int(np.argmax(hist))
    dominant_perp = float((edges[peak_index] + edges[peak_index + 1]) / 2.0)
    bin_width = float(max(1e-6, edges[peak_index + 1] - edges[peak_index]))
    dominant_band = np.abs(selected_perp_coordinate - dominant_perp) <= bin_width * 1.75
    width_source = np.abs(selected_perp_coordinate[dominant_band] - dominant_perp)
    if width_source.size < max(5, len(selected_perp_coordinate) // 6):
        width_source = np.abs(selected_perp_coordinate - dominant_perp)
    percentile = float(auto.get("zipper_tooth_structure_corridor_percentile", 82.0))
    learned_width = float(np.percentile(width_source, max(50.0, min(98.0, percentile))))
    max_width = float(max(crop.shape) * float(auto.get("zipper_tooth_structure_max_width_fraction", 0.18)))
    min_width = float(auto.get("zipper_tooth_structure_min_width_px", 8.0))
    corridor_half_width = max(min_width, min(max_width, learned_width))
    perp_coordinate = grid_coordinates @ perpendicular
    corridor = np.abs(perp_coordinate - dominant_perp) <= corridor_half_width
    cleaned = crop & corridor

    heatmap_path = _selected_candidate_heatmap_path(artifacts)
    if heatmap_path and Path(heatmap_path).exists() and int(cleaned.sum()) > 0:
        heat = Image.open(heatmap_path).convert("L").resize(image_size, Image.Resampling.BILINEAR)
        heat_crop = np.asarray(heat, dtype=np.float32)[top:bottom, left:right] / 255.0
        if heat_crop.shape == crop.shape and np.any(heat_crop[cleaned]):
            heat_percentile = float(auto.get("zipper_tooth_structure_heat_percentile", 45.0))
            threshold = float(np.percentile(heat_crop[crop], max(1.0, min(95.0, heat_percentile))))
            cleaned = cleaned & ((heat_crop >= threshold) | _dilate_bool(cleaned & (heat_crop >= threshold), radius=1))

    cleaned = _filter_components_by_area(
        cleaned,
        min_area=max(2, int(auto.get("min_component_area", 16)) // 3),
    )
    min_keep = int(round(original_pixels * float(auto.get("zipper_tooth_structure_min_keep_fraction", 0.55))))
    if int(cleaned.sum()) < max(1, min_keep):
        return mask, {
            "zipper_structure_cleaned": False,
            "zipper_structure_reason": "would_remove_too_much",
            "zipper_structure_original_pixels": original_pixels,
            "zipper_structure_candidate_pixels": int(cleaned.sum()),
            "zipper_structure_min_keep_pixels": int(min_keep),
        }
    if int(cleaned.sum()) >= int(original_pixels * 0.98):
        return mask, {
            "zipper_structure_cleaned": False,
            "zipper_structure_reason": "no_effective_cleanup",
            "zipper_structure_original_pixels": original_pixels,
        }

    output = np.zeros_like(mask_arr, dtype=bool)
    output[top:bottom, left:right] = cleaned
    orientation_degrees = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)
    return Image.fromarray(output.astype(np.uint8) * 255, mode="L"), {
        "zipper_structure_cleaned": True,
        "zipper_structure_profile": "tooth_axis_corridor",
        "zipper_structure_original_pixels": original_pixels,
        "zipper_structure_pixels": int(cleaned.sum()),
        "zipper_structure_orientation_degrees": orientation_degrees,
        "zipper_structure_corridor_half_width": float(corridor_half_width),
        "zipper_structure_heatmap_path": heatmap_path,
    }


def _zipper_fabric_border_cleanup_eval_mask(
    mask: Image.Image,
    *,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    width, _ = image_size
    left, top, right, bottom = region
    if right <= left or bottom <= top:
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "empty_region"}
    mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    crop = mask_arr[top:bottom, left:right]
    original_pixels = int(crop.sum())
    if original_pixels <= 0:
        return mask, {"zipper_structure_cleaned": False, "zipper_structure_reason": "empty_mask"}

    fractions = auto.get("zipper_fabric_layout_side_fractions", [0.08, 0.34, 0.66, 0.94])
    if not isinstance(fractions, list | tuple) or len(fractions) != 4:
        fractions = [0.08, 0.34, 0.66, 0.94]
    left_a, left_b, right_a, right_b = [float(value) for value in fractions]
    pad_fraction = float(auto.get("zipper_fabric_structure_side_padding_fraction", 0.035))
    left_a = max(0.0, left_a - pad_fraction)
    left_b = min(1.0, left_b + pad_fraction)
    right_a = max(0.0, right_a - pad_fraction)
    right_b = min(1.0, right_b + pad_fraction)
    global_x = np.arange(left, right, dtype=np.float32) / max(1, width)
    side_x = ((global_x >= left_a) & (global_x <= left_b)) | ((global_x >= right_a) & (global_x <= right_b))
    side_fence = np.repeat(side_x[None, :], crop.shape[0], axis=0)
    cleaned = crop & side_fence
    min_keep = int(round(original_pixels * float(auto.get("zipper_fabric_structure_min_keep_fraction", 0.45))))
    if int(cleaned.sum()) < max(1, min_keep):
        return mask, {
            "zipper_structure_cleaned": False,
            "zipper_structure_reason": "would_remove_too_much",
            "zipper_structure_original_pixels": original_pixels,
            "zipper_structure_candidate_pixels": int(cleaned.sum()),
            "zipper_structure_min_keep_pixels": int(min_keep),
        }
    cleaned = _filter_components_by_area(
        cleaned,
        min_area=max(2, int(auto.get("min_component_area", 16)) // 3),
    )
    if int(cleaned.sum()) <= 0 or int(cleaned.sum()) >= int(original_pixels * 0.98):
        return mask, {
            "zipper_structure_cleaned": False,
            "zipper_structure_reason": "no_effective_cleanup",
            "zipper_structure_original_pixels": original_pixels,
        }

    output = np.zeros_like(mask_arr, dtype=bool)
    output[top:bottom, left:right] = cleaned
    return Image.fromarray(output.astype(np.uint8) * 255, mode="L"), {
        "zipper_structure_cleaned": True,
        "zipper_structure_profile": "fabric_side_band",
        "zipper_structure_original_pixels": original_pixels,
        "zipper_structure_pixels": int(cleaned.sum()),
        "zipper_structure_side_fractions": [left_a, left_b, right_a, right_b],
    }


def _precision_tight_eval_mask(
    mask: Image.Image,
    *,
    artifacts: dict[str, Any],
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    if not bool(auto.get("precision_tight_eval_enabled", True)):
        return mask, {"precision_tightened": False, "precision_tight_reason": "disabled"}

    profile = _precision_tight_profile(
        category=category,
        defect_type=defect_type,
        selected_refinement=str(artifacts.get("selected_refinement", "")),
        qc=dict(artifacts.get("qc", {})),
        region=region,
        image_size=image_size,
        auto=auto,
    )
    if profile is None:
        return mask, {"precision_tightened": False, "precision_tight_reason": "no_profile"}

    heatmap_path = _selected_candidate_heatmap_path(artifacts)
    if not heatmap_path:
        return mask, {
            "precision_tightened": False,
            "precision_tight_reason": "missing_selected_heatmap",
            "precision_tight_profile": profile["name"],
        }

    left, top, right, bottom = region
    if right <= left or bottom <= top:
        return mask, {
            "precision_tightened": False,
            "precision_tight_reason": "empty_region",
            "precision_tight_profile": profile["name"],
        }

    mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    crop = mask_arr[top:bottom, left:right]
    original_pixels = int(crop.sum())
    if original_pixels <= 0:
        return mask, {
            "precision_tightened": False,
            "precision_tight_reason": "empty_mask",
            "precision_tight_profile": profile["name"],
        }

    image_area = max(1, image_size[0] * image_size[1])
    region_pixels = max(1, crop.size)
    max_by_box = int(region_pixels * float(profile["max_box_fraction"]))
    max_by_image = int(image_area * float(profile["max_image_fraction"]))
    max_pixels = max(1, min(max_by_box, max_by_image))
    min_keep_pixels = int(round(original_pixels * float(profile["min_keep_fraction"])))
    max_pixels = max(max_pixels, min_keep_pixels)
    if original_pixels <= int(max_pixels * 1.05):
        return mask, {
            "precision_tightened": False,
            "precision_tight_reason": "already_within_profile",
            "precision_tight_profile": profile["name"],
            "precision_tight_original_pixels": original_pixels,
            "precision_tight_max_pixels": int(max_pixels),
        }

    heat = Image.open(heatmap_path).convert("L").resize(image_size, Image.Resampling.BILINEAR)
    heat_arr = np.asarray(heat, dtype=np.float32) / 255.0
    heat_crop = heat_arr[top:bottom, left:right]
    if heat_crop.shape != crop.shape or not np.any(heat_crop[crop]):
        return mask, {
            "precision_tightened": False,
            "precision_tight_reason": "unusable_selected_heatmap",
            "precision_tight_profile": profile["name"],
        }

    capped = _cap_mask_by_heat(crop, heat_crop, max_area_fraction=max_pixels / region_pixels)
    capped = _filter_components(
        capped,
        defect_type,
        max(2, int(auto.get("min_component_area", 12)) // 2),
        require_elongated=False,
    )
    if int(capped.sum()) < min_keep_pixels:
        capped = _cap_mask_by_heat(crop, heat_crop, max_area_fraction=min_keep_pixels / region_pixels)
    if int(capped.sum()) <= 0 or int(capped.sum()) >= original_pixels:
        return mask, {
            "precision_tightened": False,
            "precision_tight_reason": "no_effective_tightening",
            "precision_tight_profile": profile["name"],
            "precision_tight_original_pixels": original_pixels,
        }

    tightened = np.zeros_like(mask_arr, dtype=bool)
    tightened[top:bottom, left:right] = capped
    return Image.fromarray(tightened.astype(np.uint8) * 255, mode="L"), {
        "precision_tightened": True,
        "precision_tight_profile": profile["name"],
        "precision_tight_original_pixels": original_pixels,
        "precision_tight_pixels": int(capped.sum()),
        "precision_tight_max_pixels": int(max_pixels),
        "precision_tight_heatmap_path": heatmap_path,
    }


def _precision_tight_profile(
    *,
    category: str,
    defect_type: str,
    selected_refinement: str,
    qc: dict[str, Any],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> dict[str, Any] | None:
    width, height = image_size
    left, top, right, bottom = region
    region_fraction = max(0, right - left) * max(0, bottom - top) / max(1, width * height)
    mask_box_fraction = float(qc.get("mask_to_qwen_box_fraction", 0.0))
    mask_image_fraction = float(qc.get("mask_area_fraction", 0.0))

    if category == "bottle" and defect_type == "broken_small":
        if mask_box_fraction <= 0.075 and mask_image_fraction <= 0.035:
            return None
        return {
            "name": "bottle_small_chip_precision_tight",
            "max_box_fraction": float(auto.get("bottle_small_precision_max_box_fraction", 0.080)),
            "max_image_fraction": float(auto.get("bottle_small_precision_max_image_fraction", 0.045)),
            "min_keep_fraction": float(auto.get("bottle_small_precision_min_keep_fraction", 0.45)),
        }

    if category == "zipper" and defect_type in {"broken_teeth", "split_teeth"}:
        if selected_refinement not in {"patchcore_guided", "normal_residual_fusion", "nearest_normal_residual"}:
            return None
        if mask_box_fraction <= 0.09 and mask_image_fraction <= 0.030:
            return None
        return {
            "name": "zipper_tooth_chain_precision_tight",
            "max_box_fraction": float(auto.get("zipper_tooth_precision_max_box_fraction", 0.090)),
            "max_image_fraction": float(auto.get("zipper_tooth_precision_max_image_fraction", 0.038)),
            "min_keep_fraction": float(auto.get("zipper_tooth_precision_min_keep_fraction", 0.55)),
        }

    if category == "zipper" and defect_type == "fabric_border":
        reasons = set(qc.get("reasons", []))
        if region_fraction < 0.50 or "touches_image_border" not in reasons:
            return None
        return {
            "name": "zipper_fabric_border_full_frame_precision_tight",
            "max_box_fraction": float(auto.get("zipper_fabric_border_precision_max_box_fraction", 0.020)),
            "max_image_fraction": float(auto.get("zipper_fabric_border_precision_max_image_fraction", 0.018)),
            "min_keep_fraction": float(auto.get("zipper_fabric_border_precision_min_keep_fraction", 0.55)),
        }

    return None


def _variant_or_default(artifacts: dict[str, Any], variant: str, fallback_key: str) -> str:
    variants = artifacts.get("mask_variant_paths", {})
    if variant in variants:
        return str(variants[variant])
    if fallback_key in artifacts:
        return str(artifacts[fallback_key])
    raise ValueError(f"Mask variant {variant!r} is unavailable")


def _uncertainty_mask_variants(
    base: Image.Image,
    candidate_refined_paths: dict[str, str],
    candidate_heatmap_paths: dict[str, str],
    morphology: str,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    base_arr = np.asarray(base.convert("L"), dtype=np.uint8) > 0
    mask_arrays: list[np.ndarray] = [base_arr]
    for path_value in candidate_refined_paths.values():
        path = Path(str(path_value))
        if not path.exists():
            continue
        arr = np.asarray(Image.open(path).convert("L").resize(image_size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
        if arr.any():
            mask_arrays.append(arr)

    stack = np.stack(mask_arrays, axis=0)
    vote = stack.mean(axis=0)
    if (
        morphology == "multi_scuff"
        and bool(auto.get("calibrated_soft_scuff_ensemble", True))
        and candidate_heatmap_paths
    ):
        variants, params = _calibrated_soft_scuff_variants(
            base_arr=base_arr,
            vote=vote,
            candidate_heatmap_paths=candidate_heatmap_paths,
            region=region,
            image_size=image_size,
            auto=auto,
        )
        if variants is not None:
            params["candidate_vote_count"] = len(mask_arrays)
            return variants, params

    min_vote = float(auto.get("uncertainty_possible_vote_threshold", 0.34))
    core_vote = float(auto.get("uncertainty_core_vote_threshold", 0.67))
    positive_core = vote >= core_vote
    possible_region = vote >= min_vote
    if len(mask_arrays) < 3:
        possible_region = possible_region | _dilate_bool(base_arr, radius=int(auto.get("uncertainty_fallback_dilate_radius", 3)))
        positive_core = positive_core | base_arr

    left, top, right, bottom = region
    fence = np.zeros(base_arr.shape, dtype=bool)
    fence[top:bottom, left:right] = True
    possible_region &= fence
    positive_core &= fence
    uncertainty = possible_region & ~positive_core
    boundary = _dilate_bool(base_arr, radius=int(auto.get("uncertainty_boundary_radius", 2))) & ~base_arr
    uncertainty = (uncertainty | boundary) & fence
    soft = np.zeros(base_arr.shape, dtype=np.uint8)
    soft[possible_region] = 128
    soft[positive_core] = 255
    soft_image = Image.fromarray(soft, mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(auto.get("training_soft_blur", 1.2)))
    )
    soft_arr = np.array(soft_image, dtype=np.uint8, copy=True)
    soft_arr[positive_core] = 255
    soft_image = Image.fromarray(soft_arr, mode="L")
    variants = {
        "training_soft": soft_image,
        "positive_core": Image.fromarray(positive_core.astype(np.uint8) * 255, mode="L"),
        "possible_region": Image.fromarray(possible_region.astype(np.uint8) * 255, mode="L"),
        "uncertainty_map": Image.fromarray(uncertainty.astype(np.uint8) * 255, mode="L"),
    }
    params = {
        "candidate_vote_count": len(mask_arrays),
        "core_vote_threshold": core_vote,
        "possible_vote_threshold": min_vote,
        "positive_core_pixels": int(positive_core.sum()),
        "possible_region_pixels": int(possible_region.sum()),
        "uncertainty_pixels": int(uncertainty.sum()),
    }
    return variants, params


def _calibrated_soft_scuff_variants(
    *,
    base_arr: np.ndarray,
    vote: np.ndarray,
    candidate_heatmap_paths: dict[str, str],
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[dict[str, Image.Image], dict[str, Any]] | tuple[None, dict[str, Any]]:
    left, top, right, bottom = region
    fence = np.zeros(base_arr.shape, dtype=bool)
    fence[top:bottom, left:right] = True
    weighted_sum = np.zeros(base_arr.shape, dtype=np.float32)
    total_weight = 0.0
    weights = {
        "patchcore_guided": float(auto.get("soft_scuff_patchcore_weight", 0.42)),
        "nearest_normal_residual": float(auto.get("soft_scuff_nearest_residual_weight", 0.26)),
        "fft_texture_suppression": float(auto.get("soft_scuff_fft_weight", 0.18)),
        "normal_anomaly": float(auto.get("soft_scuff_normal_anomaly_weight", 0.14)),
        "multi_scuff_fusion": float(auto.get("soft_scuff_multi_scuff_weight", 0.10)),
    }
    used_heatmaps: list[str] = []
    for mode, weight in weights.items():
        path_value = candidate_heatmap_paths.get(mode)
        if weight <= 0 or not path_value:
            continue
        path = Path(str(path_value))
        if not path.exists():
            continue
        heat = np.asarray(Image.open(path).convert("L").resize(image_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        heat_crop = heat[top:bottom, left:right]
        if heat_crop.size == 0 or float(heat_crop.max()) <= 0.0:
            continue
        normalized = np.zeros_like(heat, dtype=np.float32)
        normalized[top:bottom, left:right] = _robust_heatmap_probability(heat_crop)
        weighted_sum += normalized * weight
        total_weight += weight
        used_heatmaps.append(mode)
    if total_weight <= 0.0:
        return None, {"soft_scuff_ensemble_reason": "no_usable_heatmaps"}

    probability = weighted_sum / max(1e-6, total_weight)
    vote_weight = float(auto.get("soft_scuff_vote_weight", 0.22))
    base_weight = float(auto.get("soft_scuff_base_weight", 0.12))
    probability = _normalize_float(
        probability * max(0.0, 1.0 - vote_weight - base_weight)
        + vote.astype(np.float32) * vote_weight
        + base_arr.astype(np.float32) * base_weight
    )
    probability[~fence] = 0.0
    probability = np.asarray(
        Image.fromarray(np.uint8(np.clip(probability, 0, 1) * 255), mode="L").filter(
            ImageFilter.GaussianBlur(radius=float(auto.get("soft_scuff_probability_blur", 1.1)))
        ),
        dtype=np.float32,
    ) / 255.0
    probability[~fence] = 0.0

    crop = probability[top:bottom, left:right]
    nonzero = crop[crop > 0]
    if nonzero.size == 0:
        return None, {"soft_scuff_ensemble_reason": "empty_probability"}
    possible_percentile = float(auto.get("soft_scuff_possible_percentile", 72.0))
    core_percentile = float(auto.get("soft_scuff_core_percentile", 92.0))
    possible_threshold = max(float(auto.get("soft_scuff_min_possible_probability", 0.22)), float(np.percentile(nonzero, possible_percentile)))
    core_threshold = max(float(auto.get("soft_scuff_min_core_probability", 0.62)), float(np.percentile(nonzero, core_percentile)))
    possible_region = (probability >= possible_threshold) & fence
    positive_core = ((probability >= core_threshold) | (base_arr & (probability >= possible_threshold))) & fence

    possible_region |= _dilate_bool(base_arr, radius=int(auto.get("soft_scuff_base_possible_dilate_radius", 2))) & fence
    possible_region = _limit_mask_area(
        possible_region[top:bottom, left:right],
        probability[top:bottom, left:right],
        max_area_fraction=float(auto.get("soft_scuff_possible_max_box_fraction", 0.22)),
    )
    possible_full = np.zeros_like(base_arr, dtype=bool)
    possible_full[top:bottom, left:right] = possible_region
    possible_region = possible_full
    positive_core &= possible_region
    positive_core = _limit_mask_area(
        positive_core[top:bottom, left:right],
        probability[top:bottom, left:right],
        max_area_fraction=float(auto.get("soft_scuff_core_max_box_fraction", 0.10)),
    )
    core_full = np.zeros_like(base_arr, dtype=bool)
    core_full[top:bottom, left:right] = positive_core
    positive_core = core_full
    uncertainty = (possible_region & ~positive_core) | (
        _dilate_bool(base_arr, radius=int(auto.get("uncertainty_boundary_radius", 2))) & ~base_arr & fence
    )

    soft = np.zeros(base_arr.shape, dtype=np.uint8)
    soft_values = np.uint8(np.clip(probability * 255.0, 0, 255))
    soft[possible_region] = np.maximum(
        soft_values[possible_region],
        int(auto.get("soft_scuff_possible_floor", 96)),
    )
    soft[positive_core] = 255
    soft_image = Image.fromarray(soft, mode="L").filter(
        ImageFilter.GaussianBlur(radius=float(auto.get("training_soft_blur", 1.2)))
    )
    soft_arr = np.array(soft_image, dtype=np.uint8, copy=True)
    soft_arr[positive_core] = 255
    soft_arr[~possible_region] = 0
    variants = {
        "training_soft": Image.fromarray(soft_arr, mode="L"),
        "positive_core": Image.fromarray(positive_core.astype(np.uint8) * 255, mode="L"),
        "possible_region": Image.fromarray(possible_region.astype(np.uint8) * 255, mode="L"),
        "uncertainty_map": Image.fromarray(uncertainty.astype(np.uint8) * 255, mode="L"),
    }
    params = {
        "soft_scuff_ensemble": True,
        "soft_scuff_heatmaps_used": used_heatmaps,
        "soft_scuff_possible_threshold": float(possible_threshold),
        "soft_scuff_core_threshold": float(core_threshold),
        "soft_scuff_possible_pixels": int(possible_region.sum()),
        "soft_scuff_positive_core_pixels": int(positive_core.sum()),
        "soft_scuff_uncertainty_pixels": int(uncertainty.sum()),
        "soft_scuff_probability_mean": float(probability[fence].mean()) if fence.any() else 0.0,
        "soft_scuff_probability_p95": float(np.percentile(crop, 95.0)),
    }
    return variants, params


def _robust_heatmap_probability(crop: np.ndarray) -> np.ndarray:
    values = crop[np.isfinite(crop)]
    if values.size == 0:
        return np.zeros_like(crop, dtype=np.float32)
    low = float(np.percentile(values, 35.0))
    high = float(np.percentile(values, 98.0))
    if high <= low:
        return _normalize_float(crop)
    return np.clip((crop - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _variant_mask(mask: Image.Image, *, radius: int, blur: float) -> Image.Image:
    output = mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    for _ in range(max(0, radius)):
        output = output.filter(ImageFilter.MaxFilter(size=3))
    if blur > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=blur))
    return output.convert("L")

def _anisotropic_variant_mask(mask: Image.Image, radius: int, blur: float = 0.0, eccentricity_threshold: float = 2.5) -> Image.Image:
    import cv2
    if radius <= 0:
        if blur > 0:
            return mask.filter(ImageFilter.GaussianBlur(radius=blur))
        return mask
        
    arr = np.asarray(mask.convert("L"))
    coords = cv2.findNonZero(arr)
    kernel_size = radius * 2 + 1
    
    if coords is not None and len(coords) >= 5:
        rect = cv2.minAreaRect(coords)
        width, height = rect[1]
        angle = rect[2]
        
        if width < height:
            width, height = height, width
            angle += 90
            
        eccentricity = width / max(height, 1)
        if eccentricity > eccentricity_threshold:
            kernel = np.zeros((kernel_size * 2, kernel_size * 2), dtype=np.uint8)
            center = (kernel_size, kernel_size)
            axes = (max(1, radius // 3), radius)
            cv2.ellipse(kernel, center, axes, angle, 0, 360, 255, -1)
            kx, ky, kw, kh = cv2.boundingRect(kernel)
            cropped_kernel = kernel[ky:ky+kh, kx:kx+kw]
            
            dilated = cv2.dilate(arr, cropped_kernel)
            output = Image.fromarray(dilated, mode="L")
            if blur > 0:
                output = output.filter(ImageFilter.GaussianBlur(radius=blur))
            return output

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(arr, kernel)
    output = Image.fromarray(dilated, mode="L")
    if blur > 0:
        output = output.filter(ImageFilter.GaussianBlur(radius=blur))
    return output


def _adaptive_inpaint_soft_mask(
    mask: Image.Image,
    image: Image.Image,
    morphology: str,
    selected: str,
    auto: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    import math
    if morphology == "multi_scuff" or selected == "scuff_cluster":
        core_radius = int(auto.get("scuff_inpaint_core_radius", 5))
        halo_radius = int(auto.get("scuff_inpaint_halo_radius", 14))
        blur = float(auto.get("scuff_inpaint_blur", 3.0))
        core = _variant_mask(mask, radius=core_radius, blur=0.0)
        halo = _variant_mask(mask, radius=halo_radius, blur=blur)
        blended = _blend_soft_masks(core, halo, core_weight=float(auto.get("scuff_inpaint_core_weight", 0.65)))
        params = {
            "adaptive": True,
            "morphology": morphology,
            "core_radius": core_radius,
            "halo_radius": halo_radius,
            "blur": blur,
        }
    elif morphology == "scratch_band" or selected in {"scratch_band_clean", "linear", "multi_linear"}:
        core_radius = int(auto.get("scratch_band_inpaint_core_radius", 3))
        halo_radius = int(auto.get("scratch_band_inpaint_halo_radius", 8))
        blur = float(auto.get("scratch_band_inpaint_blur", 1.8))
        core = _variant_mask(mask, radius=core_radius, blur=0.0)
        halo = _variant_mask(mask, radius=halo_radius, blur=blur)
        blended = _blend_soft_masks(core, halo, core_weight=float(auto.get("scratch_band_inpaint_core_weight", 0.78)))
        params = {
            "adaptive": True,
            "morphology": morphology,
            "core_radius": core_radius,
            "halo_radius": halo_radius,
            "blur": blur,
        }
    else:
        radius = int(auto.get("inpaint_variant_radius", 12))
        blur = float(auto.get("inpaint_variant_blur", 2.0))
        blended = _variant_mask(mask, radius=radius, blur=blur)
        params = {
            "adaptive": False,
            "morphology": morphology,
            "radius": radius,
            "blur": blur,
        }
        
    if bool(auto.get("use_alpha_matting", True)):
        blended = _apply_alpha_matting(blended, image, radius=int(auto.get("matting_radius", 8)))
        
    arr = np.asarray(blended, dtype=np.float32)
    non_zero = arr[arr > 0]
    mean_density = float(np.mean(non_zero)) / 255.0 if non_zero.size > 0 else 0.0
    severity_class = "Severe" if mean_density > 0.6 else "Moderate" if mean_density > 0.3 else "Mild"
    
    params["severity_class"] = severity_class
    params["alpha_density"] = round(mean_density, 3)
    
    return blended, params


def _apply_alpha_matting(mask: Image.Image, image: Image.Image, radius: int = 8, eps: float = 1e-3) -> Image.Image:
    """Apply an O(1) Guided Filter to refine the mask alpha matte based on image edge guidance."""
    import scipy.ndimage as ndimage
    
    guide = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    p = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    
    mean_I = ndimage.uniform_filter(guide, size=2 * radius + 1, mode="reflect")
    mean_p = ndimage.uniform_filter(p, size=2 * radius + 1, mode="reflect")
    mean_Ip = ndimage.uniform_filter(guide * p, size=2 * radius + 1, mode="reflect")
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = ndimage.uniform_filter(guide * guide, size=2 * radius + 1, mode="reflect")
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = ndimage.uniform_filter(a, size=2 * radius + 1, mode="reflect")
    mean_b = ndimage.uniform_filter(b, size=2 * radius + 1, mode="reflect")
    
    q = mean_a * guide + mean_b
    return Image.fromarray(np.clip(q * 255.0, 0, 255).astype(np.uint8), mode="L")


def _blend_soft_masks(core: Image.Image, halo: Image.Image, *, core_weight: float) -> Image.Image:
    core_arr = np.asarray(core.convert("L"), dtype=np.float32)
    halo_arr = np.asarray(halo.convert("L"), dtype=np.float32)
    weight = max(0.0, min(1.0, core_weight))
    blended = np.maximum(core_arr * weight, halo_arr * (1.0 - weight))
    blended = np.maximum(blended, core_arr)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="L")


def _scuff_eval_tight_mask(
    mask: Image.Image,
    region: tuple[int, int, int, int],
    *,
    radius: int,
    max_components: int,
    max_box_fraction: float,
) -> Image.Image:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    left, top, right, bottom = region
    crop = arr[top:bottom, left:right]
    if crop.size == 0 or not crop.any():
        return mask.convert("L").point(lambda value: 255 if value > 0 else 0)
    if radius > 0:
        support = _dilate_bool(crop, radius=radius)
        crop = crop | (support & _morph_close(crop, radius=1))
    components = _connected_components(crop)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    for component in components:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        aspect = max(width / max(1, height), height / max(1, width))
        score = len(component) + min(80.0, aspect * 6.0)
        scored.append((score, component))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = np.zeros_like(crop, dtype=bool)
    for _, component in scored[: max(1, max_components)]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
    selected = _limit_mask_area(selected, crop.astype(np.float32), max_area_fraction=max_box_fraction)
    output = Image.new("L", mask.size, 0)
    output.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return output


def residual_refined_mask(
    image: Image.Image,
    normal_path: Path | None,
    region: tuple[int, int, int, int],
    defect_type: str,
    *,
    text_hint: str,
    percentile: float,
    min_component_area: int,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    defect_gray = image.convert("L")
    if normal_path is None:
        normal_gray = defect_gray.filter(ImageFilter.GaussianBlur(radius=9.0))
        normal_source = "blurred_defect_image"
    else:
        normal_gray = Image.open(normal_path).convert("L").resize(image.size, Image.Resampling.BILINEAR)
        normal_source = "nearest_normal"
    defect_crop = np.asarray(defect_gray.crop(region), dtype=np.float32)
    normal_crop = np.asarray(normal_gray.crop(region), dtype=np.float32)
    residual = np.abs(defect_crop - normal_crop)

    hint = text_hint.lower()
    if any(word in hint for word in ("dark", "black", "brown", "shadow", "burn")):
        signed = np.clip(normal_crop - defect_crop, 0, None)
        residual = np.maximum(residual * 0.35, signed)
        polarity = "dark"
    elif any(word in hint for word in ("bright", "white", "light", "silver", "pale")):
        signed = np.clip(defect_crop - normal_crop, 0, None)
        residual = np.maximum(residual * 0.35, signed)
        polarity = "bright"
    else:
        polarity = "absolute"

    threshold = float(np.percentile(residual, max(50.0, min(99.5, percentile))))
    active = residual >= threshold
    active = _filter_components(active, defect_type, min_component_area)
    if active.sum() == 0:
        active = residual >= float(np.percentile(residual, 90.0))
        active = _filter_components(active, defect_type, max(4, min_component_area // 2))
    mask_crop = Image.fromarray(active.astype(np.uint8) * 255, mode="L")
    mask = Image.new("L", image.size, 0)
    mask.paste(mask_crop, (left, top))
    return mask, {
        "residual_source": normal_source,
        "residual_polarity": polarity,
        "residual_percentile": percentile,
        "residual_threshold": threshold,
        "residual_active_pixels": int(active.sum()),
    }


def parse_qwen_bbox_payload(text: str, image_size: tuple[int, int]) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        recovered = _recover_truncated_qwen_payload(text, image_size)
        if recovered is None:
            raise ValueError(f"Qwen did not return a JSON object: {text!r}")
        return recovered
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        recovered = _recover_truncated_qwen_payload(text, image_size)
        if recovered is None:
            raise ValueError(f"Qwen returned malformed bbox JSON: {text!r}") from exc
        return recovered
    if not isinstance(payload, dict):
        raise ValueError(f"Qwen bbox payload must be a JSON object: {text!r}")
    raw_box = payload.get("bbox_xyxy") or payload.get("bbox") or payload.get("box") or payload.get("region")
    if not isinstance(raw_box, list | tuple) or len(raw_box) != 4:
        raise ValueError(f"Qwen bbox payload missing bbox_xyxy: {payload!r}")
    box = _coerce_box(raw_box, image_size)
    raw_sub_boxes = (
        payload.get("sub_boxes_xyxy")
        or payload.get("boxes_xyxy")
        or payload.get("defect_boxes_xyxy")
        or payload.get("sub_regions")
        or []
    )
    sub_boxes: list[tuple[int, int, int, int]] = []
    if isinstance(raw_sub_boxes, list | tuple):
        for raw_sub_box in raw_sub_boxes:
            if isinstance(raw_sub_box, list | tuple) and len(raw_sub_box) == 4:
                try:
                    sub_boxes.append(_coerce_box(raw_sub_box, image_size))
                except Exception:
                    continue
    return {
        "bbox_xyxy": box,
        "sub_boxes_xyxy": sub_boxes,
        "defect_type": str(payload["defect_type"]) if payload.get("defect_type") is not None else None,
        "confidence": _optional_float(payload.get("confidence")),
        "evidence": str(payload["evidence"]) if payload.get("evidence") is not None else None,
    }


def validate_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(1, min(width, int(round(x2))))
    y2 = max(1, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid empty bbox after clipping: {bbox_xyxy}")
    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, width * height)
    if area_ratio < min_area_ratio:
        raise ValueError(f"Bbox area ratio {area_ratio:.6f} is below minimum {min_area_ratio:.6f}: {(x1, y1, x2, y2)}")
    if area_ratio > max_area_ratio:
        raise ValueError(f"Bbox area ratio {area_ratio:.6f} is above maximum {max_area_ratio:.6f}: {(x1, y1, x2, y2)}")
    return (x1, y1, x2, y2)


def _validated_or_fallback_region(
    bbox_xyxy: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    auto: dict[str, Any],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    min_area_ratio = float(auto.get("min_box_area_ratio", 0.001))
    max_area_ratio = float(auto.get("max_box_area_ratio", 0.5))
    try:
        region = validate_bbox(
            bbox_xyxy,
            image_size,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        return region, {
            "status": "valid",
            "raw_bbox_xyxy": list(bbox_xyxy),
            "region_xyxy": list(region),
            "fallback_mode": None,
            "failure_reason": None,
        }
    except ValueError as exc:
        if not bool(auto.get("allow_localization_fallback", False)):
            raise
        mode = str(auto.get("localization_fallback_mode", "full_image"))
        width, height = image_size
        if mode == "full_image":
            region = (0, 0, width, height)
        elif mode == "center_inset":
            inset_fraction = max(0.0, min(0.45, float(auto.get("localization_fallback_inset_fraction", 0.04))))
            inset_x = int(round(width * inset_fraction))
            inset_y = int(round(height * inset_fraction))
            region = (inset_x, inset_y, width - inset_x, height - inset_y)
        else:
            raise ValueError(f"Unsupported localization_fallback_mode: {mode}") from exc
        return region, {
            "status": "fallback",
            "raw_bbox_xyxy": list(bbox_xyxy),
            "region_xyxy": list(region),
            "fallback_mode": mode,
            "failure_reason": str(exc),
        }


def _validated_sub_boxes(
    boxes: object,
    image_size: tuple[int, int],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[tuple[int, int, int, int]]:
    if not isinstance(boxes, list | tuple):
        return []
    valid: list[tuple[int, int, int, int]] = []
    for box in boxes:
        if not isinstance(box, list | tuple) or len(box) != 4:
            continue
        try:
            valid.append(validate_bbox(tuple(int(value) for value in box), image_size, min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio))
        except Exception:
            continue
    return valid


def _fuse_regions(
    regions: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    *,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    width, height = image_size
    x1 = min(region[0] for region in regions)
    y1 = min(region[1] for region in regions)
    x2 = max(region[2] for region in regions)
    y2 = max(region[3] for region in regions)
    pad_x = int(round((x2 - x1) * max(0.0, padding_fraction)))
    pad_y = int(round((y2 - y1) * max(0.0, padding_fraction)))
    return (max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y))


def _select_qwen_region_from_sub_boxes(
    region: tuple[int, int, int, int],
    valid_sub_boxes: list[tuple[int, int, int, int]],
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    selection: dict[str, Any] = {
        "sub_boxes_xyxy": [list(box) for box in valid_sub_boxes],
        "sub_box_region_xyxy": None,
        "sub_box_region_selected": False,
    }
    if not valid_sub_boxes:
        return region, selection
    padding = float(auto.get("qwen_sub_box_padding", 0.08))
    sub_region = _fuse_regions(valid_sub_boxes, image_size, padding_fraction=padding)
    selection["sub_box_region_xyxy"] = list(sub_region)
    if bool(auto.get("qwen_sub_boxes_primary", False)):
        selection["sub_box_region_selected"] = True
        selection["sub_box_selection_mode"] = "primary"
        return sub_region, selection
    if bool(auto.get("qwen_fuse_sub_boxes", True)):
        selection["sub_box_selection_mode"] = "fused_with_main"
        return _fuse_regions([region, *valid_sub_boxes], image_size, padding_fraction=padding), selection
    selection["sub_box_selection_mode"] = "ignored"
    return region, selection


def bottle_rim_guided_qwen_region(
    *,
    image_size: tuple[int, int],
    qwen_region: tuple[int, int, int, int],
    defect_type: str,
    auto: dict[str, Any],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    if defect_type == "broken_large":
        fractions = auto.get("bottle_large_rim_region_fraction", [0.13, 0.14, 0.90, 0.93])
    else:
        fractions = auto.get("bottle_small_rim_region_fraction", [0.20, 0.16, 0.85, 0.78])
    rim_region = _fractional_region(fractions, image_size, default=(0.13, 0.14, 0.90, 0.93))
    union_qwen = bool(auto.get("bottle_rim_union_qwen", False))
    region = _union_candidate_regions([rim_region, qwen_region], image_size) if union_qwen else rim_region
    return region, {
        "status": "valid",
        "model": "bottle_rim_guided_qwen",
        "defect_type": defect_type,
        "qwen_region_xyxy": list(qwen_region),
        "rim_region_xyxy": list(rim_region),
        "region_xyxy": list(region),
        "union_qwen": union_qwen,
    }


def bottle_surface_guided_region(
    *,
    image_size: tuple[int, int],
    qwen_region: tuple[int, int, int, int],
    auto: dict[str, Any],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    fractions = auto.get("bottle_surface_region_fraction", [0.23, 0.23, 0.77, 0.77])
    surface_region = _fractional_region(fractions, image_size, default=(0.23, 0.23, 0.77, 0.77))
    qwen_area = _region_area_fraction(qwen_region, image_size)
    min_qwen_area = float(auto.get("bottle_surface_min_qwen_area_fraction", 0.01))
    max_qwen_area = float(auto.get("bottle_surface_max_qwen_area_fraction", 0.45))
    use_qwen_union = bool(auto.get("bottle_surface_union_valid_qwen", False)) and min_qwen_area <= qwen_area <= max_qwen_area
    region = _union_candidate_regions([surface_region, qwen_region], image_size) if use_qwen_union else surface_region
    return region, {
        "status": "valid",
        "model": "bottle_surface_guided",
        "qwen_region_xyxy": list(qwen_region),
        "qwen_area_fraction": qwen_area,
        "surface_region_xyxy": list(surface_region),
        "region_xyxy": list(region),
        "qwen_union_used": use_qwen_union,
    }


def _fractional_region(
    value: object,
    image_size: tuple[int, int],
    *,
    default: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    fractions = default
    if isinstance(value, list | tuple) and len(value) == 4:
        try:
            candidate = tuple(float(item) for item in value)
            if 0.0 <= candidate[0] < candidate[2] <= 1.0 and 0.0 <= candidate[1] < candidate[3] <= 1.0:
                fractions = candidate  # type: ignore[assignment]
        except (TypeError, ValueError):
            pass
    width, height = image_size
    return (
        max(0, min(width - 1, int(round(fractions[0] * width)))),
        max(0, min(height - 1, int(round(fractions[1] * height)))),
        max(1, min(width, int(round(fractions[2] * width)))),
        max(1, min(height, int(round(fractions[3] * height)))),
    )


def _region_area_fraction(region: tuple[int, int, int, int], image_size: tuple[int, int]) -> float:
    width, height = image_size
    area = max(0, region[2] - region[0]) * max(0, region[3] - region[1])
    return float(area / max(1, width * height))


def normal_guided_qwen_verify_region(
    *,
    image: Image.Image,
    normal_paths: list[Path],
    category: str,
    defect_type: str,
    description: str,
    extractor: QwenFeatureExtractor,
    auto: dict[str, Any],
    output_dir: Path | None = None,
    stem: str = "sample",
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    width, height = image.size
    full_region = (0, 0, width, height)
    heatmap, heat_params = patchcore_guided_heatmap(
        image,
        normal_paths,
        full_region,
        description,
        patch_size=int(auto.get("normal_guided_patch_size", auto.get("patchcore_guided_patch_size", 17))),
        stride=int(auto.get("normal_guided_stride", auto.get("patchcore_guided_stride", 6))),
        max_normals=int(auto.get("normal_guided_max_normals", auto.get("patchcore_guided_max_normals", 16))),
        max_memory_patches=int(auto.get("normal_guided_max_memory_patches", auto.get("patchcore_guided_max_memory_patches", 4096))),
    )
    candidates = _normal_guided_candidate_windows(heatmap, image.size, auto)
    if not candidates:
        raise ValueError("normal_guided_qwen_verify could not build candidate windows")

    montage = _normal_guided_candidate_montage(image, candidates)
    prompt = _normal_guided_verify_prompt(category, defect_type, description, candidates)
    result = extractor.analyze(montage, prompt)
    selected_ids = _parse_qwen_candidate_ids(str(result["text"]), len(candidates))
    fallback_used = False
    if not selected_ids:
        selected_ids = [int(candidates[0]["id"])]
        fallback_used = True
    max_selected = max(1, int(auto.get("normal_guided_max_selected_candidates", 3)))
    selected_ids = selected_ids[:max_selected]
    selected = [candidate for candidate in candidates if int(candidate["id"]) in set(selected_ids)]
    if not selected:
        selected = candidates[:1]
        selected_ids = [int(selected[0]["id"])]
        fallback_used = True

    region = _union_candidate_regions([tuple(candidate["region_xyxy"]) for candidate in selected], image.size)
    diagnostics: dict[str, Any] = {
        "status": "valid",
        "model": "normal_guided_qwen_verify",
        "region_xyxy": list(region),
        "candidate_ids_selected": selected_ids,
        "qwen_verifier_fallback_used": fallback_used,
        "qwen_verifier_text": str(result["text"]),
        "qwen_cache": result.get("localization_cache", {"enabled": False, "cache_hit": False}),
        "candidate_windows": _json_safe(candidates),
        "heatmap_parameters": heat_params,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = output_dir / f"{stem}_normal_guided_heatmap.png"
        montage_path = output_dir / f"{stem}_normal_guided_candidates.png"
        Image.fromarray(np.uint8(_normalize_float(heatmap) * 255), mode="L").save(heatmap_path)
        montage.save(montage_path)
        diagnostics["heatmap_path"] = str(heatmap_path)
        diagnostics["candidate_montage_path"] = str(montage_path)
    return region, diagnostics


def _normal_guided_candidate_windows(
    heatmap: np.ndarray,
    image_size: tuple[int, int],
    auto: dict[str, Any],
) -> list[dict[str, Any]]:
    width, height = image_size
    x_min_fraction, x_max_fraction = _fraction_pair(auto.get("normal_guided_x_range_fraction", [0.25, 0.75]), 0.25, 0.75)
    left = int(round(width * x_min_fraction))
    right = int(round(width * x_max_fraction))
    right = max(left + 1, min(width, right))
    left = max(0, min(left, right - 1))
    window_h = max(24, int(auto.get("normal_guided_window_height_px", 160)))
    stride = max(8, int(auto.get("normal_guided_window_stride_px", 48)))
    top_k = max(1, int(auto.get("normal_guided_candidate_count", 6)))
    padding_y = max(0, int(auto.get("normal_guided_window_padding_y_px", 12)))

    scored: list[dict[str, Any]] = []
    for top in range(0, max(1, height - window_h + 1), stride):
        bottom = min(height, top + window_h)
        crop = heatmap[top:bottom, left:right]
        if crop.size == 0:
            continue
        score = float(np.percentile(crop, 98.0) * 0.65 + np.mean(np.sort(crop.reshape(-1))[-max(1, crop.size // 20) :]) * 0.35)
        scored.append(
            {
                "id": len(scored) + 1,
                "region_xyxy": [left, max(0, top - padding_y), right, min(height, bottom + padding_y)],
                "score": score,
            }
        )
    if not scored:
        return []
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    min_center_gap = max(1, int(auto.get("normal_guided_min_center_gap_px", stride // 2)))
    for candidate in scored:
        y1, y2 = int(candidate["region_xyxy"][1]), int(candidate["region_xyxy"][3])
        center = (y1 + y2) / 2.0
        if any(abs(center - ((int(other["region_xyxy"][1]) + int(other["region_xyxy"][3])) / 2.0)) < min_center_gap for other in selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    selected.sort(key=lambda item: int(item["region_xyxy"][1]))
    for index, candidate in enumerate(selected, start=1):
        candidate["id"] = index
    return selected


def _normal_guided_candidate_montage(image: Image.Image, candidates: list[dict[str, Any]]) -> Image.Image:
    crop_w, crop_h = 384, 160
    label_h = 28
    cols = 2
    rows = int(np.ceil(len(candidates) / cols))
    montage = Image.new("RGB", (cols * crop_w, rows * (crop_h + label_h)), "white")
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for index, candidate in enumerate(candidates):
        col = index % cols
        row = index // cols
        x = col * crop_w
        y = row * (crop_h + label_h)
        region = tuple(int(value) for value in candidate["region_xyxy"])
        crop = image.crop(region).resize((crop_w, crop_h), Image.Resampling.BILINEAR)
        montage.paste(crop, (x, y + label_h))
        draw.rectangle((x, y, x + crop_w - 1, y + label_h - 1), fill=(235, 240, 248))
        draw.text((x + 8, y + 8), f"C{candidate['id']} score={float(candidate['score']):.3f} y={region[1]}-{region[3]}", fill=(20, 30, 45), font=font)
        draw.rectangle((x, y + label_h, x + crop_w - 1, y + label_h + crop_h - 1), outline=(255, 80, 40), width=3)
    return montage


def _normal_guided_verify_prompt(category: str, defect_type: str, description: str, candidates: list[dict[str, Any]]) -> str:
    candidate_summary = ", ".join(
        f"C{candidate['id']} y={candidate['region_xyxy'][1]}-{candidate['region_xyxy'][3]}" for candidate in candidates
    )
    return (
        f"You are verifying candidate crops from one industrial {category} image. "
        f"Target defect: {defect_type}. Description: {description!r}. "
        f"Candidate labels and original y ranges: {candidate_summary}. "
        "Select the candidate crops that contain the target defect. For zipper tooth defects, prefer crops with "
        "missing, broken, deformed, separated, or non-interlocking teeth; reject crops showing only normal repeated teeth. "
        "Return JSON only with keys \"candidate_ids\", \"confidence\", and \"evidence\". "
        "\"candidate_ids\" must be a list of candidate numbers, for example [2] or [1,3]. Select at most three."
    )


def _parse_qwen_candidate_ids(text: str, candidate_count: int) -> list[int]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    values: object = None
    if match:
        try:
            payload = json.loads(match.group(0))
            values = payload.get("candidate_ids") or payload.get("selected_candidate_ids") or payload.get("ids") or payload.get("candidate_id")
        except json.JSONDecodeError:
            values = None
    ids: list[int] = []
    if isinstance(values, int | float | str):
        values = [values]
    if isinstance(values, list | tuple):
        for value in values:
            try:
                candidate_id = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= candidate_id <= candidate_count and candidate_id not in ids:
                ids.append(candidate_id)
    if ids:
        return ids
    for value in re.findall(r"\bC?([1-9]\d*)\b", text):
        candidate_id = int(value)
        if 1 <= candidate_id <= candidate_count and candidate_id not in ids:
            ids.append(candidate_id)
    return ids


def _union_candidate_regions(regions: list[tuple[int, int, int, int]], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    return (
        max(0, min(region[0] for region in regions)),
        max(0, min(region[1] for region in regions)),
        min(width, max(region[2] for region in regions)),
        min(height, max(region[3] for region in regions)),
    )


def _fraction_pair(value: object, default_left: float, default_right: float) -> tuple[float, float]:
    if isinstance(value, list | tuple) and len(value) == 2:
        try:
            left = float(value[0])
            right = float(value[1])
            if 0.0 <= left < right <= 1.0:
                return left, right
        except (TypeError, ValueError):
            pass
    return default_left, default_right


def _coerce_box(raw_box: list[Any] | tuple[Any, ...], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    values = [float(value) for value in raw_box]
    width, height = image_size
    if max(values) <= 1.0:
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    elif max(values) <= 1000.0 and (values[2] > width * 1.2 or values[3] > height * 1.2):
        values = [values[0] / 1000.0 * width, values[1] / 1000.0 * height, values[2] / 1000.0 * width, values[3] / 1000.0 * height]
    return tuple(int(round(value)) for value in values)  # type: ignore[return-value]


def _recover_truncated_qwen_payload(text: str, image_size: tuple[int, int]) -> dict[str, Any] | None:
    bbox_match = re.search(r'"?bbox_xyxy"?\s*:\s*\[([^\]]+)\]', text)
    if not bbox_match:
        return None
    try:
        raw_box = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", bbox_match.group(1))]
    except ValueError:
        return None
    if len(raw_box) != 4:
        return None
    defect_match = re.search(r'"?defect_type"?\s*:\s*"([^"]*)"', text)
    confidence_match = re.search(r'"?confidence"?\s*:\s*(-?\d+(?:\.\d+)?)', text)
    evidence_match = re.search(r'"?evidence"?\s*:\s*"([^"]*)', text)
    return {
        "bbox_xyxy": _coerce_box(raw_box, image_size),
        "defect_type": defect_match.group(1) if defect_match else None,
        "confidence": _optional_float(confidence_match.group(1)) if confidence_match else None,
        "evidence": evidence_match.group(1) if evidence_match else None,
    }


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _defect_images(config: AppConfig) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    auto = config.data.get("auto_masks", {})
    max_per_target = int(auto.get("max_images_per_target", 0)) if isinstance(auto, dict) else 0
    for target in config.targets:
        test_dir = config.dataset_root / target.category / "test" / target.defect_type
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing defect image directory for auto-masks: {test_dir}")
        images = sorted(path for path in test_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"No defect images found for auto-masks: {test_dir}")
        if max_per_target > 0:
            images = images[:max_per_target]
        rows.extend((target.category, target.defect_type, path) for path in images)
    return rows


def _normal_index(config: AppConfig) -> dict[str, list[Path]]:
    categories = sorted({target.category for target in config.targets})
    index: dict[str, list[Path]] = {}
    for category in categories:
        clean_dir = config.dataset_root / category / "train" / "good"
        index[category] = sorted(
            path for path in clean_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if clean_dir.exists() else []
    return index


def _nearest_normal(image: Image.Image, candidates: list[Path], image_size: tuple[int, int]) -> Path | None:
    if not candidates:
        return None
    target = np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32)
    best: tuple[float, Path] | None = None
    for path in candidates:
        normal = np.asarray(
            Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        score = float(np.mean(np.abs(target - normal)))
        if best is None or score < best[0]:
            best = (score, path)
    return best[1] if best else None


def _combine_and_filter_masks(
    pixel_mask: Image.Image,
    residual_mask: Image.Image,
    defect_type: str,
    min_component_area: int,
) -> Image.Image:
    pixel = np.asarray(pixel_mask.convert("L"), dtype=np.uint8) > 0
    residual = np.asarray(residual_mask.convert("L"), dtype=np.uint8) > 0
    combined = residual if residual.sum() >= max(4, min_component_area) else pixel
    if pixel.any() and residual.any():
        combined = residual | (pixel & _dilate_bool(residual, radius=3))
    filtered = _filter_components(combined, defect_type, min_component_area)
    if filtered.sum() == 0:
        filtered = combined
    return Image.fromarray(filtered.astype(np.uint8) * 255).convert("L")


def _filter_components(
    mask: np.ndarray,
    defect_type: str,
    min_area: int,
    *,
    require_elongated: bool = True,
) -> np.ndarray:
    labels = _connected_components(mask)
    output = np.zeros_like(mask, dtype=bool)
    for component in labels:
        area = len(component)
        if area < min_area:
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        aspect = max(width / max(1, height), height / max(1, width))
        fill = area / max(1, width * height)
        if require_elongated and defect_type == "scratch" and area >= min_area * 3 and aspect < 1.8:
            continue
        if fill > 0.85 and area > min_area * 4:
            continue
        output[ys, xs] = True
    return output


def _filter_components_by_area(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    output = np.zeros_like(mask, dtype=bool)
    for component in _connected_components(mask):
        if len(component) < min_area:
            continue
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        output[ys, xs] = True
    return output


def polar_rim_residual_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    angle_bins: int,
    inner_radius_fraction: float,
    outer_radius_fraction: float,
    profile_percentile: float,
    support_percentile: float,
    min_sector_degrees: float,
    max_sector_degrees: float,
    close_radius: int,
    dilate_radius: int,
    min_component_area: int,
    max_box_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    import scipy.ndimage as ndimage

    width, height = image_size
    left, top, right, bottom = region
    if right <= left or bottom <= top:
        return Image.new("L", image_size, 0), {"polar_rim_reason": "empty_region"}

    center_x = (left + right - 1) / 2.0
    center_y = (top + bottom - 1) / 2.0
    base_radius = max(1.0, min(right - left, bottom - top) / 2.0)
    inner_radius = base_radius * max(0.05, min(0.95, float(inner_radius_fraction)))
    outer_radius = base_radius * max(inner_radius_fraction + 0.05, min(1.25, float(outer_radius_fraction)))
    yy, xx = np.mgrid[0:height, 0:width]
    dx = xx.astype(np.float32) - center_x
    dy = yy.astype(np.float32) - center_y
    radius = np.sqrt(dx * dx + dy * dy)
    angle = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
    annulus = (
        (radius >= inner_radius)
        & (radius <= outer_radius)
        & (xx >= left)
        & (xx < right)
        & (yy >= top)
        & (yy < bottom)
    )
    if not annulus.any():
        return Image.new("L", image_size, 0), {"polar_rim_reason": "empty_annulus"}

    heat = np.asarray(heatmap, dtype=np.float32)
    bins = max(36, int(angle_bins))
    angle_indices = np.floor(angle / 360.0 * bins).astype(np.int32) % bins
    profile = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        values = heat[annulus & (angle_indices == index)]
        if values.size:
            profile[index] = float(np.percentile(values, max(1.0, min(99.0, profile_percentile))))
    profile = ndimage.gaussian_filter1d(profile, sigma=max(1.0, bins / 180.0 * 2.0), mode="wrap")
    median = float(np.median(profile))
    mad = float(np.median(np.abs(profile - median)))
    profile_threshold = max(
        float(np.percentile(profile, 78.0)),
        median + max(0.80 * 1.4826 * mad, 0.15 * float(profile.max() - profile.min())),
    )
    peak_bin = int(np.argmax(profile))
    support = profile >= profile_threshold
    support_bins = {peak_bin}
    for direction in (-1, 1):
        cursor = peak_bin
        for _ in range(bins - 1):
            cursor = (cursor + direction) % bins
            if not support[cursor]:
                break
            support_bins.add(cursor)

    min_bins = max(1, int(round(max(1.0, min_sector_degrees) / 360.0 * bins)))
    max_bins = max(min_bins, int(round(max(min_sector_degrees, max_sector_degrees) / 360.0 * bins)))
    target_bins = max(min_bins, min(max_bins, len(support_bins)))
    half_left = target_bins // 2
    selected_bins = {(peak_bin + offset) % bins for offset in range(-half_left, target_bins - half_left)}
    sector = np.isin(angle_indices, list(selected_bins))
    sector_values = heat[annulus & sector]
    if sector_values.size == 0:
        return Image.new("L", image_size, 0), {"polar_rim_reason": "empty_sector"}

    heat_threshold = float(
        np.percentile(sector_values, max(1.0, min(99.0, float(support_percentile))))
    )
    active = annulus & sector & (heat >= heat_threshold - 1e-6)
    active = _morph_close(active, radius=max(0, int(close_radius)))
    active = _dilate_bool(active, radius=max(0, int(dilate_radius))) & annulus & sector
    active = _filter_components_by_area(active, min_area=max(2, int(min_component_area) // 2))
    if int(active.sum()) < max(3, int(min_component_area) // 2):
        active = annulus & sector & (heat >= float(np.percentile(sector_values, 40.0)))
        active = _morph_close(active, radius=max(0, int(close_radius)))
        active = _filter_components_by_area(active, min_area=max(2, int(min_component_area) // 3))

    region_area = max(1, (right - left) * (bottom - top))
    active_crop = active[top:bottom, left:right].copy()
    heat_crop = heat[top:bottom, left:right]
    active_crop = _cap_mask_by_heat(
        active_crop,
        heat_crop,
        max_area_fraction=max(0.005, min(0.40, float(max_box_fraction))),
    )
    active[:] = False
    active[top:bottom, left:right] = active_crop

    output = Image.fromarray(active.astype(np.uint8) * 255, mode="L")
    selected_degrees = target_bins / bins * 360.0
    return output, {
        "polar_rim_center_xy": [float(center_x), float(center_y)],
        "polar_rim_inner_radius": float(inner_radius),
        "polar_rim_outer_radius": float(outer_radius),
        "polar_rim_angle_bins": int(bins),
        "polar_rim_peak_degrees": float((peak_bin + 0.5) / bins * 360.0),
        "polar_rim_sector_degrees": float(selected_degrees),
        "polar_rim_profile_threshold": float(profile_threshold),
        "polar_rim_heat_threshold": float(heat_threshold),
        "polar_rim_pixels": int(active.sum()),
        "polar_rim_box_fraction": float(active.sum() / region_area),
    }


def repeated_chain_refiner_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    orientation_degrees: float = 90.0,
    percentile: float,
    corridor_width_fraction: float,
    max_window_height: int,
    min_window_height: int,
    close_radius: int,
    dilate_radius: int,
    min_component_area: int,
    max_components: int,
    max_box_fraction: float,
) -> tuple[Image.Image, dict[str, Any]]:
    import scipy.ndimage as ndimage

    left, top, right, bottom = region
    crop = np.asarray(heatmap[top:bottom, left:right], dtype=np.float32)
    if crop.size == 0:
        return Image.new("L", image_size, 0), {"repeated_chain_reason": "empty_crop"}

    crop_height, crop_width = crop.shape
    angle = float(orientation_degrees) % 180.0
    theta = np.deg2rad(angle)
    axis_vector = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
    perpendicular_vector = np.asarray([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    yy, xx = np.indices(crop.shape, dtype=np.float32)
    center_x = (crop_width - 1) / 2.0
    center_y = (crop_height - 1) / 2.0
    centered_x = xx - center_x
    centered_y = yy - center_y
    axis_coordinate = centered_x * axis_vector[0] + centered_y * axis_vector[1]
    perpendicular_coordinate = (
        centered_x * perpendicular_vector[0] + centered_y * perpendicular_vector[1]
    )
    axis_min = float(axis_coordinate.min())
    axis_max = float(axis_coordinate.max())
    perpendicular_span = float(perpendicular_coordinate.max() - perpendicular_coordinate.min() + 1.0)
    corridor_width = max(
        10.0,
        perpendicular_span * max(0.12, min(0.80, float(corridor_width_fraction))),
    )
    corridor = np.abs(perpendicular_coordinate) <= corridor_width / 2.0

    axis_count = max(2, int(np.ceil(axis_max - axis_min)) + 1)
    axis_indices = np.clip(np.floor(axis_coordinate - axis_min).astype(np.int32), 0, axis_count - 1)
    axis_signal = np.full(axis_count, np.nan, dtype=np.float32)
    for index in range(axis_count):
        values_at_axis = crop[corridor & (axis_indices == index)]
        if values_at_axis.size:
            axis_signal[index] = float(np.percentile(values_at_axis, 82.0))
    valid_signal = np.flatnonzero(np.isfinite(axis_signal))
    if valid_signal.size == 0:
        return Image.new("L", image_size, 0), {"repeated_chain_reason": "empty_axis_signal"}
    axis_signal = np.interp(
        np.arange(axis_count, dtype=np.float32),
        valid_signal.astype(np.float32),
        axis_signal[valid_signal],
    ).astype(np.float32)
    axis_signal = ndimage.gaussian_filter1d(axis_signal, sigma=3.0, mode="nearest")
    peak_axis = int(np.argmax(axis_signal))
    signal_threshold = max(
        float(np.percentile(axis_signal, 65.0)),
        float(np.median(axis_signal) + 0.35 * np.std(axis_signal)),
    )
    support_rows = axis_signal >= signal_threshold
    start = peak_axis
    end = peak_axis + 1
    while start > 0 and support_rows[start - 1]:
        start -= 1
    while end < axis_count and support_rows[end]:
        end += 1
    start -= 8
    end += 8

    min_height = max(12, min(axis_count, int(min_window_height)))
    max_height = max(min_height, min(axis_count, int(max_window_height)))
    current_height = end - start
    target_height = max(min_height, min(max_height, current_height))
    window_center = (start + end) // 2
    start = max(0, min(axis_count - target_height, window_center - target_height // 2))
    end = min(axis_count, start + target_height)

    window = (axis_indices >= start) & (axis_indices < end)
    fence = corridor & window
    values = crop[fence]
    if values.size == 0:
        return Image.new("L", image_size, 0), {"repeated_chain_reason": "empty_structure_fence"}

    threshold_percentile = max(55.0, min(99.5, float(percentile)))
    threshold = float(np.percentile(values, threshold_percentile))
    active = (crop >= threshold) & fence
    active = _morph_close(active, radius=max(0, int(close_radius)))
    active = _dilate_bool(active, radius=max(0, int(dilate_radius))) & fence
    active = _filter_components_by_area(active, min_area=max(3, int(min_component_area) // 2))
    if int(active.sum()) < max(4, int(min_component_area) // 2):
        threshold_percentile = max(55.0, threshold_percentile - 7.0)
        threshold = float(np.percentile(values, threshold_percentile))
        active = (crop >= threshold) & fence
        active = _morph_close(active, radius=max(0, int(close_radius)))
        active = _filter_components_by_area(active, min_area=max(2, int(min_component_area) // 3))

    components = _connected_components(active)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    for component in components:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        component_axis = axis_coordinate[ys, xs]
        component_perpendicular = perpendicular_coordinate[ys, xs]
        component_height = float(component_axis.max() - component_axis.min() + 1.0)
        component_width = float(component_perpendicular.max() - component_perpendicular.min() + 1.0)
        span_penalty = max(0.0, component_height / max(1, target_height) - 0.45)
        compactness = len(component) / max(1.0, component_height * component_width)
        component_score = float(crop[ys, xs].mean()) + min(0.18, compactness * 0.18) - 0.20 * span_penalty
        scored.append((component_score, component))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = np.zeros_like(active, dtype=bool)
    for _, component in scored[: max(1, int(max_components))]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
    selected = _cap_mask_by_heat(
        selected,
        crop,
        max_area_fraction=max(0.005, min(0.25, float(max_box_fraction))),
    )

    selected_values = crop[selected]
    background_values = crop[fence & ~selected]
    selected_mean = float(selected_values.mean()) if selected_values.size else 0.0
    background_mean = float(background_values.mean()) if background_values.size else float(values.mean())
    background_std = float(background_values.std()) if background_values.size else float(values.std())
    evidence_z = (selected_mean - background_mean) / max(1e-6, background_std)
    explanation_score = float(np.clip(evidence_z / 3.0, 0.0, 1.0))

    if selected.any():
        ys, xs = np.where(selected)
        selected_axis = axis_coordinate[ys, xs]
        selected_perpendicular = perpendicular_coordinate[ys, xs]
        vertical_span_fraction = float(
            (selected_axis.max() - selected_axis.min() + 1.0) / max(1.0, axis_max - axis_min + 1.0)
        )
        horizontal_center_error = float(
            abs(selected_perpendicular.mean()) / max(1.0, corridor_width / 2.0)
        )
    else:
        vertical_span_fraction = 1.0
        horizontal_center_error = 1.0
    span_score = float(np.clip(1.0 - max(0.0, vertical_span_fraction - 0.35) / 0.65, 0.0, 1.0))
    center_score = float(np.clip(1.0 - horizontal_center_error, 0.0, 1.0))
    structure_score = 0.70 * span_score + 0.30 * center_score

    output = Image.new("L", image_size, 0)
    output.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    corridor_ys, corridor_xs = np.where(corridor)
    fence_ys, fence_xs = np.where(fence)
    peak_coordinate = axis_min + peak_axis + 0.5
    peak_x = int(round(left + center_x + peak_coordinate * axis_vector[0]))
    peak_y = int(round(top + center_y + peak_coordinate * axis_vector[1]))
    return output, {
        "repeated_chain_percentile": float(threshold_percentile),
        "repeated_chain_threshold": float(threshold),
        "repeated_chain_orientation_degrees": angle,
        "repeated_chain_corridor_xyxy": [
            left + int(corridor_xs.min()),
            top + int(corridor_ys.min()),
            left + int(corridor_xs.max()) + 1,
            top + int(corridor_ys.max()) + 1,
        ],
        "repeated_chain_window_xyxy": [
            left + int(fence_xs.min()),
            top + int(fence_ys.min()),
            left + int(fence_xs.max()) + 1,
            top + int(fence_ys.max()) + 1,
        ],
        "repeated_chain_peak_xy": [peak_x, peak_y],
        "repeated_chain_peak_y": peak_y,
        "repeated_chain_pixels": int(selected.sum()),
        "repeated_chain_components_seen": len(components),
        "repeated_chain_components_kept": min(len(scored), max(1, int(max_components))),
        "repeated_chain_vertical_span_fraction": vertical_span_fraction,
        "repeated_chain_explanation_score": explanation_score,
        "repeated_chain_structure_score": float(structure_score),
    }


def zipper_fabric_border_layout_mask(
    heatmap: np.ndarray,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    percentile: float,
    close_radius: int,
    min_component_area: int,
    max_components: int,
    max_image_fraction: float,
    side_fractions: Any,
) -> tuple[Image.Image, dict[str, Any]]:
    width, height = image_size
    left, top, right, bottom = region
    heat_crop = heatmap[top:bottom, left:right]
    if heat_crop.size == 0:
        return Image.new("L", image_size, 0), {"zipper_fabric_layout_reason": "empty_crop"}

    fractions = side_fractions if isinstance(side_fractions, list | tuple) and len(side_fractions) == 4 else [0.08, 0.34, 0.66, 0.94]
    left_a, left_b, right_a, right_b = [float(value) for value in fractions]
    global_x = np.arange(left, right, dtype=np.float32) / max(1, width)
    side_x = ((global_x >= left_a) & (global_x <= left_b)) | ((global_x >= right_a) & (global_x <= right_b))
    fence = np.repeat(side_x[None, :], heat_crop.shape[0], axis=0)
    if not fence.any():
        fence = np.ones_like(heat_crop, dtype=bool)

    values = heat_crop[fence]
    if values.size == 0:
        return Image.new("L", image_size, 0), {"zipper_fabric_layout_reason": "empty_fence"}
    threshold_percentile = max(50.0, min(99.5, percentile))
    threshold = float(np.percentile(values, threshold_percentile))
    active = (heat_crop >= threshold) & fence
    active = _morph_close(active, radius=max(0, close_radius))
    active &= _dilate_bool(fence, radius=1)
    active = _filter_components_by_area(active, min_area=max(2, min_component_area // 2))

    if not active.any():
        threshold_percentile = max(50.0, threshold_percentile - 8.0)
        threshold = float(np.percentile(values, threshold_percentile))
        active = (heat_crop >= threshold) & fence
        active = _morph_close(active, radius=max(0, close_radius))
        active = _filter_components_by_area(active, min_area=max(2, min_component_area // 3))

    components = _connected_components(active)
    scored: list[tuple[float, list[tuple[int, int]]]] = []
    center_x = width / 2.0
    for component in components:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        global_component_x = xs + left
        width_c = int(xs.max() - xs.min() + 1)
        height_c = int(ys.max() - ys.min() + 1)
        aspect = max(width_c / max(1, height_c), height_c / max(1, width_c))
        side_distance = float(np.mean(np.abs(global_component_x - center_x)) / max(1.0, center_x))
        score = float(heat_crop[ys, xs].mean()) + min(0.20, len(component) / max(1, heat_crop.size) * 4.0)
        score += min(0.12, side_distance * 0.12) + min(0.06, aspect / 80.0)
        scored.append((score, component))
    scored.sort(key=lambda item: item[0], reverse=True)

    selected = np.zeros_like(active, dtype=bool)
    for _, component in scored[: max(1, max_components)]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        selected[ys, xs] = True
    if not selected.any():
        selected = active

    max_pixels = int(max(1, width * height * max(0.001, min(0.10, max_image_fraction))))
    selected = _cap_mask_by_heat(selected, heat_crop, max_area_fraction=max_pixels / max(1, selected.size))
    output = Image.new("L", image_size, 0)
    output.paste(Image.fromarray(selected.astype(np.uint8) * 255, mode="L"), (left, top))
    return output, {
        "zipper_fabric_layout_percentile": float(threshold_percentile),
        "zipper_fabric_layout_threshold": float(threshold),
        "zipper_fabric_layout_close_radius": int(close_radius),
        "zipper_fabric_layout_components_seen": len(components),
        "zipper_fabric_layout_components_kept": min(len(scored), max_components),
        "zipper_fabric_layout_pixels": int(selected.sum()),
        "zipper_fabric_layout_max_image_fraction": float(max_image_fraction),
        "zipper_fabric_layout_side_fractions": [left_a, left_b, right_a, right_b],
    }


def _keep_largest_components(mask: np.ndarray, *, max_components: int) -> np.ndarray:
    if max_components <= 0:
        return mask
    components = sorted(_connected_components(mask), key=len, reverse=True)
    output = np.zeros_like(mask, dtype=bool)
    for component in components[:max_components]:
        ys = np.asarray([point[0] for point in component])
        xs = np.asarray([point[1] for point in component])
        output[ys, xs] = True
    return output


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    return components


def _dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    for _ in range(radius):
        image = image.filter(ImageFilter.MaxFilter(size=3))
    return np.asarray(image, dtype=np.uint8) > 0


def _morph_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    size = max(3, radius * 2 + 1)
    if size % 2 == 0:
        size += 1
    image = image.filter(ImageFilter.MaxFilter(size=size))
    image = image.filter(ImageFilter.MinFilter(size=size))
    image = image.filter(ImageFilter.MedianFilter(size=3))
    return np.asarray(image, dtype=np.uint8) > 0


def _padded_bbox(mask: Image.Image, image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int] | None:
    bbox = mask.getbbox()
    if bbox is None:
        return None
    left, top, right, bottom = bbox
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _qc_mask(
    mask: Image.Image,
    qwen_region: tuple[int, int, int, int],
    shrunk_region: tuple[int, int, int, int],
    image_size: tuple[int, int],
    defect_type: str,
) -> dict[str, Any]:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    mask_area = int(arr.sum())
    image_area = image_size[0] * image_size[1]
    left, top, right, bottom = qwen_region
    qwen_area = max(1, (right - left) * (bottom - top))
    sx1, sy1, sx2, sy2 = shrunk_region
    shrink_area = max(1, (sx2 - sx1) * (sy2 - sy1))
    touches = int(arr[0].any()) + int(arr[-1].any()) + int(arr[:, 0].any()) + int(arr[:, -1].any())
    components = _connected_components(arr)
    status = "pass"
    reasons: list[str] = []
    if mask_area == 0:
        status = "reject"
        reasons.append("empty_mask")
    if mask_area > 0 and mask_area / qwen_area < 0.006:
        status = "warning"
        reasons.append("tiny_fraction_of_qwen_box")
    if mask_area / max(1, image_area) > 0.08:
        status = "warning"
        reasons.append("large_image_area")
    if mask_area / qwen_area > 0.45:
        status = "warning"
        reasons.append("large_fraction_of_qwen_box")
    if touches:
        status = "warning"
        reasons.append("touches_image_border")
    if len(components) > 24:
        status = "warning"
        reasons.append("many_components")
    if defect_type == "scratch" and shrink_area > 0:
        aspect = max((sx2 - sx1) / max(1, sy2 - sy1), (sy2 - sy1) / max(1, sx2 - sx1))
        if aspect < 1.5:
            status = "warning"
            reasons.append("scratch_not_elongated")
    return {
        "status": status,
        "reasons": reasons,
        "mask_area": mask_area,
        "mask_area_fraction": mask_area / max(1, image_area),
        "mask_to_qwen_box_fraction": mask_area / qwen_area,
        "component_count": len(components),
        "touching_border_count": touches,
        "shrunk_region_xyxy": shrunk_region,
    }


def _score_candidate(
    mask: Image.Image,
    image: Image.Image,
    qc: dict[str, Any],
    mode: str,
    defect_type: str,
    description: str,
) -> float:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if not arr.any():
        return -1.0
    score = 0.65
    if qc["status"] == "pass":
        score += 0.20
    elif qc["status"] == "warning":
        score -= 0.08
    else:
        score -= 0.45
    score -= min(0.25, 0.006 * max(0, int(qc.get("component_count", 0)) - 8))
    score -= min(0.20, 2.5 * max(0.0, float(qc.get("mask_area_fraction", 0.0)) - 0.02))
    score -= 0.08 * int(qc.get("touching_border_count", 0))
    if "tiny_fraction_of_qwen_box" in qc.get("reasons", []):
        score -= 0.24
    if defect_type == "scratch":
        score += _elongation_bonus(qc)
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.35 and aspect < 2.0:
            score -= 0.22
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.30:
            score -= 0.12
    score += _description_alignment(mask, image, description)
    if mode == "pixel":
        score += 0.05
    elif mode == "dinov2_fusion":
        score += 0.28
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.14
    elif mode == "dinov2_memory":
        score += 0.24
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.12
    elif mode == "normal_anomaly":
        score += 0.20
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.18
    elif mode == "foundation_anomaly_field":
        score += 0.27
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.20
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.46 and aspect < 1.6:
            score -= 0.14
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.24:
            score -= 0.12
        if int(qc.get("component_count", 0)) > 30:
            score -= 0.14
        if 1 <= int(qc.get("component_count", 0)) <= 18:
            score += 0.06
    elif mode == "nearest_normal_residual":
        score += 0.22
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.34
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.48 and aspect < 1.6:
            score -= 0.16
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.22:
            score -= 0.12
        if 1 <= int(qc.get("component_count", 0)) <= 18:
            score += 0.06
    elif mode == "patchcore_guided":
        score += 0.24
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.36
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.44 and aspect < 1.6:
            score -= 0.14
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.20:
            score -= 0.12
        if int(qc.get("component_count", 0)) > 24:
            score -= 0.12
        if 1 <= int(qc.get("component_count", 0)) <= 16:
            score += 0.06
    elif mode == "normal_residual_fusion":
        score += 0.24
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.38
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.46 and aspect < 1.6:
            score -= 0.14
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.22:
            score -= 0.12
        if int(qc.get("component_count", 0)) > 36:
            score -= 0.18
        if "many_components" in qc.get("reasons", []):
            score -= 0.10
        if 1 <= int(qc.get("component_count", 0)) <= 18:
            score += 0.06
    elif mode == "repeated_chain_refiner":
        score += 0.30
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.12:
            score -= 0.20
        if int(qc.get("component_count", 0)) > 10:
            score -= 0.12
    elif mode == "polar_rim_residual":
        score += 0.28
        if 0.01 <= float(qc.get("mask_to_qwen_box_fraction", 0.0)) <= 0.18:
            score += 0.08
        if int(qc.get("component_count", 0)) > 8:
            score -= 0.12
    elif mode == "scuff_cluster":
        score += 0.20
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.30
        if int(qc.get("component_count", 0)) > 36:
            score -= 0.12
    elif mode == "multi_scuff_fusion":
        score += 0.24
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.36
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.42 and aspect < 1.7:
            score -= 0.18
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.20:
            score -= 0.12
        if 1 <= int(qc.get("component_count", 0)) <= 16:
            score += 0.06
    elif mode == "support_constrained_fusion":
        score += 0.22
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.32
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.42 and aspect < 1.7:
            score -= 0.18
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.20:
            score -= 0.12
        if int(qc.get("component_count", 0)) > 36:
            score -= 0.10
        if 1 <= int(qc.get("component_count", 0)) <= 16:
            score += 0.06
    elif mode == "fft_texture_suppression":
        score += 0.18
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.34
        fill, aspect = _mask_fill_and_aspect(arr)
        if fill > 0.45 and aspect < 1.6:
            score -= 0.12
        if int(qc.get("component_count", 0)) > 28:
            score -= 0.10
    elif mode == "scratch_band_clean":
        score += 0.18
        if int(qc.get("component_count", 0)) <= 8:
            score += 0.08
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.16:
            score -= 0.10
    elif mode == "structure_tensor_ridge":
        score += 0.19
        if defect_type in {"scratch", "crack"}:
            score += 0.08
        if int(qc.get("component_count", 0)) <= 10:
            score += 0.05
        if _description_implies_patch(description) or _description_implies_multiple(description):
            score -= 0.10
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.16:
            score -= 0.12
    elif mode == "soft_patch":
        score += 0.16
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.32
        if 1 <= int(qc.get("component_count", 0)) <= 12:
            score += 0.05
    elif mode == "sam2_heatmap":
        score += 0.10
        if mask.size and qc.get("component_count", 0) <= 8:
            score += 0.04
        if defect_type == "scratch" and (_description_implies_multiple(description) or _description_implies_patch(description)):
            score -= 0.16
    elif mode == "sam_prompt_regularized":
        score += 0.14
        if int(qc.get("component_count", 0)) <= 10:
            score += 0.05
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.18:
            score -= 0.08
        if defect_type == "scratch" and (_description_implies_multiple(description) or _description_implies_patch(description)):
            score -= 0.12
    elif mode == "musc_mutual_score":
        score += 0.21
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score += 0.12
        if int(qc.get("component_count", 0)) > 24:
            score -= 0.10
        if float(qc.get("mask_to_qwen_box_fraction", 0.0)) > 0.22:
            score -= 0.10
    elif mode == "residual":
        score += 0.02
        if int(qc.get("component_count", 0)) > 24:
            score -= 0.10
    elif mode == "linear":
        score += 0.12
        if defect_type == "scratch":
            score += 0.08
        if _description_implies_patch(description) or _description_implies_multiple(description):
            score -= 0.18
    elif mode == "multi_linear":
        score += 0.14
        if defect_type == "scratch":
            score += 0.10
        if _description_implies_multiple(description):
            score += 0.06
        if _description_implies_patch(description):
            score -= 0.34
    elif mode == "procedural":
        score -= 0.05
        if _description_implies_multiple(description) or _description_implies_patch(description):
            score -= 0.16
    score = float(max(-1.0, min(1.0, score)))
    if qc.get("status") == "warning":
        score = min(score, float(auto_score_warning_cap(qc)))
    elif qc.get("status") == "pass":
        score = min(score, 0.985)
    return round(score, 4)


def auto_score_warning_cap(qc: dict[str, Any]) -> float:
    cap = 0.94
    reasons = set(qc.get("reasons", []))
    if "many_components" in reasons:
        cap -= 0.04
    if "large_fraction_of_qwen_box" in reasons or "large_image_area" in reasons:
        cap -= 0.04
    if "scratch_not_elongated" in reasons:
        cap -= 0.03
    return max(0.72, cap)


def _mask_fill_and_aspect(arr: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return 0.0, 0.0
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    bbox_area = max(1, width * height)
    aspect = max(width / max(1, height), height / max(1, width))
    return float(len(xs) / bbox_area), float(aspect)


def _description_implies_multiple(description: str) -> bool:
    hint = description.lower()
    return any(
        phrase in hint
        for phrase in (
            "many",
            "multiple",
            "several",
            "a lot",
            "lots",
            "group",
            "cluster",
            "patch",
            "scratched area",
            "scratches",
            "scuff",
            "scuffed",
            "abrasion",
            "scraped",
        )
    )


def _description_implies_patch(description: str) -> bool:
    hint = description.lower()
    return any(
        phrase in hint
        for phrase in (
            "patch",
            "area",
            "scuff",
            "scuffed",
            "abrasion",
            "scraped",
            "rubbed",
            "wear",
            "worn",
            "surface damage",
        )
    )


def _elongation_bonus(qc: dict[str, Any]) -> float:
    x1, y1, x2, y2 = qc.get("shrunk_region_xyxy", (0, 0, 1, 1))
    aspect = max((x2 - x1) / max(1, y2 - y1), (y2 - y1) / max(1, x2 - x1))
    if aspect >= 4.0:
        return 0.18
    if aspect >= 2.0:
        return 0.10
    return -0.08


def _description_alignment(mask: Image.Image, image: Image.Image, description: str) -> float:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if not arr.any():
        return -0.30
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ys, xs = np.where(arr)
    hint = description.lower()
    bonus = 0.0
    masked_mean = float(gray[arr].mean())
    outside_mean = float(gray[~arr].mean()) if (~arr).any() else masked_mean
    if any(word in hint for word in ("dark", "black", "brown", "shadow", "burn")):
        bonus += 0.10 if masked_mean < outside_mean else -0.08
    if any(word in hint for word in ("bright", "white", "light", "silver", "pale")):
        bonus += 0.10 if masked_mean > outside_mean else -0.08
    width, height = image.size
    cx = float(xs.mean()) / max(1, width)
    cy = float(ys.mean()) / max(1, height)
    if "left" in hint:
        bonus += 0.06 if cx < 0.45 else -0.04
    if "right" in hint:
        bonus += 0.06 if cx > 0.55 else -0.04
    if any(word in hint for word in ("top", "upper")):
        bonus += 0.06 if cy < 0.45 else -0.04
    if any(word in hint for word in ("bottom", "lower")):
        bonus += 0.06 if cy > 0.55 else -0.04
    if "center" in hint or "middle" in hint:
        bonus += 0.06 if 0.35 <= cx <= 0.65 and 0.35 <= cy <= 0.65 else -0.04
    angle = _mask_angle_degrees(xs, ys)
    if "horizontal" in hint:
        bonus += 0.12 if min(abs(angle), abs(180 - abs(angle))) < 25 else -0.08
    if "vertical" in hint:
        bonus += 0.12 if abs(abs(angle) - 90) < 25 else -0.08
    if "diagonal" in hint:
        bonus += 0.12 if 20 <= abs(angle) <= 70 or 110 <= abs(angle) <= 160 else -0.08
    return bonus


def _mask_angle_degrees(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    values, vectors = np.linalg.eigh(cov)
    vector = vectors[:, int(values.argmax())]
    return float(np.degrees(np.arctan2(vector[1], vector[0])))


def _dominant_texture_angle_degrees(
    image: Image.Image,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
) -> float | None:
    source = Image.open(normal_paths[0]).convert("L").resize(image.size, Image.Resampling.BILINEAR) if normal_paths else image.convert("L")
    arr = np.asarray(source.crop(region), dtype=np.float32)
    if arr.size < 9:
        return None
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, 1:] = arr[:, 1:] - arr[:, :-1]
    gy[1:, :] = arr[1:, :] - arr[:-1, :]
    magnitude = np.hypot(gx, gy)
    if float(magnitude.max()) <= 1e-6:
        return None
    threshold = float(np.percentile(magnitude, 90.0))
    active = magnitude >= threshold
    if int(active.sum()) < 8:
        return None
    # Gradient direction is perpendicular to line/texture direction.
    line_angles = np.arctan2(gy[active], gx[active]) + np.pi / 2.0
    weights = magnitude[active]
    sin2 = float(np.sum(np.sin(2.0 * line_angles) * weights))
    cos2 = float(np.sum(np.cos(2.0 * line_angles) * weights))
    return float(np.degrees(0.5 * np.arctan2(sin2, cos2)))


def _apply_gabor_suppression(image: Image.Image, dominant_angle_degrees: float) -> Image.Image:
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    theta = np.radians(dominant_angle_degrees)
    # Gabor responds to edges perpendicular to theta. 
    # To capture lines oriented at dominant_angle_degrees, theta should be perpendicular to them.
    filt_real, filt_imag = skimage.filters.gabor(arr, frequency=0.15, theta=theta + np.pi / 2.0)
    magnitude = np.hypot(filt_real, filt_imag)
    
    # Mute the grain by smoothing/blurring the high-magnitude Gabor responses.
    # Instead of subtracting, we blend the image with a heavily blurred version of itself
    # proportionally to the Gabor magnitude, effectively "erasing" the sharp grain.
    blurred = skimage.filters.gaussian(arr, sigma=3.0)
    blend_mask = np.clip(magnitude * 2.5, 0.0, 1.0)
    muted = arr * (1.0 - blend_mask) + blurred * blend_mask
    
    # If original image is RGB, apply the muting to the luminance channel and convert back
    if image.mode == "RGB":
        hsv = image.convert("HSV")
        h, s, v = hsv.split()
        v = Image.fromarray((np.clip(muted, 0, 1) * 255).astype(np.uint8), mode="L")
        return Image.merge("HSV", (h, s, v)).convert("RGB")
    return Image.fromarray((np.clip(muted, 0, 1) * 255).astype(np.uint8), mode="L")



def _angle_difference_degrees(a: float, b: float) -> float:
    difference = abs((a - b + 90.0) % 180.0 - 90.0)
    return float(difference)


def _prompt(category: str, defect_type: str, image_size: tuple[int, int], auto: dict[str, Any], description: str) -> str:
    style = str(auto.get("qwen_prompt_style", "defect_localization_json"))
    if style == "defect_localization_json_v2":
        return _prompt_v2(category, defect_type, image_size, description)
    if style != "defect_localization_json":
        raise ValueError(f"Unsupported qwen_prompt_style: {style}")
    width, height = image_size
    return (
        f"You are inspecting one industrial {category} image of size {width}x{height}. "
        f"Find the visible {defect_type} defect that matches this user description: {description!r}. "
        "Use the description as the primary localization cue. "
        "Return JSON only with keys "
        '"bbox_xyxy", "defect_type", "confidence", and "evidence". '
        '"bbox_xyxy" must be pixel coordinates [x1, y1, x2, y2] around only the defect, '
        "not the whole object and not a mask. Do not include markdown."
    )


def _prompt_v2(category: str, defect_type: str, image_size: tuple[int, int], description: str) -> str:
    width, height = image_size
    rules = _bbox_prompt_rules(category, defect_type)
    return (
        f"You are inspecting one industrial {category} image of size {width}x{height}. "
        f"Localize the visible {defect_type} defect matching this description: {description!r}. "
        "Return JSON only, with no markdown, using exactly these keys: "
        '"bbox_xyxy", "sub_boxes_xyxy", "defect_type", "confidence", "evidence", "self_check". '
        '"bbox_xyxy" must be pixel coordinates [x1, y1, x2, y2] around only the abnormal defect pixels. '
        '"sub_boxes_xyxy" must be a list of 0 to 4 tighter local boxes when the defect has multiple parts or when a '
        "single broad box would include normal repeated texture. If there is one local defect, put one tight local "
        "box in sub_boxes_xyxy and make bbox_xyxy the union of that local box. "
        "Do not select the whole object, the whole surface, the whole zipper chain, the background, or normal "
        "repeating texture. Prefer a slightly tight box that may miss weak boundary pixels over a broad box full "
        "of normal material. "
        f"{rules} "
        '"evidence" must briefly name the visual abnormality and its location. '
        '"self_check" must state why the box is local and defect-only.'
    )


def _bbox_prompt_rules(category: str, defect_type: str) -> str:
    key = f"{category}/{defect_type}".lower()
    if key == "bottle/broken_large":
        return (
            "Bottle rule: the target is a chipped or missing rim/glass section. Box the broken rim arc only; "
            "do not box the full circular opening, full bottle cap, full bottle body, reflection, or background."
        )
    if key == "bottle/broken_small":
        return (
            "Bottle rule: the target is a small local rim chip. The box should be small and local on the rim; "
            "do not box the entire circular rim, cap, highlight, or bottle body."
        )
    if key == "bottle/contamination":
        return (
            "Bottle rule: the target is foreign material, stain, or contamination on the bottle surface. "
            "Box the visible contamination patch only; ignore transparent glass, normal shadows, specular highlights, "
            "cap, central opening, and background."
        )
    if key == "zipper/broken_teeth":
        return (
            "Zipper tooth rule: the target is one tooth or a short group of missing, broken, or deformed teeth. "
            "The box must cover a short local segment of the central tooth chain, not the full vertical chain. "
            "For a 512px image, bbox height should usually be under 130px unless many adjacent teeth are damaged. "
            "Use sub_boxes_xyxy for individual tooth gaps."
        )
    if key == "zipper/split_teeth":
        return (
            "Zipper tooth rule: the target is the local place where left and right teeth separate or fail to interlock. "
            "The box must cover only the split segment, not the full vertical chain. For a 512px image, bbox height "
            "should usually be under 150px unless the split truly spans many teeth. Use sub_boxes_xyxy for separated "
            "local segments."
        )
    if key == "zipper/fabric_border":
        return (
            "Zipper fabric rule: the target is a damaged, frayed, cut, or irregular fabric border. Box only the "
            "abnormal border segment; do not box intact woven fabric, the central teeth, or the whole image edge."
        )
    return (
        "General industrial-defect rule: box only the abnormal visible defect and exclude normal object structure, "
        "normal texture, shadows, reflections, and background."
    )


def _load_descriptions(config: AppConfig, auto: dict[str, Any]) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    configured = auto.get("description_by_target", {})
    if isinstance(configured, dict):
        for key, value in configured.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    nested = str(nested_key)
                    descriptions[f"{key}/{nested}"] = str(nested_value)
                    if isinstance(nested_key, int):
                        descriptions[f"{key}/{nested_key:03d}"] = str(nested_value)
            else:
                descriptions[str(key)] = str(value)
    if auto.get("description"):
        descriptions["*"] = str(auto["description"])
    path_value = auto.get("descriptions_path")
    if path_value:
        path = config.resolve_path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"auto_masks.descriptions_path does not exist: {path}")
        descriptions.update(_read_description_file(path))
    return descriptions


def _read_description_file(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {_description_key(row): str(row["description"]) for row in rows}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "records" in payload and isinstance(payload["records"], list):
            return {_description_key(row): str(row["description"]) for row in payload["records"]}
        return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return {_description_key(row): str(row["description"]) for row in payload}
    raise ValueError(f"Unsupported description file format: {path}")


def _description_key(row: Any) -> str:
    if not isinstance(row, dict) or "description" not in row:
        raise ValueError(f"Description records must contain a description field: {row!r}")
    if row.get("image"):
        return str(row["image"])
    if row.get("image_path"):
        return str(row["image_path"])
    if row.get("category") and row.get("defect_type") and row.get("stem"):
        return f"{row['category']}/{row['defect_type']}/{row['stem']}"
    if row.get("category") and row.get("defect_type"):
        return f"{row['category']}/{row['defect_type']}"
    raise ValueError(f"Description record needs image, image_path, or category/defect_type: {row!r}")


def _description_for(
    config: AppConfig,
    descriptions: dict[str, str],
    category: str,
    defect_type: str,
    image_path: Path,
) -> str:
    relative = image_path.relative_to(config.dataset_root).as_posix()
    keys = (
        str(image_path),
        relative,
        f"{category}/{defect_type}/{image_path.name}",
        f"{category}/{defect_type}/{image_path.stem}",
        f"{category}/{defect_type}/{image_path.stem}.png",
        image_path.name,
        image_path.stem,
        f"{category}/{defect_type}",
        category,
        "*",
    )
    for key in keys:
        value = descriptions.get(key)
        if value:
            return value
    prompt_by_defect = config.data.get("generation", {}).get("prompt_by_defect", {})
    return str(prompt_by_defect.get(defect_type, f"a visible {defect_type} defect"))


def _auto_config(config: AppConfig) -> dict[str, Any]:
    auto = dict(config.data.get("auto_masks", {}))
    sd15 = dict(config.data.get("models", {}).get("sd15", {}))
    if sd15:
        auto.setdefault("sd15_base_model", sd15.get("base_model"))
        auto.setdefault("sd15_cache_dir", sd15.get("cache_dir"))
        auto.setdefault("sd15_local_files_only", sd15.get("local_files_only", True))
    return auto


def _auto_architecture(auto: dict[str, Any]) -> str:
    architecture = str(auto.get("architecture", "legacy_specialist"))
    if architecture not in {"legacy_specialist", "generic_evidence"}:
        raise ValueError("auto_masks.architecture must be legacy_specialist or generic_evidence")
    specialists = str(auto.get("specialists", "structural" if architecture == "legacy_specialist" else "disabled"))
    if specialists not in {"disabled", "structural"}:
        raise ValueError("auto_masks.specialists must be disabled or structural")
    return architecture


def _run_generic_mask_artifacts(
    *,
    config: AppConfig,
    auto: dict[str, Any],
    image: Image.Image,
    image_path: Path,
    normal_paths: list[Path],
    region: tuple[int, int, int, int],
    semantic_regions: list[tuple[int, int, int, int]],
    output_dir: Path,
    variant_dir: Path,
    artifact_stem: str,
    seed: int,
) -> dict[str, Any]:
    generic = auto.get("generic_evidence", {})
    if not isinstance(generic, dict):
        raise ValueError("auto_masks.generic_evidence must be a mapping")
    providers: list[Any] = []
    if bool(generic.get("dinov2_enabled", True)):
        providers.append(
            MultiScaleDinoProvider(
                model_id=str(auto.get("dinov2_model", "facebook/dinov2-small")),
                cache_dir=auto.get("dinov2_cache_dir"),
                device=str(auto.get("dinov2_device", "auto")),
                scales=tuple(int(value) for value in generic.get("dinov2_scales", [448, 672])),
                layers=tuple(int(value) for value in generic.get("dinov2_layers", [-4, -1])),
                max_normals=int(generic.get("max_normals", 16)),
                memory_stride=int(generic.get("dinov2_memory_stride", 1)),
                artifact_dir=config.output_dir / "auto_masks" / "evidence_cache" / "dinov2_multiscale",
            )
        )
    if bool(generic.get("registered_residual_enabled", True)):
        providers.append(
            RegisteredDinoResidualProvider(
                model_id=str(auto.get("dinov2_model", "facebook/dinov2-small")),
                cache_dir=auto.get("dinov2_cache_dir"),
                device=str(auto.get("dinov2_device", "auto")),
                scale=int(generic.get("registration_scale", 448)),
                max_normals=int(generic.get("max_normals", 16)),
                min_inlier_ratio=float(generic.get("registration_min_inlier_ratio", 0.25)),
                max_reprojection_error=float(generic.get("registration_max_reprojection_error", 8.0)),
                artifact_dir=config.output_dir / "auto_masks" / "evidence_cache" / "registered_normal_residual",
            )
        )
    if bool(generic.get("musc_enabled", True)):
        providers.append(
            FunctionalEvidenceProvider(
                "musc_mutual_rarity",
                lambda target, normals, target_region: musc_mutual_score_heatmap(
                    target,
                    normals,
                    target_region,
                    "object-agnostic anomalous appearance",
                    patch_size=int(generic.get("musc_patch_size", 17)),
                    stride=int(generic.get("musc_stride", 6)),
                    max_normals=int(generic.get("max_normals", 16)),
                    max_memory_patches=int(generic.get("musc_max_memory_patches", 4096)),
                    knn_k=int(generic.get("musc_knn_k", 5)),
                ),
                base_reliability=0.72,
                calibration_normals=int(generic.get("normal_calibration_holdouts", 3)),
                artifact_dir=config.output_dir / "auto_masks" / "evidence_cache" / "musc_mutual_rarity",
                cache_identity=fingerprint(
                    {
                        "patch_size": generic.get("musc_patch_size", 17),
                        "stride": generic.get("musc_stride", 6),
                        "max_normals": generic.get("max_normals", 16),
                        "max_memory_patches": generic.get("musc_max_memory_patches", 4096),
                        "knn_k": generic.get("musc_knn_k", 5),
                    }
                ),
            )
        )
    if bool(generic.get("texture_residual_enabled", True)):
        providers.append(
            FunctionalEvidenceProvider(
                "normal_texture_residual",
                lambda target, normals, target_region: normal_texture_anomaly_heatmap(
                    target,
                    normals,
                    target_region,
                    "object-agnostic anomalous appearance",
                    max_normals=int(generic.get("max_normals", 16)),
                ),
                base_reliability=0.58,
                calibration_normals=int(generic.get("normal_calibration_holdouts", 3)),
                artifact_dir=config.output_dir / "auto_masks" / "evidence_cache" / "normal_texture_residual",
                cache_identity=fingerprint({"max_normals": generic.get("max_normals", 16)}),
            )
        )
    if not providers:
        raise ValueError("generic_evidence requires at least one enabled evidence provider")
    context = AutoMaskContext(
        image_path=image_path,
        normal_paths=tuple(normal_paths[: int(generic.get("max_normals", 16))]),
        image_size=image.size,
        semantic_regions=tuple(dict.fromkeys(semantic_regions)),
        semantic_attributes=_generic_structure_attributes(image, region),
        foreground=_generic_foreground(image, normal_paths),
        cache_key=fingerprint(
            {
                "image": _content_identity(image_path),
                "normals": [_content_identity(path) for path in normal_paths[: int(generic.get("max_normals", 16))]],
                "semantic_regions": list(dict.fromkeys(semantic_regions)),
            }
        ),
    )
    selector_path = auto.get("selector_model_path")
    selector = GenericCandidateSelector(
        Path(str(selector_path)) if selector_path else None,
        edge_swap_margin=float(generic.get("edge_swap_margin", 0.02)),
    )
    specialist_masks: dict[str, np.ndarray] = {}
    if str(auto.get("specialists", "disabled")) == "structural":
        profile = str(context.semantic_attributes.get("structure_profile", "unknown"))
        attributes = {"scale": str(context.semantic_attributes.get("scale", "unknown"))}
        for mode in ("repeated_chain_refiner", "polar_rim_residual", "edge_border_layout"):
            if not specialist_applicability(mode, structure_profile=profile, attributes=attributes):
                continue
            try:
                candidate = _make_mask_candidate(
                    mode="zipper_fabric_border_layout" if mode == "edge_border_layout" else mode,
                    image=image,
                    category="generic",
                    defect_type="anomaly",
                    region=region,
                    output_dir=output_dir / "structural_specialists",
                    artifact_stem=f"{artifact_stem}_{mode}",
                    seed=seed,
                    auto=auto,
                    description="object-agnostic anomalous structure",
                    normal_paths=normal_paths,
                    feature_cache={},
                )
                specialist_masks[mode] = np.asarray(Image.open(candidate["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
            except Exception:
                continue

    def sam_refiner(fused: np.ndarray, proposal: np.ndarray, target_region: tuple[int, int, int, int]) -> list[np.ndarray]:
        if not bool(generic.get("sam2_enabled", True)):
            return []
        points, labels = heatmap_prompt_points(
            fused,
            target_region,
            positive_count=int(generic.get("sam_positive_points", 5)),
            negative_count=int(generic.get("sam_negative_points", 4)),
        )
        mask, _ = _predict_sam_mask(image, target_region, points, labels, auto)
        return [np.asarray(mask.convert("L"), dtype=np.uint8) > 0] if mask is not None else []

    artifacts = run_generic_evidence_pipeline(
        context,
        providers,
        output_dir=output_dir,
        variant_dir=variant_dir,
        artifact_stem=artifact_stem,
        selector=selector,
        sam_refiner=sam_refiner,
        edge_refine=bool(generic.get("edge_refine", True)),
        min_component_area=int(auto.get("min_component_area", 12)),
        specialist_masks=specialist_masks,
    )
    artifacts["parameters"]["specialists"] = str(auto.get("specialists", "disabled"))
    artifacts["parameters"]["active_structural_specialists"] = sorted(specialist_masks)
    return artifacts


def _generic_structure_attributes(image: Image.Image, region: tuple[int, int, int, int]) -> dict[str, Any]:
    left, top, right, bottom = region
    width, height = max(1, right - left), max(1, bottom - top)
    aspect = max(width / height, height / width)
    repeated = _repeated_texture_score(image)
    rim = _rim_geometry_score(image)
    boundary_contact = float(left <= 2 or top <= 2 or right >= image.width - 2 or bottom >= image.height - 2)
    profile = infer_structure_profile_from_measurements(
        repeated_texture_score=repeated,
        rim_geometry_score=rim,
        boundary_contact=boundary_contact,
        elongation=aspect,
    )
    area_fraction = width * height / max(1, image.width * image.height)
    scale = "micro" if area_fraction < 0.01 else "small" if area_fraction < 0.05 else "large"
    return {
        "structure_profile": profile,
        "scale": scale,
        "region_aspect": float(aspect),
        "region_area_fraction": float(area_fraction),
        "repeated_texture_score": float(repeated),
        "rim_geometry_score": float(rim),
    }


def _generic_foreground(image: Image.Image, normal_paths: list[Path]) -> np.ndarray:
    sources = normal_paths[:8]
    arrays = [
        np.asarray(Image.open(path).convert("RGB").resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32)
        for path in sources
    ]
    if not arrays:
        return np.ones((image.height, image.width), dtype=bool)
    consensus = np.median(np.stack(arrays, axis=0), axis=0)
    border = np.concatenate((consensus[0], consensus[-1], consensus[:, 0], consensus[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(consensus - background[None, None], axis=2)
    threshold = max(8.0, float(np.percentile(distance, 35.0)))
    foreground = distance >= threshold
    if float(foreground.mean()) < 0.05:
        return np.ones_like(foreground)
    return foreground


def _content_identity(path: Path) -> dict[str, Any]:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _localization_policy(auto: dict[str, Any], category: str, defect_type: str) -> str:
    configured = auto.get("localization_policy_by_target", {})
    if isinstance(configured, dict):
        for key in (f"{category}/{defect_type}", defect_type, category):
            value = configured.get(key)
            if value:
                return str(value)
    return str(auto.get("localization_policy", "qwen_bbox"))


def resolve_auto_localization_policy(
    *,
    auto: dict[str, Any],
    category: str,
    defect_type: str,
    description: str,
    image: Image.Image,
    qwen_region: tuple[int, int, int, int],
    localization: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    configured = _localization_policy(auto, category, defect_type)
    diagnostics = diagnose_localization_context(
        category=category,
        defect_type=defect_type,
        description=description,
        image=image,
        qwen_region=qwen_region,
        localization=localization or {},
    )
    diagnostics["configured_policy"] = configured
    if configured != "auto":
        diagnostics["router_mode"] = "configured"
        diagnostics["selected_policy"] = configured
        return configured, diagnostics

    text = diagnostics["text"]
    repeated = float(diagnostics["repeated_texture_score"])
    rim = float(diagnostics["rim_geometry_score"])
    qwen_sane = bool(diagnostics["qwen_bbox_sane"])
    repeated_threshold = float(auto.get("auto_policy_repeated_texture_threshold", 0.20))
    rim_threshold = float(auto.get("auto_policy_rim_geometry_threshold", 1.15))
    fabric_border_hint = any(
        term in text
        for term in (
            "fabric_border",
            "fabric border",
            "fabric",
            "border",
            "woven",
            "frayed",
            "thread",
            "cloth",
        )
    )
    tooth_defect_hint = defect_type in {"broken_teeth", "split_teeth"} or any(
        term in text
        for term in (
            "broken_teeth",
            "split_teeth",
            "broken tooth",
            "broken teeth",
            "missing tooth",
            "missing teeth",
            "split tooth",
            "split teeth",
            "separated tooth",
            "separated teeth",
            "interlock",
        )
    )

    if fabric_border_hint and not tooth_defect_hint:
        policy = "qwen_bbox"
    elif ("zipper" in text or "teeth" in text or "tooth" in text or "interlock" in text) and (
        repeated >= repeated_threshold or "zipper" in text or "teeth" in text
    ):
        if tooth_defect_hint:
            policy = "normal_guided_qwen_verify"
        else:
            policy = "qwen_bbox"
    elif ("bottle" in text or rim >= rim_threshold) and any(
        word in text for word in ("contamination", "stain", "foreign", "spot", "surface")
    ):
        policy = "bottle_surface_guided"
    elif ("bottle" in text or rim >= rim_threshold) and any(
        word in text for word in ("broken", "chip", "chipped", "missing", "rim", "glass")
    ):
        policy = "bottle_rim_guided_qwen"
    elif not qwen_sane and repeated >= repeated_threshold:
        policy = "normal_guided_qwen_verify"
    else:
        policy = "qwen_bbox"

    diagnostics["router_mode"] = "auto"
    diagnostics["selected_policy"] = policy
    diagnostics["reason"] = _auto_policy_reason(policy, diagnostics)
    return policy, diagnostics


def diagnose_localization_context(
    *,
    category: str,
    defect_type: str,
    description: str,
    image: Image.Image,
    qwen_region: tuple[int, int, int, int],
    localization: dict[str, Any],
) -> dict[str, Any]:
    width, height = image.size
    area_fraction = _region_area_fraction(qwen_region, image.size)
    box_width = max(1, qwen_region[2] - qwen_region[0])
    box_height = max(1, qwen_region[3] - qwen_region[1])
    aspect = max(box_width / box_height, box_height / box_width)
    status = str(localization.get("status", "unknown"))
    text = f"{category} {defect_type} {description}".lower()
    sane = status == "valid" and 0.0001 <= area_fraction <= 0.72 and aspect <= 25.0
    return {
        "text": text,
        "qwen_status": status,
        "qwen_region_xyxy": list(qwen_region),
        "qwen_area_fraction": round(area_fraction, 4),
        "qwen_aspect": round(float(aspect), 4),
        "qwen_bbox_sane": sane,
        "repeated_texture_score": round(_repeated_texture_score(image), 4),
        "rim_geometry_score": round(_rim_geometry_score(image), 4),
    }


def _auto_policy_reason(policy: str, diagnostics: dict[str, Any]) -> str:
    if policy == "normal_guided_qwen_verify":
        return (
            "dense repeated texture or tooth/interlock language detected; use normal-memory candidate windows "
            "and Qwen semantic verification"
        )
    if policy == "bottle_rim_guided_qwen":
        return "rim/chip/broken-glass language or circular geometry detected; use high-recall bottle rim prior"
    if policy == "bottle_surface_guided":
        return "surface contamination language or bottle-like geometry detected; use high-recall bottle surface prior"
    return "Qwen bbox appears sufficient or no specialized zero-shot policy matched"


def _repeated_texture_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    x1, x2 = 32, 96
    central_vertical = gx[:, x1:x2].mean()
    central_horizontal = gy[:, x1:x2].mean()
    global_edge = (gx.mean() + gy.mean()) / 2.0 + 1e-6
    directional = central_vertical / (central_horizontal + 1e-6)
    score = (central_vertical / global_edge) * min(1.0, directional / 1.6)
    return float(max(0.0, min(2.0, score)) / 2.0)


def _rim_geometry_score(image: Image.Image) -> float:
    edge = np.asarray(image.convert("L").resize((160, 160), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    edge = skimage.filters.sobel(edge)
    yy, xx = np.indices(edge.shape)
    cx = (edge.shape[1] - 1) / 2.0
    cy = (edge.shape[0] - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    norm_radius = radius / max(1.0, min(edge.shape) / 2.0)
    annulus = (norm_radius >= 0.34) & (norm_radius <= 0.78)
    center = norm_radius < 0.28
    annulus_edge = float(edge[annulus].mean()) if annulus.any() else 0.0
    center_edge = float(edge[center].mean()) if center.any() else 0.0
    global_edge = float(edge.mean()) + 1e-6
    return float(max(0.0, min(3.0, (annulus_edge / global_edge) + 0.25 * (annulus_edge / (center_edge + 1e-6)))))


def _resolve_auto_paths(config: AppConfig, auto: dict[str, Any]) -> None:
    for key in ("sam2_checkpoint", "sam_checkpoint"):
        value = auto.get(key)
        if value:
            auto[key] = str(config.resolve_path(value))
    for key in (
        "qwen_cache_dir",
        "qwen_localization_cache_dir",
        "dinov2_cache_dir",
        "delta_deno_cache_dir",
        "sd15_cache_dir",
        "selector_model_path",
    ):
        value = auto.get(key)
        if value:
            auto[key] = str(config.resolve_path(value))


def _fallback_qwen(config: AppConfig, key: str, default: Any) -> Any:
    return config.data.get("auto_masks", {}).get(key, config.data.get("phase2", {}).get(key, default))


def _qwen_availability(config: AppConfig, auto: dict[str, Any]):
    return qwen_availability(
        model_id=str(auto.get("qwen_model", _fallback_qwen(config, "qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct"))),
        cache_dir=auto.get("qwen_cache_dir", _fallback_qwen(config, "qwen_cache_dir", None)),
        local_files_only=bool(auto.get("qwen_local_files_only", _fallback_qwen(config, "qwen_local_files_only", True))),
        min_free_gib=float(auto.get("qwen_min_free_gib", _fallback_qwen(config, "qwen_min_free_gib", 30.0))),
    )


def _auto_fingerprint(config: AppConfig) -> str:
    return fingerprint({"split_spec_fingerprint": config.split_spec_fingerprint(), "auto_masks": _json_safe(_auto_config(config))})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_summary(path: Path, rows: list[AutoMaskRecord], metadata_path: Path, auto: dict[str, Any]) -> None:
    mask_pixels: list[int] = []
    for row in rows:
        mask = np.asarray(Image.open(row.mask_path).convert("L"), dtype=np.uint8) > 0
        mask_pixels.append(int(mask.sum()))
    lines = [
        "# Auto Mask Report: qwen",
        "",
        "Mask truth source: `qwen_auto_refined_masks`",
        f"Metadata: `{metadata_path}`",
        f"Records written: {len(rows)}",
        f"Overwrite enabled: `{bool(auto.get('overwrite', False))}`",
        f"Mask refinement: `{auto.get('mask_refinement', 'pixel')}`",
    ]
    if mask_pixels:
        lines.extend(
            [
                f"Minimum refined mask pixels: {min(mask_pixels)}",
                f"Mean refined mask pixels: {sum(mask_pixels) / len(mask_pixels):.1f}",
                f"Maximum refined mask pixels: {max(mask_pixels)}",
            ]
        )
    qc = _qc_counts(rows)
    lines.extend(
        [
            "",
            "| QC status | Count |",
            "| --- | ---: |",
        ]
    )
    for status in ("pass", "warning", "reject"):
        lines.append(f"| {status} | {qc.get(status, 0)} |")
    if rows:
        lines.extend(
            [
                "",
                "| Image | Selected refinement | Morphology | QC | Eval pixels | Uncertainty pixels | Training variant |",
                "| --- | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            eval_pixels = 0
            uncertainty_pixels = 0
            if row.eval_mask_path and Path(row.eval_mask_path).exists():
                eval_pixels = int((np.asarray(Image.open(row.eval_mask_path).convert("L"), dtype=np.uint8) > 0).sum())
            if row.uncertainty_mask_path and Path(row.uncertainty_mask_path).exists():
                uncertainty_pixels = int(
                    (np.asarray(Image.open(row.uncertainty_mask_path).convert("L"), dtype=np.uint8) > 0).sum()
                )
            training_variant = row.settings.get("label_policy", {}).get("adapter_training_mask_variant", "")
            lines.append(
                f"| {Path(row.image_path).name} | {row.settings.get('selected_refinement')} | "
                f"{row.settings.get('scratch_morphology_class')} | {row.settings.get('qc', {}).get('status')} | "
                f"{eval_pixels} | {uncertainty_pixels} | {training_variant} |"
            )
    lines.append("")
    lines.append(
        "These masks are automatic pseudo-labels bootstrapped from Qwen boxes and candidate-mask fusion. "
        "Uncertainty maps mark pixels where candidate methods disagree or where the pseudo-label boundary is weak; "
        "prefer human/official masks for final claims."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_contact_sheet(path: Path, rows: list[AutoMaskRecord]) -> None:
    if not rows:
        return
    def training_used_overlay(row: AutoMaskRecord) -> str | None:
        if not row.training_mask_path:
            return None
        for name, mask_path in row.mask_variant_paths.items():
            if str(mask_path) == str(row.training_mask_path):
                return row.mask_variant_overlay_paths.get(name) or row.training_mask_path
        return row.training_mask_path

    columns: list[tuple[str, Any]] = [
        ("selected", lambda row: row.overlay_path),
        ("eval_tight", lambda row: row.mask_variant_overlay_paths.get("eval_tight")),
        ("training_used", training_used_overlay),
        ("training_soft", lambda row: row.mask_variant_overlay_paths.get("training_soft")),
        ("uncertainty", lambda row: row.mask_variant_overlay_paths.get("uncertainty_map")),
        ("training_medium", lambda row: row.mask_variant_overlay_paths.get("training_medium")),
        ("inpaint_soft", lambda row: row.mask_variant_overlay_paths.get("inpaint_soft")),
    ]
    thumb_w, thumb_h, label_h = 260, 260, 24
    sheet = Image.new("RGB", (thumb_w * len(columns), (thumb_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        selected = str(row.settings.get("selected_refinement", "unknown"))
        for col_index, (label, getter) in enumerate(columns):
            value = getter(row)
            if not value or not Path(value).exists():
                continue
            image = Image.open(value).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = col_index * thumb_w
            y = row_index * (thumb_h + label_h)
            sheet.paste(image, (x, y + label_h))
            text = f"{Path(row.image_path).name} {label}"
            if label == "selected":
                text += f" ({selected})"
            draw.text((x + 4, y + 4), text, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _write_candidate_comparison_sheet(path: Path, rows: list[AutoMaskRecord]) -> None:
    if not rows:
        return
    if any(str(row.settings.get("auto_mask_architecture", "")).startswith("generic_evidence") for row in rows):
        _write_generic_candidate_comparison_sheet(path, rows)
        return
    candidate_order = [
        "selected",
        "normal_anomaly",
        "nearest_normal_residual",
        "patchcore_guided",
        "soft_patch",
        "scuff_cluster",
        "multi_scuff_fusion",
        "support_constrained_fusion",
        "normal_residual_fusion",
        "fft_texture_suppression",
        "structure_tensor_ridge",
        "scratch_band_clean",
        "multi_linear",
        "linear",
    ]
    thumb_w, thumb_h, label_h = 220, 220, 24
    sheet = Image.new("RGB", (thumb_w * len(candidate_order), (thumb_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        stem_prefix = f"{row.category}_{row.defect_type}_{Path(row.image_path).stem}"
        mask_dir = Path(row.refined_mask_path).parent
        selected = str(row.settings.get("selected_refinement", "unknown"))
        for col_index, mode in enumerate(candidate_order):
            if mode == "selected":
                mask_path = Path(row.refined_mask_path)
            else:
                mask_path = mask_dir / f"{stem_prefix}_{mode}_refined.png"
            if not mask_path.exists():
                continue
            image = Image.open(row.image_path).convert("RGB")
            overlay = _overlay_for_sheet(image, Path(row.box_mask_path), mask_path)
            overlay = overlay.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = col_index * thumb_w
            y = row_index * (thumb_h + label_h)
            sheet.paste(overlay, (x, y + label_h))
            label = f"{Path(row.image_path).name} {mode}"
            if mode == "selected":
                label += f" ({selected})"
            draw.text((x + 4, y + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _write_generic_candidate_comparison_sheet(path: Path, rows: list[AutoMaskRecord]) -> None:
    thumb_w, thumb_h, label_h = 220, 220, 40
    column_count = 6
    sheet = Image.new("RGB", (thumb_w * column_count, (thumb_h + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        image = Image.open(row.image_path).convert("RGB")
        for col_index, (slot, mode, mask_path) in enumerate(_generic_candidate_comparison_entries(row, column_count)):
            overlay = _overlay_for_sheet(image, Path(row.box_mask_path), mask_path)
            overlay = overlay.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = col_index * thumb_w
            y = row_index * (thumb_h + label_h)
            sheet.paste(overlay, (x, y + label_h))
            draw.text((x + 4, y + 3), f"{Path(row.image_path).name} {slot}", fill="black")
            draw.text((x + 4, y + 19), mode[:34], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _generic_candidate_comparison_entries(
    row: AutoMaskRecord,
    limit: int = 6,
) -> list[tuple[str, str, Path]]:
    candidate_paths = {
        str(mode): Path(value)
        for mode, value in row.settings.get("candidate_refined_paths", {}).items()
        if value and Path(value).exists()
    }
    policy_scores = {str(mode): float(value) for mode, value in row.settings.get("policy_scores", {}).items()}
    raw_scores = {str(mode): float(value) for mode, value in row.settings.get("candidate_scores", {}).items()}
    selected = str(row.settings.get("selected_refinement", "unknown"))
    entries: list[tuple[str, str, Path]] = []
    used: set[str] = set()

    def add(slot: str, mode: str | None, fallback_path: Path | None = None) -> None:
        if not mode or mode in used or len(entries) >= limit:
            return
        candidate_path = candidate_paths.get(mode, fallback_path)
        if candidate_path is None or not candidate_path.exists():
            return
        entries.append((slot, mode, candidate_path))
        used.add(mode)

    def best(modes: list[str], scores: dict[str, float]) -> str | None:
        available = [mode for mode in modes if mode in candidate_paths and mode not in used]
        return max(available, key=lambda mode: (scores.get(mode, float("-inf")), mode)) if available else None

    add("selected", selected, Path(row.refined_mask_path))
    ranked_modes = sorted(candidate_paths, key=lambda mode: (policy_scores.get(mode, float("-inf")), mode), reverse=True)
    add("selector runner-up", best(ranked_modes, policy_scores))
    add("best unused edge", best([mode for mode in candidate_paths if mode.startswith("edge_")], policy_scores))
    add("best unused SAM2", best([mode for mode in candidate_paths if mode.startswith("sam2_")], policy_scores))
    add("highest unused raw", best(list(candidate_paths), raw_scores))
    while len(entries) < limit:
        mode = best(ranked_modes, policy_scores)
        if mode is None:
            break
        add("ranked alternative", mode)
    return entries


def _overlay_for_sheet(image: Image.Image, box_path: Path, mask_path: Path) -> Image.Image:
    base = image.convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    box = Image.open(box_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    red = Image.new("RGBA", image.size, (255, 0, 0, 100))
    yellow = Image.new("RGBA", image.size, (255, 220, 0, 170))
    overlay = Image.composite(red, overlay, mask)
    box_edges = box.filter(ImageFilter.FIND_EDGES).point(lambda value: 255 if value > 0 else 0)
    overlay = Image.composite(yellow, overlay, box_edges)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _qc_counts(rows: list[AutoMaskRecord]) -> dict[str, int]:
    counts = {"pass": 0, "warning": 0, "reject": 0}
    for row in rows:
        status = str(row.settings.get("qc", {}).get("status", "warning"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _stable_seed(value: str) -> int:
    return int(fingerprint(value)[:8], 16)
