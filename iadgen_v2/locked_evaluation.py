from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from iadgen_v2.config import AppConfig
from iadgen_v2.governance import (
    governance_settings,
    validate_frozen_architecture,
    validate_governance_config,
    validate_sealed_runtime_manifest,
)
from iadgen_v2.records import write_json
from iadgen_v2.segmentation import segmentation_metrics


def run_locked_evaluation(config: AppConfig, *, runtime_manifest: Path, reference_manifest: Path) -> Path:
    governance = validate_governance_config(config)
    if not governance["enabled"]:
        raise ValueError("locked-evaluate requires research_governance.enabled=true")
    target_categories = {target.category for target in config.targets}
    locked = set(governance["locked_categories"])
    if not target_categories or not target_categories <= locked:
        raise ValueError("locked-evaluate config targets must contain locked categories only")
    runtime_manifest = runtime_manifest.resolve()
    reference_manifest = reference_manifest.resolve()
    official_roots = [Path(value).resolve() for value in governance["official_mask_roots"]]
    if not any(_is_within(reference_manifest, root) for root in official_roots):
        raise ValueError("reference manifest must live under a configured locked official-mask root")
    if any(_is_within(runtime_manifest, root) for root in official_roots):
        raise ValueError("runtime manifest must remain outside locked official-mask roots")
    auto = config.data.get("auto_masks", {})
    if isinstance(auto, dict) and str(auto.get("architecture", "legacy_specialist")) == "generic_evidence":
        validate_frozen_architecture(config, summary=governance)
        validate_sealed_runtime_manifest(config, runtime_manifest)
    architecture = str(governance["architecture_tag"])
    output_dir = config.report_dir / "locked_evaluation" / architecture
    seal_path = output_dir / "seal.json"
    if seal_path.exists() and not bool(governance_settings(config).get("allow_locked_evaluation_rerun", False)):
        raise RuntimeError(f"Locked evaluation is already sealed for {architecture}: {seal_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    attempt = {
        "status": "started",
        "architecture_tag": architecture,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_manifest": _file_record(runtime_manifest),
        "reference_manifest": _file_record(reference_manifest),
        "rerun_forbidden": not bool(governance_settings(config).get("allow_locked_evaluation_rerun", False)),
    }
    # This is deliberately written before resolving or opening any official
    # mask. An interrupted attempt remains sealed and auditable.
    write_json(seal_path, attempt)
    runtime_rows = _read_rows(runtime_manifest)
    if not runtime_rows:
        raise ValueError("Locked runtime manifest contains no prediction rows")
    references = _reference_index(reference_manifest, official_roots)
    metrics: list[dict[str, Any]] = []
    for row in runtime_rows:
        key = _sample_key(row)
        reference = references.get(key)
        if reference is None:
            raise ValueError(f"No locked reference mask for runtime sample {key}")
        truth = np.asarray(Image.open(reference).convert("L"), dtype=np.uint8) > 0
        eval_path = Path(str(row.get("eval_mask_path") or row.get("refined_mask_path"))).resolve()
        if any(_is_within(eval_path, root) for root in official_roots):
            raise ValueError(f"Runtime prediction illegally resolves into official masks: {eval_path}")
        prediction = np.asarray(Image.open(eval_path).convert("L").resize((truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST), dtype=np.uint8) > 0
        score = _score_map(row, prediction, truth.shape)
        intersection = int((prediction & truth).sum())
        predicted = int(prediction.sum())
        positive = int(truth.sum())
        union = int((prediction | truth).sum())
        region = tuple(int(value) for value in row.get("region_xyxy", [0, 0, truth.shape[1], truth.shape[0]]))
        region_mask = np.zeros_like(truth, dtype=bool)
        region_mask[max(0, region[1]) : min(truth.shape[0], region[3]), max(0, region[0]) : min(truth.shape[1], region[2])] = True
        ranking = segmentation_metrics(score, truth, threshold=0.5)
        settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
        decision = settings.get("selection_decision", {}) if isinstance(settings.get("selection_decision"), dict) else {}
        metrics.append(
            {
                "sample_id": "/".join(key),
                "category": key[0],
                "defect_type": key[1],
                "dice": 2 * intersection / max(1, predicted + positive),
                "iou": intersection / max(1, union),
                "precision": intersection / max(1, predicted),
                "recall": intersection / max(1, positive),
                "predicted_positive_rate": predicted / prediction.size,
                "search_region_recall": int((region_mask & truth).sum()) / max(1, positive),
                "pixel_auroc": ranking["pixel_auroc"],
                "pixel_ap": float(average_precision_score(truth.reshape(-1), score.reshape(-1))),
                "aupro": ranking["aupro"],
                "disposition": str(decision.get("disposition", "unknown")),
                "confidence": float(decision.get("confidence", math.nan)),
            }
        )
    csv_path = output_dir / "locked_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    report_path = output_dir / "locked_evaluation_report.md"
    report_path.write_text(_report(metrics, architecture), encoding="utf-8")
    write_json(
        seal_path,
        {
            "status": "complete",
            "architecture_tag": architecture,
            "started_at_utc": attempt["started_at_utc"],
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_manifest": _file_record(runtime_manifest),
            "reference_manifest": _file_record(reference_manifest),
            "metrics_csv": _file_record(csv_path),
            "report": _file_record(report_path),
            "rerun_forbidden": not bool(governance_settings(config).get("allow_locked_evaluation_rerun", False)),
        },
    )
    return report_path


def _score_map(row: dict[str, Any], prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
    parameters = settings.get("mask_parameters", {}) if isinstance(settings.get("mask_parameters"), dict) else {}
    path_value = parameters.get("fused_evidence_path")
    if path_value and Path(str(path_value)).exists():
        return np.asarray(Image.open(str(path_value)).convert("L").resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return prediction.astype(np.float32)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must contain a row list: {path}")
    return [row for row in rows if isinstance(row, dict)]


def _reference_index(path: Path, official_roots: list[Path]) -> dict[tuple[str, str, str], Path]:
    rows = _read_rows(path)
    index = {}
    for row in rows:
        key = _sample_key(row)
        mask = Path(str(row.get("official_mask_path") or row.get("mask_path"))).resolve()
        if not any(_is_within(mask, root) for root in official_roots):
            raise ValueError(f"Official mask is outside isolated roots: {mask}")
        index[key] = mask
    return index


def _sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    category = str(row.get("category", ""))
    defect = str(row.get("defect_type", ""))
    image_value = str(row.get("image_path") or row.get("image") or row.get("sample_id") or "")
    stem = Path(image_value.split("/")[-1]).stem.removesuffix("_mask")
    if not category or not defect or not stem:
        raise ValueError(f"Manifest row lacks category/defect/image identity: {row}")
    return category, defect, stem


def _report(rows: list[dict[str, Any]], architecture: str) -> str:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    lines = [f"# Locked Evaluation: {architecture}", "", "Official masks were opened only by `locked-evaluate` after runtime outputs were sealed.", "", "| Category | Images | Dice | IoU | Precision | Recall | Search Recall | AUPRO | Pixel AP |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category, category_rows in sorted(groups.items()):
        mean = lambda key: float(np.nanmean([float(row[key]) for row in category_rows]))
        lines.append(f"| {category} | {len(category_rows)} | `{mean('dice'):.4f}` | `{mean('iou'):.4f}` | `{mean('precision'):.4f}` | `{mean('recall'):.4f}` | `{mean('search_region_recall'):.4f}` | `{mean('aupro'):.4f}` | `{mean('pixel_ap'):.4f}` |")
    macro_dice = float(np.mean([np.mean([float(row["dice"]) for row in category_rows]) for category_rows in groups.values()]))
    accepted = [row for row in rows if row["disposition"] != "needs_review"]
    lines.extend(["", f"Category-macro Dice: `{macro_dice:.4f}`", f"Accepted-mask coverage: `{len(accepted) / max(1, len(rows)):.4f}`", ""])
    return "\n".join(lines)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest}
