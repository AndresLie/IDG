from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from iadgen_v2.config import AppConfig
from iadgen_v2.governance import validate_governance_config
from iadgen_v2.records import write_json
from iadgen_v2.segmentation import segmentation_metrics


def run_development_evaluation(config: AppConfig, metadata_paths: list[Path] | None = None) -> Path:
    paths = metadata_paths or sorted((config.output_dir / "auto_masks").glob("*/metadata.jsonl"))
    if not paths:
        raise FileNotFoundError("No auto-mask metadata found for development evaluation")
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in _read_jsonl(path):
            rows.append(_evaluate_row(config, row, path))
    output_dir = config.report_dir / "development_generalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "development_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "development_generalization_report.md"
    report_path.write_text(_write_report(rows), encoding="utf-8")
    _write_gate(config, rows, output_dir / "generic_mask_gate.json")
    return report_path


def _write_gate(config: AppConfig, rows: list[dict[str, Any]], path: Path) -> None:
    generic_rows = [row for row in rows if row["architecture"] == "generic_evidence"]
    governance = validate_governance_config(config)
    if not generic_rows:
        write_json(path, {"go": False, "architecture_tag": governance["architecture_tag"], "reason": "no_generic_evidence_rows"})
        return
    categories = sorted({str(row["category"]) for row in generic_rows})
    category_dice = [float(np.mean([float(row["dice"]) for row in generic_rows if row["category"] == category])) for category in categories]
    macro_dice = float(np.mean(category_dice))
    worst_dice = float(min(category_dice))
    search_recall = float(np.mean([float(row["search_region_recall"]) for row in generic_rows]))
    accepted_coverage = sum(row["disposition"] != "needs_review" for row in generic_rows) / len(generic_rows)
    selector_regret = float(np.mean([float(row["selector_regret"]) for row in generic_rows]))
    checks = {
        "macro_dice": macro_dice >= 0.55,
        "worst_category_dice": worst_dice >= 0.30,
        "search_region_recall": search_recall >= 0.90,
        "accepted_coverage": accepted_coverage >= 0.75,
        "selector_regret": selector_regret <= 0.05,
    }
    write_json(
        path,
        {
            "go": all(checks.values()),
            "architecture_tag": governance["architecture_tag"],
            "checks": checks,
            "metrics": {
                "macro_dice": macro_dice,
                "worst_category_dice": worst_dice,
                "search_region_recall": search_recall,
                "accepted_coverage": accepted_coverage,
                "selector_regret": selector_regret,
            },
            "sample_count": len(generic_rows),
            "categories": categories,
        },
    )


def _evaluate_row(config: AppConfig, row: dict[str, Any], source: Path) -> dict[str, Any]:
    category = str(row["category"])
    defect = str(row["defect_type"])
    image_path = Path(str(row["image_path"]))
    truth_path = config.dataset_root / category / "ground_truth" / defect / f"{image_path.stem}_mask.png"
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing development reference mask: {truth_path}")
    truth = np.asarray(Image.open(truth_path).convert("L"), dtype=np.uint8) > 0
    prediction = _load_mask(Path(str(row.get("eval_mask_path") or row["refined_mask_path"])), truth.shape)
    score = _score_map(row, prediction, truth.shape)
    intersection = int((prediction & truth).sum())
    predicted = int(prediction.sum())
    positive = int(truth.sum())
    union = int((prediction | truth).sum())
    region = tuple(int(value) for value in row.get("region_xyxy", [0, 0, truth.shape[1], truth.shape[0]]))
    region_mask = np.zeros_like(truth, dtype=bool)
    region_mask[max(0, region[1]) : min(truth.shape[0], region[3]), max(0, region[0]) : min(truth.shape[1], region[2])] = True
    oracle = 0.0
    settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
    candidates = settings.get("candidate_refined_paths", {}) if isinstance(settings.get("candidate_refined_paths"), dict) else {}
    for path_value in candidates.values():
        path = Path(str(path_value))
        if not path.exists():
            continue
        candidate = _load_mask(path, truth.shape)
        candidate_intersection = int((candidate & truth).sum())
        oracle = max(oracle, 2 * candidate_intersection / max(1, int(candidate.sum()) + positive))
    ranking = segmentation_metrics(score, truth, threshold=0.5)
    decision = settings.get("selection_decision", {}) if isinstance(settings.get("selection_decision"), dict) else {}
    architecture = str(settings.get("auto_mask_architecture", "legacy_specialist"))
    return {
        "architecture": architecture,
        "metadata_source": str(source),
        "sample_id": f"{category}/{defect}/{image_path.name}",
        "category": category,
        "defect_type": defect,
        "dice": 2 * intersection / max(1, predicted + positive),
        "iou": intersection / max(1, union),
        "precision": intersection / max(1, predicted),
        "recall": intersection / max(1, positive),
        "predicted_positive_rate": predicted / prediction.size,
        "search_region_recall": int((region_mask & truth).sum()) / max(1, positive),
        "pixel_auroc": ranking["pixel_auroc"],
        "pixel_ap": float(average_precision_score(truth.reshape(-1), score.reshape(-1))),
        "aupro": ranking["aupro"],
        "oracle_dice": oracle,
        "selector_regret": max(0.0, oracle - 2 * intersection / max(1, predicted + positive)),
        "disposition": str(decision.get("disposition", settings.get("label_policy", {}).get("label_policy", "unknown"))),
        "confidence": float(decision.get("confidence", np.nan)),
    }


def _write_report(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["architecture"]), str(row["category"]))].append(row)
    lines = ["# Development Generalization Report", "", "Official development masks are evaluation-only and are not selector-training inputs.", "", "| Architecture | Category | Images | Dice | IoU | Precision | Recall | Search Recall | AUPRO | Pixel AP | Regret |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for (architecture, category), group in sorted(grouped.items()):
        mean = lambda key: float(np.nanmean([float(row[key]) for row in group]))
        lines.append(f"| {architecture} | {category} | {len(group)} | `{mean('dice'):.4f}` | `{mean('iou'):.4f}` | `{mean('precision'):.4f}` | `{mean('recall'):.4f}` | `{mean('search_region_recall'):.4f}` | `{mean('aupro'):.4f}` | `{mean('pixel_ap'):.4f}` | `{mean('selector_regret'):.4f}` |")
    lines.extend(["", "## Architecture Summary", "", "| Architecture | Category-Macro Dice | Worst-Quartile Dice | Accepted Coverage | Accepted Risk | 95% Bootstrap CI |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for architecture in sorted({str(row["architecture"]) for row in rows}):
        architecture_rows = [row for row in rows if row["architecture"] == architecture]
        category_values = []
        for category in sorted({str(row["category"]) for row in architecture_rows}):
            category_values.append(float(np.mean([float(row["dice"]) for row in architecture_rows if row["category"] == category])))
        accepted = [row for row in architecture_rows if row["disposition"] != "needs_review"]
        accepted_risk = 1.0 - float(np.mean([float(row["dice"]) for row in accepted])) if accepted else 1.0
        low, high = _hierarchical_bootstrap_ci(architecture_rows)
        lines.append(f"| {architecture} | `{np.mean(category_values):.4f}` | `{np.mean(sorted(category_values)[:max(1, len(category_values) // 4)]):.4f}` | `{len(accepted) / max(1, len(architecture_rows)):.4f}` | `{accepted_risk:.4f}` | `[{low:.4f}, {high:.4f}]` |")
    lines.append("")
    return "\n".join(lines)


def _hierarchical_bootstrap_ci(rows: list[dict[str, Any]], samples: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(17)
    categories = sorted({str(row["category"]) for row in rows})
    by_category = {category: [row for row in rows if row["category"] == category] for category in categories}
    values = []
    for _ in range(samples):
        sampled_categories = rng.choice(categories, size=len(categories), replace=True)
        category_scores = []
        for category in sampled_categories:
            group = by_category[str(category)]
            sampled_rows = rng.choice(group, size=len(group), replace=True)
            category_scores.append(float(np.mean([float(row["dice"]) for row in sampled_rows])))
        values.append(float(np.mean(category_scores)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((shape[1], shape[0]), Image.Resampling.NEAREST), dtype=np.uint8) > 0


def _score_map(row: dict[str, Any], prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
    parameters = settings.get("mask_parameters", {}) if isinstance(settings.get("mask_parameters"), dict) else {}
    path_value = parameters.get("fused_evidence_path")
    if path_value and Path(str(path_value)).exists():
        return np.asarray(Image.open(str(path_value)).convert("L").resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return prediction.astype(np.float32)
