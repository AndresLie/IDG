"""Analyze a sealed locked candidate pool without retraining or changing masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.contracts import CandidateProposal
from iadgen_v2.auto_mask.proposals import MEASUREMENT_NAMES
from iadgen_v2.auto_mask.selection import GenericCandidateSelector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--synthetic-selector", required=True, type=Path)
    parser.add_argument("--nonwidened-selector", required=True, type=Path)
    parser.add_argument("--widened-selector", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    selectors = {
        "synthetic_baseline": GenericCandidateSelector(args.synthetic_selector),
        "real_candidate_nonwidened": GenericCandidateSelector(args.nonwidened_selector),
        "real_candidate_widened": GenericCandidateSelector(args.widened_selector),
    }
    references = _references(args.reference_manifest)
    records: list[dict[str, Any]] = []
    calibration: dict[str, dict[str, list[float]]] = {
        name: {"predicted": [], "actual": []} for name in selectors
    }
    signature = hashlib.sha256()
    deployed_matches = 0
    candidate_count = 0

    for line in args.metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = _sample_key(row)
        truth_path = references.get(key)
        if truth_path is None:
            raise ValueError(f"Missing locked reference for {'/'.join(key)}")
        truth = np.asarray(Image.open(truth_path).convert("L"), dtype=np.uint8) > 0
        settings = row.get("settings", {})
        measurements = settings.get("candidate_measurements", {})
        paths = settings.get("candidate_refined_paths", {})
        scores = settings.get("candidate_scores", {})
        actual_by_mode: dict[str, dict[str, float]] = {}
        proposals: list[CandidateProposal] = []
        for mode in settings.get("candidate_modes", measurements):
            values = measurements.get(mode)
            mask_path = Path(str(paths.get(mode, "")))
            if not isinstance(values, dict) or not mask_path.is_file():
                continue
            mask = np.asarray(
                Image.open(mask_path)
                .convert("L")
                .resize((truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST),
                dtype=np.uint8,
            ) > 0
            intersection = int((mask & truth).sum())
            predicted = int(mask.sum())
            positive = int(truth.sum())
            union = predicted + positive - intersection
            actual_by_mode[str(mode)] = {
                "iou": intersection / max(1, union),
                "precision": intersection / max(1, predicted),
                "recall": intersection / max(1, positive),
            }
            vector = {
                name: float(values.get(name, 0.0))
                for name in MEASUREMENT_NAMES
            }
            proposals.append(
                CandidateProposal(
                    mode=str(mode),
                    mask=np.zeros((1, 1), dtype=bool),
                    score=float(scores.get(mode, 0.0)),
                    measurements=vector,
                )
            )
            signature.update("/".join(key).encode("utf-8"))
            signature.update(str(mode).encode("utf-8"))
            signature.update(np.asarray(list(vector.values()), dtype=np.float32).tobytes())
        if not proposals:
            raise ValueError(f"No candidates for {'/'.join(key)}")
        candidate_count += len(proposals)
        oracle_mode = max(actual_by_mode, key=lambda mode: actual_by_mode[mode]["iou"])
        deployed = str(settings.get("selection_decision", {}).get("selected_mode", ""))
        sample = {
            "sample_id": "/".join(key),
            "category": key[0],
            "defect_type": key[1],
            "candidate_count": len(proposals),
            "oracle_mode": oracle_mode,
            "oracle_iou": actual_by_mode[oracle_mode]["iou"],
            "oracle_dice": _dice(actual_by_mode[oracle_mode]["iou"]),
            "selectors": {},
        }
        for name, selector in selectors.items():
            decision, predictions = selector.select(proposals)
            selected_mode = str(decision.selected_mode)
            selected = actual_by_mode[selected_mode]
            sample["selectors"][name] = {
                "selected_mode": selected_mode,
                "selected_iou": selected["iou"],
                "selected_dice": _dice(selected["iou"]),
                "selected_precision": selected["precision"],
                "selected_recall": selected["recall"],
                "dice_regret": sample["oracle_dice"] - _dice(selected["iou"]),
                "expected_iou": decision.expected_iou,
                "disposition": decision.disposition,
            }
            for prediction in predictions:
                calibration[name]["predicted"].append(prediction.expected_iou)
                calibration[name]["actual"].append(actual_by_mode[prediction.mode]["iou"])
        deployed_matches += int(
            deployed == sample["selectors"]["real_candidate_widened"]["selected_mode"]
        )
        records.append(sample)

    aggregates = {
        name: _aggregate(records, name, calibration[name])
        for name in selectors
    }
    report = {
        "sealed_inputs": {
            "metadata": _file_record(args.metadata),
            "reference_manifest": _file_record(args.reference_manifest),
            "selectors": {
                "synthetic_baseline": _file_record(args.synthetic_selector),
                "real_candidate_nonwidened": _file_record(args.nonwidened_selector),
                "real_candidate_widened": _file_record(args.widened_selector),
            },
        },
        "pool": {
            "images": len(records),
            "candidates": candidate_count,
            "pool_hash": signature.hexdigest(),
            "deployed_widened_selection_matches": deployed_matches,
        },
        "aggregates": aggregates,
        "primary_paired_bootstrap": _paired_bootstrap(
            records,
            baseline="synthetic_baseline",
            method="real_candidate_widened",
        ),
        "per_sample": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sample"}, indent=2))


def _aggregate(
    records: list[dict[str, Any]],
    selector: str,
    calibration: dict[str, list[float]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["category"])].append(row)

    def summary(rows: list[dict[str, Any]]) -> dict[str, float]:
        selected = [row["selectors"][selector] for row in rows]
        return {
            "images": len(rows),
            "selected_dice": float(np.mean([row["selected_dice"] for row in selected])),
            "selected_iou": float(np.mean([row["selected_iou"] for row in selected])),
            "precision": float(np.mean([row["selected_precision"] for row in selected])),
            "recall": float(np.mean([row["selected_recall"] for row in selected])),
            "oracle_dice": float(np.mean([row["oracle_dice"] for row in rows])),
            "dice_regret": float(np.mean([row["dice_regret"] for row in selected])),
            "accepted_coverage": float(
                np.mean([row["disposition"] != "needs_review" for row in selected])
            ),
        }

    categories = {category: summary(rows) for category, rows in sorted(groups.items())}
    macro_keys = (
        "selected_dice",
        "selected_iou",
        "precision",
        "recall",
        "oracle_dice",
        "dice_regret",
        "accepted_coverage",
    )
    predicted = np.asarray(calibration["predicted"], dtype=np.float64)
    actual = np.asarray(calibration["actual"], dtype=np.float64)
    return {
        "overall": summary(records),
        "category_macro": {
            key: float(np.mean([row[key] for row in categories.values()]))
            for key in macro_keys
        },
        "per_category": categories,
        "candidate_calibration": {
            "mae": float(np.mean(np.abs(predicted - actual))),
            "pearson": (
                float(np.corrcoef(predicted, actual)[0, 1])
                if np.std(predicted) > 1e-12 and np.std(actual) > 1e-12
                else 0.0
            ),
        },
    }


def _paired_bootstrap(
    records: list[dict[str, Any]],
    *,
    baseline: str,
    method: str,
    iterations: int = 10_000,
    seed: int = 20260724,
) -> dict[str, Any]:
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_category[str(row["category"])].append(
            float(row["selectors"][method]["selected_dice"])
            - float(row["selectors"][baseline]["selected_dice"])
        )
    categories = sorted(by_category)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled_categories = rng.choice(categories, len(categories), replace=True)
        means = []
        for category in sampled_categories:
            values = np.asarray(by_category[str(category)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(values, len(values), replace=True))))
        bootstrap[index] = float(np.mean(means))
    deltas = [value for values in by_category.values() for value in values]
    return {
        "comparison": f"{method}_minus_{baseline}",
        "mean": float(np.mean([np.mean(values) for values in by_category.values()])),
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "probability_nonpositive": float(np.mean(bootstrap <= 0.0)),
        "wins": int(sum(value > 0 for value in deltas)),
        "ties": int(sum(value == 0 for value in deltas)),
        "losses": int(sum(value < 0 for value in deltas)),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Locked Candidate-Pool Analysis",
        "",
        "This report re-scores sealed candidates with preregistered fixed selectors.",
        "It does not train, regenerate, or modify any mask.",
        "",
        "| Selector | Macro Dice | Oracle Dice | Regret | Precision | Recall | Coverage | MAE | Pearson |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["aggregates"].items():
        macro = result["category_macro"]
        calibration = result["candidate_calibration"]
        lines.append(
            f"| {name} | `{macro['selected_dice']:.4f}` | `{macro['oracle_dice']:.4f}` "
            f"| `{macro['dice_regret']:.4f}` | `{macro['precision']:.4f}` "
            f"| `{macro['recall']:.4f}` | `{macro['accepted_coverage']:.4f}` "
            f"| `{calibration['mae']:.4f}` | `{calibration['pearson']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Primary Paired Result",
            "",
            "```json",
            json.dumps(report["primary_paired_bootstrap"], indent=2),
            "```",
            "",
            "## Per-Category Selected Dice",
            "",
            "| Category | Synthetic | Real nonwidened | Real widened | Oracle |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    categories = sorted(
        report["aggregates"]["real_candidate_widened"]["per_category"]
    )
    for category in categories:
        syn = report["aggregates"]["synthetic_baseline"]["per_category"][category]
        non = report["aggregates"]["real_candidate_nonwidened"]["per_category"][category]
        wide = report["aggregates"]["real_candidate_widened"]["per_category"][category]
        lines.append(
            f"| {category} | `{syn['selected_dice']:.4f}` | `{non['selected_dice']:.4f}` "
            f"| `{wide['selected_dice']:.4f}` | `{wide['oracle_dice']:.4f}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _references(path: Path) -> dict[tuple[str, str, str], Path]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result[_sample_key(row)] = Path(str(row["official_mask_path"]))
    return result


def _sample_key(row: dict[str, Any]) -> tuple[str, str, str]:
    image = Path(str(row.get("image_path") or row.get("sample_id"))).stem
    return str(row["category"]), str(row["defect_type"]), image.removesuffix("_mask")


def _dice(iou: float) -> float:
    return 2.0 * iou / max(1e-12, 1.0 + iou)


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    main()
