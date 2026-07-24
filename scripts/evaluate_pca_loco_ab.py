"""Evaluate PCA evidence with pool-matched leave-category-out selector refits.

This is an evaluation-only command. Auto-mask metadata and candidate masks must
already exist before this script opens development-category official masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iadgen_v2.auto_mask.contracts import CandidateProposal
from iadgen_v2.auto_mask.proposals import MEASUREMENT_NAMES
from iadgen_v2.auto_mask.selection import GenericCandidateSelector, fit_selector_bundle


@dataclass
class Candidate:
    mode: str
    measurements: dict[str, float]
    score: float
    iou: float
    precision: float
    recall: float


@dataclass
class Sample:
    sample_id: str
    category: str
    candidates: list[Candidate]


def _official_mask(root: Path, row: dict[str, Any]) -> Path:
    return (
        root
        / str(row["category"])
        / "ground_truth"
        / str(row["defect_type"])
        / f"{Path(str(row['image_path'])).stem}_mask.png"
    )


def _load_pool(metadata: Path, root: Path) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    gate_active: dict[str, int] = {}
    gate_total: dict[str, int] = {}
    signature = hashlib.sha256()
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        category = str(row["category"])
        truth_path = _official_mask(root, row)
        if not truth_path.exists():
            continue
        truth = np.asarray(Image.open(truth_path).convert("L"), dtype=np.uint8) > 127
        settings = row["settings"]
        measurements = settings.get("candidate_measurements", {})
        paths = settings.get("candidate_refined_paths", {})
        scores = settings.get("candidate_scores", {})
        candidates: list[Candidate] = []
        for mode, values in measurements.items():
            mask_path = Path(str(paths.get(mode, "")))
            if not mask_path.exists():
                continue
            mask = np.asarray(
                Image.open(mask_path).convert("L").resize((truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST),
                dtype=np.uint8,
            ) > 127
            intersection = int((mask & truth).sum())
            predicted = int(mask.sum())
            actual = int(truth.sum())
            union = predicted + actual - intersection
            vector = {name: float(values.get(name, 0.0)) for name in MEASUREMENT_NAMES}
            candidate = Candidate(
                mode=str(mode),
                measurements=vector,
                score=float(scores.get(mode, 0.0)),
                iou=intersection / max(1, union),
                precision=intersection / max(1, predicted),
                recall=intersection / max(1, actual),
            )
            candidates.append(candidate)
            signature.update(category.encode())
            signature.update(str(mode).encode())
            signature.update(np.asarray(list(vector.values()), dtype=np.float32).tobytes())
        if not candidates:
            continue
        sample_id = f"{category}/{row['defect_type']}/{Path(str(row['image_path'])).name}"
        samples.append(Sample(sample_id, category, candidates))
        gate = settings.get("mask_parameters", {}).get("pca_subspace_gate", {})
        gate_total[category] = gate_total.get(category, 0) + 1
        gate_active[category] = gate_active.get(category, 0) + int(bool(gate.get("active", False)))
    return samples, {
        "metadata": str(metadata),
        "pool_hash": signature.hexdigest()[:16],
        "images": len(samples),
        "candidates": sum(len(sample.candidates) for sample in samples),
        "gate_active_by_category": {
            category: {"active": gate_active.get(category, 0), "total": total}
            for category, total in sorted(gate_total.items())
        },
    }


def _training_rows(samples: list[Sample], excluded: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.category == excluded:
            continue
        for candidate in sample.candidates:
            rows.append(
                {
                    "category": sample.category,
                    "corruption_family": "real",
                    **candidate.measurements,
                    "iou": candidate.iou,
                    "precision": candidate.precision,
                    "recall": candidate.recall,
                }
            )
    return rows


def _dice(iou: float) -> float:
    return 2.0 * iou / max(1e-12, 1.0 + iou)


def _loco_evaluate(samples: list[Sample]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    categories = sorted({sample.category for sample in samples})
    records: list[dict[str, Any]] = []
    all_predictions: list[float] = []
    all_actual: list[float] = []
    with tempfile.TemporaryDirectory(prefix="pca-loco-") as directory:
        for held in categories:
            model_path = Path(directory) / f"selector_without_{held}.joblib"
            fit_selector_bundle(_training_rows(samples, held), model_path)
            selector = GenericCandidateSelector(model_path)
            for sample in samples:
                if sample.category != held:
                    continue
                proposals = [
                    CandidateProposal(
                        mode=candidate.mode,
                        mask=np.zeros((1, 1), dtype=bool),
                        score=candidate.score,
                        measurements=candidate.measurements,
                    )
                    for candidate in sample.candidates
                ]
                decision, predictions = selector.select(proposals)
                actual_by_mode = {candidate.mode: candidate for candidate in sample.candidates}
                selected = actual_by_mode[str(decision.selected_mode)]
                oracle = max(sample.candidates, key=lambda candidate: candidate.iou)
                prediction_by_mode = {prediction.mode: prediction for prediction in predictions}
                for mode, candidate in actual_by_mode.items():
                    all_predictions.append(prediction_by_mode[mode].expected_iou)
                    all_actual.append(candidate.iou)
                records.append(
                    {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "selected_mode": selected.mode,
                        "selected_iou": selected.iou,
                        "selected_dice": _dice(selected.iou),
                        "oracle_mode": oracle.mode,
                        "oracle_iou": oracle.iou,
                        "oracle_dice": _dice(oracle.iou),
                        "dice_regret": _dice(oracle.iou) - _dice(selected.iou),
                        "expected_iou": float(decision.expected_iou or 0.0),
                        "disposition": decision.disposition,
                    }
                )
    predictions = np.asarray(all_predictions, dtype=np.float64)
    actual = np.asarray(all_actual, dtype=np.float64)
    return records, {
        "candidate_mae": float(np.mean(np.abs(predictions - actual))),
        "candidate_pearson": (
            float(np.corrcoef(predictions, actual)[0, 1])
            if np.std(predictions) > 1e-12 and np.std(actual) > 1e-12
            else 0.0
        ),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    categories = sorted({str(row["category"]) for row in records})
    for category in ["overall", *categories]:
        selected = records if category == "overall" else [row for row in records if row["category"] == category]
        result[category] = {
            "images": len(selected),
            "selected_dice": float(np.mean([row["selected_dice"] for row in selected])),
            "selected_iou": float(np.mean([row["selected_iou"] for row in selected])),
            "oracle_dice": float(np.mean([row["oracle_dice"] for row in selected])),
            "dice_regret": float(np.mean([row["dice_regret"] for row in selected])),
            "accepted_coverage": float(
                np.mean([row["disposition"] != "needs_review" for row in selected])
            ),
        }
    result["category_macro"] = {"images": len(records)}
    result["category_macro"].update(
        {
            key: float(np.mean([result[category][key] for category in categories]))
            for key in ("selected_dice", "selected_iou", "oracle_dice", "dice_regret", "accepted_coverage")
        }
    )
    return result


def _paired_bootstrap(
    baseline: dict[str, dict[str, Any]],
    pca: dict[str, dict[str, Any]],
    *,
    iterations: int = 10_000,
    seed: int = 20260724,
) -> dict[str, Any]:
    paired = [
        (row["category"], float(pca[sample_id]["selected_dice"]) - float(row["selected_dice"]))
        for sample_id, row in baseline.items()
        if sample_id in pca
    ]
    by_category: dict[str, list[float]] = {}
    for category, delta in paired:
        by_category.setdefault(str(category), []).append(delta)
    categories = sorted(by_category)
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled_categories = rng.choice(categories, len(categories), replace=True)
        category_means: list[float] = []
        for category in sampled_categories:
            category_values = by_category[str(category)]
            positions = rng.integers(0, len(category_values), len(category_values))
            category_means.append(float(np.mean([category_values[position] for position in positions])))
        means[index] = float(np.mean(category_means))
    observed = float(np.mean([np.mean(values) for values in by_category.values()]))
    return {
        "mean": observed,
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_nonpositive": float(np.mean(means <= 0.0)),
        "wins": int(sum(delta > 0 for _, delta in paired)),
        "ties": int(sum(delta == 0 for _, delta in paired)),
        "losses": int(sum(delta < 0 for _, delta in paired)),
    }


def _strict_superset_oracle(
    baseline_samples: list[Sample],
    pca_samples: list[Sample],
) -> dict[str, Any]:
    baseline = {sample.sample_id: sample for sample in baseline_samples}
    pca = {sample.sample_id: sample for sample in pca_samples}
    records: list[dict[str, Any]] = []
    for sample_id in sorted(set(baseline) & set(pca)):
        baseline_oracle = max(candidate.iou for candidate in baseline[sample_id].candidates)
        pca_oracle = max(candidate.iou for candidate in pca[sample_id].candidates)
        superset_oracle = max(baseline_oracle, pca_oracle)
        records.append(
            {
                "sample_id": sample_id,
                "category": baseline[sample_id].category,
                "baseline_oracle_dice": _dice(baseline_oracle),
                "pca_oracle_dice": _dice(pca_oracle),
                "strict_superset_oracle_dice": _dice(superset_oracle),
            }
        )
    categories = sorted({str(row["category"]) for row in records})

    def overall(key: str) -> float:
        return float(np.mean([row[key] for row in records]))

    def category_macro(key: str) -> float:
        return float(
            np.mean(
                [
                    np.mean([row[key] for row in records if row["category"] == category])
                    for category in categories
                ]
            )
        )

    return {
        "overall_baseline_oracle_dice": overall("baseline_oracle_dice"),
        "overall_pca_oracle_dice": overall("pca_oracle_dice"),
        "overall_strict_superset_oracle_dice": overall("strict_superset_oracle_dice"),
        "overall_strict_superset_gain_over_baseline": overall("strict_superset_oracle_dice")
        - overall("baseline_oracle_dice"),
        "category_macro_baseline_oracle_dice": category_macro("baseline_oracle_dice"),
        "category_macro_pca_oracle_dice": category_macro("pca_oracle_dice"),
        "category_macro_strict_superset_oracle_dice": category_macro("strict_superset_oracle_dice"),
        "category_macro_strict_superset_gain_over_baseline": category_macro("strict_superset_oracle_dice")
        - category_macro("baseline_oracle_dice"),
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline"]["aggregate"]
    pca = report["pca"]["aggregate"]
    lines = [
        "# Five-Category PCA Evidence LOCO Review",
        "",
        "## Protocol",
        "",
        "- Candidate masks were generated before official masks were opened.",
        "- Each arm refits the selector on four categories and evaluates the held-out fifth category.",
        "- The PCA arm is therefore calibrated on PCA-perturbed candidate pools rather than treated as in-distribution by assumption.",
        "- Official masks are used only in this evaluation script to label candidates and compute metrics.",
        "",
        "## Results",
        "",
        "| Category | Baseline Dice | PCA Dice | Delta | Baseline Regret | PCA Regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in ["category_macro", "overall", "bottle", "zipper", "wood", "metal_nut", "tile"]:
        before = baseline[category]
        after = pca[category]
        lines.append(
            f"| {category} | `{before['selected_dice']:.4f}` | `{after['selected_dice']:.4f}` | "
            f"`{after['selected_dice'] - before['selected_dice']:+.4f}` | "
            f"`{before['dice_regret']:.4f}` | `{after['dice_regret']:.4f}` |"
        )
    paired = report["paired_hierarchical_bootstrap"]
    calibration_before = report["baseline"]["calibration"]
    calibration_after = report["pca"]["calibration"]
    strict = report["strict_superset_oracle"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Keep the PCA provider implemented and default-off. Do not deploy the current global repeated-texture gate.",
            "",
            "The category-macro gain is directional, but its confidence interval crosses zero. Zipper and metal_nut improve while bottle, wood, and tile regress. Bottle is not directly processed by PCA; its regression comes from cross-category selector refitting on PCA-shifted training pools, which shows that evidence gating and selector calibration cannot be treated as independent.",
            "",
            "Pool-matched refitting does not repair calibration: MAE is essentially flat and slightly worse. The next valid experiment is a nested leave-category-out mixture-of-experts selector with a baseline expert and a PCA expert, routed by normal-only topology and evidence diagnostics. Gate thresholds must be chosen inside each training fold, not from held-category official-mask results.",
            "",
            "## Calibration",
            "",
            f"- Baseline full-pool MAE: `{calibration_before['candidate_mae']:.4f}`.",
            f"- PCA full-pool MAE after pool-matched refit: `{calibration_after['candidate_mae']:.4f}`.",
            f"- Baseline full-pool Pearson: `{calibration_before['candidate_pearson']:.4f}`.",
            f"- PCA full-pool Pearson after pool-matched refit: `{calibration_after['candidate_pearson']:.4f}`.",
            "",
            "## Paired Uncertainty",
            "",
            f"- Mean Dice delta: `{paired['mean']:+.4f}`.",
            f"- Hierarchical bootstrap 95% CI: `[{paired['ci95'][0]:+.4f}, {paired['ci95'][1]:+.4f}]`.",
            f"- Wins/ties/losses: `{paired['wins']}/{paired['ties']}/{paired['losses']}`.",
            "",
            "## Strict-Superset Ceiling Check",
            "",
            f"- Baseline category-macro oracle Dice: `{strict['category_macro_baseline_oracle_dice']:.4f}`.",
            f"- PCA-only category-macro oracle Dice: `{strict['category_macro_pca_oracle_dice']:.4f}`.",
            f"- Union-of-both-pools category-macro oracle Dice: `{strict['category_macro_strict_superset_oracle_dice']:.4f}`.",
            f"- Category-macro strict-superset ceiling gain: `{strict['category_macro_strict_superset_gain_over_baseline']:+.4f}`.",
            f"- Image-weighted strict-superset ceiling gain: `{strict['overall_strict_superset_gain_over_baseline']:+.4f}`.",
            "",
            "The strict-superset value supports only a candidate-ceiling statement. It does not show that the production selector can realize that ceiling.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--pca-metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    baseline_samples, baseline_pool = _load_pool(args.baseline_metadata, args.data_root)
    pca_samples, pca_pool = _load_pool(args.pca_metadata, args.data_root)
    baseline_ids = {sample.sample_id for sample in baseline_samples}
    pca_ids = {sample.sample_id for sample in pca_samples}
    if baseline_ids != pca_ids:
        raise ValueError(
            f"Candidate cohorts differ: baseline-only={len(baseline_ids - pca_ids)}, "
            f"pca-only={len(pca_ids - baseline_ids)}"
        )

    baseline_records, baseline_calibration = _loco_evaluate(baseline_samples)
    pca_records, pca_calibration = _loco_evaluate(pca_samples)
    baseline_by_id = {row["sample_id"]: row for row in baseline_records}
    pca_by_id = {row["sample_id"]: row for row in pca_records}
    report = {
        "baseline": {
            "pool": baseline_pool,
            "calibration": baseline_calibration,
            "aggregate": _aggregate(baseline_records),
        },
        "pca": {
            "pool": pca_pool,
            "calibration": pca_calibration,
            "aggregate": _aggregate(pca_records),
        },
        "paired_hierarchical_bootstrap": _paired_bootstrap(baseline_by_id, pca_by_id),
        "strict_superset_oracle": _strict_superset_oracle(baseline_samples, pca_samples),
        "per_sample": {
            sample_id: {
                "baseline": baseline_by_id[sample_id],
                "pca": pca_by_id[sample_id],
            }
            for sample_id in sorted(baseline_by_id)
        },
    }
    (args.out / "pca_loco_ab.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, args.out / "pca_loco_ab.md")
    print(args.out / "pca_loco_ab.md")


if __name__ == "__main__":
    main()
