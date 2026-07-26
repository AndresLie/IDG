from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from iadgen_v2.segmentation import binary_auroc


R5_METRICS = (
    "pixel_ap",
    "aupro",
    "pixel_auroc",
    "dice",
    "image_auroc",
    "predicted_positive_rate",
)
R5_FUSION_FREE_NOTE = "supervised_resnet18_unet_no_teacher_no_score_fusion"


def review_r5_utility(
    results_path: Path,
    output_dir: Path,
    *,
    evaluator: str = "supervised_resnet18_unet",
    baseline_variant: str = "real_only",
    candidate_variant: str = "critic_arbitrated",
    candidate_ratio: float = 0.25,
    primary_metric: str = "pixel_ap",
    minimum_gain: float = 0.02,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 20260726,
) -> dict[str, Any]:
    if primary_metric not in R5_METRICS:
        raise ValueError(f"Unsupported R5 primary metric: {primary_metric}")
    rows = _read_results(results_path)
    baseline = _select_rows(rows, evaluator, baseline_variant, 0.0)
    candidate = _select_rows(rows, evaluator, candidate_variant, candidate_ratio)
    seeds = sorted(set(baseline) & set(candidate))
    if len(seeds) < 2:
        raise ValueError("R5 review requires at least two paired seeds")
    if set(baseline) != set(candidate):
        raise ValueError("R5 baseline and candidate seed sets do not match")

    fairness = _fairness_checks(baseline, candidate, seeds)
    paired = {
        seed: _paired_image_rows(
            _read_jsonl(Path(baseline[seed]["per_image_metrics_path"])),
            _read_jsonl(Path(candidate[seed]["per_image_metrics_path"])),
        )
        for seed in seeds
    }
    intervals = {
        metric: _hierarchical_interval(
            paired,
            metric,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index * 1009,
        )
        for index, metric in enumerate(R5_METRICS)
    }
    by_category = _category_deltas(paired)
    primary = intervals[primary_metric]
    gate = {
        "fairness_contract_passed": all(fairness.values()),
        "primary_metric": primary_metric,
        "minimum_gain": minimum_gain,
        "mean_gain_passed": float(primary["mean_delta"]) >= minimum_gain,
        "interval_excludes_zero": float(primary["ci95_low"]) > 0.0,
    }
    gate["passed"] = all(
        (
            gate["fairness_contract_passed"],
            gate["mean_gain_passed"],
            gate["interval_excludes_zero"],
        )
    )
    report = {
        "results_path": str(results_path.resolve()),
        "evaluator": evaluator,
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "candidate_ratio": candidate_ratio,
        "paired_seeds": seeds,
        "fairness": fairness,
        "intervals": intervals,
        "by_category": by_category,
        "gate": gate,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "hierarchy": "seed -> category -> image",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "r5_synthetic_utility_review.json"
    md_path = output_dir / "r5_synthetic_utility_review.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_report(report), encoding="utf-8")
    return report


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 5 results: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_rows(
    rows: list[dict[str, str]],
    evaluator: str,
    variant: str,
    ratio: float,
) -> dict[int, dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("evaluator") == evaluator
        and row.get("variant") == variant
        and math.isclose(float(row.get("synthetic_ratio", "nan")), ratio, abs_tol=1e-9)
    ]
    by_seed = {int(row["run_seed"]): row for row in selected}
    if len(by_seed) != len(selected):
        raise ValueError(f"Duplicate R5 rows for evaluator={evaluator}, variant={variant}, ratio={ratio}")
    if not by_seed:
        raise ValueError(f"Missing R5 rows for evaluator={evaluator}, variant={variant}, ratio={ratio}")
    return by_seed


def _fairness_checks(
    baseline: dict[int, dict[str, str]],
    candidate: dict[int, dict[str, str]],
    seeds: list[int],
) -> dict[str, bool]:
    return {
        "paired_run_seed_sets": set(baseline) == set(candidate),
        "training_seed_matches_run_seed": all(
            int(baseline[seed]["training_seed"]) == seed
            and int(candidate[seed]["training_seed"]) == seed
            for seed in seeds
        ),
        "matched_optimizer_steps": all(
            int(baseline[seed]["optimizer_steps"]) == int(candidate[seed]["optimizer_steps"])
            for seed in seeds
        ),
        "same_student_architecture": all(
            baseline[seed].get("student_architecture") == candidate[seed].get("student_architecture") == "resnet18_unet"
            for seed in seeds
        ),
        "fusion_free_student": all(
            baseline[seed].get("note") == candidate[seed].get("note") == R5_FUSION_FREE_NOTE
            for seed in seeds
        ),
        "strict_deterministic_training": all(
            baseline[seed].get("deterministic_training") == "1"
            and candidate[seed].get("deterministic_training") == "1"
            and baseline[seed].get("deterministic_algorithms") == "1"
            and candidate[seed].get("deterministic_algorithms") == "1"
            and baseline[seed].get("cudnn_deterministic") == "1"
            and candidate[seed].get("cudnn_deterministic") == "1"
            and baseline[seed].get("cudnn_benchmark") == "0"
            and candidate[seed].get("cudnn_benchmark") == "0"
            and baseline[seed].get("cublas_workspace_config") in {":4096:8", ":16:8"}
            and candidate[seed].get("cublas_workspace_config") in {":4096:8", ":16:8"}
            for seed in seeds
        ),
        "training_schedule_recorded": all(
            len(baseline[seed].get("training_schedule_fingerprint", "")) == 64
            and len(candidate[seed].get("training_schedule_fingerprint", "")) == 64
            for seed in seeds
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing per-image Phase 5 metrics: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _paired_image_rows(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    baseline_index = {str(row["image_path"]): row for row in baseline}
    candidate_index = {str(row["image_path"]): row for row in candidate}
    if baseline_index.keys() != candidate_index.keys():
        missing = sorted(baseline_index.keys() ^ candidate_index.keys())
        raise ValueError(f"R5 held-out image sets do not match: {missing[:5]}")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for image_path in sorted(baseline_index):
        base = baseline_index[image_path]
        cand = candidate_index[image_path]
        if str(base.get("category", "")) != str(cand.get("category", "")):
            raise ValueError(f"R5 category mismatch for {image_path}")
        grouped.setdefault(str(base.get("category", "unknown")), []).append((base, cand))
    return grouped


def _hierarchical_interval(
    paired: dict[int, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]],
    metric: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float]:
    point = _paired_macro_delta(paired, metric)
    rng = np.random.default_rng(bootstrap_seed)
    seeds = sorted(paired)
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for draw_index in range(bootstrap_samples):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_deltas: list[float] = []
        for seed in selected_seeds:
            categories = sorted(paired[int(seed)])
            selected_categories = rng.choice(categories, size=len(categories), replace=True)
            category_deltas: list[float] = []
            for category in selected_categories:
                values = paired[int(seed)][str(category)]
                sampled_indices = rng.integers(0, len(values), size=len(values))
                sampled = [values[int(index)] for index in sampled_indices]
                delta = _sample_delta(sampled, metric)
                if math.isfinite(delta):
                    category_deltas.append(delta)
            if category_deltas:
                seed_deltas.append(float(np.mean(category_deltas)))
        draws[draw_index] = float(np.mean(seed_deltas)) if seed_deltas else math.nan
    finite = draws[np.isfinite(draws)]
    if not finite.size:
        return {"mean_delta": point, "ci95_low": math.nan, "ci95_high": math.nan}
    return {
        "mean_delta": point,
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
    }


def _paired_macro_delta(
    paired: dict[int, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]],
    metric: str,
) -> float:
    seed_deltas: list[float] = []
    for categories in paired.values():
        category_deltas = [
            delta
            for values in categories.values()
            if math.isfinite(delta := _sample_delta(values, metric))
        ]
        if category_deltas:
            seed_deltas.append(float(np.mean(category_deltas)))
    return float(np.mean(seed_deltas)) if seed_deltas else math.nan


def _sample_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
) -> float:
    if metric == "image_auroc":
        labels = np.asarray([int(base.get("is_anomaly", 0)) for base, _ in pairs], dtype=np.uint8)
        baseline_scores = np.asarray([float(base["image_score"]) for base, _ in pairs], dtype=np.float64)
        candidate_scores = np.asarray([float(candidate["image_score"]) for _, candidate in pairs], dtype=np.float64)
        base_value = binary_auroc(baseline_scores, labels)
        candidate_value = binary_auroc(candidate_scores, labels)
        return candidate_value - base_value if math.isfinite(base_value) and math.isfinite(candidate_value) else math.nan
    differences = []
    for baseline, candidate in pairs:
        base_value = baseline.get(metric)
        candidate_value = candidate.get(metric)
        if base_value is None or candidate_value is None:
            continue
        base_numeric = float(base_value)
        candidate_numeric = float(candidate_value)
        if math.isfinite(base_numeric) and math.isfinite(candidate_numeric):
            differences.append(candidate_numeric - base_numeric)
    return float(np.mean(differences)) if differences else math.nan


def _category_deltas(
    paired: dict[int, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]],
) -> dict[str, dict[str, float]]:
    categories = sorted({category for values in paired.values() for category in values})
    result: dict[str, dict[str, float]] = {}
    for category in categories:
        result[category] = {}
        for metric in R5_METRICS:
            values = [
                delta
                for seed_rows in paired.values()
                if category in seed_rows and math.isfinite(delta := _sample_delta(seed_rows[category], metric))
            ]
            result[category][metric] = float(np.mean(values)) if values else math.nan
    return result


def _render_report(report: dict[str, Any]) -> str:
    gate = report["gate"]
    lines = [
        "# R5 Independent Synthetic-Utility Review",
        "",
        "## Decision",
        "",
        (
            "**PASS.** The preregistered independent utility gate passed."
            if gate["passed"]
            else "**DOES NOT PASS.** The preregistered independent utility gate did not pass."
        ),
        "",
        f"Evaluator: `{report['evaluator']}`",
        f"Baseline: `{report['baseline_variant']}`",
        f"Candidate: `{report['candidate_variant']}`, ratio `{report['candidate_ratio']:.2f}`",
        f"Paired seeds: `{report['paired_seeds']}`",
        "",
        "## Fairness Contract",
        "",
        "| Check | Status |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{key}` | {'PASS' if value else 'FAIL'} |"
        for key, value in report["fairness"].items()
    )
    lines.extend(
        [
            "",
            "The student receives no PatchCore-refined labels and no PatchCore score fusion. "
            "Both arms use the same initialization seed and optimizer-step count. "
            "Strict deterministic training must be enabled and every training schedule is hash-recorded.",
            "",
            "## Paired Hierarchical Bootstrap",
            "",
            "| Metric | Mean delta | 95% CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric, interval in report["intervals"].items():
        lines.append(
            f"| `{metric}` | `{interval['mean_delta']:+.4f}` | "
            f"`[{interval['ci95_low']:+.4f}, {interval['ci95_high']:+.4f}]` |"
        )
    lines.extend(
        [
            "",
            "Bootstrap hierarchy: seed, then category, then image. Categories are macro-weighted.",
            "",
            "## Category Diagnosis",
            "",
            "| Category | Pixel AP Δ | AUPRO Δ | Dice Δ | Image AUROC Δ | Pred+ Δ |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for category, values in report["by_category"].items():
        lines.append(
            f"| `{category}` | {values['pixel_ap']:+.4f} | {values['aupro']:+.4f} | "
            f"{values['dice']:+.4f} | {values['image_auroc']:+.4f} | "
            f"{values['predicted_positive_rate']:+.4f} |"
        )
    primary = report["intervals"][gate["primary_metric"]]
    lines.extend(
        [
            "",
            "## Preregistered Gate",
            "",
            f"- Primary metric: `{gate['primary_metric']}`.",
            f"- Required mean gain: `>= {gate['minimum_gain']:.4f}`.",
            f"- Observed mean gain: `{primary['mean_delta']:+.4f}`.",
            f"- Observed interval: `[{primary['ci95_low']:+.4f}, {primary['ci95_high']:+.4f}]`.",
            f"- Fairness contract: `{'PASS' if gate['fairness_contract_passed'] else 'FAIL'}`.",
            f"- Joint gate: `{'PASS' if gate['passed'] else 'FAIL'}`.",
            "",
            "This development result is not a locked-category claim. It must not be used to "
            "promote R4 arbitration before the independent blind-review gate is complete.",
            "",
        ]
    )
    return "\n".join(lines)
