from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from iadgen_v2.r5_utility import R5_FUSION_FREE_NOTE, review_r5_utility


def test_r5_review_enforces_fair_paired_hierarchical_gate(tmp_path: Path) -> None:
    result_rows = []
    for seed in (11, 23):
        baseline_path = tmp_path / f"baseline_{seed}.jsonl"
        candidate_path = tmp_path / f"candidate_{seed}.jsonl"
        baseline_rows = []
        candidate_rows = []
        for category in ("part_a", "part_b"):
            for index, is_anomaly in enumerate((0, 0, 1, 1)):
                image_path = str(tmp_path / category / f"{index}.png")
                common = {
                    "image_path": image_path,
                    "category": category,
                    "is_anomaly": is_anomaly,
                    "predicted_positive_rate": 0.02 if not is_anomaly else 0.08,
                    "pixel_ap": None if not is_anomaly else 0.50,
                    "aupro": None if not is_anomaly else 0.40,
                    "pixel_auroc": None if not is_anomaly else 0.70,
                    "dice": None if not is_anomaly else 0.30,
                }
                baseline_rows.append(
                    {
                        **common,
                        "image_score": 0.35 if not is_anomaly else 0.65,
                    }
                )
                candidate_rows.append(
                    {
                        **common,
                        "image_score": 0.10 if not is_anomaly else 0.90,
                        "predicted_positive_rate": common["predicted_positive_rate"] + 0.01,
                        "pixel_ap": None if not is_anomaly else 0.60,
                        "aupro": None if not is_anomaly else 0.48,
                        "pixel_auroc": None if not is_anomaly else 0.76,
                        "dice": None if not is_anomaly else 0.35,
                    }
                )
        baseline_path.write_text(
            "\n".join(json.dumps(row) for row in baseline_rows) + "\n",
            encoding="utf-8",
        )
        candidate_path.write_text(
            "\n".join(json.dumps(row) for row in candidate_rows) + "\n",
            encoding="utf-8",
        )
        common_result = {
            "evaluator": "supervised_resnet18_unet",
            "run_seed": seed,
            "training_seed": seed,
            "optimizer_steps": 100,
            "training_schedule_fingerprint": f"{seed:064x}",
            "deterministic_training": 1,
            "deterministic_algorithms": 1,
            "cudnn_deterministic": 1,
            "cudnn_benchmark": 0,
            "cublas_workspace_config": ":4096:8",
            "student_architecture": "resnet18_unet",
            "note": R5_FUSION_FREE_NOTE,
        }
        result_rows.extend(
            [
                {
                    **common_result,
                    "variant": "real_only",
                    "synthetic_ratio": 0.0,
                    "per_image_metrics_path": str(baseline_path),
                },
                {
                    **common_result,
                    "variant": "critic_arbitrated",
                    "synthetic_ratio": 0.25,
                    "per_image_metrics_path": str(candidate_path),
                },
            ]
        )
    results_path = tmp_path / "segmentation_results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    report = review_r5_utility(
        results_path,
        tmp_path / "review",
        bootstrap_samples=500,
        minimum_gain=0.02,
    )

    assert report["gate"]["fairness_contract_passed"]
    assert report["gate"]["passed"]
    assert report["intervals"]["pixel_ap"]["mean_delta"] == pytest.approx(0.10)
    assert report["intervals"]["pixel_ap"]["ci95_low"] > 0.0
    assert (tmp_path / "review" / "r5_synthetic_utility_review.md").exists()
    assert (tmp_path / "review" / "r5_synthetic_utility_review.json").exists()


def test_r5_review_rejects_unpaired_training_seed(tmp_path: Path) -> None:
    per_image = tmp_path / "metrics.jsonl"
    per_image.write_text(
        json.dumps(
            {
                "image_path": "a.png",
                "category": "part",
                "is_anomaly": 1,
                "image_score": 0.5,
                "pixel_ap": 0.5,
                "aupro": 0.4,
                "pixel_auroc": 0.6,
                "dice": 0.3,
                "predicted_positive_rate": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    for seed in (11, 23):
        for variant, ratio, training_seed in (
            ("real_only", 0.0, seed),
            ("critic_arbitrated", 0.25, seed + 1),
        ):
            rows.append(
                {
                    "evaluator": "supervised_resnet18_unet",
                    "variant": variant,
                    "synthetic_ratio": ratio,
                    "run_seed": seed,
                    "training_seed": training_seed,
                    "optimizer_steps": 10,
                    "training_schedule_fingerprint": f"{seed:064x}",
                    "deterministic_training": 1,
                    "deterministic_algorithms": 1,
                    "cudnn_deterministic": 1,
                    "cudnn_benchmark": 0,
                    "cublas_workspace_config": ":4096:8",
                    "student_architecture": "resnet18_unet",
                    "note": R5_FUSION_FREE_NOTE,
                    "per_image_metrics_path": str(per_image),
                }
            )
    results_path = tmp_path / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = review_r5_utility(results_path, tmp_path / "review", bootstrap_samples=20)

    assert not report["fairness"]["training_seed_matches_run_seed"]
    assert not report["gate"]["passed"]
