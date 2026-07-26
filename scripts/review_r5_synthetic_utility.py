from __future__ import annotations

import argparse
from pathlib import Path

from iadgen_v2.r5_utility import review_r5_utility


def main() -> None:
    parser = argparse.ArgumentParser(description="Review the paired, fusion-free R5 synthetic-utility ablation.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator", default="supervised_resnet18_unet")
    parser.add_argument("--baseline-variant", default="real_only")
    parser.add_argument("--candidate-variant", default="critic_arbitrated")
    parser.add_argument("--candidate-ratio", type=float, default=0.25)
    parser.add_argument("--primary-metric", default="pixel_ap")
    parser.add_argument("--minimum-gain", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    args = parser.parse_args()
    report = review_r5_utility(
        args.results,
        args.output_dir,
        evaluator=args.evaluator,
        baseline_variant=args.baseline_variant,
        candidate_variant=args.candidate_variant,
        candidate_ratio=args.candidate_ratio,
        primary_metric=args.primary_metric,
        minimum_gain=args.minimum_gain,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(args.output_dir / "r5_synthetic_utility_review.md")
    print(f"gate_passed={report['gate']['passed']}")


if __name__ == "__main__":
    main()
