from __future__ import annotations

import argparse
from pathlib import Path

from iadgen_v2.auto_masks import run_auto_masks
from iadgen_v2.baseline import run_baseline
from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.feasibility import write_feasibility_report
from iadgen_v2.metrics import evaluate_generation
from iadgen_v2.phase2 import evaluate_phase2_placement, run_phase2_proposals
from iadgen_v2.phase3 import build_phase3_adaptation_cache, train_phase3_adapter, validate_phase3_adapter
from iadgen_v2.phase4 import run_phase4_generation
from iadgen_v2.phase5 import run_phase5_evaluation
from iadgen_v2.phase6 import write_phase6_preflight
from iadgen_v2.phase9_visual import write_phase9_visual_report
from iadgen_v2.phase10_mask_quality import run_phase10_mask_quality_ablation
from iadgen_v2.phase11_tfidg_critic import run_phase11_tfidg_critic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IADGen v2 research pipeline commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("feasibility", "prepare", "auto-masks"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        if command == "prepare":
            sub.add_argument("--no-download", action="store_true")
    for command in ("generate", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--model", required=True, choices=("mock", "sd15"))
    for command in ("phase2-propose", "phase2-evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
    for command in ("phase3-cache", "phase3-train", "phase3-validate"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
    for command in ("phase4-generate",):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
        sub.add_argument("--variant", default=None)
    for command in ("phase5-evaluate",):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
    for command in ("phase9-visual-report",):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
    for command in ("phase10-mask-quality",):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
    for command in ("phase11-tfidg-critic",):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--provider", default=None, choices=("heuristic", "qwen"))
    sub = commands.add_parser("phase6-preflight")
    sub.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "feasibility":
        print(f"feasibility report written to {write_feasibility_report(config)}")
    elif args.command == "auto-masks":
        print(f"auto-mask metadata written to {run_auto_masks(config)}")
    elif args.command == "prepare":
        download = False if args.no_download else None
        print(f"split manifest written to {prepare_splits(config, download=download)}")
    elif args.command == "generate":
        print(f"metadata written to {run_baseline(config, args.model)}")
    elif args.command == "evaluate":
        print(f"metrics written to {evaluate_generation(config, args.model)}")
    elif args.command == "phase2-propose":
        print(f"phase 2 metadata written to {run_phase2_proposals(config, args.provider)}")
    elif args.command == "phase2-evaluate":
        print(f"phase 2 metrics written to {evaluate_phase2_placement(config, args.provider)}")
    elif args.command == "phase3-cache":
        print(f"phase 3 cache metadata written to {build_phase3_adaptation_cache(config, args.provider)}")
    elif args.command == "phase3-train":
        print(f"phase 3 adapter checkpoint written to {train_phase3_adapter(config, args.provider)}")
    elif args.command == "phase3-validate":
        print(f"phase 3 validation written to {validate_phase3_adapter(config, args.provider)}")
    elif args.command == "phase4-generate":
        print(f"phase 4 metadata written to {run_phase4_generation(config, args.provider, args.variant)}")
    elif args.command == "phase5-evaluate":
        print(f"phase 5 results written to {run_phase5_evaluation(config, args.provider)}")
    elif args.command == "phase9-visual-report":
        print(f"phase 9 visual report written to {write_phase9_visual_report(config, args.provider)}")
    elif args.command == "phase10-mask-quality":
        print(f"phase 10 mask quality report written to {run_phase10_mask_quality_ablation(config, args.provider)}")
    elif args.command == "phase11-tfidg-critic":
        print(f"phase 11 TF-IDG-lite critic report written to {run_phase11_tfidg_critic(config, args.provider)}")
    elif args.command == "phase6-preflight":
        print(f"phase 6 preflight written to {write_phase6_preflight(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
