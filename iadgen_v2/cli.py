from __future__ import annotations

import argparse
import time
from pathlib import Path

from iadgen_v2.auto_masks import reselect_auto_masks, run_auto_masks
from iadgen_v2.baseline import run_baseline
from iadgen_v2.config import AppConfig, load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.feasibility import write_feasibility_report
from iadgen_v2.governance import (
    finalize_experiment_manifest,
    reset_runtime_resource_counters,
    validate_governance_for_command,
    write_experiment_manifest,
)
from iadgen_v2.metrics import evaluate_generation
from iadgen_v2.phase2 import evaluate_phase2_placement, run_phase2_proposals
from iadgen_v2.phase3 import build_phase3_adaptation_cache, train_phase3_adapter, validate_phase3_adapter
from iadgen_v2.phase4 import run_phase4_generation
from iadgen_v2.phase5 import configure_phase5_runtime, run_phase5_evaluation
from iadgen_v2.phase6 import write_phase6_preflight
from iadgen_v2.phase9_visual import write_phase9_visual_report
from iadgen_v2.phase10_mask_quality import run_phase10_mask_quality_ablation
from iadgen_v2.phase11_tfidg_critic import run_phase11_tfidg_critic
from iadgen_v2.r6_evidence import build_r6_evidence_package


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IADGen v2 research pipeline commands.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("feasibility", "prepare", "auto-masks", "auto-masks-reselect", "auto-mask-train-selector"):
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
    sub = commands.add_parser("experiment-manifest")
    sub.add_argument("--config", required=True, type=Path)
    sub.add_argument("--for-command", default="manual-review")
    sub.add_argument("--provider", default=None)
    sub.add_argument("--model", default=None)
    sub = commands.add_parser("locked-evaluate")
    sub.add_argument("--config", required=True, type=Path)
    sub.add_argument("--runtime-manifest", required=True, type=Path)
    sub.add_argument("--reference-manifest", required=True, type=Path)
    sub = commands.add_parser("development-evaluate")
    sub.add_argument("--config", required=True, type=Path)
    sub.add_argument("--metadata", action="append", type=Path, default=[])
    sub = commands.add_parser("prepare-locked-benchmark")
    sub.add_argument("--config", required=True, type=Path)
    sub.add_argument("--source-root", required=True, type=Path)
    sub.add_argument("--download-source", action="store_true")
    sub = commands.add_parser("prepare-visa-benchmark")
    sub.add_argument("--config", required=True, type=Path)
    sub.add_argument("--download", action="store_true")
    sub = commands.add_parser("freeze-architecture")
    sub.add_argument("--config", required=True, type=Path)
    sub = commands.add_parser("r6-evidence-package")
    sub.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    manifest_command = args.for_command if args.command == "experiment-manifest" else args.command
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    allow_official_masks = args.command in {
        "locked-evaluate",
        "prepare-locked-benchmark",
        "prepare-visa-benchmark",
    }
    validate_governance_for_command(config, manifest_command, allow_official_masks=allow_official_masks)
    if args.command == "phase5-evaluate":
        configure_phase5_runtime(config)
    started = time.monotonic()
    manifest_path = write_experiment_manifest(
        config,
        manifest_command,
        provider=provider,
        model=model,
    )
    reset_runtime_resource_counters()
    try:
        label, output_path = _execute_command(args, config, manifest_path)
    except BaseException as exc:
        finalize_experiment_manifest(
            manifest_path,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            started_monotonic=started,
            error=exc,
        )
        raise
    finalize_experiment_manifest(
        manifest_path,
        status="succeeded",
        started_monotonic=started,
        output_paths=(output_path,),
    )
    print(f"{label} {output_path}")
    return 0


def _execute_command(args: argparse.Namespace, config: AppConfig, manifest_path: Path) -> tuple[str, Path]:
    if args.command == "feasibility":
        return "feasibility report written to", write_feasibility_report(config)
    if args.command == "auto-masks":
        return "auto-mask metadata written to", run_auto_masks(config)
    if args.command == "auto-masks-reselect":
        return "auto-mask metadata reselected from cached candidates at", reselect_auto_masks(config)
    if args.command == "auto-mask-train-selector":
        from iadgen_v2.auto_mask.selector_training import train_generic_selector

        return "generic selector written to", train_generic_selector(config)
    if args.command == "prepare":
        download = False if args.no_download else None
        return "split manifest written to", prepare_splits(config, download=download)
    if args.command == "generate":
        return "metadata written to", run_baseline(config, args.model)
    if args.command == "evaluate":
        return "metrics written to", evaluate_generation(config, args.model)
    if args.command == "phase2-propose":
        return "phase 2 metadata written to", run_phase2_proposals(config, args.provider)
    if args.command == "phase2-evaluate":
        return "phase 2 metrics written to", evaluate_phase2_placement(config, args.provider)
    if args.command == "phase3-cache":
        return "phase 3 cache metadata written to", build_phase3_adaptation_cache(config, args.provider)
    if args.command == "phase3-train":
        return "phase 3 adapter checkpoint written to", train_phase3_adapter(config, args.provider)
    if args.command == "phase3-validate":
        return "phase 3 validation written to", validate_phase3_adapter(config, args.provider)
    if args.command == "phase4-generate":
        return "phase 4 metadata written to", run_phase4_generation(config, args.provider, args.variant)
    if args.command == "phase5-evaluate":
        return "phase 5 results written to", run_phase5_evaluation(config, args.provider)
    if args.command == "phase9-visual-report":
        return "phase 9 visual report written to", write_phase9_visual_report(config, args.provider)
    if args.command == "phase10-mask-quality":
        return "phase 10 mask quality report written to", run_phase10_mask_quality_ablation(config, args.provider)
    if args.command == "phase11-tfidg-critic":
        return "phase 11 TF-IDG-lite critic report written to", run_phase11_tfidg_critic(config, args.provider)
    if args.command == "phase6-preflight":
        return "phase 6 preflight written to", write_phase6_preflight(config)
    if args.command == "experiment-manifest":
        return "experiment manifest written to", manifest_path
    if args.command == "locked-evaluate":
        from iadgen_v2.locked_evaluation import run_locked_evaluation

        return "locked evaluation written to", run_locked_evaluation(
            config,
            runtime_manifest=args.runtime_manifest,
            reference_manifest=args.reference_manifest,
        )
    if args.command == "development-evaluate":
        from iadgen_v2.development_evaluation import run_development_evaluation

        return "development evaluation written to", run_development_evaluation(config, args.metadata or None)
    if args.command == "prepare-locked-benchmark":
        from iadgen_v2.locked_dataset import prepare_locked_benchmark

        return "locked runtime dataset manifest written to", prepare_locked_benchmark(
            config,
            source_root=args.source_root,
            download_source=args.download_source,
        )
    if args.command == "prepare-visa-benchmark":
        from iadgen_v2.visa_dataset import prepare_visa_benchmark

        return "VisA runtime dataset manifest written to", prepare_visa_benchmark(
            config,
            download=args.download,
        )
    if args.command == "freeze-architecture":
        from iadgen_v2.architecture_freeze import freeze_generic_architecture

        return "frozen architecture manifest written to", freeze_generic_architecture(config)
    if args.command == "r6-evidence-package":
        return "R6 evidence package written to", build_r6_evidence_package(config)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
