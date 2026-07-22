from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = ROOT / "configs" / "mvtec_bottle_zipper_auto_mask.yaml"
REPORT_ROOT = ROOT / "reports" / "mvtec_bottle_zipper_foundation_ablation"
BASELINE_REPORT_DIR = ROOT / "reports" / "mvtec_bottle_zipper_auto_mask" / "official_mask_evaluation"
BASELINE_METADATA = ROOT / "outputs" / "mvtec_bottle_zipper_auto_mask" / "auto_masks" / "qwen" / "metadata.jsonl"


VARIANTS = {
    "baseline": {
        "title": "Primary baseline, no foundation",
        "config": BASE_CONFIG_PATH,
        "output_dir": ROOT / "outputs" / "mvtec_bottle_zipper_auto_mask",
        "report_dir": ROOT / "reports" / "mvtec_bottle_zipper_auto_mask",
        "metadata": BASELINE_METADATA,
        "evaluation": BASELINE_REPORT_DIR,
        "run": False,
    },
    "foundation_diagnostic": {
        "title": "Foundation generated, selector bonus scaled to zero",
        "config": ROOT / "configs" / "mvtec_bottle_zipper_foundation_diagnostic.yaml",
        "output_dir": ROOT / "outputs" / "mvtec_bottle_zipper_foundation_diagnostic",
        "report_dir": ROOT / "reports" / "mvtec_bottle_zipper_foundation_diagnostic",
        "metadata": ROOT / "outputs" / "mvtec_bottle_zipper_foundation_diagnostic" / "auto_masks" / "qwen" / "metadata.jsonl",
        "evaluation": ROOT / "reports" / "mvtec_bottle_zipper_foundation_diagnostic" / "official_mask_evaluation",
        "foundation_selector_bonus_scale": 0.0,
        "run": True,
    },
    "foundation_select": {
        "title": "Foundation generated and selector-eligible",
        "config": ROOT / "configs" / "mvtec_bottle_zipper_foundation_select.yaml",
        "output_dir": ROOT / "outputs" / "mvtec_bottle_zipper_foundation_select",
        "report_dir": ROOT / "reports" / "mvtec_bottle_zipper_foundation_select",
        "metadata": ROOT / "outputs" / "mvtec_bottle_zipper_foundation_select" / "auto_masks" / "qwen" / "metadata.jsonl",
        "evaluation": ROOT / "reports" / "mvtec_bottle_zipper_foundation_select" / "official_mask_evaluation",
        "foundation_selector_bonus_scale": 1.0,
        "run": True,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or summarize MVTec bottle/zipper foundation ablation.")
    parser.add_argument("--write-configs", action="store_true", help="Write ablation YAML configs.")
    parser.add_argument("--run", action="store_true", help="Run auto-masks and evaluation for selected variants.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[name for name in VARIANTS if name != "baseline"],
        help="Variant to run. Repeatable. Defaults to both foundation variants.",
    )
    parser.add_argument("--report", action="store_true", help="Write comparison report.")
    args = parser.parse_args()

    if args.write_configs:
        write_configs()

    if args.run:
        variants = args.variant or ["foundation_diagnostic", "foundation_select"]
        for variant in variants:
            run_variant(variant)

    if args.report or not (args.write_configs or args.run):
        write_report()

    return 0


def write_configs() -> None:
    base = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    for name, spec in VARIANTS.items():
        if name == "baseline":
            continue
        config = deepcopy(base)
        config["project"]["output_dir"] = str(Path(spec["output_dir"]).relative_to(ROOT))
        config["project"]["report_dir"] = str(Path(spec["report_dir"]).relative_to(ROOT))
        auto = config.setdefault("auto_masks", {})
        modes = list(auto.get("auto_candidate_modes", []))
        if "foundation_anomaly_field" not in modes:
            insert_at = modes.index("normal_anomaly") + 1 if "normal_anomaly" in modes else 0
            modes.insert(insert_at, "foundation_anomaly_field")
        auto["auto_candidate_modes"] = modes
        auto["enable_patch_feature_cache"] = True
        auto["foundation_selector_bonus_scale"] = float(spec["foundation_selector_bonus_scale"])
        auto["foundation_patchcore_weight"] = 0.34
        auto["foundation_musc_weight"] = 0.24
        auto["foundation_nearest_residual_weight"] = 0.20
        auto["foundation_normal_texture_weight"] = 0.14
        auto["foundation_fft_weight"] = 0.08
        auto["foundation_patchcore_patch_size"] = auto.get("patchcore_guided_patch_size", 17)
        auto["foundation_patchcore_stride"] = auto.get("patchcore_guided_stride", 6)
        auto["foundation_musc_patch_size"] = auto.get("musc_patch_size", auto.get("patchcore_guided_patch_size", 17))
        auto["foundation_musc_stride"] = auto.get("musc_stride", auto.get("patchcore_guided_stride", 6))
        auto["foundation_max_normals"] = auto.get("patchcore_guided_max_normals", 16)
        auto["foundation_max_memory_patches"] = auto.get("patchcore_guided_max_memory_patches", 4096)
        auto["foundation_percentile"] = auto.get("patchcore_guided_percentile", 88.0)
        auto["foundation_close_radius"] = auto.get("patchcore_guided_close_radius", 2)
        auto["foundation_dilate_radius"] = auto.get("patchcore_guided_dilate_radius", 1)
        auto["foundation_max_components"] = auto.get("patchcore_guided_max_components", 14)
        auto["foundation_max_box_fraction"] = auto.get("patchcore_guided_max_box_fraction", 0.18)
        auto["foundation_min_contrast"] = auto.get("patchcore_guided_min_contrast", 0.025)
        config_path = Path(spec["config"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(config_path.relative_to(ROOT))


def run_variant(name: str) -> None:
    spec = VARIANTS[name]
    config_path = Path(spec["config"])
    if not config_path.exists():
        write_configs()
    start = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "iadgen_v2.cli", "auto-masks", "--config", str(config_path.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
    )
    elapsed = time.perf_counter() - start
    elapsed_path = Path(spec["evaluation"]) / "runtime_seconds.txt"
    elapsed_path.parent.mkdir(parents=True, exist_ok=True)
    elapsed_path.write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_mvtec_bottle_zipper_auto_masks.py",
            "--metadata-path",
            str(Path(spec["metadata"]).relative_to(ROOT)),
            "--report-dir",
            str(Path(spec["evaluation"]).relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
    )


def load_metrics(evaluation_dir: Path) -> list[dict[str, Any]]:
    path = evaluation_dir / "auto_mask_metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        return []
    return [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize_variant(name: str) -> dict[str, Any]:
    spec = VARIANTS[name]
    metrics = load_metrics(Path(spec["evaluation"]))
    metadata = load_metadata(Path(spec["metadata"]))
    selected = Counter(row.get("selected_refinement", "") for row in metrics)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        grouped[str(row["category"])].append(row)
    foundation_candidates = 0
    foundation_selected = 0
    cache_hits = 0
    cache_misses = 0
    foundation_oracle_best = 0
    for row in metadata:
        settings = row.get("settings", {})
        candidate_paths = settings.get("candidate_refined_paths", {})
        if "foundation_anomaly_field" in candidate_paths:
            foundation_candidates += 1
        if settings.get("selected_refinement") == "foundation_anomaly_field":
            foundation_selected += 1
        params = settings.get("mask_parameters", {})
        for key, value in params.items():
            if key.endswith("_feature_cache_hits"):
                cache_hits = max(cache_hits, int(value))
            elif key.endswith("_feature_cache_misses"):
                cache_misses = max(cache_misses, int(value))
    for row in metrics:
        if row.get("best_candidate_mode_oracle") == "foundation_anomaly_field":
            foundation_oracle_best += 1
    runtime_path = Path(spec["evaluation"]) / "runtime_seconds.txt"
    runtime_seconds = float(runtime_path.read_text(encoding="utf-8").strip()) if runtime_path.exists() else None
    return {
        "metrics": metrics,
        "metadata": metadata,
        "selected": selected,
        "bottle": grouped.get("bottle", []),
        "zipper": grouped.get("zipper", []),
        "foundation_candidates": foundation_candidates,
        "foundation_selected": foundation_selected,
        "foundation_oracle_best": foundation_oracle_best,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "runtime_seconds": runtime_seconds,
    }


def write_report() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {name: summarize_variant(name) for name in VARIANTS}
    lines = [
        "# Foundation Candidate Ablation",
        "",
        "## Protocol",
        "",
        "- Official MVTec masks are used only by `scripts/evaluate_mvtec_bottle_zipper_auto_masks.py` after auto-mask generation.",
        "- The primary baseline is the existing no-foundation bottle/zipper run.",
        "- `foundation_diagnostic` generates `foundation_anomaly_field` candidates but scales selector bonuses to `0.0`.",
        "- `foundation_select` generates the same candidate and leaves selector bonuses at normal scale.",
        "",
        "## Aggregate Metrics",
        "",
        "| Variant | Bottle Dice | Bottle Regret | Zipper Dice | Zipper Regret | Foundation selected | Foundation oracle-best | Max cache hits/misses | Runtime |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, spec in VARIANTS.items():
        summary = summaries[name]
        bottle = summary["bottle"]
        zipper = summary["zipper"]
        runtime = summary["runtime_seconds"]
        if runtime is None and summary["metrics"]:
            runtime_text = "cached/reselect"
        else:
            runtime_text = "pending" if runtime is None else f"{runtime / 60.0:.1f} min"
        lines.append(
            f"| `{name}` | {mean(bottle, 'dice'):.4f} | {mean(bottle, 'selection_dice_regret'):.4f} | "
            f"{mean(zipper, 'dice'):.4f} | {mean(zipper, 'selection_dice_regret'):.4f} | "
            f"{summary['foundation_selected']} | {summary['foundation_oracle_best']} | "
            f"{summary['cache_hits']}/{summary['cache_misses']} | {runtime_text} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            *_decision_lines(summaries),
            "",
            "## Selected Modes",
            "",
        ]
    )
    for name, summary in summaries.items():
        selected = ", ".join(f"`{mode}`: {count}" for mode, count in sorted(summary["selected"].items()))
        lines.append(f"- `{name}`: {selected or 'pending'}")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "Promote `foundation_anomaly_field` only if:",
            "",
            "```text",
            "Bottle Dice >= 0.7361",
            "Zipper Dice > 0.6229",
            "Regret remains close to baseline",
            "Runtime remains practical",
            "```",
            "",
            "If the diagnostic variant improves oracle-best counts but not selected Dice, keep foundation as a diagnostic candidate and improve selector routing later.",
            "",
        ]
    )
    (REPORT_ROOT / "foundation_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_ROOT / "foundation_ablation_report.md")


def _decision_lines(summaries: dict[str, dict[str, Any]]) -> list[str]:
    baseline = summaries["baseline"]
    diagnostic = summaries["foundation_diagnostic"]
    select = summaries["foundation_select"]
    baseline_bottle = mean(baseline["bottle"], "dice")
    baseline_zipper = mean(baseline["zipper"], "dice")
    select_bottle = mean(select["bottle"], "dice")
    select_zipper = mean(select["zipper"], "dice")
    if not select["metrics"]:
        return [
            "The selector-eligible foundation run is pending. Do not promote until A/B/C metrics are complete.",
        ]
    promote = select_bottle >= 0.7361 and select_zipper > baseline_zipper
    if promote:
        return [
            "Promote `foundation_anomaly_field` into the primary bottle/zipper config.",
            "",
            f"- Bottle Dice is stable: `{select_bottle:.4f}` versus baseline `{baseline_bottle:.4f}`.",
            f"- Zipper Dice improves: `{select_zipper:.4f}` versus baseline `{baseline_zipper:.4f}`.",
        ]
    return [
        "Do **not** promote `foundation_anomaly_field` into the primary bottle/zipper config yet.",
        "",
        f"- Bottle Dice is stable: `{select_bottle:.4f}` versus baseline `{baseline_bottle:.4f}`.",
        f"- Zipper Dice regresses: `{select_zipper:.4f}` versus baseline `{baseline_zipper:.4f}`.",
        f"- Foundation is selected `{select['foundation_selected']}` time(s).",
        f"- Foundation is oracle-best only `{select['foundation_oracle_best']}` time(s).",
        f"- Diagnostic runtime was `{diagnostic['runtime_seconds'] / 60.0:.1f}` minutes." if diagnostic["runtime_seconds"] else "- Diagnostic runtime was not recorded.",
        "",
        "Interpretation: the candidate is useful as an ablation/diagnostic source, but it does not currently improve the full selected-mask benchmark. Keep it out of the production config and move the next improvement toward the remaining weak zipper localization/structure cases.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
