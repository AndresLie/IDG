"""Strictly-additive PCA evaluation, within a single additive pool.

Loads one pool generated with pca_subspace_additive=true (baseline candidates
preserved + PCA-residual candidates appended), then forms two views of the SAME
pool — baseline-only (PCA modes removed) and full — and LOCO-evaluates each with
a real-candidate selector refit. Because both views share identical baseline
candidates, the comparison is strictly additive: the oracle can only rise, so any
selected-Dice change is attributable to the selector using PCA candidates, not to
perturbed baseline candidates.

Evaluation-only: reads existing candidate masks + official development masks.

Usage:
  python scripts/evaluate_additive_pca_loco.py \
    --additive-metadata outputs/as1b_calib_widen_pca_additive/auto_masks/qwen/metadata.jsonl \
    --data-root data/mvtec_ad --out reports/additive_pca_loco/result.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_pca_loco_ab import (  # noqa: E402
    Sample, _load_pool, _loco_evaluate, _aggregate, _paired_bootstrap,
)

PCA_TAG = "dinov2_subspace"  # additive PCA candidate modes (raw + edge_-refined)


def strip_pca(samples: list[Sample]) -> list[Sample]:
    out: list[Sample] = []
    for s in samples:
        cands = [c for c in s.candidates if PCA_TAG not in c.mode]
        if cands:
            out.append(Sample(s.sample_id, s.category, cands))
    return out


def monotone_oracle_check(baseline: list[Sample], full: list[Sample]) -> dict:
    b = {s.sample_id: max(c.iou for c in s.candidates) for s in baseline}
    f = {s.sample_id: max(c.iou for c in s.candidates) for s in full}
    shared = sorted(set(b) & set(f))
    violations = [sid for sid in shared if f[sid] + 1e-9 < b[sid]]
    raised = [sid for sid in shared if f[sid] > b[sid] + 1e-9]
    return {"images": len(shared), "oracle_never_regresses": not violations,
            "violations": violations[:5], "images_oracle_raised_by_pca": len(raised)}


def per_category_dice_delta(base_agg, full_agg) -> dict:
    cats = [k for k in base_agg if k not in ("overall", "category_macro")]
    return {c: round(full_agg[c]["selected_dice"] - base_agg[c]["selected_dice"], 4) for c in sorted(cats)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--additive-metadata", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    full_samples, pool_meta = _load_pool(args.additive_metadata, args.data_root)
    baseline_samples = strip_pca(full_samples)

    n_pca = sum(len(s.candidates) for s in full_samples) - sum(len(s.candidates) for s in baseline_samples)
    base_records, base_cal = _loco_evaluate(baseline_samples)
    full_records, full_cal = _loco_evaluate(full_samples)
    base_agg, full_agg = _aggregate(base_records), _aggregate(full_records)

    base_by_id = {r["sample_id"]: r for r in base_records}
    full_by_id = {r["sample_id"]: r for r in full_records}
    boot = _paired_bootstrap(base_by_id, full_by_id)  # delta = full(additive) - baseline

    report = {
        "pool": pool_meta,
        "pca_candidates_appended": n_pca,
        "strict_additivity_oracle_check": monotone_oracle_check(baseline_samples, full_samples),
        "baseline_only": {"category_macro_dice": round(base_agg["category_macro"]["selected_dice"], 4),
                          "overall_dice": round(base_agg["overall"]["selected_dice"], 4),
                          "macro_regret": round(base_agg["category_macro"]["dice_regret"], 4),
                          "calibration": {k: round(v, 4) for k, v in base_cal.items()}},
        "additive_full": {"category_macro_dice": round(full_agg["category_macro"]["selected_dice"], 4),
                         "overall_dice": round(full_agg["overall"]["selected_dice"], 4),
                         "macro_regret": round(full_agg["category_macro"]["dice_regret"], 4),
                         "calibration": {k: round(v, 4) for k, v in full_cal.items()}},
        "per_category_selected_dice_delta": per_category_dice_delta(base_agg, full_agg),
        "paired_bootstrap_selected_dice_delta_full_minus_baseline": boot,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
