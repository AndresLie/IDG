"""R1 selector evidence pack — reporting only, NO selector retuning of the
deployed model. Emits leave-category-out calibration (MAE, Pearson, regret) for
the synthetic vs real-candidate selector on an identical candidate pool, with
paired hierarchical bootstrap intervals, plus calibration-size and risk-coverage
curves. The 144-image OOF calibration cohort is kept separate from the 18-image
deployed pilot. Locked categories are never touched.

Usage:
  python scripts/build_selector_evidence_pack.py \
    --metadata outputs/as1b_calib_widen/auto_masks/qwen/metadata.jsonl \
    --synthetic outputs/v3_generic_evidence_development/auto_masks/selector/generic_selector.synthetic_backup.joblib \
    --out reports/selector_evidence_pack
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from iadgen_v2.auto_mask.proposals import MEASUREMENT_NAMES


def gt_for(root, ip, cat, dt):
    return Path(root) / cat / "ground_truth" / dt / f"{Path(ip).stem}_mask.png"


def build_grouped(metadata: Path, root: Path):
    """Return per-image lists of (measurement-vec, actual IoU) plus flat arrays."""
    images = []  # (category, image_id, [(vec, iou), ...])
    for line in Path(metadata).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cat, dt, ip = r["category"], r["defect_type"], r["image_path"]
        gp = gt_for(root, ip, cat, dt)
        if not gp.exists():
            continue
        gt = np.asarray(Image.open(gp).convert("L")) > 127
        s = r["settings"]; cms = s.get("candidate_measurements", {}); crp = s.get("candidate_refined_paths", {})
        cands = []
        for mode, meas in cms.items():
            mp = crp.get(mode)
            if not mp or not Path(mp).exists():
                continue
            m = np.asarray(Image.open(mp).convert("L").resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
            inter = int((m & gt).sum()); pred = int(m.sum()); act = int(gt.sum()); union = pred + act - inter
            cands.append(([float(meas.get(k, 0.0)) for k in MEASUREMENT_NAMES], inter / max(1, union)))
        if cands:
            images.append((cat, f"{dt}/{Path(ip).name}", cands))
    return images


def stats(pred, actual):
    pred = np.clip(np.asarray(pred), 0, 1); actual = np.asarray(actual)
    mae = float(np.mean(np.abs(pred - actual)))
    pear = float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 1e-9 and actual.std() > 1e-9 else 0.0
    return mae, pear


def regret_per_image(images, predict):
    """predict: (category, X[n,D]) -> pred[n]. Returns per-image (category, regret)."""
    out = []
    for cat, _iid, cands in images:
        X = np.asarray([c[0] for c in cands], dtype=np.float32)
        actual = np.asarray([c[1] for c in cands])
        p = predict(cat, X)
        out.append((cat, float(actual.max() - actual[int(np.argmax(p))])))
    return out


def hier_bootstrap(pairs_syn, pairs_real, n=2000, seed=12345):
    """Paired hierarchical bootstrap of mean regret delta (syn - real): resample
    categories, then images within category. Positive => real reduces regret."""
    rng = np.random.default_rng(seed)
    by_cat = {}
    for (cat, rs), (_c, rr) in zip(pairs_syn, pairs_real):
        by_cat.setdefault(cat, []).append(rs - rr)
    cats = list(by_cat)
    means = []
    for _ in range(n):
        drawn_cats = rng.choice(len(cats), len(cats), replace=True)
        deltas = []
        for ci in drawn_cats:
            vals = by_cat[cats[ci]]
            idx = rng.integers(0, len(vals), len(vals))
            deltas.extend(vals[i] for i in idx)
        means.append(float(np.mean(deltas)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean([d for v in by_cat.values() for d in v])), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    images = build_grouped(args.metadata, args.root)
    cats = sorted({c for c, _, _ in images})
    n_cand = sum(len(c) for _, _, c in images)
    # candidate-pool hash (selector-independent): identity of the pool the
    # selectors are compared on.
    pool_sig = hashlib.sha256()
    for cat, iid, cands in images:
        pool_sig.update(f"{cat}|{iid}|{len(cands)}".encode())
        for vec, iou in cands:
            pool_sig.update(np.asarray(vec, dtype=np.float32).tobytes())
    pool_hash = pool_sig.hexdigest()[:16]

    X = np.asarray([v for _, _, cs in images for v, _ in cs], dtype=np.float32)
    y = np.asarray([iou for _, _, cs in images for _, iou in cs])
    groups = np.asarray([c for c, _, cs in images for _ in cs])

    # (1) synthetic selector applied to real candidates (fixed model)
    syn = joblib.load(args.synthetic)
    syn_pred = np.clip(syn["iou_model"].predict(X), 0, 1)
    if syn.get("iou_calibrator") is not None:
        syn_pred = np.clip(syn["iou_calibrator"].predict(syn_pred), 0, 1)
    syn_mae, syn_pear = stats(syn_pred, y)

    # (2) real-candidate selector, leave-category-out OOF (analysis-only fit;
    #     deployed model is untouched)
    est = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.1, random_state=17)
    real_oof = np.clip(cross_val_predict(est, X, y, groups=groups, cv=GroupKFold(n_splits=len(cats))), 0, 1)
    real_mae, real_pear = stats(real_oof, y)

    # per-image regret for each selector on the identical pool
    def syn_predict(cat, Xi):
        p = np.clip(syn["iou_model"].predict(Xi), 0, 1)
        return np.clip(syn["iou_calibrator"].predict(p), 0, 1) if syn.get("iou_calibrator") is not None else p
    # real OOF regret: predict held-out category with a model trained on the others
    real_by_cat = {}
    for held in cats:
        tr = groups != held
        m = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.1, random_state=17).fit(X[tr], y[tr])
        real_by_cat[held] = m
    reg_syn = regret_per_image(images, syn_predict)
    reg_real = regret_per_image(images, lambda cat, Xi: np.clip(real_by_cat[cat].predict(Xi), 0, 1))
    mean_delta, lo, hi = hier_bootstrap(reg_syn, reg_real)

    # (3) calibration-size curve: real selector trained on k categories, OOF regret on held-out
    size_curve = []
    for k in range(1, len(cats)):
        # train on first k categories, evaluate regret on the rest (held-out)
        train_cats = set(cats[:k]); heldout = [im for im in images if im[0] not in train_cats]
        trmask = np.isin(groups, list(train_cats))
        if trmask.sum() < 50 or not heldout:
            continue
        m = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.1, random_state=17).fit(X[trmask], y[trmask])
        reg = regret_per_image(heldout, lambda cat, Xi: np.clip(m.predict(Xi), 0, 1))
        size_curve.append({"train_categories": k, "heldout_mean_regret": float(np.mean([r for _, r in reg]))})

    # (4) risk-coverage: use real OOF max-pred confidence per image; sort by confidence,
    #     report mean regret over the most-confident fraction (coverage).
    conf, per_img_reg = [], []
    offset = 0
    for cat, _iid, cands in images:
        n = len(cands)
        p = real_oof[offset:offset + n]; offset += n
        actual = np.asarray([c[1] for c in cands])
        conf.append(float(p.max())); per_img_reg.append(float(actual.max() - actual[int(np.argmax(p))]))
    order = np.argsort(conf)[::-1]
    risk_cov = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(round(frac * len(order))))
        sel = order[:k]
        risk_cov.append({"coverage": frac, "mean_regret": float(np.mean([per_img_reg[i] for i in sel]))})

    report = {
        "candidate_pool_hash": pool_hash,
        "cohort": {"images": len(images), "candidates": n_cand, "categories": cats,
                   "note": "144-image OOF calibration cohort; distinct from the 18-image deployed pilot"},
        "calibration_144img_oof": {
            "synthetic_selector": {"mae": round(syn_mae, 4), "pearson": round(syn_pear, 4)},
            "real_selector_lco": {"mae": round(real_mae, 4), "pearson": round(real_pear, 4)},
        },
        "regret_144img": {
            "synthetic_mean": round(float(np.mean([r for _, r in reg_syn])), 4),
            "real_lco_mean": round(float(np.mean([r for _, r in reg_real])), 4),
            "paired_delta_syn_minus_real": {"mean": round(mean_delta, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                            "excludes_zero": bool(lo > 0 or hi < 0)},
        },
        "calibration_size_curve": size_curve,
        "risk_coverage_real_lco": risk_cov,
        "deployed_pilot_18img": {"note": "reported separately in reports/current_pipeline_checkpoint; NOT this OOF cohort"},
    }
    (args.out / "selector_evidence_pack.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
