"""R1 selector evidence pack — reporting only, NO selector retuning of the
deployed model. Emits leave-category-out calibration (MAE, Pearson, regret,
selected Dice/IoU) for the synthetic vs real-candidate selector on hash-verified
candidate pools, with paired hierarchical bootstrap intervals per category and
per morphology, plus calibration-size, reliability, and risk-coverage curves. The
144-image OOF calibration cohort is kept separate from the 18-image deployed
pilot. Locked categories are never touched.

The A-S1b (non-widened) and A-S1b+A-S2 (widened) pools are reported side by side;
each synthetic-vs-real comparison is made on an *identical* pool, so deltas are
attributable to the selector, not to a different candidate set.

Usage:
  python scripts/build_selector_evidence_pack.py \
    --metadata outputs/as1b_calib_widen/auto_masks/qwen/metadata.jsonl \
    --metadata-nonwiden outputs/as1b_calib/auto_masks/qwen/metadata.jsonl \
    --synthetic outputs/v3_generic_evidence_development/auto_masks/selector/generic_selector.synthetic_backup.joblib \
    --deployed outputs/v3_generic_evidence_development/auto_masks/selector/generic_selector.joblib \
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

HGB = dict(max_iter=160, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.1, random_state=17)


def gt_for(root, ip, cat, dt):
    return Path(root) / cat / "ground_truth" / dt / f"{Path(ip).stem}_mask.png"


def build_grouped(metadata: Path, root: Path):
    """Per-image list of (category, morphology, image_id, [(vec, iou, dice), ...])."""
    images = []
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
            inter = int((m & gt).sum()); pred = int(m.sum()); act = int(gt.sum())
            union = pred + act - inter
            iou = inter / max(1, union); dice = (2 * inter) / max(1, pred + act)
            cands.append(([float(meas.get(k, 0.0)) for k in MEASUREMENT_NAMES], iou, dice))
        if cands:
            images.append((cat, dt, f"{dt}/{Path(ip).name}", cands))
    return images


def stats(pred, actual):
    pred = np.clip(np.asarray(pred), 0, 1); actual = np.asarray(actual)
    mae = float(np.mean(np.abs(pred - actual)))
    pear = float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 1e-9 and actual.std() > 1e-9 else 0.0
    return round(mae, 4), round(pear, 4)


def load_selector(path):
    return joblib.load(path)


def syn_predict(syn, Xi):
    p = np.clip(syn["iou_model"].predict(Xi), 0, 1)
    if syn.get("iou_calibrator") is not None:
        p = np.clip(syn["iou_calibrator"].predict(p), 0, 1)
    return p


def per_image_selection(images, predict):
    """Return list of dicts with the selected candidate's regret / selected iou&dice."""
    rows = []
    for cat, morph, _iid, cands in images:
        X = np.asarray([c[0] for c in cands], dtype=np.float32)
        iou = np.asarray([c[1] for c in cands]); dice = np.asarray([c[2] for c in cands])
        j = int(np.argmax(predict(cat, X)))
        rows.append({"cat": cat, "morph": morph,
                     "regret": float(iou.max() - iou[j]),
                     "sel_iou": float(iou[j]), "sel_dice": float(dice[j]),
                     "oracle_iou": float(iou.max())})
    return rows


def hier_bootstrap_delta(rows_syn, rows_real, unit, field="regret", n=2000, seed=12345):
    """Paired hierarchical bootstrap of mean (syn - real) for `field`, resampling
    `unit` groups then images within group. Positive regret-delta => real better."""
    rng = np.random.default_rng(seed)
    by = {}
    for a, b in zip(rows_syn, rows_real):
        by.setdefault(a[unit], []).append(a[field] - b[field])
    keys = list(by)
    means = []
    for _ in range(n):
        drawn = rng.choice(len(keys), len(keys), replace=True)
        vals = []
        for gi in drawn:
            g = by[keys[gi]]
            idx = rng.integers(0, len(g), len(g))
            vals.extend(g[i] for i in idx)
        means.append(float(np.mean(vals)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    pt = float(np.mean([d for v in by.values() for d in v]))
    return round(pt, 4), round(float(lo), 4), round(float(hi), 4)


def grouped_delta(rows_syn, rows_real, unit, field="regret"):
    """Point delta (syn - real) per group + image-level bootstrap CI within group."""
    out = {}
    keys = sorted({r[unit] for r in rows_syn})
    for k in keys:
        s = [a[field] for a in rows_syn if a[unit] == k]
        r = [b[field] for b in rows_real if b[unit] == k]
        d = np.asarray(s) - np.asarray(r)
        rng = np.random.default_rng(777)
        boot = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(1000)] if len(d) > 1 else [float(d.mean())]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[k] = {"n": len(d), "delta": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    return out


def pool_hash(images):
    h = hashlib.sha256()
    for cat, morph, iid, cands in images:
        h.update(f"{cat}|{iid}|{len(cands)}".encode())
        for vec, iou, dice in cands:
            h.update(np.asarray(vec, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def evaluate_pool(metadata, root, syn):
    images = build_grouped(metadata, root)
    cats = sorted({c for c, _, _, _ in images})
    X = np.asarray([v for _, _, _, cs in images for v, _, _ in cs], dtype=np.float32)
    y = np.asarray([iou for _, _, _, cs in images for _, iou, _ in cs])
    groups = np.asarray([c for c, _, _, cs in images for _ in cs])

    syn_mae, syn_pear = stats(syn_predict(syn, X), y)
    est = HistGradientBoostingRegressor(**HGB)
    real_oof = np.clip(cross_val_predict(est, X, y, groups=groups, cv=GroupKFold(n_splits=len(cats))), 0, 1)
    real_mae, real_pear = stats(real_oof, y)

    real_by_cat = {}
    for held in cats:
        tr = groups != held
        real_by_cat[held] = HistGradientBoostingRegressor(**HGB).fit(X[tr], y[tr])
    rows_syn = per_image_selection(images, lambda cat, Xi: syn_predict(syn, Xi))
    rows_real = per_image_selection(images, lambda cat, Xi: np.clip(real_by_cat[cat].predict(Xi), 0, 1))

    reg_pt, reg_lo, reg_hi = hier_bootstrap_delta(rows_syn, rows_real, unit="cat", field="regret")

    # calibration-size curve
    size_curve = []
    for k in range(1, len(cats)):
        train = set(cats[:k]); heldout = [im for im in images if im[0] not in train]
        trmask = np.isin(groups, list(train))
        if trmask.sum() < 50 or not heldout:
            continue
        m = HistGradientBoostingRegressor(**HGB).fit(X[trmask], y[trmask])
        reg = per_image_selection(heldout, lambda cat, Xi: np.clip(m.predict(Xi), 0, 1))
        size_curve.append({"train_categories": k, "heldout_mean_regret": round(float(np.mean([r["regret"] for r in reg])), 4)})

    # reliability curve (real OOF): mean actual iou per predicted-iou decile
    reliability = []
    order = np.argsort(real_oof)
    for b in range(10):
        seg = order[b * len(order) // 10:(b + 1) * len(order) // 10]
        if len(seg):
            reliability.append({"bin": b, "pred_mean": round(float(real_oof[seg].mean()), 3),
                                "actual_mean": round(float(y[seg].mean()), 3), "n": len(seg)})

    # risk-coverage (real OOF confidence = max pred per image)
    conf, per_reg, offset = [], [], 0
    for cat, morph, _iid, cands in images:
        n = len(cands); p = real_oof[offset:offset + n]; offset += n
        iou = np.asarray([c[1] for c in cands])
        conf.append(float(p.max())); per_reg.append(float(iou.max() - iou[int(np.argmax(p))]))
    ranked = np.argsort(conf)[::-1]
    risk_cov = [{"coverage": f, "mean_regret": round(float(np.mean([per_reg[i] for i in ranked[:max(1, int(round(f * len(ranked))))]])), 4)}
                for f in (0.25, 0.5, 0.75, 1.0)]

    return {
        "candidate_pool_hash": pool_hash(images),
        "cohort": {"images": len(images), "candidates": int(len(y)), "categories": cats},
        "calibration_oof": {"synthetic": {"mae": syn_mae, "pearson": syn_pear},
                            "real_lco": {"mae": real_mae, "pearson": real_pear}},
        "selection": {
            "synthetic": {"mean_regret": round(float(np.mean([r["regret"] for r in rows_syn])), 4),
                          "mean_selected_iou": round(float(np.mean([r["sel_iou"] for r in rows_syn])), 4),
                          "mean_selected_dice": round(float(np.mean([r["sel_dice"] for r in rows_syn])), 4)},
            "real_lco": {"mean_regret": round(float(np.mean([r["regret"] for r in rows_real])), 4),
                         "mean_selected_iou": round(float(np.mean([r["sel_iou"] for r in rows_real])), 4),
                         "mean_selected_dice": round(float(np.mean([r["sel_dice"] for r in rows_real])), 4)},
            "oracle_mean_iou": round(float(np.mean([r["oracle_iou"] for r in rows_syn])), 4),
        },
        "regret_delta_syn_minus_real": {"mean": reg_pt, "ci95": [reg_lo, reg_hi], "excludes_zero": bool(reg_lo > 0 or reg_hi < 0)},
        "per_category_regret_delta": grouped_delta(rows_syn, rows_real, unit="cat"),
        "per_morphology_regret_delta": grouped_delta(rows_syn, rows_real, unit="morph"),
        "per_morphology_hier_bootstrap": dict(zip(("mean", "ci95_lo", "ci95_hi"),
                                                  hier_bootstrap_delta(rows_syn, rows_real, unit="morph"))),
        "calibration_size_curve": size_curve,
        "reliability_curve_real_lco": reliability,
        "risk_coverage_real_lco": risk_cov,
    }


def file_hash(path):
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True, help="widened pool (A-S1b+A-S2)")
    ap.add_argument("--metadata-nonwiden", type=Path, default=None, help="non-widened pool (A-S1b)")
    ap.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--deployed", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    syn = load_selector(args.synthetic)
    report = {
        "selector_parent_hashes": {"synthetic_backup": file_hash(args.synthetic),
                                   "deployed": file_hash(args.deployed)},
        "cohort_separation_note": "OOF calibration cohort below is the 144-image dev set; the 18-image deployed pilot is reported separately in reports/current_pipeline_checkpoint and is NOT mixed in.",
        "structure_profile_caveat": "settings.structure_profile is degenerate on this cohort (ring_sector-dominant, no linear/edge classes); per-morphology deltas use defect_type, not structure_profile.",
        "variants": {
            "A-S1b+A-S2 (widened, deployed pool)": evaluate_pool(args.metadata, args.root, syn),
        },
    }
    if args.metadata_nonwiden and args.metadata_nonwiden.exists():
        report["variants"]["A-S1b (non-widened pool)"] = evaluate_pool(args.metadata_nonwiden, args.root, syn)

    (args.out / "selector_evidence_pack.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
