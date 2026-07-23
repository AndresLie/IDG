"""Build a real-candidate selector calibration set and fit a selector bundle.

Firewall: candidates were generated WITHOUT official masks; this script reads
development-category official masks ONLY afterward to compute IoU labels. Use
--exclude to hold categories out for leave-category-out validation.

Usage:
  python scripts/build_real_selector_calibration.py \
      --metadata outputs/as1b_calib/auto_masks/qwen/metadata.jsonl \
      --data-root data/mvtec_ad \
      --exclude bottle zipper \
      --out outputs/as1b_calib/selector/real_lco_selector.joblib
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from iadgen_v2.auto_mask.proposals import MEASUREMENT_NAMES
from iadgen_v2.auto_mask.selection import fit_selector_bundle


def gt_for(data_root: Path, image_path: str, category: str, defect_type: str) -> Path:
    return data_root / category / "ground_truth" / defect_type / f"{Path(image_path).stem}_mask.png"


def build_rows(metadata: Path, data_root: Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(metadata).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cat, dt, ip = r["category"], r["defect_type"], r["image_path"]
        gp = gt_for(data_root, ip, cat, dt)
        if not gp.exists():
            continue
        gt = np.asarray(Image.open(gp).convert("L")) > 127
        s = r["settings"]
        cms = s.get("candidate_measurements", {}); crp = s.get("candidate_refined_paths", {})
        for mode, meas in cms.items():
            mp = crp.get(mode)
            if not mp or not Path(mp).exists():
                continue
            m = np.asarray(Image.open(mp).convert("L").resize((gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
            inter = int((m & gt).sum()); pred = int(m.sum()); act = int(gt.sum()); union = pred + act - inter
            rows.append({
                "category": cat, "corruption_family": "real",
                **{k: float(meas.get(k, 0.0)) for k in MEASUREMENT_NAMES},
                "iou": inter / max(1, union),
                "precision": inter / max(1, pred),
                "recall": inter / max(1, act),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = build_rows(args.metadata, args.data_root)
    kept = [r for r in rows if r["category"] not in set(args.exclude)]
    cats = sorted({r["category"] for r in kept})
    print(f"real rows total={len(rows)} kept={len(kept)} (excluded {args.exclude}) train categories={cats}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fit_selector_bundle(kept, args.out)
    print(f"fitted real-data selector -> {args.out}")


if __name__ == "__main__":
    main()
