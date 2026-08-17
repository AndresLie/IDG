"""Does a segmentation student denoise our pseudo-labels?

This is the Boxes2Pixels protocol without the bounding box: train a segmentation
model on the pipeline's *pseudo-masks* only, then score the student's prediction
against official masks on a HELD-OUT category. If the student beats the
pseudo-labels it was never shown, the pipeline's publishable mask quality is the
student's output, not the raw pseudo-label.

Firewall: official masks are NEVER used for training or model selection. They are
opened only to score (a) the pseudo-label and (b) the student prediction on the
held-out category. Development categories only.

Deterministic: fixed seeds, deterministic torch algorithms, fixed step count.

Usage:
  python scripts/evaluate_pseudo_label_student.py \
    --metadata outputs/as1b_calib_widen/auto_masks/qwen/metadata.jsonl \
    --data-root data/mvtec_ad --out reports/pseudo_label_student/result.json
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np
from PIL import Image

SIZE = 256
STEPS = 300
BATCH = 8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def load_samples(metadata: Path, root: Path) -> list[dict]:
    """Each sample: image, pseudo-mask (training input), official mask (scoring only)."""
    out = []
    for line in metadata.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = Path(r["image_path"])
        pseudo = Path(str(r.get("eval_mask_path") or r.get("refined_mask_path") or ""))
        official = root / r["category"] / "ground_truth" / r["defect_type"] / f"{img.stem}_mask.png"
        if not (img.exists() and pseudo.exists() and official.exists()):
            continue
        out.append({"category": r["category"], "defect": r["defect_type"], "image": img,
                    "pseudo": pseudo, "official": official})
    return out


def load_pair(s: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(Image.open(s["image"]).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    pse = np.asarray(Image.open(s["pseudo"]).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127
    off = np.asarray(Image.open(s["official"]).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127
    return img, pse, off


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    return 2.0 * inter / max(1, int(a.sum()) + int(b.sum()))


def build_student():
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    class Student(nn.Module):
        """ResNet18-encoder U-Net; same family as the R5 evaluator."""

        def __init__(self) -> None:
            super().__init__()
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # /2, 64
            self.pool = net.maxpool                                    # /4
            self.l1, self.l2, self.l3, self.l4 = net.layer1, net.layer2, net.layer3, net.layer4
            def up(cin, cout):
                return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
            self.d4, self.d3, self.d2, self.d1 = up(512 + 256, 256), up(256 + 128, 128), up(128 + 64, 64), up(64 + 64, 64)
            self.head = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            import torch.nn.functional as F
            s = self.stem(x)                 # /2
            c1 = self.l1(self.pool(s))       # /4
            c2 = self.l2(c1)                 # /8
            c3 = self.l3(c2)                 # /16
            c4 = self.l4(c3)                 # /32
            def merge(dec, feat, block):
                dec = F.interpolate(dec, size=feat.shape[-2:], mode="bilinear", align_corners=False)
                import torch
                return block(torch.cat([dec, feat], dim=1))
            y = merge(c4, c3, self.d4)
            y = merge(y, c2, self.d3)
            y = merge(y, c1, self.d2)
            y = merge(y, s, self.d1)
            y = F.interpolate(y, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
            return self.head(y)

    return Student()


def train_and_predict(train: list[dict], test: list[dict], seed: int) -> list[np.ndarray]:
    """Train on PSEUDO labels only; return per-test-image binary predictions."""
    import torch
    import torch.nn.functional as F
    seed_all(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_student().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    cache = [load_pair(s)[:2] for s in train]          # (image, pseudo) — official never touched
    xs = torch.from_numpy(np.stack([c[0] for c in cache])).permute(0, 3, 1, 2)
    ys = torch.from_numpy(np.stack([c[1] for c in cache]).astype(np.float32)).unsqueeze(1)
    pos_weight = torch.tensor([float((ys.numel() - ys.sum()) / max(1.0, ys.sum().item()))], device=dev).clamp(1, 50)

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(STEPS):
        idx = rng.integers(0, len(cache), min(BATCH, len(cache)))
        xb, yb = xs[idx].to(dev), ys[idx].to(dev)
        logit = model(xb)
        loss = F.binary_cross_entropy_with_logits(logit, yb, pos_weight=pos_weight)
        p = torch.sigmoid(logit)
        loss = loss + (1 - (2 * (p * yb).sum() + 1) / ((p + yb).sum() + 1))   # + soft Dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for s in test:
            img, _, _ = load_pair(s)
            x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev)
            preds.append((torch.sigmoid(model(x))[0, 0].cpu().numpy() >= 0.5))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/mvtec_ad"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument(
        "--mode", choices=("category", "image"), default="category",
        help="category = leave-category-out (tests transfer to an unseen category). "
             "image = leave-image-out folds with every category present in training "
             "(tests in-distribution label denoising, the Boxes2Pixels analogue).",
    )
    args = ap.parse_args()

    samples = load_samples(args.metadata, args.data_root)
    if args.mode == "category":
        groups = {c: [s for s in samples if s["category"] == c] for c in sorted({s["category"] for s in samples})}
    else:
        # Stratified 2-fold image split: every category appears in train and test.
        by_cat: dict[str, list[dict]] = {}
        for s in sorted(samples, key=lambda s: (s["category"], str(s["image"]))):
            by_cat.setdefault(s["category"], []).append(s)
        groups = {"fold_a": [], "fold_b": []}
        for cat_samples in by_cat.values():
            for i, s in enumerate(cat_samples):
                groups["fold_a" if i % 2 == 0 else "fold_b"].append(s)
    per_cat: dict[str, dict] = {}
    for held, test in groups.items():
        train = [s for s in samples if s not in test]
        pseudo_d = [dice(load_pair(s)[1], load_pair(s)[2]) for s in test]
        student_by_seed = []
        for seed in args.seeds:
            preds = train_and_predict(train, test, seed)
            student_by_seed.append([dice(p, load_pair(s)[2]) for p, s in zip(preds, test)])
        student_mean = np.mean(student_by_seed, axis=0)
        per_cat[held] = {
            "images": len(test),
            "pseudo_label_dice": round(float(np.mean(pseudo_d)), 4),
            "student_dice": round(float(np.mean(student_mean)), 4),
            "delta": round(float(np.mean(student_mean) - np.mean(pseudo_d)), 4),
            "per_seed_student_dice": [round(float(np.mean(s)), 4) for s in student_by_seed],
        }
        print(f"{held:16} pseudo {per_cat[held]['pseudo_label_dice']:.4f} -> student {per_cat[held]['student_dice']:.4f} ({per_cat[held]['delta']:+.4f})")

    macro_p = float(np.mean([v["pseudo_label_dice"] for v in per_cat.values()]))
    macro_s = float(np.mean([v["student_dice"] for v in per_cat.values()]))
    # paired hierarchical bootstrap over categories of the macro delta
    rng = np.random.default_rng(20260811)
    deltas = np.array([v["delta"] for v in per_cat.values()])
    boot = [float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))])) for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    report = {
        "protocol": "train on pseudo-labels only; official masks used solely to score held-out samples",
        "mode": args.mode,
        "mode_meaning": ("transfer to an unseen category" if args.mode == "category"
                         else "in-distribution label denoising (Boxes2Pixels analogue)"),
        "steps": STEPS, "image_size": SIZE, "seeds": args.seeds,
        "leave_category_out": per_cat,
        "macro_pseudo_label_dice": round(macro_p, 4),
        "macro_student_dice": round(macro_s, 4),
        "macro_delta": round(macro_s - macro_p, 4),
        "macro_delta_ci95": [round(float(lo), 4), round(float(hi), 4)],
        "student_denoises": bool(lo > 0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "leave_category_out"}, indent=2))


if __name__ == "__main__":
    main()
