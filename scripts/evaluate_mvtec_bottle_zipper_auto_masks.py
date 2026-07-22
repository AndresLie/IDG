from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iadgen_v2.auto_masks import _candidate_mode_applicable


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "mvtec_bottle_zipper_auto_mask"
METADATA_PATH = ROOT / "outputs" / "mvtec_bottle_zipper_auto_mask" / "auto_masks" / "qwen" / "metadata.jsonl"
REPORT_DIR = ROOT / "reports" / "mvtec_bottle_zipper_auto_mask" / "official_mask_evaluation"
CSV_PATH = REPORT_DIR / "auto_mask_metrics.csv"
REPORT_PATH = REPORT_DIR / "bottle_zipper_auto_mask_review.md"
SHEET_PATH = REPORT_DIR / "bottle_zipper_auto_mask_visual_review.png"


def main() -> None:
    global DATA_ROOT, METADATA_PATH, REPORT_DIR, CSV_PATH, REPORT_PATH, SHEET_PATH
    parser = argparse.ArgumentParser(description="Evaluate bottle/zipper auto masks against isolated official masks.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    METADATA_PATH = args.metadata_path
    REPORT_DIR = args.report_dir
    CSV_PATH = REPORT_DIR / "auto_mask_metrics.csv"
    REPORT_PATH = REPORT_DIR / "bottle_zipper_auto_mask_review.md"
    SHEET_PATH = REPORT_DIR / "bottle_zipper_auto_mask_visual_review.png"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((DATA_ROOT / "pilot_manifest.json").read_text(encoding="utf-8"))
    references = {
        (record["category"], record["defect_type"], Path(record["target_image"]).name): record
        for record in pilot["records"]
        if record["role"] == "defect"
    }
    rows = [json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [_evaluate(row, references[(row["category"], row["defect_type"], Path(row["image_path"]).name)]) for row in rows]
    records.sort(key=lambda row: (row["category"], row["defect_type"], row["image"]))
    _write_csv(records)
    _write_sheet(rows, references)
    _write_report(records)
    print(REPORT_PATH)
    print(SHEET_PATH)


def _evaluate(row: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    truth = load_binary(reference["official_reference_mask"])
    prediction = load_binary(row["eval_mask_path"], truth.shape[::-1])
    region = tuple(int(value) for value in row["region_xyxy"])
    region_mask = np.zeros_like(truth)
    region_mask[region[1] : region[3], region[0] : region[2]] = True
    selected_metrics = metrics(prediction, truth)
    best_mode, best_dice = best_candidate(row, truth)
    localization = row["settings"].get("qwen_localization", {})
    qc = row["settings"].get("qc", {})
    arbitration = row["settings"].get("selection_arbitration", {})
    oracle_regret_available = bool(row["settings"].get("oracle_regret_available", True))
    return {
        "sample_id": f"{row['category']}/{row['defect_type']}/{Path(row['image_path']).name}",
        "category": row["category"],
        "defect_type": row["defect_type"],
        "image": Path(row["image_path"]).name,
        "localization_status": localization.get("status", "unknown"),
        "qwen_region_recall": ratio((region_mask & truth).sum(), truth.sum()),
        "qwen_region_iou": ratio((region_mask & truth).sum(), (region_mask | truth).sum()),
        "selected_refinement": row["settings"].get("selected_refinement", ""),
        "selection_arbitration_applied": bool(arbitration.get("applied", False)),
        "selection_arbitration_reason": arbitration.get("reason", ""),
        "candidate_cache_status": row["settings"].get("candidate_cache_status", "unknown"),
        "oracle_regret_available": oracle_regret_available,
        "qc_status": qc.get("status", ""),
        "qc_reasons": ",".join(qc.get("reasons", [])),
        **selected_metrics,
        "best_candidate_mode_oracle": best_mode,
        "best_candidate_dice_oracle": round(best_dice, 4),
        "selection_dice_regret": round(max(0.0, best_dice - selected_metrics["dice"]), 4),
        "prediction_path": row["eval_mask_path"],
        "official_mask_path": reference["official_reference_mask"],
    }


def load_binary(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    dice = 2 * tp / max(1, 2 * tp + fp + fn)
    iou = tp / max(1, tp + fp + fn)
    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "predicted_pixels": int(prediction.sum()),
        "truth_pixels": int(truth.sum()),
    }


def best_candidate(row: dict[str, Any], truth: np.ndarray) -> tuple[str, float]:
    candidates = {
        str(mode): path
        for mode, path in row["settings"].get("candidate_refined_paths", {}).items()
        if _candidate_mode_applicable(
            str(mode),
            category=str(row["category"]),
            defect_type=str(row["defect_type"]),
        )
    }
    candidates["selected_eval_tight"] = row["eval_mask_path"]
    best_mode = ""
    best_dice = -1.0
    for mode, path in candidates.items():
        candidate_path = Path(str(path))
        if not candidate_path.exists():
            continue
        candidate = load_binary(str(candidate_path), truth.shape[::-1])
        score = float(metrics(candidate, truth)["dice"])
        if score > best_dice:
            best_mode, best_dice = str(mode), score
    return best_mode, max(0.0, best_dice)


def ratio(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return round(float(numerator) / max(1.0, float(denominator)), 4)


def _write_csv(records: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_sheet(rows: list[dict[str, Any]], references: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (row["category"], row["defect_type"], Path(row["image_path"]).name))
    columns = ("input", "official mask", "Qwen region", "generation core", "eval mask", "errors", "uncertainty")
    thumb = 180
    header = 40
    label = 42
    sheet = Image.new("RGB", (len(columns) * thumb, header + len(rows) * (thumb + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, title in enumerate(columns):
        draw.rectangle((index * thumb, 0, (index + 1) * thumb, header), fill=(230, 235, 242))
        draw.text((index * thumb + 7, 13), title, fill=(25, 35, 50), font=font)

    for row_index, row in enumerate(rows):
        y = header + row_index * (thumb + label)
        key = (row["category"], row["defect_type"], Path(row["image_path"]).name)
        reference = references[key]
        image = Image.open(row["image_path"]).convert("RGB")
        truth = load_binary(reference["official_reference_mask"], image.size)
        prediction = load_binary(row["eval_mask_path"], image.size)
        variant_paths = row.get("mask_variant_paths", {}) if isinstance(row.get("mask_variant_paths"), dict) else {}
        generation_core = load_binary(variant_paths.get("generation_core", row["refined_mask_path"]), image.size)
        panels = [
            image,
            overlay(image, truth, (40, 190, 90), 0.55),
            region_overlay(image, tuple(row["region_xyxy"])),
            overlay(image, generation_core, (255, 145, 40), 0.52),
            overlay(image, prediction, (235, 55, 55), 0.55),
            error_overlay(image, prediction, truth),
            overlay(image, load_binary(row["uncertainty_mask_path"], image.size), (60, 110, 235), 0.50),
        ]
        for col, panel in enumerate(panels):
            sheet.paste(panel.resize((thumb, thumb), Image.Resampling.BILINEAR), (col * thumb, y))
        score = metrics(prediction, truth)
        loc = row["settings"].get("qwen_localization", {}).get("status", "?")
        text = f"{row['category']}/{row['defect_type']}/{Path(row['image_path']).name} | Qwen={loc} | Dice={score['dice']:.3f}"
        draw.text((6, y + thumb + 11), text, fill=(25, 35, 50), font=font)
    sheet.save(SHEET_PATH)


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(Image.fromarray(np.uint8(mask) * round(255 * alpha), mode="L"))
    return Image.alpha_composite(base, layer).convert("RGB")


def region_overlay(image: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
    result = image.copy()
    ImageDraw.Draw(result).rectangle(region, outline=(255, 190, 0), width=4)
    return result


def error_overlay(image: Image.Image, prediction: np.ndarray, truth: np.ndarray) -> Image.Image:
    result = image.convert("RGBA")
    colors = np.zeros((truth.shape[0], truth.shape[1], 4), dtype=np.uint8)
    colors[prediction & truth] = (40, 200, 90, 150)
    colors[prediction & ~truth] = (235, 55, 55, 150)
    colors[~prediction & truth] = (50, 110, 240, 170)
    return Image.alpha_composite(result, Image.fromarray(colors, mode="RGBA")).convert("RGB")


def _write_report(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["category"], record["defect_type"])].append(record)
    localization = Counter(record["localization_status"] for record in records)
    qc = Counter(record["qc_status"] for record in records)
    lines = [
        "# MVTec Bottle And Zipper Auto-Mask Review",
        "",
        "## Experimental Protocol",
        "",
        "- Source dataset: MVTec AD, CC BY-NC-SA 4.0.",
        "- Categories: `bottle` and `zipper`.",
        "- Three deterministic samples per selected defect type.",
        "- Images and official masks were resized to `512 x 512` in an isolated pilot copy.",
        "- Official masks were stored outside the MVTec-style `ground_truth` path.",
        "- Qwen and the automatic candidate ensemble could not read official masks.",
        "- Official masks were used only after generation for this evaluation.",
        "",
        "## Qwen Prompt Adaptation",
        "",
        "Bottle prompts explicitly distinguish rim chips and contamination from normal reflections, caps, transparency, and background.",
        "",
        "Zipper prompts restrict attention to the central tooth chain or fabric border and explicitly reject intact repeating teeth and normal woven texture.",
        "",
        "## Aggregate Results",
        "",
            f"- Qwen localization: `{localization.get('valid', 0)}` valid, `{localization.get('fallback', 0)}` fallback.",
            f"- Internal mask QC: `{qc.get('pass', 0)}` pass, `{qc.get('warning', 0)}` warning.",
            f"- Oracle/regret available: `{sum(bool(row.get('oracle_regret_available', True)) for row in records)}` of `{len(records)}` rows.",
            "",
        "| Category / defect | N | Qwen recall | Dice | IoU | Precision | Recall | Oracle candidate Dice | Selection regret |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, rows in sorted(grouped.items()):
        lines.append(
            f"| `{key[0]}/{key[1]}` | {len(rows)} | {_mean(rows, 'qwen_region_recall'):.4f} | "
            f"{_mean(rows, 'dice'):.4f} | {_mean(rows, 'iou'):.4f} | {_mean(rows, 'precision'):.4f} | "
            f"{_mean(rows, 'recall'):.4f} | {_mean(rows, 'best_candidate_dice_oracle'):.4f} | "
            f"{_mean(rows, 'selection_dice_regret'):.4f} |"
        )
    bottle = [row for row in records if row["category"] == "bottle"]
    zipper = [row for row in records if row["category"] == "zipper"]
    lines.extend(
        [
            f"| **Bottle overall** | {len(bottle)} | {_mean(bottle, 'qwen_region_recall'):.4f} | {_mean(bottle, 'dice'):.4f} | "
            f"{_mean(bottle, 'iou'):.4f} | {_mean(bottle, 'precision'):.4f} | {_mean(bottle, 'recall'):.4f} | "
            f"{_mean(bottle, 'best_candidate_dice_oracle'):.4f} | {_mean(bottle, 'selection_dice_regret'):.4f} |",
            f"| **Zipper overall** | {len(zipper)} | {_mean(zipper, 'qwen_region_recall'):.4f} | {_mean(zipper, 'dice'):.4f} | "
            f"{_mean(zipper, 'iou'):.4f} | {_mean(zipper, 'precision'):.4f} | {_mean(zipper, 'recall'):.4f} | "
            f"{_mean(zipper, 'best_candidate_dice_oracle'):.4f} | {_mean(zipper, 'selection_dice_regret'):.4f} |",
            "",
            "The oracle candidate score is diagnostic only. It asks whether any generated candidate happened to agree with official ground truth; it is not available to the automatic selector.",
            "If `Oracle/regret available` is below the row count, the candidate cache was incomplete and oracle/regret columns should be treated as compatibility diagnostics only.",
            "",
            "## Interpretation",
            "",
            "- The visual sheet separates `generation_core` from the benchmark `eval mask`.",
            "- `generation_core` is the visible defect evidence used for generation/reference conditioning.",
            "- `eval mask` may use a broader MVTec-style pseudo-mask when the official annotation includes surrounding damaged area.",
            "- High Qwen recall with low Dice means localization succeeded but pixel refinement failed.",
            "- Low Qwen recall means the language-guided search region excluded part of the real defect.",
            "- Large selection regret means candidate generation is adequate but automatic candidate ranking needs improvement.",
            "- Zipper is expected to be harder because defects compete with dense repeating tooth and fabric texture.",
            "",
        ]
    )
    chain_validation_path = REPORT_DIR / "repeated_chain_validation.csv"
    if chain_validation_path.exists():
        with chain_validation_path.open(encoding="utf-8", newline="") as handle:
            chain_rows = list(csv.DictReader(handle))
        if chain_rows:
            current_chain = _mean(chain_rows, "current_dice")
            ungated_chain = _mean(chain_rows, "repeated_chain_dice")
            gated_chain = _mean(chain_rows, "gated_dice")
            gate_count = sum(str(row.get("gate_enabled", "")).lower() == "true" for row in chain_rows)
            lines.extend(
                [
                    "## Repeated-Chain Specialist Validation",
                    "",
                    "The repeated-chain specialist is integrated into the aggregate production masks above.",
                    "Its gate uses localization status and PCA-aligned baseline-mask geometry without reading official masks.",
                    "",
                    f"- Production gate active for `{gate_count}` of `{len(chain_rows)}` tooth samples.",
                    f"- Current production tooth-mask mean Dice: `{current_chain:.4f}`.",
                    f"- Independently recomputed specialist mean Dice: `{ungated_chain:.4f}`.",
                    f"- Gate replay mean Dice: `{gated_chain:.4f}`.",
                    "",
                ]
            )
    polar_count = sum(row["selected_refinement"] == "polar_rim_residual" for row in records)
    lines.extend(
        [
            "## Polar Rim Specialist",
            "",
            f"- `polar_rim_residual` selected for `{polar_count}` sample(s).",
            "- It is restricted to small bottle-rim chips and competes with SAM and consensus candidates.",
            "- Normal-reference residual evidence is aggregated by rim angle and mapped back to Cartesian pixels.",
            "",
        ]
    )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Visual review: `{SHEET_PATH}`",
            f"- Per-sample metrics: `{CSV_PATH}`",
            f"- Repeated-chain validation: `{REPORT_DIR / 'repeated_chain_validation.md'}`",
            f"- Repeated-chain visual comparison: `{REPORT_DIR / 'repeated_chain_validation.png'}`",
            f"- PCA/polar upgrade review: `{REPORT_DIR / 'pca_polar_upgrade_review.md'}`",
            f"- Recall-safe selector sprint review: `{REPORT_DIR / 'recall_safe_selector_sprint_review.md'}`",
            f"- Auto-mask summary: `{ROOT / 'reports/mvtec_bottle_zipper_auto_mask/auto_masks/qwen/summary.md'}`",
            f"- Mask-policy ablation: `{ROOT / 'reports/mvtec_bottle_zipper_auto_mask/phase10_mask_quality/qwen/mask_quality_ablation_report.md'}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


if __name__ == "__main__":
    main()
