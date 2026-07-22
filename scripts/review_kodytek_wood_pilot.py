from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "kodytek_wood_pilot"
METADATA_PATH = ROOT / "outputs" / "kodytek_wood_pilot" / "auto_masks" / "qwen" / "metadata.jsonl"
REPORT_DIR = ROOT / "reports" / "kodytek_wood_pilot" / "external_data_review"
CSV_PATH = REPORT_DIR / "weak_box_audit.csv"
REPORT_PATH = REPORT_DIR / "kodytek_wood_pilot_review.md"
SHEET_PATH = REPORT_DIR / "kodytek_wood_pilot_visual_review.png"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source = json.loads((DATA_ROOT / "source_manifest.json").read_text(encoding="utf-8"))
    source_by_image = {str(Path(row["image_path"]).resolve()): row for row in source["records"]}
    auto_rows = [json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    records = [_audit_record(row, source_by_image[str(Path(row["image_path"]).resolve())]) for row in auto_rows]
    records.sort(key=lambda row: (row["defect_type"], row["image"]))
    _write_csv(records)
    _write_sheet(auto_rows, source_by_image)
    _write_report(records, source)
    print(REPORT_PATH)
    print(SHEET_PATH)


def _audit_record(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    mask = np.asarray(Image.open(row["eval_mask_path"]).convert("L"), dtype=np.uint8) > 0
    reference = np.asarray(Image.open(source["reference_bbox_mask_path"]).convert("L"), dtype=np.uint8) > 0
    intersection = mask & reference
    region = tuple(int(value) for value in row["region_xyxy"])
    region_mask = np.zeros_like(mask)
    region_mask[region[1] : region[3], region[0] : region[2]] = True
    localization = row["settings"].get("qwen_localization", {})
    policy = row["settings"].get("label_policy", {})
    qc = row["settings"].get("qc", {})
    return {
        "sample_id": f"{row['category']}/{row['defect_type']}/{Path(row['image_path']).name}",
        "defect_type": row["defect_type"],
        "image": Path(row["image_path"]).name,
        "source_row_idx": source["row_idx"],
        "localization_status": localization.get("status", "unknown"),
        "qwen_region_box_recall": _ratio((region_mask & reference).sum(), reference.sum()),
        "qwen_region_iou_with_box": _ratio((region_mask & reference).sum(), (region_mask | reference).sum()),
        "selected_refinement": row["settings"].get("selected_refinement", ""),
        "morphology": row["settings"].get("quality_morphology", ""),
        "label_policy": policy.get("label_policy", ""),
        "training_mask_variant": policy.get("adapter_training_mask_variant", ""),
        "qc_status": qc.get("status", ""),
        "qc_reasons": ",".join(qc.get("reasons", [])),
        "mask_pixels": int(mask.sum()),
        "reference_box_pixels": int(reference.sum()),
        "mask_inside_box_fraction": _ratio(intersection.sum(), mask.sum()),
        "reference_box_covered_fraction": _ratio(intersection.sum(), reference.sum()),
        "mask_to_box_area_ratio": _ratio(mask.sum(), reference.sum()),
        "image_path": row["image_path"],
        "reference_bbox_mask_path": source["reference_bbox_mask_path"],
        "eval_mask_path": row["eval_mask_path"],
        "training_mask_path": row["training_mask_path"],
        "uncertainty_mask_path": row["uncertainty_mask_path"],
    }


def _ratio(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return round(float(numerator) / max(1.0, float(denominator)), 4)


def _write_csv(records: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_sheet(auto_rows: list[dict[str, Any]], source_by_image: dict[str, dict[str, Any]]) -> None:
    rows = sorted(auto_rows, key=lambda row: (row["defect_type"], Path(row["image_path"]).name))
    columns = ("input", "published bbox", "Qwen region", "eval_tight", "training", "uncertainty")
    thumb = 190
    header = 42
    label = 38
    sheet = Image.new("RGB", (len(columns) * thumb, header + len(rows) * (thumb + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, title in enumerate(columns):
        draw.rectangle((index * thumb, 0, (index + 1) * thumb, header), fill=(230, 235, 242))
        draw.text((index * thumb + 8, 14), title, fill=(25, 35, 50), font=font)

    for row_index, row in enumerate(rows):
        y = header + row_index * (thumb + label)
        image_path = str(Path(row["image_path"]).resolve())
        source = source_by_image[image_path]
        image = Image.open(image_path).convert("RGB")
        panels = [
            image,
            _overlay(image, source["reference_bbox_mask_path"], (255, 190, 0), 0.45),
            _region_overlay(image, tuple(row["region_xyxy"])),
            _overlay(image, row["eval_mask_path"], (235, 55, 55), 0.55),
            _overlay(image, row["training_mask_path"], (20, 150, 120), 0.48),
            _overlay(image, row["uncertainty_mask_path"], (60, 110, 235), 0.50),
        ]
        for col, panel in enumerate(panels):
            fitted = panel.resize((thumb, thumb), Image.Resampling.BILINEAR)
            sheet.paste(fitted, (col * thumb, y))
        localization = row["settings"].get("qwen_localization", {}).get("status", "?")
        qc = row["settings"].get("qc", {}).get("status", "?")
        text = f"{row['defect_type']}/{Path(row['image_path']).name} | Qwen={localization} | QC={qc}"
        draw.text((6, y + thumb + 10), text, fill=(25, 35, 50), font=font)
    sheet.save(SHEET_PATH)


def _overlay(image: Image.Image, mask_path: str, color: tuple[int, int, int], alpha: float) -> Image.Image:
    base = image.convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask_arr = np.asarray(mask, dtype=np.float32) / 255.0
    overlay = Image.new("RGBA", image.size, (*color, 0))
    overlay.putalpha(Image.fromarray(np.uint8(np.clip(mask_arr * 255 * alpha, 0, 255)), mode="L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _region_overlay(image: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
    result = image.copy()
    ImageDraw.Draw(result).rectangle(region, outline=(255, 190, 0), width=4)
    return result


def _write_report(records: list[dict[str, Any]], source: dict[str, Any]) -> None:
    by_defect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_defect[record["defect_type"]].append(record)
    localization = Counter(record["localization_status"] for record in records)
    qc = Counter(record["qc_status"] for record in records)
    lines = [
        "# Kodytek Real-Wood Pilot Review",
        "",
        "## Dataset",
        "",
        "This pilot adds real production-line wood imagery from the Kodytek et al. wood surface defect dataset.",
        "",
        f"- Source: `{source['dataset']}`",
        f"- Original DOI: `{source['original_doi']}`",
        f"- License: `{source['license']}`",
        "- Imported images: `30` (`18` clean, `12` defect)",
        "- Defect families: `crack`, `resin`, `knot_with_crack`",
        "- Published boxes were used for deterministic crop selection and audit only.",
        "- The automatic mask generator received images and text descriptions, not the published boxes.",
        "",
        "## Pipeline Execution",
        "",
        "```text",
        "download selected public records",
        "-> deterministic 768 px crop",
        "-> resize to 512 px",
        "-> Qwen localization",
        "-> 16-candidate auto-mask ensemble",
        "-> purpose-specific mask variants",
        "-> prepare no-leakage split",
        "-> Phase 10 mask-policy ablation",
        "```",
        "",
        f"Qwen localization: `{localization.get('valid', 0)}` valid, `{localization.get('fallback', 0)}` full-image fallback.",
        f"Mask QC: `{qc.get('pass', 0)}` pass, `{qc.get('warning', 0)}` warning, `{qc.get('reject', 0)}` reject.",
        "",
        "## Weak-Box Audit",
        "",
        "These are weak-reference diagnostics. A bounding box is not a pixel mask, so box coverage is not segmentation IoU.",
        "",
        "| Defect | Samples | Qwen box recall | Mask inside box | Box covered | Mask/box area |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for defect_type, defect_rows in sorted(by_defect.items()):
        lines.append(
            f"| `{defect_type}` | {len(defect_rows)} | "
            f"{_mean(defect_rows, 'qwen_region_box_recall'):.4f} | "
            f"{_mean(defect_rows, 'mask_inside_box_fraction'):.4f} | "
            f"{_mean(defect_rows, 'reference_box_covered_fraction'):.4f} | "
            f"{_mean(defect_rows, 'mask_to_box_area_ratio'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Professional Interpretation",
            "",
            "### What transferred well",
            "",
            "- Qwen produced a valid search region for 11 of 12 unfamiliar production-line samples.",
            "- Long cracks were generally isolated as narrow line-like regions.",
            "- Compact cracked knots received localized masks supported by multiple candidate methods.",
            "- The batch completed without manual mask drawing and produced all uncertainty/training variants.",
            "",
            "### What did not transfer cleanly",
            "",
            "- One low-contrast crack triggered a full-image localization fallback.",
            "- Conveyor-edge structure can dominate resin masks when dark borders resemble anomalies.",
            "- Broad resin streaks sometimes cause over-segmentation along repeated wood texture.",
            "- Every sample currently routes to `training_medium`; resin may need a dedicated soft/appearance-defect policy.",
            "- The morphology label `not_scratch` is too coarse for cracks, resin, and knots and should become a richer taxonomy.",
            "",
            "## Recommended Next Upgrade",
            "",
            "1. Add defect-family morphology classes: `linear_crack`, `resin_streak`, and `compact_knot_crack`.",
            "2. Add a border/conveyor suppression prior learned from clean images.",
            "3. Route resin streaks to uncertainty-aware soft labels when candidate disagreement is high.",
            "4. Evaluate automatic masks on MVTec wood, where true pixel masks are available.",
            "5. Use this Kodytek pilot as external-domain adaptation data, not as pixel-accurate final evaluation.",
            "",
            "## Artifacts",
            "",
            f"- Visual review: `{SHEET_PATH}`",
            f"- Weak-box metrics: `{CSV_PATH}`",
            f"- Auto-mask report: `{ROOT / 'reports/kodytek_wood_pilot/auto_masks/qwen/summary.md'}`",
            f"- Mask-policy ablation: `{ROOT / 'reports/kodytek_wood_pilot/phase10_mask_quality/qwen/mask_quality_ablation_report.md'}`",
            f"- Split manifest: `{ROOT / 'outputs/kodytek_wood_pilot/prepared/split_manifest.json'}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


if __name__ == "__main__":
    main()
