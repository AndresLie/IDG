from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iadgen_v2.auto_masks import (
    _attach_mask_variants,
    _auto_config,
    _make_mask_candidate,
    _repeated_chain_correction_gate,
    _resolve_auto_paths,
)
from iadgen_v2.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "mvtec_bottle_zipper_auto_mask.yaml"
METADATA_PATH = ROOT / "outputs" / "mvtec_bottle_zipper_auto_mask" / "auto_masks" / "qwen" / "metadata.jsonl"
REPORT_DIR = ROOT / "reports" / "mvtec_bottle_zipper_auto_mask" / "official_mask_evaluation"
CSV_PATH = REPORT_DIR / "repeated_chain_validation.csv"
REPORT_PATH = REPORT_DIR / "repeated_chain_validation.md"
SHEET_PATH = REPORT_DIR / "repeated_chain_validation.png"


def main() -> None:
    config = load_config(CONFIG_PATH)
    auto = _auto_config(config)
    _resolve_auto_paths(config, auto)
    rows = [json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    references = _reference_index()
    records: list[dict[str, Any]] = []
    panels: list[dict[str, Any]] = []

    for row in rows:
        if row["category"] != "zipper" or row["defect_type"] not in {"broken_teeth", "split_teeth"}:
            continue
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        normal_paths = sorted((config.dataset_root / row["category"] / "train" / "good").glob("*.png"))
        artifact = _make_mask_candidate(
            mode="repeated_chain_refiner",
            image=image,
            category=row["category"],
            defect_type=row["defect_type"],
            region=tuple(int(value) for value in row["region_xyxy"]),
            output_dir=config.output_dir / "auto_masks" / "qwen" / "masks" / "zipper" / "repeated_chain_validation",
            artifact_stem=f"zipper_{row['defect_type']}_{image_path.stem}_repeated_chain_refiner",
            seed=int(row.get("seed", 0)),
            auto=auto,
            description=str(row.get("description", "")),
            normal_paths=normal_paths,
        )
        baseline_paths = dict(row.get("settings", {}).get("candidate_refined_paths", {}))
        baseline_heatmaps = dict(row.get("settings", {}).get("candidate_heatmap_paths", {}))
        heatmap_path = str(artifact.get("parameters", {}).get("heatmap_path", ""))
        artifact.update(
            {
                "selected_refinement": "repeated_chain_refiner",
                "candidate_refined_paths": {
                    **baseline_paths,
                    "repeated_chain_refiner": artifact["refined_mask_path"],
                },
                "candidate_heatmap_paths": {
                    **baseline_heatmaps,
                    **({"repeated_chain_refiner": heatmap_path} if heatmap_path else {}),
                },
                "scratch_morphology_class": "not_scratch",
                "structure_profile": "repeated_chain",
                "description": str(row.get("description", "")),
            }
        )
        production_artifact = _attach_mask_variants(
            image=image,
            category=row["category"],
            defect_type=row["defect_type"],
            region=tuple(int(value) for value in row["region_xyxy"]),
            output_dir=(
                config.output_dir
                / "auto_masks"
                / "qwen"
                / "mask_variants"
                / "zipper"
                / "repeated_chain_validation"
            ),
            artifact_stem=f"zipper_{row['defect_type']}_{image_path.stem}_repeated_chain_refiner",
            artifacts=artifact,
            auto=auto,
        )
        gate = _repeated_chain_correction_gate(
            _baseline_candidates(row),
            region=tuple(int(value) for value in row["region_xyxy"]),
            localization_status=str(row.get("settings", {}).get("qwen_localization", {}).get("status", "unknown")),
        )
        truth_path = references[(row["category"], row["defect_type"], image_path.name)]
        truth = _binary(truth_path, image.size)
        current = _binary(row["eval_mask_path"], image.size)
        repeated_raw = _binary(artifact["refined_mask_path"], image.size)
        repeated = _binary(production_artifact["eval_mask_path"], image.size)
        gated = repeated if gate["repeated_chain_gate_enabled"] else current
        current_metrics = _metrics(current, truth)
        raw_metrics = _metrics(repeated_raw, truth)
        repeated_metrics = _metrics(repeated, truth)
        gated_metrics = _metrics(gated, truth)
        params = artifact.get("parameters", {})
        records.append(
            {
                "sample": f"{row['defect_type']}/{image_path.name}",
                "gate_enabled": bool(gate["repeated_chain_gate_enabled"]),
                "gate_reason": gate["repeated_chain_gate_reason"],
                "current_dice": current_metrics["dice"],
                "raw_repeated_chain_dice": raw_metrics["dice"],
                "repeated_chain_dice": repeated_metrics["dice"],
                "gated_dice": gated_metrics["dice"],
                "gated_delta": round(float(gated_metrics["dice"]) - float(current_metrics["dice"]), 4),
                "explanation_score": round(float(params.get("repeated_chain_explanation_score", 0.0)), 4),
                "structure_score": round(float(params.get("repeated_chain_structure_score", 0.0)), 4),
                "vertical_span": round(float(params.get("repeated_chain_vertical_span_fraction", 0.0)), 4),
                "candidate_path": production_artifact["eval_mask_path"],
            }
        )
        panels.append(
            {
                "label": f"{row['defect_type']}/{image_path.name}",
                "image": image,
                "truth": truth,
                "current": current,
                "repeated_raw": repeated_raw,
                "repeated": repeated,
                "gated": gated,
                "gate_enabled": bool(gate["repeated_chain_gate_enabled"]),
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(records)
    _write_report(records)
    _write_sheet(panels)
    print(REPORT_PATH)
    print(SHEET_PATH)


def _baseline_candidates(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    settings = row.get("settings", {})
    paths = settings.get("candidate_refined_paths", {})
    scores = settings.get("candidate_scores", {})
    candidates: dict[str, dict[str, Any]] = {}
    for mode in ("patchcore_guided", "normal_residual_fusion", "nearest_normal_residual"):
        path = paths.get(mode)
        if path and Path(str(path)).exists():
            candidates[mode] = {
                "refined_mask_path": str(path),
                "score": float(scores.get(mode, 0.9)),
                "qc": {"status": "pass", "mask_area": 1},
            }
    return candidates


def _reference_index() -> dict[tuple[str, str, str], str]:
    manifest = json.loads((ROOT / "data" / "mvtec_bottle_zipper_auto_mask" / "pilot_manifest.json").read_text())
    return {
        (record["category"], record["defect_type"], Path(record["target_image"]).name): str(
            ROOT / record["official_reference_mask"]
        )
        for record in manifest["records"]
        if record.get("role") == "defect"
    }


def _binary(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    intersection = int((prediction & truth).sum())
    dice = 2 * intersection / max(1, int(prediction.sum()) + int(truth.sum()))
    return {"dice": round(float(dice), 4)}


def _write_csv(records: list[dict[str, Any]]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_report(records: list[dict[str, Any]]) -> None:
    current = float(np.mean([float(record["current_dice"]) for record in records]))
    raw = float(np.mean([float(record["raw_repeated_chain_dice"]) for record in records]))
    repeated = float(np.mean([float(record["repeated_chain_dice"]) for record in records]))
    gated = float(np.mean([float(record["gated_dice"]) for record in records]))
    enabled = sum(bool(record["gate_enabled"]) for record in records)
    lines = [
        "# Repeated-Chain Refiner Validation",
        "",
        "Official masks are used only after candidate generation and label-free gate decisions.",
        "",
        f"- Samples: `{len(records)}`",
        f"- Gate enabled: `{enabled}`",
        f"- Current tooth-mask mean Dice: `{current:.4f}`",
        f"- Raw repeated-chain mean Dice: `{raw:.4f}`",
        f"- Production-processed repeated-chain mean Dice: `{repeated:.4f}`",
        f"- Gated repeated-chain mean Dice: `{gated:.4f}`",
        f"- Gated improvement: `{gated - current:+.4f}`",
        "",
        "| Sample | Gate | Reason | Current Dice | Raw Refiner | Production Refiner | Gated Dice | Delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| `{record['sample']}` | `{record['gate_enabled']}` | `{record['gate_reason']}` | "
            f"{float(record['current_dice']):.4f} | {float(record['raw_repeated_chain_dice']):.4f} | "
            f"{float(record['repeated_chain_dice']):.4f} | "
            f"{float(record['gated_dice']):.4f} | {float(record['gated_delta']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "The refiner is intentionally specialized. It is selected only when a valid localization contains a dense baseline mask spanning nearly the full repeated chain.",
            "",
            f"- Metrics: `{CSV_PATH}`",
            f"- Visual comparison: `{SHEET_PATH}`",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sheet(records: list[dict[str, Any]]) -> None:
    columns = ("input", "official", "current", "raw refiner", "production refiner", "gated")
    thumb = 180
    header = 38
    label_height = 34
    sheet = Image.new("RGB", (thumb * len(columns), header + len(records) * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, title in enumerate(columns):
        draw.rectangle((index * thumb, 0, (index + 1) * thumb, header), fill=(230, 235, 242))
        draw.text((index * thumb + 7, 12), title, fill=(25, 35, 50), font=font)
    for row_index, record in enumerate(records):
        y = header + row_index * (thumb + label_height)
        image = record["image"]
        panels = [
            image,
            _overlay(image, record["truth"], (40, 190, 90)),
            _overlay(image, record["current"], (235, 55, 55)),
            _overlay(image, record["repeated_raw"], (245, 190, 35)),
            _overlay(image, record["repeated"], (255, 150, 35)),
            _overlay(image, record["gated"], (70, 110, 235)),
        ]
        for column, panel in enumerate(panels):
            sheet.paste(panel.resize((thumb, thumb), Image.Resampling.BILINEAR), (column * thumb, y))
        draw.text(
            (6, y + thumb + 10),
            f"{record['label']} | gate={record['gate_enabled']}",
            fill=(25, 35, 50),
            font=font,
        )
    sheet.save(SHEET_PATH)


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(Image.fromarray(np.uint8(mask) * 145, mode="L"))
    return Image.alpha_composite(base, layer).convert("RGB")


if __name__ == "__main__":
    main()
