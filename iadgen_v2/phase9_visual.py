from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from iadgen_v2.config import AppConfig


def write_phase9_visual_report(config: AppConfig, provider: str | None = None) -> Path:
    phase5 = dict(config.data.get("phase5", {}))
    phase4 = dict(config.data.get("phase4", {}))
    auto = dict(config.data.get("auto_masks", {}))
    provider = provider or str(phase5.get("provider", phase4.get("provider", auto.get("provider", "qwen"))))
    report_dir = config.report_dir / "phase9_visual" / provider
    report_dir.mkdir(parents=True, exist_ok=True)

    auto_rows = _load_jsonl(config.output_dir / "auto_masks" / provider / "metadata.jsonl")
    phase4_rows = _load_phase4_rows(config, provider)
    prediction_sheets = _load_prediction_sheets(config, provider)

    contact_sheet = report_dir / "phase9_visual_evidence_contact_sheet.png"
    _write_contact_sheet(contact_sheet, auto_rows, phase4_rows, prediction_sheets)
    summary = report_dir / "summary.md"
    _write_summary(summary, contact_sheet, auto_rows, phase4_rows, prediction_sheets, provider)
    return contact_sheet


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _load_phase4_rows(config: AppConfig, provider: str) -> list[dict[str, Any]]:
    root = config.output_dir / "phase4" / provider
    if not root.exists():
        return []
    paths = sorted(path for path in root.glob("*/metadata.jsonl") if path.parent.name != provider)
    if not paths and (root / "metadata.jsonl").exists():
        paths = [root / "metadata.jsonl"]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in paths:
        for row in _load_jsonl(path):
            key = (str(row.get("variant", path.parent.name)), str(row.get("output_path", "")), int(row.get("generation_seed", 0)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: float(row.get("generation_quality_score", 0.0)), reverse=True)


def _load_prediction_sheets(config: AppConfig, provider: str) -> list[Path]:
    csv_path = config.report_dir / "phase5" / provider / "segmentation_results.csv"
    sheets: list[Path] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get("prediction_contact_sheet")
                if value and Path(value).exists():
                    sheets.append(Path(value))
    if not sheets:
        sheets = sorted((config.report_dir / "phase5" / provider / "prediction_examples").glob("**/prediction_contact_sheet.png"))
    return sheets


def _write_contact_sheet(
    path: Path,
    auto_rows: list[dict[str, Any]],
    phase4_rows: list[dict[str, Any]],
    prediction_sheets: list[Path],
) -> None:
    selected_phase4 = phase4_rows[:8]
    selected_auto = auto_rows[: max(0, 4 - len(selected_phase4))]
    rows: list[tuple[str, dict[str, Any]]] = [("phase4", row) for row in selected_phase4] + [("auto", row) for row in selected_auto]
    if not rows:
        rows = [("empty", {})]
    columns = ["input", "selected_mask", "training_used", "generated", "leakage", "prediction"]
    thumb, label_h = 160, 28
    sheet = Image.new("RGB", (thumb * len(columns), (thumb + label_h) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (kind, row) in enumerate(rows):
        panels = _panels_for_row(kind, row, prediction_sheets[row_index % len(prediction_sheets)] if prediction_sheets else None)
        for col_index, column in enumerate(columns):
            x = col_index * thumb
            y = row_index * (thumb + label_h)
            panel = panels.get(column) or _placeholder(thumb, thumb, column)
            sheet.paste(panel.resize((thumb, thumb), Image.Resampling.BILINEAR), (x, y + label_h))
            label = _row_label(kind, row, column)
            draw.text((x + 4, y + 4), label[:42], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _panels_for_row(kind: str, row: dict[str, Any], prediction_sheet: Path | None) -> dict[str, Image.Image]:
    panels: dict[str, Image.Image] = {}
    if kind == "phase4":
        input_image = _open_image(row.get("background_path"))
        output = _open_image(row.get("output_path"))
        refined = _open_image(row.get("refined_mask_path"), mode="L")
        inpaint = _open_image(row.get("inpaint_mask_path"), mode="L")
        panels["input"] = input_image
        panels["selected_mask"] = _overlay_mask(input_image, refined, (255, 40, 40))
        panels["training_used"] = _overlay_mask(input_image, inpaint, (40, 180, 255))
        panels["generated"] = output
        panels["leakage"] = _diff_heatmap(input_image, output, refined)
    elif kind == "auto":
        image = _open_image(row.get("image_path"))
        panels["input"] = image
        panels["selected_mask"] = _open_image(row.get("overlay_path")) or _overlay_mask(image, _open_image(row.get("refined_mask_path"), "L"), (255, 40, 40))
        panels["training_used"] = _training_used_panel(row, image)
    if prediction_sheet and prediction_sheet.exists():
        panels["prediction"] = Image.open(prediction_sheet).convert("RGB")
    return panels


def _training_used_panel(row: dict[str, Any], image: Image.Image) -> Image.Image:
    training_path = row.get("training_mask_path") or row.get("settings", {}).get("training_mask_path")
    mask = _open_image(training_path, mode="L")
    return _overlay_mask(image, mask, (40, 180, 255))


def _row_label(kind: str, row: dict[str, Any], column: str) -> str:
    if kind == "phase4":
        return f"{row.get('variant', 'phase4')} {row.get('quality_profile', '')} {column}"
    if kind == "auto":
        return f"{Path(str(row.get('image_path', 'auto'))).name} {column}"
    return column


def _open_image(value: object, mode: str = "RGB") -> Image.Image:
    path = Path(str(value)) if value else Path("")
    if path.exists():
        return Image.open(path).convert(mode)
    return _placeholder(160, 160, "missing")


def _overlay_mask(image: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    resized_mask = mask.convert("L").resize(image.size, Image.Resampling.BILINEAR)
    alpha = (np.asarray(resized_mask, dtype=np.float32) / 255.0)[..., None]
    overlay = base * (1.0 - 0.55 * alpha) + np.asarray(color, dtype=np.float32)[None, None, :] * 0.55 * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).convert("RGB")


def _diff_heatmap(background: Image.Image, output: Image.Image, refined_mask: Image.Image) -> Image.Image:
    source = np.asarray(background.convert("RGB"), dtype=np.float32)
    generated = np.asarray(output.resize(background.size, Image.Resampling.BILINEAR).convert("RGB"), dtype=np.float32)
    diff = np.abs(generated - source).mean(axis=2)
    refined = np.asarray(refined_mask.convert("L").resize(background.size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
    diff[refined] *= 0.35
    diff = diff / max(1.0, float(np.percentile(diff, 99)))
    heat = np.zeros((*diff.shape, 3), dtype=np.float32)
    heat[..., 0] = np.clip(diff * 255.0, 0, 255)
    heat[..., 1] = np.clip((1.0 - np.abs(diff - 0.5) * 2.0) * 180.0, 0, 180)
    heat[..., 2] = np.clip((1.0 - diff) * 120.0, 0, 120)
    return Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8)).convert("RGB")


def _placeholder(width: int, height: int, text: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(160, 160, 160))
    draw.text((8, 8), text, fill=(40, 40, 40))
    return image


def _write_summary(
    path: Path,
    contact_sheet: Path,
    auto_rows: list[dict[str, Any]],
    phase4_rows: list[dict[str, Any]],
    prediction_sheets: list[Path],
    provider: str,
) -> None:
    lines = [
        f"# Phase 9 Visual Evidence Report: {provider}",
        "",
        f"Visual contact sheet: `{contact_sheet}`",
        f"Auto-mask rows available: `{len(auto_rows)}`",
        f"Phase 4 generation rows available: `{len(phase4_rows)}`",
        f"Phase 5 prediction sheets available: `{len(prediction_sheets)}`",
        "",
        "Columns: input, selected mask, training-used mask, generated image, leakage map, segmentation prediction.",
        "",
        "If a cell says `missing`, that artifact has not been generated yet. This is intentional: the visual report exposes incomplete evidence instead of hiding it.",
    ]
    if phase4_rows:
        best = phase4_rows[0]
        lines.extend(
            [
                "",
                "## Best Generation Row Shown",
                "",
                f"- Variant: `{best.get('variant')}`",
                f"- Quality profile: `{best.get('quality_profile')}`",
                f"- Quality score: `{float(best.get('generation_quality_score', 0.0)):.6f}`",
                f"- Output: `{best.get('output_path')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
