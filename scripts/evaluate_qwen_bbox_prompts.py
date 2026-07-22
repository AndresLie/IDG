from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iadgen_v2.auto_masks import (  # noqa: E402
    _auto_config,
    _description_for,
    _defect_images,
    _fallback_qwen,
    _load_descriptions,
    _normal_index,
    _prompt,
    _qwen_availability,
    _resolve_auto_paths,
    _select_qwen_region_from_sub_boxes,
    _validated_or_fallback_region,
    _validated_sub_boxes,
    bottle_rim_guided_qwen_region,
    bottle_surface_guided_region,
    normal_guided_qwen_verify_region,
    parse_qwen_bbox_payload,
    resolve_auto_localization_policy,
)
from iadgen_v2.config import load_config  # noqa: E402
from iadgen_v2.qwen_provider import QwenFeatureExtractor, write_availability  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen bbox localization without running mask refinement.")
    parser.add_argument("--config", required=True, help="Path to v2 config.")
    parser.add_argument("--provider", default="qwen", help="Provider label. Only qwen is supported.")
    parser.add_argument(
        "--baseline",
        default=str(ROOT / "reports/mvtec_bottle_zipper_auto_mask/official_mask_evaluation/auto_mask_metrics.csv"),
        help="Optional baseline metrics CSV with qwen_region_recall/qwen_region_iou.",
    )
    parser.add_argument(
        "--baseline-metadata",
        default=str(ROOT / "outputs/mvtec_bottle_zipper_auto_mask/auto_masks/qwen/metadata.jsonl"),
        help="Optional baseline auto-mask metadata JSONL with old Qwen region geometry.",
    )
    args = parser.parse_args()
    if args.provider != "qwen":
        raise ValueError("Only provider=qwen is supported")

    config = load_config(args.config)
    auto = _auto_config(config)
    _resolve_auto_paths(config, auto)
    report_dir = config.report_dir / "qwen_bbox_eval" / args.provider
    output_dir = config.output_dir / "qwen_bbox_eval" / args.provider
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    availability = _qwen_availability(config, auto)
    write_availability(output_dir / "qwen_availability.json", availability)
    if not availability.ready:
        raise RuntimeError(f"Qwen provider is not ready: {availability.failure_reason()}")

    references = _load_references(config.dataset_root)
    descriptions = _load_descriptions(config, auto)
    normal_index = _normal_index(config)
    extractor = QwenFeatureExtractor(
        model_id=str(auto.get("qwen_model", _fallback_qwen(config, "qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct"))),
        cache_dir=auto.get("qwen_cache_dir", _fallback_qwen(config, "qwen_cache_dir", None)),
        local_files_only=bool(auto.get("qwen_local_files_only", _fallback_qwen(config, "qwen_local_files_only", True))),
        device=str(auto.get("qwen_device", _fallback_qwen(config, "qwen_device", "auto"))),
        torch_dtype=str(auto.get("qwen_dtype", _fallback_qwen(config, "qwen_dtype", "auto"))),
        token_count=int(auto.get("token_count", _fallback_qwen(config, "token_count", 16))),
        min_free_gib=float(auto.get("qwen_min_free_gib", _fallback_qwen(config, "qwen_min_free_gib", 30.0))),
    )

    rows: list[dict[str, Any]] = []
    raw_path = output_dir / "qwen_bbox_raw.jsonl"
    raw_path.write_text("", encoding="utf-8")
    try:
        for category, defect_type, image_path in _defect_images(config):
            reference = references.get((category, defect_type, image_path.name))
            if reference is None:
                continue
            image = Image.open(image_path).convert("RGB")
            prompt = _prompt(
                category,
                defect_type,
                image.size,
                auto,
                _description_for(config, descriptions, category, defect_type, image_path),
            )
            result = extractor.analyze(image, prompt)
            parsed = parse_qwen_bbox_payload(str(result["text"]), image.size)
            region, localization = _validated_or_fallback_region(parsed["bbox_xyxy"], image.size, auto=auto)
            sub_boxes = _validated_sub_boxes(
                parsed.get("sub_boxes_xyxy", []),
                image.size,
                min_area_ratio=float(auto.get("min_sub_box_area_ratio", auto.get("min_box_area_ratio", 0.001) / 4.0)),
                max_area_ratio=float(auto.get("max_sub_box_area_ratio", auto.get("max_box_area_ratio", 0.5))),
            )
            region, sub_selection = _select_qwen_region_from_sub_boxes(region, sub_boxes, image.size, auto)
            localization.update(sub_selection)
            strategy = "prompt"
            normal_guided_record: dict[str, Any] | None = None
            grid_record: dict[str, Any] | None = None
            description = _description_for(config, descriptions, category, defect_type, image_path)
            localization_policy, policy_diagnostics = resolve_auto_localization_policy(
                auto=auto,
                category=category,
                defect_type=defect_type,
                description=description,
                image=image,
                qwen_region=region,
                localization=localization,
            )
            localization["policy"] = localization_policy
            localization["policy_diagnostics"] = policy_diagnostics
            if localization_policy == "normal_guided_qwen_verify":
                normal_guided_region, normal_guided_record = normal_guided_qwen_verify_region(
                    image=image,
                    normal_paths=normal_index.get(category, []),
                    category=category,
                    defect_type=defect_type,
                    description=description,
                    extractor=extractor,
                    auto=auto,
                    output_dir=output_dir / "localization" / category / defect_type,
                    stem=f"{category}_{defect_type}_{image_path.stem}",
                )
                region = normal_guided_region
                strategy = localization_policy
            elif localization_policy == "bottle_rim_guided_qwen":
                region, bottle_record = bottle_rim_guided_qwen_region(
                    image_size=image.size,
                    qwen_region=region,
                    defect_type=defect_type,
                    auto=auto,
                )
                normal_guided_record = bottle_record
                strategy = localization_policy
            elif localization_policy == "bottle_surface_guided":
                region, bottle_record = bottle_surface_guided_region(
                    image_size=image.size,
                    qwen_region=region,
                    auto=auto,
                )
                normal_guided_record = bottle_record
                strategy = localization_policy
            elif _should_run_grid_rescue(category, defect_type, auto):
                grid_record = _run_grid_rescue(
                    extractor=extractor,
                    image=image,
                    category=category,
                    defect_type=defect_type,
                    description=_description_for(config, descriptions, category, defect_type, image_path),
                    auto=auto,
                )
                if grid_record.get("selected"):
                    region = tuple(int(value) for value in grid_record["region_xyxy"])  # type: ignore[assignment]
                    strategy = "spatial_grid"
            truth = _load_binary(reference["official_reference_mask"], image.size)
            metrics = _bbox_metrics(region, truth, image.size)
            row = {
                "sample_id": f"{category}/{defect_type}/{image_path.name}",
                "category": category,
                "defect_type": defect_type,
                "image": image_path.name,
                "prompt_style": str(auto.get("qwen_prompt_style", "defect_localization_json")),
                "localization_status": localization.get("status", ""),
                "raw_bbox_xyxy": json.dumps(localization.get("raw_bbox_xyxy", [])),
                "region_xyxy": json.dumps(list(region)),
                "sub_boxes_xyxy": json.dumps(localization.get("sub_boxes_xyxy", [])),
                "sub_box_region_selected": bool(localization.get("sub_box_region_selected", False)),
                "localization_strategy": strategy,
                "auto_router_reason": policy_diagnostics.get("reason", ""),
                "repeated_texture_score": policy_diagnostics.get("repeated_texture_score", 0.0),
                "rim_geometry_score": policy_diagnostics.get("rim_geometry_score", 0.0),
                "qwen_confidence": parsed.get("confidence"),
                "qwen_evidence": parsed.get("evidence"),
                **metrics,
                "image_path": str(image_path),
                "official_mask_path": str(reference["official_reference_mask"]),
            }
            rows.append(row)
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "sample_id": row["sample_id"],
                            "prompt": prompt,
                            "qwen_text": result["text"],
                            "parsed": _json_safe(parsed),
                            "localization": _json_safe(localization),
                            "normal_guided_qwen_verify": _json_safe(normal_guided_record),
                            "grid_rescue": _json_safe(grid_record),
                            "metrics": metrics,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        extractor.close()

    rows.sort(key=lambda item: (item["category"], item["defect_type"], item["image"]))
    baseline = _load_baseline(Path(args.baseline))
    baseline_geometry = _load_baseline_geometry(Path(args.baseline_metadata))
    csv_path = report_dir / "qwen_bbox_metrics.csv"
    sheet_path = report_dir / "qwen_bbox_visual_review.png"
    report_path = report_dir / "qwen_bbox_prompt_v2_review.md"
    _write_csv(csv_path, rows)
    _write_sheet(sheet_path, rows, baseline, baseline_geometry)
    _write_report(report_path, rows, baseline, csv_path, sheet_path, raw_path)
    print(report_path)
    print(sheet_path)


def _load_references(dataset_root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest_path = dataset_root / "pilot_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected isolated pilot manifest: {manifest_path}")
    pilot = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        (str(record["category"]), str(record["defect_type"]), Path(record["target_image"]).name): record
        for record in pilot["records"]
        if record.get("role") == "defect" and record.get("official_reference_mask")
    }


def _should_run_grid_rescue(category: str, defect_type: str, auto: dict[str, Any]) -> bool:
    if not bool(auto.get("qwen_spatial_grid_rescue", False)):
        return False
    targets = auto.get("qwen_spatial_grid_targets", [])
    if not isinstance(targets, list | tuple):
        return False
    key = f"{category}/{defect_type}"
    return key in {str(target) for target in targets}


def _run_grid_rescue(
    *,
    extractor: QwenFeatureExtractor,
    image: Image.Image,
    category: str,
    defect_type: str,
    description: str,
    auto: dict[str, Any],
) -> dict[str, Any]:
    step = max(32, int(auto.get("qwen_spatial_grid_step_px", 64)))
    grid = _grid_image(image, step)
    prompt = _grid_prompt(category, defect_type, image.size, step, description)
    result = extractor.analyze(grid, prompt)
    record: dict[str, Any] = {
        "selected": False,
        "qwen_text": result["text"],
        "prompt": prompt,
        "failure_reason": None,
    }
    try:
        parsed = parse_qwen_bbox_payload(str(result["text"]), image.size)
        grid_auto = dict(auto)
        grid_auto["min_box_area_ratio"] = float(auto.get("qwen_spatial_grid_min_area_ratio", auto.get("min_box_area_ratio", 0.001)))
        region, localization = _validated_or_fallback_region(parsed["bbox_xyxy"], image.size, auto=grid_auto)
    except Exception as exc:
        record["failure_reason"] = str(exc)
        return record
    max_height_fraction = float(auto.get("qwen_spatial_grid_max_height_fraction", 1.0))
    height_fraction = (region[3] - region[1]) / max(1, image.size[1])
    record.update(
        {
            "parsed": _json_safe(parsed),
            "localization": _json_safe(localization),
            "region_xyxy": list(region),
            "height_fraction": round(height_fraction, 4),
        }
    )
    if localization.get("status") == "valid" and height_fraction <= max_height_fraction:
        record["selected"] = True
    else:
        record["failure_reason"] = f"grid region height_fraction={height_fraction:.4f} exceeds {max_height_fraction:.4f}"
    return record


def _grid_image(image: Image.Image, step: int) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    width, height = result.size
    for y in range(0, height, step):
        y2 = min(height - 1, y + step - 1)
        draw.rectangle((0, y, width - 1, y), fill=(255, 40, 40), width=2)
        draw.text((5, y + 5), f"Y{y}-{y2}", fill=(255, 40, 40), font=font)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(255, 40, 40), width=2)
    return result


def _grid_prompt(category: str, defect_type: str, image_size: tuple[int, int], step: int, description: str) -> str:
    width, height = image_size
    return (
        f"You are inspecting one gridded industrial {category} image of size {width}x{height}. "
        f"Horizontal red grid lines mark {step}px y-bands. Localize the {defect_type} defect matching: {description!r}. "
        "Use the red y-band labels to avoid selecting the whole vertical zipper chain. "
        "Return JSON only with keys \"bbox_xyxy\", \"defect_type\", \"confidence\", \"evidence\", and \"self_check\". "
        "\"bbox_xyxy\" must use original image pixel coordinates [x1, y1, x2, y2], not normalized coordinates. "
        "For zipper broken_teeth or split_teeth, choose the short local y-band where the abnormal teeth appear. "
        "Do not box the full height of the zipper. Do not box normal repeating teeth."
    )


def _load_binary(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def _bbox_metrics(region: tuple[int, int, int, int], truth: np.ndarray, image_size: tuple[int, int]) -> dict[str, float | int]:
    region_mask = np.zeros_like(truth, dtype=bool)
    region_mask[region[1] : region[3], region[0] : region[2]] = True
    intersection = int((region_mask & truth).sum())
    union = int((region_mask | truth).sum())
    region_pixels = int(region_mask.sum())
    truth_pixels = int(truth.sum())
    recall = intersection / max(1, truth_pixels)
    precision = intersection / max(1, region_pixels)
    iou = intersection / max(1, union)
    dice = 2 * intersection / max(1, region_pixels + truth_pixels)
    area_fraction = region_pixels / max(1, image_size[0] * image_size[1])
    return {
        "qwen_region_recall": round(recall, 4),
        "qwen_region_precision": round(precision, 4),
        "qwen_region_iou": round(iou, 4),
        "qwen_region_dice": round(dice, 4),
        "qwen_region_area_fraction": round(area_fraction, 4),
        "truth_pixels": truth_pixels,
        "region_pixels": region_pixels,
    }


def _load_baseline(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", ""))
            rows[sample_id] = {
                "qwen_region_recall": _as_float(row.get("qwen_region_recall")),
                "qwen_region_iou": _as_float(row.get("qwen_region_iou")),
            }
    return rows


def _load_baseline_geometry(path: Path) -> dict[str, tuple[int, int, int, int]]:
    if not path.exists():
        return {}
    geometry: dict[str, tuple[int, int, int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = f"{row['category']}/{row['defect_type']}/{Path(row['image_path']).name}"
        region = row.get("region_xyxy")
        if isinstance(region, list | tuple) and len(region) == 4:
            geometry[sample_id] = tuple(int(value) for value in region)  # type: ignore[assignment]
    return geometry


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sheet(
    path: Path,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, float]],
    baseline_geometry: dict[str, tuple[int, int, int, int]],
) -> None:
    thumb = 190
    label = 50
    header = 38
    columns = ("input", "official mask", "old qwen box", "new qwen box")
    sheet = Image.new("RGB", (thumb * len(columns), header + len(rows) * (thumb + label)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for col, title in enumerate(columns):
        draw.rectangle((col * thumb, 0, (col + 1) * thumb, header), fill=(232, 237, 245))
        draw.text((col * thumb + 8, 12), title, fill=(25, 35, 50), font=font)
    for index, row in enumerate(rows):
        y = header + index * (thumb + label)
        image = Image.open(row["image_path"]).convert("RGB")
        truth = _load_binary(row["official_mask_path"], image.size)
        old_region = baseline_geometry.get(row["sample_id"])
        new_region = tuple(json.loads(str(row["region_xyxy"])))
        panels = [
            image,
            _overlay(image, truth, (45, 190, 90), 0.55),
            _region_overlay(image, old_region, (255, 150, 30)) if old_region else image,
            _region_overlay(image, new_region, (40, 120, 255)),
        ]
        for col, panel in enumerate(panels):
            sheet.paste(panel.resize((thumb, thumb), Image.Resampling.BILINEAR), (col * thumb, y))
        old = baseline.get(row["sample_id"], {})
        text = (
            f"{row['sample_id']} | old R={old.get('qwen_region_recall', 0.0):.3f} "
            f"new R={float(row['qwen_region_recall']):.3f} | "
            f"new IoU={float(row['qwen_region_iou']):.3f}"
        )
        draw.text((8, y + thumb + 10), text, fill=(25, 35, 50), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", image.size, (*color, 0))
    layer.putalpha(Image.fromarray(np.uint8(mask) * round(255 * alpha), mode="L"))
    return Image.alpha_composite(base, layer).convert("RGB")


def _region_overlay(image: Image.Image, region: tuple[int, int, int, int], color: tuple[int, int, int]) -> Image.Image:
    result = image.copy()
    ImageDraw.Draw(result).rectangle(region, outline=color, width=4)
    return result


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    baseline: dict[str, dict[str, float]],
    csv_path: Path,
    sheet_path: Path,
    raw_path: Path,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["defect_type"])].append(row)
    lines = [
        "# Qwen BBox Localization Review",
        "",
        "This report evaluates Qwen localization plus optional bottle geometry and normal-guided Qwen verification policies. "
        "Official masks are used only after localization has returned boxes.",
        "",
        "## Aggregate Results",
        "",
        "| Category / defect | N | Old recall | New recall | Delta recall | Old IoU | New IoU | Delta IoU | New area frac | Sub-box primary |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, group in sorted(grouped.items()):
        old_recall = _baseline_mean(group, baseline, "qwen_region_recall")
        old_iou = _baseline_mean(group, baseline, "qwen_region_iou")
        new_recall = _mean(group, "qwen_region_recall")
        new_iou = _mean(group, "qwen_region_iou")
        lines.append(
            f"| `{key[0]}/{key[1]}` | {len(group)} | {old_recall:.4f} | {new_recall:.4f} | "
            f"{new_recall - old_recall:+.4f} | {old_iou:.4f} | {new_iou:.4f} | {new_iou - old_iou:+.4f} | "
            f"{_mean(group, 'qwen_region_area_fraction'):.4f} | "
            f"{sum(1 for row in group if row['sub_box_region_selected'])}/{len(group)} |"
        )
    lines.extend(_overall_lines(rows, baseline))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Better Qwen recall means the search region contains more official defect pixels.",
            "- Better Qwen IoU means the search region is both more complete and less broad.",
            "- If recall improves but IoU drops, Qwen found the defect but boxed too much normal texture.",
            "- `bottle_rim_guided_qwen` and `bottle_surface_guided` prioritize recall for bottle defects by using "
            "official-mask-free object geometry priors.",
            "- `normal_guided_qwen_verify` prioritizes recall: normal-memory proposes y-windows and Qwen selects "
            "candidate strips instead of drawing coordinates from scratch.",
            "- Zipper fabric-border defects intentionally stay on the direct Qwen bbox path. They are border/fabric "
            "abnormalities, not tooth-chain failures, so the tooth-window verifier would search the wrong structure.",
            "- Broad high-recall boxes are intentional at this stage because downstream mask refinement removes "
            "normal texture inside the search region.",
            "",
            "## Artifacts",
            "",
            f"- Metrics CSV: `{csv_path}`",
            f"- Raw Qwen JSONL: `{raw_path}`",
            f"- Visual review: `{sheet_path}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overall_lines(rows: list[dict[str, Any]], baseline: dict[str, dict[str, float]]) -> list[str]:
    lines = []
    for category in sorted({row["category"] for row in rows}):
        group = [row for row in rows if row["category"] == category]
        old_recall = _baseline_mean(group, baseline, "qwen_region_recall")
        old_iou = _baseline_mean(group, baseline, "qwen_region_iou")
        new_recall = _mean(group, "qwen_region_recall")
        new_iou = _mean(group, "qwen_region_iou")
        lines.append(
            f"| **{category.title()} overall** | {len(group)} | {old_recall:.4f} | {new_recall:.4f} | "
            f"{new_recall - old_recall:+.4f} | {old_iou:.4f} | {new_iou:.4f} | {new_iou - old_iou:+.4f} | "
            f"{_mean(group, 'qwen_region_area_fraction'):.4f} | "
            f"{sum(1 for row in group if row['sub_box_region_selected'])}/{len(group)} |"
        )
    return lines


def _baseline_mean(rows: list[dict[str, Any]], baseline: dict[str, dict[str, float]], key: str) -> float:
    values = [baseline[row["sample_id"]][key] for row in rows if row["sample_id"] in baseline]
    return float(np.mean(values)) if values else 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
