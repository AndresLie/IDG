from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from iadgen_v2.config import AppConfig


POLICIES = ("hard_binary", "vote_soft", "calibrated_soft")


def run_phase10_mask_quality_ablation(config: AppConfig, provider: str | None = None) -> Path:
    provider = provider or str(config.data.get("auto_masks", {}).get("provider", "qwen"))
    metadata_path = config.output_dir / "auto_masks" / provider / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing auto-mask metadata for Phase 10 mask-quality ablation: {metadata_path}")

    report_dir = config.report_dir / "phase10_mask_quality" / provider
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: (str(row.get("category", "")), str(row.get("defect_type", "")), Path(str(row.get("image_path", ""))).stem))

    records: list[dict[str, Any]] = []
    generated_policy_paths: dict[tuple[str, str], str] = {}
    for row in rows:
        policy_masks = _policy_masks(row, report_dir)
        generated_policy_paths.update({(str(row["image_path"]), key): value for key, value in policy_masks.items()})
        for policy, mask_path in policy_masks.items():
            records.append(_critic_record(row, policy, Path(mask_path)))

    csv_path = report_dir / "mask_quality_metrics.csv"
    jsonl_path = report_dir / "mask_quality_metrics.jsonl"
    _write_csv(csv_path, records)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    contact_sheet = report_dir / "mask_quality_ablation_contact_sheet.png"
    _write_visual_sheet(contact_sheet, rows, generated_policy_paths)
    summary = report_dir / "mask_quality_ablation_report.md"
    _write_summary(summary, records, contact_sheet, csv_path, jsonl_path)
    return summary


def _policy_masks(row: dict[str, Any], report_dir: Path) -> dict[str, str]:
    image_path = str(row["image_path"])
    stem = f"{row.get('category', 'unknown')}_{row.get('defect_type', 'unknown')}_{Path(image_path).stem}"
    policy_dir = report_dir / "policy_masks"
    policy_dir.mkdir(parents=True, exist_ok=True)

    variant_paths = row.get("mask_variant_paths", {})
    settings = row.get("settings", {})
    candidate_paths = settings.get("candidate_refined_paths", {})
    image_size = Image.open(image_path).size
    region = tuple(int(v) for v in row.get("region_xyxy", (0, 0, image_size[0], image_size[1])))

    hard_path = str(variant_paths.get("eval_tight") or row.get("eval_mask_path") or row.get("refined_mask_path") or row.get("training_mask_path"))
    calibrated_path = str(variant_paths.get("training_soft") or row.get("training_mask_path") or row.get("refined_mask_path"))
    vote_soft = _vote_soft_mask(candidate_paths, row.get("refined_mask_path"), region, image_size)
    vote_path = policy_dir / f"{stem}_vote_soft.png"
    vote_soft.save(vote_path)
    return {
        "hard_binary": hard_path,
        "vote_soft": str(vote_path),
        "calibrated_soft": calibrated_path,
    }


def _vote_soft_mask(
    candidate_refined_paths: dict[str, str],
    fallback_path: str | None,
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> Image.Image:
    arrays: list[np.ndarray] = []
    if fallback_path and Path(fallback_path).exists():
        arrays.append(np.asarray(Image.open(fallback_path).convert("L").resize(image_size, Image.Resampling.NEAREST), dtype=np.uint8) > 0)
    for value in candidate_refined_paths.values():
        path = Path(str(value))
        if not path.exists():
            continue
        arr = np.asarray(Image.open(path).convert("L").resize(image_size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
        if arr.any():
            arrays.append(arr)
    if not arrays:
        return Image.new("L", image_size, 0)
    vote = np.stack(arrays, axis=0).mean(axis=0)
    left, top, right, bottom = region
    fence = np.zeros(vote.shape, dtype=bool)
    fence[top:bottom, left:right] = True
    soft = np.zeros(vote.shape, dtype=np.uint8)
    soft[vote >= 0.34] = 128
    soft[vote >= 0.67] = 255
    soft[~fence] = 0
    return Image.fromarray(soft, mode="L").filter(ImageFilter.GaussianBlur(radius=1.2))


def _critic_record(row: dict[str, Any], policy: str, mask_path: Path) -> dict[str, Any]:
    image_path = Path(str(row["image_path"]))
    image = Image.open(image_path).convert("RGB")
    mask = np.asarray(Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    binary = mask > 0.05
    region = tuple(int(v) for v in row.get("region_xyxy", (0, 0, image.width, image.height)))
    qwen_area = max(1, (region[2] - region[0]) * (region[3] - region[1]))
    settings = row.get("settings", {})
    morphology = str(settings.get("quality_morphology") or settings.get("scratch_morphology_class") or "unknown")
    heat_support, heatmaps_used = _heatmap_support(row, mask, image.size)
    agreement = _candidate_agreement(row, binary, image.size)
    components = _connected_components(binary)
    component_count = len(components)
    area_fraction = float(binary.sum() / max(1, mask.size))
    box_fraction = float(binary.sum() / qwen_area)
    entropy = _soft_entropy(mask)
    grain_penalty = _grain_alignment_penalty(image, binary, region, morphology)
    area_score = _area_sanity_score(box_fraction, morphology)
    fragmentation_score = max(0.0, 1.0 - max(0, component_count - _component_target(morphology)) / 32.0)
    soft_score = _soft_label_score(entropy, policy, morphology)
    critic = (
        0.36 * heat_support
        + 0.24 * agreement
        + 0.16 * area_score
        + 0.12 * fragmentation_score
        + 0.08 * soft_score
        - 0.12 * grain_penalty
    )
    return {
        "image": image_path.name,
        "category": row.get("category", ""),
        "defect_type": row.get("defect_type", ""),
        "selected_refinement": settings.get("selected_refinement", ""),
        "morphology": morphology,
        "label_policy": settings.get("label_policy", {}).get("label_policy", ""),
        "policy": policy,
        "mask_path": str(mask_path),
        "critic_score": round(float(max(0.0, min(1.0, critic))), 4),
        "heatmap_support": round(float(heat_support), 4),
        "candidate_agreement": round(float(agreement), 4),
        "area_sanity": round(float(area_score), 4),
        "fragmentation_score": round(float(fragmentation_score), 4),
        "soft_label_score": round(float(soft_score), 4),
        "grain_alignment_penalty": round(float(grain_penalty), 4),
        "mask_area_fraction": round(area_fraction, 6),
        "mask_to_qwen_box_fraction": round(box_fraction, 6),
        "component_count": component_count,
        "soft_entropy": round(float(entropy), 4),
        "heatmaps_used": ",".join(heatmaps_used),
    }


def _heatmap_support(row: dict[str, Any], mask: np.ndarray, image_size: tuple[int, int]) -> tuple[float, list[str]]:
    paths = row.get("settings", {}).get("candidate_heatmap_paths", {})
    preferred = ("patchcore_guided", "nearest_normal_residual", "fft_texture_suppression", "normal_anomaly")
    heatmaps: list[np.ndarray] = []
    used: list[str] = []
    for mode in preferred:
        path_value = paths.get(mode)
        if not path_value or not Path(str(path_value)).exists():
            continue
        heat = np.asarray(Image.open(path_value).convert("L").resize(image_size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        if float(heat.max()) <= 0.0:
            continue
        heatmaps.append(_normalize(heat))
        used.append(mode)
    if not heatmaps:
        return 0.0, used
    fused = np.mean(np.stack(heatmaps, axis=0), axis=0)
    weights = mask / max(float(mask.max()), 1e-6)
    active = weights > 0.02
    if not active.any():
        return 0.0, used
    inside = float(np.average(fused[active], weights=np.clip(weights[active], 1e-4, None)))
    outside = float(fused[~active].mean()) if (~active).any() else 0.0
    return max(0.0, min(1.0, inside - 0.35 * outside)), used


def _candidate_agreement(row: dict[str, Any], binary: np.ndarray, image_size: tuple[int, int]) -> float:
    paths = row.get("settings", {}).get("candidate_refined_paths", {})
    arrays = []
    for value in paths.values():
        path = Path(str(value))
        if path.exists():
            arr = np.asarray(Image.open(path).convert("L").resize(image_size, Image.Resampling.NEAREST), dtype=np.uint8) > 0
            if arr.any():
                arrays.append(arr)
    if not arrays or not binary.any():
        return 0.0
    vote = np.stack(arrays, axis=0).mean(axis=0)
    return float(vote[binary].mean())


def _area_sanity_score(box_fraction: float, morphology: str) -> float:
    if morphology == "multi_scuff":
        low, high = 0.025, 0.24
    else:
        low, high = 0.004, 0.22
    if low <= box_fraction <= high:
        return 1.0
    if box_fraction < low:
        return max(0.0, box_fraction / max(low, 1e-6))
    return max(0.0, 1.0 - (box_fraction - high) / max(high, 1e-6))


def _component_target(morphology: str) -> int:
    return 18 if morphology == "multi_scuff" else 8


def _soft_label_score(entropy: float, policy: str, morphology: str) -> float:
    if morphology == "multi_scuff":
        if policy == "hard_binary":
            return 0.25
        target = 0.70
    else:
        target = 0.02 if policy == "hard_binary" else 0.10
    return max(0.0, 1.0 - abs(entropy - target) / 0.45)


def _soft_entropy(mask: np.ndarray) -> float:
    values = mask[(mask > 0.0) & (mask < 1.0)]
    if values.size == 0:
        return 0.0
    entropy = -(values * np.log2(values + 1e-6) + (1.0 - values) * np.log2(1.0 - values + 1e-6))
    return float(entropy.mean())


def _grain_alignment_penalty(
    image: Image.Image,
    binary: np.ndarray,
    region: tuple[int, int, int, int],
    morphology: str,
) -> float:
    if not binary.any() or morphology == "multi_scuff":
        return 0.0
    ys, xs = np.where(binary)
    if len(xs) < 8:
        return 0.0
    aspect = _aspect(xs, ys)
    if aspect < 2.0:
        return 0.0
    mask_angle = _angle_degrees(xs, ys)
    texture_angle = _dominant_texture_angle(image, region)
    if texture_angle is None:
        return 0.0
    diff = abs((mask_angle - texture_angle + 90.0) % 180.0 - 90.0)
    return max(0.0, 1.0 - diff / 18.0) * 0.5


def _dominant_texture_angle(image: Image.Image, region: tuple[int, int, int, int]) -> float | None:
    left, top, right, bottom = region
    crop = np.asarray(image.convert("L"), dtype=np.float32)[top:bottom, left:right]
    if crop.size < 16:
        return None
    gy, gx = np.gradient(crop)
    magnitude = np.hypot(gx, gy)
    mask = magnitude > np.percentile(magnitude, 75.0)
    if int(mask.sum()) < 8:
        return None
    angles = np.degrees(np.arctan2(gy[mask], gx[mask])) + 90.0
    doubled = np.deg2rad(2.0 * angles)
    mean_angle = math.degrees(math.atan2(float(np.sin(doubled).mean()), float(np.cos(doubled).mean()))) / 2.0
    return mean_angle % 180.0


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            components.append(component)
    return components


def _angle_degrees(xs: np.ndarray, ys: np.ndarray) -> float:
    points = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    return float((math.degrees(math.atan2(float(direction[1]), float(direction[0]))) + 180.0) % 180.0)


def _aspect(xs: np.ndarray, ys: np.ndarray) -> float:
    return max((xs.max() - xs.min() + 1) / max(1, ys.max() - ys.min() + 1), (ys.max() - ys.min() + 1) / max(1, xs.max() - xs.min() + 1))


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 98.0))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "image",
        "category",
        "defect_type",
        "selected_refinement",
        "morphology",
        "label_policy",
        "policy",
        "critic_score",
        "heatmap_support",
        "candidate_agreement",
        "area_sanity",
        "fragmentation_score",
        "soft_label_score",
        "grain_alignment_penalty",
        "mask_area_fraction",
        "mask_to_qwen_box_fraction",
        "component_count",
        "soft_entropy",
        "heatmaps_used",
        "mask_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_visual_sheet(path: Path, rows: list[dict[str, Any]], policy_paths: dict[tuple[str, str], str]) -> None:
    thumb_w, thumb_h, label_h, header_h = 190, 190, 44, 44
    columns = ["input", "selected", *POLICIES, "uncertainty"]
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (thumb_w * len(columns), header_h + len(rows) * (label_h + thumb_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(columns):
        x = col * thumb_w
        draw.rectangle((x, 0, x + thumb_w, header_h), fill=(235, 235, 235), outline=(210, 210, 210))
        draw.text((x + 6, 10), title, fill="black", font=font)
    for row_index, row in enumerate(rows):
        image_path = str(row["image_path"])
        y = header_h + row_index * (label_h + thumb_h)
        settings = row.get("settings", {})
        variants = row.get("mask_variant_paths", {})
        content = {
            "input": Image.open(image_path).convert("RGB"),
            "selected": _overlay(image_path, row.get("refined_mask_path"), color=(255, 0, 0), soft=False),
            "hard_binary": _overlay(image_path, policy_paths[(image_path, "hard_binary")], color=(255, 0, 0), soft=False),
            "vote_soft": _overlay(image_path, policy_paths[(image_path, "vote_soft")], color=(0, 150, 255), soft=True),
            "calibrated_soft": _overlay(image_path, policy_paths[(image_path, "calibrated_soft")], color=(255, 90, 0), soft=True),
            "uncertainty": _heat(variants.get("uncertainty_map") or row.get("uncertainty_mask_path")),
        }
        labels = {
            "input": Path(image_path).name,
            "selected": f"selected: {settings.get('selected_refinement', '')}",
            "hard_binary": "hard binary",
            "vote_soft": "vote soft",
            "calibrated_soft": "calibrated soft",
            "uncertainty": "uncertainty",
        }
        for col, key in enumerate(columns):
            x = col * thumb_w
            draw.rectangle((x, y, x + thumb_w, y + label_h), fill=(248, 248, 248), outline=(220, 220, 220))
            draw.text((x + 5, y + 7), labels[key][:31], fill="black", font=font)
            sheet.paste(_fit(content[key], thumb_w, thumb_h), (x, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _overlay(image_path: str, mask_path: str | None, *, color: tuple[int, int, int], soft: bool) -> Image.Image:
    base = Image.open(image_path).convert("RGB")
    if not mask_path or not Path(str(mask_path)).exists():
        return Image.new("RGB", base.size, (245, 245, 245))
    mask = Image.open(mask_path).convert("L").resize(base.size, Image.Resampling.BILINEAR if soft else Image.Resampling.NEAREST)
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    if not soft:
        alpha = (alpha > 0.0).astype(np.float32)
    out = np.asarray(base, dtype=np.float32)
    tint = np.zeros_like(out)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    blend = np.clip(alpha * 0.62, 0.0, 0.62)[..., None]
    out = out * (1.0 - blend) + tint * blend
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _heat(mask_path: str | None) -> Image.Image:
    if not mask_path or not Path(str(mask_path)).exists():
        return Image.new("RGB", (64, 64), (245, 245, 245))
    gray = Image.open(mask_path).convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(np.clip(arr * 255, 0, 255))
    rgb[..., 1] = np.uint8(np.clip((1.0 - np.abs(arr - 0.5) * 2.0) * 180, 0, 180))
    rgb[..., 2] = np.uint8(np.clip((1.0 - arr) * 90, 0, 90))
    return Image.fromarray(rgb, mode="RGB")


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    scale = min(width / image.width, height / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def _write_summary(summary_path: Path, records: list[dict[str, Any]], contact_sheet: Path, csv_path: Path, jsonl_path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {policy: [] for policy in POLICIES}
    for record in records:
        grouped.setdefault(str(record["policy"]), []).append(record)
    lines = [
        "# Phase 10 Mask Quality Ablation",
        "",
        "This report compares mask-training policies using auto-mask pseudo-label evidence.",
        "",
        "## Artifacts",
        "",
        "```text",
        str(contact_sheet),
        str(csv_path),
        str(jsonl_path),
        "```",
        "",
        "## Policy Summary",
        "",
        "| Policy | Mean critic | Mean heat support | Mean agreement | Mean area sanity | Mean entropy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in POLICIES:
        rows = grouped.get(policy, [])
        lines.append(
            f"| `{policy}` | {_mean(rows, 'critic_score'):.4f} | {_mean(rows, 'heatmap_support'):.4f} | "
            f"{_mean(rows, 'candidate_agreement'):.4f} | {_mean(rows, 'area_sanity'):.4f} | {_mean(rows, 'soft_entropy'):.4f} |"
        )
    best_by_image: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["image"])
        if key not in best_by_image or float(record["critic_score"]) > float(best_by_image[key]["critic_score"]):
            best_by_image[key] = record
    lines.extend(["", "## Per-Image Winner", "", "| Image | Best policy | Critic | Selected refinement | Morphology |", "| --- | --- | ---: | --- | --- |"])
    for image_name in sorted(best_by_image):
        row = best_by_image[image_name]
        lines.append(
            f"| `{image_name}` | `{row['policy']}` | {float(row['critic_score']):.4f} | "
            f"`{row['selected_refinement']}` | `{row['morphology']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `hard_binary` represents the old crisp training-region style.",
            "- `vote_soft` represents uncertainty from binary candidate agreement only.",
            "- `calibrated_soft` represents the current heatmap-calibrated soft target.",
            "",
            "The critic is diagnostic, not ground truth. It is meant to flag whether a mask policy is supported by normal-feature anomaly heatmaps, candidate agreement, area sanity, and acceptable fragmentation.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0.0)) for row in rows]))
