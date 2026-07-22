from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from iadgen_v2.config import AppConfig
from iadgen_v2.dataset import load_manifest


PHASE11_SCHEMA_VERSION = 1


def run_phase11_tfidg_critic(config: AppConfig, provider: str | None = None) -> Path:
    """Score Phase 4 generations with a TF-IDG-inspired training-free critic.

    The real TF-IDG paper uses diffusion-internal feature priors. This local
    critic keeps the same spirit but stays dependency-light: it checks whether a
    generated defect visually matches a real defect exemplar, edits enough of
    the selected mask, preserves texture outside the mask, and avoids leakage.
    """

    phase11 = _phase11_config(config)
    provider = provider or str(config.data.get("phase4", {}).get("provider", "qwen"))
    phase4_rows = _load_phase4_rows(config, provider)
    if not phase4_rows:
        raise FileNotFoundError(f"No Phase 4 metadata found under {config.output_dir / 'phase4' / provider}")
    phase4_rows = _filter_rows(phase4_rows, phase11)

    max_rows = int(phase11.get("max_rows", len(phase4_rows)))
    if max_rows > 0:
        phase4_rows = phase4_rows[:max_rows]
    references = _reference_index(config)
    report_dir = config.report_dir / "phase11_tfidg_critic" / provider
    report_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for row in phase4_rows:
        record = _critic_record(row, references, phase11)
        records.append(record)

    csv_path = report_dir / "tfidg_lite_metrics.csv"
    jsonl_path = report_dir / "tfidg_lite_metrics.jsonl"
    _write_csv(csv_path, records)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    contact_sheet = report_dir / "tfidg_lite_contact_sheet.png"
    _write_contact_sheet(contact_sheet, records, phase11)
    summary = report_dir / "tfidg_lite_report.md"
    _write_summary(summary, records, contact_sheet, csv_path, jsonl_path, phase11)
    return summary


def _load_phase4_rows(config: AppConfig, provider: str) -> list[dict[str, Any]]:
    phase4_dir = config.output_dir / "phase4" / provider
    combined = phase4_dir / "metadata.jsonl"
    paths = [combined] if combined.exists() else sorted(phase4_dir.glob("*/metadata.jsonl"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("output_path") or f"{path}:{len(rows)}")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row.get("category", "")),
            str(row.get("defect_type", "")),
            str(row.get("variant", "")),
            str(row.get("quality_profile", "")),
            int(row.get("generation_seed", 0)),
        )
    )
    return rows


def _filter_rows(rows: list[dict[str, Any]], phase11: dict[str, Any]) -> list[dict[str, Any]]:
    categories = {str(value) for value in phase11.get("categories", [])}
    defect_types = {str(value) for value in phase11.get("defect_types", [])}
    variants = {str(value) for value in phase11.get("variants", [])}
    profiles = {str(value) for value in phase11.get("quality_profiles", [])}
    filtered = []
    for row in rows:
        if categories and str(row.get("category", "")) not in categories:
            continue
        if defect_types and str(row.get("defect_type", "")) not in defect_types:
            continue
        if variants and str(row.get("variant", "")) not in variants:
            continue
        if profiles and str(row.get("quality_profile", "")) not in profiles:
            continue
        filtered.append(row)
    return filtered


def _reference_index(config: AppConfig) -> dict[tuple[str, str], list[dict[str, str]]]:
    manifest = load_manifest(config)
    targets = manifest.get("targets", {})
    index: dict[tuple[str, str], list[dict[str, str]]] = {}
    if not isinstance(targets, dict):
        return index
    for value in targets.values():
        if not isinstance(value, dict):
            continue
        category = str(value.get("category", ""))
        defect_type = str(value.get("defect_type", ""))
        refs: list[dict[str, str]] = []
        for sample in value.get("adaptation", []):
            if not isinstance(sample, dict):
                continue
            image_path = sample.get("image_path")
            mask_path = sample.get("training_mask_path") or sample.get("mask_path") or sample.get("eval_mask_path")
            if image_path and mask_path:
                refs.append({"image_path": str(image_path), "mask_path": str(mask_path)})
        index[(category, defect_type)] = refs
    return index


def _critic_record(row: dict[str, Any], references: dict[tuple[str, str], list[dict[str, str]]], phase11: dict[str, Any]) -> dict[str, Any]:
    category = str(row.get("category", ""))
    defect_type = str(row.get("defect_type", ""))
    output_path = Path(str(row.get("output_path", "")))
    background_path = Path(str(row.get("background_path", "")))
    refined_mask_path = Path(str(row.get("refined_mask_path", "")))
    inpaint_mask_path = Path(str(row.get("inpaint_mask_path") or refined_mask_path))
    settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}

    generated_full = Image.open(output_path).convert("RGB")
    background_full = Image.open(background_path).convert("RGB").resize(generated_full.size, Image.Resampling.BILINEAR)
    critic_size = int(phase11.get("critic_image_size", 160))
    generated = _resize_for_critic(generated_full, critic_size)
    background = _resize_for_critic(background_full, critic_size)
    refined = _load_mask(refined_mask_path, generated.size)
    inpaint = _load_mask(inpaint_mask_path, generated.size)
    effective_mask = np.maximum(refined, inpaint)
    reference = _select_reference(row, references.get((category, defect_type), []))

    if reference:
        ref_image = _resize_for_critic(Image.open(reference["image_path"]).convert("RGB"), critic_size)
        ref_mask = _load_mask(Path(reference["mask_path"]), ref_image.size)
        alignment = _feature_alignment_score(generated, background, effective_mask, ref_image, ref_mask)
        reference_path = reference["image_path"]
    else:
        alignment = 0.0
        reference_path = ""

    coverage = _adaptive_mask_coverage_score(generated, background, effective_mask, phase11)
    texture = _texture_preservation_score(generated, background, effective_mask)
    leakage = _leakage_score(generated, background, refined, inpaint, phase11)
    morphology = str(settings.get("quality_morphology") or settings.get("label_policy", {}).get("quality_morphology", "") or "unknown")
    morphology_fit = _morphology_fit_score(generated, background, refined, morphology)

    weights = _weights(phase11)
    score = (
        weights["feature_alignment"] * alignment
        + weights["mask_coverage"] * coverage
        + weights["texture_preservation"] * texture
        + weights["leakage"] * leakage
        + weights["morphology_fit"] * morphology_fit
    )
    reject_reasons = _reject_reasons(alignment, coverage, texture, leakage, morphology_fit, phase11, bool(reference))
    return {
        "category": category,
        "defect_type": defect_type,
        "variant": str(row.get("variant", "")),
        "quality_profile": str(row.get("quality_profile", "")),
        "morphology": morphology,
        "generation_seed": int(row.get("generation_seed", 0)),
        "output_path": str(output_path),
        "background_path": str(background_path),
        "refined_mask_path": str(refined_mask_path),
        "inpaint_mask_path": str(inpaint_mask_path),
        "reference_path": reference_path,
        "tfidg_lite_score": round(float(max(0.0, min(1.0, score))), 4),
        "feature_alignment_score": round(float(alignment), 4),
        "adaptive_mask_coverage_score": round(float(coverage), 4),
        "texture_preservation_score": round(float(texture), 4),
        "leakage_score": round(float(leakage), 4),
        "morphology_fit_score": round(float(morphology_fit), 4),
        "phase4_generation_quality_score": round(float(row.get("generation_quality_score", 0.0)), 4),
        "phase4_defect_visibility_score": round(float(row.get("defect_visibility_score", 0.0)), 4),
        "phase4_background_l1": round(float(row.get("background_preservation_l1", 0.0)), 4),
        "phase4_outside_refined_change_fraction": round(float(row.get("outside_refined_change_fraction", 0.0)), 4),
        "reject_reasons": ",".join(reject_reasons),
        "accepted": not reject_reasons,
        "schema_version": PHASE11_SCHEMA_VERSION,
    }


def _select_reference(row: dict[str, Any], references: list[dict[str, str]]) -> dict[str, str] | None:
    if not references:
        return None
    seed = int(row.get("generation_seed", 0))
    return references[seed % len(references)]


def _feature_alignment_score(
    generated: Image.Image,
    background: Image.Image,
    mask: np.ndarray,
    reference: Image.Image,
    reference_mask: np.ndarray,
) -> float:
    gen_vec = _defect_feature_vector(generated, mask, background)
    ref_vec = _defect_feature_vector(reference, reference_mask, None)
    if gen_vec is None or ref_vec is None:
        return 0.0
    distance = float(np.linalg.norm(gen_vec - ref_vec) / math.sqrt(len(gen_vec)))
    return float(math.exp(-2.4 * distance))


def _defect_feature_vector(image: Image.Image, mask: np.ndarray, background: Image.Image | None) -> np.ndarray | None:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    weights = np.clip(mask.astype(np.float32), 0.0, 1.0)
    active = weights > 0.03
    if int(active.sum()) < 8:
        return None
    weights = weights / max(float(weights.sum()), 1e-6)
    gray = _gray(rgb)
    grad = _gradient_magnitude(gray)
    detail = np.abs(gray - _box_blur(gray, radius=3))
    features: list[float] = []
    for channel in range(3):
        values = rgb[:, :, channel]
        features.extend(_weighted_mean_std(values, weights))
    features.extend(_weighted_mean_std(gray, weights))
    features.extend(_weighted_mean_std(grad, weights))
    features.extend(_weighted_mean_std(detail, weights))
    if background is not None:
        bg = np.asarray(background, dtype=np.float32) / 255.0
        diff = np.abs(rgb - bg).mean(axis=2)
        features.extend(_weighted_mean_std(diff, weights))
    else:
        features.extend([0.0, 0.0])
    return np.asarray(features, dtype=np.float32)


def _adaptive_mask_coverage_score(generated: Image.Image, background: Image.Image, mask: np.ndarray, phase11: dict[str, Any]) -> float:
    active = mask > 0.03
    if int(active.sum()) < 8:
        return 0.0
    diff = _image_diff(generated, background)
    threshold = float(phase11.get("change_threshold", 0.045))
    changed = diff > threshold
    soft_weights = np.clip(mask, 0.0, 1.0)
    weighted_changed = float((changed.astype(np.float32) * soft_weights).sum() / max(float(soft_weights.sum()), 1e-6))
    target = float(phase11.get("target_mask_coverage", 0.38))
    if weighted_changed <= target:
        return max(0.0, weighted_changed / max(target, 1e-6))
    over = max(0.0, weighted_changed - float(phase11.get("max_mask_coverage", 0.92)))
    return max(0.0, 1.0 - over / 0.25)


def _texture_preservation_score(generated: Image.Image, background: Image.Image, mask: np.ndarray) -> float:
    outside = mask <= 0.02
    if int(outside.sum()) < 8:
        return 0.0
    gen = np.asarray(generated, dtype=np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32) / 255.0
    rgb_l1 = float(np.abs(gen - bg).mean(axis=2)[outside].mean())
    gen_grad = _gradient_magnitude(_gray(gen))
    bg_grad = _gradient_magnitude(_gray(bg))
    grad_l1 = float(np.abs(gen_grad - bg_grad)[outside].mean())
    return float(max(0.0, min(1.0, 1.0 - 8.0 * rgb_l1 - 2.5 * grad_l1)))


def _leakage_score(generated: Image.Image, background: Image.Image, refined: np.ndarray, inpaint: np.ndarray, phase11: dict[str, Any]) -> float:
    diff = _image_diff(generated, background)
    refined_active = refined > 0.03
    inpaint_active = inpaint > 0.03
    outside = ~refined_active
    shell = inpaint_active & ~refined_active
    threshold = float(phase11.get("change_threshold", 0.045))
    outside_changed = float((diff[outside] > threshold).mean()) if outside.any() else 0.0
    shell_changed = float((diff[shell] > threshold).mean()) if shell.any() else 0.0
    max_outside = float(phase11.get("max_outside_change_fraction", 0.18))
    outside_score = 1.0 - outside_changed / max(max_outside, 1e-6)
    shell_score = 1.0 - max(0.0, shell_changed - 0.35) / 0.45
    return float(max(0.0, min(1.0, 0.75 * outside_score + 0.25 * shell_score)))


def _morphology_fit_score(generated: Image.Image, background: Image.Image, refined: np.ndarray, morphology: str) -> float:
    active = refined > 0.05
    if int(active.sum()) < 8:
        return 0.0
    diff = _image_diff(generated, background)
    strong = active & (diff > max(0.045, float(np.percentile(diff[active], 60.0))))
    if int(strong.sum()) < 8:
        strong = active
    ys, xs = np.where(strong)
    aspect = _aspect(xs, ys)
    components = _connected_component_count(strong)
    if morphology in {"scratch_band", "single_stroke", "crack_band"}:
        aspect_score = min(1.0, max(0.0, (aspect - 1.5) / 6.0))
        frag_score = max(0.0, 1.0 - max(0, components - 8) / 20.0)
        return float(0.7 * aspect_score + 0.3 * frag_score)
    if morphology == "multi_scuff":
        area = float(strong.sum() / max(1, refined.size))
        area_score = max(0.0, min(1.0, area / 0.025))
        aspect_score = max(0.0, 1.0 - max(0.0, aspect - 8.0) / 12.0)
        return float(0.55 * area_score + 0.45 * aspect_score)
    return 0.6


def _reject_reasons(
    alignment: float,
    coverage: float,
    texture: float,
    leakage: float,
    morphology_fit: float,
    phase11: dict[str, Any],
    has_reference: bool,
) -> list[str]:
    reasons: list[str] = []
    if not has_reference:
        reasons.append("missing_reference")
    if alignment < float(phase11.get("min_feature_alignment_score", 0.28)):
        reasons.append("low_feature_alignment")
    if coverage < float(phase11.get("min_mask_coverage_score", 0.35)):
        reasons.append("low_mask_coverage")
    if texture < float(phase11.get("min_texture_preservation_score", 0.55)):
        reasons.append("poor_texture_preservation")
    if leakage < float(phase11.get("min_leakage_score", 0.45)):
        reasons.append("high_leakage")
    if morphology_fit < float(phase11.get("min_morphology_fit_score", 0.20)):
        reasons.append("poor_morphology_fit")
    return reasons


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=np.float32)
    return np.asarray(Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def _resize_for_critic(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0 or max(image.size) <= max_size:
        return image
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return resized


def _image_diff(a: Image.Image, b: Image.Image) -> np.ndarray:
    arr_a = np.asarray(a, dtype=np.float32) / 255.0
    arr_b = np.asarray(b, dtype=np.float32) / 255.0
    return np.abs(arr_a - arr_b).mean(axis=2)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray.astype(np.float32))
    return np.hypot(gx, gy).astype(np.float32)


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    pad = radius
    padded = np.pad(values, pad, mode="reflect")
    out = np.zeros_like(values, dtype=np.float32)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return out / float(size * size)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> list[float]:
    mean_value = float((values * weights).sum())
    var = float((((values - mean_value) ** 2) * weights).sum())
    return [mean_value, math.sqrt(max(0.0, var))]


def _aspect(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size == 0 or ys.size == 0:
        return 0.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return max(width / max(1.0, height), height / max(1.0, width))


def _connected_component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    count = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
    return count


def _weights(phase11: dict[str, Any]) -> dict[str, float]:
    configured = phase11.get("weights", {}) if isinstance(phase11.get("weights"), dict) else {}
    weights = {
        "feature_alignment": float(configured.get("feature_alignment", 0.32)),
        "mask_coverage": float(configured.get("mask_coverage", 0.22)),
        "texture_preservation": float(configured.get("texture_preservation", 0.24)),
        "leakage": float(configured.get("leakage", 0.14)),
        "morphology_fit": float(configured.get("morphology_fit", 0.08)),
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total > 0 else weights


def _phase11_config(config: AppConfig) -> dict[str, Any]:
    return dict(config.data.get("phase11", {}))


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "category",
        "defect_type",
        "variant",
        "quality_profile",
        "morphology",
        "generation_seed",
        "tfidg_lite_score",
        "feature_alignment_score",
        "adaptive_mask_coverage_score",
        "texture_preservation_score",
        "leakage_score",
        "morphology_fit_score",
        "phase4_generation_quality_score",
        "phase4_defect_visibility_score",
        "phase4_background_l1",
        "phase4_outside_refined_change_fraction",
        "accepted",
        "reject_reasons",
        "output_path",
        "background_path",
        "refined_mask_path",
        "inpaint_mask_path",
        "reference_path",
        "schema_version",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _write_summary(
    path: Path,
    records: list[dict[str, Any]],
    contact_sheet: Path,
    csv_path: Path,
    jsonl_path: Path,
    phase11: dict[str, Any],
) -> None:
    accepted = [row for row in records if row["accepted"]]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        by_group.setdefault((str(row["variant"]), str(row["quality_profile"])), []).append(row)
    group_rows = sorted(
        (
            (variant, profile, len(rows), mean(float(row["tfidg_lite_score"]) for row in rows), sum(1 for row in rows if row["accepted"]))
            for (variant, profile), rows in by_group.items()
        ),
        key=lambda item: item[3],
        reverse=True,
    )
    best = sorted(records, key=lambda row: float(row["tfidg_lite_score"]), reverse=True)[:8]
    worst = sorted(records, key=lambda row: float(row["tfidg_lite_score"]))[:8]
    reason_counts: dict[str, int] = {}
    for row in records:
        for reason in str(row["reject_reasons"]).split(","):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines = [
        "# Phase 11 TF-IDG-Lite Generation Critic",
        "",
        "This report is inspired by TF-IDG's training-free philosophy: generated defects are ranked without training another model.",
        "The local critic scores visual reference alignment, adaptive mask coverage, outside-mask texture preservation, leakage, and morphology fit.",
        "",
        "These are automatic research diagnostics, not human quality labels.",
        "",
        "## Summary",
        "",
        f"- Records scored: `{len(records)}`",
        f"- Accepted records: `{len(accepted)}`",
        f"- Mean TF-IDG-lite score: `{mean(float(row['tfidg_lite_score']) for row in records) if records else 0.0:.4f}`",
        f"- Contact sheet: `{contact_sheet}`",
        f"- CSV: `{csv_path}`",
        f"- JSONL: `{jsonl_path}`",
        "",
        "## Reject Reasons",
        "",
    ]
    if reason_counts:
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Mean Score By Variant/Profile", "", "| Variant | Profile | Count | Accepted | Mean Score |", "| --- | --- | ---: | ---: | ---: |"])
    for variant, profile, count, avg, accepted_count in group_rows[:30]:
        lines.append(f"| `{variant}` | `{profile}` | {count} | {accepted_count} | `{avg:.4f}` |")
    lines.extend(["", "## Best Samples", "", "| Score | Variant | Profile | Category | Output | Reject Reasons |", "| ---: | --- | --- | --- | --- | --- |"])
    for row in best:
        lines.append(
            f"| `{float(row['tfidg_lite_score']):.4f}` | `{row['variant']}` | `{row['quality_profile']}` | "
            f"`{row['category']}/{row['defect_type']}` | `{Path(str(row['output_path'])).name}` | `{row['reject_reasons']}` |"
        )
    lines.extend(["", "## Weakest Samples", "", "| Score | Variant | Profile | Category | Output | Reject Reasons |", "| ---: | --- | --- | --- | --- | --- |"])
    for row in worst:
        lines.append(
            f"| `{float(row['tfidg_lite_score']):.4f}` | `{row['variant']}` | `{row['quality_profile']}` | "
            f"`{row['category']}/{row['defect_type']}` | `{Path(str(row['output_path'])).name}` | `{row['reject_reasons']}` |"
        )
    lines.extend(
        [
            "",
            "## Thresholds",
            "",
            "```json",
            json.dumps(
                {
                    "min_feature_alignment_score": phase11.get("min_feature_alignment_score", 0.28),
                    "min_mask_coverage_score": phase11.get("min_mask_coverage_score", 0.35),
                    "min_texture_preservation_score": phase11.get("min_texture_preservation_score", 0.55),
                    "min_leakage_score": phase11.get("min_leakage_score", 0.45),
                    "min_morphology_fit_score": phase11.get("min_morphology_fit_score", 0.20),
                    "weights": _weights(phase11),
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_contact_sheet(path: Path, records: list[dict[str, Any]], phase11: dict[str, Any]) -> None:
    limit = int(phase11.get("contact_sheet_rows", 18))
    best = sorted(records, key=lambda row: float(row["tfidg_lite_score"]), reverse=True)[: max(0, limit // 2)]
    weak = sorted(records, key=lambda row: float(row["tfidg_lite_score"]))[: max(0, limit - len(best))]
    rows = best + weak
    if not rows:
        Image.new("RGB", (640, 160), "white").save(path)
        return
    thumb_w, thumb_h, label_h, header_h = 180, 180, 54, 42
    columns = ["reference", "background", "generated", "mask", "diff"]
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (thumb_w * len(columns), header_h + len(rows) * (label_h + thumb_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, title in enumerate(columns):
        x = col * thumb_w
        draw.rectangle((x, 0, x + thumb_w, header_h), fill=(235, 235, 235), outline=(210, 210, 210))
        draw.text((x + 6, 12), title, fill="black", font=font)
    for row_index, row in enumerate(rows):
        y = header_h + row_index * (label_h + thumb_h)
        generated = Image.open(row["output_path"]).convert("RGB")
        background = Image.open(row["background_path"]).convert("RGB").resize(generated.size, Image.Resampling.BILINEAR)
        reference = Image.open(row["reference_path"]).convert("RGB") if row["reference_path"] else Image.new("RGB", generated.size, "white")
        mask = _load_mask(Path(str(row["refined_mask_path"])), generated.size)
        tiles = [
            reference,
            background,
            generated,
            _overlay(generated, mask, color=(255, 0, 0)),
            _diff_heatmap(generated, background, mask),
        ]
        label = (
            f"{row['category']}/{row['defect_type']} {row['variant']} {row['quality_profile']}\n"
            f"score={float(row['tfidg_lite_score']):.3f} align={float(row['feature_alignment_score']):.2f} "
            f"cov={float(row['adaptive_mask_coverage_score']):.2f} tex={float(row['texture_preservation_score']):.2f}"
        )
        for col, tile in enumerate(tiles):
            x = col * thumb_w
            draw.rectangle((x, y, x + thumb_w, y + label_h + thumb_h), fill="white", outline=(225, 225, 225))
            if col == 0:
                draw.text((x + 5, y + 4), label[:86], fill="black", font=font)
            tile = _thumb(tile, (thumb_w, thumb_h))
            sheet.paste(tile, (x, y + label_h))
    sheet.save(path)


def _thumb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = np.clip(mask, 0.0, 1.0)[:, :, None] * 0.45
    color_arr = np.asarray(color, dtype=np.float32)[None, None, :]
    blended = arr * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def _diff_heatmap(generated: Image.Image, background: Image.Image, mask: np.ndarray) -> Image.Image:
    diff = _image_diff(generated, background)
    norm = _normalize(diff)
    heat = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
    heat[:, :, 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    heat[:, :, 1] = np.clip((1.0 - np.abs(norm - 0.5) * 2.0) * 180, 0, 180).astype(np.uint8)
    heat[:, :, 2] = np.clip((1.0 - norm) * 120, 0, 120).astype(np.uint8)
    outline = mask > 0.05
    heat[outline, 1] = np.maximum(heat[outline, 1], 180)
    return Image.fromarray(heat, mode="RGB")


def _normalize(values: np.ndarray) -> np.ndarray:
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 99.0))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
