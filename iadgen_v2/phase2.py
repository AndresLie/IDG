from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.dataset import load_manifest
from iadgen_v2.masks import surface_map, write_mask_overlay, write_refined_bbox_masks
from iadgen_v2.qwen_provider import QwenFeatureExtractor, parse_normalized_box, qwen_availability, save_qwen_cache, write_availability
from iadgen_v2.records import write_json


PHASE2_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Phase2Record:
    category: str
    defect_type: str
    background_path: str
    provider: str
    prompt: str
    region_xyxy: tuple[int, int, int, int]
    normalized_region_xyxy: tuple[float, float, float, float]
    region_surface_coverage: float
    box_mask_path: str
    refined_mask_path: str
    inpaint_mask_path: str
    overlay_path: str
    feature_cache_path: str
    seed: int
    settings: dict[str, Any]


def run_phase2_proposals(config: AppConfig, provider: str | None = None) -> Path:
    manifest = load_manifest(config)
    phase2 = _phase2_config(config)
    provider = provider or str(phase2.get("provider", "heuristic"))
    if provider not in {"heuristic", "qwen"}:
        raise ValueError(f"Unsupported Phase 2 provider: {provider}")
    output_dir = config.output_dir / "phase2" / provider
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.write_text("", encoding="utf-8")
    samples_per_category = int(phase2.get("samples_per_category", config.data["generation"].get("samples_per_category", 1)))
    base_seed = int(config.data["generation"].get("seed", 1337))
    run_fingerprint = _phase2_fingerprint(config, provider)
    qwen_status = _qwen_status(phase2)
    write_json(
        output_dir / "provider_status.json",
        {
            "provider": provider,
            "qwen_status": qwen_status,
            "phase2_fingerprint": run_fingerprint,
            "note": _provider_note(provider),
        },
    )
    if provider == "qwen":
        availability = _qwen_availability(phase2)
        write_availability(output_dir / "qwen_availability.json", availability)
        if not availability.ready:
            raise RuntimeError(f"Qwen provider is not ready: {availability.failure_reason()}")
        extractor: QwenFeatureExtractor | None = QwenFeatureExtractor(
            model_id=str(phase2.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")),
            cache_dir=phase2.get("qwen_cache_dir"),
            local_files_only=bool(phase2.get("qwen_local_files_only", True)),
            device=str(phase2.get("qwen_device", phase2.get("device", "auto"))),
            torch_dtype=str(phase2.get("qwen_dtype", "auto")),
            token_count=int(phase2.get("token_count", 16)),
            min_free_gib=float(phase2.get("qwen_min_free_gib", 30.0)),
        )
    else:
        extractor = None
    rows: list[Phase2Record] = []
    try:
        for target_key, target_data in manifest["targets"].items():
            category = str(target_data["category"])
            defect_type = str(target_data["defect_type"])
            backgrounds = list(target_data["clean_targets"])
            for index in range(samples_per_category):
                seed = base_seed + _stable_seed(f"phase2:{target_key}:{index}")
                background_path = Path(backgrounds[index % len(backgrounds)])
                image = Image.open(background_path).convert("RGB")
                prompt = _prompt(category, defect_type)
                qwen_result: dict[str, Any] | None = None
                if extractor is None:
                    region, coverage = _propose_region(image, category, defect_type, seed, phase2)
                    region_metadata = None
                else:
                    qwen_result = extractor.analyze(image, prompt)
                    region, coverage, region_metadata = _qwen_region(image, category, defect_type, qwen_result["text"], phase2)
                masks = _write_masks(image, category, defect_type, region, output_dir / "masks" / category, f"{category}_{defect_type}_{index:04d}", seed)
                overlay_path = output_dir / "overlays" / category / f"{category}_{defect_type}_{index:04d}.png"
                _write_overlay(image, region, Path(masks["box_mask_path"]), Path(masks["refined_mask_path"]), overlay_path)
                feature_path = output_dir / "feature_cache" / category / f"{category}_{defect_type}_{index:04d}.pt"
                if qwen_result is None:
                    _write_feature_cache(image, region, prompt, feature_path, seed, phase2)
                    qwen_text = None
                else:
                    qwen_text = str(qwen_result["text"])
                    save_qwen_cache(
                        feature_path,
                        tokens=qwen_result["tokens"],
                        metadata={
                            "provider": "qwen",
                            "model_id": str(phase2.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")),
                            "prompt": prompt,
                            "qwen_text": qwen_text,
                            "region_xyxy": region,
                            "normalized_region_xyxy": _normalize_region(region, image.size),
                            "tensor_shape": list(qwen_result["tokens"].shape),
                            "hidden_width": qwen_result["hidden_width"],
                            "selected_token_indices": qwen_result["selected_token_indices"],
                            "input_token_count": qwen_result["input_token_count"],
                        },
                    )
                record = Phase2Record(
                    category=category,
                    defect_type=defect_type,
                    background_path=str(background_path),
                    provider=provider,
                    prompt=prompt,
                    region_xyxy=region,
                    normalized_region_xyxy=_normalize_region(region, image.size),
                    region_surface_coverage=coverage,
                    box_mask_path=masks["box_mask_path"],
                    refined_mask_path=masks["refined_mask_path"],
                    inpaint_mask_path=masks["inpaint_mask_path"],
                    overlay_path=str(overlay_path),
                    feature_cache_path=str(feature_path),
                    seed=seed,
                    settings={
                        "phase2_fingerprint": run_fingerprint,
                        "phase2_schema_version": PHASE2_SCHEMA_VERSION,
                        "split_spec_fingerprint": manifest["split_spec_fingerprint"],
                        "mask_seed": seed,
                        "mask_parameters": masks["parameters"],
                        "qwen_prompt_template": _prompt_template(),
                        "qwen_text": qwen_text,
                        "qwen_region_metadata": region_metadata,
                    },
                )
                rows.append(record)
    finally:
        if extractor is not None:
            extractor.close()
    with metadata_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    return metadata_path


def evaluate_phase2_placement(config: AppConfig, provider: str | None = None) -> Path:
    phase2 = _phase2_config(config)
    provider = provider or str(phase2.get("provider", "heuristic"))
    output_dir = config.output_dir / "phase2" / provider
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing Phase 2 metadata: {metadata_path}")
    expected = _phase2_fingerprint(config, provider)
    raw_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("settings", {}).get("phase2_schema_version") != PHASE2_SCHEMA_VERSION for row in raw_rows):
        raise ValueError("Phase 2 metadata schema is stale; regenerate proposals first.")
    if any(row.get("settings", {}).get("phase2_fingerprint") != expected for row in raw_rows):
        raise ValueError("Phase 2 metadata does not match the active configuration; regenerate proposals first.")
    rows = [Phase2Record(**row) for row in raw_rows]
    metrics = [_metrics_for_record(row) for row in rows]
    report_dir = config.report_dir / "phase2" / provider
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "placement_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "defect_type",
                "region_surface_coverage",
                "box_area_fraction",
                "refined_area_fraction",
                "refined_to_box_area",
                "refined_surface_coverage",
                "refined_off_surface_fraction",
                "inpaint_to_refined_area",
                "inpaint_off_surface_fraction",
                "boundary_clearance",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)
    _write_summary(report_dir / "summary.md", provider, metrics, output_dir / "provider_status.json")
    return csv_path


def _phase2_config(config: AppConfig) -> dict[str, Any]:
    return dict(config.data.get("phase2", {}))


def _phase2_fingerprint(config: AppConfig, provider: str) -> str:
    return fingerprint(
        {
            "split": config.split_spec_fingerprint(),
            "phase2": config.data.get("phase2", {}),
            "provider": provider,
            "schema_version": PHASE2_SCHEMA_VERSION,
        }
    )


def _prompt_template() -> str:
    return (
        "Analyze the clean industrial image. For a requested {defect_type}, "
        "choose one plausible surface-valid target region. Respond with JSON only: "
        '{{"box":[x1,y1,x2,y2],"confidence":0.0,"rationale":"short reason"}}. '
        "Coordinates must be normalized floats from 0 to 1, x2>x1, y2>y1. "
        "Do not produce a pixel mask."
    )


def _prompt(category: str, defect_type: str) -> str:
    return _prompt_template().format(defect_type=defect_type) + f" Category: {category}."


def _qwen_status(phase2: dict[str, Any]) -> dict[str, object]:
    availability = _qwen_availability(phase2)
    return availability.__dict__ | {"ready": availability.ready, "failure_reason": availability.failure_reason()}


def _qwen_availability(phase2: dict[str, Any]):
    return qwen_availability(
        model_id=str(phase2.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")),
        cache_dir=phase2.get("qwen_cache_dir"),
        local_files_only=bool(phase2.get("qwen_local_files_only", True)),
        min_free_gib=float(phase2.get("qwen_min_free_gib", 30.0)),
    )


def _provider_note(provider: str) -> str:
    if provider == "qwen":
        return "qwen provider writes Qwen-selected regions and Qwen hidden-state token caches"
    return "heuristic provider writes Phase 2 artifacts without running Qwen"


def _propose_region(
    image: Image.Image,
    category: str,
    defect_type: str,
    seed: int,
    phase2: dict[str, Any],
) -> tuple[tuple[int, int, int, int], float]:
    rng = random.Random(seed)
    surface = surface_map(image, category)
    area_fraction = float(phase2.get("box_area_fraction", {}).get(defect_type, 0.08))
    target_area = max(64, int(image.width * image.height * area_fraction))
    aspect = rng.uniform(2.5, 6.0) if defect_type == "scratch" else rng.uniform(1.6, 4.0)
    box_w = max(12, min(image.width, int(math.sqrt(target_area * aspect))))
    box_h = max(8, min(image.height, int(math.sqrt(target_area / aspect))))
    best: tuple[float, int, int] | None = None
    for _ in range(768):
        left = rng.randint(0, max(0, image.width - box_w))
        top = rng.randint(0, max(0, image.height - box_h))
        coverage = float(surface[top : top + box_h, left : left + box_w].mean())
        if best is None or coverage > best[0]:
            best = (coverage, left, top)
    if best is None:
        raise RuntimeError("No region candidates could be sampled")
    coverage, left, top = best
    if coverage < float(phase2.get("min_surface_coverage", 0.90)):
        raise RuntimeError(f"No surface-valid region found for {category}/{defect_type}: coverage={coverage:.3f}")
    return (left, top, left + box_w, top + box_h), coverage


def _qwen_region(
    image: Image.Image,
    category: str,
    defect_type: str,
    text: str,
    phase2: dict[str, Any],
) -> tuple[tuple[int, int, int, int], float, dict[str, Any]]:
    normalized = parse_normalized_box(text)
    if normalized is None:
        raise RuntimeError(f"Qwen did not return a parseable normalized box: {text!r}")
    x1, y1, x2, y2 = normalized
    region = (
        max(0, min(image.width - 1, round(x1 * image.width))),
        max(0, min(image.height - 1, round(y1 * image.height))),
        max(1, min(image.width, round(x2 * image.width))),
        max(1, min(image.height, round(y2 * image.height))),
    )
    if region[2] <= region[0] or region[3] <= region[1]:
        raise RuntimeError(f"Qwen returned an empty region after scaling: {text!r}")
    original_region = region
    original_coverage = _surface_coverage(image, category, original_region)
    region, size_metadata = _normalize_qwen_region_size(image, defect_type, region, phase2)
    coverage = _surface_coverage(image, category, region)
    min_coverage = float(phase2.get("min_surface_coverage", 0.90))
    metadata: dict[str, Any] = {
        "raw_normalized_region_xyxy": normalized,
        "raw_region_xyxy": original_region,
        "size_normalized_region_xyxy": region,
        "raw_region_surface_coverage": original_coverage,
        "surface_repaired": False,
        **size_metadata,
    }
    if coverage < min_coverage and bool(phase2.get("qwen_repair_low_coverage", True)):
        repaired, repaired_coverage = _repair_region_to_surface(image, category, region)
        if repaired_coverage >= coverage:
            metadata.update(
                {
                    "surface_repaired": repaired != region,
                    "repaired_region_xyxy": repaired,
                    "repaired_region_surface_coverage": repaired_coverage,
                }
            )
            region = repaired
            coverage = repaired_coverage
    if coverage < min_coverage:
        raise RuntimeError(f"Qwen region is not surface-valid for {category}: coverage={coverage:.3f}")
    return region, coverage, metadata


def _normalize_qwen_region_size(
    image: Image.Image,
    defect_type: str,
    region: tuple[int, int, int, int],
    phase2: dict[str, Any],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    left, top, right, bottom = region
    width = max(1, right - left)
    height = max(1, bottom - top)
    image_area = image.width * image.height
    area_fraction = (width * height) / max(1, image_area)
    target_fraction = float(phase2.get("box_area_fraction", {}).get(defect_type, 0.08))
    max_fraction = target_fraction * float(phase2.get("qwen_max_area_multiplier", 1.5))
    if area_fraction <= max_fraction:
        return region, {
            "size_normalized": False,
            "raw_area_fraction": area_fraction,
            "max_area_fraction": max_fraction,
        }
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    aspect = width / max(1, height)
    if defect_type == "scratch":
        aspect = max(3.0, min(6.0, aspect))
    elif defect_type == "crack":
        aspect = max(1.5, min(4.0, aspect))
    target_area = max(64, round(image_area * target_fraction))
    new_width = max(8, min(image.width, round(math.sqrt(target_area * aspect))))
    new_height = max(8, min(image.height, round(math.sqrt(target_area / aspect))))
    resized = _box_from_center(center_x, center_y, new_width, new_height, image.size)
    return resized, {
        "size_normalized": True,
        "raw_area_fraction": area_fraction,
        "max_area_fraction": max_fraction,
        "target_area_fraction": target_fraction,
        "size_normalized_aspect": aspect,
    }


def _repair_region_to_surface(
    image: Image.Image,
    category: str,
    region: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], float]:
    surface = surface_map(image, category)
    left, top, right, bottom = region
    width = max(8, right - left)
    height = max(8, bottom - top)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    best_region = region
    best_coverage = _surface_coverage(image, category, region)
    max_shift = max(width, height)
    step = max(4, min(width, height) // 4)
    for scale in (1.0, 0.85, 0.70, 0.55):
        candidate_w = max(8, min(image.width, round(width * scale)))
        candidate_h = max(8, min(image.height, round(height * scale)))
        for dy in range(-max_shift, max_shift + 1, step):
            for dx in range(-max_shift, max_shift + 1, step):
                cx = center_x + dx
                cy = center_y + dy
                candidate = _box_from_center(cx, cy, candidate_w, candidate_h, image.size)
                cand_left, cand_top, cand_right, cand_bottom = candidate
                if cand_right <= cand_left or cand_bottom <= cand_top:
                    continue
                coverage = float(surface[cand_top:cand_bottom, cand_left:cand_right].mean())
                if coverage > best_coverage:
                    best_region = candidate
                    best_coverage = coverage
    return best_region, best_coverage


def _box_from_center(
    center_x: int,
    center_y: int,
    width: int,
    height: int,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    left = max(0, min(image_width - width, round(center_x - width / 2)))
    top = max(0, min(image_height - height, round(center_y - height / 2)))
    return (left, top, left + width, top + height)


def _surface_coverage(image: Image.Image, category: str, region: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = region
    surface = surface_map(image, category)
    return float(surface[top:bottom, left:right].mean())


def _normalize_region(
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    left, top, right, bottom = region
    return (left / width, top / height, right / width, bottom / height)


def _write_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    seed: int,
) -> dict[str, Any]:
    return write_refined_bbox_masks(image, category, defect_type, region, output_dir, stem, seed)


def _refined_mask(
    image_size: tuple[int, int],
    defect_type: str,
    region: tuple[int, int, int, int],
    seed: int,
) -> tuple[Image.Image, dict[str, Any]]:
    rng = random.Random(seed)
    left, top, right, bottom = region
    width = max(1, right - left)
    height = max(1, bottom - top)
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    if defect_type == "crack":
        points = _random_walk_points(left, top, width, height, rng, branches=False)
        branch_count = rng.randint(1, 3)
        line_width = max(2, min(width, height) // rng.randint(14, 22))
        draw.line(points, fill=255, width=line_width, joint="curve")
        for _ in range(branch_count):
            anchor = points[rng.randrange(len(points))]
            branch = _branch_points(anchor, left, top, width, height, rng)
            draw.line(branch, fill=255, width=max(1, line_width // 2))
        params = {"kind": "jagged_random_walk", "line_width": line_width, "branches": branch_count}
    else:
        points = _bezier_like_points(left, top, width, height, rng)
        line_width = max(2, min(width, height) // rng.randint(10, 18))
        draw.line(points, fill=255, width=line_width, joint="curve")
        params = {"kind": "curved_polyline", "line_width": line_width, "points": len(points)}
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35)).point(lambda value: 255 if value > 48 else 0)
    return mask, params


def _clip_mask_to_surface(
    mask: Image.Image,
    image: Image.Image,
    category: str,
    region: tuple[int, int, int, int],
) -> tuple[Image.Image, dict[str, Any]]:
    surface = surface_map(image, category)
    mask_arr = np.asarray(mask, dtype=np.uint8) > 0
    clipped = mask_arr & surface
    left, top, right, bottom = region
    fallback_used = False
    if not clipped.any():
        surface_region = surface[top:bottom, left:right]
        if not surface_region.any():
            raise RuntimeError(f"Refined mask for {category} has no valid surface pixels inside proposed region")
        ys, xs = np.where(surface_region)
        cy = top + int(np.median(ys))
        cx = left + int(np.median(xs))
        length = max(4, min(right - left, bottom - top) // 4)
        fallback = Image.new("L", image.size, 0)
        ImageDraw.Draw(fallback).line((cx - length, cy, cx + length, cy), fill=255, width=2)
        clipped = (np.asarray(fallback, dtype=np.uint8) > 0) & surface
        fallback_used = True
    clipped_image = Image.fromarray(clipped.astype(np.uint8) * 255).convert("L")
    return clipped_image, {"surface_clipped": True, "surface_clip_fallback": fallback_used}


def _clip_soft_mask_to_surface(mask: Image.Image, image: Image.Image, category: str) -> Image.Image:
    surface = surface_map(image, category)
    mask_arr = np.array(mask, dtype=np.uint8, copy=True)
    mask_arr[~surface] = 0
    return Image.fromarray(mask_arr).convert("L")


def _bezier_like_points(left: int, top: int, width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
    y = top + rng.randint(max(0, height // 4), max(1, height * 3 // 4))
    points = []
    for step in range(7):
        frac = step / 6
        x = left + int(frac * (width - 1))
        offset = int(math.sin(frac * math.pi * rng.uniform(0.8, 1.5)) * rng.uniform(-height * 0.18, height * 0.18))
        points.append((x, max(top, min(top + height - 1, y + offset + rng.randint(-height // 10, max(1, height // 10))))))
    return points


def _random_walk_points(left: int, top: int, width: int, height: int, rng: random.Random, branches: bool) -> list[tuple[int, int]]:
    y = top + rng.randint(height // 4, max(height // 4, height * 3 // 4))
    points = []
    for step in range(9):
        frac = step / 8
        x = left + int(frac * (width - 1))
        y = max(top, min(top + height - 1, y + rng.randint(-max(1, height // 8), max(1, height // 8))))
        points.append((x, y))
    return points


def _branch_points(anchor: tuple[int, int], left: int, top: int, width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
    angle = rng.uniform(-1.2, 1.2)
    length = rng.uniform(0.12, 0.30) * width
    end = (
        max(left, min(left + width - 1, int(anchor[0] + math.cos(angle) * length))),
        max(top, min(top + height - 1, int(anchor[1] + math.sin(angle) * length))),
    )
    return [anchor, end]


def _write_overlay(
    image: Image.Image,
    region: tuple[int, int, int, int],
    box_mask_path: Path,
    refined_mask_path: Path,
    output_path: Path,
) -> None:
    write_mask_overlay(image, region, refined_mask_path, output_path)


def _write_feature_cache(
    image: Image.Image,
    region: tuple[int, int, int, int],
    prompt: str,
    output_path: Path,
    seed: int,
    phase2: dict[str, Any],
) -> None:
    import torch

    rng = np.random.default_rng(seed)
    token_count = int(phase2.get("token_count", 16))
    feature_dim = int(phase2.get("feature_dim", 64))
    left, top, right, bottom = region
    patch = np.asarray(image.crop(region).resize((16, 16)).convert("RGB"), dtype=np.float32) / 255.0
    stats = np.concatenate(
        [
            patch.mean(axis=(0, 1)),
            patch.std(axis=(0, 1)),
            np.asarray([left / image.width, top / image.height, right / image.width, bottom / image.height], dtype=np.float32),
        ]
    )
    tokens = rng.normal(0.0, 0.01, size=(token_count, feature_dim)).astype(np.float32)
    for idx, value in enumerate(np.resize(stats, feature_dim)):
        tokens[:, idx] += value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": torch.from_numpy(tokens).unsqueeze(0),
            "metadata": {
                "provider": "heuristic",
                "prompt": prompt,
                "region_xyxy": region,
                "normalized_region_xyxy": _normalize_region(region, image.size),
                "tensor_shape": [1, token_count, feature_dim],
                "note": "placeholder cache for Phase 2 plumbing; not Qwen hidden states",
            },
        },
        output_path,
    )


def _metrics_for_record(record: Phase2Record) -> dict[str, object]:
    image = Image.open(record.background_path).convert("RGB")
    box = np.asarray(Image.open(record.box_mask_path).convert("L"), dtype=np.uint8) > 0
    refined = np.asarray(Image.open(record.refined_mask_path).convert("L"), dtype=np.uint8) > 0
    inpaint = np.asarray(Image.open(record.inpaint_mask_path).convert("L"), dtype=np.uint8) > 0
    surface = surface_map(image, record.category)
    box_area = int(box.sum())
    refined_area = int(refined.sum())
    refined_off_surface = int((refined & ~surface).sum())
    inpaint_area = int(inpaint.sum())
    inpaint_off_surface = int((inpaint & ~surface).sum())
    left, top, right, bottom = record.region_xyxy
    clearance = min(left, top, image.width - right, image.height - bottom) / max(1, min(image.size))
    return {
        "category": record.category,
        "defect_type": record.defect_type,
        "region_surface_coverage": record.region_surface_coverage,
        "box_area_fraction": box_area / (image.width * image.height),
        "refined_area_fraction": refined_area / (image.width * image.height),
        "refined_to_box_area": refined_area / box_area if box_area else math.nan,
        "refined_surface_coverage": float(surface[refined].mean()) if refined_area else math.nan,
        "refined_off_surface_fraction": refined_off_surface / refined_area if refined_area else math.nan,
        "inpaint_to_refined_area": inpaint_area / refined_area if refined_area else math.nan,
        "inpaint_off_surface_fraction": inpaint_off_surface / inpaint_area if inpaint_area else math.nan,
        "boundary_clearance": clearance,
    }


def _write_summary(path: Path, provider: str, metrics: list[dict[str, object]], provider_status_path: Path) -> None:
    lines = [
        f"# Phase 2 Placement Report: {provider}",
        "",
        f"Samples: {len(metrics)}",
        f"Provider status: `{provider_status_path}`",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
    ]
    for key in (
        "region_surface_coverage",
        "box_area_fraction",
        "refined_area_fraction",
        "refined_to_box_area",
        "refined_surface_coverage",
        "refined_off_surface_fraction",
        "inpaint_to_refined_area",
        "inpaint_off_surface_fraction",
        "boundary_clearance",
    ):
        values = [float(row[key]) for row in metrics]
        lines.append(f"| {key} | {mean(values):.4f} |")
    lines.append("")
    if provider == "qwen":
        lines.append("This report uses real Qwen placement proposals and Qwen hidden-state token caches.")
    else:
        lines.append("The heuristic provider validates Phase 2 artifact flow. It is not a Qwen result.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stable_seed(value: str) -> int:
    return int(fingerprint(value)[:8], 16)
