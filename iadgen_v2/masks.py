from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def write_refined_bbox_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    seed: int,
    *,
    clip_to_surface: bool = True,
) -> dict[str, Any]:
    """Write rectangular debug, refined binary, and soft inpaint masks for a proposed bbox."""
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    refined, parameters = refined_defect_mask(image.size, defect_type, region, seed)
    if clip_to_surface:
        refined, clip_parameters = clip_mask_to_surface(refined, image, category, region)
        parameters.update(clip_parameters)
    else:
        parameters.update({"surface_clipped": False, "surface_clip_fallback": False})
    inpaint = refined.filter(ImageFilter.MaxFilter(size=9)).filter(ImageFilter.GaussianBlur(radius=2.0))
    if clip_to_surface:
        inpaint = clip_soft_mask_to_surface(inpaint, image, category)

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": parameters,
    }


def write_pixel_refined_bbox_masks(
    image: Image.Image,
    category: str,
    defect_type: str,
    region: tuple[int, int, int, int],
    output_dir: Path,
    stem: str,
    *,
    clip_to_surface: bool = True,
    text_hint: str = "",
    percentile: float = 93.0,
) -> dict[str, Any]:
    """Write masks by extracting high-contrast defect evidence inside a proposed bbox."""
    output_dir.mkdir(parents=True, exist_ok=True)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((region[0], region[1], region[2] - 1, region[3] - 1), fill=255)
    refined, parameters = pixel_refined_defect_mask(image, region, defect_type, text_hint=text_hint, percentile=percentile)
    if clip_to_surface:
        refined, clip_parameters = clip_mask_to_surface(refined, image, category, region)
        parameters.update(clip_parameters)
    else:
        parameters.update({"surface_clipped": False, "surface_clip_fallback": False})
    inpaint = refined.filter(ImageFilter.MaxFilter(size=7)).filter(ImageFilter.GaussianBlur(radius=1.8))
    if clip_to_surface:
        inpaint = clip_soft_mask_to_surface(inpaint, image, category)

    box_path = output_dir / f"{stem}_box.png"
    refined_path = output_dir / f"{stem}_refined.png"
    inpaint_path = output_dir / f"{stem}_inpaint.png"
    box.save(box_path)
    refined.save(refined_path)
    inpaint.save(inpaint_path)
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "inpaint_mask_path": str(inpaint_path),
        "parameters": parameters,
    }


def refined_defect_mask(
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
        points = _random_walk_points(left, top, width, height, rng)
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


def pixel_refined_defect_mask(
    image: Image.Image,
    region: tuple[int, int, int, int],
    defect_type: str,
    *,
    text_hint: str = "",
    percentile: float = 93.0,
) -> tuple[Image.Image, dict[str, Any]]:
    left, top, right, bottom = region
    crop = image.crop(region).convert("L")
    gray = np.asarray(crop, dtype=np.float32)
    if gray.size == 0:
        raise ValueError(f"Cannot refine an empty region: {region}")

    blurred = np.asarray(crop.filter(ImageFilter.GaussianBlur(radius=5.0)), dtype=np.float32)
    dark = np.clip(blurred - gray, 0, None)
    bright = np.clip(gray - blurred, 0, None)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    edge = np.maximum(gx, gy) * 0.45

    hint = text_hint.lower()
    if any(word in hint for word in ("white", "bright", "light", "silver", "pale")):
        contrast = bright + edge
        polarity = "bright"
    elif any(word in hint for word in ("black", "dark", "brown", "burn", "shadow")):
        contrast = dark + edge
        polarity = "dark"
    else:
        contrast = np.maximum(dark, bright) + edge
        polarity = "absolute"

    threshold = float(np.percentile(contrast, max(50.0, min(99.5, percentile))))
    active = contrast >= threshold
    if active.sum() < max(8, int(active.size * 0.002)):
        threshold = float(np.percentile(contrast, 88.0))
        active = contrast >= threshold
    mask_crop = Image.fromarray(active.astype(np.uint8) * 255, mode="L")
    mask_crop = mask_crop.filter(ImageFilter.MedianFilter(size=3))
    if defect_type in {"scratch", "crack"}:
        mask_crop = mask_crop.filter(ImageFilter.MaxFilter(size=3))
    mask_crop = mask_crop.point(lambda value: 255 if value > 0 else 0)

    mask = Image.new("L", image.size, 0)
    mask.paste(mask_crop, (left, top))
    if mask.getbbox() is None:
        fallback, fallback_params = refined_defect_mask(image.size, defect_type, region, seed=17)
        fallback_params.update({"kind": "pixel_contrast_fallback", "pixel_refine_failed": True})
        return fallback, fallback_params
    return mask, {
        "kind": "pixel_contrast",
        "polarity": polarity,
        "percentile": percentile,
        "threshold": threshold,
        "active_pixels": int((np.asarray(mask_crop, dtype=np.uint8) > 0).sum()),
    }


def clip_mask_to_surface(
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


def clip_soft_mask_to_surface(mask: Image.Image, image: Image.Image, category: str) -> Image.Image:
    surface = surface_map(image, category)
    mask_arr = np.array(mask, dtype=np.uint8, copy=True)
    mask_arr[~surface] = 0
    return Image.fromarray(mask_arr).convert("L")


def write_mask_overlay(
    image: Image.Image,
    region: tuple[int, int, int, int],
    refined_mask_path: Path,
    output_path: Path,
) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(region, outline=(255, 200, 0), width=max(2, image.width // 256))
    refined = Image.open(refined_mask_path).convert("L")
    color = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.blend(overlay, Image.composite(color, overlay, refined), 0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def place_adaptation_mask(
    background: Image.Image,
    source_mask_path: Path,
    output_dir: Path,
    output_stem: str,
    seed: int,
    *,
    category: str,
    placement_mode: str,
) -> tuple[Path, Path, tuple[int, int, int, int], float | None]:
    """Place adaptation morphology on a clean target for the text-only baseline."""
    source_mask = Image.open(source_mask_path).convert("L")
    crop_box = source_mask.getbbox()
    if crop_box is None:
        raise ValueError(f"Empty source anomaly mask: {source_mask_path}")
    crop = source_mask.crop(crop_box).point(lambda value: 255 if value > 0 else 0)
    crop.thumbnail((max(8, background.width // 3), max(8, background.height // 3)), Image.Resampling.NEAREST)

    rng = random.Random(seed)
    surface_coverage: float | None
    if placement_mode == "random_smoke":
        left, top = _random_location(background.size, crop.size, rng)
        surface_coverage = None
    elif placement_mode == "surface_constrained":
        left, top, surface_coverage = _surface_location(background, crop, category, rng)
    else:
        raise ValueError(f"Unsupported placement_mode: {placement_mode}")
    bbox = (left, top, left + crop.width, top + crop.height)

    binary = Image.new("L", background.size, 0)
    binary.paste(crop, (left, top))
    if binary.getbbox() is None:
        raise ValueError("Placed mask unexpectedly became empty")
    inpaint = binary.filter(ImageFilter.MaxFilter(size=9)).filter(ImageFilter.GaussianBlur(radius=2.0))

    output_dir.mkdir(parents=True, exist_ok=True)
    binary_path = output_dir / f"{output_stem}_binary.png"
    inpaint_path = output_dir / f"{output_stem}_inpaint.png"
    binary.save(binary_path)
    inpaint.save(inpaint_path)
    return binary_path, inpaint_path, bbox, surface_coverage


def _random_location(
    background_size: tuple[int, int], crop_size: tuple[int, int], rng: random.Random
) -> tuple[int, int]:
    return (
        rng.randint(0, max(0, background_size[0] - crop_size[0])),
        rng.randint(0, max(0, background_size[1] - crop_size[1])),
    )


def _surface_location(
    background: Image.Image, crop: Image.Image, category: str, rng: random.Random
) -> tuple[int, int, float]:
    surface = surface_map(background, category)
    defect = np.asarray(crop, dtype=np.uint8) > 0
    candidates: list[tuple[float, int, int]] = []
    for _ in range(512):
        left, top = _random_location(background.size, crop.size, rng)
        region = surface[top : top + crop.height, left : left + crop.width]
        coverage = float(region[defect].mean()) if defect.any() else 0.0
        candidates.append((coverage, left, top))
    coverage, left, top = max(candidates)
    if coverage < 0.90:
        raise ValueError(
            f"Could not place {category} mask on a sufficiently valid surface: coverage={coverage:.3f}"
        )
    return left, top, coverage


def _bezier_like_points(left: int, top: int, width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
    y = top + rng.randint(max(0, height // 4), max(1, height * 3 // 4))
    points = []
    for step in range(7):
        frac = step / 6
        x = left + int(frac * (width - 1))
        offset = int(math.sin(frac * math.pi * rng.uniform(0.8, 1.5)) * rng.uniform(-height * 0.18, height * 0.18))
        points.append((x, max(top, min(top + height - 1, y + offset + rng.randint(-height // 10, max(1, height // 10))))))
    return points


def _random_walk_points(left: int, top: int, width: int, height: int, rng: random.Random) -> list[tuple[int, int]]:
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


def surface_map(image: Image.Image, category: str) -> np.ndarray:
    width, height = image.size
    margin = max(2, min(width, height) // 50)
    if category in {"tile", "wood"}:
        surface = np.zeros((height, width), dtype=bool)
        surface[margin : height - margin, margin : width - margin] = True
        return surface
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    background = np.median(border, axis=0)
    surface = np.abs(arr - background).mean(axis=2) > 18.0
    surface[:margin] = False
    surface[-margin:] = False
    surface[:, :margin] = False
    surface[:, -margin:] = False
    return surface
