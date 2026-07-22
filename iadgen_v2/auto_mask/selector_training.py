from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from iadgen_v2.auto_mask.contracts import EvidenceMap
from iadgen_v2.auto_mask.evidence import fuse_evidence_maps, robust_probability
from iadgen_v2.auto_mask.proposals import generate_generic_proposals
from iadgen_v2.auto_mask.refinement import EdgeAwareRefiner, edge_align_field
from iadgen_v2.auto_mask.selection import fit_selector_bundle
from iadgen_v2.config import AppConfig


def train_generic_selector(config: AppConfig) -> Path:
    auto = config.data.get("auto_masks", {})
    settings = auto.get("selector_training", {}) if isinstance(auto, dict) else {}
    if not isinstance(settings, dict):
        raise ValueError("auto_masks.selector_training must be a mapping")
    seed = int(settings.get("seed", 20260722))
    max_normals = int(settings.get("max_normals_per_category", 12))
    corruptions_per_image = int(settings.get("corruptions_per_image", 6))
    generic = auto.get("generic_evidence", {}) if isinstance(auto, dict) else {}
    edge_refine = bool(generic.get("edge_refine", True)) if isinstance(generic, dict) else True
    rng = np.random.default_rng(seed)
    categories = sorted({target.category for target in config.targets})
    rows: list[dict[str, Any]] = []
    for category in categories:
        normal_dir = config.dataset_root / category / "train" / "good"
        paths = sorted(path for path in normal_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})[:max_normals]
        for path in paths:
            clean = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            for family_index in range(corruptions_per_image):
                family = CORRUPTION_FAMILIES[family_index % len(CORRUPTION_FAMILIES)]
                corrupted, truth = _corrupt(clean, family, rng)
                evidence = _synthetic_evidence(clean, corrupted, rng)
                fused, disagreement, _ = fuse_evidence_maps(evidence)
                # Include edge-refined candidates in training so the reliability
                # model learns to score them, instead of meeting them for the
                # first time at inference (which mis-ranks them).
                edge_refiner = None
                if edge_refine:
                    try:
                        edge_refiner = EdgeAwareRefiner(edge_align_field(corrupted, fused))
                    except Exception:
                        edge_refiner = None
                proposals = generate_generic_proposals(
                    fused,
                    disagreement,
                    evidence,
                    (0, 0, clean.width, clean.height),
                    foreground=np.ones((clean.height, clean.width), dtype=bool),
                    min_area=8,
                    edge_refiner=edge_refiner,
                )
                for proposal in proposals:
                    intersection = int((proposal.mask & truth).sum())
                    predicted = int(proposal.mask.sum())
                    actual = int(truth.sum())
                    union = predicted + actual - intersection
                    rows.append(
                        {
                            "category": category,
                            "corruption_family": family,
                            **proposal.measurements,
                            "iou": intersection / max(1, union),
                            "precision": intersection / max(1, predicted),
                            "recall": intersection / max(1, actual),
                        }
                    )
    output_dir = config.output_dir / "auto_masks" / "selector"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "synthetic_candidate_training_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    model_path = output_dir / "generic_selector.joblib"
    fit_selector_bundle(rows, model_path)
    return model_path


CORRUPTION_FAMILIES = (
    "cutpaste_patch",
    "cutpaste_scar",
    "perlin_texture",
    "local_appearance",
    "component_occlusion",
    "thin_crack",
)


def _corrupt(image: Image.Image, family: str, rng: np.random.Generator) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    output = image.copy()
    mask_image = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask_image)
    cx = int(rng.integers(width // 5, 4 * width // 5))
    cy = int(rng.integers(height // 5, 4 * height // 5))
    if family == "cutpaste_patch":
        w, h = int(rng.integers(24, 110)), int(rng.integers(24, 110))
        source_x, source_y = int(rng.integers(0, max(1, width - w))), int(rng.integers(0, max(1, height - h)))
        patch = image.crop((source_x, source_y, source_x + w, source_y + h)).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        output.paste(patch, (cx - w // 2, cy - h // 2))
        draw.rectangle((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), fill=255)
    elif family == "cutpaste_scar":
        length, thickness = int(rng.integers(50, 180)), int(rng.integers(3, 14))
        angle = float(rng.uniform(0, np.pi))
        dx, dy = int(np.cos(angle) * length / 2), int(np.sin(angle) * length / 2)
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=255, width=thickness)
        output = _apply_mask_color(output, mask_image, rng)
    elif family == "perlin_texture":
        noise = rng.normal(0, 1, (height // 16 + 1, width // 16 + 1)).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        threshold = float(np.quantile(noise, 0.82))
        blob = (noise >= threshold).astype(np.uint8) * 255
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        mask_image = Image.fromarray(blob, mode="L")
        output = _apply_mask_color(output, mask_image, rng)
    elif family == "local_appearance":
        radius = int(rng.integers(20, 70))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
        changed = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(3.0, 8.0))))
        output = Image.composite(changed, output, mask_image)
    elif family == "component_occlusion":
        w, h = int(rng.integers(20, 90)), int(rng.integers(20, 90))
        draw.rounded_rectangle((cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), radius=max(2, min(w, h) // 5), fill=255)
        output = _apply_mask_color(output, mask_image, rng, muted=True)
    elif family == "thin_crack":
        points = [(cx, cy)]
        for _ in range(int(rng.integers(3, 7))):
            last_x, last_y = points[-1]
            points.append((int(np.clip(last_x + rng.integers(-45, 46), 0, width - 1)), int(np.clip(last_y + rng.integers(-45, 46), 0, height - 1))))
        draw.line(points, fill=255, width=int(rng.integers(2, 7)), joint="curve")
        output = _apply_mask_color(output, mask_image, rng)
    else:
        raise ValueError(f"Unsupported corruption family: {family}")
    truth = np.asarray(mask_image, dtype=np.uint8) > 0
    return output, truth


def _apply_mask_color(image: Image.Image, mask: Image.Image, rng: np.random.Generator, *, muted: bool = False) -> Image.Image:
    if muted:
        color = tuple(int(value) for value in rng.integers(80, 176, size=3))
    else:
        color = tuple(int(value) for value in rng.integers(10, 246, size=3))
    layer = Image.new("RGB", image.size, color)
    alpha = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    return Image.composite(layer, image, alpha)


def _synthetic_evidence(clean: Image.Image, corrupted: Image.Image, rng: np.random.Generator) -> list[EvidenceMap]:
    clean_rgb = np.asarray(clean, dtype=np.float32) / 255.0
    changed_rgb = np.asarray(corrupted, dtype=np.float32) / 255.0
    color = np.linalg.norm(changed_rgb - clean_rgb, axis=2)
    clean_gray = cv2.cvtColor(np.uint8(clean_rgb * 255), cv2.COLOR_RGB2GRAY).astype(np.float32)
    changed_gray = cv2.cvtColor(np.uint8(changed_rgb * 255), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = np.abs(cv2.Laplacian(changed_gray, cv2.CV_32F) - cv2.Laplacian(clean_gray, cv2.CV_32F)) / 255.0
    blur = cv2.GaussianBlur(color, (0, 0), 5.0)
    maps = []
    for name, raw, reliability in (
        ("synthetic_color_residual", color, 0.82),
        ("synthetic_gradient_residual", gradient, 0.68),
        ("synthetic_context_residual", blur, 0.72),
    ):
        noisy = np.clip(raw + rng.normal(0, 0.015, raw.shape), 0.0, None)
        probability, calibration = robust_probability(noisy)
        maps.append(EvidenceMap(name, probability, reliability, calibration, augmentation_consistency=0.8))
    return maps
