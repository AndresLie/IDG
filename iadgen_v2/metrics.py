from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image

from iadgen_v2.config import AppConfig
from iadgen_v2.records import read_records


def evaluate_generation(config: AppConfig, model: str) -> Path:
    metadata = read_records(config.output_dir / "generated" / model / "metadata.jsonl")
    if not metadata:
        raise FileNotFoundError(f"No generated metadata found for model {model}; run generation first.")
    expected_fingerprint = config.generation_fingerprint(model)
    mismatched = [
        row for row in metadata if row.settings.get("generation_fingerprint") != expected_fingerprint
    ]
    if mismatched:
        raise ValueError(
            "Generated metadata does not match the active generation configuration; "
            "regenerate outputs before evaluation."
        )
    successful = [row for row in metadata if row.error is None and Path(row.output_path).exists()]
    if len(successful) != len(metadata):
        raise RuntimeError(
            f"Cannot evaluate an incomplete {model} run: {len(successful)} of {len(metadata)} required outputs exist."
        )
    rows: list[dict[str, str | float | int]] = []
    visibility_threshold = float(config.data["evaluation"].get("visibility_threshold", 0.0))
    for row in successful:
        output_image = Image.open(row.output_path).convert("RGB")
        output_size = output_image.size
        background = np.asarray(
            Image.open(row.background_path).convert("RGB").resize(output_size, Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        output = np.asarray(output_image, dtype=np.float32)
        binary_mask = (
            np.asarray(Image.open(row.insertion_mask_path).convert("L").resize(output_size, Image.Resampling.NEAREST), dtype=np.uint8)
            > 0
        )
        inpaint_mask_path = row.settings.get("inpaint_mask_path")
        if not inpaint_mask_path:
            raise ValueError(f"Missing inpaint_mask_path in generated metadata for {row.output_path}")
        inpaint_mask = (
            np.asarray(Image.open(inpaint_mask_path).convert("L").resize(output_size, Image.Resampling.LANCZOS), dtype=np.uint8)
            > 0
        )
        visibility = masked_visibility_l1(background, output, binary_mask)
        peak_memory = row.settings.get("peak_cuda_memory_bytes")
        rows.append(
            {
                "category": row.category,
                "defect_type": row.defect_type,
                "model": model,
                "outside_binary_background_l1": background_preservation_l1(background, output, binary_mask),
                "outside_binary_leakage": mask_leakage(background, output, binary_mask),
                "outside_inpaint_background_l1": background_preservation_l1(background, output, inpaint_mask),
                "outside_inpaint_leakage": mask_leakage(background, output, inpaint_mask),
                "masked_visibility_l1": visibility,
                "visibility_accepted": visibility >= visibility_threshold,
                "latency_sec": row.latency_sec if row.latency_sec is not None else math.nan,
                "peak_cuda_memory_gib": float(peak_memory) / (1024**3) if peak_memory is not None else math.nan,
            }
        )
    output_dir = config.report_dir / model
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rendering_metrics.csv"
    _write_csv(csv_path, rows)
    _write_summary(output_dir / "summary.md", model, metadata, rows, config)
    return csv_path


def background_preservation_l1(background: np.ndarray, output: np.ndarray, mask: np.ndarray) -> float:
    outside = ~mask
    return float(np.abs(background[outside] - output[outside]).mean() / 255.0) if outside.any() else math.nan


def masked_visibility_l1(background: np.ndarray, output: np.ndarray, mask: np.ndarray) -> float:
    return float(np.abs(background[mask] - output[mask]).mean() / 255.0) if mask.any() else math.nan


def mask_leakage(background: np.ndarray, output: np.ndarray, mask: np.ndarray, threshold: float = 12.0) -> float:
    outside = ~mask
    if not outside.any():
        return math.nan
    pixel_change = np.abs(background - output).mean(axis=2)
    return float((pixel_change[outside] > threshold).mean())


def _write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    fields = [
        "category",
        "defect_type",
        "model",
        "outside_binary_background_l1",
        "outside_binary_leakage",
        "outside_inpaint_background_l1",
        "outside_inpaint_leakage",
        "masked_visibility_l1",
        "visibility_accepted",
        "latency_sec",
        "peak_cuda_memory_gib",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    model: str,
    records: list[object],
    rows: list[dict[str, str | float | int]],
    config: AppConfig,
) -> None:
    failed = len(records) - len(rows)
    sample_count = int(config.data["generation"].get("samples_per_category", 1))
    run_type = "smoke validation" if sample_count == 1 else "baseline batch"
    lines = [
        f"# Phase 1 Rendering Report: {model}",
        "",
        f"Run type: {run_type}",
        f"Samples per category: {sample_count}",
        f"Rendered outputs: {len(rows)}",
        f"Rendering failures: {failed}",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| Metric | Mean |",
                "| --- | ---: |",
                f"| Background L1 outside binary label | {mean(float(row['outside_binary_background_l1']) for row in rows):.4f} |",
                f"| Leakage outside binary label | {mean(float(row['outside_binary_leakage']) for row in rows):.4f} |",
                f"| Background L1 outside inpaint support | {mean(float(row['outside_inpaint_background_l1']) for row in rows):.4f} |",
                f"| Leakage outside inpaint support | {mean(float(row['outside_inpaint_leakage']) for row in rows):.4f} |",
                f"| Masked visibility L1 | {mean(float(row['masked_visibility_l1']) for row in rows):.4f} |",
                f"| Visibility accepted | {sum(bool(row['visibility_accepted']) for row in rows)} / {len(rows)} |",
                f"| Mean latency (sec) | {mean(float(row['latency_sec']) for row in rows):.3f} |",
            ]
        )
        measured_memory = [float(row["peak_cuda_memory_gib"]) for row in rows if not math.isnan(float(row["peak_cuda_memory_gib"]))]
        if measured_memory:
            lines.append(f"| Peak CUDA memory (GiB) | {max(measured_memory):.3f} |")
    else:
        lines.append("No successful generated images were available for measurement.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
