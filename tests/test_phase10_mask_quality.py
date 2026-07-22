from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.phase10_mask_quality import run_phase10_mask_quality_ablation


def test_phase10_mask_quality_ablation_writes_reports(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    output = tmp_path / "outputs" / "wood_quality"
    report = tmp_path / "reports" / "wood_quality"
    image_dir = dataset / "custom_part" / "test" / "scratch"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "000.png"
    image = Image.new("RGB", (96, 64), (128, 116, 92))
    ImageDraw.Draw(image).rectangle((24, 22, 54, 42), fill=(170, 150, 122))
    image.save(image_path)

    mask_dir = output / "auto_masks" / "qwen" / "masks"
    variant_dir = output / "auto_masks" / "qwen" / "variants"
    mask_dir.mkdir(parents=True)
    variant_dir.mkdir(parents=True)
    selected = Image.new("L", image.size, 0)
    ImageDraw.Draw(selected).rectangle((30, 28, 38, 36), fill=255)
    selected_path = mask_dir / "selected.png"
    selected.save(selected_path)
    training_medium = Image.new("L", image.size, 0)
    ImageDraw.Draw(training_medium).rectangle((24, 22, 54, 42), fill=255)
    training_medium_path = variant_dir / "training_medium.png"
    training_medium.save(training_medium_path)
    training_soft = Image.new("L", image.size, 0)
    ImageDraw.Draw(training_soft).rectangle((24, 22, 54, 42), fill=110)
    ImageDraw.Draw(training_soft).rectangle((30, 28, 38, 36), fill=255)
    training_soft_path = variant_dir / "training_soft.png"
    training_soft.save(training_soft_path)
    uncertainty = Image.new("L", image.size, 0)
    ImageDraw.Draw(uncertainty).rectangle((24, 22, 54, 42), fill=255)
    uncertainty_path = variant_dir / "uncertainty.png"
    uncertainty.save(uncertainty_path)
    heat = Image.new("L", image.size, 0)
    ImageDraw.Draw(heat).rectangle((24, 22, 54, 42), fill=210)
    heat_path = mask_dir / "patchcore_heatmap.png"
    heat.save(heat_path)
    candidate_path = mask_dir / "candidate.png"
    training_medium.save(candidate_path)

    metadata_dir = output / "auto_masks" / "qwen"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "category": "custom_part",
        "defect_type": "scratch",
        "image_path": str(image_path),
        "refined_mask_path": str(selected_path),
        "training_mask_path": str(training_soft_path),
        "region_xyxy": [12, 12, 72, 52],
        "mask_variant_paths": {
            "training_medium": str(training_medium_path),
            "training_soft": str(training_soft_path),
            "uncertainty_map": str(uncertainty_path),
        },
        "settings": {
            "selected_refinement": "patchcore_guided",
            "quality_morphology": "multi_scuff",
            "label_policy": {"label_policy": "soft_mask_only"},
            "candidate_refined_paths": {"patchcore_guided": str(candidate_path)},
            "candidate_heatmap_paths": {"patchcore_guided": str(heat_path)},
        },
    }
    (metadata_dir / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "phase10.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        f"""
project:
  output_dir: {output}
  report_dir: {report}
dataset:
  root: {dataset}
  targets:
    custom_part: [scratch]
generation: {{}}
models: {{}}
evaluation: {{}}
auto_masks:
  provider: qwen
""",
        encoding="utf-8",
    )
    summary = run_phase10_mask_quality_ablation(load_config(config_path), "qwen")

    assert summary.exists()
    csv_path = report / "phase10_mask_quality" / "qwen" / "mask_quality_metrics.csv"
    contact_sheet = report / "phase10_mask_quality" / "qwen" / "mask_quality_ablation_contact_sheet.png"
    assert csv_path.exists()
    assert contact_sheet.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert {row["policy"] for row in rows} == {"hard_binary", "vote_soft", "calibrated_soft"}
    assert {row["sample_id"] for row in rows} == {"custom_part/scratch/000.png"}
    assert all(float(row["critic_score"]) >= 0.0 for row in rows)
