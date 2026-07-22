from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.phase11_tfidg_critic import run_phase11_tfidg_critic


def test_phase11_tfidg_critic_scores_phase4_generations(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    output = tmp_path / "outputs" / "tfidg"
    report = tmp_path / "reports" / "tfidg"
    train_good = dataset / "custom_part" / "train" / "good"
    test_dir = dataset / "custom_part" / "test" / "scratch"
    gt_dir = dataset / "custom_part" / "ground_truth" / "scratch"
    train_good.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    gt_dir.mkdir(parents=True)

    background = Image.new("RGB", (96, 64), (128, 116, 92))
    ImageDraw.Draw(background).line((0, 20, 95, 25), fill=(145, 132, 104), width=2)
    background_path = train_good / "000.png"
    background.save(background_path)

    reference = background.copy()
    ImageDraw.Draw(reference).line((24, 30, 72, 34), fill=(70, 62, 48), width=3)
    reference_path = test_dir / "000.png"
    reference.save(reference_path)
    reference_mask = Image.new("L", reference.size, 0)
    ImageDraw.Draw(reference_mask).line((24, 30, 72, 34), fill=255, width=5)
    reference_mask_path = gt_dir / "000_mask.png"
    reference_mask.save(reference_mask_path)

    held_out = background.copy()
    ImageDraw.Draw(held_out).line((25, 40, 74, 43), fill=(65, 60, 46), width=3)
    held_out.save(test_dir / "001.png")
    held_mask = Image.new("L", held_out.size, 0)
    ImageDraw.Draw(held_mask).line((25, 40, 74, 43), fill=255, width=5)
    held_mask.save(gt_dir / "001_mask.png")

    phase4_dir = output / "phase4" / "qwen"
    image_dir = phase4_dir / "fixed_mask_adapter" / "images"
    image_dir.mkdir(parents=True)
    good_output = background.copy()
    ImageDraw.Draw(good_output).line((24, 30, 72, 34), fill=(68, 60, 48), width=3)
    good_output_path = image_dir / "good.png"
    good_output.save(good_output_path)
    weak_output_path = image_dir / "weak.png"
    background.save(weak_output_path)

    refined_mask_path = output / "phase4" / "mask.png"
    reference_mask.save(refined_mask_path)
    inpaint_mask_path = output / "phase4" / "inpaint.png"
    reference_mask.save(inpaint_mask_path)

    rows = [
        {
            "category": "custom_part",
            "defect_type": "scratch",
            "variant": "fixed_mask_adapter",
            "quality_profile": "scratch_thin_detail",
            "background_path": str(background_path),
            "output_path": str(good_output_path),
            "refined_mask_path": str(refined_mask_path),
            "inpaint_mask_path": str(inpaint_mask_path),
            "generation_seed": 0,
            "generation_quality_score": 0.5,
            "defect_visibility_score": 0.05,
            "background_preservation_l1": 0.01,
            "outside_refined_change_fraction": 0.02,
            "settings": {"quality_morphology": "scratch_band"},
        },
        {
            "category": "custom_part",
            "defect_type": "scratch",
            "variant": "fixed_mask_adapter",
            "quality_profile": "scratch_thin_detail",
            "background_path": str(background_path),
            "output_path": str(weak_output_path),
            "refined_mask_path": str(refined_mask_path),
            "inpaint_mask_path": str(inpaint_mask_path),
            "generation_seed": 1,
            "generation_quality_score": 0.0,
            "defect_visibility_score": 0.0,
            "background_preservation_l1": 0.0,
            "outside_refined_change_fraction": 0.0,
            "settings": {"quality_morphology": "scratch_band"},
        },
    ]
    (phase4_dir / "metadata.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (phase4_dir / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "phase11.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        f"""
project:
  output_dir: {output}
  report_dir: {report}
dataset:
  root: {dataset}
  seed: 1
  adaptation_per_defect: 1
  targets:
    custom_part: [scratch]
generation: {{}}
models: {{}}
evaluation: {{}}
phase4:
  provider: qwen
phase11:
  max_rows: 2
  contact_sheet_rows: 2
""",
        encoding="utf-8",
    )

    summary = run_phase11_tfidg_critic(load_config(config_path), "qwen")

    assert summary.exists()
    csv_path = report / "phase11_tfidg_critic" / "qwen" / "tfidg_lite_metrics.csv"
    jsonl_path = report / "phase11_tfidg_critic" / "qwen" / "tfidg_lite_metrics.jsonl"
    contact_sheet = report / "phase11_tfidg_critic" / "qwen" / "tfidg_lite_contact_sheet.png"
    assert csv_path.exists()
    assert jsonl_path.exists()
    assert contact_sheet.exists()
    metrics = list(csv.DictReader(csv_path.open()))
    assert len(metrics) == 2
    by_name = {Path(row["output_path"]).name: row for row in metrics}
    assert float(by_name["good.png"]["tfidg_lite_score"]) > float(by_name["weak.png"]["tfidg_lite_score"])
    assert "low_mask_coverage" in by_name["weak.png"]["reject_reasons"]
