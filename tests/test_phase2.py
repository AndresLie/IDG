from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.phase2 import evaluate_phase2_placement, run_phase2_proposals


TARGETS = {"metal_nut": "scratch", "tile": "crack", "wood": "scratch"}


def test_phase2_heuristic_proposals_emit_masks_overlays_and_feature_cache(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    metadata_path = run_phase2_proposals(config, "heuristic")
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 3
    assert {row["category"] for row in records} == set(TARGETS)
    for record in records:
        for key in ("box_mask_path", "refined_mask_path", "inpaint_mask_path", "overlay_path", "feature_cache_path"):
            assert Path(record[key]).exists()
        box = Image.open(record["box_mask_path"]).convert("L")
        refined = Image.open(record["refined_mask_path"]).convert("L")
        assert _mask_area(refined) > 0
        assert _mask_area(refined) < _mask_area(box)
        assert record["normalized_region_xyxy"]
        assert record["settings"]["phase2_schema_version"] == 2
        assert record["settings"]["mask_parameters"]["surface_clipped"] is True
        cache = torch.load(record["feature_cache_path"], map_location="cpu", weights_only=False)
        assert tuple(cache["tokens"].shape) == (1, 8, 32)
        assert cache["metadata"]["provider"] == "heuristic"
        assert cache["metadata"]["tensor_shape"] == [1, 8, 32]
        assert "not Qwen hidden states" in cache["metadata"]["note"]


def test_phase2_placement_evaluation_writes_metrics(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    run_phase2_proposals(config, "heuristic")

    csv_path = evaluate_phase2_placement(config, "heuristic")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(float(row["region_surface_coverage"]) >= 0.90 for row in rows)
    assert all(0.0 < float(row["refined_to_box_area"]) < 1.0 for row in rows)
    assert all(float(row["refined_surface_coverage"]) == 1.0 for row in rows)
    assert all(float(row["refined_off_surface_fraction"]) == 0.0 for row in rows)
    assert all(float(row["inpaint_off_surface_fraction"]) == 0.0 for row in rows)
    assert (config.report_dir / "phase2" / "heuristic" / "summary.md").exists()


def test_phase2_qwen_provider_fails_until_real_dependencies_exist(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    with pytest.raises(RuntimeError, match="Qwen provider is not ready"):
        run_phase2_proposals(config, "qwen")


def test_phase2_evaluation_rejects_stale_proposals(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    run_phase2_proposals(config, "heuristic")

    config.data["phase2"]["min_surface_coverage"] = 0.95
    with pytest.raises(ValueError, match="does not match"):
        evaluate_phase2_placement(config, "heuristic")


def _mask_area(image: Image.Image) -> int:
    return int((np.asarray(image) > 0).sum())


def _fixture_config(tmp_path: Path):
    dataset_root = tmp_path / "data" / "mvtec_ad"
    for category, defect_type in TARGETS.items():
        clean_dir = dataset_root / category / "train" / "good"
        test_dir = dataset_root / category / "test" / defect_type
        mask_dir = dataset_root / category / "ground_truth" / defect_type
        clean_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        clean = Image.new("RGB", (48, 48), (15, 15, 15) if category == "metal_nut" else (80, 80, 80))
        if category == "metal_nut":
            ImageDraw.Draw(clean).ellipse((8, 8, 40, 40), fill=(150, 150, 150))
        clean.save(clean_dir / "000.png")
        for index in range(4):
            image = clean.copy()
            mask = Image.new("L", (48, 48), 0)
            draw = ImageDraw.Draw(mask)
            draw.line((8, 18 + index, 38, 20 + index), fill=255, width=2)
            image.paste(Image.new("RGB", image.size, (180, 40, 40)), mask=mask)
            image.save(test_dir / f"{index:03d}.png")
            mask.save(mask_dir / f"{index:03d}_mask.png")
    config_path = tmp_path / "phase2.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {dataset_root}",
                "  download: false",
                "  seed: 11",
                "  adaptation_per_defect: 2",
                "  targets:",
                "    metal_nut: [scratch]",
                "    tile: [crack]",
                "    wood: [scratch]",
                "generation:",
                "  samples_per_category: 1",
                "  seed: 11",
                "  device: cpu",
                "  prompt_by_defect: {scratch: scratch, crack: crack}",
                "models:",
                "  mock: {enabled: true}",
                "  sd15:",
                "    enabled: false",
                "    base_model: example/base",
                "    controlnet_model: example/control",
                "evaluation: {}",
                "phase2:",
                "  provider: heuristic",
                "  qwen_model: Qwen/Qwen2.5-VL-3B-Instruct",
                "  samples_per_category: 1",
                "  token_count: 8",
                "  feature_dim: 32",
                "  box_area_fraction: {scratch: 0.06, crack: 0.06}",
                "  min_surface_coverage: 0.90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
