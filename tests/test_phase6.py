from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.phase6 import write_phase6_preflight
from iadgen_v2.qwen_provider import parse_normalized_box


def test_parse_normalized_qwen_box_formats() -> None:
    assert parse_normalized_box("[0.1, 0.2, 0.7, 0.8]") == (0.1, 0.2, 0.7, 0.8)
    assert parse_normalized_box('{"box":[0.1,0.2,0.7,0.8],"confidence":0.9}') == (0.1, 0.2, 0.7, 0.8)
    assert parse_normalized_box("x1=100 y1=200 x2=700 y2=800") == (0.1, 0.2, 0.7, 0.8)
    assert parse_normalized_box("[0.7, 0.2, 0.1, 0.8]") is None


def test_phase6_preflight_writes_qwen_readiness_report(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    report_path = write_phase6_preflight(config)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert "cuda_available" in report
    assert "qwen" in report
    assert report["qwen"]["model_id"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert "next_action" in report


def _fixture_config(tmp_path: Path):
    dataset_root = tmp_path / "data" / "mvtec_ad"
    for category, defect_type in {"metal_nut": "scratch", "tile": "crack", "wood": "scratch"}.items():
        clean_dir = dataset_root / category / "train" / "good"
        test_dir = dataset_root / category / "test" / defect_type
        mask_dir = dataset_root / category / "ground_truth" / defect_type
        clean_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        clean = Image.new("RGB", (32, 32), (80, 80, 80))
        clean.save(clean_dir / "000.png")
        for index in range(3):
            image = clean.copy()
            mask = Image.new("L", (32, 32), 0)
            ImageDraw.Draw(mask).line((5, 10 + index, 25, 12 + index), fill=255, width=2)
            image.paste(Image.new("RGB", image.size, (180, 40, 40)), mask=mask)
            image.save(test_dir / f"{index:03d}.png")
            mask.save(mask_dir / f"{index:03d}_mask.png")
    config_path = tmp_path / "phase6.yaml"
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
                "  adaptation_per_defect: 1",
                "  targets:",
                "    metal_nut: [scratch]",
                "    tile: [crack]",
                "    wood: [scratch]",
                "generation:",
                "  samples_per_category: 1",
                "  seed: 11",
                "  device: cpu",
                "models:",
                "  mock: {enabled: true}",
                "  sd15:",
                "    enabled: false",
                "    base_model: example/base",
                "    controlnet_model: example/control",
                "evaluation: {}",
                "phase2:",
                "  qwen_model: Qwen/Qwen2.5-VL-3B-Instruct",
                "  qwen_local_files_only: true",
                "  qwen_min_free_gib: 30.0",
                "phase3: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
