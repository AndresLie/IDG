from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import iadgen_v2.baseline as baseline_module
from iadgen_v2.baseline import run_baseline
from iadgen_v2.config import load_config
from iadgen_v2.dataset import _extract_tar_safely, load_manifest, prepare_splits
from iadgen_v2.masks import place_adaptation_mask
from iadgen_v2.metrics import evaluate_generation
from iadgen_v2.segmentation import segmentation_metrics


TARGETS = {"metal_nut": "scratch", "tile": "crack", "wood": "scratch"}


def test_prepare_creates_disjoint_adaptation_and_held_out_splits(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    manifest_path = prepare_splits(config, download=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for target in manifest["targets"].values():
        adaptation = {row["image_path"] for row in target["adaptation"]}
        held_out = {row["image_path"] for row in target["held_out"]}
        assert len(adaptation) == 2
        assert len(held_out) == 2
        assert not adaptation & held_out


def test_mock_baseline_uses_adaptation_samples_and_emits_metrics(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    metadata_path = run_baseline(config, "mock")
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((config.output_dir / "prepared" / "split_manifest.json").read_text(encoding="utf-8"))
    adaptation_paths = {
        row["image_path"]
        for target in manifest["targets"].values()
        for row in target["adaptation"]
    }
    assert len(records) == 3
    assert all(record["source_image_path"] in adaptation_paths for record in records)
    assert all(record["error"] is None for record in records)
    metrics_path = evaluate_generation(config, "mock")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert all(float(row["masked_visibility_l1"]) > 0.0 for row in rows)
    assert all(row["visibility_accepted"] == "True" for row in rows)
    assert all("outside_inpaint_leakage" in row for row in rows)


def test_segmentation_metric_loop_reports_required_metrics() -> None:
    truth = np.zeros((8, 8), dtype=np.uint8)
    truth[2:5, 2:5] = 1
    prediction = truth.astype(np.float32)
    result = segmentation_metrics(prediction, truth)
    assert result["pixel_auroc"] == 1.0
    assert result["pixel_ap"] == 1.0
    assert result["iou"] == 1.0
    assert result["dice"] == 1.0
    assert result["aupro"] > 0.99


def test_pixel_auroc_treats_tied_scores_as_chance() -> None:
    truth = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    prediction = np.full(truth.shape, 0.5, dtype=np.float32)
    assert segmentation_metrics(prediction, truth)["pixel_auroc"] == 0.5


def test_generation_raises_when_required_sample_fails(tmp_path: Path, monkeypatch) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    class FailingRunner:
        def generate(self, **kwargs):
            raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(baseline_module, "_make_runner", lambda config, model: FailingRunner())
    with pytest.raises(RuntimeError, match="generation failed for 3 required sample"):
        run_baseline(config, "mock")
    records = (config.output_dir / "generated" / "mock" / "metadata.jsonl").read_text(encoding="utf-8")
    assert "deliberate test failure" in records


def test_manifest_is_rejected_after_split_configuration_changes(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    config.data["dataset"]["adaptation_per_defect"] = 1
    with pytest.raises(ValueError, match="does not match"):
        load_manifest(config)


def test_evaluation_rejects_outputs_after_generation_settings_change(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    run_baseline(config, "mock")
    config.data["generation"]["seed"] = 12
    with pytest.raises(ValueError, match="does not match"):
        evaluate_generation(config, "mock")


def test_surface_constrained_object_mask_stays_on_foreground(tmp_path: Path) -> None:
    background = Image.new("RGB", (64, 64), (0, 0, 0))
    ImageDraw.Draw(background).ellipse((12, 12, 52, 52), fill=(160, 160, 160))
    source = Image.new("L", (16, 16), 0)
    ImageDraw.Draw(source).line((3, 8, 12, 8), fill=255, width=2)
    source_path = tmp_path / "source.png"
    source.save(source_path)
    binary_path, _, _, coverage = place_adaptation_mask(
        background,
        source_path,
        tmp_path,
        "sample",
        7,
        category="metal_nut",
        placement_mode="surface_constrained",
    )
    assert coverage is not None and coverage >= 0.90
    binary = np.asarray(Image.open(binary_path)) > 0
    assert binary.any()


def test_archive_extraction_rejects_symlinks(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("metal_nut/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        handle.addfile(member)
    with pytest.raises(RuntimeError, match="link member"):
        _extract_tar_safely(archive, tmp_path / "data")


def _fixture_config(tmp_path: Path):
    dataset_root = tmp_path / "data" / "mvtec_ad"
    for category, defect_type in TARGETS.items():
        clean_dir = dataset_root / category / "train" / "good"
        test_dir = dataset_root / category / "test" / defect_type
        mask_dir = dataset_root / category / "ground_truth" / defect_type
        clean_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32), (80, 80, 80)).save(clean_dir / "000.png")
        for index in range(4):
            image = Image.new("RGB", (32, 32), (80, 80, 80))
            mask = Image.new("L", (32, 32), 0)
            draw = ImageDraw.Draw(mask)
            draw.line((5, 10 + index, 25, 12 + index), fill=255, width=2)
            image.paste(Image.new("RGB", image.size, (180, 40, 40)), mask=mask)
            image.save(test_dir / f"{index:03d}.png")
            mask.save(mask_dir / f"{index:03d}_mask.png")
    config_path = tmp_path / "phase1.yaml"
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
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
