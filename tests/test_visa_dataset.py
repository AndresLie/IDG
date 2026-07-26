from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iadgen_v2.config import load_config
from iadgen_v2.visa_dataset import prepare_visa_benchmark


def test_prepare_visa_benchmark_isolates_masks_and_is_deterministic(tmp_path: Path) -> None:
    config_path = _visa_fixture(tmp_path)
    config = load_config(config_path)

    first_path = prepare_visa_benchmark(config)
    first = first_path.read_bytes()
    second_path = prepare_visa_benchmark(config)
    second = second_path.read_bytes()

    assert first == second
    manifest = json.loads(first)
    runtime = tmp_path / "runtime"
    official = tmp_path / "official"
    assert manifest["dataset"] == "VisA"
    assert manifest["categories"] == ["candle"]
    assert manifest["runtime"]["file_count"] == 3
    assert manifest["runtime"]["contains_official_masks"] is False
    assert manifest["reference"]["row_count"] == 1
    assert manifest["split_counts"] == {
        "test_bad": 1,
        "test_good": 1,
        "train_good": 1,
    }
    assert not list(runtime.rglob("ground_truth"))
    assert (runtime / "candle" / "train" / "good" / "000.JPG").is_file()
    assert (runtime / "candle" / "test" / "good" / "002.JPG").is_file()
    assert (runtime / "candle" / "test" / "bad" / "001.JPG").is_file()
    output_mask = official / "candle" / "bad" / "001.png"
    assert output_mask.is_file()
    with Image.open(output_mask) as mask:
        assert set(np.unique(np.asarray(mask))) == {0, 255}
    assert (official / "visa_reference_manifest.jsonl").is_file()


def test_prepare_visa_benchmark_rejects_unpinned_archive(tmp_path: Path) -> None:
    config_path = _visa_fixture(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        f"    archive_sha256: {_sha256(tmp_path / 'archive.tar')}",
        f"    archive_sha256: {'0' * 64}",
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        prepare_visa_benchmark(load_config(config_path))


def test_prepare_visa_benchmark_rejects_stale_runtime_files(tmp_path: Path) -> None:
    config = load_config(_visa_fixture(tmp_path))
    stale = tmp_path / "runtime" / "candle" / "test" / "bad" / "stale.JPG"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="Unexpected stale files"):
        prepare_visa_benchmark(config)


def _visa_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    normal = source / "VisA" / "candle" / "Data" / "Images" / "Normal"
    anomaly = source / "VisA" / "candle" / "Data" / "Images" / "Anomaly"
    masks = source / "VisA" / "candle" / "Data" / "Masks" / "Anomaly"
    for directory in (normal, anomaly, masks):
        directory.mkdir(parents=True)
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(normal / "000.JPG")
    Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8)).save(normal / "002.JPG")
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8)).save(anomaly / "001.JPG")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 7
    Image.fromarray(mask).save(masks / "001.png")

    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(source / "VisA", arcname="VisA")
    split = tmp_path / "1cls.csv"
    split.write_text(
        "\n".join(
            [
                "object,split,label,image,mask",
                "candle,train,normal,candle/Data/Images/Normal/000.JPG,",
                "candle,test,normal,candle/Data/Images/Normal/002.JPG,",
                "candle,test,anomaly,candle/Data/Images/Anomaly/001.JPG,candle/Data/Masks/Anomaly/001.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preregistration = tmp_path / "preregister.yaml"
    preregistration.write_text("primary_endpoint: category_macro_dice\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {tmp_path / 'runtime'}",
                "  targets:",
                "    candle: [bad]",
                "generation: {}",
                "models: {}",
                "evaluation: {}",
                "external_benchmarks:",
                "  visa:",
                "    archive_url: https://example.invalid/visa.tar",
                f"    archive_path: {archive}",
                f"    archive_size: {archive.stat().st_size}",
                f"    archive_sha256: {_sha256(archive)}",
                "    split_url: https://example.invalid/1cls.csv",
                f"    split_path: {split}",
                f"    split_size: {split.stat().st_size}",
                f"    split_sha256: {_sha256(split)}",
                f"    source_extract_root: {tmp_path / 'extracted'}",
                "    official_repository: https://github.com/amazon-science/spot-diff",
                "    official_commit: test",
                "research_governance:",
                "  enabled: true",
                "  architecture_tag: visa-test",
                "  development_categories: [part]",
                "  locked_categories: [candle]",
                f"  locked_official_mask_roots: [{tmp_path / 'official'}]",
                f"  locked_preregistration_path: {preregistration}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
