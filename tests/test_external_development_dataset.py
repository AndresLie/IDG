from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from iadgen_v2.config import load_config
from iadgen_v2.external_development_dataset import (
    ExternalDatasetSpec,
    Ksdd2LayoutAdapter,
    prepare_external_development_benchmark,
)


def test_external_development_preparation_isolates_masks_and_is_deterministic(tmp_path: Path) -> None:
    config = load_config(_fixture_config(tmp_path))

    first_path = prepare_external_development_benchmark(config)
    first = first_path.read_bytes()
    second_path = prepare_external_development_benchmark(config)
    second = second_path.read_bytes()

    assert first == second
    manifest = json.loads(first)
    runtime = tmp_path / "runtime"
    reference = tmp_path / "reference"
    assert manifest["datasets"] == ["btad", "ksdd2"]
    assert manifest["categories"] == ["btad_01", "ksdd2"]
    assert manifest["runtime"]["file_count"] == 4
    assert manifest["runtime"]["contains_official_masks"] is False
    assert manifest["reference"]["row_count"] == 2
    assert manifest["counts"] == {
        "btad": {"test_bad": 1, "train_good": 1},
        "ksdd2": {"test_bad": 1, "train_good": 1},
    }
    assert not list(runtime.rglob("ground_truth"))
    assert not list(runtime.rglob("*_GT.png"))
    assert (runtime / "btad_01" / "train" / "good" / "0000.bmp").is_file()
    assert (runtime / "ksdd2" / "test" / "bad" / "20001.png").is_file()
    rows = [
        json.loads(line)
        for line in (reference / "external_development_reference_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["dataset_id"] for row in rows} == {"btad", "ksdd2"}
    for row in rows:
        with Image.open(row["official_mask_path"]) as mask:
            assert set(np.unique(np.asarray(mask))) <= {0, 255}


def test_external_development_rejects_unpinned_archive(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    btad = tmp_path / "btad.zip"
    text = text.replace(_sha256(btad), "0" * 64)
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="archive SHA-256 mismatch for btad"):
        prepare_external_development_benchmark(load_config(config_path))


def test_external_development_rejects_stale_runtime_files(tmp_path: Path) -> None:
    config = load_config(_fixture_config(tmp_path))
    prepare_external_development_benchmark(config)
    stale = tmp_path / "runtime" / "ksdd2" / "test" / "bad" / "stale.png"
    stale.write_bytes(b"stale")

    with pytest.raises(RuntimeError, match="unexpected stale files"):
        prepare_external_development_benchmark(config)


def test_ksdd2_adapter_rejects_unmatched_test_image(tmp_path: Path) -> None:
    archive_path = tmp_path / "ksdd2.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("test/20001.png", _image_bytes("PNG", 100))
    spec = ExternalDatasetSpec(
        dataset_id="ksdd2",
        layout="ksdd2",
        archive_url="https://example.invalid/ksdd2.zip",
        archive_path=archive_path,
        archive_size=archive_path.stat().st_size,
        archive_sha256=_sha256(archive_path),
        license="test-only",
        max_train_good_per_category=1,
        max_test_anomaly_per_category=1,
    )

    with ZipFile(archive_path) as archive, pytest.raises(FileNotFoundError, match="lacks a paired mask"):
        Ksdd2LayoutAdapter().discover(archive, spec)


def _fixture_config(tmp_path: Path) -> Path:
    btad = tmp_path / "btad.zip"
    ksdd2 = tmp_path / "ksdd2.zip"
    with ZipFile(btad, "w") as archive:
        archive.writestr("BTech_Dataset_transformed/01/train/ok/0000.bmp", _image_bytes("BMP", 80))
        archive.writestr("BTech_Dataset_transformed/01/test/ko/0001.bmp", _image_bytes("BMP", 130))
        archive.writestr("BTech_Dataset_transformed/01/ground_truth/ko/0001.bmp", _mask_bytes(True, "BMP"))
    with ZipFile(ksdd2, "w") as archive:
        archive.writestr("train/10001.png", _image_bytes("PNG", 90))
        archive.writestr("train/10001_GT.png", _mask_bytes(False))
        archive.writestr("train/10002.png", _image_bytes("PNG", 120))
        archive.writestr("train/10002_GT.png", _mask_bytes(True))
        archive.writestr("train/10003 (copy).png", _image_bytes("PNG", 100))
        archive.writestr("test/20001.png", _image_bytes("PNG", 140))
        archive.writestr("test/20001_GT.png", _mask_bytes(True))
        archive.writestr("test/20002.png", _image_bytes("PNG", 100))
        archive.writestr("test/20002_GT.png", _mask_bytes(False))
    preregistration = tmp_path / "preregister.yaml"
    preregistration.write_text("primary_endpoint: leave_dataset_out_macro_dice\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {tmp_path / 'runtime'}",
                "  targets:",
                "    btad_01: [bad]",
                "    ksdd2: [bad]",
                "generation: {}",
                "models: {}",
                "evaluation: {}",
                "external_benchmarks:",
                "  development:",
                f"    reference_root: {tmp_path / 'reference'}",
                f"    preregistration_path: {preregistration}",
                "    seed: 7",
                "    sources:",
                "      - dataset_id: btad",
                "        layout: btad",
                "        archive_url: https://example.invalid/btad.zip",
                f"        archive_path: {btad}",
                f"        archive_size: {btad.stat().st_size}",
                f"        archive_sha256: '{_sha256(btad)}'",
                "        license: test-only",
                "        max_train_good_per_category: 1",
                "        max_test_anomaly_per_category: 1",
                "      - dataset_id: ksdd2",
                "        layout: ksdd2",
                "        archive_url: https://example.invalid/ksdd2.zip",
                f"        archive_path: {ksdd2}",
                f"        archive_size: {ksdd2.stat().st_size}",
                f"        archive_sha256: '{_sha256(ksdd2)}'",
                "        license: test-only",
                "        max_train_good_per_category: 1",
                "        max_test_anomaly_per_category: 1",
                "research_governance:",
                "  enabled: true",
                "  architecture_tag: external-development-test",
                "  development_categories: [btad_01, ksdd2]",
                "  locked_categories: []",
                f"  locked_official_mask_roots: [{tmp_path / 'reference'}]",
                "  enforce_official_mask_isolation: true",
                "  dataset_fingerprint_mode: sha256",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _image_bytes(format_name: str, value: int) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((12, 16, 3), value, dtype=np.uint8)).save(buffer, format=format_name)
    return buffer.getvalue()


def _mask_bytes(positive: bool, format_name: str = "PNG") -> bytes:
    values = np.zeros((12, 16), dtype=np.uint8)
    if positive:
        values[3:8, 5:11] = 9
    buffer = io.BytesIO()
    Image.fromarray(values).save(buffer, format=format_name)
    return buffer.getvalue()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
