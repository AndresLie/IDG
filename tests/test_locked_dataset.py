from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from iadgen_v2.config import load_config
from iadgen_v2.dataset import download_categories
from iadgen_v2.locked_dataset import prepare_locked_benchmark


def test_unconfigured_mvtec_category_uses_full_archive_mirror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_mirror(
        root: Path,
        archive_dir: Path,
        categories: list[str],
        official_failures: list[str],
        *,
        unconfigured_categories: list[str] | None = None,
    ) -> None:
        calls.append(
            {
                "root": root,
                "archive_dir": archive_dir,
                "categories": categories,
                "official_failures": official_failures,
                "unconfigured_categories": unconfigured_categories,
            }
        )

    monkeypatch.setattr("iadgen_v2.dataset._download_from_mirror", fake_mirror)

    download_categories(tmp_path / "source", ["cable", "capsule"])

    assert calls == [
        {
            "root": tmp_path / "source",
            "archive_dir": tmp_path / "source" / "_archives",
            "categories": ["cable", "capsule"],
            "official_failures": [],
            "unconfigured_categories": ["cable", "capsule"],
        }
    ]


def test_prepare_locked_benchmark_isolates_masks_and_hashes_inventories(tmp_path: Path) -> None:
    config = load_config(_locked_config(tmp_path))
    source = tmp_path / "source"
    _write_source_sample(source)

    manifest_path = prepare_locked_benchmark(config, source_root=source)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = tmp_path / "runtime"
    official = tmp_path / "official"
    reference_manifest = official / "locked_reference_manifest.jsonl"
    assert manifest["schema_version"] == 2
    assert manifest["categories"] == ["part"]
    assert manifest["source"]["file_count"] == 3
    assert manifest["runtime"]["file_count"] == 2
    assert manifest["runtime"]["contains_official_masks"] is False
    assert len(manifest["source"]["inventory_fingerprint"]) == 64
    assert len(manifest["runtime"]["inventory_fingerprint"]) == 64
    assert manifest["preregistration"]["sha256"] == _sha256(tmp_path / "preregister.yaml")
    assert manifest["reference"]["manifest_sha256"] == _sha256(reference_manifest)
    assert (runtime / "part" / "train" / "good" / "000.png").is_file()
    assert (runtime / "part" / "test" / "defect" / "001.png").is_file()
    assert not list(runtime.rglob("ground_truth"))
    assert (official / "part" / "defect" / "001_mask.png").is_file()
    assert (official / "locked_reference_manifest_summary.json").is_file()


def test_prepare_locked_benchmark_downloads_only_through_official_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_locked_config(tmp_path))
    source = tmp_path / "source"
    calls: list[tuple[Path, list[str]]] = []

    def fake_download(root: Path, categories: list[str]) -> None:
        calls.append((root, categories))
        _write_source_sample(root)

    monkeypatch.setattr("iadgen_v2.locked_dataset.download_categories", fake_download)

    prepare_locked_benchmark(config, source_root=source, download_source=True)

    assert calls == [(source.resolve(), ["part"])]


def test_prepare_locked_benchmark_rejects_overlapping_source(tmp_path: Path) -> None:
    config = load_config(_locked_config(tmp_path))

    with pytest.raises(ValueError, match="mutually isolated"):
        prepare_locked_benchmark(config, source_root=tmp_path / "runtime")


def test_prepare_locked_benchmark_requires_preregistration(tmp_path: Path) -> None:
    config_path = _locked_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        text.replace(f"  locked_preregistration_path: {tmp_path / 'preregister.yaml'}\n", ""),
        encoding="utf-8",
    )
    config = load_config(config_path)

    with pytest.raises(ValueError, match="locked_preregistration_path"):
        prepare_locked_benchmark(config, source_root=tmp_path / "source")


def _locked_config(tmp_path: Path) -> Path:
    preregistration = tmp_path / "preregister.yaml"
    preregistration.write_text("primary_endpoint: category_macro_dice\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {tmp_path / 'runtime'}",
                "  targets:",
                "    part: [defect]",
                "generation: {}",
                "models: {}",
                "evaluation: {}",
                "research_governance:",
                "  enabled: true",
                "  architecture_tag: locked-test",
                "  development_categories: [development_part]",
                "  locked_categories: [part]",
                f"  locked_official_mask_roots: [{tmp_path / 'official'}]",
                f"  locked_preregistration_path: {preregistration}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_source_sample(root: Path) -> None:
    files = {
        root / "part" / "train" / "good" / "000.png": b"normal",
        root / "part" / "test" / "defect" / "001.png": b"defect",
        root / "part" / "ground_truth" / "defect" / "001_mask.png": b"mask",
    }
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
