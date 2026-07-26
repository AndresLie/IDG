from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.governance import governance_settings, validate_governance_config
from iadgen_v2.records import write_json


VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


def prepare_visa_benchmark(config: AppConfig, *, download: bool = False) -> Path:
    governance = validate_governance_config(config)
    if not governance["enabled"]:
        raise ValueError("prepare-visa-benchmark requires research governance")
    categories = sorted({target.category for target in config.targets})
    defects = {target.defect_type for target in config.targets}
    if not categories or not set(categories) <= set(VISA_CATEGORIES):
        raise ValueError("prepare-visa-benchmark targets must be official VisA categories")
    if defects != {"bad"}:
        raise ValueError("prepare-visa-benchmark targets must use the official one-class defect label 'bad'")
    if not set(categories) <= set(governance["locked_categories"]):
        raise ValueError("prepare-visa-benchmark targets must all be governed locked categories")

    official_roots = [Path(value).resolve() for value in governance["official_mask_roots"]]
    if len(official_roots) != 1:
        raise ValueError("prepare-visa-benchmark requires exactly one official-mask root")
    runtime_root = config.dataset_root.resolve()
    official_root = official_roots[0]
    if _overlap(runtime_root, official_root):
        raise ValueError("VisA runtime and official reference roots must not overlap")

    settings = _settings(config)
    archive_path = config.resolve_path(str(settings["archive_path"]))
    split_path = config.resolve_path(str(settings["split_path"]))
    source_root = config.resolve_path(str(settings["source_extract_root"]))
    for path in (archive_path.parent, split_path.parent, source_root.parent):
        path.mkdir(parents=True, exist_ok=True)
    if download:
        _download(
            str(settings["archive_url"]),
            archive_path,
            int(settings["archive_size"]),
        )
        _download(
            str(settings["split_url"]),
            split_path,
            int(settings.get("split_size", 0)) or None,
        )
    _verify_artifact(
        archive_path,
        expected_size=int(settings["archive_size"]),
        expected_sha256=str(settings["archive_sha256"]),
        label="VisA archive",
    )
    _verify_artifact(
        split_path,
        expected_size=int(settings.get("split_size", 0)) or None,
        expected_sha256=str(settings["split_sha256"]),
        label="VisA official split",
    )

    if not source_root.exists() or not _locate_dataset_root(source_root, categories):
        _extract_tar_safely(archive_path, source_root)
    dataset_root = _locate_dataset_root(source_root, categories)
    if dataset_root is None:
        raise FileNotFoundError("Extracted VisA archive does not contain the expected category layout")
    if _overlap(dataset_root, runtime_root) or _overlap(dataset_root, official_root):
        raise ValueError("VisA source, runtime, and official reference roots must be mutually isolated")

    split_rows = _read_split(split_path, categories)
    preregistration = _preregistration(config)
    runtime_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    expected_runtime: set[Path] = set()
    expected_reference: set[Path] = set()
    counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = {
        category: Counter() for category in categories
    }

    for row in split_rows:
        category = row["object"]
        split = row["split"]
        label = row["label"]
        image_source = _safe_source_path(dataset_root, row["image"])
        if not image_source.is_file():
            raise FileNotFoundError(f"Missing VisA source image: {image_source}")
        output_label = "good" if label == "normal" else "bad"
        image_destination = runtime_root / category / split / output_label / image_source.name
        _copy_verified(image_source, image_destination)
        expected_runtime.add(image_destination.resolve())
        runtime_rows.append(_file_row(image_destination, runtime_root))
        source_rows.append(_file_row(image_source, dataset_root))
        counts[f"{split}_{output_label}"] += 1
        category_counts[category][f"{split}_{output_label}"] += 1

        if split == "test" and output_label == "bad":
            if not row["mask"]:
                raise ValueError(f"Anomalous VisA row lacks a mask path: {row}")
            mask_source = _safe_source_path(dataset_root, row["mask"])
            if not mask_source.is_file():
                raise FileNotFoundError(f"Missing VisA source mask: {mask_source}")
            mask_destination = official_root / category / "bad" / mask_source.name
            _write_binary_mask(mask_source, mask_destination)
            expected_reference.add(mask_destination.resolve())
            source_rows.append(_file_row(mask_source, dataset_root))
            reference_rows.append(
                {
                    "category": category,
                    "defect_type": "bad",
                    "image_path": str(image_destination),
                    "image_sha256": _sha256(image_destination),
                    "official_mask_path": str(mask_destination),
                    "official_mask_sha256": _sha256(mask_destination),
                }
            )

    _reject_stale_files(runtime_root, expected_runtime, "runtime")
    _reject_stale_files(official_root, expected_reference, "official reference")
    if any(path.name == "ground_truth" for path in runtime_root.rglob("ground_truth")):
        raise RuntimeError("Official VisA masks leaked into the runtime dataset")

    reference_manifest = official_root / "visa_reference_manifest.jsonl"
    reference_manifest.parent.mkdir(parents=True, exist_ok=True)
    with reference_manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(reference_rows, key=lambda item: (item["category"], item["image_path"])):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    reference_manifest_sha256 = _sha256(reference_manifest)

    manifest_path = runtime_root / "visa_runtime_dataset_manifest.json"
    manifest = {
        "schema_version": 1,
        "dataset": "VisA",
        "license": "CC-BY-4.0",
        "categories": categories,
        "official_source": {
            "archive_url": str(settings["archive_url"]),
            "archive_path": str(archive_path),
            "archive_size": archive_path.stat().st_size,
            "archive_sha256": _sha256(archive_path),
            "repository": str(settings["official_repository"]),
            "repository_commit": str(settings["official_commit"]),
            "split_url": str(settings["split_url"]),
            "split_path": str(split_path),
            "split_sha256": _sha256(split_path),
        },
        "preregistration": preregistration,
        "source": {
            "root": str(dataset_root),
            "referenced_file_count": len(source_rows),
            "inventory_fingerprint": fingerprint(source_rows),
        },
        "runtime": {
            "root": str(runtime_root),
            "file_count": len(runtime_rows),
            "inventory_fingerprint": fingerprint(runtime_rows),
            "contains_official_masks": False,
        },
        "reference": {
            "root": str(official_root),
            "row_count": len(reference_rows),
            "manifest_path": str(reference_manifest),
            "manifest_sha256": reference_manifest_sha256,
            "inventory_fingerprint": fingerprint(reference_rows),
        },
        "split_counts": dict(sorted(counts.items())),
        "category_counts": {
            category: dict(sorted(category_counts[category].items()))
            for category in categories
        },
        "rows": runtime_rows,
    }
    write_json(manifest_path, manifest)
    write_json(
        official_root / "visa_reference_manifest_summary.json",
        {
            "schema_version": 1,
            "dataset": "VisA",
            "categories": categories,
            "preregistration": preregistration,
            "reference_manifest_path": str(reference_manifest),
            "reference_manifest_sha256": reference_manifest_sha256,
            "row_count": len(reference_rows),
            "inventory_fingerprint": fingerprint(reference_rows),
        },
    )
    return manifest_path


def _settings(config: AppConfig) -> dict[str, Any]:
    external = config.data.get("external_benchmarks", {})
    visa = external.get("visa", {}) if isinstance(external, dict) else {}
    if not isinstance(visa, dict):
        raise ValueError("external_benchmarks.visa must be a mapping")
    required = {
        "archive_url",
        "archive_path",
        "archive_size",
        "archive_sha256",
        "split_url",
        "split_path",
        "split_sha256",
        "source_extract_root",
        "official_repository",
        "official_commit",
    }
    missing = required - set(visa)
    if missing:
        raise ValueError(f"external_benchmarks.visa is missing: {sorted(missing)}")
    return visa


def _download(url: str, destination: Path, expected_size: int | None) -> None:
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise ValueError(
            f"Downloaded size mismatch for {destination.name}: "
            f"expected {expected_size}, got {partial.stat().st_size}"
        )
    partial.replace(destination)


def _verify_artifact(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size}, got {path.stat().st_size}"
        )
    actual = _sha256(path)
    if len(expected_sha256) != 64 or actual != expected_sha256.lower():
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")


def _extract_tar_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not _is_within(target, root):
                raise ValueError(f"Unsafe path in VisA archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not permitted in the VisA archive: {member.name}")
        handle.extractall(destination)


def _locate_dataset_root(source_root: Path, categories: list[str]) -> Path | None:
    for candidate in (source_root, *sorted(path for path in source_root.rglob("*") if path.is_dir())):
        if all((candidate / category / "Data" / "Images").is_dir() for category in categories):
            return candidate.resolve()
    return None


def _read_split(path: Path, categories: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["object", "split", "label", "image", "mask"]:
            raise ValueError(f"Unexpected VisA split columns: {reader.fieldnames}")
        rows = [
            {key: str(value or "").strip() for key, value in row.items()}
            for row in reader
            if str(row.get("object", "")).strip() in categories
        ]
    if not rows:
        raise ValueError("Official VisA split contains no configured categories")
    seen_categories = {row["object"] for row in rows}
    if seen_categories != set(categories):
        raise ValueError(
            f"Official VisA split is missing configured categories: {sorted(set(categories) - seen_categories)}"
        )
    for row in rows:
        if row["split"] not in {"train", "test"} or row["label"] not in {"normal", "anomaly"}:
            raise ValueError(f"Invalid official VisA split row: {row}")
        if row["split"] == "train" and row["label"] != "normal":
            raise ValueError("VisA one-class split must contain only normal training images")
    return rows


def _safe_source_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not _is_within(path, root.resolve()):
        raise ValueError(f"VisA split path escapes the source root: {relative}")
    return path


def _copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and _sha256(destination) == _sha256(source):
        return
    shutil.copy2(source, destination)


def _write_binary_mask(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        values = np.asarray(image)
    binary = np.where(values != 0, 255, 0).astype(np.uint8)
    if binary.ndim == 3:
        binary = np.max(binary, axis=2)
    output = Image.fromarray(binary, mode="L")
    if destination.is_file():
        with Image.open(destination) as existing:
            if np.array_equal(np.asarray(existing.convert("L")), binary):
                return
    output.save(destination)


def _reject_stale_files(root: Path, expected: set[Path], label: str) -> None:
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {
            "visa_runtime_dataset_manifest.json",
            "visa_reference_manifest.jsonl",
            "visa_reference_manifest_summary.json",
        }
    }
    stale = sorted(actual - expected)
    if stale:
        raise RuntimeError(f"Unexpected stale files in VisA {label} root: {stale[:5]}")


def _file_row(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _preregistration(config: AppConfig) -> dict[str, str]:
    value = governance_settings(config).get("locked_preregistration_path")
    if not value:
        raise ValueError("prepare-visa-benchmark requires locked_preregistration_path")
    path = config.resolve_path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"VisA preregistration does not exist: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)
