from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iadgen_v2.config import AppConfig
from iadgen_v2.dataset import download_categories
from iadgen_v2.governance import governance_settings, validate_governance_config


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def prepare_locked_benchmark(
    config: AppConfig,
    *,
    source_root: Path,
    download_source: bool = False,
) -> Path:
    governance = validate_governance_config(config)
    if not governance["enabled"]:
        raise ValueError("prepare-locked-benchmark requires research governance")
    targets = {(target.category, target.defect_type) for target in config.targets}
    categories = sorted({category for category, _ in targets})
    if not targets or not set(categories) <= set(governance["locked_categories"]):
        raise ValueError("prepare-locked-benchmark targets must all be locked categories")
    official_roots = [Path(value).resolve() for value in governance["official_mask_roots"]]
    if len(official_roots) != 1:
        raise ValueError("prepare-locked-benchmark requires exactly one locked official-mask root")
    source_root = source_root.resolve()
    runtime_root = config.dataset_root.resolve()
    official_root = official_roots[0]
    if _overlap(runtime_root, official_root):
        raise ValueError("Runtime and official locked roots must not overlap")
    if _overlap(source_root, runtime_root) or _overlap(source_root, official_root):
        raise ValueError("Locked source, runtime, and official roots must be mutually isolated")
    preregistration = _preregistration(config)
    if download_source:
        download_categories(source_root, categories)
    runtime_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for category in categories:
        source_rows.extend(
            _file_row(path, source_root)
            for path in _images(source_root / category)
        )
        for source in _images(source_root / category / "train" / "good"):
            destination = runtime_root / category / "train" / "good" / source.name
            _copy(source, destination)
            runtime_rows.append(_file_row(destination, runtime_root))
    for category, defect in sorted(targets):
        test_dir = source_root / category / "test" / defect
        truth_dir = source_root / category / "ground_truth" / defect
        if not test_dir.exists() or not truth_dir.exists():
            raise FileNotFoundError(f"Missing locked source data for {category}/{defect}")
        for image in _images(test_dir):
            runtime_image = runtime_root / category / "test" / defect / image.name
            _copy(image, runtime_image)
            runtime_rows.append(_file_row(runtime_image, runtime_root))
            candidates = [path for path in _images(truth_dir) if path.stem.removesuffix("_mask") == image.stem]
            if not candidates:
                raise FileNotFoundError(f"Missing official mask for {category}/{defect}/{image.name}")
            official_mask = official_root / category / defect / candidates[0].name
            _copy(candidates[0], official_mask)
            reference_rows.append(
                {
                    "category": category,
                    "defect_type": defect,
                    "image_path": str(runtime_image),
                    "official_mask_path": str(official_mask),
                    "official_mask_sha256": _sha256(official_mask),
                }
            )
    if any(path.name == "ground_truth" for path in runtime_root.rglob("ground_truth")):
        raise RuntimeError("Official masks leaked into the locked runtime dataset")
    runtime_manifest = runtime_root / "locked_runtime_dataset_manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    reference_manifest = official_root / "locked_reference_manifest.jsonl"
    reference_manifest.parent.mkdir(parents=True, exist_ok=True)
    with reference_manifest.open("w", encoding="utf-8") as handle:
        for row in reference_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    reference_manifest_hash = _sha256(reference_manifest)
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "categories": categories,
        "preregistration": preregistration,
        "source": {
            "root": str(source_root),
            "file_count": len(source_rows),
            "inventory_fingerprint": _inventory_fingerprint(source_rows),
        },
        "runtime": {
            "root": str(runtime_root),
            "file_count": len(runtime_rows),
            "inventory_fingerprint": _inventory_fingerprint(runtime_rows),
            "contains_official_masks": False,
        },
        "reference": {
            "manifest_path": str(reference_manifest),
            "manifest_sha256": reference_manifest_hash,
            "row_count": len(reference_rows),
        },
        "rows": runtime_rows,
    }
    runtime_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reference_summary = official_root / "locked_reference_manifest_summary.json"
    reference_summary.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "preregistration": preregistration,
                "reference_manifest_path": str(reference_manifest),
                "reference_manifest_sha256": reference_manifest_hash,
                "row_count": len(reference_rows),
                "inventory_fingerprint": _inventory_fingerprint(reference_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return runtime_manifest


def _images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _sha256(destination) == _sha256(source):
        return
    shutil.copy2(source, destination)


def _file_row(path: Path, root: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": _sha256(path)}


def _preregistration(config: AppConfig) -> dict[str, str]:
    value = governance_settings(config).get("locked_preregistration_path")
    if not value:
        raise ValueError(
            "prepare-locked-benchmark requires research_governance.locked_preregistration_path"
        )
    path = config.resolve_path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"Locked preregistration does not exist: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _inventory_fingerprint(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False
