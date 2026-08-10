from __future__ import annotations

import hashlib
import io
import json
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from zipfile import ZipFile, ZipInfo

import numpy as np
from PIL import Image

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.governance import governance_settings, validate_governance_config
from iadgen_v2.records import write_json


@dataclass(frozen=True)
class ExternalDatasetSpec:
    """Pinned source and deterministic subset contract for exposed development data."""

    dataset_id: str
    layout: str
    archive_url: str
    archive_path: Path
    archive_size: int
    archive_sha256: str
    license: str
    max_train_good_per_category: int
    max_test_anomaly_per_category: int
    seed: int = 20260810

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("external development dataset_id must be non-empty")
        if self.layout not in {"btad", "ksdd2"}:
            raise ValueError(f"unsupported external development layout: {self.layout}")
        if self.archive_size <= 0 or len(self.archive_sha256) != 64:
            raise ValueError("external development archives require pinned size and SHA-256")
        if self.max_train_good_per_category < 1 or self.max_test_anomaly_per_category < 1:
            raise ValueError("external development subset caps must be positive")


@dataclass(frozen=True)
class ExternalArchiveSample:
    category: str
    split: str
    label: str
    image_member: str
    mask_member: str | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"}:
            raise ValueError(f"unsupported external sample split: {self.split}")
        if self.label not in {"good", "defect"}:
            raise ValueError(f"unsupported external sample label: {self.label}")
        if self.label == "defect" and not self.mask_member:
            raise ValueError("external defect samples require a reference mask")


class ExternalLayoutAdapter(Protocol):
    def discover(self, archive: ZipFile, spec: ExternalDatasetSpec) -> tuple[ExternalArchiveSample, ...]: ...


class BtadLayoutAdapter:
    def discover(self, archive: ZipFile, spec: ExternalDatasetSpec) -> tuple[ExternalArchiveSample, ...]:
        names = {_safe_member(info).as_posix() for info in archive.infolist() if not info.is_dir()}
        masks_by_stem: dict[tuple[str, str], str] = {}
        for name in sorted(names):
            parts = PurePosixPath(name).parts
            if len(parts) == 5 and parts[0] == "BTech_Dataset_transformed" and parts[2:4] == ("ground_truth", "ko"):
                key = (parts[1], Path(parts[4]).stem)
                if key in masks_by_stem:
                    raise ValueError(f"BTAD has duplicate masks for {key}: {masks_by_stem[key]}, {name}")
                masks_by_stem[key] = name
        samples: list[ExternalArchiveSample] = []
        for name in sorted(names):
            parts = PurePosixPath(name).parts
            if len(parts) != 5 or parts[0] != "BTech_Dataset_transformed":
                continue
            raw_category, split, label, filename = parts[1:]
            category = f"{spec.dataset_id}_{raw_category}"
            if split == "train" and label == "ok":
                samples.append(ExternalArchiveSample(category, "train", "good", name))
            elif split == "test" and label == "ko":
                mask = masks_by_stem.get((raw_category, Path(filename).stem))
                if mask is None:
                    raise FileNotFoundError(f"BTAD defect image lacks a mask: {name}")
                samples.append(ExternalArchiveSample(category, "test", "defect", name, mask))
        return _cap_samples(samples, spec)


class Ksdd2LayoutAdapter:
    def discover(self, archive: ZipFile, spec: ExternalDatasetSpec) -> tuple[ExternalArchiveSample, ...]:
        infos = {
            _safe_member(info).as_posix(): info
            for info in archive.infolist()
            if not info.is_dir()
        }
        samples: list[ExternalArchiveSample] = []
        for name in sorted(infos):
            parts = PurePosixPath(name).parts
            if len(parts) != 2 or parts[0] not in {"train", "test"} or Path(parts[1]).stem.endswith("_GT"):
                continue
            mask = f"{parts[0]}/{Path(parts[1]).stem}_GT.png"
            mask_info = infos.get(mask)
            if mask_info is None:
                if parts[0] == "train":
                    # KSDD2 contains two duplicate-looking training images without
                    # masks. Their label cannot be established, so exclude them.
                    continue
                raise FileNotFoundError(f"KSDD2 image lacks a paired mask: {name}")
            positive = _archive_mask_has_positive(archive, mask_info)
            if parts[0] == "train" and not positive:
                samples.append(ExternalArchiveSample(spec.dataset_id, "train", "good", name))
            elif parts[0] == "test" and positive:
                samples.append(ExternalArchiveSample(spec.dataset_id, "test", "defect", name, mask))
        return _cap_samples(samples, spec)


LAYOUT_ADAPTERS: dict[str, ExternalLayoutAdapter] = {
    "btad": BtadLayoutAdapter(),
    "ksdd2": Ksdd2LayoutAdapter(),
}


def prepare_external_development_benchmark(config: AppConfig) -> Path:
    """Prepare isolated runtime/reference trees from pinned exposed datasets."""

    governance = validate_governance_config(config)
    if not governance["enabled"]:
        raise ValueError("prepare-external-development requires research governance")
    settings = external_development_settings(config)
    reference_value = settings.get("reference_root")
    if not reference_value:
        raise ValueError("external_benchmarks.development.reference_root is required")
    runtime_root = config.dataset_root.resolve()
    reference_root = config.resolve_path(str(reference_value))
    official_roots = {Path(value).resolve() for value in governance["official_mask_roots"]}
    if reference_root.resolve() not in official_roots:
        raise ValueError("external development reference_root must be a governed official-mask root")
    if _overlap(runtime_root, reference_root):
        raise ValueError("external development runtime and reference roots must not overlap")
    specs = _parse_specs(config, settings)
    preregistration = _preregistration(config, settings)

    runtime_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    expected_runtime: set[Path] = set()
    expected_reference: set[Path] = set()
    counts: dict[str, dict[str, int]] = {}
    categories: set[str] = set()

    for spec in specs:
        _verify_archive(spec)
        if _overlap(spec.archive_path.resolve(), runtime_root) or _overlap(spec.archive_path.resolve(), reference_root):
            raise ValueError("external development archives must be isolated from runtime/reference roots")
        with ZipFile(spec.archive_path) as archive:
            samples = LAYOUT_ADAPTERS[spec.layout].discover(archive, spec)
            if not samples:
                raise ValueError(f"external development source {spec.dataset_id} produced no samples")
            source_counts = {"train_good": 0, "test_bad": 0}
            for sample in samples:
                output_label = "good" if sample.label == "good" else "bad"
                image_name = PurePosixPath(sample.image_member).name
                image_path = runtime_root / sample.category / sample.split / output_label / image_name
                image_sha256 = _write_archive_member(archive, sample.image_member, image_path)
                expected_runtime.add(image_path.resolve())
                runtime_rows.append(_file_row(image_path, runtime_root))
                source_rows.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "member": sample.image_member,
                        "sha256": image_sha256,
                    }
                )
                categories.add(sample.category)
                source_counts[f"{sample.split}_{output_label}"] += 1
                if sample.mask_member:
                    mask_name = f"{PurePosixPath(sample.mask_member).stem}_mask.png"
                    mask_path = reference_root / sample.category / "bad" / mask_name
                    mask_sha256, source_mask_sha256 = _write_binary_archive_mask(
                        archive,
                        sample.mask_member,
                        mask_path,
                    )
                    expected_reference.add(mask_path.resolve())
                    reference_rows.append(
                        {
                            "dataset_id": spec.dataset_id,
                            "category": sample.category,
                            "defect_type": "bad",
                            "image_path": str(image_path.resolve()),
                            "image_sha256": image_sha256,
                            "official_mask_path": str(mask_path.resolve()),
                            "official_mask_sha256": mask_sha256,
                        }
                    )
                    source_rows.append(
                        {
                            "dataset_id": spec.dataset_id,
                            "member": sample.mask_member,
                            "sha256": source_mask_sha256,
                        }
                    )
            counts[spec.dataset_id] = source_counts

    expected_targets = {(category, "bad") for category in categories}
    actual_targets = {(target.category, target.defect_type) for target in config.targets}
    if actual_targets != expected_targets:
        raise ValueError(
            "external development targets must exactly match prepared categories: "
            f"expected {sorted(expected_targets)}, got {sorted(actual_targets)}"
        )
    if not categories <= set(governance["development_categories"]):
        raise ValueError("external development categories must be governed development categories")

    _reject_stale_files(runtime_root, expected_runtime, "runtime")
    _reject_stale_files(reference_root, expected_reference, "reference")
    if any(path.name == "ground_truth" for path in runtime_root.rglob("ground_truth")):
        raise RuntimeError("official masks leaked into the external development runtime dataset")

    reference_manifest = reference_root / "external_development_reference_manifest.jsonl"
    _write_jsonl(reference_manifest, reference_rows)
    reference_hash = _sha256_file(reference_manifest)
    manifest_path = runtime_root / "external_development_runtime_manifest.json"
    manifest = {
        "schema_version": 1,
        "datasets": [spec.dataset_id for spec in specs],
        "categories": sorted(categories),
        "subset_policy": "sha256_rank_v1",
        "preregistration": preregistration,
        "sources": [
            {
                "dataset_id": spec.dataset_id,
                "layout": spec.layout,
                "archive_url": spec.archive_url,
                "archive_path": str(spec.archive_path),
                "archive_size": spec.archive_size,
                "archive_sha256": spec.archive_sha256,
                "license": spec.license,
                "max_train_good_per_category": spec.max_train_good_per_category,
                "max_test_anomaly_per_category": spec.max_test_anomaly_per_category,
                "seed": spec.seed,
            }
            for spec in specs
        ],
        "source_inventory_fingerprint": fingerprint(sorted(source_rows, key=lambda row: (row["dataset_id"], row["member"]))),
        "runtime": {
            "root": str(runtime_root),
            "file_count": len(runtime_rows),
            "inventory_fingerprint": fingerprint(sorted(runtime_rows, key=lambda row: row["path"])),
            "contains_official_masks": False,
        },
        "reference": {
            "root": str(reference_root),
            "row_count": len(reference_rows),
            "manifest_path": str(reference_manifest),
            "manifest_sha256": reference_hash,
            "inventory_fingerprint": fingerprint(sorted(reference_rows, key=lambda row: row["image_path"])),
        },
        "counts": counts,
        "rows": sorted(runtime_rows, key=lambda row: row["path"]),
    }
    write_json(manifest_path, manifest)
    write_json(
        reference_root / "external_development_reference_summary.json",
        {
            "schema_version": 1,
            "datasets": manifest["datasets"],
            "categories": manifest["categories"],
            "preregistration": preregistration,
            "manifest_path": str(reference_manifest),
            "manifest_sha256": reference_hash,
            "row_count": len(reference_rows),
        },
    )
    return manifest_path


def external_development_settings(config: AppConfig) -> dict[str, Any]:
    external = config.data.get("external_benchmarks", {})
    settings = external.get("development", {}) if isinstance(external, dict) else {}
    if not isinstance(settings, dict):
        raise ValueError("external_benchmarks.development must be a mapping")
    return settings


def _parse_specs(config: AppConfig, settings: dict[str, Any]) -> tuple[ExternalDatasetSpec, ...]:
    raw_sources = settings.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise ValueError("external development preparation requires at least two sources")
    specs = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError("external development sources must be mappings")
        required = {
            "dataset_id",
            "layout",
            "archive_url",
            "archive_path",
            "archive_size",
            "archive_sha256",
            "license",
            "max_train_good_per_category",
            "max_test_anomaly_per_category",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(f"external development source is missing: {sorted(missing)}")
        specs.append(
            ExternalDatasetSpec(
                dataset_id=str(raw["dataset_id"]),
                layout=str(raw["layout"]),
                archive_url=str(raw["archive_url"]),
                archive_path=config.resolve_path(str(raw["archive_path"])),
                archive_size=int(raw["archive_size"]),
                archive_sha256=str(raw["archive_sha256"]).lower(),
                license=str(raw["license"]),
                max_train_good_per_category=int(raw["max_train_good_per_category"]),
                max_test_anomaly_per_category=int(raw["max_test_anomaly_per_category"]),
                seed=int(raw.get("seed", settings.get("seed", 20260810))),
            )
        )
    dataset_ids = [spec.dataset_id for spec in specs]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("external development dataset_id values must be unique")
    return tuple(specs)


def _cap_samples(
    samples: list[ExternalArchiveSample],
    spec: ExternalDatasetSpec,
) -> tuple[ExternalArchiveSample, ...]:
    grouped: dict[tuple[str, str, str], list[ExternalArchiveSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.category, sample.split, sample.label), []).append(sample)
    selected: list[ExternalArchiveSample] = []
    for key, values in sorted(grouped.items()):
        cap = spec.max_train_good_per_category if key[1:] == ("train", "good") else spec.max_test_anomaly_per_category
        ranked = sorted(
            values,
            key=lambda sample: (
                hashlib.sha256(f"{spec.seed}:{spec.dataset_id}:{sample.image_member}".encode("utf-8")).digest(),
                sample.image_member,
            ),
        )
        selected.extend(ranked[:cap])
    return tuple(sorted(selected, key=lambda sample: (sample.category, sample.split, sample.image_member)))


def _safe_member(info: ZipInfo) -> PurePosixPath:
    if "\\" in info.filename:
        raise ValueError(f"unsafe backslash in ZIP member: {info.filename}")
    path = PurePosixPath(info.filename)
    file_type = (info.external_attr >> 16) & 0o170000
    if path.is_absolute() or ".." in path.parts or file_type == stat.S_IFLNK:
        raise ValueError(f"unsafe external development ZIP member: {info.filename}")
    return path


def _archive_mask_has_positive(archive: ZipFile, info: ZipInfo) -> bool:
    with archive.open(info) as handle, Image.open(handle) as image:
        return bool(np.any(np.asarray(image.convert("L"), dtype=np.uint8) != 0))


def _write_archive_member(archive: ZipFile, member: str, destination: Path) -> str:
    info = archive.getinfo(member)
    _safe_member(info)
    data = archive.read(info)
    digest = hashlib.sha256(data).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256_file(destination) != digest:
        destination.write_bytes(data)
    return digest


def _write_binary_archive_mask(archive: ZipFile, member: str, destination: Path) -> tuple[str, str]:
    info = archive.getinfo(member)
    _safe_member(info)
    source_data = archive.read(info)
    with Image.open(io.BytesIO(source_data)) as image:
        binary = np.where(np.asarray(image.convert("L"), dtype=np.uint8) != 0, 255, 0).astype(np.uint8)
    output = Image.fromarray(binary, mode="L")
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    output.save(buffer, format="PNG", optimize=False)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    if not destination.is_file() or _sha256_file(destination) != digest:
        destination.write_bytes(data)
    return digest, hashlib.sha256(source_data).hexdigest()


def _verify_archive(spec: ExternalDatasetSpec) -> None:
    if not spec.archive_path.is_file():
        raise FileNotFoundError(f"external development archive is missing: {spec.archive_path}")
    if spec.archive_path.stat().st_size != spec.archive_size:
        raise ValueError(
            f"external development archive size mismatch for {spec.dataset_id}: "
            f"expected {spec.archive_size}, got {spec.archive_path.stat().st_size}"
        )
    actual = _sha256_file(spec.archive_path)
    if actual != spec.archive_sha256:
        raise ValueError(
            f"external development archive SHA-256 mismatch for {spec.dataset_id}: "
            f"expected {spec.archive_sha256}, got {actual}"
        )


def _preregistration(config: AppConfig, settings: dict[str, Any]) -> dict[str, str]:
    value = settings.get("preregistration_path") or governance_settings(config).get("locked_preregistration_path")
    if not value:
        raise ValueError("prepare-external-development requires a preregistration_path")
    path = config.resolve_path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"external development preregistration does not exist: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["dataset_id"], row["category"], row["image_path"]))
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")


def _reject_stale_files(root: Path, expected: set[Path], label: str) -> None:
    ignored = {
        "external_development_runtime_manifest.json",
        "external_development_reference_manifest.jsonl",
        "external_development_reference_summary.json",
    }
    actual = {path.resolve() for path in root.rglob("*") if path.is_file() and path.name not in ignored}
    stale = sorted(actual - expected)
    if stale:
        raise RuntimeError(f"unexpected stale files in external development {label} root: {stale[:5]}")


def _file_row(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
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
