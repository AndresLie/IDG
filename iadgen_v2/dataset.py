from __future__ import annotations

import hashlib
import json
import random
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from iadgen_v2.config import AppConfig, SurfaceTarget
from iadgen_v2.records import AnomalySample, write_json


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MVTEC_URLS = {
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937637-1629959294/metal_nut.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938133-1629960456/tile.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938383-1629960649/wood.tar.xz",
}
MVTEC_MIRROR_ARCHIVE_URL = (
    "https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/"
    "resolve/main/mvtec_anomaly_detection.tar.xz?download=true"
)


def prepare_splits(config: AppConfig, download: bool | None = None) -> Path:
    root = config.dataset_root
    should_download = config.data["dataset"].get("download", False) if download is None else download
    categories = sorted({target.category for target in config.targets})
    if should_download:
        download_categories(root, categories)
    _validate_structure(root, config.targets)

    seed = int(config.data["dataset"].get("seed", 1337))
    adaptation_count = int(config.data["dataset"].get("adaptation_per_defect", 5))
    if adaptation_count < 1:
        raise ValueError("dataset.adaptation_per_defect must be positive")

    target_rows: dict[str, dict[str, object]] = {}
    auto_mask_index = _auto_mask_variant_index(config)
    for target in config.targets:
        samples = _collect_anomalies(root, target, auto_mask_index)
        if len(samples) <= adaptation_count:
            raise ValueError(
                f"{target.category}/{target.defect_type} contains {len(samples)} images; "
                f"need more than adaptation_per_defect={adaptation_count} for held-out evaluation"
            )
        rng = random.Random(seed + _stable_int(f"{target.category}:{target.defect_type}"))
        rng.shuffle(samples)
        key = f"{target.category}/{target.defect_type}"
        target_rows[key] = {
            "category": target.category,
            "defect_type": target.defect_type,
            "adaptation": [asdict(row) for row in samples[:adaptation_count]],
            "held_out": [asdict(row) for row in samples[adaptation_count:]],
            "clean_targets": [str(path) for path in _iter_images(root / target.category / "train" / "good")],
        }

    manifest = {
        "dataset_root": str(root),
        "seed": seed,
        "adaptation_per_defect": adaptation_count,
        "split_spec": config.split_spec(),
        "split_spec_fingerprint": config.split_spec_fingerprint(),
        "no_leakage_rule": (
            "held_out samples are excluded from references, source masks, placement calibration, "
            "prompt selection, and hyperparameter selection"
        ),
        "targets": target_rows,
    }
    manifest_path = config.output_dir / "prepared" / "split_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def load_manifest(config: AppConfig) -> dict[str, object]:
    path = config.output_dir / "prepared" / "split_manifest.json"
    if not path.exists():
        prepare_splits(config)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    found = manifest.get("split_spec_fingerprint")
    expected = config.split_spec_fingerprint()
    if found != expected:
        raise ValueError(
            "Prepared split manifest does not match the active dataset split configuration; "
            "run the prepare command again before generation."
        )
    return manifest


def download_categories(root: Path, categories: list[str]) -> None:
    archive_dir = root / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    unconfigured_categories = sorted(category for category in categories if category not in MVTEC_URLS)
    if unconfigured_categories:
        print(
            "no direct category URL configured for "
            f"{', '.join(unconfigured_categories)}; using full-archive mirror",
            flush=True,
        )
        _download_from_mirror(
            root,
            archive_dir,
            categories,
            [],
            unconfigured_categories=unconfigured_categories,
        )
        return
    failed_official_urls: list[str] = []
    for category in categories:
        if (root / category).exists():
            continue
        archive = archive_dir / f"{category}.tar.xz"
        if not archive.exists():
            print(f"downloading MVTec AD {category} to {archive}", flush=True)
            try:
                _download(MVTEC_URLS[category], archive, category)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                failed_official_urls.append(MVTEC_URLS[category])
                print(f"official category link unavailable for {category}; using full-archive mirror", flush=True)
                _download_from_mirror(root, archive_dir, categories, failed_official_urls)
                return
        print(f"extracting {archive}", flush=True)
        _extract_tar_safely(archive, root)
    if failed_official_urls:
        write_json(root / "download_source.json", {"official_failures": failed_official_urls})


def _download_from_mirror(
    root: Path,
    archive_dir: Path,
    categories: list[str],
    official_failures: list[str],
    *,
    unconfigured_categories: list[str] | None = None,
) -> None:
    archive = archive_dir / "mvtec_anomaly_detection.tar.xz"
    if not archive.exists():
        print(f"downloading MVTec AD fallback archive to {archive}", flush=True)
        _download(MVTEC_MIRROR_ARCHIVE_URL, archive, "mvtec_ad_archive")
    print(f"extracting configured categories only from {archive}", flush=True)
    _extract_tar_safely(archive, root, included_categories=set(categories))
    write_json(
        root / "download_source.json",
        {
            "official_dataset_page": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
            "official_category_links_unavailable": official_failures,
            "direct_category_links_unconfigured": unconfigured_categories or [],
            "fallback_mirror": MVTEC_MIRROR_ARCHIVE_URL,
            "extracted_categories": categories,
            "license": "CC BY-NC-SA 4.0",
        },
    )
    archive.unlink()


def _download(url: str, destination: Path, label: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(url, partial, _progress(label))
        print("", flush=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _validate_structure(root: Path, targets: list[SurfaceTarget]) -> None:
    missing: list[str] = []
    for target in targets:
        for relative in ("train/good", f"test/{target.defect_type}", f"ground_truth/{target.defect_type}"):
            path = root / target.category / relative
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing MVTec AD Phase 1 data:\n  " + "\n  ".join(missing))


def _collect_anomalies(root: Path, target: SurfaceTarget, auto_mask_index: dict[str, dict[str, object]] | None = None) -> list[AnomalySample]:
    masks = sorted(_iter_images(root / target.category / "ground_truth" / target.defect_type))
    rows: list[AnomalySample] = []
    for mask in masks:
        stem = mask.stem.removesuffix("_mask")
        matches = [path for path in _iter_images(root / target.category / "test" / target.defect_type) if path.stem == stem]
        if matches:
            variants = (auto_mask_index or {}).get(str(mask), {})
            rows.append(
                AnomalySample(
                    target.category,
                    target.defect_type,
                    str(matches[0]),
                    str(mask),
                    training_mask_path=str(variants["training_mask_path"]) if variants.get("training_mask_path") else None,
                    eval_mask_path=str(variants["eval_mask_path"]) if variants.get("eval_mask_path") else None,
                    uncertainty_mask_path=str(variants["uncertainty_mask_path"]) if variants.get("uncertainty_mask_path") else None,
                    positive_core_path=str(variants["positive_core_path"]) if variants.get("positive_core_path") else None,
                    possible_region_path=str(variants["possible_region_path"]) if variants.get("possible_region_path") else None,
                    label_policy=variants.get("label_policy") if isinstance(variants.get("label_policy"), dict) else None,
                )
            )
    if not rows:
        raise FileNotFoundError(f"No paired anomaly masks found for {target.category}/{target.defect_type}")
    return rows


def _auto_mask_variant_index(config: AppConfig) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for metadata_path in (config.output_dir / "auto_masks").glob("*/metadata.jsonl"):
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mask_path = row.get("mask_path")
            if not mask_path:
                continue
            variants: dict[str, object] = {}
            if row.get("training_mask_path"):
                variants["training_mask_path"] = str(row["training_mask_path"])
            if row.get("eval_mask_path"):
                variants["eval_mask_path"] = str(row["eval_mask_path"])
            if row.get("uncertainty_mask_path"):
                variants["uncertainty_mask_path"] = str(row["uncertainty_mask_path"])
            mask_variants = row.get("mask_variant_paths", {})
            if isinstance(mask_variants, dict):
                if mask_variants.get("positive_core"):
                    variants["positive_core_path"] = str(mask_variants["positive_core"])
                if mask_variants.get("possible_region"):
                    variants["possible_region_path"] = str(mask_variants["possible_region"])
            if isinstance(row.get("label_policy"), dict):
                variants["label_policy"] = row["label_policy"]
            settings = row.get("settings", {})
            if isinstance(settings, dict):
                if settings.get("training_mask_path"):
                    variants["training_mask_path"] = str(settings["training_mask_path"])
                if settings.get("eval_mask_path"):
                    variants["eval_mask_path"] = str(settings["eval_mask_path"])
                if settings.get("uncertainty_mask_path"):
                    variants["uncertainty_mask_path"] = str(settings["uncertainty_mask_path"])
                settings_variants = settings.get("mask_variant_paths", {})
                if isinstance(settings_variants, dict):
                    if settings_variants.get("positive_core"):
                        variants["positive_core_path"] = str(settings_variants["positive_core"])
                    if settings_variants.get("possible_region"):
                        variants["possible_region_path"] = str(settings_variants["possible_region"])
                if isinstance(settings.get("label_policy"), dict):
                    variants["label_policy"] = settings["label_policy"]
            if variants:
                index[str(mask_path)] = variants
    return index


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _progress(label: str):
    state = {"last": -5}

    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        percent = min(100, int(block_count * block_size * 100 / total_size))
        if percent >= state["last"] + 5:
            state["last"] = percent
            print(f"\r{label}: {percent:3d}%", end="", flush=True)

    return report


def _extract_tar_safely(
    archive_path: Path, destination: Path, included_categories: set[str] | None = None
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe tar link member: {member.name}")
            extracted = (destination_resolved / member.name).resolve()
            if extracted != destination_resolved and not extracted.is_relative_to(destination_resolved):
                raise RuntimeError(f"Unsafe tar member: {member.name}")
        selected = members if included_categories is None else [
            member for member in members if _member_category(member.name) in included_categories
        ]
        archive.extractall(destination_resolved, members=selected)
    nested = destination / "mvtec_anomaly_detection"
    if nested.exists():
        for path in nested.iterdir():
            target = destination / path.name
            if not target.exists():
                shutil.move(str(path), str(target))


def _member_category(name: str) -> str | None:
    parts = Path(name).parts
    if not parts:
        return None
    if parts[0] == "mvtec_anomaly_detection" and len(parts) > 1:
        return parts[1]
    return parts[0]
