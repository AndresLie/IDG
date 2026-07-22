from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.records import write_json


GOVERNANCE_SECTION = "research_governance"
COMMAND_POLICIES = {
    "feasibility": "runtime",
    "prepare": "runtime",
    "auto-masks": "runtime",
    "auto-masks-reselect": "runtime",
    "auto-mask-train-selector": "runtime",
    "development-evaluate": "runtime",
    "generate": "runtime",
    "evaluate": "runtime",
    "phase2-propose": "runtime",
    "phase2-evaluate": "runtime",
    "phase3-cache": "runtime",
    "phase3-train": "runtime",
    "phase3-validate": "runtime",
    "phase4-generate": "runtime",
    "phase5-evaluate": "runtime",
    "phase6-preflight": "runtime",
    "phase9-visual-report": "runtime",
    "phase10-mask-quality": "runtime",
    "phase11-tfidg-critic": "runtime",
    "experiment-manifest": "runtime",
    "manual-review": "runtime",
    "locked-evaluate": "official-evaluation",
    "prepare-locked-benchmark": "official-preparation",
    "freeze-architecture": "runtime",
}


def governance_settings(config: AppConfig) -> dict[str, Any]:
    raw = config.data.get(GOVERNANCE_SECTION, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{GOVERNANCE_SECTION} must be a mapping")
    return dict(raw)


def validate_governance_config(config: AppConfig) -> dict[str, Any]:
    settings = governance_settings(config)
    if not settings or not bool(settings.get("enabled", False)):
        return {
            "enabled": False,
            "architecture_tag": str(settings.get("architecture_tag", "unversioned")),
            "development_categories": [],
            "locked_categories": [],
            "target_roles": {target.category: "unassigned" for target in config.targets},
            "official_mask_roots": [],
        }

    development = _category_set(settings.get("development_categories", []), "development_categories")
    locked = _category_set(settings.get("locked_categories", []), "locked_categories")
    overlap = development & locked
    if overlap:
        raise ValueError(
            "research_governance development_categories and locked_categories must be disjoint; "
            f"overlap: {sorted(overlap)}"
        )

    target_categories = {target.category for target in config.targets}
    unassigned = target_categories - development - locked
    if unassigned and not bool(settings.get("allow_unassigned_target_categories", False)):
        raise ValueError(
            "Governed dataset targets must be declared development or locked categories; "
            f"unassigned: {sorted(unassigned)}"
        )

    official_roots = [
        config.resolve_path(value)
        for value in _string_list(settings.get("locked_official_mask_roots", []), "locked_official_mask_roots")
    ]
    if locked and bool(settings.get("enforce_official_mask_isolation", True)) and not official_roots:
        raise ValueError(
            "research_governance.locked_official_mask_roots is required when locked categories are configured"
        )

    protected_runtime_roots = (config.dataset_root, config.output_dir, config.report_dir)
    for official_root in official_roots:
        for runtime_root in protected_runtime_roots:
            if _paths_overlap(official_root, runtime_root):
                raise ValueError(
                    "Locked official masks must be isolated from dataset/output/report roots: "
                    f"{official_root} overlaps {runtime_root}"
                )

    target_roles = {
        category: "development" if category in development else "locked" if category in locked else "unassigned"
        for category in sorted(target_categories)
    }
    return {
        "enabled": True,
        "architecture_tag": str(settings.get("architecture_tag", "unversioned")),
        "experiment_track": str(settings.get("experiment_track", "unspecified")),
        "development_categories": sorted(development),
        "locked_categories": sorted(locked),
        "target_roles": target_roles,
        "official_mask_roots": [str(path) for path in official_roots],
        "enforce_official_mask_isolation": bool(settings.get("enforce_official_mask_isolation", True)),
    }


def validate_governance_for_command(
    config: AppConfig,
    command: str,
    *,
    allow_official_masks: bool = False,
) -> dict[str, Any]:
    summary = validate_governance_config(config)
    policy = COMMAND_POLICIES.get(command)
    if policy is None:
        raise ValueError(f"Command {command!r} has no registered governance policy")
    if allow_official_masks and policy not in {"official-evaluation", "official-preparation"}:
        raise ValueError(f"Command {command!r} is not permitted to access official masks")
    if policy in {"official-evaluation", "official-preparation"} and not allow_official_masks:
        raise ValueError(f"Command {command!r} requires the isolated official-evaluation path")
    if not summary["enabled"] or policy in {"official-evaluation", "official-preparation"}:
        return summary
    if not bool(summary.get("enforce_official_mask_isolation", False)):
        return summary

    official_roots = [Path(value).resolve() for value in summary["official_mask_roots"]]
    references = list(_config_path_references(config))
    violations = [
        f"{key}={path}"
        for key, path in references
        if any(_is_within(path, official_root) for official_root in official_roots)
    ]
    if violations:
        raise ValueError(
            f"Command {command!r} cannot access locked official masks; forbidden config references: "
            + ", ".join(sorted(violations))
        )
    _validate_phase_gate(config, command, summary)
    _validate_locked_architecture_freeze(config, command, summary)
    return summary


def _validate_phase_gate(config: AppConfig, command: str, summary: dict[str, Any]) -> None:
    if command not in {"phase4-generate", "phase5-evaluate", "phase11-tfidg-critic"}:
        return
    auto = config.data.get("auto_masks", {})
    if not isinstance(auto, dict) or str(auto.get("architecture", "legacy_specialist")) != "generic_evidence":
        return
    settings = governance_settings(config)
    value = settings.get("generic_mask_gate_path")
    if not value:
        raise ValueError(f"Command {command!r} is frozen until research_governance.generic_mask_gate_path is configured")
    gate_path = config.resolve_path(str(value))
    if not gate_path.exists():
        raise ValueError(f"Command {command!r} is frozen until development mask validation passes: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("go", False)):
        raise ValueError(f"Command {command!r} is frozen because the development mask gate did not pass")
    if str(gate.get("architecture_tag")) != str(summary.get("architecture_tag")):
        raise ValueError("Development mask gate architecture does not match the active architecture tag")


def architecture_core_fingerprint(config: AppConfig) -> str:
    auto = config.data.get("auto_masks", {})
    if not isinstance(auto, dict):
        auto = {}
    selector_value = auto.get("selector_model_path")
    selector_path = config.resolve_path(str(selector_value)) if selector_value else None
    selector_hash = _sha256_file(selector_path) if selector_path is not None and selector_path.is_file() else None
    core = {
        "architecture": auto.get("architecture", "legacy_specialist"),
        "specialists": auto.get("specialists"),
        "qwen_model": auto.get("qwen_model"),
        "dinov2_model": auto.get("dinov2_model"),
        "generic_evidence": auto.get("generic_evidence", {}),
        "min_box_area_ratio": auto.get("min_box_area_ratio"),
        "max_box_area_ratio": auto.get("max_box_area_ratio"),
        "min_component_area": auto.get("min_component_area"),
        "selector_sha256": selector_hash,
        "code": code_inventory(config)["content_fingerprint"],
    }
    return fingerprint(core)


def _validate_locked_architecture_freeze(config: AppConfig, command: str, summary: dict[str, Any]) -> None:
    if command not in {"auto-masks", "auto-masks-reselect"}:
        return
    if not any(role == "locked" for role in summary.get("target_roles", {}).values()):
        return
    settings = governance_settings(config)
    value = settings.get("frozen_architecture_manifest")
    if not value:
        raise ValueError("Locked auto-mask runs require research_governance.frozen_architecture_manifest")
    path = config.resolve_path(str(value))
    if not path.exists():
        raise ValueError(f"Frozen architecture manifest does not exist: {path}")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if not bool(frozen.get("go", False)):
        raise ValueError("Frozen architecture manifest is not approved for locked evaluation")
    if str(frozen.get("architecture_tag")) != str(summary.get("architecture_tag")):
        raise ValueError("Frozen architecture tag does not match the locked config")
    if str(frozen.get("architecture_core_fingerprint")) != architecture_core_fingerprint(config):
        raise ValueError("Locked config/code/selector does not match the frozen architecture")


def build_experiment_manifest(
    config: AppConfig,
    command: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    governance = validate_governance_config(config)
    return {
        "schema_version": 1,
        "status": "started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "provider": provider,
        "model": model,
        "architecture_tag": governance["architecture_tag"],
        "experiment_track": governance.get("experiment_track", "unspecified"),
        "governance": governance,
        "config": {
            "path": str(config.path),
            "file_sha256": _sha256_file(config.path),
            "effective_fingerprint": fingerprint(config.data),
            "split_spec_fingerprint": config.split_spec_fingerprint(),
        },
        "dataset": dataset_inventory(config),
        "code": code_inventory(config),
        "models_fingerprint": fingerprint(config.data.get("models", {})),
        "runtime": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
        },
        "parent_inputs": input_artifact_inventory(config, command),
    }


def write_experiment_manifest(
    config: AppConfig,
    command: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    manifest = build_experiment_manifest(config, command, provider=provider, model=model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    directory = config.output_dir / "experiment_manifests" / command
    path = directory / f"{run_id}.json"
    write_json(path, {**manifest, "run_id": run_id})
    write_json(directory / "latest.json", {**manifest, "run_id": run_id, "manifest_path": str(path)})
    return path


def finalize_experiment_manifest(
    path: Path,
    *,
    status: str,
    started_monotonic: float,
    output_paths: Iterable[Path] = (),
    error: BaseException | None = None,
) -> Path:
    if status not in {"succeeded", "failed", "interrupted"}:
        raise ValueError(f"Unsupported manifest final status: {status}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"] = round(max(0.0, time.monotonic() - started_monotonic), 6)
    manifest["outputs"] = [_artifact_record(item) for item in output_paths if item.exists()]
    if error is not None:
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    write_json(path, manifest)
    latest_path = path.parent / "latest.json"
    write_json(latest_path, {**manifest, "manifest_path": str(path)})
    return path


def dataset_inventory(config: AppConfig) -> dict[str, Any]:
    settings = governance_settings(config)
    mode = str(settings.get("dataset_fingerprint_mode", "metadata")).lower()
    if mode not in {"metadata", "sha256"}:
        raise ValueError("research_governance.dataset_fingerprint_mode must be 'metadata' or 'sha256'")
    content_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for category in sorted({target.category for target in config.targets}):
        category_root = config.dataset_root / category
        if not category_root.exists():
            content_rows.append({"category": category, "status": "missing"})
            continue
        for path in sorted(item for item in category_root.rglob("*") if item.is_file()):
            stat = path.stat()
            metadata_row = {
                "category": category,
                "path": str(path.relative_to(config.dataset_root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            content_row = {
                "category": category,
                "path": str(path.relative_to(config.dataset_root)),
                "size": stat.st_size,
            }
            if mode == "sha256":
                content_row["sha256"] = _sha256_file(path)
            content_rows.append(content_row)
            metadata_rows.append(metadata_row)
    return {
        "root": str(config.dataset_root),
        "inventory_kind": "relative_path_size_sha256" if mode == "sha256" else "relative_path_size",
        "file_count": sum(1 for row in content_rows if "path" in row),
        "inventory_fingerprint": fingerprint(content_rows),
        "metadata_fingerprint": fingerprint(metadata_rows),
        "target_categories": sorted({target.category for target in config.targets}),
    }


def input_artifact_inventory(config: AppConfig, command: str) -> list[dict[str, Any]]:
    patterns_by_command = {
        "auto-masks-reselect": ("auto_masks/*/metadata.jsonl",),
        "phase2-propose": ("prepared/split_manifest.json", "auto_masks/*/metadata.jsonl"),
        "phase2-evaluate": ("phase2/*/metadata.jsonl",),
        "phase3-cache": ("prepared/split_manifest.json", "phase2/*/metadata.jsonl"),
        "phase3-train": ("prepared/split_manifest.json", "phase3/*/cache_metadata.jsonl"),
        "phase3-validate": ("phase3/*/adapter.pt",),
        "phase4-generate": ("prepared/split_manifest.json", "phase3/*/adapter.pt"),
        "phase5-evaluate": ("prepared/split_manifest.json", "phase4/*/*/metadata.jsonl"),
        "phase9-visual-report": ("phase5/*/segmentation_results.csv",),
        "phase11-tfidg-critic": ("phase4/*/*/metadata.jsonl",),
    }
    paths: set[Path] = set()
    for pattern in patterns_by_command.get(command, ()):
        paths.update(path for path in config.output_dir.glob(pattern) if path.is_file())
    settings = governance_settings(config)
    for value in _string_list(settings.get("parent_manifest_paths", []), "parent_manifest_paths"):
        path = config.resolve_path(value)
        if path.is_file():
            paths.add(path)
    return [_artifact_record(path) for path in sorted(paths)]


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        return {"path": str(resolved), "kind": "file", "size": resolved.stat().st_size, "sha256": _sha256_file(resolved)}
    rows = [
        {"path": str(item.relative_to(resolved)), "size": item.stat().st_size, "sha256": _sha256_file(item)}
        for item in sorted(resolved.rglob("*"))
        if item.is_file()
    ]
    return {"path": str(resolved), "kind": "directory", "file_count": len(rows), "sha256": fingerprint(rows)}


def code_inventory(config: AppConfig) -> dict[str, Any]:
    package_root = config.base_dir / "iadgen_v2"
    rows = [
        {"path": str(path.relative_to(config.base_dir)), "sha256": _sha256_file(path)}
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    ]
    return {
        "package_root": str(package_root),
        "python_file_count": len(rows),
        "content_fingerprint": fingerprint(rows),
    }


def _category_set(value: object, key: str) -> set[str]:
    return set(_string_list(value, key))


def _string_list(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"research_governance.{key} must be a list of non-empty strings")
    return [item.strip() for item in value]


def _config_path_references(config: AppConfig) -> Iterable[tuple[str, Path]]:
    def visit(value: object, prefix: str) -> Iterable[tuple[str, Path]]:
        if isinstance(value, dict):
            for key, child in value.items():
                if prefix == "" and key == GOVERNANCE_SECTION:
                    continue
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from visit(child, child_prefix)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from visit(child, f"{prefix}[{index}]")
        elif isinstance(value, str) and _looks_like_path(value):
            yield prefix, config.resolve_path(value).resolve()

    yield from visit(config.data, "")


def _looks_like_path(value: str) -> bool:
    lowered = value.lower()
    return "/" in value or "\\" in value or lowered.endswith(
        (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".json", ".jsonl", ".yaml", ".yml")
    )


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return _is_within(first, second) or _is_within(second, first)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "numpy",
        "pillow",
        "pyyaml",
        "scikit-image",
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "sam2",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
