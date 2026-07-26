from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iadgen_v2.config import AppConfig
from iadgen_v2.governance import architecture_core_fingerprint, governance_settings
from iadgen_v2.records import write_json


def freeze_generic_architecture(config: AppConfig) -> Path:
    settings = governance_settings(config)
    gate_value = settings.get("generic_mask_gate_path")
    if not gate_value:
        raise ValueError("Architecture freeze requires research_governance.generic_mask_gate_path")
    gate_path = config.resolve_path(str(gate_value))
    if not gate_path.exists():
        raise FileNotFoundError(f"Development gate not found: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("go", False)):
        raise ValueError("Architecture cannot be frozen because the development gate did not pass")
    architecture_tag = str(settings.get("release_architecture_tag", settings.get("architecture_tag", "v3-generic-evidence-rc1")))
    output_path = config.report_dir / "architecture_freeze" / f"{architecture_tag}.json"
    write_json(
        output_path,
        {
            "go": True,
            "architecture_tag": architecture_tag,
            "architecture_core_fingerprint": architecture_core_fingerprint(config),
            "development_gate_path": str(gate_path),
            "development_gate": gate,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "post_freeze_category_tuning_forbidden": True,
        },
    )
    return output_path


def reseal_behavior_equivalent_architecture(config: AppConfig) -> Path:
    settings = governance_settings(config)
    parent_path = _required_path(config, settings, "parent_frozen_architecture_manifest")
    baseline_path = _required_path(config, settings, "equivalence_baseline_metadata_path")
    candidate_path = _required_path(config, settings, "equivalence_candidate_metadata_path")
    parent = _read_json(parent_path)
    if not bool(parent.get("go", False)):
        raise ValueError("Parent frozen architecture is not approved")

    architecture_tag = str(settings.get("release_architecture_tag", "")).strip()
    active_tag = str(settings.get("architecture_tag", "")).strip()
    if not architecture_tag or architecture_tag != active_tag:
        raise ValueError("Operational reseal requires matching architecture_tag and release_architecture_tag")
    if architecture_tag == str(parent.get("architecture_tag", "")):
        raise ValueError("Operational reseal must issue a new architecture tag")

    auto = config.data.get("auto_masks", {})
    if not isinstance(auto, dict):
        raise ValueError("auto_masks must be a mapping")
    selector_value = auto.get("selector_model_path")
    selector_path = config.resolve_path(str(selector_value)) if selector_value else None
    selector_sha256 = _sha256_file(selector_path) if selector_path is not None and selector_path.is_file() else None
    parent_selector = parent.get("selector_sha256")
    if parent_selector and selector_sha256 != str(parent_selector):
        raise ValueError("Operational reseal cannot change the frozen selector")
    if parent.get("specialists") is not None and str(auto.get("specialists", "disabled")) != str(parent["specialists"]):
        raise ValueError("Operational reseal cannot change the frozen specialist policy")
    generic = auto.get("generic_evidence", {}) if isinstance(auto.get("generic_evidence"), dict) else {}
    current_pca = bool(generic.get("pca_subspace_enabled", False))
    if parent.get("pca_subspace_enabled") is not None and current_pca != bool(parent["pca_subspace_enabled"]):
        raise ValueError("Operational reseal cannot change the frozen PCA policy")

    equivalence = _compare_auto_mask_runs(
        baseline_path,
        candidate_path,
        minimum_samples=int(settings.get("operational_equivalence_min_samples", 12)),
        forbidden_roots=[
            config.resolve_path(value)
            for value in settings.get("locked_official_mask_roots", [])
            if isinstance(value, str)
        ],
    )
    equivalence["baseline_metadata_path"] = _portable_path(config, baseline_path)
    equivalence["candidate_metadata_path"] = _portable_path(config, candidate_path)
    output_dir = config.report_dir / "architecture_freeze"
    equivalence_path = output_dir / f"{architecture_tag}-operational-equivalence.json"
    write_json(
        equivalence_path,
        {
            **equivalence,
            "parent_architecture_tag": parent.get("architecture_tag"),
            "candidate_architecture_tag": architecture_tag,
            "parent_manifest_path": _portable_path(config, parent_path),
            "parent_manifest_sha256": _sha256_file(parent_path),
            "official_masks_opened": False,
        },
    )

    output_path = output_dir / f"{architecture_tag}.json"
    write_json(
        output_path,
        {
            "go": True,
            "approval_scope": parent.get("approval_scope", "locked_confirmation_only"),
            "architecture_tag": architecture_tag,
            "architecture_core_fingerprint": architecture_core_fingerprint(config),
            "selector_sha256": selector_sha256,
            "specialists": str(auto.get("specialists", "disabled")),
            "pca_subspace_enabled": current_pca,
            "parent_frozen_architecture_manifest": _portable_path(config, parent_path),
            "parent_frozen_architecture_sha256": _sha256_file(parent_path),
            "operational_equivalence_manifest": _portable_path(config, equivalence_path),
            "operational_equivalence_sha256": _sha256_file(equivalence_path),
            "resealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reseal_scope": "behavior-equivalent operational hardening only",
            "post_freeze_category_tuning_forbidden": True,
        },
    )
    return output_path


def _compare_auto_mask_runs(
    baseline_path: Path,
    candidate_path: Path,
    *,
    minimum_samples: int,
    forbidden_roots: list[Path],
) -> dict[str, Any]:
    for path in (baseline_path, candidate_path):
        _reject_forbidden_path(path, forbidden_roots)
    baseline = _metadata_index(baseline_path)
    candidate = _metadata_index(candidate_path)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"Operational equivalence sample mismatch: missing={missing}, extra={extra}")
    if len(baseline) < minimum_samples:
        raise ValueError(
            f"Operational equivalence requires at least {minimum_samples} samples; found {len(baseline)}"
        )

    differences = {
        "qwen_region": 0,
        "selected_refinement": 0,
        "candidate_scores": 0,
        "candidate_measurements": 0,
        "candidate_predictions": 0,
        "candidate_modes": 0,
        "candidate_rejections": 0,
        "selection_decision": 0,
        "fused_evidence": 0,
        "mask_roles": 0,
    }
    role_comparisons = 0
    for key in sorted(baseline):
        before = baseline[key]
        after = candidate[key]
        if before.get("region_xyxy") != after.get("region_xyxy"):
            differences["qwen_region"] += 1
        before_settings = before.get("settings", {})
        after_settings = after.get("settings", {})
        for field in (
            "selected_refinement",
            "candidate_scores",
            "candidate_measurements",
            "candidate_predictions",
            "candidate_modes",
            "candidate_rejections",
            "selection_decision",
        ):
            if before_settings.get(field) != after_settings.get(field):
                differences[field] += 1

        before_fused = Path(str(before_settings.get("mask_parameters", {}).get("fused_evidence_path", "")))
        after_fused = Path(str(after_settings.get("mask_parameters", {}).get("fused_evidence_path", "")))
        if not _artifacts_equal(before_fused, after_fused, forbidden_roots):
            differences["fused_evidence"] += 1

        before_roles = dict(before.get("mask_variant_paths", {}))
        after_roles = dict(after.get("mask_variant_paths", {}))
        if set(before_roles) != set(after_roles):
            raise ValueError(f"Operational equivalence mask-role mismatch for sample: {key}")
        for role in sorted(before_roles):
            role_comparisons += 1
            if not _artifacts_equal(
                Path(str(before_roles[role])),
                Path(str(after_roles[role])),
                forbidden_roots,
            ):
                differences["mask_roles"] += 1

    nonzero = {key: value for key, value in differences.items() if value}
    if nonzero:
        raise ValueError(f"Operational equivalence failed: {nonzero}")
    return {
        "go": True,
        "schema_version": 1,
        "baseline_metadata_path": str(baseline_path),
        "baseline_metadata_sha256": _sha256_file(baseline_path),
        "candidate_metadata_path": str(candidate_path),
        "candidate_metadata_sha256": _sha256_file(candidate_path),
        "samples_compared": len(baseline),
        "mask_roles_compared": role_comparisons,
        "differences": differences,
    }


def _metadata_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Auto-mask metadata does not exist: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("category", "")),
            str(row.get("defect_type", "")),
            Path(str(row.get("image_path", ""))).name,
        )
        if key in index:
            raise ValueError(f"Duplicate auto-mask metadata sample: {key}")
        index[key] = row
    return index


def _artifacts_equal(first: Path, second: Path, forbidden_roots: list[Path]) -> bool:
    for path in (first, second):
        _reject_forbidden_path(path, forbidden_roots)
        if not path.is_file():
            raise FileNotFoundError(f"Operational equivalence artifact does not exist: {path}")
    return _sha256_file(first) == _sha256_file(second)


def _reject_forbidden_path(path: Path, forbidden_roots: list[Path]) -> None:
    resolved = path.resolve()
    for root in forbidden_roots:
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            continue
        raise ValueError(f"Operational equivalence cannot access official-mask path: {resolved}")


def _required_path(config: AppConfig, settings: dict[str, Any], key: str) -> Path:
    value = settings.get(key)
    if not value:
        raise ValueError(f"Operational reseal requires research_governance.{key}")
    path = config.resolve_path(str(value))
    if not path.is_file():
        raise FileNotFoundError(f"Operational reseal input does not exist: {path}")
    return path


def _portable_path(config: AppConfig, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(config.base_dir.resolve()))
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
