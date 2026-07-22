from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iadgen_v2.auto_mask.contracts import CandidateProposal, EvidenceMap, SelectionDecision
from iadgen_v2.auto_mask.specialists import specialist_applicability
from iadgen_v2.config import load_config
from iadgen_v2.governance import (
    architecture_core_fingerprint,
    build_experiment_manifest,
    validate_governance_for_command,
    write_experiment_manifest,
)


def test_governance_rejects_development_locked_overlap(tmp_path: Path) -> None:
    config_path = _config_path(
        tmp_path,
        governance=[
            "  development_categories: [part]",
            "  locked_categories: [part]",
            f"  locked_official_mask_roots: [{tmp_path / 'official'}]",
        ],
    )
    with pytest.raises(ValueError, match="must be disjoint"):
        load_config(config_path)


def test_governance_rejects_official_masks_inside_runtime_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    config_path = _config_path(
        tmp_path,
        governance=[
            "  development_categories: []",
            "  locked_categories: [part]",
            f"  locked_official_mask_roots: [{dataset_root / 'official_masks'}]",
        ],
    )
    with pytest.raises(ValueError, match="must be isolated"):
        load_config(config_path)


def test_runtime_command_rejects_config_reference_to_locked_masks(tmp_path: Path) -> None:
    official_root = tmp_path / "official"
    config_path = _config_path(
        tmp_path,
        governance=[
            "  development_categories: []",
            "  locked_categories: [part]",
            f"  locked_official_mask_roots: [{official_root}]",
        ],
        extra=["auto_masks:", f"  seed_mask_path: {official_root / 'part' / '000_mask.png'}"],
    )
    config = load_config(config_path)
    with pytest.raises(ValueError, match="cannot access locked official masks"):
        validate_governance_for_command(config, "auto-masks")


def test_experiment_manifest_records_governance_dataset_and_code_fingerprints(tmp_path: Path) -> None:
    official_root = tmp_path / "official"
    config = load_config(
        _config_path(
            tmp_path,
            governance=[
                "  architecture_tag: test-general-core",
                "  experiment_track: scarce-defect-pseudo-labeling",
                "  development_categories: [part]",
                "  locked_categories: [held_out_part]",
                f"  locked_official_mask_roots: [{official_root}]",
            ],
        )
    )
    image_path = config.dataset_root / "part" / "train" / "good" / "000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-placeholder")

    manifest = build_experiment_manifest(config, "auto-masks", provider="qwen")

    assert manifest["architecture_tag"] == "test-general-core"
    assert manifest["governance"]["target_roles"] == {"part": "development"}
    assert manifest["dataset"]["file_count"] == 1
    assert len(manifest["dataset"]["inventory_fingerprint"]) == 64
    assert len(manifest["code"]["content_fingerprint"]) == 64
    assert manifest["provider"] == "qwen"

    path = write_experiment_manifest(config, "auto-masks", provider="qwen")
    written = json.loads(path.read_text(encoding="utf-8"))
    latest = json.loads((path.parent / "latest.json").read_text(encoding="utf-8"))
    assert written["run_id"] == latest["run_id"]
    assert latest["manifest_path"] == str(path)


def test_generalization_contracts_validate_shapes_confidence_and_disposition() -> None:
    values = np.zeros((8, 8), dtype=np.float32)
    evidence = EvidenceMap(source="normal_memory", values=values, reliability=0.8)
    proposal = CandidateProposal(mode="evidence_threshold_90", mask=values > 0.5, score=0.7)
    decision = SelectionDecision(
        selected_mode=proposal.mode,
        expected_iou=0.6,
        expected_precision=0.7,
        expected_recall=0.55,
        confidence=0.75,
        disposition="hard_mask_ok",
    )
    assert evidence.values.shape == proposal.mask.shape
    assert decision.disposition == "hard_mask_ok"
    with pytest.raises(ValueError, match="disposition"):
        SelectionDecision(None, None, None, None, 0.5, "force_accept")


def test_specialists_activate_from_structure_not_category_name() -> None:
    assert specialist_applicability(
        "repeated_chain_refiner",
        structure_profile="repeated_chain",
        attributes={"scale": "small"},
    )
    assert specialist_applicability(
        "polar_rim_residual",
        structure_profile="ring_sector",
        attributes={"scale": "micro"},
    )
    assert not specialist_applicability(
        "polar_rim_residual",
        structure_profile="ring_sector",
        attributes={"scale": "large"},
    )
    assert specialist_applicability("edge_border_layout", structure_profile="edge_border")


def test_architecture_fingerprint_covers_behavior_but_not_operational_device(tmp_path: Path) -> None:
    common = [
        "auto_masks:",
        "  architecture: generic_evidence",
        "  qwen_prompt_style: defect_localization_json",
        "  qwen_device: cpu",
        "  generic_evidence:",
        "    sam_positive_points: 5",
    ]
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
            extra=common,
        )
    )
    original = architecture_core_fingerprint(config)
    config.data["auto_masks"]["qwen_device"] = "cuda"
    assert architecture_core_fingerprint(config) == original
    config.data["auto_masks"]["max_images_per_target"] = 1
    assert architecture_core_fingerprint(config) != original
    config.data["auto_masks"].pop("max_images_per_target")
    config.data["auto_masks"]["generic_evidence"]["sam_positive_points"] = 7
    assert architecture_core_fingerprint(config) != original


def test_generic_generation_is_blocked_by_failed_development_gate(tmp_path: Path) -> None:
    gate = tmp_path / "reports" / "gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text('{"go": false, "architecture_tag": "test"}\n', encoding="utf-8")
    config = load_config(
        _config_path(
            tmp_path,
            governance=[
                "  architecture_tag: test",
                "  development_categories: [part]",
                "  locked_categories: []",
                f"  generic_mask_gate_path: {gate}",
            ],
            extra=["auto_masks:", "  architecture: generic_evidence"],
        )
    )

    with pytest.raises(ValueError, match="development mask gate did not pass"):
        validate_governance_for_command(config, "phase4-generate")


def test_unregistered_command_has_no_official_mask_policy(tmp_path: Path) -> None:
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
        )
    )
    with pytest.raises(ValueError, match="no registered governance policy"):
        validate_governance_for_command(config, "unregistered-command")


def _config_path(
    tmp_path: Path,
    *,
    governance: list[str],
    extra: list[str] | None = None,
) -> Path:
    path = tmp_path / "configs" / "test.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {tmp_path / 'data'}",
                "  targets:",
                "    part: [defect]",
                "generation: {}",
                "models: {}",
                "evaluation: {}",
                "research_governance:",
                "  enabled: true",
                "  enforce_official_mask_isolation: true",
                *governance,
                *(extra or []),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
