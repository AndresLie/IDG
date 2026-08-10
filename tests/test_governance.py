from __future__ import annotations

import hashlib
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
    configured_artifact_inventory,
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
    assert len(manifest["architecture_core_fingerprint"]) == 64
    assert manifest["provider"] == "qwen"

    path = write_experiment_manifest(config, "auto-masks", provider="qwen")
    written = json.loads(path.read_text(encoding="utf-8"))
    latest = json.loads((path.parent / "latest.json").read_text(encoding="utf-8"))
    assert written["run_id"] == latest["run_id"]
    assert latest["manifest_path"] == str(path)


def test_experiment_manifest_hashes_mutable_selector_artifact(tmp_path: Path) -> None:
    selector = tmp_path / "models" / "selector.joblib"
    selector.parent.mkdir(parents=True)
    selector.write_bytes(b"selector-v1")
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
            extra=["auto_masks:", f"  selector_model_path: {selector}"],
        )
    )

    first = build_experiment_manifest(config, "auto-masks")
    selector.write_bytes(b"selector-v2")
    second = build_experiment_manifest(config, "auto-masks")

    assert first["configured_artifacts"] == [
        {
            "config_key": "auto_masks.selector_model_path",
            "path": str(selector),
            "kind": "file",
            "size": len(b"selector-v1"),
            "sha256": hashlib.sha256(b"selector-v1").hexdigest(),
        }
    ]
    assert second["configured_artifacts"][0]["sha256"] == hashlib.sha256(b"selector-v2").hexdigest()
    assert first["architecture_core_fingerprint"] != second["architecture_core_fingerprint"]


def test_phase4_manifest_records_pinned_huggingface_snapshot(tmp_path: Path) -> None:
    revision = "fixed-revision"
    cache = tmp_path / "model_cache"
    snapshot = cache / "models--vendor--inpainting" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
        )
    )
    config.data["models"]["sd15"] = {
        "base_model": "vendor/inpainting",
        "cache_dir": str(cache),
        "revision": revision,
    }

    manifest = build_experiment_manifest(config, "phase4-generate", provider="qwen")

    assert manifest["configured_artifacts"] == [
        {
            "config_key": "models.sd15.base_model",
            "kind": "huggingface_snapshot",
            "model_id": "vendor/inpainting",
            "status": "resolved",
            "revision": revision,
            "file_count": 1,
            "snapshot_fingerprint": manifest["configured_artifacts"][0]["snapshot_fingerprint"],
        }
    ]
    assert len(manifest["configured_artifacts"][0]["snapshot_fingerprint"]) == 64


def test_phase5_manifest_hashes_explicit_synthetic_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "external" / "metadata.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"sample": 1}\n', encoding="utf-8")
    expected = hashlib.sha256(metadata.read_bytes()).hexdigest()
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
        )
    )
    config.data["phase5"] = {
        "synthetic_metadata_path": str(metadata),
        "synthetic_metadata_sha256": expected,
    }

    manifest = build_experiment_manifest(config, "phase5-evaluate", provider="qwen")

    assert manifest["configured_artifacts"] == [
        {
            "config_key": "phase5.synthetic_metadata_path",
            "configured_sha256": expected,
            "path": str(metadata),
            "kind": "file",
            "size": len(metadata.read_bytes()),
            "sha256": expected,
        }
    ]


def test_r6_manifest_hashes_every_configured_evidence_input(tmp_path: Path) -> None:
    first = tmp_path / "reports" / "r1.json"
    second = tmp_path / "reports" / "r3.json"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"r1")
    second.write_bytes(b"r3")
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
        )
    )
    config.data["r6_evidence"] = {
        "inputs": {
            "selector_evidence": {
                "path": str(first),
                "sha256": hashlib.sha256(b"r1").hexdigest(),
            },
            "locked_candidate_analysis": {
                "path": str(second),
                "sha256": hashlib.sha256(b"r3").hexdigest(),
            },
        }
    }

    records = configured_artifact_inventory(config, "r6-evidence-package")

    assert [row["config_key"] for row in records] == [
        "r6_evidence.inputs.locked_candidate_analysis.path",
        "r6_evidence.inputs.selector_evidence.path",
    ]
    assert [row["sha256"] for row in records] == [
        hashlib.sha256(b"r3").hexdigest(),
        hashlib.sha256(b"r1").hexdigest(),
    ]
    validate_governance_for_command(config, "r6-evidence-package")


def test_visa_preparation_manifest_hashes_archive_and_official_split(tmp_path: Path) -> None:
    archive = tmp_path / "visa.tar"
    split = tmp_path / "1cls.csv"
    archive.write_bytes(b"archive")
    split.write_bytes(b"split")
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
        )
    )
    config.data["external_benchmarks"] = {
        "visa": {
            "archive_path": str(archive),
            "archive_sha256": hashlib.sha256(b"archive").hexdigest(),
            "split_path": str(split),
            "split_sha256": hashlib.sha256(b"split").hexdigest(),
        }
    }

    records = configured_artifact_inventory(config, "prepare-visa-benchmark")

    assert [row["sha256"] for row in records] == [
        hashlib.sha256(b"archive").hexdigest(),
        hashlib.sha256(b"split").hexdigest(),
    ]


def test_posterior_training_manifest_hashes_calibration_inputs(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    reference = tmp_path / "references.jsonl"
    metadata.write_bytes(b"metadata")
    reference.write_bytes(b"references")
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
        )
    )
    config.data["posterior_calibration"] = {
        "sources": [
            {
                "dataset_id": "development",
                "metadata_path": str(metadata),
                "reference": {"kind": "manifest", "manifest_path": str(reference)},
            }
        ]
    }

    records = configured_artifact_inventory(config, "auto-mask-train-posterior")

    assert [row["config_key"] for row in records] == [
        "posterior_calibration.sources[0].metadata_path",
        "posterior_calibration.sources[0].reference.manifest_path",
    ]
    assert [row["sha256"] for row in records] == [
        hashlib.sha256(b"metadata").hexdigest(),
        hashlib.sha256(b"references").hexdigest(),
    ]


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
    config.data["auto_masks"]["artifact_retention"] = "locked_evaluation"
    config.data["auto_masks"]["write_contact_sheets"] = False
    config.data["auto_masks"]["contact_sheet_max_rows"] = 12
    config.data["auto_masks"]["resume_incomplete"] = True
    config.data["auto_masks"]["generic_evidence"]["persist_target_evidence_cache"] = False
    assert architecture_core_fingerprint(config) == original
    config.data["auto_masks"]["max_images_per_target"] = 1
    assert architecture_core_fingerprint(config) != original
    config.data["auto_masks"].pop("max_images_per_target")
    config.data["auto_masks"]["generic_evidence"]["sam_positive_points"] = 7
    assert architecture_core_fingerprint(config) != original


def test_architecture_fingerprint_covers_posterior_model_content(tmp_path: Path) -> None:
    posterior = tmp_path / "posterior.joblib"
    posterior.write_bytes(b"posterior-v1")
    config = load_config(
        _config_path(
            tmp_path,
            governance=["  development_categories: [part]", "  locked_categories: []"],
            extra=[
                "auto_masks:",
                "  architecture: generic_evidence",
                f"  posterior_model_path: {posterior}",
            ],
        )
    )

    original = architecture_core_fingerprint(config)
    posterior.write_bytes(b"posterior-v2")

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
