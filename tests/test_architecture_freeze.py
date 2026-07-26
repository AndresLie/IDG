from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from iadgen_v2.architecture_freeze import reseal_behavior_equivalent_architecture
from iadgen_v2.config import AppConfig


def test_operational_reseal_requires_exact_runtime_equivalence(tmp_path: Path) -> None:
    config, baseline_path, candidate_path = _fixture(tmp_path)

    output_path = reseal_behavior_equivalent_architecture(config)
    frozen = json.loads(output_path.read_text(encoding="utf-8"))
    equivalence = json.loads(
        config.resolve_path(frozen["operational_equivalence_manifest"]).read_text(encoding="utf-8")
    )

    assert frozen["go"] is True
    assert frozen["architecture_tag"] == "rc2"
    assert frozen["selector_sha256"] == hashlib.sha256(b"selector").hexdigest()
    assert equivalence["samples_compared"] == 1
    assert equivalence["mask_roles_compared"] == 2
    assert not any(equivalence["differences"].values())
    assert equivalence["official_masks_opened"] is False

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["settings"]["selection_decision"]["selected_mode"] = "different"
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection_decision"):
        reseal_behavior_equivalent_architecture(config)

    assert baseline_path.is_file()


def _fixture(tmp_path: Path) -> tuple[AppConfig, Path, Path]:
    selector = tmp_path / "selector.joblib"
    selector.write_bytes(b"selector")
    parent = tmp_path / "rc1.json"
    parent.write_text(
        json.dumps(
            {
                "go": True,
                "architecture_tag": "rc1",
                "approval_scope": "locked_confirmation_only",
                "selector_sha256": hashlib.sha256(selector.read_bytes()).hexdigest(),
                "specialists": "disabled",
                "pca_subspace_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline_fused = baseline_dir / "fused.png"
    candidate_fused = candidate_dir / "fused.png"
    baseline_eval = baseline_dir / "eval.png"
    candidate_eval = candidate_dir / "eval.png"
    baseline_soft = baseline_dir / "soft.png"
    candidate_soft = candidate_dir / "soft.png"
    for path, content in (
        (baseline_fused, b"fused"),
        (candidate_fused, b"fused"),
        (baseline_eval, b"eval"),
        (candidate_eval, b"eval"),
        (baseline_soft, b"soft"),
        (candidate_soft, b"soft"),
    ):
        path.write_bytes(content)

    baseline_path = baseline_dir / "metadata.jsonl"
    candidate_path = candidate_dir / "metadata.jsonl"
    baseline_path.write_text(
        json.dumps(_row(baseline_fused, baseline_eval, baseline_soft)) + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_row(candidate_fused, candidate_eval, candidate_soft)) + "\n",
        encoding="utf-8",
    )
    config = AppConfig(
        path=tmp_path / "config.yaml",
        data={
            "project": {
                "output_dir": str(tmp_path / "outputs"),
                "report_dir": str(tmp_path / "reports"),
            },
            "dataset": {
                "root": str(tmp_path / "dataset"),
                "targets": {"part": ["bad"]},
            },
            "auto_masks": {
                "architecture": "generic_evidence",
                "specialists": "disabled",
                "selector_model_path": str(selector),
                "generic_evidence": {"pca_subspace_enabled": False},
            },
            "research_governance": {
                "enabled": True,
                "architecture_tag": "rc2",
                "release_architecture_tag": "rc2",
                "parent_frozen_architecture_manifest": str(parent),
                "equivalence_baseline_metadata_path": str(baseline_path),
                "equivalence_candidate_metadata_path": str(candidate_path),
                "operational_equivalence_min_samples": 1,
                "locked_official_mask_roots": [str(tmp_path / "official")],
            },
        },
    )
    return config, baseline_path, candidate_path


def _row(fused: Path, eval_mask: Path, soft_mask: Path) -> dict[str, object]:
    return {
        "category": "part",
        "defect_type": "bad",
        "image_path": "/runtime/000.png",
        "region_xyxy": [1, 2, 8, 9],
        "mask_variant_paths": {
            "eval_tight": str(eval_mask),
            "training_soft": str(soft_mask),
        },
        "settings": {
            "selected_refinement": "fused_q95",
            "candidate_scores": {"fused_q95": 0.8},
            "candidate_measurements": {"fused_q95": {"area_fraction": 0.1}},
            "candidate_predictions": [{"mode": "fused_q95", "expected_iou": 0.5}],
            "candidate_modes": ["fused_q95"],
            "candidate_rejections": {},
            "selection_decision": {
                "selected_mode": "fused_q95",
                "disposition": "soft_mask_only",
            },
            "mask_parameters": {"fused_evidence_path": str(fused)},
        },
    }
