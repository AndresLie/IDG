from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iadgen_v2.auto_mask.contracts import AutoMaskContext, CandidateProposal, EvidenceMap, SelectionDecision
from iadgen_v2.auto_mask.evidence import apply_soft_spatial_prior, fuse_evidence_maps
from iadgen_v2.auto_mask.evidence.base import FunctionalEvidenceProvider
from iadgen_v2.auto_mask.mask_roles import posterior_mask_roles
from iadgen_v2.auto_mask.pipeline import run_generic_evidence_pipeline
from iadgen_v2.auto_mask.proposals import MEASUREMENT_NAMES, generate_generic_proposals
from iadgen_v2.auto_mask.selection import GenericCandidateSelector, _iou_from_precision_recall, fit_selector_bundle
from iadgen_v2.auto_masks import reselect_auto_masks
from iadgen_v2.config import load_config
from iadgen_v2.governance import (
    architecture_core_fingerprint,
    dataset_inventory,
    finalize_experiment_manifest,
    write_experiment_manifest,
)
from iadgen_v2.locked_evaluation import run_locked_evaluation


class _FixedProvider:
    def __init__(self, name: str, values: np.ndarray, reliability: float = 0.8) -> None:
        self.name = name
        self.values = values
        self.reliability = reliability

    def compute(self, context: AutoMaskContext) -> EvidenceMap:
        return EvidenceMap(self.name, self.values, self.reliability, augmentation_consistency=0.9)


def test_soft_spatial_prior_does_not_erase_outside_evidence() -> None:
    values = np.ones((20, 20), dtype=np.float32)
    weighted = apply_soft_spatial_prior(values, ((5, 5, 15, 15),))
    assert weighted[10, 10] == pytest.approx(1.0)
    assert weighted[0, 0] == pytest.approx(0.2)
    assert weighted[4, 4] == pytest.approx(0.5)


def test_soft_policy_keeps_selected_conservative_eval_proposal() -> None:
    fused = np.zeros((12, 12), dtype=np.float32)
    fused[3:9, 3:9] = 0.6
    fused[5:7, 5:7] = 0.9
    selected_mask = np.zeros_like(fused, dtype=bool)
    selected_mask[3:9, 3:9] = True
    evidence = [EvidenceMap(source="test", values=fused, reliability=1.0)]
    proposal = CandidateProposal(mode="selected", mask=selected_mask, score=0.5)
    decision = SelectionDecision(
        selected_mode="selected",
        disposition="soft_mask_only",
        expected_iou=0.4,
        expected_precision=0.6,
        expected_recall=0.5,
        conformal_iou_lower_bound=0.3,
        source_disagreement=0.3,
        confidence=0.7,
    )

    roles = posterior_mask_roles(
        fused,
        np.zeros_like(fused),
        evidence,
        proposal,
        decision,
    )

    assert np.array_equal(roles["eval_tight"] > 0.5, selected_mask)
    assert np.count_nonzero(roles["positive_core"]) < np.count_nonzero(roles["eval_tight"])


def test_selector_iou_projection_is_consistent_with_precision_and_recall() -> None:
    assert _iou_from_precision_recall(0.8, 0.5) == pytest.approx(0.4444444)
    assert _iou_from_precision_recall(0.0, 0.8) == 0.0


def test_functional_provider_calibrates_from_held_out_normals(tmp_path: Path) -> None:
    defect = tmp_path / "defect.png"
    normals = tuple(tmp_path / f"normal_{index}.png" for index in range(3))
    Image.new("RGB", (16, 16), (180, 180, 180)).save(defect)
    for index, path in enumerate(normals):
        Image.new("RGB", (16, 16), (100 + index, 100 + index, 100 + index)).save(path)

    regions: list[tuple[int, int, int, int]] = []

    def builder(image: Image.Image, references: list[Path], region: tuple[int, int, int, int]):
        regions.append(region)
        value = float(np.asarray(image, dtype=np.float32).mean() / 255.0)
        return np.full((16, 16), value, dtype=np.float32), {"reference_count": len(references)}

    context = AutoMaskContext(
        image_path=defect,
        normal_paths=normals,
        image_size=(16, 16),
        semantic_regions=((0, 0, 16, 16),),
    )
    provider = FunctionalEvidenceProvider("test", builder, artifact_dir=tmp_path / "cache")
    evidence = provider.compute(context)
    cached = provider.compute(context)

    assert evidence.metadata["calibration_source"] == "leave_one_normal_out"
    assert evidence.calibration["median"] < 0.5
    assert cached.metadata["cache_hit"] is True
    assert np.array_equal(cached.values, evidence.values)
    assert set(regions) == {(0, 0, 16, 16)}


def test_generic_proposals_selector_and_pipeline_write_contract_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "defect.png"
    normal_path = tmp_path / "normal.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(image_path)
    Image.new("RGB", (64, 64), (120, 120, 120)).save(normal_path)
    first = np.full((64, 64), 0.08, dtype=np.float32)
    second = np.full((64, 64), 0.12, dtype=np.float32)
    first[20:40, 22:42] = 0.95
    second[21:41, 21:41] = 0.90
    context = AutoMaskContext(
        image_path=image_path,
        normal_paths=(normal_path,),
        image_size=(64, 64),
        semantic_regions=((12, 12, 52, 52),),
        foreground=np.ones((64, 64), dtype=bool),
        cache_key="test",
    )
    artifacts = run_generic_evidence_pipeline(
        context,
        [_FixedProvider("first", first), _FixedProvider("second", second)],
        output_dir=tmp_path / "masks",
        variant_dir=tmp_path / "variants",
        artifact_stem="sample",
        selector=GenericCandidateSelector(),
        min_component_area=4,
    )
    assert artifacts["parameters"]["architecture"] == "v3-generic-evidence"
    assert artifacts["selection_decision"]["disposition"] == "hard_mask_ok"
    assert artifacts["candidate_measurements"]
    assert artifacts["candidate_predictions"]
    for name in ("eval_tight", "training_soft", "positive_core", "possible_region", "uncertainty_map", "inpaint_soft"):
        assert Path(artifacts["mask_variant_paths"][name]).exists()


def test_selector_bundle_uses_leave_category_out_training(tmp_path: Path) -> None:
    rows = []
    for category_index, category in enumerate(("a", "b", "c")):
        for index in range(24):
            quality = (index + 1) / 25.0
            row = {name: quality * (0.7 + 0.05 * category_index) for name in MEASUREMENT_NAMES}
            row.update({"category": category, "iou": quality, "precision": min(1.0, quality + 0.05), "recall": max(0.0, quality - 0.05)})
            rows.append(row)
    path = fit_selector_bundle(rows, tmp_path / "selector.joblib")
    assert path.exists()
    selector = GenericCandidateSelector(path)
    assert selector.bundle is not None
    assert set(selector.bundle["development_categories"]) == {"a", "b", "c"}


def test_sha_dataset_fingerprint_ignores_mtime(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, targets="part: [defect]", development="[part]", locked="[locked]"))
    image = config.dataset_root / "part" / "train" / "good" / "000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"stable-content")
    first = dataset_inventory(config)
    stat = image.stat()
    os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    second = dataset_inventory(config)
    assert first["inventory_fingerprint"] == second["inventory_fingerprint"]
    assert first["metadata_fingerprint"] != second["metadata_fingerprint"]


def test_manifest_finalization_records_status_and_output_hash(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, targets="part: [defect]", development="[part]", locked="[locked]"))
    path = write_experiment_manifest(config, "auto-masks")
    output = tmp_path / "result.txt"
    output.write_text("done", encoding="utf-8")
    started = time.monotonic()
    finalize_experiment_manifest(path, status="succeeded", started_monotonic=started, output_paths=(output,))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["outputs"][0]["sha256"]
    assert manifest["resources"]["peak_process_rss_bytes"] > 0


def test_locked_evaluation_is_isolated_and_sealed(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, targets="locked: [defect]", development="[]", locked="[locked]"))
    config.data["auto_masks"] = {"architecture": "generic_evidence"}
    prediction_path = tmp_path / "runtime" / "prediction.png"
    prediction_path.parent.mkdir(parents=True)
    prediction = np.zeros((32, 32), dtype=np.uint8)
    prediction[10:20, 12:22] = 255
    Image.fromarray(prediction).save(prediction_path)
    runtime_manifest = tmp_path / "runtime" / "metadata.jsonl"
    runtime_manifest.write_text(json.dumps({"category": "locked", "defect_type": "defect", "image_path": "000.png", "eval_mask_path": str(prediction_path), "region_xyxy": [8, 8, 24, 24], "settings": {"selection_decision": {"disposition": "hard_mask_ok", "confidence": 0.8}}}) + "\n", encoding="utf-8")
    official_root = tmp_path / "official"
    official_root.mkdir()
    truth_path = official_root / "000_mask.png"
    Image.fromarray(prediction).save(truth_path)
    reference_manifest = official_root / "references.jsonl"
    reference_manifest.write_text(json.dumps({"category": "locked", "defect_type": "defect", "image_path": "000.png", "official_mask_path": str(truth_path)}) + "\n", encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    config.data["research_governance"]["frozen_architecture_manifest"] = str(freeze_path)
    freeze_path.write_text(
        json.dumps(
            {
                "go": True,
                "architecture_tag": "test-architecture",
                "architecture_core_fingerprint": architecture_core_fingerprint(config),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    latest = config.output_dir / "experiment_manifests" / "auto-masks" / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "architecture_tag": "test-architecture",
                "outputs": [
                    {
                        "path": str(runtime_manifest.resolve()),
                        "sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_payload = runtime_manifest.read_text(encoding="utf-8")
    report = run_locked_evaluation(config, runtime_manifest=runtime_manifest, reference_manifest=reference_manifest)
    assert "Category-macro Dice: `1.0000`" in report.read_text(encoding="utf-8")
    runtime_manifest.write_text(runtime_payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an unchanged output"):
        run_locked_evaluation(config, runtime_manifest=runtime_manifest, reference_manifest=reference_manifest)
    runtime_manifest.write_text(runtime_payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="already sealed"):
        run_locked_evaluation(config, runtime_manifest=runtime_manifest, reference_manifest=reference_manifest)


def test_generic_path_rejects_legacy_cached_candidate_reselection(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, targets="part: [defect]", development="[part]", locked="[]"))
    config.data["auto_masks"] = {"architecture": "generic_evidence"}

    with pytest.raises(ValueError, match="restricted to the frozen legacy specialist path"):
        reselect_auto_masks(config)


def _config(tmp_path: Path, *, targets: str, development: str, locked: str) -> Path:
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
                f"    {targets}",
                "generation: {}",
                "models: {}",
                "evaluation: {}",
                "research_governance:",
                "  enabled: true",
                "  architecture_tag: test-architecture",
                "  experiment_track: test",
                f"  development_categories: {development}",
                f"  locked_categories: {locked}",
                f"  locked_official_mask_roots: [{tmp_path / 'official'}]",
                "  enforce_official_mask_isolation: true",
                "  dataset_fingerprint_mode: sha256",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
