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
from iadgen_v2.auto_mask.refinement import EdgeAwareRefiner, edge_align_field, guided_filter
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


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    union = int((first | second).sum())
    return float((first & second).sum() / union) if union else 0.0


def test_guided_filter_sharpens_a_blurred_edge_toward_the_guide() -> None:
    import cv2

    guide = np.zeros((24, 24), dtype=np.float32)
    guide[:, 12:] = 1.0  # sharp guide edge at column 12
    src = cv2.GaussianBlur(guide, (0, 0), sigmaX=4.0)  # same edge, badly blurred
    output = guided_filter(guide, src, radius=6, eps=1e-4)
    # The guided output tracks the guide's sharp transition, so its steepest
    # horizontal gradient exceeds the blurred source's.
    src_gradient = float(np.abs(np.diff(src.mean(axis=0))).max())
    output_gradient = float(np.abs(np.diff(output.mean(axis=0))).max())
    assert output_gradient > src_gradient


def test_edge_refiner_snaps_blurry_proposal_to_the_image_edge() -> None:
    height = width = 48
    image_array = np.zeros((height, width, 3), dtype=np.uint8)
    image_array[:, :24] = 45
    image_array[:, 24:] = 205  # sharp vertical edge at column 24
    image = Image.fromarray(image_array, "RGB")

    truth = np.zeros((height, width), dtype=bool)
    truth[12:36, 24:40] = True  # true defect sits entirely on the bright side

    import cv2

    fused = cv2.GaussianBlur(truth.astype(np.float32), (0, 0), sigmaX=6.0)
    fused = fused / float(fused.max())
    proposal = fused >= 0.5

    aligned = edge_align_field(image, fused, radius=6)
    candidates = EdgeAwareRefiner(aligned, search_radius=6)(fused, proposal, (0, 0, width, height))

    assert candidates, "edge refiner should propose at least one variant"
    wrong_side = np.arange(width)[None, :] < 24
    best = max(candidates, key=lambda mask: _iou(mask, truth))

    # The refined boundary hugs the real image edge: higher IoU and no pixels
    # bleeding across the edge into the dark (non-defect) side.
    assert _iou(best, truth) > _iou(proposal, truth)
    assert int((best & wrong_side).sum()) < int((proposal & wrong_side).sum())


def test_edge_refinement_also_snaps_sam_outputs() -> None:
    import cv2

    height = width = 48
    probability = np.zeros((height, width), dtype=np.float32)
    probability[14:34, 26:42] = 0.9
    probability = cv2.GaussianBlur(probability, (0, 0), sigmaX=2.0)
    evidence = [EvidenceMap(source="x", values=np.clip(probability, 0.0, 1.0), reliability=0.8)]
    fused, disagreement, _ = fuse_evidence_maps(evidence)

    guide = np.zeros((height, width), dtype=np.float32)
    guide[:, 24:] = 1.0
    aligned = np.clip(0.5 * fused + 0.5 * (fused * guide), 0.0, 1.0).astype(np.float32)

    def fake_sam(_fused, mask, _region):
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
        return [grown]

    proposals = generate_generic_proposals(
        fused,
        disagreement,
        evidence,
        (0, 0, width, height),
        sam_refiner=fake_sam,
        edge_refiner=EdgeAwareRefiner(aligned, search_radius=6),
    )
    modes = [proposal.mode for proposal in proposals]
    assert any(mode.startswith("sam2_") for mode in modes)
    assert any(mode.startswith("edge_fused_") for mode in modes)
    # The SAM output is itself edge-snapped, not left as a raw dilated box.
    assert any(mode.startswith("edge_sam2_") for mode in modes)


def test_nearest_distance_cpu_path_matches_numpy() -> None:
    from iadgen_v2.auto_mask.evidence.foundation import _nearest_distance, _nearest_distance_numpy

    rng = np.random.default_rng(3)
    def norm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    target = norm(rng.standard_normal((64, 32)).astype(np.float32))
    memory = norm(rng.standard_normal((500, 32)).astype(np.float32))
    # Explicit CPU must produce exactly the NumPy result regardless of GPU presence.
    assert np.allclose(_nearest_distance(target, memory, use_gpu=False), _nearest_distance_numpy(target, memory), atol=1e-6)


def test_atomic_token_save_roundtrips_without_leftover_temp(tmp_path: Path) -> None:
    from iadgen_v2.auto_mask.evidence.foundation import MultiScaleDinoProvider

    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    target = tmp_path / "sub" / "tokens.npy"
    MultiScaleDinoProvider._atomic_save(target, array)
    assert target.exists()
    assert np.array_equal(np.load(target), array)
    assert not list(target.parent.glob("*.tmp"))


def test_edge_refiner_is_safe_on_empty_proposal() -> None:
    aligned = np.zeros((16, 16), dtype=np.float32)
    assert EdgeAwareRefiner(aligned)(aligned, np.zeros((16, 16), dtype=bool), (0, 0, 16, 16)) == []


def _proposal(mode: str, coverage: float, agreement: float, area: float) -> CandidateProposal:
    mask = np.zeros((16, 16), dtype=bool)
    span = max(1, int(round((area * 256) ** 0.5)))
    mask[:span, :span] = True
    return CandidateProposal(
        mode=mode,
        mask=mask,
        score=0.0,
        measurements={
            "evidence_coverage": coverage,
            "source_agreement": agreement,
            "source_disagreement": 0.05,
            "area_fraction": area,
        },
    )


def test_edge_candidate_needs_margin_to_displace_non_edge() -> None:
    selector = GenericCandidateSelector(edge_swap_margin=0.05)
    # Edge candidate is only marginally "better" by the heuristic score.
    non_edge = _proposal("fused_q850", coverage=0.60, agreement=0.60, area=0.05)
    edge_tie = _proposal("edge_fused_q850_1", coverage=0.61, agreement=0.60, area=0.05)
    decision, _ = selector.select([non_edge, edge_tie])
    # Near-tie: the in-distribution non-edge candidate is kept.
    assert decision.selected_mode == "fused_q850"

    # A clearly better edge candidate crosses the margin and is selected.
    edge_strong = _proposal("edge_fused_q850_1", coverage=0.95, agreement=0.95, area=0.06)
    decision2, _ = selector.select([non_edge, edge_strong])
    assert decision2.selected_mode == "edge_fused_q850_1"


class _ColumnModel:
    """Stub regressor returning a chosen feature column (index 0 = evidence_coverage)."""

    def predict(self, features):
        return np.asarray(features, dtype=float)[:, 0]


class _ConstModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


class _IdentityCalibrator:
    def predict(self, values):
        return np.asarray(values, dtype=float)


def _paired_bundle(predicted_gain: float) -> dict:
    return {
        "schema_version": 1,
        "iou_model": _ColumnModel(),
        "precision_model": _ColumnModel(),
        "recall_model": _ColumnModel(),
        "iou_calibrator": _IdentityCalibrator(),
        "conformal_residual_q90": 0.0,
        "paired_model": _ConstModel(predicted_gain),
        "paired_residual_q90": 0.05,
    }


def _edge_of(parent: CandidateProposal, coverage: float) -> CandidateProposal:
    mask = np.zeros((16, 16), dtype=bool)
    mask[:4, :4] = True
    return CandidateProposal(
        mode=f"edge_{parent.mode}_1",
        mask=mask,
        score=0.0,
        measurements={"evidence_coverage": coverage, "source_agreement": coverage, "source_disagreement": 0.05, "area_fraction": 0.05},
        parent_mode=parent.mode,
    )


def test_paired_model_gates_edge_by_predicted_gain() -> None:
    non_edge = _proposal("fused_q850", coverage=0.5, agreement=0.5, area=0.05)
    edge = _edge_of(non_edge, coverage=0.9)  # would rank higher by predicted IoU

    selector = GenericCandidateSelector()
    selector.bundle = _paired_bundle(predicted_gain=0.20)  # 0.20 - 0.05 > 0 -> eligible
    decision, _ = selector.select([non_edge, edge])
    assert decision.selected_mode == "edge_fused_q850_1"

    selector.bundle = _paired_bundle(predicted_gain=0.02)  # 0.02 - 0.05 < 0 -> ineligible
    decision2, _ = selector.select([non_edge, edge])
    assert decision2.selected_mode == "fused_q850"


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


def test_subspace_pca_residual_flags_off_subspace_patches() -> None:
    from iadgen_v2.auto_mask.evidence.foundation import SubspacePcaDinoProvider

    rng = np.random.default_rng(0)
    # normal patches lie in a 3D subspace of a 16D space; anomalies get an
    # orthogonal off-subspace component -> large reconstruction residual.
    basis = np.linalg.qr(rng.standard_normal((16, 3)))[0]
    def normal_block(n):
        return (rng.standard_normal((n, 3)) @ basis.T).astype(np.float32)
    normals = [normal_block(64) for _ in range(4)]
    target = normal_block(50)
    off = target.copy()
    off[:10] += 5.0 * rng.standard_normal((10, 16)).astype(np.float32)  # off-subspace anomalies

    prov = SubspacePcaDinoProvider(variance=0.95)
    res_clean = prov._reconstruction_residual(np.concatenate(normals), target)
    res_anom = prov._reconstruction_residual(np.concatenate(normals), off)
    # the injected off-subspace patches must have far larger residual than clean ones
    assert res_anom[:10].mean() > res_clean.mean() + 1.0
    assert res_anom[10:].mean() < res_anom[:10].mean()


def test_subspace_pca_cache_reuse_and_invalidation(tmp_path: Path) -> None:
    from iadgen_v2.auto_mask.evidence import foundation
    from iadgen_v2.auto_mask.evidence.foundation import SubspacePcaDinoProvider

    foundation._PCA_SUBSPACE_CACHE.clear()
    rng = np.random.default_rng(1)
    normals = [rng.standard_normal((32, 8)).astype(np.float32) for _ in range(3)]
    paths = []
    for i in range(3):
        p = tmp_path / f"normal_{i}.png"
        Image.fromarray(np.full((4, 4), i, dtype=np.uint8)).save(p)
        paths.append(p)
    prov = SubspacePcaDinoProvider(variance=0.9)

    key = prov._subspace_key(tuple(paths))
    _, _, hit_first = prov._cached_subspace(normals, key)
    pca_a, _, hit_second = prov._cached_subspace(normals, key)
    assert hit_first is False and hit_second is True  # reused within the process

    # Invalidation: a changed normal file (different size) yields a different key,
    # and a different parameter (variance) also yields a different key.
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(paths[0])
    assert prov._subspace_key(tuple(paths)) != key
    assert SubspacePcaDinoProvider(variance=0.95)._subspace_key(tuple(paths)) != key
    _, _, hit_after_change = prov._cached_subspace(normals, prov._subspace_key(tuple(paths)))
    assert hit_after_change is False  # stale entry not reused


def test_subspace_pca_loo_calibration_is_deterministic() -> None:
    from iadgen_v2.auto_mask.evidence.foundation import SubspacePcaDinoProvider

    rng = np.random.default_rng(2)
    normals = [rng.standard_normal((40, 8)).astype(np.float32) for _ in range(4)]
    prov = SubspacePcaDinoProvider(variance=0.9)
    first = prov._loo_calibration(normals)
    second = prov._loo_calibration([n.copy() for n in normals])
    assert first is not None and first["mad"] > 0
    assert first == second  # svd_solver="full" LOO is deterministic
