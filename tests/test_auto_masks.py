from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import iadgen_v2.auto_masks as auto_masks
from iadgen_v2.auto_masks import (
    _generic_candidate_comparison_entries,
    _prompt,
    musc_mutual_score_heatmap,
    patchcore_guided_heatmap,
    _select_mask_artifacts,
    _validated_or_fallback_region,
    parse_qwen_bbox_payload,
    reselect_auto_masks,
    run_auto_masks,
    support_constrained_fusion_mask,
    validate_bbox,
    write_delta_deno_bbox_masks,
    write_dinov2_memory_bbox_masks,
    write_fft_texture_suppression_bbox_masks,
    write_foundation_anomaly_field_bbox_masks,
    write_multi_scuff_fusion_bbox_masks,
    write_musc_mutual_score_bbox_masks,
    write_nearest_normal_residual_bbox_masks,
    write_patchcore_guided_bbox_masks,
    write_normal_residual_fusion_bbox_masks,
    write_normal_anomaly_bbox_masks,
    write_sam_heatmap_bbox_masks,
    write_structure_tensor_ridge_bbox_masks,
    write_linear_refined_bbox_masks,
    write_multi_linear_refined_bbox_masks,
    write_residual_refined_bbox_masks,
    write_soft_patch_bbox_masks,
)
from iadgen_v2.auto_mask.contracts import AutoMaskRecord
from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.masks import write_refined_bbox_masks
from iadgen_v2.qwen_provider import CachedQwenLocalizationExtractor, QwenAvailability


def test_parse_qwen_bbox_payload_accepts_json_and_rejects_malformed() -> None:
    parsed = parse_qwen_bbox_payload(
        '{"bbox_xyxy": [0.1, 0.2, 0.8, 0.6], "defect_type": "crack", "confidence": 0.7, "evidence": "thin dark line"}',
        (100, 50),
    )

    assert parsed["bbox_xyxy"] == (10, 10, 80, 30)
    assert parsed["defect_type"] == "crack"
    assert parsed["confidence"] == 0.7

    with pytest.raises(ValueError, match="JSON object"):
        parse_qwen_bbox_payload("bbox: 1,2,3,4", (100, 50))
    with pytest.raises(ValueError, match="missing bbox_xyxy"):
        parse_qwen_bbox_payload('{"defect_type": "scratch"}', (100, 50))


def test_generic_candidate_comparison_uses_current_candidate_families(tmp_path: Path) -> None:
    modes = [
        "fused_q850",
        "fused_q900",
        "edge_fused_q850_1",
        "edge_fused_q900_1",
        "sam2_fused_q850_1",
        "fused_q950_component_1",
    ]
    paths = {}
    for index, mode in enumerate(modes):
        candidate_path = tmp_path / f"{mode}.png"
        Image.new("L", (8, 8), 255 if index % 2 else 0).save(candidate_path)
        paths[mode] = str(candidate_path)
    row = AutoMaskRecord(
        category="bottle",
        defect_type="broken_small",
        image_path=str(tmp_path / "input.png"),
        mask_path=paths["fused_q850"],
        provider="qwen",
        prompt="",
        description="",
        qwen_text=None,
        qwen_defect_type=None,
        confidence=None,
        evidence=None,
        region_xyxy=(0, 0, 8, 8),
        box_mask_path=paths["fused_q850"],
        refined_mask_path=paths["fused_q850"],
        inpaint_mask_path=paths["fused_q850"],
        eval_mask_path=paths["fused_q850"],
        training_mask_path=paths["fused_q850"],
        uncertainty_mask_path=paths["fused_q850"],
        mask_variant_paths={},
        mask_variant_overlay_paths={},
        overlay_path=None,
        seed=1,
        settings={
            "selected_refinement": "fused_q850",
            "candidate_refined_paths": paths,
            "policy_scores": {mode: 1.0 - index * 0.1 for index, mode in enumerate(modes)},
            "candidate_scores": {mode: index * 0.1 for index, mode in enumerate(modes)},
        },
    )

    entries = _generic_candidate_comparison_entries(row)

    selected_modes = [mode for _, mode, _ in entries]
    assert selected_modes[0] == "fused_q850"
    assert len(selected_modes) == len(set(selected_modes)) == 6
    assert any(mode.startswith("edge_") for mode in selected_modes)
    assert any(mode.startswith("sam2_") for mode in selected_modes)


def test_parse_qwen_bbox_payload_recovers_truncated_json_like_output() -> None:
    parsed = parse_qwen_bbox_payload(
        '```json\n{"bbox_xyxy": [10, 20, 90, 40], "defect_type": "scratch", "confidence": 0.9, "evidence": "visible scratch',
        (100, 50),
    )

    assert parsed["bbox_xyxy"] == (10, 20, 90, 40)
    assert parsed["defect_type"] == "scratch"
    assert parsed["confidence"] == 0.9


def test_parse_qwen_bbox_payload_preserves_sub_boxes() -> None:
    parsed = parse_qwen_bbox_payload(
        json.dumps(
            {
                "bbox_xyxy": [10, 10, 90, 90],
                "sub_boxes_xyxy": [[12, 20, 40, 30], [50, 60, 80, 70]],
                "defect_type": "scratch",
            }
        ),
        (100, 100),
    )

    assert parsed["bbox_xyxy"] == (10, 10, 90, 90)
    assert parsed["sub_boxes_xyxy"] == [(12, 20, 40, 30), (50, 60, 80, 70)]


def test_qwen_localization_cache_replays_and_invalidates_inputs(tmp_path: Path) -> None:
    class CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, image: Image.Image, prompt: str) -> dict[str, object]:
            self.calls += 1
            return {"text": f"response-{self.calls}-{image.getpixel((0, 0))}-{prompt}"}

        def close(self) -> None:
            pass

    delegate = CountingExtractor()
    cached = CachedQwenLocalizationExtractor(
        delegate,  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        model_id="test/model",
        model_revision="revision-a",
        torch_dtype="float16",
    )
    image = Image.new("RGB", (8, 8), (10, 20, 30))

    first = cached.analyze(image, "locate defect")
    replay = cached.analyze(image, "locate defect")
    changed_prompt = cached.analyze(image, "locate anomaly")
    changed_image = image.copy()
    changed_image.putpixel((0, 0), (11, 20, 30))
    cached.analyze(changed_image, "locate defect")

    assert delegate.calls == 3
    assert replay["text"] == first["text"]
    assert first["localization_cache"]["cache_hit"] is False
    assert replay["localization_cache"]["cache_hit"] is True
    assert replay["localization_cache"]["response_sha256"] == first["localization_cache"]["response_sha256"]
    assert changed_prompt["localization_cache"]["cache_key"] != first["localization_cache"]["cache_key"]

    changed_revision = CachedQwenLocalizationExtractor(
        delegate,  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        model_id="test/model",
        model_revision="revision-b",
        torch_dtype="float16",
    )
    changed_revision.analyze(image, "locate defect")
    assert delegate.calls == 4


def test_qwen_localization_cache_rejects_corrupt_response(tmp_path: Path) -> None:
    class CountingExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, image: Image.Image, prompt: str) -> dict[str, object]:
            self.calls += 1
            return {"text": f"valid-{self.calls}"}

        def close(self) -> None:
            pass

    delegate = CountingExtractor()
    cached = CachedQwenLocalizationExtractor(
        delegate,  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        model_id="test/model",
        model_revision="revision-a",
        torch_dtype="auto",
    )
    image = Image.new("RGB", (4, 4), "white")
    first = cached.analyze(image, "prompt")
    cache_path = Path(str(first["localization_cache"]["cache_path"]))
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["response"]["text"] = "tampered"
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = cached.analyze(image, "prompt")

    assert delegate.calls == 2
    assert recovered["text"] == "valid-2"
    assert recovered["localization_cache"]["cache_hit"] is False
    assert "hash mismatch" in recovered["localization_cache"]["invalid_reason"]


def test_qwen_bbox_v2_prompt_requests_local_zipper_sub_boxes() -> None:
    prompt = _prompt(
        "zipper",
        "broken_teeth",
        (512, 512),
        {"qwen_prompt_style": "defect_localization_json_v2"},
        "find the missing or broken zipper teeth",
    )

    assert "sub_boxes_xyxy" in prompt
    assert "short local segment" in prompt
    assert "not the full vertical chain" in prompt
    assert "bbox height should usually be under 130px" in prompt


def test_qwen_bbox_v2_prompt_keeps_bottle_small_chip_local() -> None:
    prompt = _prompt(
        "bottle",
        "broken_small",
        (512, 512),
        {"qwen_prompt_style": "defect_localization_json_v2"},
        "find the small local rim chip",
    )

    assert "sub_boxes_xyxy" in prompt
    assert "small local rim chip" in prompt
    assert "do not box the entire circular rim" in prompt


def test_structure_profile_routes_common_industrial_shapes() -> None:
    assert auto_masks._structure_profile("bottle", "broken_large", "large broken rim area") == "ring_sector"
    assert auto_masks._structure_profile("bottle", "contamination", "surface stain") == "flat_surface_patch"
    assert auto_masks._structure_profile("zipper", "broken_teeth", "missing tooth in zipper chain") == "repeated_chain"
    assert auto_masks._structure_profile("zipper", "fabric_border", "frayed edge beside zipper") == "edge_border"
    assert auto_masks._structure_profile("wood", "scratch", "thin linear scratch") == "thin_linear"


def test_selection_policy_records_structure_profile_for_bottle_and_zipper() -> None:
    bottle_policy = auto_masks._selection_policy(
        category="bottle",
        defect_type="broken_large",
        region=(10, 10, 90, 90),
        image_size=(100, 100),
        description="large broken glass rim",
        candidates={},
    )
    zipper_policy = auto_masks._selection_policy(
        category="zipper",
        defect_type="broken_teeth",
        region=(35, 0, 65, 100),
        image_size=(100, 100),
        description="missing teeth in repeated zipper chain",
        candidates={},
    )

    assert bottle_policy["structure_profile"] == "ring_sector"
    assert zipper_policy["structure_profile"] == "repeated_chain"
    assert "patchcore_guided" in zipper_policy["mode_score_adjustments"]


def _candidate_for_arbitration(*, area: int, box_fraction: float, components: int, status: str = "pass", reasons: list[str] | None = None) -> dict:
    return {
        "score": 0.9,
        "qc": {
            "mask_area": area,
            "mask_to_qwen_box_fraction": box_fraction,
            "component_count": components,
            "status": status,
            "reasons": reasons or [],
        },
    }


def test_recall_safe_arbitration_prefers_compact_polar_small_chip() -> None:
    mode, _, arbitration = auto_masks._choose_candidate_with_arbitration(
        {
            "sam_prompt_regularized": _candidate_for_arbitration(area=13000, box_fraction=0.13, components=4),
            "polar_rim_residual": _candidate_for_arbitration(area=3500, box_fraction=0.033, components=2),
        },
        adjusted_scores={"sam_prompt_regularized": 1.5, "polar_rim_residual": 0.98},
        hierarchy=["sam_prompt_regularized", "polar_rim_residual"],
        policy={"selection_policy": "bottle_small_chip_sam_gate"},
    )

    assert mode == "polar_rim_residual"
    assert arbitration["reason"] == "compact_polar_rim_chip_over_broad_sam"


def test_recall_safe_arbitration_prefers_local_tooth_consensus() -> None:
    mode, _, arbitration = auto_masks._choose_candidate_with_arbitration(
        {
            "patchcore_guided": _candidate_for_arbitration(area=7000, box_fraction=0.13, components=1),
            "ensemble_consensus": _candidate_for_arbitration(area=7050, box_fraction=0.13, components=1),
        },
        adjusted_scores={"patchcore_guided": 1.43, "ensemble_consensus": 0.78},
        hierarchy=["patchcore_guided", "ensemble_consensus"],
        policy={
            "selection_policy": "zipper_tooth_chain_prefer_patchcore_normal_residual",
            "qwen_region_xyxy": (128, 272, 384, 480),
            "image_size": (512, 512),
        },
    )

    assert mode == "ensemble_consensus"
    assert arbitration["reason"] == "local_tooth_consensus_matches_patchcore_extent"


def test_recall_safe_arbitration_prefers_edge_normal_residual_fusion() -> None:
    mode, _, arbitration = auto_masks._choose_candidate_with_arbitration(
        {
            "patchcore_guided": _candidate_for_arbitration(area=7400, box_fraction=0.10, components=2),
            "normal_residual_fusion": _candidate_for_arbitration(
                area=8850,
                box_fraction=0.12,
                components=33,
                status="warning",
                reasons=["touches_image_border", "many_components"],
            ),
        },
        adjusted_scores={"patchcore_guided": 1.43, "normal_residual_fusion": 0.66},
        hierarchy=["patchcore_guided", "normal_residual_fusion"],
        policy={
            "selection_policy": "zipper_tooth_chain_prefer_patchcore_normal_residual",
            "qwen_region_xyxy": (128, 0, 384, 288),
            "image_size": (512, 512),
        },
    )

    assert mode == "normal_residual_fusion"
    assert arbitration["reason"] == "edge_split_teeth_preserve_normal_residual_recall"


def test_recall_safe_arbitration_prefers_long_tooth_agreement() -> None:
    mode, _, arbitration = auto_masks._choose_candidate_with_arbitration(
        {
            "patchcore_guided": _candidate_for_arbitration(area=10000, box_fraction=0.16, components=2),
            "zipper_tooth_agreement": _candidate_for_arbitration(area=8200, box_fraction=0.13, components=2),
        },
        adjusted_scores={"patchcore_guided": 1.43, "zipper_tooth_agreement": 0.80},
        hierarchy=["patchcore_guided", "zipper_tooth_agreement"],
        policy={
            "selection_policy": "zipper_tooth_chain_prefer_patchcore_normal_residual",
            "qwen_region_xyxy": (180, 0, 330, 512),
            "image_size": (512, 512),
        },
    )

    assert mode == "zipper_tooth_agreement"
    assert arbitration["reason"] == "long_tooth_region_patchcore_musc_agreement"


def test_recall_safe_arbitration_prefers_compact_fabric_border_consensus() -> None:
    mode, _, arbitration = auto_masks._choose_candidate_with_arbitration(
        {
            "sam2_heatmap": _candidate_for_arbitration(
                area=8800,
                box_fraction=0.30,
                components=3,
                status="warning",
                reasons=["touches_image_border"],
            ),
            "ensemble_consensus": _candidate_for_arbitration(
                area=7200,
                box_fraction=0.24,
                components=1,
                status="warning",
                reasons=["touches_image_border"],
            ),
        },
        adjusted_scores={"sam2_heatmap": 1.05, "ensemble_consensus": 0.82},
        hierarchy=["sam2_heatmap", "ensemble_consensus"],
        policy={"selection_policy": "zipper_fabric_border_prefer_sam_patchcore_border_support"},
    )

    assert mode == "ensemble_consensus"
    assert arbitration["reason"] == "border_touching_sam_prefers_compact_consensus"


def test_recall_safe_arbitration_prefers_bottle_contamination_recall_fusion() -> None:
    mode, _, arbitration = auto_masks._choose_candidate_with_arbitration(
        {
            "fft_texture_suppression": _candidate_for_arbitration(
                area=28648,
                box_fraction=0.38,
                components=1,
                status="warning",
                reasons=["large_image_area"],
            ),
            "bottle_contamination_recall_fusion": _candidate_for_arbitration(
                area=45098,
                box_fraction=0.59,
                components=10,
                status="warning",
                reasons=["large_image_area", "large_fraction_of_qwen_box"],
            ),
        },
        adjusted_scores={"fft_texture_suppression": 1.49, "bottle_contamination_recall_fusion": 1.14},
        hierarchy=["fft_texture_suppression", "bottle_contamination_recall_fusion"],
        policy={"selection_policy": "bottle_contamination_prefer_texture_or_sam"},
    )

    assert mode == "bottle_contamination_recall_fusion"
    assert arbitration["reason"] == "broad_fft_uses_coherent_sam_recall_fusion"


def test_repeated_chain_candidate_is_only_applicable_to_chain_profiles() -> None:
    assert auto_masks._candidate_mode_applicable(
        "repeated_chain_refiner",
        category="zipper",
        defect_type="broken_teeth",
    )
    assert not auto_masks._candidate_mode_applicable(
        "repeated_chain_refiner",
        category="zipper",
        defect_type="fabric_border",
    )
    assert not auto_masks._candidate_mode_applicable(
        "zipper_fabric_border_layout",
        category="bottle",
        defect_type="broken_small",
    )
    assert auto_masks._candidate_mode_applicable(
        "polar_rim_residual",
        category="bottle",
        defect_type="broken_small",
    )
    assert not auto_masks._candidate_mode_applicable(
        "polar_rim_residual",
        category="bottle",
        defect_type="broken_large",
    )
    assert auto_masks._candidate_mode_applicable(
        "repeated_chain_refiner",
        category="custom_fastener",
        defect_type="missing_periodic_chain_link",
        description="one link is missing from a repeated chain",
    )


def test_repeated_chain_refiner_keeps_local_anomaly_and_rejects_long_chain() -> None:
    heatmap = np.zeros((160, 96), dtype=np.float32)
    heatmap[:, 40:56] = 0.28
    heatmap[62:88, 39:57] = 1.0
    heatmap[10:24, 42:54] = 0.48
    heatmap[126:145, 42:54] = 0.46

    mask, params = auto_masks.repeated_chain_refiner_mask(
        heatmap,
        (20, 0, 76, 160),
        (96, 160),
        percentile=82.0,
        corridor_width_fraction=0.50,
        max_window_height=72,
        min_window_height=40,
        close_radius=1,
        dilate_radius=1,
        min_component_area=8,
        max_components=4,
        max_box_fraction=0.10,
    )

    arr = np.asarray(mask.convert("L")) > 0
    assert arr[68:84, 42:54].any()
    assert not arr[0:28].any()
    assert not arr[120:].any()
    assert params["repeated_chain_vertical_span_fraction"] < 0.55
    assert params["repeated_chain_explanation_score"] > 0.0
    assert params["repeated_chain_structure_score"] > 0.5


def test_repeated_chain_refiner_supports_diagonal_chain_orientation() -> None:
    heatmap = np.zeros((160, 160), dtype=np.float32)
    for offset in range(-5, 6):
        rows = np.arange(24, 136)
        columns = np.clip(rows + offset, 0, 159)
        heatmap[rows, columns] = 0.25
    for offset in range(-7, 8):
        rows = np.arange(70, 94)
        columns = np.clip(rows + offset, 0, 159)
        heatmap[rows, columns] = 1.0

    mask, params = auto_masks.repeated_chain_refiner_mask(
        heatmap,
        (12, 12, 148, 148),
        (160, 160),
        orientation_degrees=45.0,
        percentile=80.0,
        corridor_width_fraction=0.28,
        max_window_height=74,
        min_window_height=42,
        close_radius=1,
        dilate_radius=1,
        min_component_area=8,
        max_components=4,
        max_box_fraction=0.08,
    )

    arr = np.asarray(mask.convert("L")) > 0
    assert arr[74:92, 74:92].any()
    assert not arr[18:42, 18:42].any()
    assert not arr[120:145, 120:145].any()
    assert params["repeated_chain_orientation_degrees"] == 45.0
    assert params["repeated_chain_vertical_span_fraction"] < 0.60


def test_repeated_chain_gate_only_enables_for_valid_dense_full_chain(tmp_path: Path) -> None:
    dense = Image.new("L", (100, 240), 0)
    ImageDraw.Draw(dense).rectangle((40, 6, 62, 232), fill=255)
    dense_path = tmp_path / "dense_chain.png"
    dense.save(dense_path)
    candidates = {
        "patchcore_guided": {
            "refined_mask_path": str(dense_path),
            "score": 0.9,
            "qc": {"status": "pass", "mask_area": 1},
        }
    }

    enabled = auto_masks._repeated_chain_correction_gate(
        candidates,
        region=(20, 0, 80, 240),
        localization_status="valid",
    )
    fallback = auto_masks._repeated_chain_correction_gate(
        candidates,
        region=(20, 0, 80, 240),
        localization_status="fallback",
    )

    assert enabled["repeated_chain_gate_enabled"] is True
    assert enabled["repeated_chain_gate_reason"] == "dense_full_chain_overcoverage"
    assert fallback["repeated_chain_gate_enabled"] is False
    assert fallback["repeated_chain_gate_reason"] == "localization_not_valid"


def test_repeated_chain_gate_is_orientation_and_scale_normalized(tmp_path: Path) -> None:
    dense = Image.new("L", (240, 240), 0)
    draw = ImageDraw.Draw(dense)
    draw.line((28, 28, 212, 212), fill=255, width=24)
    dense_path = tmp_path / "diagonal_chain.png"
    dense.save(dense_path)

    result = auto_masks._repeated_chain_correction_gate(
        {
            "patchcore_guided": {
                "refined_mask_path": str(dense_path),
                "score": 0.9,
                "qc": {"status": "pass", "mask_area": 5000},
            }
        },
        region=(20, 20, 220, 220),
        localization_status="valid",
    )

    assert result["repeated_chain_gate_enabled"] is True
    assert 40.0 <= result["repeated_chain_gate_orientation_degrees"] <= 50.0
    assert result["repeated_chain_gate_axis_span_fraction"] >= 0.88
    assert result["repeated_chain_gate_region_scale_fraction"] >= 0.32


def test_zipper_tooth_structure_cleanup_removes_off_axis_spill() -> None:
    mask = Image.new("L", (120, 180), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((52, 12, 68, 166), fill=255)
    draw.rectangle((6, 66, 30, 114), fill=255)

    cleaned, params = auto_masks._zipper_structure_cleanup_eval_mask(
        mask,
        artifacts={"selected_refinement": "patchcore_guided", "candidate_heatmap_paths": {}},
        category="zipper",
        defect_type="broken_teeth",
        region=(0, 0, 120, 180),
        image_size=(120, 180),
        auto={
            "zipper_tooth_structure_min_box_fraction": 0.02,
            "zipper_tooth_structure_corridor_percentile": 80.0,
            "zipper_tooth_structure_max_width_fraction": 0.15,
            "zipper_tooth_structure_min_keep_fraction": 0.50,
            "min_component_area": 8,
        },
    )

    arr = np.asarray(cleaned.convert("L")) > 0
    assert params["zipper_structure_cleaned"] is True
    assert arr[30:150, 54:66].any()
    assert not arr[72:108, 8:28].any()
    assert params["zipper_structure_pixels"] < params["zipper_structure_original_pixels"]


def test_zipper_fabric_structure_cleanup_removes_central_tooth_spill() -> None:
    mask = Image.new("L", (100, 120), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((8, 10, 20, 110), fill=255)
    draw.rectangle((80, 10, 92, 110), fill=255)
    draw.rectangle((42, 10, 58, 110), fill=255)

    cleaned, params = auto_masks._zipper_structure_cleanup_eval_mask(
        mask,
        artifacts={"selected_refinement": "sam2_heatmap"},
        category="zipper",
        defect_type="fabric_border",
        region=(0, 0, 100, 120),
        image_size=(100, 120),
        auto={
            "zipper_fabric_layout_side_fractions": [0.05, 0.28, 0.72, 0.95],
            "zipper_fabric_structure_side_padding_fraction": 0.0,
            "zipper_fabric_structure_min_keep_fraction": 0.40,
            "min_component_area": 8,
        },
    )

    arr = np.asarray(cleaned.convert("L")) > 0
    assert params["zipper_structure_cleaned"] is True
    assert arr[20:100, 10:18].any()
    assert arr[20:100, 82:90].any()
    assert not arr[20:100, 46:54].any()


def test_polar_rim_residual_keeps_local_angular_anomaly() -> None:
    size = (160, 160)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    center = np.asarray([80.0, 80.0])
    radius = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    angle = (np.degrees(np.arctan2(yy - center[1], xx - center[0])) + 360.0) % 360.0
    heatmap = np.zeros((size[1], size[0]), dtype=np.float32)
    annulus = (radius >= 48) & (radius <= 68)
    heatmap[annulus] = 0.15
    heatmap[annulus & (angle >= 28) & (angle <= 48)] = 1.0

    mask, params = auto_masks.polar_rim_residual_mask(
        heatmap,
        (12, 12, 148, 148),
        size,
        angle_bins=180,
        inner_radius_fraction=0.52,
        outer_radius_fraction=1.02,
        profile_percentile=72.0,
        support_percentile=55.0,
        min_sector_degrees=10.0,
        max_sector_degrees=70.0,
        close_radius=1,
        dilate_radius=1,
        min_component_area=8,
        max_box_fraction=0.16,
    )

    arr = np.asarray(mask.convert("L")) > 0
    assert arr[104:132, 118:148].any()
    assert not arr[20:55, 65:95].any()
    assert 20.0 <= params["polar_rim_peak_degrees"] <= 60.0
    assert params["polar_rim_sector_degrees"] <= 70.0


def test_qwen_sub_boxes_can_be_selected_as_primary_region() -> None:
    region, selection = auto_masks._select_qwen_region_from_sub_boxes(
        (0, 0, 100, 100),
        [(40, 45, 55, 60)],
        (100, 100),
        {"qwen_sub_boxes_primary": True, "qwen_sub_box_padding": 0.0},
    )

    assert region == (40, 45, 55, 60)
    assert selection["sub_box_region_selected"] is True
    assert selection["sub_box_selection_mode"] == "primary"


def test_normal_guided_candidate_windows_rank_heatmap_bands() -> None:
    heatmap = np.zeros((128, 128), dtype=np.float32)
    heatmap[70:85, 40:80] = 1.0

    candidates = auto_masks._normal_guided_candidate_windows(
        heatmap,
        (128, 128),
        {
            "normal_guided_x_range_fraction": [0.25, 0.75],
            "normal_guided_window_height_px": 32,
            "normal_guided_window_stride_px": 16,
            "normal_guided_candidate_count": 3,
            "normal_guided_window_padding_y_px": 0,
        },
    )

    assert candidates
    best = max(candidates, key=lambda item: float(item["score"]))
    assert best["region_xyxy"][1] <= 80 <= best["region_xyxy"][3]


def test_parse_qwen_candidate_ids_accepts_json_and_text() -> None:
    assert auto_masks._parse_qwen_candidate_ids('{"candidate_ids": [2, 4]}', 5) == [2, 4]
    assert auto_masks._parse_qwen_candidate_ids("Select C3 and C1.", 4) == [3, 1]


def test_bottle_rim_guided_region_uses_configured_rim_prior() -> None:
    region, diagnostics = auto_masks.bottle_rim_guided_qwen_region(
        image_size=(100, 100),
        qwen_region=(1, 1, 5, 5),
        defect_type="broken_small",
        auto={"bottle_small_rim_region_fraction": [0.1, 0.2, 0.8, 0.9]},
    )

    assert region == (10, 20, 80, 90)
    assert diagnostics["model"] == "bottle_rim_guided_qwen"
    assert diagnostics["union_qwen"] is False


def test_bottle_surface_guided_region_ignores_tiny_wrong_qwen_by_default() -> None:
    region, diagnostics = auto_masks.bottle_surface_guided_region(
        image_size=(100, 100),
        qwen_region=(2, 80, 6, 84),
        auto={"bottle_surface_region_fraction": [0.25, 0.25, 0.75, 0.75]},
    )

    assert region == (25, 25, 75, 75)
    assert diagnostics["qwen_union_used"] is False


def test_auto_localization_policy_routes_zipper_teeth_to_normal_guided() -> None:
    image = Image.new("RGB", (128, 128), "gray")
    draw = ImageDraw.Draw(image)
    for x in range(42, 86, 4):
        draw.line((x, 0, x, 127), fill="black", width=1)

    policy, diagnostics = auto_masks.resolve_auto_localization_policy(
        auto={"localization_policy": "auto"},
        category="zipper",
        defect_type="broken_teeth",
        description="one or more zipper teeth are broken or missing",
        image=image,
        qwen_region=(40, 0, 88, 128),
        localization={"status": "valid"},
    )

    assert policy == "normal_guided_qwen_verify"
    assert diagnostics["router_mode"] == "auto"


def test_auto_localization_policy_keeps_zipper_fabric_border_on_qwen_bbox() -> None:
    image = Image.new("RGB", (128, 128), "gray")
    draw = ImageDraw.Draw(image)
    for x in range(42, 86, 4):
        draw.line((x, 0, x, 127), fill="black", width=1)

    policy, diagnostics = auto_masks.resolve_auto_localization_policy(
        auto={"localization_policy": "auto"},
        category="zipper",
        defect_type="fabric_border",
        description="frayed damaged fabric border beside the zipper teeth",
        image=image,
        qwen_region=(8, 0, 46, 128),
        localization={"status": "valid"},
    )

    assert policy == "qwen_bbox"
    assert diagnostics["router_mode"] == "auto"


def test_auto_localization_policy_routes_bottle_chip_to_rim_prior() -> None:
    policy, diagnostics = auto_masks.resolve_auto_localization_policy(
        auto={"localization_policy": "auto"},
        category="bottle",
        defect_type="broken_small",
        description="small chipped missing glass on the bottle rim",
        image=Image.new("RGB", (128, 128), "black"),
        qwen_region=(20, 20, 80, 80),
        localization={"status": "valid"},
    )

    assert policy == "bottle_rim_guided_qwen"
    assert diagnostics["router_mode"] == "auto"


def test_auto_localization_policy_routes_bottle_contamination_to_surface_prior() -> None:
    policy, diagnostics = auto_masks.resolve_auto_localization_policy(
        auto={"localization_policy": "auto"},
        category="bottle",
        defect_type="contamination",
        description="visible contamination stain or foreign material on the bottle surface",
        image=Image.new("RGB", (128, 128), "black"),
        qwen_region=(5, 100, 12, 110),
        localization={"status": "valid"},
    )

    assert policy == "bottle_surface_guided"
    assert diagnostics["router_mode"] == "auto"


def test_auto_localization_policy_keeps_unknown_case_on_qwen_bbox() -> None:
    policy, diagnostics = auto_masks.resolve_auto_localization_policy(
        auto={"localization_policy": "auto"},
        category="wood",
        defect_type="scratch",
        description="thin scratch on wooden surface",
        image=Image.new("RGB", (128, 128), "gray"),
        qwen_region=(20, 20, 80, 80),
        localization={"status": "valid"},
    )

    assert policy == "qwen_bbox"
    assert diagnostics["router_mode"] == "auto"


def test_validate_bbox_clips_and_rejects_bad_area() -> None:
    assert validate_bbox((-5, 3, 110, 40), (100, 50), min_area_ratio=0.001, max_area_ratio=0.95) == (0, 3, 100, 40)

    with pytest.raises(ValueError, match="below minimum"):
        validate_bbox((1, 1, 2, 2), (100, 100), min_area_ratio=0.01, max_area_ratio=0.95)
    with pytest.raises(ValueError, match="above maximum"):
        validate_bbox((0, 0, 100, 100), (100, 100), min_area_ratio=0.001, max_area_ratio=0.5)


def test_localization_fallback_preserves_batch_when_qwen_returns_full_frame() -> None:
    region, diagnostics = _validated_or_fallback_region(
        (0, 0, 100, 100),
        (100, 100),
        auto={
            "min_box_area_ratio": 0.001,
            "max_box_area_ratio": 0.5,
            "allow_localization_fallback": True,
            "localization_fallback_mode": "full_image",
        },
    )

    assert region == (0, 0, 100, 100)
    assert diagnostics["status"] == "fallback"
    assert diagnostics["fallback_mode"] == "full_image"
    assert "above maximum" in diagnostics["failure_reason"]


def test_refined_bbox_masks_are_non_rectangular(tmp_path: Path) -> None:
    image = Image.new("RGB", (64, 64), (120, 120, 120))
    artifacts = write_refined_bbox_masks(
        image,
        "custom_part",
        "scratch",
        (8, 18, 56, 34),
        tmp_path,
        "sample",
        13,
        clip_to_surface=False,
    )

    box = np.asarray(Image.open(artifacts["box_mask_path"]).convert("L"), dtype=np.uint8) > 0
    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert refined.any()
    assert int(refined.sum()) < int(box.sum())
    assert not np.array_equal(box, refined)


def test_auto_masks_use_description_guided_prompt_and_pixel_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    config.data["auto_masks"]["description_by_target"] = {"custom_part/scratch": "a thin dark horizontal scratch"}
    config.data["auto_masks"]["mask_refinement"] = "pixel"
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)

    metadata_path = run_auto_masks(config)
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["description"] == "a thin dark horizontal scratch"
    assert "a thin dark horizontal scratch" in rows[0]["prompt"]
    assert rows[0]["settings"]["mask_refinement"] == "pixel"
    assert rows[0]["settings"]["mask_parameters"]["kind"] == "pixel_contrast"


def test_auto_masks_use_per_image_description_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fixture_config(tmp_path)
    config.data["auto_masks"]["description_by_target"] = {
        "custom_part/scratch": {
            0: "one long narrow vertical scratch",
            "001": "a broad scuffed patch",
        }
    }
    config.data["auto_masks"]["mask_refinement"] = "pixel"
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)

    metadata_path = run_auto_masks(config)
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["description"] == "one long narrow vertical scratch"
    assert rows[1]["description"] == "a broad scuffed patch"
    assert rows[2]["description"] == "scratch"


def test_auto_masks_overwrite_replays_qwen_localization_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)
    first_path = run_auto_masks(config)
    first_rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]

    class NoAnalyzeExtractor(_FakeQwenExtractor):
        def analyze(self, image: Image.Image, prompt: str) -> dict[str, object]:
            raise AssertionError("Qwen generation must be bypassed by localization cache replay")

    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", NoAnalyzeExtractor)
    replay_path = run_auto_masks(config)
    replay_rows = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]

    assert [row["qwen_text"] for row in replay_rows] == [row["qwen_text"] for row in first_rows]
    assert all(row["settings"]["qwen_localization"]["qwen_cache"]["cache_hit"] is True for row in replay_rows)
    assert all(row["settings"]["qwen_localization"]["qwen_cache"]["response_sha256"] for row in replay_rows)


def test_auto_masks_write_mvtec_ground_truth_and_prepare_can_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)

    metadata_path = run_auto_masks(config)
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    current_run = json.loads(
        (config.output_dir / "auto_masks" / "qwen" / "current_run.json").read_text(encoding="utf-8")
    )
    run_output_dir = Path(current_run["run_output_dir"])

    assert len(rows) == 3
    assert current_run["status"] == "complete"
    assert run_output_dir.is_dir()
    assert Path(current_run["run_metadata_path"]).read_bytes() == metadata_path.read_bytes()
    assert (config.report_dir / "auto_masks" / "qwen" / "contact_sheet_mask_variants_current.png").exists()
    assert (config.report_dir / "auto_masks" / "qwen" / "contact_sheet_candidate_comparison.png").exists()
    assert all(row["settings"]["mask_truth_source"] == "qwen_auto_refined_masks" for row in rows)
    assert all(row["settings"]["qwen_localization"]["qwen_cache"]["enabled"] is True for row in rows)
    assert all(row["settings"]["qwen_localization"]["qwen_cache"]["cache_hit"] is False for row in rows)
    assert all(row["defect_type"] == "scratch" for row in rows)
    assert all(row["qwen_defect_type"] == "crack" for row in rows)
    for row in rows:
        assert run_output_dir in Path(row["refined_mask_path"]).parents
        assert Path(row["mask_path"]).exists()
        assert Path(row["overlay_path"]).exists()
        assert Path(row["mask_path"]).name.endswith("_mask.png")
        assert Path(row["eval_mask_path"]).exists()
        assert Path(row["training_mask_path"]).exists()
        assert {
            "eval_tight",
            "training_medium",
            "training_wide",
            "training_soft",
            "positive_core",
            "possible_region",
            "uncertainty_map",
            "inpaint_soft",
        } <= set(row["mask_variant_paths"])
        assert row["settings"]["ground_truth_mask_variant"] == "eval_tight"
        assert Path(row["settings"]["eval_mask_path"]).exists()
        assert Path(row["settings"]["training_mask_path"]).exists()
        assert Path(row["settings"]["uncertainty_mask_path"]).exists()
        assert Path(row["mask_variant_paths"]["training_soft"]).exists()
        assert row["settings"]["scratch_morphology_class"] in {"single_stroke", "multi_scuff", "scratch_band"}
        assert row["settings"]["label_policy"]["label_policy"] in {"hard_mask_ok", "soft_mask_only"}
        assert row["settings"]["selection_policy"]
        assert isinstance(row["settings"]["candidate_rejections"], dict)
        eval_area = np.asarray(Image.open(row["eval_mask_path"]).convert("L"), dtype=np.uint8) > 0
        train_area = np.asarray(Image.open(row["training_mask_path"]).convert("L"), dtype=np.uint8) > 0
        assert int(train_area.sum()) >= int(eval_area.sum())

    manifest_path = prepare_splits(config, download=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest["targets"]["custom_part/scratch"]
    assert len(target["adaptation"]) == 2
    assert len(target["held_out"]) == 1
    assert all(sample["training_mask_path"] for sample in target["adaptation"])
    assert all(sample["eval_mask_path"] for sample in target["adaptation"])
    assert all(sample["uncertainty_mask_path"] for sample in target["adaptation"])
    assert all(sample["label_policy"]["label_policy"] in {"hard_mask_ok", "soft_mask_only"} for sample in target["adaptation"])


def test_auto_masks_unavailable_qwen_does_not_clear_existing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    metadata_path = config.output_dir / "auto_masks" / "qwen" / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text('{"existing": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        "iadgen_v2.auto_masks.qwen_availability",
        lambda **kwargs: QwenAvailability(
            model_id="Qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir="",
            cached_locally=False,
            qwen_vl_utils_available=True,
            transformers_available=True,
            free_gib=1.0,
            min_free_gib=30.0,
            local_files_only=True,
        ),
    )

    with pytest.raises(RuntimeError, match="not ready"):
        run_auto_masks(config)
    assert metadata_path.read_text(encoding="utf-8") == '{"existing": true}\n'


def test_auto_masks_interruption_preserves_stable_metadata_and_writes_partial_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    metadata_path = config.output_dir / "auto_masks" / "qwen" / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text('{"existing": true}\n', encoding="utf-8")

    class InterruptingExtractor(_FakeQwenExtractor):
        calls = 0

        def analyze(self, image: Image.Image, prompt: str) -> dict[str, object]:
            type(self).calls += 1
            if type(self).calls == 2:
                raise KeyboardInterrupt
            return super().analyze(image, prompt)

    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", InterruptingExtractor)

    with pytest.raises(KeyboardInterrupt):
        run_auto_masks(config)

    status = json.loads(
        (config.output_dir / "auto_masks" / "qwen" / "last_run_status.json").read_text(encoding="utf-8")
    )
    assert metadata_path.read_text(encoding="utf-8") == '{"existing": true}\n'
    assert status["status"] == "interrupted"
    assert status["records_completed"] == 1
    assert Path(status["partial_metadata_path"]).exists()
    assert not list((config.dataset_root / "custom_part" / "ground_truth" / "scratch").glob("*_mask.png"))


def test_auto_masks_overwrite_false_carries_existing_rows_without_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)
    metadata_path = run_auto_masks(config)
    original_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

    class NoAnalyzeExtractor(_FakeQwenExtractor):
        def analyze(self, image: Image.Image, prompt: str) -> dict[str, object]:
            raise AssertionError("Qwen must not run for carried-forward samples")

    config.data["auto_masks"]["overwrite"] = False
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", NoAnalyzeExtractor)
    carried_path = run_auto_masks(config)
    carried_rows = [json.loads(line) for line in carried_path.read_text(encoding="utf-8").splitlines()]
    current_run = json.loads(
        (config.output_dir / "auto_masks" / "qwen" / "current_run.json").read_text(encoding="utf-8")
    )

    assert carried_rows == original_rows
    assert current_run["records"] == 3
    assert Path(current_run["run_metadata_path"]).read_bytes() == carried_path.read_bytes()


def test_auto_masks_can_reselect_cached_candidates_without_qwen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    config.data["auto_masks"]["mask_refinement"] = "auto"
    config.data["auto_masks"]["auto_candidate_modes"] = ["pixel", "procedural"]
    config.data["auto_masks"]["use_ensemble_consensus"] = False
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)
    initial_path = run_auto_masks(config)
    initial_rows = [json.loads(line) for line in initial_path.read_text(encoding="utf-8").splitlines()]

    reselected_path = reselect_auto_masks(config)
    reselected_rows = [json.loads(line) for line in reselected_path.read_text(encoding="utf-8").splitlines()]
    current_run = json.loads(
        (config.output_dir / "auto_masks" / "qwen" / "current_run.json").read_text(encoding="utf-8")
    )

    assert len(reselected_rows) == len(initial_rows) == 3
    assert current_run["run_kind"] == "cached_candidate_reselection"
    assert all(row["settings"]["reselected_from_cached_candidates"] is True for row in reselected_rows)
    assert all(Path(current_run["run_output_dir"]) in Path(row["eval_mask_path"]).parents for row in reselected_rows)
    assert all(Path(row["mask_path"]).exists() for row in reselected_rows)


def test_auto_masks_reselect_repairs_cleaned_candidate_cache_with_selected_mask_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    config.data["auto_masks"]["mask_refinement"] = "auto"
    config.data["auto_masks"]["auto_candidate_modes"] = ["pixel", "procedural"]
    config.data["auto_masks"]["use_ensemble_consensus"] = False
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)
    initial_path = run_auto_masks(config)
    initial_rows = [json.loads(line) for line in initial_path.read_text(encoding="utf-8").splitlines()]

    missing_dir = tmp_path / "deleted_candidate_cache"
    for row in initial_rows:
        settings = row["settings"]
        settings["candidate_refined_paths"] = {
            mode: str(missing_dir / f"{Path(row['image_path']).stem}_{mode}_refined.png")
            for mode in settings["candidate_refined_paths"]
        }
        settings["candidate_heatmap_paths"] = {
            mode: str(missing_dir / f"{Path(row['image_path']).stem}_{mode}_heatmap.png")
            for mode in settings.get("candidate_heatmap_paths", {})
        }
    initial_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in initial_rows) + "\n",
        encoding="utf-8",
    )

    reselected_path = reselect_auto_masks(config)
    reselected_rows = [json.loads(line) for line in reselected_path.read_text(encoding="utf-8").splitlines()]
    current_run = json.loads(
        (config.output_dir / "auto_masks" / "qwen" / "current_run.json").read_text(encoding="utf-8")
    )

    assert len(reselected_rows) == len(initial_rows) == 3
    assert current_run["source_cache_audit"]["existing_candidate_paths"] == 0
    assert current_run["output_cache_audit"]["rows_with_existing_candidates"] == 3
    assert all(row["settings"]["candidate_cache_status"] == "selected_mask_compatibility_fallback" for row in reselected_rows)
    assert all(row["settings"]["oracle_regret_available"] is False for row in reselected_rows)
    assert all(row["settings"]["selection_arbitration"]["reason"] == "selected_mask_compatibility_fallback" for row in reselected_rows)
    assert all(Path(row["eval_mask_path"]).exists() for row in reselected_rows)


def test_auto_masks_reselect_can_fail_strictly_when_candidate_cache_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture_config(tmp_path)
    config.data["auto_masks"]["mask_refinement"] = "auto"
    config.data["auto_masks"]["auto_candidate_modes"] = ["pixel", "procedural"]
    config.data["auto_masks"]["use_ensemble_consensus"] = False
    config.data["auto_masks"]["allow_selected_mask_reselect_fallback"] = False
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)
    initial_path = run_auto_masks(config)
    initial_rows = [json.loads(line) for line in initial_path.read_text(encoding="utf-8").splitlines()]

    for row in initial_rows:
        row["settings"]["candidate_refined_paths"] = {
            mode: str(tmp_path / "missing" / f"{mode}.png")
            for mode in row["settings"]["candidate_refined_paths"]
        }
    initial_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in initial_rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Regenerate full auto-mask candidates"):
        reselect_auto_masks(config)


def test_auto_refinement_records_selected_candidate_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fixture_config(tmp_path)
    config.data["auto_masks"]["mask_refinement"] = "auto"
    config.data["auto_masks"]["auto_candidate_modes"] = ["pixel", "procedural"]
    config.data["auto_masks"]["use_ensemble_consensus"] = False
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)

    metadata_path = run_auto_masks(config)
    row = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])

    assert row["settings"]["mask_refinement"] == "auto"
    assert row["settings"]["candidate_modes"] == ["pixel", "procedural"]
    assert row["settings"]["selected_refinement"] in {"pixel", "procedural"}
    assert set(row["settings"]["candidate_scores"]) == {"pixel", "procedural"}
    assert row["settings"]["candidate_failures"] == {}
    assert Path(row["refined_mask_path"]).name.endswith("_refined.png")


def test_auto_refinement_selects_highest_scoring_valid_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (64, 64), (100, 100, 100))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 20, 50, 24), fill=255)
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        score = 0.2 if mode == "low_score" else 0.9
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {"status": "pass", "mask_area": 205},
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="scratch",
        region=(8, 18, 56, 30),
        output_dir=tmp_path,
        artifact_stem="sample",
        seed=1,
        auto={"auto_candidate_modes": ["low_score", "high_score"], "use_ensemble_consensus": False},
        description="multiple scratches",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "high_score"
    assert selected["candidate_scores"] == {"low_score": 0.2, "high_score": 0.9}


def _patch_fake_candidates(
    monkeypatch: pytest.MonkeyPatch,
    image: Image.Image,
    scores: dict[str, float],
) -> None:
    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 20, 54, 48), fill=255)
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {"status": "pass", "mask_area": 1260, "mask_to_qwen_box_fraction": 0.12, "component_count": 1},
            "score": scores.get(mode, 0.0),
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)


def test_foundation_anomaly_field_writes_diagnostic_artifacts(tmp_path: Path) -> None:
    width, height = 96, 80
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    base = 132.0 + 12.0 * np.sin(xx * 18.0) + 7.0 * np.cos(yy * 13.0)
    normal_arr = np.repeat(np.clip(base, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    normal = Image.fromarray(normal_arr, mode="RGB")
    defect = normal.copy()
    draw = ImageDraw.Draw(defect)
    draw.rectangle((40, 30, 64, 45), fill=(55, 55, 55))
    draw.line((38, 48, 70, 52), fill=(65, 65, 65), width=2)
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)

    artifacts = write_foundation_anomaly_field_bbox_masks(
        defect,
        [normal_path],
        "bottle",
        "contamination",
        (24, 18, 78, 62),
        tmp_path,
        "foundation_case",
        auto={
            "foundation_patchcore_patch_size": 7,
            "foundation_patchcore_stride": 6,
            "foundation_musc_patch_size": 7,
            "foundation_musc_stride": 6,
            "foundation_max_normals": 1,
            "foundation_max_memory_patches": 256,
            "foundation_search_radius": 6,
            "foundation_search_step": 4,
            "foundation_percentile": 80.0,
            "foundation_min_contrast": 0.001,
            "foundation_max_box_fraction": 0.22,
            "foundation_max_components": 8,
            "min_component_area": 4,
        },
        text_hint="dark contamination patch on bottle surface",
        padding=3,
    )

    params = artifacts["parameters"]
    assert params["kind"] == "foundation_anomaly_field"
    assert params["foundation_source_count"] >= 1
    assert params["foundation_region_contrast"] >= 0.0
    assert Path(params["heatmap_path"]).exists()
    assert Path(params["foundation_candidate_path"]).exists()
    assert Path(params["foundation_overlay_path"]).exists()
    assert Path(artifacts["refined_mask_path"]).exists()
    assert artifacts["qc"]["mask_area"] > 0


def test_patch_feature_cache_reuses_features_across_patchcore_family(tmp_path: Path) -> None:
    width, height = 72, 64
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    normal = Image.fromarray(arr, mode="RGB")
    defect = normal.copy()
    ImageDraw.Draw(defect).rectangle((30, 22, 46, 36), fill=(55, 55, 55))
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)
    cache: dict[object, object] = {}

    _, patchcore_params = patchcore_guided_heatmap(
        defect,
        [normal_path],
        (12, 12, 60, 52),
        "dark abnormal stain",
        patch_size=9,
        stride=8,
        max_normals=1,
        max_memory_patches=256,
        feature_cache=cache,
    )
    _, musc_params = musc_mutual_score_heatmap(
        defect,
        [normal_path],
        (12, 12, 60, 52),
        "dark abnormal stain",
        patch_size=9,
        stride=8,
        max_normals=1,
        max_memory_patches=256,
        knn_k=2,
        feature_cache=cache,
    )

    assert patchcore_params["patchcore_guided_feature_cache_enabled"] == 1
    assert patchcore_params["patchcore_guided_feature_cache_misses"] >= 2
    assert musc_params["musc_feature_cache_hits"] >= 2
    assert musc_params["musc_feature_cache_misses"] == patchcore_params["patchcore_guided_feature_cache_misses"]


def test_derived_tooth_agreement_intersects_patchcore_and_musc(tmp_path: Path) -> None:
    image = Image.new("RGB", (80, 80), (120, 120, 120))
    patchcore = Image.new("L", image.size, 0)
    musc = Image.new("L", image.size, 0)
    ImageDraw.Draw(patchcore).rectangle((20, 10, 50, 60), fill=255)
    ImageDraw.Draw(musc).rectangle((32, 20, 62, 70), fill=255)
    patchcore_path = tmp_path / "patchcore.png"
    musc_path = tmp_path / "musc.png"
    patchcore.save(patchcore_path)
    musc.save(musc_path)

    candidates = {
        "patchcore_guided": {
            "refined_mask_path": str(patchcore_path),
            "qc": {"status": "pass", "mask_area": int((np.asarray(patchcore) > 0).sum())},
            "score": 0.9,
        },
        "musc_mutual_score": {
            "refined_mask_path": str(musc_path),
            "qc": {"status": "pass", "mask_area": int((np.asarray(musc) > 0).sum())},
            "score": 0.8,
        },
    }
    valid_candidates = dict(candidates)

    auto_masks._add_derived_recall_safe_candidates(
        candidates=candidates,
        valid_candidates=valid_candidates,
        image=image,
        category="zipper",
        defect_type="broken_teeth",
        region=(10, 0, 70, 80),
        output_dir=tmp_path,
        artifact_stem="agreement_case",
        auto={"min_component_area": 3},
        description="broken zipper teeth",
    )

    derived = valid_candidates["zipper_tooth_agreement"]
    arr = np.asarray(Image.open(derived["refined_mask_path"]).convert("L")) > 0
    assert arr[25:55, 35:48].any()
    assert not arr[12:18, 22:28].any()
    assert derived["parameters"]["operation"] == "intersection"


def test_foundation_anomaly_field_can_win_score_only_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {
            "foundation_anomaly_field": 0.58,
            "normal_anomaly": 0.40,
            "procedural": 0.35,
        },
    )

    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="stain",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="foundation_selector",
        seed=1,
        auto={
            "auto_candidate_modes": ["foundation_anomaly_field", "normal_anomaly", "procedural"],
            "use_ensemble_consensus": False,
        },
        description="small abnormal stain on the surface",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "foundation_anomaly_field"
    assert selected["policy_scores"]["foundation_anomaly_field"] > selected["policy_scores"]["normal_anomaly"]


def test_bottle_contamination_policy_prefers_fft_over_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"ensemble_consensus": 0.95, "fft_texture_suppression": 0.45, "sam2_heatmap": 0.40},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="bottle",
        defect_type="contamination",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="bottle_contam",
        seed=1,
        auto={
            "auto_candidate_modes": ["ensemble_consensus", "fft_texture_suppression", "sam2_heatmap"],
            "use_ensemble_consensus": False,
        },
        description="visible contamination stain on bottle surface",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "fft_texture_suppression"
    assert selected["selection_policy"] == "bottle_contamination_prefer_texture_or_sam"
    assert selected["policy_scores"]["fft_texture_suppression"] > selected["policy_scores"]["ensemble_consensus"]


def test_bottle_rim_chip_policy_can_prefer_sam2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"ensemble_consensus": 0.65, "sam2_heatmap": 0.40, "nearest_normal_residual": 0.20},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="bottle",
        defect_type="broken_small",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="bottle_chip",
        seed=1,
        auto={
            "auto_candidate_modes": ["ensemble_consensus", "sam2_heatmap", "nearest_normal_residual"],
            "use_ensemble_consensus": False,
        },
        description="small chipped missing glass on the bottle rim",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "sam2_heatmap"
    assert selected["selection_policy"] == "bottle_small_chip_sam_gate"


def test_bottle_small_chip_policy_prefers_valid_polar_rim_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        is_sam = mode == "sam2_heatmap"
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).ellipse((56, 28, 76, 48), fill=255)
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {
                "kind": mode,
                **({"polar_rim_sector_degrees": 32.0} if mode == "polar_rim_residual" else {}),
            },
            "qc": {
                "status": "warning" if is_sam else "pass",
                "mask_area": 7000 if is_sam else 300,
                "mask_area_fraction": 0.76 if is_sam else 0.03,
                "mask_to_qwen_box_fraction": 0.82 if is_sam else 0.08,
                "component_count": 1,
                "reasons": ["large_fraction_of_qwen_box"] if is_sam else [],
            },
            "score": 0.85 if mode == "polar_rim_residual" else 0.55,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="bottle",
        defect_type="broken_small",
        region=(8, 8, 88, 88),
        output_dir=tmp_path,
        artifact_stem="bottle_polar_chip",
        seed=1,
        auto={
            "auto_candidate_modes": ["polar_rim_residual", "sam2_heatmap"],
            "use_ensemble_consensus": False,
        },
        description="small chipped missing glass on the bottle rim",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "polar_rim_residual"
    assert selected["policy_scores"]["polar_rim_residual"] >= selected["policy_scores"]["sam2_heatmap"]


def test_bottle_small_chip_policy_can_prefer_sam_prompt_regularized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"ensemble_consensus": 0.68, "sam_prompt_regularized": 0.30, "musc_mutual_score": 0.25},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="bottle",
        defect_type="broken_small",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="bottle_chip_regularized",
        seed=1,
        auto={
            "auto_candidate_modes": ["ensemble_consensus", "sam_prompt_regularized", "musc_mutual_score"],
            "use_ensemble_consensus": False,
        },
        description="small chipped missing glass on the bottle rim",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "sam_prompt_regularized"
    assert selected["selection_policy"] == "bottle_small_chip_sam_gate"
    assert selected["policy_scores"]["sam_prompt_regularized"] > selected["policy_scores"]["ensemble_consensus"]


def test_bottle_small_chip_policy_rejects_huge_sam2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 20, 54, 48), fill=255)
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        if mode == "sam2_heatmap":
            qc = {
                "status": "warning",
                "mask_area": 5000,
                "mask_area_fraction": 0.55,
                "mask_to_qwen_box_fraction": 0.80,
                "component_count": 1,
                "reasons": ["large_fraction_of_qwen_box"],
            }
            score = 0.55
        else:
            qc = {"status": "pass", "mask_area": 900, "mask_to_qwen_box_fraction": 0.12, "component_count": 2}
            score = 0.70
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": qc,
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="bottle",
        defect_type="broken_small",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="bottle_chip_huge_sam",
        seed=1,
        auto={
            "auto_candidate_modes": ["sam2_heatmap", "nearest_normal_residual"],
            "use_ensemble_consensus": False,
        },
        description="small chipped missing glass on the bottle rim",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "nearest_normal_residual"
    assert selected["policy_scores"]["nearest_normal_residual"] > selected["policy_scores"]["sam2_heatmap"]


def test_zipper_tooth_policy_prefers_patchcore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"ensemble_consensus": 0.70, "patchcore_guided": 0.35, "normal_residual_fusion": 0.40},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="zipper",
        defect_type="broken_teeth",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="zipper_teeth",
        seed=1,
        auto={
            "auto_candidate_modes": ["ensemble_consensus", "patchcore_guided", "normal_residual_fusion"],
            "use_ensemble_consensus": False,
        },
        description="one or more zipper teeth are broken or missing",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "patchcore_guided"
    assert selected["selection_policy"] == "zipper_tooth_chain_prefer_patchcore_normal_residual"


def test_repeated_chain_gate_arbitrates_over_score_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (120, 240), (100, 100, 100))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        if mode == "patchcore_guided":
            ImageDraw.Draw(mask).rectangle((46, 6, 74, 232), fill=255)
            score = 0.95
        else:
            ImageDraw.Draw(mask).rectangle((48, 164, 72, 196), fill=255)
            score = 0.15
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {
                "kind": mode,
                "repeated_chain_explanation_score": 0.8,
                "repeated_chain_structure_score": 0.9,
                "repeated_chain_vertical_span_fraction": 0.15,
            },
            "qc": {
                "status": "pass",
                "mask_area": int((np.asarray(mask) > 0).sum()),
                "mask_area_fraction": 0.04,
                "mask_to_qwen_box_fraction": 0.10,
                "component_count": 1,
                "reasons": [],
            },
            "score": score,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="zipper",
        defect_type="broken_teeth",
        region=(20, 0, 100, 240),
        output_dir=tmp_path,
        artifact_stem="zipper_gate_arbitration",
        seed=1,
        auto={
            "auto_candidate_modes": ["patchcore_guided", "repeated_chain_refiner"],
            "use_ensemble_consensus": False,
        },
        description="broken zipper teeth",
        normal_paths=[],
        requested="auto",
        localization_status="valid",
    )

    assert selected["selected_refinement"] == "repeated_chain_refiner"
    assert selected["selection_arbitration"]["applied"] is True
    assert selected["selection_arbitration"]["initial_selected_refinement"] == "patchcore_guided"


def test_zipper_tooth_policy_can_use_musc_when_patchcore_is_not_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"ensemble_consensus": 0.65, "musc_mutual_score": 0.43, "normal_residual_fusion": 0.35},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="zipper",
        defect_type="split_teeth",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="zipper_teeth_musc",
        seed=1,
        auto={
            "auto_candidate_modes": ["ensemble_consensus", "musc_mutual_score", "normal_residual_fusion"],
            "use_ensemble_consensus": False,
        },
        description="zipper teeth split apart near the central tooth chain",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "musc_mutual_score"
    assert selected["selection_policy"] == "zipper_tooth_chain_prefer_patchcore_normal_residual"


def test_zipper_fabric_border_policy_prefers_sam2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"ensemble_consensus": 0.70, "sam2_heatmap": 0.35, "patchcore_guided": 0.30},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="zipper",
        defect_type="fabric_border",
        region=(8, 8, 80, 80),
        output_dir=tmp_path,
        artifact_stem="zipper_border",
        seed=1,
        auto={
            "auto_candidate_modes": ["ensemble_consensus", "sam2_heatmap", "patchcore_guided"],
            "use_ensemble_consensus": False,
        },
        description="frayed damaged fabric border beside the zipper teeth",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "sam2_heatmap"
    assert selected["selection_policy"] == "zipper_fabric_border_prefer_sam_patchcore_border_support"


def test_zipper_fabric_border_policy_avoids_tiny_full_frame_sam2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 20, 54, 48), fill=255)
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        if mode == "sam2_heatmap":
            qc = {
                "status": "warning",
                "mask_area": 20,
                "mask_area_fraction": 0.002,
                "mask_to_qwen_box_fraction": 0.002,
                "component_count": 1,
                "reasons": ["tiny_fraction_of_qwen_box"],
            }
            score = 0.50
        else:
            qc = {"status": "pass", "mask_area": 800, "mask_to_qwen_box_fraction": 0.08, "component_count": 2}
            score = 0.65
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": qc,
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="zipper",
        defect_type="fabric_border",
        region=(0, 0, 96, 96),
        output_dir=tmp_path,
        artifact_stem="zipper_border_tiny_sam",
        seed=1,
        auto={
            "auto_candidate_modes": ["sam2_heatmap", "ensemble_consensus"],
            "use_ensemble_consensus": False,
        },
        description="frayed damaged fabric border beside the zipper teeth",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "ensemble_consensus"
    assert selected["policy_scores"]["ensemble_consensus"] > selected["policy_scores"]["sam2_heatmap"]


def test_zipper_fabric_border_full_frame_policy_prefers_layout_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (96, 96), (120, 120, 120))
    _patch_fake_candidates(
        monkeypatch,
        image,
        {"normal_residual_fusion": 0.70, "zipper_fabric_border_layout": 0.20, "sam2_heatmap": 0.10},
    )

    selected = _select_mask_artifacts(
        image=image,
        category="zipper",
        defect_type="fabric_border",
        region=(0, 0, 96, 96),
        output_dir=tmp_path,
        artifact_stem="zipper_border_layout",
        seed=1,
        auto={
            "auto_candidate_modes": ["normal_residual_fusion", "zipper_fabric_border_layout", "sam2_heatmap"],
            "use_ensemble_consensus": False,
        },
        description="frayed damaged fabric border beside the zipper teeth",
        normal_paths=[],
        requested="auto",
    )

    assert selected["selected_refinement"] == "zipper_fabric_border_layout"
    assert selected["selection_policy"] == "zipper_fabric_border_prefer_sam_patchcore_border_support"


def test_scuffed_patch_policy_blocks_undercovered_linear_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = Image.new("RGB", (128, 128), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        if mode == "linear":
            draw.line((40, 64, 92, 72), fill=255, width=2)
            score = 0.99
            fraction = 0.025
            area = 110
        else:
            for y in (48, 58, 70, 82):
                draw.line((36, y, 96, y + 5), fill=255, width=3)
            score = 0.74
            fraction = 0.12
            area = 620
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {
                "status": "pass",
                "mask_area": area,
                "mask_to_qwen_box_fraction": fraction,
                "component_count": 1,
                "shrunk_region_xyxy": (36, 48, 96, 86),
            },
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="scratch",
        region=(28, 36, 104, 100),
        output_dir=tmp_path,
        artifact_stem="scuff",
        seed=1,
        auto={"auto_candidate_modes": ["linear", "scuff_cluster"]},
        description="a scuffed area with several scratch marks",
        normal_paths=[],
        requested="auto",
    )

    assert selected["scratch_morphology_class"] == "multi_scuff"
    assert selected["candidate_rejections"]["linear"].startswith("scuffed")
    assert selected["selected_refinement"] == "scuff_cluster"


def test_fft_texture_suppression_detects_periodic_texture_scuff(tmp_path: Path) -> None:
    width, height = 128, 96
    x = np.arange(width, dtype=np.float32)
    grain = 120.0 + 35.0 * np.sin(x / 4.0)
    arr = np.repeat(grain[None, :], height, axis=0)
    arr[34:60, 42:88] = arr[34:60, 42:88] * 0.65 + 80.0
    image_arr = np.repeat(np.clip(arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    image = Image.fromarray(image_arr, mode="RGB")

    artifacts = write_fft_texture_suppression_bbox_masks(
        image,
        [],
        "custom_part",
        "scratch",
        (30, 24, 100, 72),
        tmp_path,
        "fft_scuff",
        auto={
            "fft_texture_percentile": 82.0,
            "fft_texture_min_contrast": 0.02,
            "fft_texture_blend_with_scuff_seeds": False,
            "min_component_area": 6,
        },
        text_hint="a rubbed scuffed area with many faint scratches",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    box = np.zeros(refined.shape, dtype=bool)
    box[24:72, 30:100] = True
    assert refined.any()
    assert int(refined.sum()) < int(box.sum())
    assert np.logical_and(refined, ~box).sum() == 0
    assert artifacts["parameters"]["kind"] == "fft_texture_suppression"
    assert artifacts["parameters"]["fft_texture_active_pixels"] > 0


def test_structure_tensor_ridge_detects_diagonal_scratch(tmp_path: Path) -> None:
    width, height = 128, 96
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.line((34, 70, 98, 30), fill=(40, 40, 40), width=3)

    artifacts = write_structure_tensor_ridge_bbox_masks(
        image,
        "custom_part",
        "scratch",
        (24, 20, 108, 80),
        tmp_path,
        "ridge_scratch",
        auto={
            "structure_tensor_percentile": 86.0,
            "structure_tensor_max_box_fraction": 0.12,
            "structure_tensor_max_components": 4,
            "min_component_area": 4,
        },
        text_hint="one long diagonal scratch",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    box = np.zeros(refined.shape, dtype=bool)
    box[20:80, 24:108] = True
    assert refined.any()
    assert np.logical_and(refined, ~box).sum() == 0
    assert int(refined.sum()) < int(box.sum() * 0.15)
    assert artifacts["parameters"]["kind"] == "structure_tensor_ridge"
    assert artifacts["parameters"]["structure_tensor_active_pixels"] > 0


def test_nearest_normal_residual_detects_scuff_against_clean_reference(tmp_path: Path) -> None:
    width, height = 128, 96
    x = np.arange(width, dtype=np.float32)
    grain = 128.0 + 30.0 * np.sin(x / 5.0)
    normal_arr = np.repeat(grain[None, :], height, axis=0)
    defect_arr = normal_arr.copy()
    defect_arr[34:60, 42:88] = defect_arr[34:60, 42:88] * 0.72 + 70.0
    normal_img = Image.fromarray(np.repeat(np.clip(normal_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    defect_img = Image.fromarray(np.repeat(np.clip(defect_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    normal_path = tmp_path / "normal.png"
    normal_img.save(normal_path)

    artifacts = write_nearest_normal_residual_bbox_masks(
        defect_img,
        [normal_path],
        "custom_part",
        "scratch",
        (30, 24, 100, 72),
        tmp_path,
        "nearest_scuff",
        auto={
            "nearest_normal_residual_percentile": 82.0,
            "nearest_normal_residual_max_box_fraction": 0.14,
            "nearest_normal_residual_close_radius": 2,
            "nearest_normal_residual_search_radius": 6,
            "nearest_normal_residual_search_step": 6,
            "min_component_area": 6,
        },
        text_hint="a rubbed scuffed area with many faint scratches",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert refined.any()
    assert int(refined.sum()) < int((100 - 30) * (72 - 24) * 0.16)
    assert artifacts["parameters"]["kind"] == "nearest_normal_patch_residual"
    assert artifacts["parameters"]["nearest_normal_residual_active_pixels"] > 0
    assert artifacts["parameters"]["nearest_normal_residual_normal_path"] == str(normal_path)


def test_patchcore_guided_detects_scuff_against_normal_memory(tmp_path: Path) -> None:
    width, height = 128, 96
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    grain = 126.0 + 24.0 * np.sin(x / 5.0)[None, :] + 8.0 * np.sin(y[:, None] / 13.0)
    normal_arr = np.clip(grain, 0, 255)
    defect_arr = normal_arr.copy()
    defect_arr[34:60, 42:88] = defect_arr[34:60, 42:88] * 0.62 + 86.0
    defect_arr[39:43, 48:84] = defect_arr[39:43, 48:84] - 35.0
    normal_img = Image.fromarray(np.repeat(np.clip(normal_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    defect_img = Image.fromarray(np.repeat(np.clip(defect_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    normal_path = tmp_path / "normal.png"
    normal_img.save(normal_path)

    artifacts = write_patchcore_guided_bbox_masks(
        defect_img,
        [normal_path],
        "custom_part",
        "scratch",
        (30, 24, 100, 72),
        tmp_path,
        "patchcore_scuff",
        auto={
            "patchcore_guided_patch_size": 13,
            "patchcore_guided_stride": 4,
            "patchcore_guided_percentile": 82.0,
            "patchcore_guided_max_box_fraction": 0.16,
            "patchcore_guided_min_contrast": 0.01,
            "patchcore_guided_max_components": 8,
            "min_component_area": 6,
        },
        text_hint="a rubbed scuffed area with many faint scratches",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    defect_support = np.zeros_like(refined, dtype=bool)
    defect_support[34:60, 42:88] = True
    assert refined.any()
    assert float(refined[defect_support].mean()) > float(refined[~defect_support].mean())
    assert int(refined.sum()) < int((100 - 30) * (72 - 24) * 0.18)
    assert artifacts["parameters"]["kind"] == "patchcore_guided"
    assert artifacts["parameters"]["patchcore_guided_active_pixels"] > 0
    assert artifacts["parameters"]["patchcore_guided_memory_patches"] > 0


def test_musc_mutual_score_detects_scuff_against_normal_memory(tmp_path: Path) -> None:
    width, height = 128, 96
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    grain = 126.0 + 24.0 * np.sin(x / 5.0)[None, :] + 8.0 * np.sin(y[:, None] / 13.0)
    normal_arr = np.clip(grain, 0, 255)
    defect_arr = normal_arr.copy()
    defect_arr[34:60, 42:88] = defect_arr[34:60, 42:88] * 0.60 + 82.0
    defect_arr[39:43, 48:84] = defect_arr[39:43, 48:84] - 38.0
    normal_img = Image.fromarray(np.repeat(np.clip(normal_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    defect_img = Image.fromarray(np.repeat(np.clip(defect_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    normal_path = tmp_path / "normal.png"
    normal_img.save(normal_path)

    artifacts = write_musc_mutual_score_bbox_masks(
        defect_img,
        [normal_path],
        "custom_part",
        "scratch",
        (30, 24, 100, 72),
        tmp_path,
        "musc_scuff",
        auto={
            "musc_patch_size": 13,
            "musc_stride": 4,
            "musc_percentile": 82.0,
            "musc_max_box_fraction": 0.16,
            "musc_min_contrast": 0.005,
            "musc_max_components": 8,
            "musc_knn_k": 3,
            "min_component_area": 6,
        },
        text_hint="a rubbed scuffed area with many faint scratches",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    defect_support = np.zeros_like(refined, dtype=bool)
    defect_support[34:60, 42:88] = True
    assert refined.any()
    assert float(refined[defect_support].mean()) > float(refined[~defect_support].mean())
    assert artifacts["parameters"]["kind"] == "musc_mutual_score"
    assert artifacts["parameters"]["musc_memory_patches"] > 0
    assert artifacts["parameters"]["musc_patchcore_guided_active_pixels"] > 0


def test_multi_scuff_policy_can_select_nearest_normal_residual(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (128, 128), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        if mode == "nearest_normal_residual":
            for y in (48, 58, 70, 82):
                draw.line((36, y, 96, y + 4), fill=255, width=3)
            score = 0.78
            fraction = 0.13
            components = 4
        else:
            draw.line((40, 62, 94, 66), fill=255, width=2)
            score = 0.94
            fraction = 0.025
            components = 1
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {
                "status": "pass",
                "mask_area": int((np.asarray(mask) > 0).sum()),
                "mask_to_qwen_box_fraction": fraction,
                "component_count": components,
                "shrunk_region_xyxy": (36, 48, 96, 86),
            },
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="scratch",
        region=(28, 36, 104, 100),
        output_dir=tmp_path,
        artifact_stem="nearest_policy",
        seed=1,
        auto={"auto_candidate_modes": ["linear", "nearest_normal_residual"], "use_ensemble_consensus": False},
        description="a scuffed rubbed patch with many faint scratches",
        normal_paths=[],
        requested="auto",
    )

    assert selected["scratch_morphology_class"] == "multi_scuff"
    assert selected["selected_refinement"] == "nearest_normal_residual"


def test_normal_residual_fusion_writer_limits_grain_leakage(tmp_path: Path) -> None:
    width, height = 128, 96
    x = np.arange(width, dtype=np.float32)
    grain = 128.0 + 30.0 * np.sin(x / 5.0)
    normal_arr = np.repeat(grain[None, :], height, axis=0)
    defect_arr = normal_arr.copy()
    defect_arr[34:60, 42:88] = defect_arr[34:60, 42:88] * 0.72 + 70.0
    normal_img = Image.fromarray(np.repeat(np.clip(normal_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    defect_img = Image.fromarray(np.repeat(np.clip(defect_arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2), mode="RGB")
    normal_path = tmp_path / "normal.png"
    normal_img.save(normal_path)

    artifacts = write_normal_residual_fusion_bbox_masks(
        defect_img,
        [normal_path],
        "custom_part",
        "scratch",
        (30, 24, 100, 72),
        tmp_path,
        "residual_fusion",
        auto={
            "normal_anomaly_percentile": 94.0,
            "soft_patch_percentile": 88.0,
            "nearest_normal_residual_percentile": 82.0,
            "nearest_normal_residual_max_box_fraction": 0.14,
            "nearest_normal_residual_search_radius": 6,
            "nearest_normal_residual_search_step": 6,
            "fft_texture_min_contrast": 0.02,
            "multi_scuff_fusion_support_percentile": 70.0,
            "multi_scuff_fusion_max_box_fraction": 0.11,
            "normal_residual_fusion_max_box_fraction": 0.12,
            "normal_residual_fusion_grain_reject_angle_degrees": 16.0,
            "min_component_area": 6,
        },
        text_hint="a rubbed scuffed area with many faint scratches",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    box_area = (100 - 30) * (72 - 24)
    assert refined.any()
    assert int(refined.sum()) <= int(box_area * 0.13)
    assert artifacts["parameters"]["kind"] == "normal_residual_fusion"
    assert artifacts["parameters"]["normal_residual_fusion_pixels"] > 0


def test_multi_scuff_policy_can_select_normal_residual_fusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (128, 128), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        if mode == "normal_residual_fusion":
            for y in (48, 58, 70, 82):
                draw.line((36, y, 96, y + 4), fill=255, width=3)
            score = 0.78
            fraction = 0.13
            components = 4
        else:
            draw.line((40, 62, 94, 66), fill=255, width=2)
            score = 0.94
            fraction = 0.025
            components = 1
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {
                "status": "pass",
                "mask_area": int((np.asarray(mask) > 0).sum()),
                "mask_to_qwen_box_fraction": fraction,
                "component_count": components,
                "shrunk_region_xyxy": (36, 48, 96, 86),
            },
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="scratch",
        region=(28, 36, 104, 100),
        output_dir=tmp_path,
        artifact_stem="normal_residual_policy",
        seed=1,
        auto={"auto_candidate_modes": ["linear", "normal_residual_fusion"], "use_ensemble_consensus": False},
        description="a scuffed rubbed patch with many faint scratches",
        normal_paths=[],
        requested="auto",
    )

    assert selected["scratch_morphology_class"] == "multi_scuff"
    assert selected["selected_refinement"] == "normal_residual_fusion"


def test_multi_scuff_policy_can_select_fft_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (128, 128), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        if mode == "fft_texture_suppression":
            draw.rectangle((38, 46, 92, 78), fill=255)
            score = 0.76
            fraction = 0.16
        else:
            draw.line((40, 62, 94, 66), fill=255, width=2)
            score = 0.94
            fraction = 0.025
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {
                "status": "pass",
                "mask_area": int((np.asarray(mask) > 0).sum()),
                "mask_to_qwen_box_fraction": fraction,
                "component_count": 1,
                "shrunk_region_xyxy": (38, 46, 92, 78),
            },
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="scratch",
        region=(28, 36, 104, 100),
        output_dir=tmp_path,
        artifact_stem="scuff_fft",
        seed=1,
        auto={"auto_candidate_modes": ["linear", "fft_texture_suppression"], "use_ensemble_consensus": False},
        description="a scuffed rubbed patch with many faint scratches",
        normal_paths=[],
        requested="auto",
    )

    assert selected["scratch_morphology_class"] == "multi_scuff"
    assert selected["selected_refinement"] == "fft_texture_suppression"


def test_multi_scuff_fusion_uses_fft_as_support_without_blob(tmp_path: Path) -> None:
    width, height = 128, 96
    x = np.arange(width, dtype=np.float32)
    grain = 120.0 + 35.0 * np.sin(x / 4.0)
    arr = np.repeat(grain[None, :], height, axis=0)
    arr[34:60, 42:88] = arr[34:60, 42:88] * 0.68 + 78.0
    image_arr = np.repeat(np.clip(arr, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    image = Image.fromarray(image_arr, mode="RGB")

    artifacts = write_multi_scuff_fusion_bbox_masks(
        image,
        [],
        "custom_part",
        "scratch",
        (30, 24, 100, 72),
        tmp_path,
        "fusion_scuff",
        auto={
            "normal_anomaly_percentile": 94.0,
            "soft_patch_percentile": 88.0,
            "fft_texture_min_contrast": 0.02,
            "multi_scuff_fusion_support_percentile": 70.0,
            "multi_scuff_fusion_max_box_fraction": 0.11,
            "multi_scuff_fusion_fine_dilate_radius": 4,
            "multi_scuff_fusion_close_radius": 2,
            "min_component_area": 6,
        },
        text_hint="a rubbed scuffed area with many faint scratches",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    box_area = (100 - 30) * (72 - 24)
    assert refined.any()
    assert int(refined.sum()) <= int(box_area * 0.12)
    assert artifacts["parameters"]["kind"] == "multi_scuff_fusion"
    assert artifacts["parameters"]["multi_scuff_fusion_pixels"] > 0


def test_multi_scuff_policy_prefers_fusion_over_raw_fft_blob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (128, 128), (120, 120, 120))

    def fake_make_candidate(**kwargs: object) -> dict[str, object]:
        mode = str(kwargs["mode"])
        output_dir = Path(kwargs["output_dir"])
        stem = str(kwargs["artifact_stem"])
        output_dir.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        if mode == "fft_texture_suppression":
            draw.rectangle((36, 46, 96, 82), fill=255)
            score = 0.91
            fraction = 0.30
        else:
            for y in (50, 61, 73):
                draw.line((42, y, 92, y + 3), fill=255, width=3)
            score = 0.78
            fraction = 0.12
        paths = {
            "box_mask_path": output_dir / f"{stem}_box.png",
            "refined_mask_path": output_dir / f"{stem}_refined.png",
            "inpaint_mask_path": output_dir / f"{stem}_inpaint.png",
        }
        for path in paths.values():
            mask.save(path)
        return {
            **{key: str(value) for key, value in paths.items()},
            "parameters": {"kind": mode},
            "qc": {
                "status": "pass",
                "mask_area": int((np.asarray(mask) > 0).sum()),
                "mask_to_qwen_box_fraction": fraction,
                "component_count": 1 if mode == "fft_texture_suppression" else 3,
                "shrunk_region_xyxy": (36, 46, 96, 82),
            },
            "score": score,
            "selected_refinement": mode,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks._make_mask_candidate", fake_make_candidate)
    selected = _select_mask_artifacts(
        image=image,
        category="custom_part",
        defect_type="scratch",
        region=(28, 36, 104, 100),
        output_dir=tmp_path,
        artifact_stem="scuff_fusion",
        seed=1,
        auto={"auto_candidate_modes": ["fft_texture_suppression", "multi_scuff_fusion"], "use_ensemble_consensus": False},
        description="a scuffed rubbed patch with many faint scratches",
        normal_paths=[],
        requested="auto",
    )

    assert selected["scratch_morphology_class"] == "multi_scuff"
    assert selected["selected_refinement"] == "multi_scuff_fusion"


def test_support_constrained_fusion_removes_unsupported_component() -> None:
    image = Image.new("RGB", (128, 96), (120, 120, 120))
    base = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(base)
    draw.rectangle((36, 36, 60, 56), fill=255)
    draw.rectangle((86, 20, 92, 82), fill=255)
    heatmap = np.zeros((96, 128), dtype=np.float32)
    heatmap[32:62, 32:66] = 1.0

    refined, params = support_constrained_fusion_mask(
        base,
        heatmap,
        image,
        [],
        (20, 12, 110, 88),
        image.size,
        min_component_area=4,
        support_percentile=80.0,
        support_dilate_radius=2,
        min_support_overlap=0.20,
        min_component_heat=0.20,
        min_keep_fraction=0.20,
        grain_reject_angle_degrees=14.0,
    )

    arr = np.asarray(refined.convert("L"), dtype=np.uint8) > 0
    assert arr[40:52, 42:54].any()
    assert not arr[24:76, 88:91].any()
    assert params["support_constrained_reverted"] is False
    assert params["support_constrained_rejected_low_support"] >= 1


def test_support_constrained_fusion_reverts_when_too_destructive() -> None:
    image = Image.new("RGB", (128, 96), (120, 120, 120))
    base = Image.new("L", image.size, 0)
    ImageDraw.Draw(base).rectangle((36, 36, 90, 56), fill=255)
    heatmap = np.zeros((96, 128), dtype=np.float32)

    refined, params = support_constrained_fusion_mask(
        base,
        heatmap,
        image,
        [],
        (20, 12, 110, 88),
        image.size,
        min_component_area=4,
        support_percentile=80.0,
        support_dilate_radius=2,
        min_support_overlap=0.20,
        min_component_heat=0.20,
        min_keep_fraction=0.80,
        grain_reject_angle_degrees=14.0,
    )

    assert np.array_equal(np.asarray(refined.convert("L")), np.asarray(base.convert("L")))
    assert params["support_constrained_reverted"] is True


def test_sam_heatmap_fallback_emits_mask_without_sam(tmp_path: Path) -> None:
    normal = Image.new("RGB", (96, 64), (130, 130, 130))
    defect = normal.copy()
    ImageDraw.Draw(defect).line((18, 32, 78, 32), fill=(35, 35, 35), width=2)
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)

    artifacts = write_sam_heatmap_bbox_masks(
        defect,
        normal_path,
        "custom_part",
        "scratch",
        (8, 20, 88, 44),
        tmp_path,
        "sam_heatmap",
        auto={"allow_heatmap_fallback": True, "heatmap_mask_percentile": 90.0},
        text_hint="a dark horizontal scratch",
        padding=3,
    )

    assert Path(artifacts["parameters"]["heatmap_path"]).exists()
    assert artifacts["parameters"]["sam_provider"] in {"heatmap_fallback", "sam", "sam2"}
    assert Path(artifacts["refined_mask_path"]).exists()


def test_delta_deno_rejects_zero_heatmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGB", (64, 64), (120, 120, 120))

    def fake_delta_heatmap(*args: object, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros((64, 64), dtype=np.float32), {"delta_deno_provider": "fake"}

    monkeypatch.setattr("iadgen_v2.auto_masks.delta_deno_heatmap", fake_delta_heatmap)
    with pytest.raises(ValueError, match="all-zero heatmap"):
        write_delta_deno_bbox_masks(
            image,
            "custom_part",
            "scratch",
            (8, 18, 56, 34),
            tmp_path,
            "delta",
            auto={},
            text_hint="a scratch",
            padding=3,
        )


def test_sam2_predictor_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_calls = {"count": 0}

    def fake_build_sam2(model_cfg: str, checkpoint: str, device: str) -> object:
        build_calls["count"] += 1
        return {"model_cfg": model_cfg, "checkpoint": checkpoint, "device": device}

    class FakePredictor:
        def __init__(self, model: object) -> None:
            self.model = model

        def set_image(self, image: np.ndarray) -> None:
            self.image_shape = image.shape

        def predict(self, **_: object) -> tuple[np.ndarray, np.ndarray, None]:
            mask = np.zeros((1, 32, 32), dtype=bool)
            mask[:, 8:16, 8:24] = True
            return mask, np.asarray([0.9], dtype=np.float32), None

    sam2_pkg = types.ModuleType("sam2")
    build_module = types.ModuleType("sam2.build_sam")
    predictor_module = types.ModuleType("sam2.sam2_image_predictor")
    build_module.build_sam2 = fake_build_sam2
    predictor_module.SAM2ImagePredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "sam2", sam2_pkg)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_module)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", predictor_module)
    auto_masks._SAM2_PREDICTOR_CACHE.clear()

    image = Image.new("RGB", (32, 32), (100, 100, 100))
    auto = {"sam2_checkpoint": str(tmp_path / "sam2.pt"), "sam2_model_cfg": "tiny.yaml", "sam_device": "cpu"}
    first, _ = auto_masks._predict_sam2_mask(image, (4, 4, 28, 28), [[12, 12]], [1], auto)
    second, _ = auto_masks._predict_sam2_mask(image, (4, 4, 28, 28), [[12, 12]], [1], auto)

    assert first is not None
    assert second is not None
    assert build_calls["count"] == 1


def test_delta_deno_restores_attention_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    original_attn2 = object()
    original_attn1 = object()
    captured: dict[str, object] = {}

    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool = True) -> object:
            ids = [11, 22, 33] if add_special_tokens else [22]
            return types.SimpleNamespace(input_ids=ids)

    class FakeUnet:
        def __init__(self) -> None:
            self.attn_processors = {
                "up_blocks.0.attn2.processor": original_attn2,
                "up_blocks.0.attn1.processor": original_attn1,
            }

        def set_attn_processor(self, processors: dict[str, object]) -> None:
            self.attn_processors = processors

    class FakePipe:
        def __init__(self) -> None:
            self.unet = FakeUnet()
            self.tokenizer = FakeTokenizer()
            self.safety_checker = None

        def to(self, device: str) -> "FakePipe":
            self.device = device
            return self

        def __call__(self, **_: object) -> None:
            grid = torch.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
            for processor in self.unet.attn_processors.values():
                if hasattr(processor, "attn_maps"):
                    processor.attn_maps.append(grid)

    fake_pipe = FakePipe()

    class FakePipelineClass:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakePipe:
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return fake_pipe

    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.StableDiffusionImg2ImgPipeline = FakePipelineClass
    models_module = types.ModuleType("diffusers.models")
    attention_module = types.ModuleType("diffusers.models.attention_processor")
    attention_module.AttnProcessor = object
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_module)
    monkeypatch.setitem(sys.modules, "diffusers.models", models_module)
    monkeypatch.setitem(sys.modules, "diffusers.models.attention_processor", attention_module)
    auto_masks._SD_PIPELINE_CACHE.clear()

    heatmap, params = auto_masks.delta_deno_heatmap(
        Image.new("RGB", (64, 64), (120, 120, 120)),
        (0, 0, 64, 64),
        "custom_part",
        "scratch",
        {
            "delta_deno_model": "fake/sd",
            "delta_deno_cache_dir": "fake-cache",
            "delta_deno_local_files_only": True,
            "sd_device": "cpu",
            "delta_deno_num_inference_steps": 3,
        },
        "a scratch on the surface",
    )

    assert np.isfinite(heatmap).all()
    assert params["delta_deno_target_indices"] == [1]
    assert fake_pipe.unet.attn_processors == {
        "up_blocks.0.attn2.processor": original_attn2,
        "up_blocks.0.attn1.processor": original_attn1,
    }
    assert captured["model_id"] == "fake/sd"
    assert captured["kwargs"]["cache_dir"] == "fake-cache"
    assert captured["kwargs"]["local_files_only"] is True


def test_normal_anomaly_refinement_uses_normal_texture_stats(tmp_path: Path) -> None:
    normal_paths: list[Path] = []
    for index in range(3):
        normal = Image.new("RGB", (96, 64), (130 + index, 120, 110))
        draw = ImageDraw.Draw(normal)
        for x in range(8, 96, 12):
            draw.line((x, 0, x + 2, 64), fill=(95, 85, 75), width=1)
        path = tmp_path / f"normal_{index}.png"
        normal.save(path)
        normal_paths.append(path)
    defect = Image.open(normal_paths[0]).convert("RGB")
    ImageDraw.Draw(defect).line((10, 42, 86, 28), fill=(235, 235, 210), width=2)

    artifacts = write_normal_anomaly_bbox_masks(
        defect,
        normal_paths,
        "custom_part",
        "scratch",
        (4, 18, 92, 52),
        tmp_path,
        "normal_anomaly",
        auto={
            "normal_anomaly_percentile": 94.0,
            "normal_anomaly_max_normals": 3,
            "normal_anomaly_max_components": 8,
            "normal_anomaly_envelope": True,
            "normal_anomaly_envelope_mode": "adaptive",
            "normal_anomaly_envelope_radius": 3,
            "normal_anomaly_envelope_percentile": 80.0,
            "normal_anomaly_suppress_parallel_texture": True,
            "min_component_area": 3,
        },
        text_hint="a light diagonal scratch across the wood surface",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert artifacts["parameters"]["kind"] == "normal_texture_anomaly"
    assert artifacts["parameters"]["normal_anomaly_normals_used"] == 3
    assert artifacts["parameters"]["normal_anomaly_envelope"] is True
    assert artifacts["parameters"]["normal_anomaly_envelope_kind"] in {"scratch_band", "patch", "patch_fallback"}
    assert refined.any()
    assert artifacts["qc"]["status"] in {"pass", "warning"}


def test_dinov2_memory_refinement_uses_patch_memory_heatmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    normal = Image.new("RGB", (96, 64), (130, 130, 130))
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)
    defect = normal.copy()
    ImageDraw.Draw(defect).line((12, 40, 82, 26), fill=(235, 235, 235), width=3)

    def fake_heatmap(*args: object, **kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        heatmap = np.zeros((64, 96), dtype=np.float32)
        for index, x in enumerate(range(12, 83)):
            y = 40 - index // 5
            heatmap[max(0, y - 1) : min(64, y + 2), x] = 1.0
        return heatmap, {
            "dinov2_provider": "fake_dinov2_memory_bank",
            "dinov2_available": True,
            "dinov2_model": "fake",
            "dinov2_normals_used": 1,
        }

    monkeypatch.setattr("iadgen_v2.auto_masks.dinov2_memory_heatmap", fake_heatmap)
    artifacts = write_dinov2_memory_bbox_masks(
        defect,
        [normal_path],
        "custom_part",
        "scratch",
        (4, 18, 92, 52),
        tmp_path,
        "dinov2",
        auto={
            "dinov2_percentile": 90.0,
            "dinov2_max_components": 8,
            "dinov2_envelope": True,
            "dinov2_envelope_radius": 3,
            "dinov2_envelope_percentile": 70.0,
            "dinov2_max_box_fraction": 0.30,
            "min_component_area": 3,
        },
        text_hint="a light diagonal scratch",
        padding=3,
    )

    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert artifacts["parameters"]["kind"] == "dinov2_memory"
    assert artifacts["parameters"]["dinov2_provider"] == "fake_dinov2_memory_bank"
    assert refined.any()
    assert artifacts["qc"]["status"] in {"pass", "warning"}


def test_soft_patch_refinement_merges_scratch_evidence_without_rectangle(tmp_path: Path) -> None:
    normal = Image.new("RGB", (96, 64), (130, 130, 130))
    defect = normal.copy()
    draw = ImageDraw.Draw(defect)
    for y in (24, 30, 37):
        draw.line((18, y, 76, y + 2), fill=(55, 55, 55), width=2)
    draw.ellipse((40, 29, 47, 35), fill=(60, 60, 60))
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)

    artifacts = write_soft_patch_bbox_masks(
        defect,
        normal_path,
        "custom_part",
        "scratch",
        (10, 16, 88, 48),
        tmp_path,
        "soft",
        auto={
            "soft_patch_percentile": 88.0,
            "soft_patch_close_radius": 2,
            "soft_patch_max_box_fraction": 0.25,
            "min_component_area": 4,
        },
        text_hint="a scratched/scuffed area with several dark scratch marks",
        padding=3,
    )

    box = np.asarray(Image.open(artifacts["box_mask_path"]).convert("L"), dtype=np.uint8) > 0
    refined = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert artifacts["parameters"]["kind"] == "soft_patch"
    assert refined.any()
    assert int(refined.sum()) < int(box.sum())
    assert not np.array_equal(box, refined)
    assert artifacts["qc"]["status"] in {"pass", "warning"}


def test_linear_refinement_turns_speckles_into_connected_scratch(tmp_path: Path) -> None:
    image = Image.new("RGB", (96, 64), (130, 130, 130))
    draw = ImageDraw.Draw(image)
    for x in range(20, 78, 6):
        draw.ellipse((x, 28 + (x % 3), x + 2, 30 + (x % 3)), fill=(35, 35, 35))

    artifacts = write_linear_refined_bbox_masks(
        image,
        "custom_part",
        "scratch",
        (10, 18, 88, 44),
        tmp_path,
        "linear",
        clip_to_surface=False,
        text_hint="a dark horizontal scratch",
        percentile=90.0,
        padding=3,
    )

    assert artifacts["parameters"]["kind"] == "linearized_scratch"
    assert artifacts["qc"]["component_count"] == 1
    assert artifacts["qc"]["status"] in {"pass", "warning"}


def test_multi_linear_refinement_keeps_multiple_scratch_strokes(tmp_path: Path) -> None:
    image = Image.new("RGB", (96, 64), (130, 130, 130))
    draw = ImageDraw.Draw(image)
    for y in (24, 32, 40):
        for x in range(18, 80, 8):
            draw.ellipse((x, y, x + 2, y + 2), fill=(35, 35, 35))

    artifacts = write_multi_linear_refined_bbox_masks(
        image,
        "custom_part",
        "scratch",
        (10, 16, 88, 48),
        tmp_path,
        "multi",
        clip_to_surface=False,
        text_hint="several dark horizontal scratches",
        percentile=90.0,
        padding=3,
        max_lines=5,
    )

    assert artifacts["parameters"]["kind"] == "multi_linearized_scratch"
    assert artifacts["parameters"]["line_count"] >= 2
    assert artifacts["qc"]["status"] in {"pass", "warning"}


def test_scratch_band_clean_removes_detached_blobs() -> None:
    normal = Image.new("L", (128, 96), 0)
    multi = Image.new("L", (128, 96), 0)
    draw_normal = ImageDraw.Draw(normal)
    draw_multi = ImageDraw.Draw(multi)
    draw_normal.line((10, 70, 112, 34), fill=255, width=5)
    draw_multi.line((10, 70, 112, 34), fill=255, width=2)
    draw_normal.rectangle((18, 18, 32, 30), fill=255)
    draw_normal.rectangle((82, 76, 96, 88), fill=255)

    cleaned, params = auto_masks.scratch_band_clean_mask(
        normal,
        multi,
        (0, 0, 128, 96),
        (128, 96),
        radius=5,
        min_component_area=4,
        max_area_fraction=0.08,
    )
    arr = np.asarray(cleaned, dtype=np.uint8) > 0

    assert params["scratch_band_clean_pixels"] < int((np.asarray(normal) > 0).sum())
    assert arr[52, 64]
    assert not arr[22, 24]
    assert not arr[82, 88]


def test_scuff_eval_tight_is_smaller_than_training_mask(tmp_path: Path) -> None:
    image = Image.new("RGB", (96, 64), (130, 130, 130))
    base = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(base)
    for x in (18, 28, 38, 52):
        draw.line((x, 18, x + 4, 44), fill=255, width=2)
    refined_path = tmp_path / "base_refined.png"
    inpaint_path = tmp_path / "base_inpaint.png"
    box_path = tmp_path / "base_box.png"
    base.save(refined_path)
    base.save(inpaint_path)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((10, 12, 72, 50), fill=255)
    box.save(box_path)

    artifacts = auto_masks._attach_mask_variants(
        image=image,
        region=(10, 12, 72, 50),
        output_dir=tmp_path / "variants",
        artifact_stem="scuff",
        artifacts={
            "box_mask_path": str(box_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path),
            "parameters": {"kind": "scuff_cluster"},
            "qc": {"status": "pass"},
            "scratch_morphology_class": "multi_scuff",
            "selected_refinement": "scuff_cluster",
        },
        auto={
            "scuff_eval_tight_radius": 0,
            "scuff_eval_tight_max_components": 2,
            "scuff_eval_tight_max_box_fraction": 0.05,
            "training_medium_radius": 4,
            "training_wide_radius": 9,
            "scuff_inpaint_core_radius": 3,
            "scuff_inpaint_halo_radius": 8,
            "scuff_inpaint_blur": 2.5,
            "scuff_inpaint_core_weight": 0.65,
            "write_variant_overlays": False,
        },
    )
    eval_area = np.asarray(Image.open(artifacts["mask_variant_paths"]["eval_tight"]).convert("L")) > 0
    train_area = np.asarray(Image.open(artifacts["mask_variant_paths"]["training_medium"]).convert("L")) > 0
    wide_area = np.asarray(Image.open(artifacts["mask_variant_paths"]["training_wide"]).convert("L")) > 0

    assert int(eval_area.sum()) < int(train_area.sum())
    assert int(train_area.sum()) < int(wide_area.sum())
    assert artifacts["training_mask_path"] == artifacts["mask_variant_paths"]["training_soft"]
    assert artifacts["label_policy"]["label_policy"] == "soft_mask_only"
    assert artifacts["parameters"]["training_mask_variant"] == "training_soft"
    assert artifacts["parameters"]["mask_variants"]["inpaint_soft"]["adaptive"] is True
    assert artifacts["parameters"]["mask_variants"]["inpaint_soft"]["morphology"] == "multi_scuff"


def test_precision_tight_eval_trims_small_chip_without_shrinking_training_mask(tmp_path: Path) -> None:
    image = Image.new("RGB", (128, 128), (130, 130, 130))
    base = Image.new("L", image.size, 0)
    ImageDraw.Draw(base).rectangle((24, 24, 104, 96), fill=255)
    refined_path = tmp_path / "chip_refined.png"
    inpaint_path = tmp_path / "chip_inpaint.png"
    box_path = tmp_path / "chip_box.png"
    heatmap_path = tmp_path / "chip_heatmap.png"
    base.save(refined_path)
    base.save(inpaint_path)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((16, 16, 112, 112), fill=255)
    box.save(box_path)
    heat = Image.new("L", image.size, 0)
    ImageDraw.Draw(heat).rectangle((30, 28, 58, 54), fill=255)
    ImageDraw.Draw(heat).rectangle((24, 24, 104, 96), outline=80, width=2)
    heat.save(heatmap_path)

    artifacts = auto_masks._attach_mask_variants(
        image=image,
        category="bottle",
        defect_type="broken_small",
        region=(16, 16, 112, 112),
        output_dir=tmp_path / "variants_precision",
        artifact_stem="chip",
        artifacts={
            "box_mask_path": str(box_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path),
            "parameters": {"kind": "nearest_normal_residual", "heatmap_path": str(heatmap_path)},
            "qc": {
                "status": "pass",
                "mask_area": 5913,
                "mask_area_fraction": 0.36,
                "mask_to_qwen_box_fraction": 0.64,
            },
            "scratch_morphology_class": "not_scratch",
            "selected_refinement": "nearest_normal_residual",
            "candidate_heatmap_paths": {"nearest_normal_residual": str(heatmap_path)},
        },
        auto={
            "precision_tight_eval_enabled": True,
            "bottle_small_precision_max_box_fraction": 0.12,
            "bottle_small_precision_max_image_fraction": 0.20,
            "bottle_small_precision_min_keep_fraction": 0.35,
            "training_medium_radius": 4,
            "training_wide_radius": 9,
            "write_variant_overlays": False,
        },
    )

    eval_arr = np.asarray(Image.open(artifacts["mask_variant_paths"]["eval_tight"]).convert("L")) > 0
    train_arr = np.asarray(Image.open(artifacts["mask_variant_paths"]["training_medium"]).convert("L")) > 0
    base_arr = np.asarray(base, dtype=np.uint8) > 0

    assert int(eval_arr.sum()) < int(base_arr.sum())
    assert int(train_arr.sum()) > int(base_arr.sum())
    assert artifacts["parameters"]["mask_variants"]["eval_tight"]["precision_tightened"] is True


def test_bottle_large_eval_tight_expands_from_heat_support(tmp_path: Path) -> None:
    image = Image.new("RGB", (128, 128), (130, 130, 130))
    base = Image.new("L", image.size, 0)
    ImageDraw.Draw(base).rectangle((48, 48, 60, 60), fill=255)
    refined_path = tmp_path / "large_refined.png"
    inpaint_path = tmp_path / "large_inpaint.png"
    box_path = tmp_path / "large_box.png"
    heatmap_path = tmp_path / "large_heatmap.png"
    base.save(refined_path)
    base.save(inpaint_path)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((24, 24, 104, 104), fill=255)
    box.save(box_path)
    heat = Image.new("L", image.size, 0)
    ImageDraw.Draw(heat).rectangle((34, 34, 88, 88), fill=220)
    ImageDraw.Draw(heat).rectangle((48, 48, 60, 60), fill=255)
    heat.save(heatmap_path)

    artifacts = auto_masks._attach_mask_variants(
        image=image,
        category="bottle",
        defect_type="broken_large",
        region=(24, 24, 104, 104),
        output_dir=tmp_path / "variants_large_recall",
        artifact_stem="large",
        artifacts={
            "box_mask_path": str(box_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path),
            "parameters": {"kind": "nearest_normal_residual", "heatmap_path": str(heatmap_path)},
            "qc": {
                "status": "pass",
                "mask_area": 169,
                "mask_area_fraction": 0.010,
                "mask_to_qwen_box_fraction": 0.026,
            },
            "scratch_morphology_class": "not_scratch",
            "selected_refinement": "nearest_normal_residual",
            "candidate_heatmap_paths": {"nearest_normal_residual": str(heatmap_path)},
        },
        auto={
            "bottle_large_recall_expand_enabled": True,
            "bottle_large_recall_expand_min_box_fraction": 0.15,
            "bottle_large_recall_expand_percentile": 70.0,
            "bottle_large_recall_expand_max_box_fraction": 0.30,
            "training_medium_radius": 4,
            "training_wide_radius": 9,
            "write_variant_overlays": False,
        },
    )

    eval_arr = np.asarray(Image.open(artifacts["mask_variant_paths"]["eval_tight"]).convert("L")) > 0
    base_arr = np.asarray(base, dtype=np.uint8) > 0

    assert int(eval_arr.sum()) > int(base_arr.sum())
    assert artifacts["parameters"]["mask_variants"]["eval_tight"]["recall_expanded"] is True


def test_zipper_fabric_eval_mvtec_expands_without_changing_generation_core(tmp_path: Path) -> None:
    image = Image.new("RGB", (128, 160), (130, 130, 130))
    base = Image.new("L", image.size, 0)
    ImageDraw.Draw(base).rectangle((94, 50, 106, 98), fill=255)
    refined_path = tmp_path / "fabric_refined.png"
    inpaint_path = tmp_path / "fabric_inpaint.png"
    box_path = tmp_path / "fabric_box.png"
    base.save(refined_path)
    base.save(inpaint_path)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((86, 0, 116, 160), fill=255)
    box.save(box_path)

    artifacts = auto_masks._attach_mask_variants(
        image=image,
        category="zipper",
        defect_type="fabric_border",
        region=(86, 0, 116, 160),
        output_dir=tmp_path / "variants_fabric_eval",
        artifact_stem="fabric",
        artifacts={
            "box_mask_path": str(box_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path),
            "parameters": {"kind": "sam2_heatmap"},
            "qc": {
                "status": "pass",
                "mask_area": 637,
                "mask_area_fraction": 0.031,
                "mask_to_qwen_box_fraction": 0.13,
                "component_count": 1,
                "reasons": [],
            },
            "scratch_morphology_class": "not_scratch",
            "selected_refinement": "sam2_heatmap",
        },
        auto={
            "eval_mvtec_enabled": True,
            "eval_mask_variant": "eval_mvtec",
            "zipper_fabric_eval_mvtec_vertical_expand_enabled": True,
            "zipper_fabric_eval_mvtec_sam_vertical_radius": 6,
            "zipper_fabric_eval_mvtec_max_image_fraction": 0.08,
            "training_medium_radius": 4,
            "training_wide_radius": 9,
            "write_variant_overlays": False,
        },
    )

    generation_core = np.asarray(Image.open(artifacts["generation_core_mask_path"]).convert("L")) > 0
    eval_mvtec = np.asarray(Image.open(artifacts["benchmark_eval_mask_path"]).convert("L")) > 0
    base_arr = np.asarray(base, dtype=np.uint8) > 0

    assert np.array_equal(generation_core, base_arr)
    assert int(eval_mvtec.sum()) > int(generation_core.sum())
    assert artifacts["eval_mask_path"] == artifacts["benchmark_eval_mask_path"]
    assert artifacts["parameters"]["mask_variants"]["eval_mvtec"]["eval_mvtec_reason"] == "zipper_fabric_border_vertical_support"


def test_multi_scuff_training_soft_uses_calibrated_heatmap_ensemble(tmp_path: Path) -> None:
    image = Image.new("RGB", (96, 64), (130, 130, 130))
    base = Image.new("L", image.size, 0)
    ImageDraw.Draw(base).rectangle((26, 26, 32, 34), fill=255)
    refined_path = tmp_path / "base_refined.png"
    inpaint_path = tmp_path / "base_inpaint.png"
    box_path = tmp_path / "base_box.png"
    heatmap_path = tmp_path / "patchcore_heatmap.png"
    base.save(refined_path)
    base.save(inpaint_path)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((12, 14, 70, 48), fill=255)
    box.save(box_path)
    heat = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(heat)
    draw.rectangle((20, 20, 54, 42), fill=150)
    draw.rectangle((26, 26, 34, 35), fill=255)
    heat.save(heatmap_path)

    artifacts = auto_masks._attach_mask_variants(
        image=image,
        region=(12, 14, 70, 48),
        output_dir=tmp_path / "variants_soft",
        artifact_stem="scuff_soft",
        artifacts={
            "box_mask_path": str(box_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path),
            "parameters": {"kind": "patchcore_guided"},
            "qc": {"status": "pass"},
            "scratch_morphology_class": "multi_scuff",
            "selected_refinement": "patchcore_guided",
            "candidate_refined_paths": {"patchcore_guided": str(refined_path)},
            "candidate_heatmap_paths": {"patchcore_guided": str(heatmap_path)},
        },
        auto={
            "calibrated_soft_scuff_ensemble": True,
            "scuff_eval_tight_radius": 0,
            "scuff_eval_tight_max_components": 2,
            "scuff_eval_tight_max_box_fraction": 0.05,
            "training_medium_radius": 4,
            "training_wide_radius": 9,
            "soft_scuff_possible_percentile": 45.0,
            "soft_scuff_core_percentile": 92.0,
            "soft_scuff_min_possible_probability": 0.08,
            "soft_scuff_min_core_probability": 0.58,
            "soft_scuff_possible_max_box_fraction": 0.32,
            "soft_scuff_core_max_box_fraction": 0.10,
            "scuff_inpaint_core_radius": 3,
            "scuff_inpaint_halo_radius": 8,
            "scuff_inpaint_blur": 2.5,
            "write_variant_overlays": False,
        },
    )

    training_soft = np.asarray(Image.open(artifacts["mask_variant_paths"]["training_soft"]).convert("L"), dtype=np.uint8)
    positive_core = np.asarray(Image.open(artifacts["mask_variant_paths"]["positive_core"]).convert("L"), dtype=np.uint8) > 0
    possible_region = np.asarray(Image.open(artifacts["mask_variant_paths"]["possible_region"]).convert("L"), dtype=np.uint8) > 0
    base_arr = np.asarray(base, dtype=np.uint8) > 0
    broad_support = np.zeros_like(base_arr, dtype=bool)
    broad_support[20:43, 20:55] = True

    assert artifacts["parameters"]["mask_variants"]["training_soft"]["soft_scuff_ensemble"] is True
    assert int(possible_region.sum()) > int(base_arr.sum())
    assert int(positive_core.sum()) >= int(base_arr.sum())
    assert float(training_soft[broad_support & ~base_arr].mean()) > 0.0
    assert artifacts["training_mask_path"] == artifacts["mask_variant_paths"]["training_soft"]


def test_scratch_band_inpaint_soft_is_adaptive_and_narrower_than_global_default(tmp_path: Path) -> None:
    image = Image.new("RGB", (128, 96), (130, 130, 130))
    base = Image.new("L", image.size, 0)
    ImageDraw.Draw(base).line((12, 72, 116, 34), fill=255, width=3)
    refined_path = tmp_path / "band_refined.png"
    inpaint_path = tmp_path / "band_inpaint.png"
    box_path = tmp_path / "band_box.png"
    base.save(refined_path)
    base.save(inpaint_path)
    box = Image.new("L", image.size, 0)
    ImageDraw.Draw(box).rectangle((0, 20, 128, 84), fill=255)
    box.save(box_path)

    artifacts = auto_masks._attach_mask_variants(
        image=image,
        region=(0, 20, 128, 84),
        output_dir=tmp_path / "variants_band",
        artifact_stem="band",
        artifacts={
            "box_mask_path": str(box_path),
            "refined_mask_path": str(refined_path),
            "inpaint_mask_path": str(inpaint_path),
            "parameters": {"kind": "scratch_band_clean"},
            "qc": {"status": "pass"},
            "scratch_morphology_class": "scratch_band",
            "selected_refinement": "scratch_band_clean",
        },
        auto={
            "training_medium_radius": 4,
            "training_wide_radius": 9,
            "scratch_band_inpaint_core_radius": 2,
            "scratch_band_inpaint_halo_radius": 5,
            "scratch_band_inpaint_blur": 1.5,
            "scratch_band_inpaint_core_weight": 0.78,
            "inpaint_variant_radius": 12,
            "inpaint_variant_blur": 2.0,
            "write_variant_overlays": False,
            "use_alpha_matting": False,
        },
    )
    adaptive = np.asarray(Image.open(artifacts["mask_variant_paths"]["inpaint_soft"]).convert("L")) > 0
    global_default = np.asarray(auto_masks._variant_mask(base, radius=12, blur=2.0).convert("L")) > 0

    assert artifacts["training_mask_path"] == artifacts["mask_variant_paths"]["training_medium"]
    assert artifacts["label_policy"]["label_policy"] == "hard_mask_ok"
    assert artifacts["parameters"]["training_mask_variant"] == "training_medium"
    assert artifacts["parameters"]["mask_variants"]["inpaint_soft"]["adaptive"] is True
    assert artifacts["parameters"]["mask_variants"]["inpaint_soft"]["morphology"] == "scratch_band"
    assert int(adaptive.sum()) < int(global_default.sum())


def test_multiple_scratch_description_favors_patch_candidates() -> None:
    image = Image.new("RGB", (96, 64), (130, 130, 130))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for y in (22, 29, 36):
        draw.line((18, y, 78, y + 1), fill=255, width=2)
    qc = {
        "status": "pass",
        "component_count": 3,
        "mask_area_fraction": 0.018,
        "mask_to_qwen_box_fraction": 0.14,
        "touching_border_count": 0,
        "reasons": [],
        "shrunk_region_xyxy": (18, 22, 78, 38),
    }
    description = "a scuffed area with multiple fine scratches"

    soft_score = auto_masks._score_candidate(mask, image, qc, "soft_patch", "scratch", description)
    normal_score = auto_masks._score_candidate(mask, image, qc, "normal_anomaly", "scratch", description)
    procedural_score = auto_masks._score_candidate(mask, image, qc, "procedural", "scratch", description)
    linear_score = auto_masks._score_candidate(mask, image, qc, "linear", "scratch", description)

    assert soft_score > procedural_score
    assert normal_score > procedural_score
    assert soft_score > linear_score


def test_residual_refinement_filters_noise_and_emits_qc(tmp_path: Path) -> None:
    normal = Image.new("RGB", (96, 64), (130, 130, 130))
    defect = normal.copy()
    ImageDraw.Draw(defect).line((18, 32, 78, 32), fill=(35, 35, 35), width=2)
    normal_path = tmp_path / "normal.png"
    normal.save(normal_path)

    artifacts = write_residual_refined_bbox_masks(
        defect,
        normal_path,
        "custom_part",
        "scratch",
        (8, 20, 88, 44),
        tmp_path,
        "residual",
        clip_to_surface=False,
        text_hint="a thin dark horizontal scratch",
        percentile=90.0,
        min_component_area=4,
        padding=3,
    )

    mask = np.asarray(Image.open(artifacts["refined_mask_path"]).convert("L"), dtype=np.uint8) > 0
    assert mask.any()
    assert artifacts["parameters"]["kind"] == "normal_residual"
    assert artifacts["qc"]["status"] in {"pass", "warning"}
    x1, y1, x2, y2 = artifacts["qc"]["shrunk_region_xyxy"]
    assert x2 - x1 < 96
    assert y2 - y1 < 64


class _FakeQwenExtractor:
    def __init__(self, **_: object) -> None:
        pass

    def analyze(self, image: Image.Image, prompt: str) -> dict[str, object]:
        assert "JSON only" in prompt
        assert "user description" in prompt
        return {
            "text": json.dumps(
                {
                    "bbox_xyxy": [10, 16, image.width - 10, 34],
                    "defect_type": "crack",
                    "confidence": 0.91,
                    "evidence": "visible narrow mark",
                }
            )
        }

    def close(self) -> None:
        pass


def _ready_qwen(**kwargs: object) -> QwenAvailability:
    return QwenAvailability(
        model_id=str(kwargs.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")),
        cache_dir=str(kwargs.get("cache_dir", "")),
        cached_locally=True,
        qwen_vl_utils_available=True,
        transformers_available=True,
        free_gib=100.0,
        min_free_gib=float(kwargs.get("min_free_gib", 30.0)),
        local_files_only=bool(kwargs.get("local_files_only", True)),
    )


def _fixture_config(tmp_path: Path):
    dataset_root = tmp_path / "data" / "custom_mvtec"
    category = "custom_part"
    defect_type = "scratch"
    clean_dir = dataset_root / category / "train" / "good"
    test_dir = dataset_root / category / "test" / defect_type
    clean_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    clean = Image.new("RGB", (64, 64), (100, 100, 100))
    clean.save(clean_dir / "000.png")
    for index in range(3):
        image = clean.copy()
        draw = ImageDraw.Draw(image)
        draw.line((10, 22 + index, 54, 24 + index), fill=(45, 45, 45), width=2)
        image.save(test_dir / f"{index:03d}.png")
    config_path = tmp_path / "custom_auto_masks.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                f"  output_dir: {tmp_path / 'outputs'}",
                f"  report_dir: {tmp_path / 'reports'}",
                "dataset:",
                f"  root: {dataset_root}",
                "  download: false",
                "  seed: 11",
                "  adaptation_per_defect: 2",
                "  targets:",
                "    custom_part: [scratch]",
                "generation:",
                "  samples_per_category: 1",
                "  seed: 11",
                "  device: cpu",
                "  prompt_by_defect: {scratch: scratch, crack: crack}",
                "models:",
                "  mock: {enabled: true}",
                "  sd15: {enabled: false}",
                "evaluation: {}",
                "auto_masks:",
                "  provider: qwen",
                "  qwen_model: Qwen/Qwen2.5-VL-3B-Instruct",
                "  qwen_cache_dir: model_cache/huggingface/hub",
                "  qwen_local_files_only: true",
                "  qwen_min_free_gib: 30.0",
                "  overwrite: true",
                "  min_box_area_ratio: 0.001",
                "  max_box_area_ratio: 0.5",
                "  mask_seed: 17",
                "  write_overlays: true",
                "  clip_to_surface: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
