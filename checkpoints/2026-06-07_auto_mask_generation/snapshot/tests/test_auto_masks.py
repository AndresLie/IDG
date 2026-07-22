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
    _select_mask_artifacts,
    parse_qwen_bbox_payload,
    run_auto_masks,
    support_constrained_fusion_mask,
    validate_bbox,
    write_delta_deno_bbox_masks,
    write_dinov2_memory_bbox_masks,
    write_fft_texture_suppression_bbox_masks,
    write_multi_scuff_fusion_bbox_masks,
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
from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.masks import write_refined_bbox_masks
from iadgen_v2.qwen_provider import QwenAvailability


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


def test_validate_bbox_clips_and_rejects_bad_area() -> None:
    assert validate_bbox((-5, 3, 110, 40), (100, 50), min_area_ratio=0.001, max_area_ratio=0.95) == (0, 3, 100, 40)

    with pytest.raises(ValueError, match="below minimum"):
        validate_bbox((1, 1, 2, 2), (100, 100), min_area_ratio=0.01, max_area_ratio=0.95)
    with pytest.raises(ValueError, match="above maximum"):
        validate_bbox((0, 0, 100, 100), (100, 100), min_area_ratio=0.001, max_area_ratio=0.5)


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


def test_auto_masks_write_mvtec_ground_truth_and_prepare_can_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _fixture_config(tmp_path)
    monkeypatch.setattr("iadgen_v2.auto_masks.qwen_availability", _ready_qwen)
    monkeypatch.setattr("iadgen_v2.auto_masks.QwenFeatureExtractor", _FakeQwenExtractor)

    metadata_path = run_auto_masks(config)
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    assert (config.report_dir / "auto_masks" / "qwen" / "contact_sheet_mask_variants_current.png").exists()
    assert (config.report_dir / "auto_masks" / "qwen" / "contact_sheet_candidate_comparison.png").exists()
    assert all(row["settings"]["mask_truth_source"] == "qwen_auto_refined_masks" for row in rows)
    assert all(row["defect_type"] == "scratch" for row in rows)
    assert all(row["qwen_defect_type"] == "crack" for row in rows)
    for row in rows:
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
