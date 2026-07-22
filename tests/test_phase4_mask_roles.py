from __future__ import annotations

from pathlib import Path

import pytest

from iadgen_v2.phase4 import _mask_path_for_profile, _mask_role_for_profile


def test_phase4_mask_role_rejects_benchmark_eval_mask_for_generation(tmp_path: Path) -> None:
    inpaint = tmp_path / "inpaint_soft.png"
    eval_mvtec = tmp_path / "eval_mvtec.png"
    core = tmp_path / "generation_core.png"
    for path in (inpaint, eval_mvtec, core):
        path.write_bytes(b"mask")
    row = {
        "refined_mask_path": str(core),
        "inpaint_mask_path": str(inpaint),
        "settings": {
            "generation_core_mask_path": str(core),
            "benchmark_eval_mask_path": str(eval_mvtec),
            "mask_variant_paths": {
                "generation_core": str(core),
                "inpaint_soft": str(inpaint),
                "eval_mvtec": str(eval_mvtec),
            },
        },
    }

    with pytest.raises(ValueError, match="benchmark-only eval mask"):
        _mask_role_for_profile(row, {"mask_variant": "eval_mvtec"}, {})


def test_phase4_mask_role_allows_explicit_benchmark_eval_ablation(tmp_path: Path) -> None:
    eval_mvtec = tmp_path / "eval_mvtec.png"
    eval_mvtec.write_bytes(b"mask")
    row = {
        "refined_mask_path": str(eval_mvtec),
        "settings": {
            "benchmark_eval_mask_path": str(eval_mvtec),
            "mask_variant_paths": {"eval_mvtec": str(eval_mvtec)},
        },
    }

    role = _mask_role_for_profile(
        row,
        {"mask_variant": "eval_mvtec"},
        {"allow_benchmark_eval_mask_for_generation": True},
    )

    assert role["path"] == str(eval_mvtec)
    assert role["role"] == "benchmark_eval"
    assert role["warning"] == "benchmark_eval_mask_used_for_generation"


def test_phase4_mask_path_compatibility_wrapper_uses_generation_mask(tmp_path: Path) -> None:
    inpaint = tmp_path / "inpaint_soft.png"
    inpaint.write_bytes(b"mask")
    row = {
        "refined_mask_path": str(tmp_path / "core.png"),
        "inpaint_mask_path": str(inpaint),
        "settings": {"mask_variant_paths": {"inpaint_soft": str(inpaint)}},
    }

    assert _mask_path_for_profile(row, {"mask_variant": "inpaint_soft"}) == str(inpaint)

