from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch

from iadgen_v2.config import AppConfig
from iadgen_v2.qwen_provider import qwen_availability
from iadgen_v2.records import write_json


def write_phase6_preflight(config: AppConfig) -> Path:
    phase2 = dict(config.data.get("phase2", {}))
    phase3 = dict(config.data.get("phase3", {}))
    model_id = str(phase3.get("qwen_model", phase2.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct")))
    availability = qwen_availability(
        model_id=model_id,
        cache_dir=phase3.get("qwen_cache_dir", phase2.get("qwen_cache_dir")),
        local_files_only=bool(phase3.get("qwen_local_files_only", phase2.get("qwen_local_files_only", True))),
        min_free_gib=float(phase3.get("qwen_min_free_gib", phase2.get("qwen_min_free_gib", 30.0))),
    )
    sd15 = dict(config.data["models"]["sd15"])
    result: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else None,
        "qwen": availability.__dict__ | {"ready": availability.ready, "failure_reason": availability.failure_reason()},
        "sd15_base_model": sd15.get("base_model"),
        "sd15_controlnet_model": sd15.get("controlnet_model"),
        "diffusers_available": importlib.util.find_spec("diffusers") is not None,
        "transformers_available": importlib.util.find_spec("transformers") is not None,
        "next_action": "Set qwen_cache_dir/HF_HOME to storage with >=30 GiB free or pre-cache Qwen locally." if not availability.ready else "Qwen provider is ready for smoke extraction.",
    }
    path = config.report_dir / "phase6" / "preflight.json"
    write_json(path, result)
    return path
