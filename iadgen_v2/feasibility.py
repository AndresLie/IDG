from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path

from iadgen_v2.config import AppConfig
from iadgen_v2.records import write_json


def write_feasibility_report(config: AppConfig) -> Path:
    packages = {name: _package_status(name) for name in ("torch", "diffusers", "transformers", "cv2")}
    hardware: dict[str, object] = {"python": platform.python_version(), "platform": platform.platform()}
    if packages["torch"]["available"]:
        import torch

        cuda = torch.cuda.is_available()
        hardware.update({"torch_version": torch.__version__, "cuda_available": cuda})
        if cuda:
            hardware.update(
                {
                    "cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                }
            )
    else:
        hardware["cuda_available"] = False

    model_status = {}
    for key in ("base_model", "controlnet_model"):
        model_id = config.data["models"]["sd15"][key]
        model_status[key] = {"id": model_id, "cached_locally": _is_hf_cached(model_id)}
    diffusion_import = _sd15_pipeline_import_status()
    measured_baseline = (config.output_dir / "generated" / "sd15" / "metadata.jsonl").exists()
    report = {
        "hardware": hardware,
        "packages": packages,
        "sd15": model_status,
        "sd15_pipeline_import": diffusion_import,
        "measured_sd15_baseline_available": measured_baseline,
        "real_sd15_ready": bool(
            hardware.get("cuda_available")
            and diffusion_import["available"]
            and all(value["cached_locally"] for value in model_status.values())
        ),
    }
    path = config.report_dir / "feasibility.json"
    write_json(path, report)
    _write_markdown(config.report_dir / "feasibility.md", report)
    return path


def _package_status(name: str) -> dict[str, object]:
    found = importlib.util.find_spec(name) is not None
    return {"available": found}


def _is_hf_cached(model_id: str) -> bool:
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = cache_home / "hub"
    model_dir = hub / f"models--{model_id.replace('/', '--')}"
    return model_dir.exists()


def _sd15_pipeline_import_status() -> dict[str, object]:
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

        return {"available": bool(ControlNetModel and StableDiffusionControlNetInpaintPipeline)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _write_markdown(path: Path, report: dict[str, object]) -> None:
    hardware = report["hardware"]
    sd15 = report["sd15"]
    lines = [
        "# Phase 1 Feasibility",
        "",
        f"- CUDA available: `{hardware.get('cuda_available', False)}`",
        f"- CUDA device: `{hardware.get('cuda_device_name', 'not detected')}`",
        f"- SD1.5 base cached: `{sd15['base_model']['cached_locally']}`",
        f"- ControlNet cached: `{sd15['controlnet_model']['cached_locally']}`",
        f"- SD1.5 pipeline import ready: `{report['sd15_pipeline_import']['available']}`",
        f"- Real SD1.5 baseline ready: `{report['real_sd15_ready']}`",
        f"- Measured SD1.5 baseline metadata available: `{report['measured_sd15_baseline_available']}`",
        "",
        (
            "Execution prerequisites are available. See `sd15/summary.md` for measured smoke-run memory and latency."
            if report["measured_sd15_baseline_available"]
            else "The report checks execution prerequisites only; run generation for memory and latency evidence."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
