from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from iadgen_v2.config import AppConfig
from iadgen_v2.dataset import load_manifest
from iadgen_v2.masks import place_adaptation_mask
from iadgen_v2.records import GenerationRecord, append_record


def run_baseline(config: AppConfig, model: str) -> Path:
    if model not in {"mock", "sd15"}:
        raise ValueError(f"Unsupported Phase 1 baseline: {model}")
    manifest = load_manifest(config)
    output_dir = config.output_dir / "generated" / model
    metadata_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(output_dir / "failures", ignore_errors=True)
    shutil.rmtree(output_dir / "images", ignore_errors=True)
    shutil.rmtree(output_dir / "masks", ignore_errors=True)
    metadata_path.write_text("", encoding="utf-8")
    samples_per_category = int(config.data["generation"].get("samples_per_category", 1))
    base_seed = int(config.data["generation"].get("seed", 1337))
    placement_mode = str(config.data["generation"].get("placement_mode", "random_smoke"))
    generation_fingerprint = config.generation_fingerprint(model)
    split_fingerprint = str(manifest["split_spec_fingerprint"])
    (output_dir / "config_snapshot.yaml").write_text(config.path.read_text(encoding="utf-8"), encoding="utf-8")
    runner = _make_runner(config, model)
    failures = 0

    for target_key, target_data in manifest["targets"].items():
        category = str(target_data["category"])
        defect_type = str(target_data["defect_type"])
        references = list(target_data["adaptation"])
        backgrounds = list(target_data["clean_targets"])
        if not references or not backgrounds:
            raise ValueError(f"Prepared target contains no adaptation references or clean targets: {target_key}")
        for index in range(samples_per_category):
            seed = base_seed + index + _target_seed(target_key)
            reference = references[index % len(references)]
            background_path = Path(backgrounds[index % len(backgrounds)])
            background = Image.open(background_path).convert("RGB")
            stem = f"{category}_{defect_type}_{index:04d}"
            binary_mask_path, inpaint_mask_path, bbox, surface_coverage = place_adaptation_mask(
                background,
                Path(reference["mask_path"]),
                output_dir / "masks" / category,
                stem,
                seed,
                category=category,
                placement_mode=placement_mode,
            )
            image_path = output_dir / "images" / category / f"{stem}.png"
            settings = {
                "prompt": _prompt(config, defect_type),
                "negative_prompt": str(config.data["generation"].get("negative_prompt", "")),
                "binary_mask_path": str(binary_mask_path),
                "inpaint_mask_path": str(inpaint_mask_path),
                "target_bbox_xyxy": bbox,
                "placement_mode": placement_mode,
                "surface_coverage": surface_coverage,
                "split": "adaptation-source-only",
                "split_spec_fingerprint": split_fingerprint,
                "generation_fingerprint": generation_fingerprint,
            }
            started = time.perf_counter()
            try:
                telemetry = runner.generate(
                    background_path=background_path,
                    source_image_path=Path(reference["image_path"]),
                    source_mask_path=Path(reference["mask_path"]),
                    binary_mask_path=binary_mask_path,
                    inpaint_mask_path=inpaint_mask_path,
                    output_path=image_path,
                    prompt=settings["prompt"],
                    negative_prompt=settings["negative_prompt"],
                    seed=seed,
                )
                settings.update(telemetry)
                error = None
            except Exception:
                failures += 1
                error = traceback.format_exc()
                failure = output_dir / "failures" / category / f"{stem}.log"
                failure.parent.mkdir(parents=True, exist_ok=True)
                failure.write_text(error, encoding="utf-8")
            append_record(
                metadata_path,
                GenerationRecord(
                    category=category,
                    defect_type=defect_type,
                    background_path=str(background_path),
                    source_image_path=str(reference["image_path"]),
                    source_mask_path=str(reference["mask_path"]),
                    insertion_mask_path=str(binary_mask_path),
                    output_path=str(image_path),
                    model=model,
                    seed=seed,
                    latency_sec=time.perf_counter() - started if error is None else None,
                    settings=settings,
                    error=error,
                ),
            )
    if failures:
        raise RuntimeError(
            f"{model} generation failed for {failures} required sample(s); inspect {output_dir / 'failures'}"
        )
    return metadata_path


class MockRunner:
    def generate(
        self,
        *,
        background_path: Path,
        source_image_path: Path,
        source_mask_path: Path,
        binary_mask_path: Path,
        inpaint_mask_path: Path,
        output_path: Path,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> dict[str, object]:
        background = Image.open(background_path).convert("RGB")
        binary = Image.open(binary_mask_path).convert("L").point(lambda value: 255 if value > 0 else 0)
        bbox = binary.getbbox()
        if bbox is None:
            raise ValueError(f"Empty insertion mask: {inpaint_mask_path}")
        source = Image.open(source_image_path).convert("RGB")
        source_mask = Image.open(source_mask_path).convert("L")
        source_box = source_mask.getbbox()
        if source_box is None:
            raise ValueError(f"Empty source mask: {source_mask_path}")
        patch = source.crop(source_box).resize((bbox[2] - bbox[0], bbox[3] - bbox[1]), Image.Resampling.LANCZOS)
        patch = ImageEnhance.Contrast(patch).enhance(1.05)
        local_alpha = binary.crop(bbox).filter(ImageFilter.GaussianBlur(radius=1.0))
        result = background.copy()
        result.paste(patch, bbox[:2], local_alpha)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        return {"device": "cpu", "dtype": "uint8"}


class SD15TextOnlyRunner:
    def __init__(self, config: dict[str, Any], device: str) -> None:
        self.config = config
        self.device = device
        self._pipe: Any | None = None

    def generate(
        self,
        *,
        background_path: Path,
        source_image_path: Path,
        source_mask_path: Path,
        binary_mask_path: Path,
        inpaint_mask_path: Path,
        output_path: Path,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> dict[str, object]:
        torch = _torch()
        torch.cuda.reset_peak_memory_stats()
        pipe = self._pipeline()
        size = int(self.config.get("image_size", 512))
        background = Image.open(background_path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        mask = Image.open(inpaint_mask_path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        control_image = _canny_image(background)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=background,
            mask_image=mask,
            control_image=control_image,
            controlnet_conditioning_scale=float(self.config.get("controlnet_conditioning_scale", 0.65)),
            guidance_scale=float(self.config.get("guidance_scale", 7.5)),
            num_inference_steps=int(self.config.get("num_inference_steps", 20)),
            strength=float(self.config.get("strength", 0.55)),
            generator=generator,
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_path)
        return {
            "device": self.device,
            "dtype": str(getattr(torch, str(self.config.get("dtype", "float16")))).replace("torch.", ""),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        }

    def _pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        torch = _torch()
        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The SD1.5 baseline requires CUDA; run the mock backend for CPU plumbing validation.")
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

        dtype = getattr(torch, str(self.config.get("dtype", "float16")))
        local_only = bool(self.config.get("local_files_only", True))
        controlnet = ControlNetModel.from_pretrained(
            self.config["controlnet_model"], torch_dtype=dtype, local_files_only=local_only
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.config["base_model"], controlnet=controlnet, torch_dtype=dtype, local_files_only=local_only
        )
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=False)
        self._pipe = pipe
        return pipe


def _make_runner(config: AppConfig, model: str) -> MockRunner | SD15TextOnlyRunner:
    if model == "mock":
        return MockRunner()
    configured_device = str(config.data["generation"].get("device", "auto"))
    torch = _torch()
    device = "cuda" if configured_device == "auto" and torch.cuda.is_available() else configured_device
    model_config = dict(config.data["models"]["sd15"])
    model_config["image_size"] = config.data["generation"].get("image_size", 512)
    return SD15TextOnlyRunner(model_config, device)


def _prompt(config: AppConfig, defect_type: str) -> str:
    return str(config.data["generation"].get("prompt_by_defect", {}).get(defect_type, f"a realistic {defect_type}"))


def _target_seed(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def _torch():
    import torch

    return torch


def _canny_image(image: Image.Image) -> Image.Image:
    try:
        import cv2

        edges = cv2.Canny(np.asarray(image), 100, 200)
        return Image.fromarray(edges).convert("RGB")
    except ImportError:
        return image.filter(ImageFilter.FIND_EDGES).convert("RGB")
