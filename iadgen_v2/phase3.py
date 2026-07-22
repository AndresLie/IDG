from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from PIL import Image

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.dataset import load_manifest
from iadgen_v2.qwen_provider import QwenFeatureExtractor, qwen_availability, save_qwen_cache, write_availability
from iadgen_v2.records import write_json


PHASE3_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Phase3CacheRecord:
    category: str
    defect_type: str
    provider: str
    source_image_path: str
    source_mask_path: str
    masked_context_path: str
    anomaly_bbox_xyxy: tuple[int, int, int, int]
    normalized_bbox_xyxy: tuple[float, float, float, float]
    feature_cache_path: str
    prompt: str
    settings: dict[str, Any]
    uncertainty_mask_path: str | None = None


class GatedProjectionAdapter(torch.nn.Module):
    """Project cached VLM tokens to SD1.5 cross-attention width and append them."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 768,
        hidden_dim: int = 512,
        initial_gate_logit: float = -2.0,
    ) -> None:
        super().__init__()
        torch = _torch()
        nn = torch.nn
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(input_dim)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_gate_logit)))

    def project_tokens(self, vlm_tokens: Any) -> Any:
        if vlm_tokens.ndim != 3:
            raise ValueError(f"vlm_tokens must have shape [B, K, D], got {tuple(vlm_tokens.shape)}")
        if vlm_tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected VLM token width {self.input_dim}, got {vlm_tokens.shape[-1]}")
        torch = _torch()
        return torch.sigmoid(self.gate_logit) * self.projection(self.norm(vlm_tokens))

    def forward(self, clip_embeddings: Any, vlm_tokens: Any) -> Any:
        if clip_embeddings.ndim != 3:
            raise ValueError(f"clip_embeddings must have shape [B, 77, 768], got {tuple(clip_embeddings.shape)}")
        if clip_embeddings.shape[-1] != self.output_dim:
            raise ValueError(f"Expected CLIP width {self.output_dim}, got {clip_embeddings.shape[-1]}")
        projected = self.project_tokens(vlm_tokens).to(dtype=clip_embeddings.dtype)
        if projected.shape[0] != clip_embeddings.shape[0]:
            raise ValueError("CLIP embeddings and VLM tokens must use the same batch size")
        torch = _torch()
        return torch.cat([clip_embeddings, projected], dim=1)

    @property
    def gate_value(self) -> float:
        torch = _torch()
        return float(torch.sigmoid(self.gate_logit).detach().cpu())


def build_phase3_adaptation_cache(config: AppConfig, provider: str | None = None) -> Path:
    manifest = load_manifest(config)
    phase3 = _phase3_config(config)
    provider = provider or str(phase3.get("provider", "heuristic"))
    if provider not in {"heuristic", "qwen"}:
        raise ValueError(f"Unsupported Phase 3 provider: {provider}")
    output_dir = config.output_dir / "phase3" / provider
    cache_dir = output_dir / "cache"
    context_dir = output_dir / "masked_contexts"
    metadata_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("", encoding="utf-8")
    run_fingerprint = _phase3_fingerprint(config, provider)
    if provider == "qwen":
        availability = _qwen_availability(config)
        write_availability(output_dir / "qwen_availability.json", availability)
        if not availability.ready:
            raise RuntimeError(f"Qwen provider is not ready: {availability.failure_reason()}")
        extractor: QwenFeatureExtractor | None = QwenFeatureExtractor(
            model_id=str(phase3.get("qwen_model", config.data.get("phase2", {}).get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct"))),
            cache_dir=phase3.get("qwen_cache_dir", config.data.get("phase2", {}).get("qwen_cache_dir")),
            local_files_only=bool(phase3.get("qwen_local_files_only", config.data.get("phase2", {}).get("qwen_local_files_only", True))),
            device=str(phase3.get("qwen_device", phase3.get("device", "auto"))),
            torch_dtype=str(phase3.get("qwen_dtype", "auto")),
            token_count=int(phase3.get("token_count", config.data.get("phase2", {}).get("token_count", 16))),
            min_free_gib=float(phase3.get("qwen_min_free_gib", config.data.get("phase2", {}).get("qwen_min_free_gib", 30.0))),
        )
    else:
        extractor = None
    rows: list[Phase3CacheRecord] = []
    try:
        for target_key, target_data in manifest["targets"].items():
            category = str(target_data["category"])
            defect_type = str(target_data["defect_type"])
            for index, sample in enumerate(target_data["adaptation"]):
                image_path = Path(str(sample["image_path"]))
                mask_path = Path(str(sample.get("training_mask_path") or sample["mask_path"]))
                uncertainty_mask_path = str(sample.get("uncertainty_mask_path") or "")
                label_policy = _sample_label_policy(sample)
                preserve_soft_mask = label_policy.get("label_policy") == "soft_mask_only"
                image = Image.open(image_path).convert("RGB")
                raw_mask = Image.open(mask_path).convert("L")
                binary_mask = raw_mask.point(lambda value: 255 if value > 0 else 0)
                mask = raw_mask if preserve_soft_mask else binary_mask
                bbox = binary_mask.getbbox()
                if bbox is None:
                    raise ValueError(f"Empty adaptation mask: {mask_path}")
                stem = f"{category}_{defect_type}_{index:04d}"
                masked_context_path = context_dir / category / f"{stem}.png"
                masked_context = _masked_context(image, mask)
                masked_context_path.parent.mkdir(parents=True, exist_ok=True)
                masked_context.save(masked_context_path)
                feature_path = cache_dir / category / f"{stem}.pt"
                prompt = _prompt(category, defect_type)
                if extractor is None:
                    _write_heuristic_cache(image, masked_context, mask, bbox, prompt, feature_path, config)
                    cache_note = "heuristic placeholder cache; not Qwen hidden states"
                else:
                    result = extractor.analyze(masked_context, prompt)
                    save_qwen_cache(
                        feature_path,
                        tokens=result["tokens"],
                        metadata={
                            "provider": "qwen",
                            "model_id": extractor.model_id,
                            "prompt": prompt,
                            "qwen_text": result["text"],
                            "bbox_xyxy": bbox,
                            "normalized_bbox_xyxy": _normalize_region(bbox, image.size),
                            "tokens_shape": list(result["tokens"].shape),
                            "hidden_width": result["hidden_width"],
                            "selected_token_indices": result["selected_token_indices"],
                            "input_token_count": result["input_token_count"],
                        },
                    )
                    cache_note = "real Qwen hidden-state token cache"
                rows.append(
                    Phase3CacheRecord(
                        category=category,
                        defect_type=defect_type,
                        provider=provider,
                        source_image_path=str(image_path),
                        source_mask_path=str(mask_path),
                        uncertainty_mask_path=uncertainty_mask_path or None,
                        masked_context_path=str(masked_context_path),
                        anomaly_bbox_xyxy=bbox,
                        normalized_bbox_xyxy=_normalize_region(bbox, image.size),
                        feature_cache_path=str(feature_path),
                        prompt=prompt,
                        settings={
                            "phase3_fingerprint": run_fingerprint,
                            "phase3_schema_version": PHASE3_SCHEMA_VERSION,
                            "split_spec_fingerprint": manifest["split_spec_fingerprint"],
                            "target_split": "adaptation-only",
                            "cache_note": cache_note,
                            "label_policy": label_policy,
                            "mask_weight_mode": "soft_alpha" if preserve_soft_mask else "binary",
                            "uncertainty_mask_path": uncertainty_mask_path or None,
                        },
                    )
                )
    finally:
        if extractor is not None:
            extractor.close()
    with metadata_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    write_json(
        output_dir / "cache_status.json",
        {
            "provider": provider,
            "records": len(rows),
            "phase3_fingerprint": run_fingerprint,
            "schema_version": PHASE3_SCHEMA_VERSION,
            "note": "No VLM is loaded during this cache build when provider=heuristic." if provider == "heuristic" else "Qwen was loaded only for offline adaptation-cache extraction.",
        },
    )
    return metadata_path


def train_phase3_adapter(config: AppConfig, provider: str | None = None) -> Path:
    torch = _torch()
    phase3 = _phase3_config(config)
    provider = provider or str(phase3.get("provider", "heuristic"))
    rows = _load_cache_records(config, provider)
    if not rows:
        raise ValueError("Phase 3 cache is empty")
    if provider == "qwen":
        return _train_phase3_denoising_adapter(config, provider, rows)
    device = _training_device(config)
    seed = int(phase3.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    first = _load_cache(rows[0])
    input_dim = int(first["tokens"].shape[-1])
    output_dim = int(phase3.get("cross_attention_dim", 768))
    adapter = GatedProjectionAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(phase3.get("adapter_hidden_dim", 512)),
        initial_gate_logit=float(phase3.get("initial_gate_logit", -2.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(phase3.get("learning_rate", 1e-3)),
        weight_decay=float(phase3.get("weight_decay", 0.0)),
    )
    epochs = int(phase3.get("adapter_epochs", 5))
    clip_token_count = int(phase3.get("clip_token_count", 77))
    losses: list[float] = []
    token_dropout = float(phase3.get("adapter_token_dropout", 0.0))
    gate_regularization = float(phase3.get("gate_regularization_weight", 0.0))
    started = time.perf_counter()
    for _ in range(epochs):
        for row in rows:
            item = _load_cache(row)
            tokens = item["tokens"].to(device=device, dtype=torch.float32)
            tokens = _drop_adapter_tokens(tokens, token_dropout, adapter.training)
            target = item["surrogate_target_tokens"].to(device=device, dtype=torch.float32)
            if target.shape[-1] != output_dim:
                raise ValueError(f"Expected surrogate target width {output_dim}, got {target.shape[-1]}")
            clip = torch.zeros((tokens.shape[0], clip_token_count, output_dim), device=device, dtype=torch.float32)
            appended = adapter(clip, tokens)[:, clip_token_count:, :]
            loss = torch.nn.functional.mse_loss(appended, target)
            if gate_regularization > 0.0:
                loss = loss + gate_regularization * torch.sigmoid(adapter.gate_logit).pow(2)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite Phase 3 loss while training on {row.feature_cache_path}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    output_dir = config.output_dir / "phase3" / provider
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "adapter_surrogate.pt"
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "adapter_config": {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": int(phase3.get("adapter_hidden_dim", 512)),
                "clip_token_count": clip_token_count,
            },
            "training": {
                "provider": provider,
                "epochs": epochs,
                "records": len(rows),
                "seed": seed,
                "loss_first": losses[0] if losses else math.nan,
                "loss_final": losses[-1] if losses else math.nan,
                "loss_mean": mean(losses) if losses else math.nan,
                "gate_value": adapter.gate_value,
                "adapter_token_dropout": token_dropout,
                "gate_regularization_weight": gate_regularization,
                "duration_sec": time.perf_counter() - started,
                "objective": "surrogate adapter-shape loss, not diffusion denoising loss",
            },
            "phase3_fingerprint": _phase3_fingerprint(config, provider),
            "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        },
        checkpoint_path,
    )
    report_dir = config.report_dir / "phase3" / provider
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_training_summary(
        report_dir / "training_summary.md",
        checkpoint_path,
        len(rows),
        losses,
        adapter.gate_value,
        "surrogate adapter-shape loss, not diffusion denoising loss",
    )
    return checkpoint_path


def validate_phase3_adapter(config: AppConfig, provider: str | None = None) -> Path:
    torch = _torch()
    phase3 = _phase3_config(config)
    provider = provider or str(phase3.get("provider", "heuristic"))
    rows = _load_cache_records(config, provider)
    if not rows:
        raise ValueError("Phase 3 cache is empty")
    checkpoint_path = _checkpoint_path(config, provider)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 adapter checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase3_schema_version") != PHASE3_SCHEMA_VERSION:
        raise ValueError("Phase 3 checkpoint schema is stale; retrain the adapter first.")
    if checkpoint.get("phase3_fingerprint") != _phase3_fingerprint(config, provider):
        raise ValueError("Phase 3 checkpoint does not match the active configuration; retrain the adapter first.")
    adapter_config = dict(checkpoint["adapter_config"])
    adapter = GatedProjectionAdapter(
        input_dim=int(adapter_config["input_dim"]),
        output_dim=int(adapter_config["output_dim"]),
        hidden_dim=int(adapter_config["hidden_dim"]),
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    item = _load_qwen_cache(rows[0]) if provider == "qwen" else _load_cache(rows[0])
    tokens = item["tokens"].to(dtype=torch.float32)
    clip = torch.zeros(
        (tokens.shape[0], int(adapter_config["clip_token_count"]), int(adapter_config["output_dim"])),
        dtype=torch.float32,
    )
    with torch.no_grad():
        combined = adapter(clip, tokens)
    result = {
        "provider": provider,
        "checkpoint_path": str(checkpoint_path),
        "input_tokens_shape": list(tokens.shape),
        "clip_embeddings_shape": list(clip.shape),
        "combined_conditioning_shape": list(combined.shape),
        "expected_sd15_text_conditioning_shape": [1, int(adapter_config["clip_token_count"]), int(adapter_config["output_dim"])],
        "gate_value": adapter.gate_value,
        "trainable_parameter_count": sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad),
        "frozen_backbone_policy": "VLM, CLIP text encoder, VAE, and diffusion U-Net are not loaded during Phase 3 adapter-shape validation.",
        "objective_note": checkpoint.get("training", {}).get("objective", "adapter validation"),
    }
    validation_path = config.report_dir / "phase3" / provider / "validation.json"
    write_json(validation_path, result)
    return validation_path


def _phase3_config(config: AppConfig) -> dict[str, Any]:
    return dict(config.data.get("phase3", {}))


def _phase3_fingerprint(config: AppConfig, provider: str) -> str:
    return fingerprint(
        {
            "split": config.split_spec_fingerprint(),
            "phase2": config.data.get("phase2", {}),
            "phase3": config.data.get("phase3", {}),
            "provider": provider,
            "schema_version": PHASE3_SCHEMA_VERSION,
        }
    )


def _qwen_availability(config: AppConfig):
    phase3 = _phase3_config(config)
    phase2 = dict(config.data.get("phase2", {}))
    return qwen_availability(
        model_id=str(phase3.get("qwen_model", phase2.get("qwen_model", "Qwen/Qwen2.5-VL-3B-Instruct"))),
        cache_dir=phase3.get("qwen_cache_dir", phase2.get("qwen_cache_dir")),
        local_files_only=bool(phase3.get("qwen_local_files_only", phase2.get("qwen_local_files_only", True))),
        min_free_gib=float(phase3.get("qwen_min_free_gib", phase2.get("qwen_min_free_gib", 30.0))),
    )


def _load_cache_records(config: AppConfig, provider: str) -> list[Phase3CacheRecord]:
    path = config.output_dir / "phase3" / provider / "metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 3 cache metadata: {path}")
    expected = _phase3_fingerprint(config, provider)
    raw_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("settings", {}).get("phase3_schema_version") != PHASE3_SCHEMA_VERSION for row in raw_rows):
        raise ValueError("Phase 3 metadata schema is stale; rebuild adaptation cache first.")
    if any(row.get("settings", {}).get("phase3_fingerprint") != expected for row in raw_rows):
        raise ValueError("Phase 3 metadata does not match the active configuration; rebuild adaptation cache first.")
    return [Phase3CacheRecord(**row) for row in raw_rows]


def _load_cache(row: Phase3CacheRecord) -> dict[str, Any]:
    torch = _torch()
    item = torch.load(row.feature_cache_path, map_location="cpu", weights_only=False)
    tokens = item.get("tokens")
    target = item.get("surrogate_target_tokens")
    if tokens is None or target is None:
        raise ValueError(f"Phase 3 cache is missing required tensors: {row.feature_cache_path}")
    if tokens.ndim != 3 or target.ndim != 3:
        raise ValueError(f"Phase 3 tensors must be [B, K, D]: {row.feature_cache_path}")
    if tokens.shape[0] != target.shape[0] or tokens.shape[1] != target.shape[1]:
        raise ValueError(f"Phase 3 token and target batch/token dimensions differ: {row.feature_cache_path}")
    if not torch.isfinite(tokens).all() or not torch.isfinite(target).all():
        raise ValueError(f"Phase 3 cache contains non-finite tensors: {row.feature_cache_path}")
    return item


def _load_qwen_cache(row: Phase3CacheRecord) -> dict[str, Any]:
    item = torch.load(row.feature_cache_path, map_location="cpu", weights_only=False)
    tokens = item.get("tokens")
    if tokens is None or tokens.ndim != 3:
        raise ValueError(f"Qwen Phase 3 cache must contain tokens shaped [B, K, D]: {row.feature_cache_path}")
    if not torch.isfinite(tokens).all():
        raise ValueError(f"Qwen Phase 3 cache contains non-finite tokens: {row.feature_cache_path}")
    return item


def _checkpoint_path(config: AppConfig, provider: str) -> Path:
    name = "adapter_denoising.pt" if provider == "qwen" else "adapter_surrogate.pt"
    return config.output_dir / "phase3" / provider / "checkpoints" / name


def _train_phase3_denoising_adapter(config: AppConfig, provider: str, rows: list[Phase3CacheRecord]) -> Path:
    phase3 = _phase3_config(config)
    device = _training_device(config)
    seed = int(phase3.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    first = _load_qwen_cache(rows[0])
    input_dim = int(first["tokens"].shape[-1])
    output_dim = int(phase3.get("cross_attention_dim", 768))
    adapter = GatedProjectionAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(phase3.get("adapter_hidden_dim", 512)),
        initial_gate_logit=float(phase3.get("initial_gate_logit", -2.0)),
    ).to(device)
    pipeline = _load_sd15_training_components(config, device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(phase3.get("learning_rate", 1e-4)),
        weight_decay=float(phase3.get("weight_decay", 0.0)),
    )
    epochs = int(phase3.get("denoising_epochs", phase3.get("adapter_epochs", 1)))
    image_size = int(phase3.get("image_size", config.data.get("generation", {}).get("image_size", 512)))
    losses: list[float] = []
    token_dropout = float(phase3.get("adapter_token_dropout", 0.0))
    gate_regularization = float(phase3.get("gate_regularization_weight", 0.0))
    started = time.perf_counter()
    for _ in range(epochs):
        for row in rows:
            item = _load_qwen_cache(row)
            tokens = item["tokens"].to(device=device, dtype=torch.float32)
            tokens = _drop_adapter_tokens(tokens, token_dropout, adapter.training)
            target_image, mask_image, masked_image = _load_inpaint_training_tensors(row, image_size, device, pipeline["dtype"])
            loss_weight = _load_uncertainty_loss_weight(row, image_size, device, pipeline["dtype"], phase3)
            with torch.no_grad():
                latents = pipeline["vae"].encode(target_image).latent_dist.sample() * pipeline["vae"].config.scaling_factor
                masked_latents = pipeline["vae"].encode(masked_image).latent_dist.sample() * pipeline["vae"].config.scaling_factor
                mask_latents = torch.nn.functional.interpolate(mask_image, size=latents.shape[-2:], mode="nearest")
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipeline["scheduler"].config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = pipeline["scheduler"].add_noise(latents, noise, timesteps)
                prompt_embeds = _text_embeddings(pipeline, [row.prompt], device)
            combined = adapter(prompt_embeds.float(), tokens).to(dtype=prompt_embeds.dtype)
            latent_model_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)
            noise_pred = pipeline["unet"](latent_model_input, timesteps, encoder_hidden_states=combined).sample
            loss = _weighted_denoising_mse(noise_pred.float(), noise.float(), loss_weight.float())
            if gate_regularization > 0.0:
                loss = loss + gate_regularization * torch.sigmoid(adapter.gate_logit).pow(2)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite denoising loss while training on {row.feature_cache_path}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    output_dir = config.output_dir / "phase3" / provider
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "adapter_denoising.pt"
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "adapter_config": {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": int(phase3.get("adapter_hidden_dim", 512)),
                "clip_token_count": 77,
            },
            "training": {
                "provider": provider,
                "epochs": epochs,
                "records": len(rows),
                "seed": seed,
                "loss_first": losses[0] if losses else math.nan,
                "loss_final": losses[-1] if losses else math.nan,
                "loss_mean": mean(losses) if losses else math.nan,
                "gate_value": adapter.gate_value,
                "adapter_token_dropout": token_dropout,
                "gate_regularization_weight": gate_regularization,
                "uncertainty_loss_weight": _uncertainty_loss_weight(phase3),
                "duration_sec": time.perf_counter() - started,
                "objective": "SD1.5 inpainting diffusion denoising loss with frozen backbones",
            },
            "phase3_fingerprint": _phase3_fingerprint(config, provider),
            "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        },
        checkpoint_path,
    )
    report_dir = config.report_dir / "phase3" / provider
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_training_summary(
        report_dir / "training_summary.md",
        checkpoint_path,
        len(rows),
        losses,
        adapter.gate_value,
        "SD1.5 inpainting diffusion denoising loss with frozen backbones",
    )
    return checkpoint_path


def _load_sd15_training_components(config: AppConfig, device: str) -> dict[str, Any]:
    from diffusers import DDPMScheduler, StableDiffusionInpaintPipeline

    model_config = dict(config.data["models"]["sd15"])
    dtype = getattr(torch, str(model_config.get("dtype", "float16"))) if device == "cuda" else torch.float32
    local_only = bool(model_config.get("local_files_only", True))
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,
        local_files_only=local_only,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.eval()
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    return {
        "tokenizer": pipe.tokenizer,
        "text_encoder": pipe.text_encoder,
        "vae": pipe.vae,
        "unet": pipe.unet,
        "scheduler": scheduler,
        "dtype": dtype,
    }


def _load_inpaint_training_tensors(
    row: Phase3CacheRecord,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image = Image.open(row.source_image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    preserve_soft_mask = row.settings.get("mask_weight_mode") == "soft_alpha"
    resample = Image.Resampling.BILINEAR if preserve_soft_mask else Image.Resampling.NEAREST
    mask = Image.open(row.source_mask_path).convert("L").resize((image_size, image_size), resample)
    image_arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    raw_mask = np.asarray(mask, dtype=np.float32) / 255.0
    mask_arr = raw_mask[None, :, :] if preserve_soft_mask else (raw_mask > 0.0).astype(np.float32)[None, :, :]
    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).to(device=device, dtype=dtype)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).to(device=device, dtype=dtype)
    masked = image_tensor * (1.0 - mask_tensor)
    return image_tensor, mask_tensor, masked


def _load_uncertainty_loss_weight(
    row: Phase3CacheRecord,
    image_size: int,
    device: str,
    dtype: torch.dtype,
    phase3: dict[str, Any],
) -> torch.Tensor:
    min_weight = _uncertainty_loss_weight(phase3)
    if min_weight >= 1.0 or not row.uncertainty_mask_path:
        return torch.ones((1, 1, image_size, image_size), device=device, dtype=dtype)
    path = Path(row.uncertainty_mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 3 uncertainty mask: {path}")
    uncertainty = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    uncertainty_arr = np.asarray(uncertainty, dtype=np.float32) / 255.0
    weight_arr = 1.0 - uncertainty_arr * (1.0 - min_weight)
    weight_arr = np.clip(weight_arr, min_weight, 1.0).astype(np.float32)[None, None, :, :]
    return torch.from_numpy(weight_arr).to(device=device, dtype=dtype)


def _weighted_denoising_mse(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.shape[-2:] != prediction.shape[-2:]:
        weight = torch.nn.functional.interpolate(weight, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
    if weight.shape[1] == 1 and prediction.shape[1] != 1:
        weight = weight.expand(-1, prediction.shape[1], -1, -1)
    squared = (prediction - target).pow(2)
    denominator = weight.sum().clamp_min(1e-6)
    return (squared * weight).sum() / denominator


def _uncertainty_loss_weight(settings: dict[str, Any]) -> float:
    raw = settings.get("uncertainty_loss_weight", 1.0)
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError("uncertainty_loss_weight must be between 0.0 and 1.0")
    return value


def _drop_adapter_tokens(tokens: torch.Tensor, probability: float, training: bool = True) -> torch.Tensor:
    if not training or probability <= 0.0:
        return tokens
    if probability >= 1.0:
        return torch.zeros_like(tokens)
    keep = torch.rand(tokens.shape[:2] + (1,), device=tokens.device, dtype=tokens.dtype) >= probability
    return tokens * keep / max(1e-6, 1.0 - probability)


def _text_embeddings(pipeline: dict[str, Any], prompts: list[str], device: str) -> torch.Tensor:
    tokenizer = pipeline["tokenizer"]
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    return pipeline["text_encoder"](input_ids)[0]


def _masked_context(image: Image.Image, mask: Image.Image) -> Image.Image:
    arr = np.array(image, dtype=np.uint8, copy=True)
    alpha = (np.asarray(mask.convert("L"), dtype=np.float32) / 255.0)[..., None]
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    fill = np.median(border, axis=0).astype(np.float32)
    blended = np.asarray(arr, dtype=np.float32) * (1.0 - alpha) + fill[None, None, :] * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).convert("RGB")


def _sample_label_policy(sample: dict[str, Any]) -> dict[str, Any]:
    policy = sample.get("label_policy")
    if isinstance(policy, dict) and policy.get("label_policy"):
        return dict(policy)
    return {"label_policy": "hard_mask_ok", "quality_morphology": None}


def _write_heuristic_cache(
    image: Image.Image,
    masked_context: Image.Image,
    mask: Image.Image,
    bbox: tuple[int, int, int, int],
    prompt: str,
    output_path: Path,
    config: AppConfig,
) -> None:
    torch = _torch()
    phase2 = dict(config.data.get("phase2", {}))
    phase3 = _phase3_config(config)
    token_count = int(phase3.get("token_count", phase2.get("token_count", 16)))
    feature_dim = int(phase3.get("feature_dim", phase2.get("feature_dim", 64)))
    output_dim = int(phase3.get("cross_attention_dim", 768))
    context_stats = _image_stats(masked_context.crop(bbox).resize((16, 16)))
    anomaly_patch = np.asarray(image.crop(bbox).resize((16, 16)).convert("RGB"), dtype=np.float32) / 255.0
    mask_patch = np.asarray(mask.crop(bbox).resize((16, 16)).convert("L"), dtype=np.float32) / 255.0
    if mask_patch.sum() > 0:
        masked_pixels = anomaly_patch[mask_patch > 0.0]
        if masked_pixels.size == 0:
            masked_pixels = anomaly_patch.reshape(-1, 3)
        anomaly_stats = np.concatenate([masked_pixels.mean(axis=0), masked_pixels.std(axis=0)])
    else:
        anomaly_stats = anomaly_patch.mean(axis=(0, 1)).repeat(2)
    bbox_stats = np.asarray(_normalize_region(bbox, image.size), dtype=np.float32)
    base = np.concatenate([context_stats, anomaly_stats, bbox_stats])
    tokens = _expand_stats(base, token_count, feature_dim)
    surrogate_target = _expand_stats(np.concatenate([anomaly_stats, bbox_stats, context_stats]), token_count, output_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": torch.from_numpy(tokens).unsqueeze(0),
            "surrogate_target_tokens": torch.from_numpy(surrogate_target).unsqueeze(0),
            "metadata": {
                "provider": "heuristic",
                "prompt": prompt,
                "bbox_xyxy": bbox,
                "normalized_bbox_xyxy": _normalize_region(bbox, image.size),
                "tokens_shape": [1, token_count, feature_dim],
                "surrogate_target_shape": [1, token_count, output_dim],
                "note": "heuristic placeholder cache for adapter plumbing; not Qwen hidden states",
            },
        },
        output_path,
    )


def _image_stats(image: Image.Image) -> np.ndarray:
    patch = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.concatenate([patch.mean(axis=(0, 1)), patch.std(axis=(0, 1))])


def _expand_stats(stats: np.ndarray, token_count: int, width: int) -> np.ndarray:
    stats = np.asarray(stats, dtype=np.float32)
    values = np.resize(stats, (token_count, width)).astype(np.float32)
    token_offsets = np.linspace(0.0, 0.02, token_count, dtype=np.float32).reshape(token_count, 1)
    dim_offsets = np.linspace(0.0, 0.01, width, dtype=np.float32).reshape(1, width)
    return values + token_offsets + dim_offsets


def _normalize_region(
    region: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    width, height = image_size
    left, top, right, bottom = region
    return (left / width, top / height, right / width, bottom / height)


def _prompt(category: str, defect_type: str) -> str:
    return f"Reconstruct a realistic {defect_type} on {category} surface within the annotated mask."


def _training_device(config: AppConfig) -> str:
    configured = str(config.data.get("phase3", {}).get("device", config.data.get("generation", {}).get("device", "cpu")))
    torch = _torch()
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 3 config requested CUDA, but torch.cuda.is_available() is false.")
    return configured


def _write_training_summary(
    path: Path,
    checkpoint_path: Path,
    records: int,
    losses: list[float],
    gate_value: float,
    objective_note: str,
) -> None:
    is_denoising = "denoising" in objective_note and "not diffusion" not in objective_note
    title = "Phase 3 Adapter Denoising Training" if is_denoising else "Phase 3 Adapter Surrogate Training"
    final_note = (
        "This is real adapter smoke training with cached Qwen features and frozen SD1.5 backbones."
        if is_denoising
        else "This is a cache/adapter/gate validation run. It is not diffusion denoising training and does not use real Qwen features."
    )
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"Checkpoint: `{checkpoint_path}`",
                f"Records: {records}",
                f"Initial loss: {losses[0]:.6f}" if losses else "Initial loss: n/a",
                f"Final loss: {losses[-1]:.6f}" if losses else "Final loss: n/a",
                f"Gate value: {gate_value:.6f}",
                f"Objective: {objective_note}",
                "",
                final_note,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _torch() -> Any:
    import torch

    return torch
