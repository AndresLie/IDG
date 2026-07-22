from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.dataset import load_manifest
from iadgen_v2.generation_critic import score_generation
from iadgen_v2.phase2 import PHASE2_SCHEMA_VERSION, _phase2_fingerprint
from iadgen_v2.phase3 import GatedProjectionAdapter, PHASE3_SCHEMA_VERSION, _phase3_fingerprint
from iadgen_v2.records import write_json


PHASE4_SCHEMA_VERSION = 8
PHASE4_QWEN_VARIANTS = (
    "qwen_mask_only",
    "full_qwen_hybrid",
    "fixed_mask_adapter",
    "clone_harmonized",
    "ip_adapter_hybrid",
    "latent_blend_harmonized",
)
PHASE4_QUALITY_PROFILES: dict[str, dict[str, Any]] = {
    "scuff_soft_low_strength": {
        "morphologies": {"multi_scuff"},
        "mask_variant": "inpaint_soft",
        "strength": 0.42,
        "guidance_scale": 6.2,
        "num_inference_steps": 16,
        "prompt_template": (
            "{base_prompt}, a broad subtle rubbed scuffed patch with many faint short surface scratches, "
            "natural material texture, no rectangular edges"
        ),
        "negative_prompt": "square boundary, hard mask edge, single dark line, dots, blobs, text, watermark",
    },
    "scratch_ridge_balanced": {
        "morphologies": {"scratch_band", "crack_band"},
        "mask_variant": "inpaint_soft",
        "strength": 0.55,
        "guidance_scale": 7.2,
        "num_inference_steps": 18,
        "prompt_template": "{base_prompt}, a realistic narrow surface ridge defect following the selected mask, blended into the material",
        "negative_prompt": "detached blobs, square boundary, overpainted patch, text, watermark, unrealistic defect",
    },
    "scratch_thin_detail": {
        "morphologies": {"scratch_band", "crack_band", "single_stroke"},
        "mask_variant": "inpaint_soft",
        "strength": 0.62,
        "guidance_scale": 8.0,
        "num_inference_steps": 22,
        "prompt_template": "{base_prompt}, a thin detailed continuous surface scratch with subtle highlights and shadow",
        "negative_prompt": "wide stain, detached chunks, square boundary, blurry patch, text, watermark",
    },
    "single_stroke_clean": {
        "morphologies": {"single_stroke"},
        "mask_variant": "inpaint_soft",
        "strength": 0.50,
        "guidance_scale": 7.0,
        "num_inference_steps": 18,
        "prompt_template": "{base_prompt}, one clean narrow line-like surface defect, physically plausible, no extra marks",
        "negative_prompt": "multiple scratches, stains, dots, blobs, square boundary, text, watermark",
    },
}


@dataclass(frozen=True)
class Phase4Record:
    category: str
    defect_type: str
    provider: str
    variant: str
    quality_profile: str
    placement_source: str
    background_path: str
    prompt: str
    refined_mask_path: str
    inpaint_mask_path: str
    generation_mask_path: str
    generation_mask_variant: str
    generation_mask_role: str
    generation_mask_role_warning: str
    benchmark_eval_mask_path: str
    phase2_feature_cache_path: str
    adapter_checkpoint_path: str | None
    output_path: str
    generation_seed: int
    conditioning_shape: list[int]
    projected_tokens_shape: list[int]
    gate_value: float
    latency_sec: float
    peak_cuda_memory_bytes: int | None
    max_rss_kb: int
    background_preservation_l1: float
    mask_changed_pixel_fraction: float
    defect_visibility_score: float
    outside_refined_change_fraction: float
    inpaint_mask_area_fraction: float
    generation_quality_score: float
    quality_flags: list[str]
    critic_guided_generation: dict[str, Any]
    settings: dict[str, Any]


def run_phase4_generation(config: AppConfig, provider: str | None = None, variant: str | None = None) -> Path:
    phase4 = _phase4_config(config)
    provider = provider or str(phase4.get("provider", "heuristic"))
    if provider not in {"heuristic", "qwen"}:
        raise ValueError(f"Unsupported Phase 4 provider: {provider}")
    selected_variants = _phase4_variants(phase4, provider, variant)
    written_paths = [_run_phase4_variant(config, provider, selected_variant) for selected_variant in selected_variants]
    if len(written_paths) == 1:
        return written_paths[0]
    combined_path = config.output_dir / "phase4" / provider / "metadata.jsonl"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text("", encoding="utf-8")
    with combined_path.open("a", encoding="utf-8") as handle:
        for path in written_paths:
            handle.write(path.read_text(encoding="utf-8"))
    write_json(
        config.output_dir / "phase4" / provider / "run_status.json",
        {
            "provider": provider,
            "variants": selected_variants,
            "records": sum(1 for path in written_paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "phase4_fingerprint": _phase4_fingerprint(config, provider),
            "schema_version": PHASE4_SCHEMA_VERSION,
            "note": "Combined Phase 4 ablation metadata; variant-specific metadata remains in variant subdirectories.",
        },
    )
    return combined_path


def _run_phase4_variant(config: AppConfig, provider: str, variant: str) -> Path:
    phase4 = _phase4_config(config)
    phase2_rows = _load_phase2_records(config, provider)
    if not phase2_rows:
        raise ValueError("Phase 4 requires non-empty Phase 2 proposal metadata")
    uses_adapter = _variant_uses_adapter(provider, variant)
    uses_reference = provider == "qwen" and variant in {"clone_harmonized", "ip_adapter_hybrid", "latent_blend_harmonized"}
    source_index = _clone_source_index(config) if uses_reference else {}
    checkpoint_path = _checkpoint_path(config, provider) if uses_adapter else None
    adapter: GatedProjectionAdapter | None
    adapter_config: dict[str, Any] | None
    checkpoint: dict[str, Any] | None
    if uses_adapter:
        adapter, adapter_config, checkpoint = _load_adapter(config, provider, checkpoint_path)
    else:
        adapter, adapter_config, checkpoint = None, None, None
    output_dir = _phase4_output_dir(config, provider, variant)
    image_dir = output_dir / "images"
    metadata_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("", encoding="utf-8")
    run_fingerprint = _phase4_fingerprint(config, provider)
    rows: list[Phase4Record] = []
    limit = int(phase4.get("max_records", len(phase2_rows)))
    device = _phase4_device(config)
    if adapter is not None:
        adapter = adapter.to(device)
        adapter.eval()
    if device == "cuda":
        torch.cuda.empty_cache()
    pipe = _load_inpaint_pipeline(config, device) if provider == "qwen" else None
    record_index = 0
    seeds_per_record = int(phase4.get("seeds_per_record", 1))
    if seeds_per_record < 1:
        raise ValueError("phase4.seeds_per_record must be positive")
    for base_index, row in enumerate(phase2_rows[:limit]):
        image = Image.open(row["background_path"]).convert("RGB")
        refined_mask = Image.open(row["refined_mask_path"]).convert("L")
        if refined_mask.size != image.size:
            raise ValueError(f"Phase 4 mask/image size mismatch for {row['background_path']}")
        tokens: torch.Tensor | None = None
        if adapter_config is not None:
            feature = torch.load(row["feature_cache_path"], map_location="cpu", weights_only=False)
            tokens = _load_phase2_tokens(feature, row["feature_cache_path"], adapter_config).to(device=device, dtype=torch.float32)
            clip = torch.zeros(
                (tokens.shape[0], int(adapter_config["clip_token_count"]), int(adapter_config["output_dim"])),
                device=device,
                dtype=torch.float32,
            )
        else:
            clip = None
        for profile_name in _quality_profiles_for_row(phase4, row):
            profile_settings = _quality_profile_settings(phase4, profile_name)
            base_render_prompt = _render_prompt(config, row, profile_settings)
            mask_role = _mask_role_for_profile(row, profile_settings, phase4)
            inpaint_mask_path = str(mask_role["path"])
            inpaint_mask = Image.open(inpaint_mask_path).convert("L")
            if inpaint_mask.size != image.size:
                raise ValueError(f"Phase 4 mask/image size mismatch for {row['background_path']}")
            for seed_index in range(seeds_per_record):
                started = time.perf_counter()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                generation_seed = _generation_seed(row, seed_index)
                base_render_phase4 = {**phase4, **profile_settings, "negative_prompt": _negative_prompt(config, profile_settings)}
                critic_settings = _critic_guided_settings(base_render_phase4, provider)
                attempt_records: list[dict[str, Any]] = []
                selected: dict[str, Any] | None = None
                max_attempts = int(critic_settings.get("max_attempts", 1))
                for attempt_index in range(max_attempts):
                    previous_attempt = attempt_records[-1] if attempt_records else None
                    attempt_phase4, attempt_prompt, attempt_seed = _critic_guided_attempt_settings(
                        base_render_phase4,
                        base_render_prompt,
                        row,
                        generation_seed,
                        attempt_index,
                        critic_settings,
                        previous_attempt=previous_attempt,
                    )
                    diversity = _diversity_metadata(attempt_phase4, attempt_seed, variant)
                    render_metadata: dict[str, Any] = {}
                    with torch.no_grad():
                        if pipe is None:
                            if adapter is None or adapter_config is None or tokens is None or clip is None:
                                raise RuntimeError("Heuristic Phase 4 requires adapter conditioning")
                            combined = adapter(clip, tokens)
                            projected = combined[:, int(adapter_config["clip_token_count"]) :, :]
                            output = _heuristic_integrated_render(
                                image,
                                refined_mask,
                                inpaint_mask,
                                projected.detach().cpu(),
                                str(row["defect_type"]),
                            )
                        elif variant == "qwen_mask_only":
                            output, combined, projected = _qwen_sd15_text_render(
                                config, attempt_phase4, pipe, attempt_prompt, row, image, inpaint_mask, device, attempt_seed
                            )
                        elif variant == "clone_harmonized":
                            output, combined, projected = _qwen_clone_harmonized_render(
                                config,
                                attempt_phase4,
                                pipe,
                                attempt_prompt,
                                row,
                                source_index,
                                image,
                                refined_mask,
                                inpaint_mask,
                                device,
                                attempt_seed,
                            )
                        elif variant == "ip_adapter_hybrid":
                            output, combined, projected, render_metadata = _qwen_ip_adapter_hybrid_render(
                                config,
                                attempt_phase4,
                                pipe,
                                attempt_prompt,
                                row,
                                source_index,
                                image,
                                refined_mask,
                                inpaint_mask,
                                device,
                                attempt_seed,
                            )
                        elif variant == "latent_blend_harmonized":
                            output, combined, projected, render_metadata = _qwen_latent_blend_harmonized_render(
                                config,
                                attempt_phase4,
                                pipe,
                                attempt_prompt,
                                row,
                                source_index,
                                image,
                                refined_mask,
                                inpaint_mask,
                                device,
                                attempt_seed,
                            )
                        else:
                            if adapter is None or adapter_config is None or tokens is None:
                                raise RuntimeError(f"Phase 4 variant {variant} requires adapter conditioning")
                            output, combined, projected = _qwen_sd15_render(
                                config,
                                attempt_phase4,
                                pipe,
                                adapter,
                                adapter_config,
                                row,
                                attempt_prompt,
                                tokens,
                                image,
                                inpaint_mask,
                                device,
                                attempt_seed,
                            )
                    output, postprocess_metadata = _critic_guided_visibility_boost(
                        image,
                        output,
                        refined_mask,
                        inpaint_mask,
                        attempt_phase4,
                        attempt_seed,
                        morphology=_morphology_for_row(row),
                    )
                    if postprocess_metadata:
                        render_metadata["critic_guided_visibility_boost"] = postprocess_metadata
                    critic_record = _phase4_generation_critic(
                        image,
                        output,
                        refined_mask,
                        inpaint_mask,
                        critic_settings,
                        morphology=_morphology_for_row(row),
                    )
                    attempt_record = {
                        **critic_record,
                        "attempt_index": attempt_index,
                        "generation_seed": attempt_seed,
                        "strength": _effective_strength(attempt_phase4, variant),
                        "guidance_scale": _effective_guidance_scale(attempt_phase4, variant),
                        "num_inference_steps": _effective_num_inference_steps(attempt_phase4, variant),
                        "retry_reason": str(attempt_phase4.get("critic_retry_reason", "")),
                        "local_mask_noise_boost": round(float(attempt_phase4.get("local_mask_noise_boost", 0.0)), 4),
                        "postprocess_mask_contrast_boost": round(float(attempt_phase4.get("postprocess_mask_contrast_boost", 0.0)), 4),
                        "prompt": attempt_prompt,
                    }
                    attempt_records.append(attempt_record)
                    candidate = {
                        "output": output,
                        "combined": combined,
                        "projected": projected,
                        "render_metadata": render_metadata,
                        "render_phase4": attempt_phase4,
                        "render_prompt": attempt_prompt,
                        "diversity": diversity,
                        "attempt_record": attempt_record,
                    }
                    if selected is None or _critic_attempt_rank(attempt_record) > _critic_attempt_rank(selected["attempt_record"]):
                        selected = candidate
                    if attempt_record["accepted"] and not bool(critic_settings.get("exhaustive", False)):
                        break
                if selected is None:
                    raise RuntimeError("Phase 4 generation produced no attempts")
                output = selected["output"]
                combined = selected["combined"]
                projected = selected["projected"]
                render_metadata = dict(selected["render_metadata"])
                render_phase4 = dict(selected["render_phase4"])
                render_prompt = str(selected["render_prompt"])
                diversity = dict(selected["diversity"])
                final_repair_attempts: list[dict[str, Any]] = []
                final_repair_metadata: dict[str, Any] = {}
                if bool(critic_settings.get("final_coverage_repair_enabled", False)) and not bool(selected["attempt_record"].get("accepted", False)):
                    repaired_output, repaired_record, final_repair_metadata = _critic_guided_final_coverage_repair(
                        image,
                        output,
                        refined_mask,
                        inpaint_mask,
                        critic_settings,
                        generation_seed,
                        selected["attempt_record"],
                        morphology=_morphology_for_row(row),
                    )
                    final_repair_attempts = list(final_repair_metadata.get("attempts", []))
                    if _critic_attempt_rank(repaired_record) > _critic_attempt_rank(selected["attempt_record"]):
                        output = repaired_output
                        selected["attempt_record"] = repaired_record
                        render_metadata["critic_guided_final_coverage_repair"] = final_repair_metadata
                critic_guided_generation = {
                    "enabled": bool(critic_settings.get("enabled", False)),
                    "selected_attempt_index": int(selected["attempt_record"]["attempt_index"]),
                    "selected_repair_index": int(selected["attempt_record"].get("repair_index", -1)),
                    "attempts": attempt_records,
                    "final_coverage_repair_enabled": bool(critic_settings.get("final_coverage_repair_enabled", False)),
                    "final_coverage_repair_attempts": final_repair_attempts,
                    "accepted": bool(selected["attempt_record"]["accepted"]),
                    "reject_reasons": list(selected["attempt_record"]["reject_reasons"]),
                }
                image_path = (
                    image_dir
                    / str(row["category"])
                    / profile_name
                    / f"{row['category']}_{row['defect_type']}_{base_index:04d}_s{seed_index:02d}.png"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                output.save(image_path)
                preservation_l1, mask_changed_fraction = _preservation_metrics(image, output, inpaint_mask)
                morphology = _morphology_for_row(row)
                diagnostics = _quality_diagnostics(
                    image,
                    output,
                    refined_mask,
                    inpaint_mask,
                    phase4,
                    preservation_l1,
                    mask_changed_fraction,
                    morphology=morphology,
                    label_policy=_label_policy_for_row(row),
                )
                peak_cuda = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
                records_settings = {
                    "phase4_fingerprint": run_fingerprint,
                    "phase4_schema_version": PHASE4_SCHEMA_VERSION,
                    "phase2_fingerprint": row["settings"]["phase2_fingerprint"],
                    "phase3_fingerprint": checkpoint["phase3_fingerprint"] if checkpoint is not None else None,
                    "phase2_schema_version": row["settings"]["phase2_schema_version"],
                    "phase3_schema_version": checkpoint["phase3_schema_version"] if checkpoint is not None else None,
                    "generator": variant,
                    "variant": variant,
                    "quality_profile": profile_name,
                    "quality_profile_settings": _json_safe(profile_settings),
                    "quality_morphology": morphology,
                    "label_policy": _label_policy_for_row(row),
                    "seed_index": seed_index,
                    "mask_variant_used": str(mask_role["variant"]),
                    "generation_mask_path": str(mask_role["path"]),
                    "generation_mask_role": str(mask_role["role"]),
                    "generation_mask_source": str(mask_role["source"]),
                    "generation_mask_role_warning": str(mask_role["warning"]),
                    "benchmark_eval_mask_path": str(mask_role["benchmark_eval_mask_path"]),
                    "generation_core_mask_path": str(mask_role["generation_core_mask_path"]),
                    "mask_role_diagnostics": _json_safe(mask_role),
                    "quality_diagnostics": diagnostics,
                    "critic_guided_generation": critic_guided_generation,
                    "diversity": diversity,
                    "reference_render": _json_safe(render_metadata),
                    "placement_source": _placement_source(provider, variant),
                    "note": _provider_note(provider),
                }
                rows.append(
                    Phase4Record(
                        category=str(row["category"]),
                        defect_type=str(row["defect_type"]),
                        provider=provider,
                        variant=variant,
                        quality_profile=profile_name,
                        placement_source=_placement_source(provider, variant),
                        background_path=str(row["background_path"]),
                        prompt=render_prompt,
                        refined_mask_path=str(row["refined_mask_path"]),
                        inpaint_mask_path=str(inpaint_mask_path),
                        generation_mask_path=str(mask_role["path"]),
                        generation_mask_variant=str(mask_role["variant"]),
                        generation_mask_role=str(mask_role["role"]),
                        generation_mask_role_warning=str(mask_role["warning"]),
                        benchmark_eval_mask_path=str(mask_role["benchmark_eval_mask_path"]),
                        phase2_feature_cache_path=str(row["feature_cache_path"]),
                        adapter_checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
                        output_path=str(image_path),
                        generation_seed=generation_seed,
                        conditioning_shape=list(combined.shape),
                        projected_tokens_shape=list(projected.shape),
                        gate_value=float(checkpoint["training"]["gate_value"]) if checkpoint is not None else 0.0,
                        latency_sec=time.perf_counter() - started,
                        peak_cuda_memory_bytes=peak_cuda,
                        max_rss_kb=_max_rss_kb(),
                        background_preservation_l1=preservation_l1,
                        mask_changed_pixel_fraction=mask_changed_fraction,
                        defect_visibility_score=float(diagnostics["defect_visibility_score"]),
                        outside_refined_change_fraction=float(diagnostics["outside_refined_change_fraction"]),
                        inpaint_mask_area_fraction=float(diagnostics["inpaint_mask_area_fraction"]),
                        generation_quality_score=float(diagnostics["generation_quality_score"]),
                        quality_flags=list(diagnostics["quality_flags"]),
                        critic_guided_generation=critic_guided_generation,
                        settings=records_settings,
                    )
                )
                record_index += 1
    with metadata_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
    summary_path = _phase4_report_dir(config, provider, variant) / "summary.md"
    _write_summary(summary_path, rows, provider, variant)
    _write_quality_contact_sheet(_phase4_report_dir(config, provider, variant) / "quality_contact_sheet.png", rows)
    write_json(
        output_dir / "run_status.json",
        {
            "provider": provider,
            "variant": variant,
            "records": len(rows),
            "phase4_fingerprint": run_fingerprint,
            "schema_version": PHASE4_SCHEMA_VERSION,
            "adapter_checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "note": _provider_note(provider),
        },
    )
    return metadata_path


def _load_phase2_tokens(feature: dict[str, Any], feature_path: str, adapter_config: dict[str, Any]) -> torch.Tensor:
    tokens = feature.get("tokens")
    if tokens is None:
        raise ValueError(f"Phase 4 feature cache is missing tokens: {feature_path}")
    if tokens.ndim != 3:
        raise ValueError(f"Phase 4 feature tokens must be [B, K, D], got {tuple(tokens.shape)} from {feature_path}")
    expected_dim = int(adapter_config["input_dim"])
    if int(tokens.shape[-1]) != expected_dim:
        raise ValueError(f"Phase 4 feature width {tokens.shape[-1]} does not match adapter input width {expected_dim}")
    if not torch.isfinite(tokens).all():
        raise ValueError(f"Phase 4 feature cache contains non-finite tokens: {feature_path}")
    return tokens


def _load_phase2_records(config: AppConfig, provider: str) -> list[dict[str, Any]]:
    path = config.output_dir / "phase2" / provider / "metadata.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 2 metadata: {path}")
    expected = _phase2_fingerprint(config, provider)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("settings", {}).get("phase2_schema_version") != PHASE2_SCHEMA_VERSION for row in rows):
        raise ValueError("Phase 2 metadata schema is stale; regenerate Phase 2 proposals first.")
    if any(row.get("settings", {}).get("phase2_fingerprint") != expected for row in rows):
        raise ValueError("Phase 2 metadata does not match the active configuration; regenerate Phase 2 proposals first.")
    return rows


def _load_adapter(
    config: AppConfig,
    provider: str,
    checkpoint_path: Path,
) -> tuple[GatedProjectionAdapter, dict[str, Any], dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Phase 3 adapter checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("phase3_schema_version") != PHASE3_SCHEMA_VERSION:
        raise ValueError("Phase 3 checkpoint schema is stale; retrain Phase 3 adapter first.")
    if checkpoint.get("phase3_fingerprint") != _phase3_fingerprint(config, provider):
        raise ValueError("Phase 3 checkpoint does not match the active configuration; retrain Phase 3 adapter first.")
    adapter_config = dict(checkpoint["adapter_config"])
    adapter = GatedProjectionAdapter(
        input_dim=int(adapter_config["input_dim"]),
        output_dim=int(adapter_config["output_dim"]),
        hidden_dim=int(adapter_config["hidden_dim"]),
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    return adapter, adapter_config, checkpoint


def _checkpoint_path(config: AppConfig, provider: str) -> Path:
    name = "adapter_denoising.pt" if provider == "qwen" else "adapter_surrogate.pt"
    return config.output_dir / "phase3" / provider / "checkpoints" / name


def _phase4_variants(phase4: dict[str, Any], provider: str, requested: str | None) -> list[str]:
    if provider == "heuristic":
        if requested not in {None, "heuristic_adapter_contract"}:
            raise ValueError(f"Unsupported Phase 4 variant for heuristic provider: {requested}")
        return ["heuristic_adapter_contract"]
    raw = requested or phase4.get("variant") or phase4.get("variants") or "full_qwen_hybrid"
    if raw == "all":
        variants = list(PHASE4_QWEN_VARIANTS)
    elif isinstance(raw, list):
        variants = [str(value) for value in raw]
    else:
        variants = [str(raw)]
    unknown = [value for value in variants if value not in PHASE4_QWEN_VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported Phase 4 variant(s): {', '.join(unknown)}")
    return variants


def _quality_profile_settings(phase4: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if profile_name not in PHASE4_QUALITY_PROFILES:
        raise ValueError(f"Unsupported Phase 4 quality_profile: {profile_name}")
    overrides = dict(phase4.get("quality_profile_overrides", {})).get(profile_name, {})
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError(f"phase4.quality_profile_overrides.{profile_name} must be a mapping")
    return {**PHASE4_QUALITY_PROFILES[profile_name], **dict(overrides or {})}


def _quality_profiles_for_row(phase4: dict[str, Any], row: dict[str, Any]) -> list[str]:
    raw = phase4.get("quality_profiles", phase4.get("quality_profile", "scratch_ridge_balanced"))
    if raw is None or raw == "auto":
        requested = [_default_quality_profile(_morphology_for_row(row))]
    elif raw == "all":
        requested = list(PHASE4_QUALITY_PROFILES)
    elif isinstance(raw, list):
        requested = [str(value) for value in raw]
    else:
        requested = [str(raw)]
    unknown = [value for value in requested if value not in PHASE4_QUALITY_PROFILES]
    if unknown:
        raise ValueError(f"Unsupported Phase 4 quality_profile(s): {', '.join(unknown)}")
    if bool(phase4.get("quality_profile_filter_by_morphology", False)):
        morphology = _morphology_for_row(row)
        compatible = [
            name
            for name in requested
            if morphology in set(_quality_profile_settings(phase4, name).get("morphologies", set()))
        ]
        if compatible:
            return compatible
        return [_default_quality_profile(morphology)]
    return requested


def _default_quality_profile(morphology: str) -> str:
    if morphology == "multi_scuff":
        return "scuff_soft_low_strength"
    if morphology == "single_stroke":
        return "single_stroke_clean"
    return "scratch_ridge_balanced"


def _morphology_for_row(row: dict[str, Any]) -> str:
    settings = row.get("settings", {})
    row_policy = row.get("label_policy")
    settings_policy = settings.get("label_policy") if isinstance(settings, dict) else None
    for candidate in (
        row.get("scratch_morphology_class"),
        row_policy.get("quality_morphology") if isinstance(row_policy, dict) else None,
        settings.get("scratch_morphology_class") if isinstance(settings, dict) else None,
        settings.get("quality_morphology") if isinstance(settings, dict) else None,
        settings_policy.get("quality_morphology") if isinstance(settings_policy, dict) else None,
    ):
        if candidate:
            return str(candidate)
    text = " ".join(str(value).lower() for value in (row.get("prompt", ""), settings.get("description", "") if isinstance(settings, dict) else ""))
    defect_type = str(row.get("defect_type", "")).lower()
    if any(hint in text for hint in ("scuff", "many", "multiple", "rubbed")):
        return "multi_scuff"
    if any(hint in text for hint in ("single", "one long", "one narrow")):
        return "single_stroke"
    if defect_type == "crack":
        return "crack_band"
    return "scratch_band"


def _label_policy_for_row(row: dict[str, Any]) -> dict[str, Any]:
    policy = row.get("label_policy")
    if isinstance(policy, dict) and policy.get("label_policy"):
        return dict(policy)
    settings = row.get("settings", {})
    policy = settings.get("label_policy") if isinstance(settings, dict) else None
    if isinstance(policy, dict) and policy.get("label_policy"):
        return dict(policy)
    return {"label_policy": "hard_mask_ok", "quality_morphology": _morphology_for_row(row)}


def _mask_path_for_profile(row: dict[str, Any], profile_settings: dict[str, Any]) -> str:
    return str(_mask_role_for_profile(row, profile_settings, {})["path"])


def _mask_role_for_profile(row: dict[str, Any], profile_settings: dict[str, Any], phase4: dict[str, Any]) -> dict[str, Any]:
    variant = str(profile_settings.get("mask_variant", "inpaint_soft"))
    settings = row.get("settings", {})
    mask_variants = settings.get("mask_variant_paths", {}) if isinstance(settings, dict) else {}
    source = ""
    if isinstance(mask_variants, dict) and variant in mask_variants:
        path = str(mask_variants[variant])
        source = "settings.mask_variant_paths"
    elif variant in row:
        path = str(row[variant])
        source = "row"
    elif variant in {"inpaint_soft", "training_medium", "training_wide"}:
        path = str(row.get("inpaint_mask_path", row["refined_mask_path"]))
        source = "fallback_inpaint_mask_path"
    elif variant == "eval_tight":
        path = str(row["refined_mask_path"])
        source = "fallback_refined_mask_path"
    else:
        raise ValueError(f"Phase 4 quality profile requested unavailable mask_variant: {variant}")

    role = _mask_variant_role(variant)
    benchmark_eval_mask_path = _benchmark_eval_mask_path(row)
    generation_core_mask_path = _generation_core_mask_path(row)
    warning = ""
    resolved = Path(path)
    benchmark_path = Path(benchmark_eval_mask_path) if benchmark_eval_mask_path else None
    if role == "benchmark_eval" or (benchmark_path is not None and benchmark_path == resolved):
        warning = "benchmark_eval_mask_used_for_generation"
        if not bool(phase4.get("allow_benchmark_eval_mask_for_generation", False)):
            raise ValueError(
                "Phase 4 generation attempted to use benchmark-only eval mask variant "
                f"{variant!r}. Use generation_core/inpaint_soft/training_* instead, or set "
                "phase4.allow_benchmark_eval_mask_for_generation=true for an explicit ablation."
            )
    return {
        "variant": variant,
        "path": path,
        "role": role,
        "source": source,
        "warning": warning,
        "benchmark_eval_mask_path": benchmark_eval_mask_path,
        "generation_core_mask_path": generation_core_mask_path,
        "available_mask_variants": sorted(str(key) for key in mask_variants.keys()) if isinstance(mask_variants, dict) else [],
    }


def _mask_variant_role(variant: str) -> str:
    if variant in {"generation_core", "inpaint_soft"}:
        return "generation"
    if variant in {"training_medium", "training_wide", "training_soft", "positive_core", "possible_region", "uncertainty_map"}:
        return "training_or_uncertainty"
    if variant in {"eval_mvtec", "benchmark_eval"}:
        return "benchmark_eval"
    if variant in {"eval_tight"}:
        return "pseudo_eval"
    return "custom"


def _benchmark_eval_mask_path(row: dict[str, Any]) -> str:
    settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
    for value in (
        row.get("benchmark_eval_mask_path"),
        settings.get("benchmark_eval_mask_path"),
        (row.get("mask_variant_paths", {}) if isinstance(row.get("mask_variant_paths"), dict) else {}).get("eval_mvtec"),
        (settings.get("mask_variant_paths", {}) if isinstance(settings.get("mask_variant_paths"), dict) else {}).get("eval_mvtec"),
    ):
        if value:
            return str(value)
    return ""


def _generation_core_mask_path(row: dict[str, Any]) -> str:
    settings = row.get("settings", {}) if isinstance(row.get("settings"), dict) else {}
    for value in (
        row.get("generation_core_mask_path"),
        settings.get("generation_core_mask_path"),
        (row.get("mask_variant_paths", {}) if isinstance(row.get("mask_variant_paths"), dict) else {}).get("generation_core"),
        (settings.get("mask_variant_paths", {}) if isinstance(settings.get("mask_variant_paths"), dict) else {}).get("generation_core"),
        row.get("refined_mask_path"),
    ):
        if value:
            return str(value)
    return ""


def _variant_uses_adapter(provider: str, variant: str) -> bool:
    return provider != "qwen" or variant in {"full_qwen_hybrid", "fixed_mask_adapter"}


def _phase4_output_dir(config: AppConfig, provider: str, variant: str) -> Path:
    base = config.output_dir / "phase4" / provider
    return base / variant if provider == "qwen" else base


def _phase4_report_dir(config: AppConfig, provider: str, variant: str) -> Path:
    base = config.report_dir / "phase4" / provider
    return base / variant if provider == "qwen" else base


def _placement_source(provider: str, variant: str) -> str:
    if provider == "qwen" and variant == "clone_harmonized":
        return "qwen_refined_mask_with_real_defect_clone"
    if provider == "qwen" and variant == "ip_adapter_hybrid":
        return "qwen_refined_mask_with_visual_reference_ip_adapter"
    if provider == "qwen" and variant == "latent_blend_harmonized":
        return "qwen_refined_mask_with_latent_reference_blend"
    if provider == "qwen":
        return "qwen_refined_mask"
    return "heuristic_refined_mask"


def _load_inpaint_pipeline(config: AppConfig, device: str) -> Any:
    from diffusers import StableDiffusionInpaintPipeline

    model_config = dict(config.data["models"]["sd15"])
    dtype = getattr(torch, str(model_config.get("dtype", "float16"))) if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_config["base_model"],
        torch_dtype=dtype,
        local_files_only=bool(model_config.get("local_files_only", True)),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _qwen_sd15_render(
    config: AppConfig,
    phase4: dict[str, Any],
    pipe: Any,
    adapter: GatedProjectionAdapter,
    adapter_config: dict[str, Any],
    row: dict[str, Any],
    prompt: str,
    tokens: torch.Tensor,
    image: Image.Image,
    inpaint_mask: Image.Image,
    device: str,
    generation_seed: int | None = None,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    size = int(phase4.get("image_size", config.data.get("generation", {}).get("image_size", 512)))
    negative_prompt = str(phase4.get("negative_prompt", config.data.get("generation", {}).get("negative_prompt", "")))
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    projected = adapter.project_tokens(tokens).to(dtype=prompt_embeds.dtype)
    noise_std = _adapter_token_noise_std(phase4, generation_seed)
    if noise_std > 0.0:
        generator = torch.Generator(device=device).manual_seed(
            int(generation_seed if generation_seed is not None else row.get("seed", config.data.get("generation", {}).get("seed", 1337)))
        )
        projected = projected + torch.randn(projected.shape, generator=generator, device=projected.device, dtype=projected.dtype) * noise_std
    combined = torch.cat([prompt_embeds, projected], dim=1)
    negative_extension = torch.zeros(
        (negative_prompt_embeds.shape[0], projected.shape[1], negative_prompt_embeds.shape[2]),
        device=device,
        dtype=negative_prompt_embeds.dtype,
    )
    negative_combined = torch.cat([negative_prompt_embeds, negative_extension], dim=1)
    generator = torch.Generator(device=device).manual_seed(
        int(generation_seed if generation_seed is not None else row.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    )
    output = pipe(
        prompt_embeds=combined,
        negative_prompt_embeds=negative_combined,
        image=_critic_guided_input_image(image, inpaint_mask, phase4, generation_seed).resize((size, size), Image.Resampling.BILINEAR),
        mask_image=inpaint_mask.resize((size, size), Image.Resampling.BILINEAR),
        guidance_scale=float(phase4.get("guidance_scale", config.data["models"]["sd15"].get("guidance_scale", 7.5))),
        num_inference_steps=int(phase4.get("num_inference_steps", config.data["models"]["sd15"].get("num_inference_steps", 20))),
        strength=float(phase4.get("strength", config.data["models"]["sd15"].get("strength", 0.55))),
        generator=generator,
    ).images[0]
    return output.resize(image.size, Image.Resampling.BILINEAR), combined, projected


def _qwen_sd15_text_render(
    config: AppConfig,
    phase4: dict[str, Any],
    pipe: Any,
    prompt: str,
    row: dict[str, Any],
    image: Image.Image,
    inpaint_mask: Image.Image,
    device: str,
    generation_seed: int | None = None,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    size = int(phase4.get("image_size", config.data.get("generation", {}).get("image_size", 512)))
    negative_prompt = str(phase4.get("negative_prompt", config.data.get("generation", {}).get("negative_prompt", "")))
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(generation_seed if generation_seed is not None else row.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    )
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=_critic_guided_input_image(image, inpaint_mask, phase4, generation_seed).resize((size, size), Image.Resampling.BILINEAR),
        mask_image=inpaint_mask.resize((size, size), Image.Resampling.BILINEAR),
        guidance_scale=float(phase4.get("guidance_scale", config.data["models"]["sd15"].get("guidance_scale", 7.5))),
        num_inference_steps=int(phase4.get("num_inference_steps", config.data["models"]["sd15"].get("num_inference_steps", 20))),
        strength=float(phase4.get("strength", config.data["models"]["sd15"].get("strength", 0.55))),
        generator=generator,
    ).images[0]
    projected = torch.empty(
        (prompt_embeds.shape[0], 0, prompt_embeds.shape[-1]),
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    return output.resize(image.size, Image.Resampling.BILINEAR), prompt_embeds, projected


def _qwen_clone_harmonized_render(
    config: AppConfig,
    phase4: dict[str, Any],
    pipe: Any,
    prompt: str,
    row: dict[str, Any],
    source_index: dict[tuple[str, str], list[dict[str, str]]],
    image: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    device: str,
    generation_seed: int | None = None,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    source = _clone_source_for_row(row, source_index)
    cloned = _normal_clone_to_target(
        source_image=Image.open(source["image_path"]).convert("RGB"),
        source_mask=Image.open(source["mask_path"]).convert("L"),
        target_image=image,
        target_mask=refined_mask,
        diversity=_diversity_metadata(phase4, generation_seed, "clone_harmonized"),
    )
    size = int(phase4.get("image_size", config.data.get("generation", {}).get("image_size", 512)))
    negative_prompt = str(phase4.get("negative_prompt", config.data.get("generation", {}).get("negative_prompt", "")))
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(generation_seed if generation_seed is not None else row.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    )
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=_critic_guided_input_image(cloned, inpaint_mask, phase4, generation_seed).resize((size, size), Image.Resampling.BILINEAR),
        mask_image=inpaint_mask.resize((size, size), Image.Resampling.BILINEAR),
        guidance_scale=float(phase4.get("clone_harmonize_guidance_scale", phase4.get("guidance_scale", 5.0))),
        num_inference_steps=int(phase4.get("clone_harmonize_steps", 8)),
        strength=float(phase4.get("clone_harmonize_strength", 0.20)),
        generator=generator,
    ).images[0]
    projected = torch.empty(
        (prompt_embeds.shape[0], 0, prompt_embeds.shape[-1]),
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    return output.resize(image.size, Image.Resampling.BILINEAR), prompt_embeds, projected


def _qwen_ip_adapter_hybrid_render(
    config: AppConfig,
    phase4: dict[str, Any],
    pipe: Any,
    prompt: str,
    row: dict[str, Any],
    source_index: dict[tuple[str, str], list[dict[str, str]]],
    image: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    device: str,
    generation_seed: int | None = None,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = _clone_source_for_row(row, source_index)
    source_image = Image.open(source["image_path"]).convert("RGB")
    source_mask = Image.open(source["mask_path"]).convert("L")
    reference_patch = _reference_patch(source_image, source_mask)
    size = int(phase4.get("image_size", config.data.get("generation", {}).get("image_size", 512)))
    negative_prompt = str(phase4.get("negative_prompt", config.data.get("generation", {}).get("negative_prompt", "")))
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    projected = torch.empty(
        (prompt_embeds.shape[0], 0, prompt_embeds.shape[-1]),
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(generation_seed if generation_seed is not None else row.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    )
    metadata = {
        "source_image_path": source["image_path"],
        "source_mask_path": source["mask_path"],
        "reference_patch_size": list(reference_patch.size),
        "conditioning": "clip_text_plus_ip_adapter_image" if _ip_adapter_enabled(phase4) else "clip_text_reference_inpaint_fallback",
    }
    if _ip_adapter_enabled(phase4) and _ensure_ip_adapter_loaded(pipe, config, phase4, metadata):
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=_critic_guided_input_image(image, inpaint_mask, phase4, generation_seed).resize((size, size), Image.Resampling.BILINEAR),
            mask_image=inpaint_mask.resize((size, size), Image.Resampling.BILINEAR),
            ip_adapter_image=reference_patch.resize((size, size), Image.Resampling.BILINEAR),
            guidance_scale=float(phase4.get("ip_adapter_guidance_scale", phase4.get("guidance_scale", 7.0))),
            num_inference_steps=int(phase4.get("ip_adapter_steps", phase4.get("num_inference_steps", 18))),
            strength=float(phase4.get("ip_adapter_strength", phase4.get("strength", 0.50))),
            generator=generator,
        ).images[0]
        metadata["status"] = "ip_adapter_loaded"
    else:
        cloned = _normal_clone_to_target(
            source_image=source_image,
            source_mask=source_mask,
            target_image=image,
            target_mask=refined_mask,
            diversity=_diversity_metadata(phase4, generation_seed, "ip_adapter_hybrid"),
        )
        output = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=_critic_guided_input_image(cloned, inpaint_mask, phase4, generation_seed).resize((size, size), Image.Resampling.BILINEAR),
            mask_image=inpaint_mask.resize((size, size), Image.Resampling.BILINEAR),
            guidance_scale=float(phase4.get("ip_adapter_fallback_guidance_scale", phase4.get("guidance_scale", 5.0))),
            num_inference_steps=int(phase4.get("ip_adapter_fallback_steps", phase4.get("clone_harmonize_steps", 8))),
            strength=float(phase4.get("ip_adapter_fallback_strength", phase4.get("clone_harmonize_strength", 0.18))),
            generator=generator,
        ).images[0]
        metadata["status"] = metadata.get("status", "reference_inpaint_fallback")
    return output.resize(image.size, Image.Resampling.BILINEAR), prompt_embeds, projected, metadata


def _qwen_latent_blend_harmonized_render(
    config: AppConfig,
    phase4: dict[str, Any],
    pipe: Any,
    prompt: str,
    row: dict[str, Any],
    source_index: dict[tuple[str, str], list[dict[str, str]]],
    image: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    device: str,
    generation_seed: int | None = None,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = _clone_source_for_row(row, source_index)
    source_image = Image.open(source["image_path"]).convert("RGB")
    source_mask = Image.open(source["mask_path"]).convert("L")
    aligned_reference = _normal_clone_to_target(
        source_image=source_image,
        source_mask=source_mask,
        target_image=image,
        target_mask=refined_mask,
        diversity=_diversity_metadata(phase4, generation_seed, "latent_blend_harmonized"),
    )
    size = int(phase4.get("image_size", config.data.get("generation", {}).get("image_size", 512)))
    negative_prompt = str(phase4.get("negative_prompt", config.data.get("generation", {}).get("negative_prompt", "")))
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )
    metadata = {
        "source_image_path": source["image_path"],
        "source_mask_path": source["mask_path"],
        "conditioning": "clip_text_plus_latent_reference_blend",
    }
    blended = _vae_latent_blend(
        pipe,
        background=image.resize((size, size), Image.Resampling.BILINEAR),
        reference=aligned_reference.resize((size, size), Image.Resampling.BILINEAR),
        mask=refined_mask.resize((size, size), Image.Resampling.BILINEAR),
        device=device,
        blend_strength=float(phase4.get("latent_blend_strength", 1.0)),
        metadata=metadata,
    )
    generator = torch.Generator(device=device).manual_seed(
        int(generation_seed if generation_seed is not None else row.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    )
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=_critic_guided_input_image(blended, inpaint_mask.resize((size, size), Image.Resampling.BILINEAR), phase4, generation_seed),
        mask_image=inpaint_mask.resize((size, size), Image.Resampling.BILINEAR),
        guidance_scale=float(phase4.get("latent_blend_guidance_scale", phase4.get("guidance_scale", 5.0))),
        num_inference_steps=int(phase4.get("latent_blend_harmonize_steps", phase4.get("clone_harmonize_steps", 8))),
        strength=float(phase4.get("latent_blend_harmonize_strength", phase4.get("clone_harmonize_strength", 0.16))),
        generator=generator,
    ).images[0]
    projected = torch.empty(
        (prompt_embeds.shape[0], 0, prompt_embeds.shape[-1]),
        device=prompt_embeds.device,
        dtype=prompt_embeds.dtype,
    )
    return output.resize(image.size, Image.Resampling.BILINEAR), prompt_embeds, projected, metadata


def _ip_adapter_enabled(phase4: dict[str, Any]) -> bool:
    raw = phase4.get("ip_adapter", {})
    return isinstance(raw, dict) and bool(raw.get("enabled", False))


def _ensure_ip_adapter_loaded(pipe: Any, config: AppConfig, phase4: dict[str, Any], metadata: dict[str, Any]) -> bool:
    if bool(getattr(pipe, "_iadgen_ip_adapter_loaded", False)):
        metadata["ip_adapter_cached_on_pipe"] = True
        return True
    if not hasattr(pipe, "load_ip_adapter"):
        metadata["status"] = "ip_adapter_unavailable_on_pipeline"
        return False
    settings = dict(phase4.get("ip_adapter", {}))
    model_id = str(settings.get("model_id", config.data.get("models", {}).get("ip_adapter", {}).get("model_id", "")))
    weight_name = str(settings.get("weight_name", config.data.get("models", {}).get("ip_adapter", {}).get("weight_name", "")))
    if not model_id or not weight_name:
        metadata["status"] = "ip_adapter_not_configured"
        return False
    kwargs: dict[str, Any] = {"weight_name": weight_name}
    for key in ("subfolder", "image_encoder_folder"):
        value = settings.get(key, config.data.get("models", {}).get("ip_adapter", {}).get(key))
        if value:
            kwargs[key] = str(value)
    if "local_files_only" in settings:
        kwargs["local_files_only"] = bool(settings["local_files_only"])
    try:
        pipe.load_ip_adapter(model_id, **kwargs)
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(float(settings.get("scale", 0.65)))
        setattr(pipe, "_iadgen_ip_adapter_loaded", True)
        metadata["ip_adapter_model_id"] = model_id
        metadata["ip_adapter_weight_name"] = weight_name
        return True
    except Exception as exc:  # pragma: no cover - depends on optional external weights.
        if not bool(settings.get("fallback_to_reference_inpaint", True)):
            raise
        metadata["status"] = "ip_adapter_load_failed_reference_inpaint_fallback"
        metadata["ip_adapter_error"] = str(exc)
        return False


def _reference_patch(source_image: Image.Image, source_mask: Image.Image) -> Image.Image:
    bbox = source_mask.convert("L").point(lambda value: 255 if value > 0 else 0).getbbox()
    if bbox is None:
        raise ValueError("IP-Adapter reference source mask is empty")
    pad = max(2, int(round(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.20)))
    padded = (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(source_image.width, bbox[2] + pad),
        min(source_image.height, bbox[3] + pad),
    )
    return source_image.crop(padded).convert("RGB")


def _vae_latent_blend(
    pipe: Any,
    *,
    background: Image.Image,
    reference: Image.Image,
    mask: Image.Image,
    device: str,
    blend_strength: float,
    metadata: dict[str, Any],
) -> Image.Image:
    if not hasattr(pipe, "vae"):
        metadata["status"] = "pixel_alpha_blend_fallback_no_vae"
        return _pixel_alpha_blend(background, reference, mask, blend_strength)
    try:
        vae = pipe.vae
        dtype = next(vae.parameters()).dtype
        bg = _image_to_vae_tensor(background, device, dtype)
        ref = _image_to_vae_tensor(reference, device, dtype)
        mask_tensor = torch.from_numpy(np.asarray(mask.convert("L"), dtype=np.float32) / 255.0)[None, None].to(device=device, dtype=dtype)
        with torch.no_grad():
            bg_latent = vae.encode(bg).latent_dist.mean * float(getattr(vae.config, "scaling_factor", 0.18215))
            ref_latent = vae.encode(ref).latent_dist.mean * float(getattr(vae.config, "scaling_factor", 0.18215))
            latent_mask = torch.nn.functional.interpolate(mask_tensor, size=bg_latent.shape[-2:], mode="bilinear", align_corners=False)
            latent_mask = latent_mask.clamp(0.0, 1.0) * float(np.clip(blend_strength, 0.0, 1.0))
            mixed = bg_latent * (1.0 - latent_mask) + ref_latent * latent_mask
            decoded = vae.decode(mixed / float(getattr(vae.config, "scaling_factor", 0.18215))).sample
        metadata["status"] = "vae_latent_blend"
        return _vae_tensor_to_image(decoded)
    except Exception as exc:  # pragma: no cover - depends on runtime VAE behavior.
        metadata["status"] = "pixel_alpha_blend_fallback_vae_error"
        metadata["latent_blend_error"] = str(exc)
        return _pixel_alpha_blend(background, reference, mask, blend_strength)


def _image_to_vae_tensor(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=dtype)


def _vae_tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = ((tensor.detach().float().clamp(-1.0, 1.0).cpu()[0].permute(1, 2, 0).numpy() + 1.0) * 127.5)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _pixel_alpha_blend(background: Image.Image, reference: Image.Image, mask: Image.Image, blend_strength: float) -> Image.Image:
    bg = np.asarray(background.convert("RGB"), dtype=np.float32)
    ref = np.asarray(reference.resize(background.size, Image.Resampling.BILINEAR).convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.resize(background.size, Image.Resampling.BILINEAR).convert("L"), dtype=np.float32) / 255.0
    alpha = np.clip(alpha * float(np.clip(blend_strength, 0.0, 1.0)), 0.0, 1.0)[..., None]
    return Image.fromarray(np.clip(bg * (1.0 - alpha) + ref * alpha, 0, 255).astype(np.uint8), mode="RGB")


def _clone_source_index(config: AppConfig) -> dict[tuple[str, str], list[dict[str, str]]]:
    manifest = load_manifest(config)
    index: dict[tuple[str, str], list[dict[str, str]]] = {}
    for target in manifest["targets"].values():
        key = (str(target["category"]), str(target["defect_type"]))
        rows: list[dict[str, str]] = []
        for row in target["adaptation"]:
            rows.append(
                {
                    "image_path": str(row["image_path"]),
                    "mask_path": str(row.get("training_mask_path") or row["mask_path"]),
                }
            )
        index[key] = rows
    return index


def _clone_source_for_row(row: dict[str, Any], source_index: dict[tuple[str, str], list[dict[str, str]]]) -> dict[str, str]:
    key = (str(row["category"]), str(row["defect_type"]))
    sources = source_index.get(key, [])
    if not sources:
        raise ValueError(f"clone_harmonized requires at least one adaptation source for {key[0]}/{key[1]}")
    seed = int(row.get("seed", 0))
    return sources[seed % len(sources)]


def _normal_clone_to_target(
    *,
    source_image: Image.Image,
    source_mask: Image.Image,
    target_image: Image.Image,
    target_mask: Image.Image,
    diversity: dict[str, Any] | None = None,
) -> Image.Image:
    import cv2

    source_bbox = source_mask.convert("L").point(lambda value: 255 if value > 0 else 0).getbbox()
    target_bbox = target_mask.convert("L").point(lambda value: 255 if value > 0 else 0).getbbox()
    if source_bbox is None:
        raise ValueError("clone_harmonized source mask is empty")
    if target_bbox is None:
        raise ValueError("clone_harmonized target mask is empty")
    target_width = max(1, target_bbox[2] - target_bbox[0])
    target_height = max(1, target_bbox[3] - target_bbox[1])
    source_patch = source_image.crop(source_bbox).resize((target_width, target_height), Image.Resampling.BILINEAR)
    mask_patch = source_mask.crop(source_bbox).resize((target_width, target_height), Image.Resampling.NEAREST)
    source_patch, mask_patch = _apply_clone_diversity(source_patch, mask_patch, diversity or {})
    mask_arr = (np.asarray(mask_patch.convert("L"), dtype=np.uint8) > 0).astype(np.uint8) * 255
    if mask_arr.sum() == 0:
        raise ValueError("clone_harmonized resized source mask is empty")
    src = np.asarray(source_patch.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
    dst = np.asarray(target_image.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
    center = ((target_bbox[0] + target_bbox[2]) // 2, (target_bbox[1] + target_bbox[3]) // 2)
    center = (max(0, min(dst.shape[1] - 1, center[0])), max(0, min(dst.shape[0] - 1, center[1])))
    cloned = cv2.seamlessClone(src, dst, mask_arr, center, cv2.NORMAL_CLONE)
    return Image.fromarray(cloned[:, :, ::-1]).convert("RGB")


def _render_prompt(config: AppConfig, row: dict[str, Any], profile_settings: dict[str, Any] | None = None) -> str:
    prompts = dict(config.data.get("generation", {}).get("prompt_by_defect", {}))
    base_prompt = str(prompts.get(str(row["defect_type"]), row["prompt"]))
    if profile_settings is None:
        return base_prompt
    template = str(profile_settings.get("prompt_template", "{base_prompt}"))
    return template.format(
        base_prompt=base_prompt,
        category=str(row.get("category", "")),
        defect_type=str(row.get("defect_type", "")),
    )


def _negative_prompt(config: AppConfig, profile_settings: dict[str, Any]) -> str:
    base = str(config.data.get("generation", {}).get("negative_prompt", ""))
    profile = str(profile_settings.get("negative_prompt", ""))
    return ", ".join(part for part in (base, profile) if part)


def _generation_seed(row: dict[str, Any], seed_index: int) -> int:
    base = int(row.get("seed", 1337))
    return base + seed_index * 1_000_003


def _diversity_metadata(phase4: dict[str, Any], generation_seed: int | None, variant: str) -> dict[str, Any]:
    raw = phase4.get("diversity", {})
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return {
            "enabled": False,
            "adapter_token_noise_std": 0.0,
            "geometry_jitter": False,
            "color_jitter": False,
            "elastic_deform": False,
        }
    seed = int(generation_seed if generation_seed is not None else 0)
    rng = np.random.default_rng(seed + 91_337)
    token_std = _adapter_token_noise_std(phase4, generation_seed)
    max_rotation = float(raw.get("max_rotation_degrees", 7.0))
    brightness = float(rng.uniform(1.0 - float(raw.get("brightness_jitter", 0.08)), 1.0 + float(raw.get("brightness_jitter", 0.08))))
    contrast = float(rng.uniform(1.0 - float(raw.get("contrast_jitter", 0.08)), 1.0 + float(raw.get("contrast_jitter", 0.08))))
    return {
        "enabled": True,
        "variant": variant,
        "adapter_token_noise_std": token_std,
        "geometry_jitter": bool(raw.get("geometry_jitter", False)),
        "color_jitter": bool(raw.get("color_jitter", False)),
        "elastic_deform": bool(raw.get("elastic_deform", False)),
        "rotation_degrees": float(rng.uniform(-max_rotation, max_rotation)) if bool(raw.get("geometry_jitter", False)) else 0.0,
        "brightness_factor": brightness if bool(raw.get("color_jitter", False)) else 1.0,
        "contrast_factor": contrast if bool(raw.get("color_jitter", False)) else 1.0,
        "seed": seed,
    }


def _adapter_token_noise_std(phase4: dict[str, Any], generation_seed: int | None) -> float:
    raw = phase4.get("diversity", {})
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return 0.0
    values = raw.get("adapter_token_noise_std", 0.0)
    if isinstance(values, list):
        parsed = [float(value) for value in values]
        if not parsed:
            return 0.0
        seed = int(generation_seed if generation_seed is not None else 0)
        return max(0.0, parsed[abs(seed) % len(parsed)])
    return max(0.0, float(values))


def _apply_clone_diversity(
    source_patch: Image.Image,
    mask_patch: Image.Image,
    diversity: dict[str, Any],
) -> tuple[Image.Image, Image.Image]:
    if not bool(diversity.get("enabled", False)):
        return source_patch, mask_patch
    patch = source_patch
    mask = mask_patch
    if bool(diversity.get("geometry_jitter", False)):
        angle = float(diversity.get("rotation_degrees", 0.0))
        patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=tuple(np.asarray(patch).reshape(-1, 3).mean(axis=0).astype(int)))
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        if bool(diversity.get("elastic_deform", False)):
            # Lightweight deterministic deformation: a tiny shear-like roll that preserves the binary mask contract.
            shift = max(1, int(round(min(patch.size) * 0.015)))
            patch_arr = np.asarray(patch.convert("RGB"), dtype=np.uint8).copy()
            mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8).copy()
            for y in range(patch_arr.shape[0]):
                offset = int(round(np.sin(y / max(1.0, patch_arr.shape[0] / 3.0)) * shift))
                patch_arr[y] = np.roll(patch_arr[y], offset, axis=0)
                mask_arr[y] = np.roll(mask_arr[y], offset, axis=0)
            patch = Image.fromarray(patch_arr, mode="RGB")
            mask = Image.fromarray(mask_arr, mode="L")
    if bool(diversity.get("color_jitter", False)):
        patch = ImageEnhance.Brightness(patch).enhance(float(diversity.get("brightness_factor", 1.0)))
        patch = ImageEnhance.Contrast(patch).enhance(float(diversity.get("contrast_factor", 1.0)))
    return patch, mask


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    return value


def _heuristic_integrated_render(
    image: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    projected_tokens: torch.Tensor,
    defect_type: str,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    refined = np.asarray(refined_mask.convert("L"), dtype=np.float32) / 255.0
    inpaint = np.asarray(inpaint_mask.convert("L"), dtype=np.float32) / 255.0
    token_mean = float(projected_tokens.mean())
    token_std = float(projected_tokens.std())
    rng = np.random.default_rng(int(abs(token_mean) * 1_000_000) % (2**32))
    noise = rng.normal(0.0, 1.0, size=base.shape[:2]).astype(np.float32)
    if defect_type == "crack":
        defect_color = np.asarray([28.0, 24.0, 22.0], dtype=np.float32)
        contrast = 0.70 + min(0.20, token_std)
    else:
        defect_color = np.asarray([185.0, 175.0, 160.0], dtype=np.float32)
        contrast = 0.45 + min(0.25, token_std)
    texture = np.clip(refined * (1.0 + 0.12 * noise), 0.0, 1.0)
    alpha = np.clip(texture * contrast + inpaint * 0.08, 0.0, 0.95)[..., None]
    rendered = base * (1.0 - alpha) + defect_color.reshape(1, 1, 3) * alpha
    preserved = np.where((inpaint[..., None] > 0.0), rendered, base)
    smoothed = np.asarray(Image.fromarray(np.clip(preserved, 0, 255).astype(np.uint8)).filter(ImageFilter.SMOOTH_MORE), dtype=np.float32)
    final = np.where((inpaint[..., None] > 0.0), smoothed, base)
    return Image.fromarray(np.clip(final, 0, 255).astype(np.uint8))


def _preservation_metrics(image: Image.Image, output: Image.Image, inpaint_mask: Image.Image) -> tuple[float, float]:
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    generated = np.asarray(output.convert("RGB"), dtype=np.float32) / 255.0
    inpaint = np.asarray(inpaint_mask.convert("L"), dtype=np.uint8) > 0
    outside = ~inpaint
    if outside.any():
        preservation_l1 = float(np.abs(generated[outside] - source[outside]).mean())
    else:
        preservation_l1 = 0.0
    changed = np.abs(generated - source).mean(axis=2) > (1.0 / 255.0)
    mask_changed_fraction = float((changed & inpaint).sum() / max(1, int(inpaint.sum())))
    return preservation_l1, mask_changed_fraction


def _quality_diagnostics(
    image: Image.Image,
    output: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    phase4: dict[str, Any],
    background_preservation_l1: float,
    mask_changed_fraction: float,
    morphology: str | None = None,
    label_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    generated = np.asarray(output.convert("RGB"), dtype=np.float32) / 255.0
    diff = np.abs(generated - source).mean(axis=2)
    refined_alpha = np.asarray(refined_mask.convert("L"), dtype=np.float32) / 255.0
    inpaint_alpha = np.asarray(inpaint_mask.convert("L"), dtype=np.float32) / 255.0
    refined = refined_alpha > 0.0
    inpaint = inpaint_alpha > 0.0
    policy_name = str((label_policy or {}).get("label_policy", "hard_mask_ok"))
    use_soft_support = policy_name == "soft_mask_only" or morphology == "multi_scuff"
    visibility_alpha = inpaint_alpha if use_soft_support else refined_alpha
    visibility_support = visibility_alpha > 0.0
    changed = diff > float(phase4.get("quality_change_threshold", 12.0 / 255.0))
    if visibility_support.any():
        weights = visibility_alpha[visibility_support]
        defect_visibility = float((diff[visibility_support] * weights).sum() / max(1e-6, float(weights.sum())))
    else:
        defect_visibility = 0.0
    outside_refined = ~refined
    outside_refined_change = float((changed & outside_refined).sum() / max(1, int(outside_refined.sum())))
    inpaint_area = float(inpaint.mean())
    flags: list[str] = []
    if defect_visibility < float(phase4.get("min_defect_visibility_score", 0.015)):
        flags.append("weak_visible_defect")
    if background_preservation_l1 > float(phase4.get("max_background_l1", 0.05)):
        flags.append("high_background_change")
    if outside_refined_change > float(phase4.get("max_outside_refined_change_fraction", 0.25)):
        flags.append("high_outside_refined_change")
    if inpaint_area > float(phase4.get("max_inpaint_area_fraction", 0.20)):
        flags.append("overbroad_inpaint_mask")
    if mask_changed_fraction < float(phase4.get("min_mask_changed_fraction", 0.30)):
        flags.append("low_mask_edit_fraction")
    quality_score = (
        defect_visibility
        - 2.0 * background_preservation_l1
        - 0.10 * max(0.0, outside_refined_change - float(phase4.get("target_outside_refined_change_fraction", 0.05)))
        - 0.05 * len(flags)
    )
    return {
        "defect_visibility_score": defect_visibility,
        "outside_refined_change_fraction": outside_refined_change,
        "inpaint_mask_area_fraction": inpaint_area,
        "generation_quality_score": float(quality_score),
        "quality_flags": flags,
    }


def _critic_guided_settings(phase4: dict[str, Any], provider: str) -> dict[str, Any]:
    raw = phase4.get("critic_guided_regeneration", {})
    if not isinstance(raw, dict):
        raw = {}
    enabled = bool(raw.get("enabled", False)) and provider == "qwen"
    max_attempts = int(raw.get("max_attempts", 3 if enabled else 1))
    if max_attempts < 1:
        raise ValueError("phase4.critic_guided_regeneration.max_attempts must be positive")
    return {
        "enabled": enabled,
        "max_attempts": max_attempts if enabled else 1,
        "exhaustive": bool(raw.get("exhaustive", False)),
        "min_adaptive_mask_coverage_score": float(raw.get("min_adaptive_mask_coverage_score", 0.35)),
        "min_mask_coverage_score": float(raw.get("min_mask_coverage_score", raw.get("min_adaptive_mask_coverage_score", 0.35))),
        "min_feature_alignment_score": float(raw.get("min_feature_alignment_score", 0.0)),
        "min_texture_preservation_score": float(raw.get("min_texture_preservation_score", 0.0)),
        "min_leakage_score": float(raw.get("min_leakage_score", 0.45)),
        "min_morphology_fit_score": float(raw.get("min_morphology_fit_score", 0.0)),
        "critic_image_size": int(raw.get("critic_image_size", 160)),
        "change_threshold": float(raw.get("change_threshold", 0.045)),
        "target_mask_coverage": float(raw.get("target_mask_coverage", 0.38)),
        "max_mask_coverage": float(raw.get("max_mask_coverage", 0.92)),
        "max_outside_change_fraction": float(raw.get("max_outside_change_fraction", 0.18)),
        "strength_step": float(raw.get("strength_step", 0.08)),
        "guidance_step": float(raw.get("guidance_step", 0.35)),
        "step_increment": int(raw.get("step_increment", 2)),
        "max_strength": float(raw.get("max_strength", 0.78)),
        "max_guidance_scale": float(raw.get("max_guidance_scale", 9.0)),
        "max_inference_steps": int(raw.get("max_inference_steps", 28)),
        "retry_seed_stride": int(raw.get("retry_seed_stride", 97_531)),
        "low_coverage_extra_strength": float(raw.get("low_coverage_extra_strength", 0.10)),
        "low_coverage_extra_guidance": float(raw.get("low_coverage_extra_guidance", 0.45)),
        "low_coverage_extra_steps": int(raw.get("low_coverage_extra_steps", 2)),
        "max_local_mask_noise_boost": float(raw.get("max_local_mask_noise_boost", 0.08)),
        "local_mask_noise_step": float(raw.get("local_mask_noise_step", 0.035)),
        "max_postprocess_mask_contrast_boost": float(raw.get("max_postprocess_mask_contrast_boost", 0.16)),
        "postprocess_mask_contrast_step": float(raw.get("postprocess_mask_contrast_step", 0.08)),
        "final_coverage_repair_enabled": bool(raw.get("final_coverage_repair_enabled", enabled)),
        "final_coverage_repair_max_attempts": int(raw.get("final_coverage_repair_max_attempts", 3)),
        "final_coverage_repair_step": float(raw.get("final_coverage_repair_step", 0.12)),
        "final_coverage_repair_max_boost": float(raw.get("final_coverage_repair_max_boost", 0.42)),
        "final_coverage_repair_profiles": raw.get("final_coverage_repair_profiles", {}),
        "low_coverage_prompt_suffix": str(
            raw.get(
                "low_coverage_prompt_suffix",
                "make the defect visibly fill the selected mask with clear local contrast and physical surface damage",
            )
        ),
        "prompt_suffix": str(
            raw.get(
                "prompt_suffix",
                "clearly visible defect occupying the selected mask, strong local material damage",
            )
        ),
    }


def _critic_guided_attempt_settings(
    phase4: dict[str, Any],
    prompt: str,
    row: dict[str, Any],
    generation_seed: int,
    attempt_index: int,
    critic_settings: dict[str, Any],
    previous_attempt: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, int]:
    attempt_phase4 = dict(phase4)
    if not bool(critic_settings.get("enabled", False)) or attempt_index <= 0:
        return attempt_phase4, prompt, generation_seed
    previous_reasons = {str(reason) for reason in (previous_attempt or {}).get("reject_reasons", [])}
    # Visibility-targeted retry: if the previous attempt was under-visible
    # (either a hard reject or below a configured target band), escalate the
    # same magnitude knobs. Opt-in via target_defect_visibility (default 0 =
    # off) so behavior is unchanged until a re-audit sets the band.
    previous_visibility = float((previous_attempt or {}).get("critic_defect_visibility_score", 1.0))
    visibility_target = float(critic_settings.get("target_defect_visibility", 0.0))
    low_visibility_retry = "low_defect_visibility" in previous_reasons or (
        visibility_target > 0.0 and previous_visibility < visibility_target
    )
    low_coverage_retry = "low_mask_coverage" in previous_reasons or not previous_reasons
    boost_retry = low_coverage_retry or low_visibility_retry
    strength_delta = float(critic_settings["strength_step"]) * attempt_index
    guidance_delta = float(critic_settings["guidance_step"]) * attempt_index
    step_delta = int(critic_settings["step_increment"]) * attempt_index
    if boost_retry:
        strength_delta += float(critic_settings.get("low_coverage_extra_strength", 0.0))
        guidance_delta += float(critic_settings.get("low_coverage_extra_guidance", 0.0))
        step_delta += int(critic_settings.get("low_coverage_extra_steps", 0))
        attempt_phase4["critic_retry_reason"] = (
            "low_defect_visibility" if (low_visibility_retry and not low_coverage_retry) else "low_mask_coverage"
        )
        morphology = _morphology_for_row(row)
        morph_multiplier = {
            "micro_chip": 1.25,
            "tooth_break": 1.15,
            "scratch_band": 1.0,
            "single_stroke": 0.9,
            "crack_band": 1.0,
            "multi_scuff": 0.7,
        }.get(morphology, 1.0)
        attempt_phase4["local_mask_noise_boost"] = min(
            float(critic_settings.get("max_local_mask_noise_boost", 0.08)),
            float(critic_settings.get("local_mask_noise_step", 0.035)) * attempt_index * morph_multiplier,
        )
        attempt_phase4["postprocess_mask_contrast_boost"] = min(
            float(critic_settings.get("max_postprocess_mask_contrast_boost", 0.16)),
            float(critic_settings.get("postprocess_mask_contrast_step", 0.08)) * attempt_index * morph_multiplier,
        )
    for key in (
        "strength",
        "clone_harmonize_strength",
        "ip_adapter_strength",
        "ip_adapter_fallback_strength",
        "latent_blend_harmonize_strength",
    ):
        if key in attempt_phase4 or key == "strength":
            attempt_phase4[key] = min(float(critic_settings["max_strength"]), float(attempt_phase4.get(key, attempt_phase4.get("strength", 0.55))) + strength_delta)
    for key in (
        "guidance_scale",
        "clone_harmonize_guidance_scale",
        "ip_adapter_guidance_scale",
        "ip_adapter_fallback_guidance_scale",
        "latent_blend_guidance_scale",
    ):
        if key in attempt_phase4 or key == "guidance_scale":
            attempt_phase4[key] = min(
                float(critic_settings["max_guidance_scale"]),
                float(attempt_phase4.get(key, attempt_phase4.get("guidance_scale", 7.5))) + guidance_delta,
            )
    for key in (
        "num_inference_steps",
        "clone_harmonize_steps",
        "ip_adapter_steps",
        "ip_adapter_fallback_steps",
        "latent_blend_harmonize_steps",
    ):
        if key in attempt_phase4 or key == "num_inference_steps":
            attempt_phase4[key] = min(
                int(critic_settings["max_inference_steps"]),
                int(attempt_phase4.get(key, attempt_phase4.get("num_inference_steps", 20))) + step_delta,
            )
    suffixes = [str(critic_settings.get("prompt_suffix", "")).strip()]
    if low_coverage_retry:
        suffixes.append(str(critic_settings.get("low_coverage_prompt_suffix", "")).strip())
    suffix = ", ".join(suffix for suffix in suffixes if suffix)
    attempt_prompt = f"{prompt}, {suffix}" if suffix else prompt
    attempt_seed = int(generation_seed) + int(critic_settings.get("retry_seed_stride", 97_531)) * attempt_index
    return attempt_phase4, attempt_prompt, attempt_seed


def _critic_guided_input_image(
    image: Image.Image,
    mask: Image.Image,
    phase4: dict[str, Any],
    generation_seed: int | None,
) -> Image.Image:
    boost = float(phase4.get("local_mask_noise_boost", 0.0))
    if boost <= 0.0:
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = np.asarray(mask.convert("L").resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    if float(alpha.max(initial=0.0)) <= 0.0:
        return image
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    seed = int(generation_seed if generation_seed is not None else phase4.get("seed", 1337))
    rng = np.random.default_rng(seed + 17_071)
    noise = rng.normal(0.0, 255.0 * boost, size=rgb.shape).astype(np.float32)
    dark_bias = -255.0 * boost * 0.35
    perturbed = np.clip(rgb + noise + dark_bias, 0.0, 255.0)
    blend = np.clip(alpha * min(0.55, boost * 4.0), 0.0, 0.55)
    out = rgb * (1.0 - blend) + perturbed * blend
    return Image.fromarray(np.uint8(np.clip(out, 0.0, 255.0)), mode="RGB")


def _phase4_generation_critic(
    image: Image.Image,
    output: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    critic_settings: dict[str, Any],
    *,
    morphology: str = "unknown",
) -> dict[str, Any]:
    if not bool(critic_settings.get("enabled", False)):
        return {
            "critic_score": 1.0,
            "tfidg_lite_score": 1.0,
            "feature_alignment_score": 1.0,
            "adaptive_mask_coverage_score": 1.0,
            "adaptive_mask_coverage_fraction": 1.0,
            "texture_preservation_score": 1.0,
            "leakage_score": 1.0,
            "morphology_fit_score": 1.0,
            "outside_change_fraction": 0.0,
            "shell_change_fraction": 0.0,
            "critic_defect_visibility_score": 1.0,
            "accepted": True,
            "reject_reasons": [],
        }
    scores = score_generation(
        output,
        image,
        refined_mask,
        inpaint_mask,
        critic_settings,
        morphology=morphology,
        require_reference=False,
    )
    return {
        "critic_score": round(float(scores.score), 4),
        "tfidg_lite_score": round(float(scores.score), 4),
        "feature_alignment_score": round(float(scores.feature_alignment_score), 4),
        "adaptive_mask_coverage_score": round(float(scores.adaptive_mask_coverage_score), 4),
        "adaptive_mask_coverage_fraction": round(float(scores.adaptive_mask_coverage_fraction), 4),
        "texture_preservation_score": round(float(scores.texture_preservation_score), 4),
        "leakage_score": round(float(scores.leakage_score), 4),
        "morphology_fit_score": round(float(scores.morphology_fit_score), 4),
        "outside_change_fraction": round(float(scores.outside_change_fraction), 4),
        "shell_change_fraction": round(float(scores.shell_change_fraction), 4),
        "critic_defect_visibility_score": round(float(scores.defect_visibility_score), 4),
        "accepted": scores.accepted,
        "reject_reasons": scores.reject_reasons,
    }


def _critic_guided_visibility_boost(
    background: Image.Image,
    output: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    phase4: dict[str, Any],
    generation_seed: int | None,
    *,
    morphology: str,
) -> tuple[Image.Image, dict[str, Any]]:
    boost = float(phase4.get("postprocess_mask_contrast_boost", 0.0))
    if boost <= 0.0:
        return output, {}
    mask = refined_mask if _mask_area_fraction(refined_mask) > 0.0 else inpaint_mask
    alpha_img = mask.convert("L").resize(output.size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=0.75))
    alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
    if float(alpha.max(initial=0.0)) <= 0.0:
        return output, {}
    rgb = np.asarray(output.convert("RGB"), dtype=np.float32)
    bg = np.asarray(background.convert("RGB").resize(output.size, Image.Resampling.BILINEAR), dtype=np.float32)
    seed = int(generation_seed if generation_seed is not None else phase4.get("seed", 1337))
    rng = np.random.default_rng(seed + 91_337)
    noise = rng.normal(0.0, 255.0 * boost * 0.16, size=rgb.shape).astype(np.float32)
    if morphology in {"multi_scuff", "contamination"}:
        direction = np.where(rgb >= bg, 1.0, -1.0)
        target = np.clip(rgb + direction * 255.0 * boost * 0.32 + noise, 0.0, 255.0)
    else:
        target = np.clip(rgb * (1.0 - boost) + noise, 0.0, 255.0)
    blend = np.clip(alpha[..., None] * boost, 0.0, 0.30)
    out = rgb * (1.0 - blend) + target * blend
    boosted = Image.fromarray(np.uint8(np.clip(out, 0.0, 255.0)), mode="RGB")
    return boosted, {
        "enabled": True,
        "boost": round(boost, 4),
        "morphology": morphology,
        "mask_area_fraction": round(_mask_area_fraction(mask), 6),
    }


def _critic_guided_final_coverage_repair(
    background: Image.Image,
    output: Image.Image,
    refined_mask: Image.Image,
    inpaint_mask: Image.Image,
    critic_settings: dict[str, Any],
    generation_seed: int | None,
    base_attempt_record: dict[str, Any],
    *,
    morphology: str,
) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
    reasons = {str(reason) for reason in base_attempt_record.get("reject_reasons", [])}
    if "low_mask_coverage" not in reasons:
        return output, base_attempt_record, {"enabled": False, "skip_reason": "not_low_mask_coverage"}
    profile = _final_coverage_repair_profile(critic_settings, morphology)
    max_attempts = max(0, int(profile.get("max_attempts", critic_settings.get("final_coverage_repair_max_attempts", 3))))
    if max_attempts <= 0:
        return output, base_attempt_record, {"enabled": False, "skip_reason": "no_repair_attempts"}
    step = float(profile.get("step", critic_settings.get("final_coverage_repair_step", 0.12)))
    max_boost = float(profile.get("max_boost", critic_settings.get("final_coverage_repair_max_boost", 0.42)))
    best_output = output
    best_record = dict(base_attempt_record)
    repair_attempts: list[dict[str, Any]] = []
    for repair_index in range(max_attempts):
        boost = min(max_boost, step * float(repair_index + 1))
        candidate = _mask_local_coverage_repair(
            background,
            output,
            refined_mask if _mask_area_fraction(refined_mask) > 0.0 else inpaint_mask,
            boost,
            generation_seed,
            morphology=morphology,
            profile=profile,
        )
        critic_record = _phase4_generation_critic(
            background,
            candidate,
            refined_mask,
            inpaint_mask,
            critic_settings,
            morphology=morphology,
        )
        repair_record = {
            **critic_record,
            "attempt_index": int(base_attempt_record.get("attempt_index", -1)),
            "repair_index": repair_index,
            "generation_seed": int(generation_seed if generation_seed is not None else 0),
            "strength": float(base_attempt_record.get("strength", 0.0)),
            "guidance_scale": float(base_attempt_record.get("guidance_scale", 0.0)),
            "num_inference_steps": int(base_attempt_record.get("num_inference_steps", 0)),
            "retry_reason": "final_coverage_repair",
            "final_coverage_repair_profile": str(profile.get("name", morphology or "unknown")),
            "final_coverage_repair_mode": str(profile.get("mode", "auto")),
            "final_coverage_repair_boost": round(boost, 4),
            "prompt": str(base_attempt_record.get("prompt", "")),
        }
        repair_attempts.append(repair_record)
        if _critic_attempt_rank(repair_record) > _critic_attempt_rank(best_record):
            best_output = candidate
            best_record = repair_record
        if repair_record["accepted"]:
            break
    return best_output, best_record, {
        "enabled": True,
        "source_attempt_index": int(base_attempt_record.get("attempt_index", -1)),
        "selected_repair_index": int(best_record.get("repair_index", -1)),
        "profile": _json_safe(profile),
        "attempts": repair_attempts,
        "accepted": bool(best_record.get("accepted", False)),
        "reject_reasons": list(best_record.get("reject_reasons", [])),
    }


def _final_coverage_repair_profile(critic_settings: dict[str, Any], morphology: str) -> dict[str, Any]:
    profiles = {
        "scratch_band": {
            "name": "scratch_band",
            "mode": "dark_line",
            "max_attempts": 3,
            "step": 0.14,
            "max_boost": 0.50,
            "blur_radius": 0.55,
            "texture_scale": 0.08,
            "blend_multiplier": 1.45,
            "blend_cap": 0.44,
            "darken_strength": 0.82,
        },
        "crack_band": {
            "name": "crack_band",
            "mode": "dark_line",
            "max_attempts": 3,
            "step": 0.16,
            "max_boost": 0.56,
            "blur_radius": 0.50,
            "texture_scale": 0.09,
            "blend_multiplier": 1.55,
            "blend_cap": 0.48,
            "darken_strength": 0.92,
        },
        "single_stroke": {
            "name": "single_stroke",
            "mode": "dark_line",
            "max_attempts": 3,
            "step": 0.12,
            "max_boost": 0.42,
            "blur_radius": 0.45,
            "texture_scale": 0.06,
            "blend_multiplier": 1.35,
            "blend_cap": 0.38,
            "darken_strength": 0.72,
        },
        "multi_scuff": {
            "name": "multi_scuff",
            "mode": "contrast_scuff",
            "max_attempts": 3,
            "step": 0.09,
            "max_boost": 0.32,
            "blur_radius": 1.25,
            "texture_scale": 0.05,
            "blend_multiplier": 1.00,
            "blend_cap": 0.28,
            "contrast_strength": 0.48,
        },
        "contamination": {
            "name": "contamination",
            "mode": "contrast_scuff",
            "max_attempts": 3,
            "step": 0.10,
            "max_boost": 0.36,
            "blur_radius": 1.10,
            "texture_scale": 0.05,
            "blend_multiplier": 1.05,
            "blend_cap": 0.30,
            "contrast_strength": 0.55,
        },
        "micro_chip": {
            "name": "micro_chip",
            "mode": "bright_chip",
            "max_attempts": 3,
            "step": 0.20,
            "max_boost": 0.68,
            "blur_radius": 0.35,
            "texture_scale": 0.05,
            "blend_multiplier": 1.70,
            "blend_cap": 0.56,
            "brighten_strength": 0.72,
        },
        "tooth_break": {
            "name": "tooth_break",
            "mode": "bright_chip",
            "max_attempts": 3,
            "step": 0.18,
            "max_boost": 0.62,
            "blur_radius": 0.35,
            "texture_scale": 0.05,
            "blend_multiplier": 1.60,
            "blend_cap": 0.52,
            "brighten_strength": 0.66,
        },
        "unknown": {
            "name": "unknown",
            "mode": "dark_line",
            "max_attempts": int(critic_settings.get("final_coverage_repair_max_attempts", 3)),
            "step": float(critic_settings.get("final_coverage_repair_step", 0.12)),
            "max_boost": float(critic_settings.get("final_coverage_repair_max_boost", 0.42)),
            "blur_radius": 0.65,
            "texture_scale": 0.08,
            "blend_multiplier": 1.35,
            "blend_cap": 0.42,
            "darken_strength": 0.75,
        },
    }
    profile = dict(profiles.get(morphology, profiles["unknown"]))
    configured = critic_settings.get("final_coverage_repair_profiles", {})
    if isinstance(configured, dict):
        override = configured.get(morphology) or configured.get(profile["name"]) or configured.get("default")
        if isinstance(override, dict):
            profile.update(override)
    profile["name"] = str(profile.get("name", morphology or "unknown"))
    return profile


def _mask_local_coverage_repair(
    background: Image.Image,
    output: Image.Image,
    mask: Image.Image,
    boost: float,
    generation_seed: int | None,
    *,
    morphology: str,
    profile: dict[str, Any] | None = None,
) -> Image.Image:
    if boost <= 0.0:
        return output
    profile = profile or _final_coverage_repair_profile({}, morphology)
    blur_radius = float(profile.get("blur_radius", 0.65))
    alpha_img = mask.convert("L").resize(output.size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
    if float(alpha.max(initial=0.0)) <= 0.0:
        return output
    rgb = np.asarray(output.convert("RGB"), dtype=np.float32)
    bg = np.asarray(background.convert("RGB").resize(output.size, Image.Resampling.BILINEAR), dtype=np.float32)
    seed = int(generation_seed if generation_seed is not None else 0)
    rng = np.random.default_rng(seed + 550_021)
    gray = np.asarray(output.convert("L"), dtype=np.float32)
    bg_gray = np.asarray(background.convert("L").resize(output.size, Image.Resampling.BILINEAR), dtype=np.float32)
    texture_scale = float(profile.get("texture_scale", 0.08))
    texture = rng.normal(0.0, 255.0 * boost * texture_scale, size=rgb.shape).astype(np.float32)
    darken_strength = float(profile.get("darken_strength", 0.75))
    brighten_strength = float(profile.get("brighten_strength", 0.55))
    contrast_strength = float(profile.get("contrast_strength", 0.45))
    darker_target = np.clip(rgb * (1.0 - darken_strength * boost) + texture, 0.0, 255.0)
    brighter_target = np.clip(rgb + 255.0 * boost * brighten_strength + texture, 0.0, 255.0)
    mode = str(profile.get("mode", "auto"))
    if mode == "contrast_scuff" or morphology in {"multi_scuff", "contamination"}:
        direction = np.where(gray[..., None] >= bg_gray[..., None], 1.0, -1.0)
        target = np.clip(rgb + direction * 255.0 * boost * contrast_strength + texture, 0.0, 255.0)
    elif mode == "bright_chip" or morphology in {"micro_chip", "tooth_break"}:
        target = brighter_target
    else:
        target = darker_target
    blend = np.clip(
        alpha[..., None] * boost * float(profile.get("blend_multiplier", 1.35)),
        0.0,
        float(profile.get("blend_cap", 0.42)),
    )
    repaired = rgb * (1.0 - blend) + target * blend
    return Image.fromarray(np.uint8(np.clip(repaired, 0.0, 255.0)), mode="RGB")


def _mask_area_fraction(mask: Image.Image) -> float:
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return float((arr > 0.03).mean()) if arr.size else 0.0


def _critic_attempt_rank(attempt: dict[str, Any]) -> tuple[int, float, float, float]:
    # critic_score already folds in the visibility weight; visibility is also
    # the explicit tiebreaker (ahead of coverage) so the controller prefers the
    # most visibly-defective attempt rather than the one that merely changed the
    # most pixels.
    return (
        1 if bool(attempt.get("accepted", False)) else 0,
        float(attempt.get("critic_score", 0.0)),
        float(attempt.get("critic_defect_visibility_score", 0.0)),
        float(attempt.get("adaptive_mask_coverage_score", 0.0)),
    )


def _effective_strength(phase4: dict[str, Any], variant: str) -> float:
    if variant == "clone_harmonized":
        return float(phase4.get("clone_harmonize_strength", phase4.get("strength", 0.20)))
    if variant == "ip_adapter_hybrid":
        if _ip_adapter_enabled(phase4):
            return float(phase4.get("ip_adapter_strength", phase4.get("strength", 0.50)))
        return float(phase4.get("ip_adapter_fallback_strength", phase4.get("clone_harmonize_strength", phase4.get("strength", 0.18))))
    if variant == "latent_blend_harmonized":
        return float(phase4.get("latent_blend_harmonize_strength", phase4.get("clone_harmonize_strength", phase4.get("strength", 0.16))))
    return float(phase4.get("strength", 0.55))


def _effective_guidance_scale(phase4: dict[str, Any], variant: str) -> float:
    if variant == "clone_harmonized":
        return float(phase4.get("clone_harmonize_guidance_scale", phase4.get("guidance_scale", 5.0)))
    if variant == "ip_adapter_hybrid":
        if _ip_adapter_enabled(phase4):
            return float(phase4.get("ip_adapter_guidance_scale", phase4.get("guidance_scale", 7.0)))
        return float(phase4.get("ip_adapter_fallback_guidance_scale", phase4.get("guidance_scale", 5.0)))
    if variant == "latent_blend_harmonized":
        return float(phase4.get("latent_blend_guidance_scale", phase4.get("guidance_scale", 5.0)))
    return float(phase4.get("guidance_scale", 7.5))


def _effective_num_inference_steps(phase4: dict[str, Any], variant: str) -> int:
    if variant == "clone_harmonized":
        return int(phase4.get("clone_harmonize_steps", phase4.get("num_inference_steps", 8)))
    if variant == "ip_adapter_hybrid":
        if _ip_adapter_enabled(phase4):
            return int(phase4.get("ip_adapter_steps", phase4.get("num_inference_steps", 18)))
        return int(phase4.get("ip_adapter_fallback_steps", phase4.get("clone_harmonize_steps", phase4.get("num_inference_steps", 8))))
    if variant == "latent_blend_harmonized":
        return int(phase4.get("latent_blend_harmonize_steps", phase4.get("clone_harmonize_steps", phase4.get("num_inference_steps", 8))))
    return int(phase4.get("num_inference_steps", 20))


def _phase4_config(config: AppConfig) -> dict[str, Any]:
    return dict(config.data.get("phase4", {}))


def _phase4_fingerprint(config: AppConfig, provider: str) -> str:
    return fingerprint(
        {
            "split": config.split_spec_fingerprint(),
            "phase2": config.data.get("phase2", {}),
            "phase3": config.data.get("phase3", {}),
            "phase4": config.data.get("phase4", {}),
            "provider": provider,
            "schema_version": PHASE4_SCHEMA_VERSION,
        }
    )


def _phase4_device(config: AppConfig) -> str:
    configured = str(config.data.get("phase4", {}).get("device", config.data.get("phase3", {}).get("device", "cpu")))
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 4 config requested CUDA, but torch.cuda.is_available() is false.")
    return configured


def _provider_note(provider: str) -> str:
    if provider == "qwen":
        return "Qwen provider uses cached Qwen tokens, a denoising-trained adapter, and real SD1.5 inpainting."
    return "Integrated contract validation; not real SD1.5 diffusion adapter inference."


def _max_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _write_summary(path: Path, rows: list[Phase4Record], provider: str, variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latencies = [row.latency_sec for row in rows]
    cuda_values = [row.peak_cuda_memory_bytes for row in rows if row.peak_cuda_memory_bytes is not None]
    preservation = [row.background_preservation_l1 for row in rows]
    changed = [row.mask_changed_pixel_fraction for row in rows]
    visibility = [row.defect_visibility_score for row in rows]
    outside_refined = [row.outside_refined_change_fraction for row in rows]
    quality_scores = [row.generation_quality_score for row in rows]
    critic_rows = [row.critic_guided_generation for row in rows if row.critic_guided_generation.get("enabled")]
    critic_selected_attempts = [int(row.get("selected_attempt_index", 0)) for row in critic_rows]
    critic_accepted = [row for row in critic_rows if bool(row.get("accepted", False))]
    critic_repair_rows = [row for row in critic_rows if row.get("final_coverage_repair_attempts")]
    critic_selected_repair_rows = [row for row in critic_rows if int(row.get("selected_repair_index", -1)) >= 0]
    repair_profile_counts: dict[str, int] = {}
    repair_coverage_gains: list[float] = []
    for row in critic_repair_rows:
        repair_attempts = row.get("final_coverage_repair_attempts", [])
        source_attempts = row.get("attempts", [])
        if repair_attempts:
            profile = str(repair_attempts[0].get("final_coverage_repair_profile", "unknown"))
            repair_profile_counts[profile] = repair_profile_counts.get(profile, 0) + 1
        source_index = int(row.get("selected_attempt_index", -1))
        repair_index = int(row.get("selected_repair_index", -1))
        if 0 <= source_index < len(source_attempts) and 0 <= repair_index < len(repair_attempts):
            before = float(source_attempts[source_index].get("adaptive_mask_coverage_score", 0.0))
            after = float(repair_attempts[repair_index].get("adaptive_mask_coverage_score", before))
            repair_coverage_gains.append(after - before)
    flag_counts: dict[str, int] = {}
    for row in rows:
        for flag in row.quality_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    critic_reasons: dict[str, int] = {}
    for row in critic_rows:
        for reason in row.get("reject_reasons", []):
            critic_reasons[str(reason)] = critic_reasons.get(str(reason), 0) + 1
    lines = [
        f"# Phase 4 Integrated Generation Report: {provider}/{variant}",
        "",
        f"Samples: {len(rows)}",
        f"Variant: {variant}",
        f"Mean latency sec: {mean(latencies):.4f}" if latencies else "Mean latency sec: n/a",
        f"Max latency sec: {max(latencies):.4f}" if latencies else "Max latency sec: n/a",
        f"Peak CUDA memory bytes: {max(cuda_values)}" if cuda_values else "Peak CUDA memory bytes: n/a",
        f"Max RSS KB: {max(row.max_rss_kb for row in rows)}" if rows else "Max RSS KB: n/a",
        f"Mean background preservation L1 outside inpaint mask: {mean(preservation):.6f}" if preservation else "Mean background preservation L1 outside inpaint mask: n/a",
        f"Mean changed-pixel fraction inside inpaint mask: {mean(changed):.4f}" if changed else "Mean changed-pixel fraction inside inpaint mask: n/a",
        f"Mean defect visibility score inside refined mask: {mean(visibility):.6f}" if visibility else "Mean defect visibility score inside refined mask: n/a",
        f"Mean outside-refined changed-pixel fraction: {mean(outside_refined):.4f}" if outside_refined else "Mean outside-refined changed-pixel fraction: n/a",
        f"Mean generation quality score: {mean(quality_scores):.6f}" if quality_scores else "Mean generation quality score: n/a",
        f"Quality profiles: {', '.join(sorted({row.quality_profile for row in rows}))}" if rows else "Quality profiles: n/a",
        f"Quality flag counts: {flag_counts}" if rows else "Quality flag counts: n/a",
        f"Critic-guided regeneration rows: {len(critic_rows)}",
        f"Critic-guided accepted rows: {len(critic_accepted)}" if critic_rows else "Critic-guided accepted rows: n/a",
        f"Mean selected critic attempt index: {mean(critic_selected_attempts):.2f}" if critic_selected_attempts else "Mean selected critic attempt index: n/a",
        f"Critic-guided final reject counts: {critic_reasons}" if critic_rows else "Critic-guided final reject counts: n/a",
        f"Final coverage repair rows: {len(critic_repair_rows)}" if critic_rows else "Final coverage repair rows: n/a",
        f"Final coverage repair selected rows: {len(critic_selected_repair_rows)}" if critic_rows else "Final coverage repair selected rows: n/a",
        f"Final coverage repair profile counts: {repair_profile_counts}" if critic_repair_rows else "Final coverage repair profile counts: n/a",
        f"Mean final repair coverage gain: {mean(repair_coverage_gains):.4f}" if repair_coverage_gains else "Mean final repair coverage gain: n/a",
        "",
        "This report validates the Phase 4 integration contract with cached tokens and a Phase 3 adapter checkpoint.",
    ]
    if rows:
        lines.extend(
            [
                "",
                "| Quality profile | Samples | Mean background L1 | Mean mask changed fraction | Mean visibility | Mean quality score |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for profile in sorted({row.quality_profile for row in rows}):
            profile_rows = [row for row in rows if row.quality_profile == profile]
            lines.append(
                "| "
                f"{profile} | "
                f"{len(profile_rows)} | "
                f"{mean(row.background_preservation_l1 for row in profile_rows):.6f} | "
                f"{mean(row.mask_changed_pixel_fraction for row in profile_rows):.4f} | "
                f"{mean(row.defect_visibility_score for row in profile_rows):.6f} | "
                f"{mean(row.generation_quality_score for row in profile_rows):.6f} |"
            )
        lines.extend(
            [
                "",
                "| Rank | Output | Profile | Seed | Quality score | Flags |",
                "| ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for rank, row in enumerate(sorted(rows, key=lambda item: item.generation_quality_score, reverse=True)[:5], start=1):
            critic = row.critic_guided_generation or {}
            lines.append(
                "| "
                f"{rank} | "
                f"`{Path(row.output_path).name}` | "
                f"{row.quality_profile} | "
                f"{row.generation_seed} | "
                f"{row.generation_quality_score:.6f} | "
                f"{', '.join(row.quality_flags) if row.quality_flags else 'none'}"
                f"{' / critic_attempt=' + str(critic.get('selected_attempt_index')) if critic.get('enabled') else ''} |"
            )
    if provider == "qwen" and variant == "qwen_mask_only":
        lines.append("This is real Qwen-mask-guided SD1.5 inpainting with ordinary CLIP text conditioning only.")
    elif provider == "qwen" and variant == "clone_harmonized":
        lines.append("This is a clone-plus-low-strength-SD harmonization ablation using adaptation defects as source patches.")
    elif provider == "qwen" and variant == "ip_adapter_hybrid":
        lines.append("This is a visual-reference ablation using IP-Adapter when configured, with explicit fallback metadata otherwise.")
    elif provider == "qwen" and variant == "latent_blend_harmonized":
        lines.append("This is a visual-reference ablation that blends aligned defect/source evidence in latent space before SD harmonization.")
    elif provider == "qwen":
        lines.append("This is real Qwen-token-conditioned SD1.5 inpainting smoke inference.")
    else:
        lines.append("It is not real Qwen-guided SD1.5 diffusion inference.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_quality_contact_sheet(path: Path, rows: list[Phase4Record], max_rows: int = 8) -> None:
    if not rows:
        return
    selected = sorted(rows, key=lambda row: row.generation_quality_score, reverse=True)[: max_rows // 2]
    selected += sorted(rows, key=lambda row: row.generation_quality_score)[: max_rows - len(selected)]
    thumb = 128
    columns = 4
    sheet = Image.new("RGB", (columns * thumb, len(selected) * thumb), (245, 245, 245))
    for row_index, row in enumerate(selected):
        background = Image.open(row.background_path).convert("RGB")
        output = Image.open(row.output_path).convert("RGB")
        refined = Image.open(row.refined_mask_path).convert("L")
        diff = _diff_heatmap(background, output)
        overlay = _overlay_mask(output, refined)
        panels = [background, output, overlay, diff]
        for col, panel in enumerate(panels):
            sheet.paste(panel.resize((thumb, thumb), Image.Resampling.BILINEAR), (col * thumb, row_index * thumb))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _overlay_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    active = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    overlay = base.copy()
    overlay[active] = overlay[active] * 0.45 + np.asarray([255.0, 40.0, 40.0]) * 0.55
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).convert("RGB")


def _diff_heatmap(background: Image.Image, output: Image.Image) -> Image.Image:
    source = np.asarray(background.convert("RGB"), dtype=np.float32)
    generated = np.asarray(output.convert("RGB"), dtype=np.float32)
    diff = np.abs(generated - source).mean(axis=2)
    diff = diff / max(1.0, float(np.percentile(diff, 99)))
    heat = np.zeros((*diff.shape, 3), dtype=np.float32)
    heat[..., 0] = np.clip(diff * 255.0, 0, 255)
    heat[..., 1] = np.clip((1.0 - np.abs(diff - 0.5) * 2.0) * 180.0, 0, 180)
    heat[..., 2] = np.clip((1.0 - diff) * 120.0, 0, 120)
    return Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8)).convert("RGB")
