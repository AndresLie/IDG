from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from iadgen_v2.config import load_config
from iadgen_v2.dataset import prepare_splits
from iadgen_v2.phase3 import (
    GatedProjectionAdapter,
    PHASE3_SCHEMA_VERSION,
    Phase3CacheRecord,
    _load_inpaint_training_tensors,
    _load_uncertainty_loss_weight,
    _masked_context,
    _phase3_fingerprint,
    _weighted_denoising_mse,
    build_phase3_adaptation_cache,
    train_phase3_adapter,
    validate_phase3_adapter,
)


TARGETS = {"metal_nut": "scratch", "tile": "crack", "wood": "scratch"}


def test_gated_projection_adapter_appends_tokens_and_respects_gate() -> None:
    torch.manual_seed(7)
    tokens = torch.randn(2, 4, 6)
    clip = torch.randn(2, 77, 768)
    adapter = GatedProjectionAdapter(input_dim=6, output_dim=768, hidden_dim=12, initial_gate_logit=-100.0)

    combined = adapter(clip, tokens)

    assert tuple(combined.shape) == (2, 81, 768)
    assert torch.allclose(combined[:, :77], clip)
    assert combined[:, 77:].abs().max() < 1e-30

    adapter.gate_logit.data.fill_(100.0)
    open_gate = adapter(clip, tokens)
    assert open_gate[:, 77:].abs().mean() > 0.01


def test_phase3_cache_train_and_validate_surrogate_adapter(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    metadata_path = build_phase3_adaptation_cache(config, "heuristic")
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 6
    assert all(row["settings"]["target_split"] == "adaptation-only" for row in records)
    first_cache = torch.load(records[0]["feature_cache_path"], map_location="cpu", weights_only=False)
    assert tuple(first_cache["tokens"].shape) == (1, 4, 10)
    assert tuple(first_cache["surrogate_target_tokens"].shape) == (1, 4, 768)
    assert torch.isfinite(first_cache["tokens"]).all()
    assert torch.isfinite(first_cache["surrogate_target_tokens"]).all()

    checkpoint_path = train_phase3_adapter(config, "heuristic")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["adapter_config"]["output_dim"] == 768
    assert checkpoint["training"]["objective"] == "surrogate adapter-shape loss, not diffusion denoising loss"

    validation_path = validate_phase3_adapter(config, "heuristic")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["input_tokens_shape"] == [1, 4, 10]
    assert validation["clip_embeddings_shape"] == [1, 77, 768]
    assert validation["combined_conditioning_shape"] == [1, 81, 768]
    assert "not diffusion denoising" in validation["objective_note"]


def test_phase3_validation_rejects_stale_checkpoint(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    build_phase3_adaptation_cache(config, "heuristic")
    train_phase3_adapter(config, "heuristic")

    config.data["phase3"]["adapter_hidden_dim"] = 20
    with pytest.raises(ValueError, match="does not match"):
        validate_phase3_adapter(config, "heuristic")


def test_phase3_qwen_validation_accepts_token_only_cache(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)
    config.data["phase3"]["provider"] = "qwen"
    config.data["phase3"]["adapter_hidden_dim"] = 8
    provider = "qwen"
    phase3_fingerprint = _phase3_fingerprint(config, provider)
    output_dir = config.output_dir / "phase3" / provider
    cache_path = output_dir / "cache" / "metal_nut" / "metal_nut_scratch_0000.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save({"tokens": torch.randn(1, 4, 6), "metadata": {"provider": "qwen"}}, cache_path)
    metadata_path = output_dir / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "category": "metal_nut",
                "defect_type": "scratch",
                "provider": provider,
                "source_image_path": "unused.png",
                "source_mask_path": "unused_mask.png",
                "masked_context_path": "unused_context.png",
                "anomaly_bbox_xyxy": [1, 2, 8, 9],
                "normalized_bbox_xyxy": [0.1, 0.2, 0.8, 0.9],
                "feature_cache_path": str(cache_path),
                "prompt": "scratch",
                "settings": {
                    "phase3_fingerprint": phase3_fingerprint,
                    "phase3_schema_version": PHASE3_SCHEMA_VERSION,
                    "split_spec_fingerprint": config.split_spec_fingerprint(),
                    "target_split": "adaptation-only",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = GatedProjectionAdapter(input_dim=6, output_dim=768, hidden_dim=8)
    checkpoint_path = output_dir / "checkpoints" / "adapter_denoising.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "adapter_config": {"input_dim": 6, "output_dim": 768, "hidden_dim": 8, "clip_token_count": 77},
            "training": {
                "gate_value": adapter.gate_value,
                "objective": "SD1.5 inpainting diffusion denoising loss with frozen backbones",
            },
            "phase3_fingerprint": phase3_fingerprint,
            "phase3_schema_version": PHASE3_SCHEMA_VERSION,
        },
        checkpoint_path,
    )

    validation_path = validate_phase3_adapter(config, provider)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    assert validation["input_tokens_shape"] == [1, 4, 6]
    assert validation["combined_conditioning_shape"] == [1, 81, 768]
    assert "denoising loss" in validation["objective_note"]


def test_phase3_qwen_provider_fails_until_real_extractor_exists(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    prepare_splits(config, download=False)

    with pytest.raises(RuntimeError, match="Qwen provider is not ready"):
        build_phase3_adaptation_cache(config, "qwen")


def test_phase3_soft_mask_context_and_training_tensor_preserve_alpha(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "soft_mask.png"
    image = Image.new("RGB", (8, 8), (100, 100, 100))
    ImageDraw.Draw(image).rectangle((3, 3, 4, 4), fill=(220, 40, 40))
    image.save(image_path)
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((3, 3), 128)
    mask.putpixel((4, 4), 255)
    mask.save(mask_path)

    context = _masked_context(image, mask)
    context_pixel = context.getpixel((3, 3))
    assert context_pixel != image.getpixel((3, 3))
    assert context_pixel != context.getpixel((4, 4))

    row = Phase3CacheRecord(
        category="wood",
        defect_type="scratch",
        provider="heuristic",
        source_image_path=str(image_path),
        source_mask_path=str(mask_path),
        masked_context_path="unused.png",
        anomaly_bbox_xyxy=(3, 3, 5, 5),
        normalized_bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        feature_cache_path="unused.pt",
        prompt="scratch",
        settings={"mask_weight_mode": "soft_alpha"},
    )

    _, mask_tensor, _ = _load_inpaint_training_tensors(row, 8, "cpu", torch.float32)

    assert float(mask_tensor[0, 0, 3, 3]) == pytest.approx(128 / 255, abs=1e-4)
    assert float(mask_tensor[0, 0, 4, 4]) == pytest.approx(1.0, abs=1e-4)


def test_phase3_uncertainty_weight_downweights_latent_denoising_loss(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    uncertainty_path = tmp_path / "uncertainty.png"
    Image.new("RGB", (8, 8), (100, 100, 100)).save(image_path)
    Image.new("L", (8, 8), 255).save(mask_path)
    uncertainty = Image.new("L", (8, 8), 0)
    ImageDraw.Draw(uncertainty).rectangle((0, 0, 3, 7), fill=255)
    uncertainty.save(uncertainty_path)
    row = Phase3CacheRecord(
        category="wood",
        defect_type="scratch",
        provider="qwen",
        source_image_path=str(image_path),
        source_mask_path=str(mask_path),
        uncertainty_mask_path=str(uncertainty_path),
        masked_context_path="unused.png",
        anomaly_bbox_xyxy=(0, 0, 8, 8),
        normalized_bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        feature_cache_path="unused.pt",
        prompt="scratch",
        settings={},
    )

    weight = _load_uncertainty_loss_weight(row, 8, "cpu", torch.float32, {"uncertainty_loss_weight": 0.25})

    assert float(weight[0, 0, 0, 0]) == pytest.approx(0.25, abs=1e-4)
    assert float(weight[0, 0, 7, 7]) == pytest.approx(1.0, abs=1e-4)

    prediction = torch.zeros((1, 4, 2, 2))
    target = torch.zeros((1, 4, 2, 2))
    prediction[:, :, :, 0] = 4.0
    weighted = _weighted_denoising_mse(prediction, target, weight)
    unweighted = _weighted_denoising_mse(prediction, target, torch.ones_like(weight))

    assert weighted < unweighted


def _fixture_config(tmp_path: Path):
    dataset_root = tmp_path / "data" / "mvtec_ad"
    for category, defect_type in TARGETS.items():
        clean_dir = dataset_root / category / "train" / "good"
        test_dir = dataset_root / category / "test" / defect_type
        mask_dir = dataset_root / category / "ground_truth" / defect_type
        clean_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        clean = Image.new("RGB", (48, 48), (15, 15, 15) if category == "metal_nut" else (80, 80, 80))
        if category == "metal_nut":
            ImageDraw.Draw(clean).ellipse((8, 8, 40, 40), fill=(150, 150, 150))
        clean.save(clean_dir / "000.png")
        for index in range(4):
            image = clean.copy()
            mask = Image.new("L", (48, 48), 0)
            draw = ImageDraw.Draw(mask)
            draw.line((8, 18 + index, 38, 20 + index), fill=255, width=2)
            image.paste(Image.new("RGB", image.size, (180, 40, 40)), mask=mask)
            image.save(test_dir / f"{index:03d}.png")
            mask.save(mask_dir / f"{index:03d}_mask.png")
    config_path = tmp_path / "phase3.yaml"
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
                "    metal_nut: [scratch]",
                "    tile: [crack]",
                "    wood: [scratch]",
                "generation:",
                "  samples_per_category: 1",
                "  seed: 11",
                "  device: cpu",
                "  prompt_by_defect: {scratch: scratch, crack: crack}",
                "models:",
                "  mock: {enabled: true}",
                "  sd15:",
                "    enabled: false",
                "    base_model: example/base",
                "    controlnet_model: example/control",
                "evaluation: {}",
                "phase2:",
                "  provider: heuristic",
                "  qwen_model: Qwen/Qwen2.5-VL-3B-Instruct",
                "  samples_per_category: 1",
                "  token_count: 4",
                "  feature_dim: 10",
                "  box_area_fraction: {scratch: 0.06, crack: 0.06}",
                "  min_surface_coverage: 0.90",
                "phase3:",
                "  provider: heuristic",
                "  device: cpu",
                "  clip_token_count: 77",
                "  cross_attention_dim: 768",
                "  adapter_hidden_dim: 16",
                "  adapter_epochs: 2",
                "  learning_rate: 0.001",
                "  weight_decay: 0.0",
                "  initial_gate_logit: -2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_config(config_path)
