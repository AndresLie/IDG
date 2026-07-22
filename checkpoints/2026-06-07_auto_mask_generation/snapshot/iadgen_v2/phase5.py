from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from iadgen_v2.config import AppConfig, fingerprint
from iadgen_v2.dataset import load_manifest
from iadgen_v2.phase4 import PHASE4_SCHEMA_VERSION, _phase4_fingerprint
from iadgen_v2.phase4 import PHASE4_QUALITY_PROFILES
from iadgen_v2.records import write_json
from iadgen_v2.segmentation import segmentation_metrics


PHASE5_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class SegmentationSample:
    image_path: str
    mask_path: str
    category: str
    defect_type: str
    source: str
    mask_mode: str = "binary"
    morphology: str = "unknown"
    uncertainty_mask_path: str | None = None
    positive_core_path: str | None = None
    possible_region_path: str | None = None


def run_phase5_evaluation(config: AppConfig, provider: str | None = None) -> Path:
    phase5 = _phase5_config(config)
    provider = provider or str(phase5.get("provider", "heuristic"))
    if provider not in {"heuristic", "qwen"}:
        raise ValueError(f"Unsupported Phase 5 provider: {provider}")
    seed = int(phase5.get("seed", config.data.get("generation", {}).get("seed", 1337)))
    _seed_everything(seed)
    manifest = load_manifest(config)
    real_train, held_out = _real_split_samples(manifest, int(phase5.get("held_out_per_category", 12)))
    clean_negatives = _clean_inpaint_negative_samples(config, manifest, phase5, provider, seed)
    normal_train = _normal_split_samples(manifest)
    supervised_normals = _supervised_normal_negative_samples(normal_train, phase5, seed)
    if not real_train:
        raise ValueError("Phase 5 needs at least one real adaptation sample")
    if not held_out:
        raise ValueError("Phase 5 needs at least one held-out real anomaly sample")
    evaluators = _phase5_evaluators(phase5)
    ratios = [float(value) for value in phase5.get("synthetic_ratios", [0.0, 0.5, 0.8])]
    run_seeds = [int(value) for value in phase5.get("repeated_seeds", [seed])]
    if not run_seeds:
        run_seeds = [seed]
    variants = _phase5_variants(config, provider)
    variant_specs = _phase5_variant_specs(config, provider, variants)
    rows: list[dict[str, Any]] = []
    synthetic_counts: dict[str, int] = {}
    selector_logs: list[dict[str, Any]] = []
    report_dir = config.report_dir / "phase5" / provider
    report_dir.mkdir(parents=True, exist_ok=True)
    audit_manifest = _write_human_audit_manifest(report_dir, manifest, phase5, provider)
    mask_policy_report = _write_mask_policy_validation_report(report_dir, manifest, provider)
    if "tiny_unet" in evaluators:
        if variant_specs:
            for run_seed in run_seeds:
                metrics = _train_and_evaluate(
                    config,
                    real_train + clean_negatives + supervised_normals,
                    held_out,
                    0.0,
                    run_seed,
                    _prediction_dir(report_dir, "tiny_unet", "real_only", "none", 0.0, run_seed),
                )
                metrics["evaluator"] = "tiny_unet"
                metrics["variant"] = "real_only"
                metrics["quality_profile"] = "none"
                metrics["run_seed"] = run_seed
                rows.append(metrics)
            for variant_index, spec in enumerate(variant_specs):
                synthetic = _phase4_synthetic_samples(
                    config,
                    provider,
                    spec["variant"],
                    spec["quality_profile"],
                    phase5,
                    selector_logs=selector_logs,
                )
                synthetic_counts[spec["label"]] = len(synthetic)
                if not synthetic:
                    continue
                for ratio_index, ratio in enumerate(ratio for ratio in ratios if ratio > 0.0):
                    train_samples = _mix_samples(real_train, synthetic, ratio, seed, bool(phase5.get("class_balanced_synthetic", False)))
                    train_samples.extend(clean_negatives)
                    train_samples.extend(supervised_normals)
                    for run_seed in run_seeds:
                        metrics = _train_and_evaluate(
                            config,
                            train_samples,
                            held_out,
                            ratio,
                            run_seed + (variant_index + 1) * 100_003 + ratio_index * 9973,
                            _prediction_dir(report_dir, "tiny_unet", spec["label"], spec["quality_profile"] or "all", ratio, run_seed),
                        )
                        metrics["evaluator"] = "tiny_unet"
                        metrics["variant"] = spec["label"]
                        metrics["quality_profile"] = spec["quality_profile"] or "all"
                        metrics["run_seed"] = run_seed
                        rows.append(metrics)
        else:
            synthetic = _phase4_synthetic_samples(config, provider, phase5=phase5, selector_logs=selector_logs)
            synthetic_counts[provider] = len(synthetic)
            for index, ratio in enumerate(ratios):
                train_samples = _mix_samples(real_train, synthetic, ratio, seed, bool(phase5.get("class_balanced_synthetic", False)))
                train_samples.extend(clean_negatives)
                train_samples.extend(supervised_normals)
                for run_seed in run_seeds:
                    metrics = _train_and_evaluate(
                        config,
                        train_samples,
                        held_out,
                        ratio,
                        run_seed + index * 9973,
                        _prediction_dir(report_dir, "tiny_unet", provider, "all", ratio, run_seed),
                    )
                    metrics["evaluator"] = "tiny_unet"
                    metrics["variant"] = provider
                    metrics["quality_profile"] = "all"
                    metrics["run_seed"] = run_seed
                    rows.append(metrics)
    if "patchcore_lite" in evaluators:
        for run_seed in run_seeds:
            metrics = _patchcore_lite_evaluate(
                config,
                normal_train,
                real_train,
                held_out,
                run_seed,
                _prediction_dir(report_dir, "patchcore_lite", "normal_only", "none", 0.0, run_seed),
            )
            metrics["evaluator"] = "patchcore_lite"
            metrics["variant"] = "normal_only"
            metrics["quality_profile"] = "none"
            metrics["run_seed"] = run_seed
            rows.append(metrics)
    if "patchcore_resnet" in evaluators:
        for run_seed in run_seeds:
            metrics = _patchcore_resnet_evaluate(
                config,
                normal_train,
                real_train,
                held_out,
                run_seed,
                _prediction_dir(report_dir, "patchcore_resnet", "normal_only", "none", 0.0, run_seed),
            )
            metrics["evaluator"] = "patchcore_resnet"
            metrics["variant"] = "normal_only"
            metrics["quality_profile"] = "none"
            metrics["run_seed"] = run_seed
            rows.append(metrics)
    csv_path = report_dir / "segmentation_results.csv"
    selector_report = _write_synthetic_selector_reports(report_dir, selector_logs, phase5)
    _write_csv(csv_path, rows)
    _write_summary(
        report_dir / "summary.md",
        rows,
        len(real_train),
        len(normal_train),
        len(clean_negatives),
        len(supervised_normals),
        synthetic_counts,
        len(held_out),
        provider,
        audit_manifest,
        evaluators,
        mask_policy_report,
        selector_report,
    )
    write_json(
        config.output_dir / "phase5" / provider / "run_status.json",
        {
            "provider": provider,
            "evaluators": evaluators,
            "variants": variants or [provider],
            "phase5_fingerprint": _phase5_fingerprint(config, provider),
            "schema_version": PHASE5_SCHEMA_VERSION,
            "real_adaptation_samples": len(real_train),
            "normal_memory_samples": len(normal_train),
            "clean_inpaint_negative_samples": len(clean_negatives),
            "supervised_normal_negative_samples": len(supervised_normals),
            "synthetic_samples": synthetic_counts,
            "held_out_samples": len(held_out),
            "held_out_by_category": _count_by_category(held_out),
            "human_audit_manifest": str(audit_manifest),
            "mask_policy_validation_report": str(mask_policy_report),
            "synthetic_selector_report": str(selector_report),
            "ratios": ratios,
            "run_seeds": run_seeds,
            "note": _provider_note(provider),
        },
    )
    return csv_path


def _real_split_samples(manifest: dict[str, Any], held_out_per_category: int) -> tuple[list[SegmentationSample], list[SegmentationSample]]:
    train: list[SegmentationSample] = []
    held_out: list[SegmentationSample] = []
    for target in manifest["targets"].values():
        category = str(target["category"])
        defect_type = str(target["defect_type"])
        for row in target["adaptation"]:
            label_policy = row.get("label_policy") if isinstance(row, dict) else None
            policy_name = str(label_policy.get("label_policy", "hard_mask_ok")) if isinstance(label_policy, dict) else "hard_mask_ok"
            morphology = str(label_policy.get("quality_morphology", "unknown")) if isinstance(label_policy, dict) else "unknown"
            train.append(
                SegmentationSample(
                    image_path=str(row["image_path"]),
                    mask_path=str(row.get("training_mask_path") or row["mask_path"]),
                    category=category,
                    defect_type=defect_type,
                    source="real_adaptation",
                    mask_mode="soft" if policy_name == "soft_mask_only" else "binary",
                    morphology=morphology,
                    uncertainty_mask_path=str(row.get("uncertainty_mask_path") or "") or None,
                    positive_core_path=str(row.get("positive_core_path") or "") or None,
                    possible_region_path=str(row.get("possible_region_path") or "") or None,
                )
            )
        for row in list(target["held_out"])[:held_out_per_category]:
            label_policy = row.get("label_policy") if isinstance(row, dict) else None
            morphology = str(label_policy.get("quality_morphology", "unknown")) if isinstance(label_policy, dict) else "unknown"
            held_out.append(
                SegmentationSample(
                    image_path=str(row["image_path"]),
                    mask_path=str(row.get("eval_mask_path") or row["mask_path"]),
                    category=category,
                    defect_type=defect_type,
                    source="real_held_out",
                    morphology=morphology,
                    uncertainty_mask_path=str(row.get("uncertainty_mask_path") or "") or None,
                    positive_core_path=str(row.get("positive_core_path") or "") or None,
                    possible_region_path=str(row.get("possible_region_path") or "") or None,
                )
            )
    return train, held_out


def _normal_split_samples(manifest: dict[str, Any]) -> list[SegmentationSample]:
    samples: list[SegmentationSample] = []
    for target in manifest["targets"].values():
        category = str(target["category"])
        defect_type = str(target["defect_type"])
        for image_path in target.get("clean_targets", []):
            samples.append(
                SegmentationSample(
                    image_path=str(image_path),
                    mask_path="",
                    category=category,
                    defect_type=defect_type,
                    source="normal_train_good",
                    morphology="normal",
                )
            )
    return samples


def _supervised_normal_negative_samples(
    normal_samples: list[SegmentationSample],
    phase5: dict[str, Any],
    seed: int,
) -> list[SegmentationSample]:
    settings = phase5.get("supervised_normal_negatives", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return []
    max_samples = int(settings.get("max_samples", len(normal_samples)))
    if max_samples <= 0:
        return []
    rng = random.Random(seed + 91_337)
    shuffled = list(normal_samples)
    rng.shuffle(shuffled)
    return [
        SegmentationSample(
            image_path=sample.image_path,
            mask_path="",
            category=sample.category,
            defect_type=sample.defect_type,
            source="supervised_normal_negative",
            morphology="normal",
        )
        for sample in shuffled[:max_samples]
    ]


def _phase5_evaluators(phase5: dict[str, Any]) -> list[str]:
    raw = phase5.get("evaluators", ["tiny_unet"])
    evaluators = [str(value) for value in raw] if isinstance(raw, list) else [str(raw)]
    aliases = {"patchcore": "patchcore_resnet", "patchcore_full": "patchcore_resnet"}
    normalized = [aliases.get(value, value) for value in evaluators]
    allowed = {"tiny_unet", "patchcore_lite", "patchcore_resnet"}
    unknown = [value for value in normalized if value not in allowed]
    if unknown:
        raise ValueError(f"Unsupported Phase 5 evaluator(s): {', '.join(unknown)}")
    return list(dict.fromkeys(normalized))


def _write_human_audit_manifest(
    report_dir: Path,
    manifest: dict[str, Any],
    phase5: dict[str, Any],
    provider: str,
) -> Path:
    settings = phase5.get("human_audit", {})
    enabled = bool(settings.get("enabled", True)) if isinstance(settings, dict) else bool(settings)
    audit_count = int(settings.get("max_samples", 24)) if isinstance(settings, dict) else 24
    path = report_dir / "human_audit_manifest.jsonl"
    if not enabled:
        path.write_text("", encoding="utf-8")
        return path
    rows: list[dict[str, Any]] = []
    for target_key, target in manifest["targets"].items():
        for split_name in ("adaptation", "held_out"):
            for row in target.get(split_name, []):
                label_policy = row.get("label_policy") if isinstance(row, dict) else None
                rows.append(
                    {
                        "provider": provider,
                        "target": target_key,
                        "split": split_name,
                        "image_path": row.get("image_path"),
                        "compatibility_mask_path": row.get("mask_path"),
                        "training_mask_path": row.get("training_mask_path"),
                        "eval_mask_path": row.get("eval_mask_path"),
                        "uncertainty_mask_path": row.get("uncertainty_mask_path"),
                        "positive_core_path": row.get("positive_core_path"),
                        "possible_region_path": row.get("possible_region_path"),
                        "label_policy": label_policy or {"label_policy": "unknown"},
                        "recommended_action": _audit_recommendation(label_policy),
                    }
                )
    priority = {"soft_mask_only": 0, "unknown": 1, "hard_mask_ok": 2}
    rows.sort(key=lambda item: priority.get(str(item["label_policy"].get("label_policy", "unknown")), 1))
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows[:audit_count]) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    summary_path = report_dir / "human_audit_manifest.md"
    summary_path.write_text(
        "\n".join(
            [
                f"# Human-Light Audit Manifest: {provider}",
                "",
                "This file is for final research validation only. The normal user workflow remains automatic.",
                "",
                f"JSONL: `{path.name}`",
                f"Samples listed: `{min(len(rows), audit_count)}` of `{len(rows)}`",
                "",
                "Priority is given to `soft_mask_only` cases because broad scuffs have fuzzy boundaries and should be human-checked before final claims.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _audit_recommendation(label_policy: object) -> str:
    if isinstance(label_policy, dict) and label_policy.get("label_policy") == "soft_mask_only":
        return "inspect soft support; do not force a crisp boundary unless final human labels are required"
    if isinstance(label_policy, dict) and label_policy.get("label_policy") == "hard_mask_ok":
        return "spot-check hard pseudo-label for line alignment and leakage"
    return "inspect pseudo-label source and choose soft or hard policy before final claims"


def _write_mask_policy_validation_report(report_dir: Path, manifest: dict[str, Any], provider: str) -> Path:
    rows: list[dict[str, Any]] = []
    for target_key, target in manifest.get("targets", {}).items():
        if not isinstance(target, dict):
            continue
        for split_name in ("adaptation", "held_out"):
            for row in target.get(split_name, []):
                if not isinstance(row, dict):
                    continue
                label_policy = row.get("label_policy") if isinstance(row.get("label_policy"), dict) else {}
                policy = str(label_policy.get("label_policy", "hard_mask_ok"))
                morphology = str(label_policy.get("quality_morphology", "unknown"))
                training_path = str(row.get("training_mask_path") or row.get("mask_path") or "")
                eval_path = str(row.get("eval_mask_path") or row.get("mask_path") or "")
                expected_training = _expected_training_mask_role(policy, morphology)
                observed_training = _mask_role(training_path)
                observed_eval = _mask_role(eval_path)
                status = _mask_policy_status(split_name, expected_training, observed_training, observed_eval)
                rows.append(
                    {
                        "provider": provider,
                        "target": str(target_key),
                        "split": split_name,
                        "image": Path(str(row.get("image_path", ""))).name,
                        "policy": policy,
                        "morphology": morphology,
                        "expected_training_role": expected_training,
                        "observed_training_role": observed_training,
                        "observed_eval_role": observed_eval,
                        "training_mask_path": training_path,
                        "eval_mask_path": eval_path,
                        "status": status,
                    }
                )
    csv_path = report_dir / "mask_policy_validation.csv"
    fields = [
        "provider",
        "target",
        "split",
        "image",
        "policy",
        "morphology",
        "expected_training_role",
        "observed_training_role",
        "observed_eval_role",
        "status",
        "training_mask_path",
        "eval_mask_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    by_policy: dict[tuple[str, str, str], int] = {}
    for row in rows:
        key = (row["policy"], row["morphology"], row["observed_training_role"])
        by_policy[key] = by_policy.get(key, 0) + 1
    md_path = report_dir / "mask_policy_validation.md"
    lines = [
        "# Mask Policy Validation",
        "",
        "This report verifies that Phase 5 consumes the intended automatic pseudo-label role.",
        "",
        f"CSV: `{csv_path.name}`",
        f"Rows: `{len(rows)}`",
        "",
        "## Status Counts",
        "",
    ]
    if status_counts:
        lines.extend(f"- `{status}`: `{count}`" for status, count in sorted(status_counts.items()))
    else:
        lines.append("- none")
    lines.extend(["", "## Training Mask Roles By Policy", ""])
    if by_policy:
        for (policy, morphology, role), count in sorted(by_policy.items()):
            lines.append(f"- `{policy}` / `{morphology}` -> `{role}`: `{count}`")
    else:
        lines.append("- none")
    failures = [row for row in rows if row["status"] != "pass"]
    if failures:
        lines.extend(["", "## Rows To Inspect", "", "| Split | Image | Policy | Morphology | Expected | Observed |", "| --- | --- | --- | --- | --- | --- |"])
        for row in failures[:30]:
            lines.append(
                f"| `{row['split']}` | `{row['image']}` | `{row['policy']}` | `{row['morphology']}` | "
                f"`{row['expected_training_role']}` | `{row['observed_training_role']}` |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _expected_training_mask_role(policy: str, morphology: str) -> str:
    if policy == "soft_mask_only" or morphology == "multi_scuff":
        return "training_soft"
    if policy == "hard_mask_ok" or morphology in {"scratch_band", "single_stroke", "crack_band"}:
        return "training_medium"
    return "mask"


def _mask_role(path: str) -> str:
    name = Path(path).name.lower()
    for role in ("training_soft", "training_medium", "eval_tight", "inpaint_soft", "training_wide", "positive_core", "possible_region"):
        if role in name:
            return role
    if name:
        return "mask"
    return "missing"


def _mask_policy_status(split_name: str, expected_training: str, observed_training: str, observed_eval: str) -> str:
    if split_name == "adaptation" and observed_training != expected_training:
        return "inspect_training_role"
    if split_name == "held_out" and observed_eval not in {"eval_tight", "mask"}:
        return "inspect_eval_role"
    return "pass"


def _prediction_dir(
    report_dir: Path,
    evaluator: str,
    variant: str,
    quality_profile: str,
    synthetic_ratio: float,
    run_seed: int,
) -> Path:
    safe_variant = _safe_name(variant)
    safe_profile = _safe_name(quality_profile)
    return report_dir / "prediction_examples" / evaluator / safe_variant / safe_profile / f"r{synthetic_ratio:.2f}_s{run_seed}"


def _safe_name(value: object) -> str:
    text = str(value).strip().replace("/", "_").replace(":", "_")
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text) or "unknown"


def _clean_inpaint_negative_samples(
    config: AppConfig,
    manifest: dict[str, Any],
    phase5: dict[str, Any],
    provider: str,
    seed: int,
) -> list[SegmentationSample]:
    settings = phase5.get("sd_clean_inpaint_negatives", {})
    if not isinstance(settings, dict) or not bool(settings.get("enabled", False)):
        return []
    per_clean = int(settings.get("samples_per_clean", 1))
    if per_clean <= 0:
        return []
    output_dir = config.output_dir / "phase5" / provider / "clean_inpaint_negatives"
    samples: list[SegmentationSample] = []
    rng = random.Random(seed + 77_001)
    image_size = int(phase5.get("image_size", 96))
    for target_key, target in manifest["targets"].items():
        category = str(target["category"])
        defect_type = str(target["defect_type"])
        clean_targets = [Path(path) for path in target.get("clean_targets", [])]
        for clean_index, clean_path in enumerate(clean_targets):
            for sample_index in range(per_clean):
                image = Image.open(clean_path).convert("RGB")
                mask = Image.new("L", image.size, 0)
                patched = _synthetic_clean_inpaint_signature(image, rng)
                stem = f"{target_key.replace('/', '_')}_{clean_index:04d}_{sample_index:02d}"
                image_path = output_dir / category / f"{stem}.png"
                mask_path = output_dir / category / f"{stem}_mask.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                patched.resize((image_size, image_size), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR).save(image_path)
                mask.save(mask_path)
                samples.append(
                    SegmentationSample(
                        image_path=str(image_path),
                        mask_path=str(mask_path),
                        category=category,
                        defect_type=defect_type,
                        source="sd_clean_inpaint_negative",
                        morphology="normal",
                    )
                )
    return samples


def _synthetic_clean_inpaint_signature(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    box_w = max(8, int(width * rng.uniform(0.12, 0.28)))
    box_h = max(8, int(height * rng.uniform(0.12, 0.28)))
    left = rng.randint(0, max(0, width - box_w))
    top = rng.randint(0, max(0, height - box_h))
    box = (left, top, left + box_w, top + box_h)
    patch = image.crop(box).filter(ImageFilter.GaussianBlur(radius=1.2))
    soft = Image.new("L", image.size, 0)
    draw = Image.new("L", (box_w, box_h), 255).filter(ImageFilter.GaussianBlur(radius=max(1.0, min(box_w, box_h) / 10.0)))
    soft.paste(draw, (left, top))
    canvas = image.copy()
    canvas.paste(patch, box)
    return Image.composite(canvas, image, soft).convert("RGB")


def _phase4_synthetic_samples(
    config: AppConfig,
    provider: str,
    variant: str | None = None,
    quality_profile: str | None = None,
    phase5: dict[str, Any] | None = None,
    selector_logs: list[dict[str, Any]] | None = None,
) -> list[SegmentationSample]:
    path = _phase4_metadata_path(config, provider, variant)
    if not path.exists():
        raise FileNotFoundError(f"Missing Phase 4 metadata: {path}")
    expected = _phase4_fingerprint(config, provider)
    tfidg_metrics = _phase11_tfidg_metric_index(config, provider, phase5 or {})
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if quality_profile is not None and str(row.get("quality_profile", "")) != quality_profile:
            continue
        settings = row.get("settings", {})
        if settings.get("phase4_schema_version") != PHASE4_SCHEMA_VERSION:
            raise ValueError("Phase 4 metadata schema is stale; regenerate Phase 4 outputs first.")
        if settings.get("phase4_fingerprint") != expected:
            raise ValueError("Phase 4 metadata does not match the active configuration; regenerate Phase 4 outputs first.")
        if not Path(row["output_path"]).exists() or not Path(row["refined_mask_path"]).exists():
            raise FileNotFoundError(f"Missing Phase 4 output or mask for {row.get('output_path')}")
        metrics = tfidg_metrics.get(str(row["output_path"]))
        if metrics is not None:
            row = {**row, "tfidg_lite": metrics}
        decision = _synthetic_selector_decision(row, phase5 or {})
        candidates.append((row, decision))
    selected_keys = _selected_candidate_keys(candidates, phase5 or {})
    samples: list[SegmentationSample] = []
    for row, decision in candidates:
        candidate_key = _phase4_candidate_key(row)
        accepted = candidate_key in selected_keys
        reasons = list(decision["reject_reasons"])
        if decision["accepted"] and not accepted:
            reasons.append("selector_group_limit")
        settings = row.get("settings", {})
        log_row = {
            "accepted": accepted,
            "reject_reasons": reasons,
            "selector_score": float(decision["selector_score"]),
            "variant": str(row.get("variant", variant or "phase4_synthetic")),
            "quality_profile": str(row.get("quality_profile", settings.get("quality_profile", quality_profile or "unknown"))),
            "category": str(row["category"]),
            "defect_type": str(row["defect_type"]),
            "morphology": str(settings.get("quality_morphology", row.get("quality_profile", "unknown"))) if isinstance(settings, dict) else "unknown",
            "output_path": str(row["output_path"]),
            "mask_path": str(row["refined_mask_path"]),
            "generation_quality_score": float(row.get("generation_quality_score", 0.0)),
            "defect_visibility_score": float(row.get("defect_visibility_score", 0.0)),
            "background_preservation_l1": float(row.get("background_preservation_l1", 0.0)),
            "outside_refined_change_fraction": float(row.get("outside_refined_change_fraction", 0.0)),
            "inpaint_mask_area_fraction": float(row.get("inpaint_mask_area_fraction", 0.0)),
            "quality_flags": list(row.get("quality_flags", [])),
            "tfidg_lite_score": float(decision.get("tfidg_lite_score", 0.0)),
            "tfidg_adaptive_mask_coverage_score": float(decision.get("tfidg_adaptive_mask_coverage_score", 0.0)),
            "tfidg_feature_alignment_score": float(decision.get("tfidg_feature_alignment_score", 0.0)),
            "tfidg_texture_preservation_score": float(decision.get("tfidg_texture_preservation_score", 0.0)),
            "tfidg_leakage_score": float(decision.get("tfidg_leakage_score", 0.0)),
        }
        if selector_logs is not None:
            selector_logs.append(log_row)
        if not accepted:
            continue
        settings = row.get("settings", {})
        samples.append(
            SegmentationSample(
                image_path=str(row["output_path"]),
                mask_path=str(row["refined_mask_path"]),
                category=str(row["category"]),
                defect_type=str(row["defect_type"]),
                source=str(row.get("variant", variant or "phase4_synthetic")),
                morphology=str(settings.get("quality_morphology", row.get("quality_profile", "unknown"))) if isinstance(settings, dict) else "unknown",
            )
        )
    return samples


def _passes_phase5_quality_filter(row: dict[str, Any], phase5: dict[str, Any]) -> bool:
    settings = phase5.get("quality_filtered_training", {})
    enabled = bool(settings.get("enabled", False)) if isinstance(settings, dict) else bool(settings)
    if not enabled:
        return True
    flags = row.get("quality_flags", [])
    if flags:
        return False
    min_visibility = float(settings.get("min_defect_visibility_score", phase5.get("min_defect_visibility_score", 0.015))) if isinstance(settings, dict) else 0.015
    max_background = float(settings.get("max_background_l1", phase5.get("max_background_l1", 0.05))) if isinstance(settings, dict) else 0.05
    max_outside = (
        float(settings.get("max_outside_refined_change_fraction", phase5.get("max_outside_refined_change_fraction", 0.25)))
        if isinstance(settings, dict)
        else 0.25
    )
    return (
        float(row.get("defect_visibility_score", 1.0)) >= min_visibility
        and float(row.get("background_preservation_l1", 0.0)) <= max_background
        and float(row.get("outside_refined_change_fraction", 0.0)) <= max_outside
    )


def _phase11_tfidg_metric_index(config: AppConfig, provider: str, phase5: dict[str, Any]) -> dict[str, dict[str, float]]:
    selector = _synthetic_selector_settings(phase5)
    tfidg_settings = selector.get("tfidg_lite", {}) if isinstance(selector.get("tfidg_lite"), dict) else {}
    if not bool(tfidg_settings.get("enabled", False)):
        return {}
    configured_path = tfidg_settings.get("metrics_path")
    metrics_path = config.resolve_path(configured_path) if configured_path else config.report_dir / "phase11_tfidg_critic" / provider / "tfidg_lite_metrics.csv"
    if not metrics_path.exists():
        if bool(tfidg_settings.get("require_metrics_file", True)):
            raise FileNotFoundError(
                f"Phase 5 synthetic selector requires TF-IDG-lite metrics but they are missing: {metrics_path}. "
                "Run phase11-tfidg-critic before phase5-evaluate."
            )
        return {}
    index: dict[str, dict[str, float]] = {}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output_path = str(row.get("output_path", ""))
            if not output_path:
                continue
            index[output_path] = {
                "tfidg_lite_score": _float(row.get("tfidg_lite_score")),
                "adaptive_mask_coverage_score": _float(row.get("adaptive_mask_coverage_score")),
                "feature_alignment_score": _float(row.get("feature_alignment_score")),
                "texture_preservation_score": _float(row.get("texture_preservation_score")),
                "leakage_score": _float(row.get("leakage_score")),
                "morphology_fit_score": _float(row.get("morphology_fit_score")),
            }
    return index


def _float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _synthetic_selector_decision(row: dict[str, Any], phase5: dict[str, Any]) -> dict[str, Any]:
    selector = _synthetic_selector_settings(phase5)
    selector_enabled = bool(selector.get("enabled", False))
    quality_filter_enabled = bool(phase5.get("quality_filtered_training", {}).get("enabled", False)) if isinstance(phase5.get("quality_filtered_training", {}), dict) else bool(phase5.get("quality_filtered_training", False))
    settings = row.get("settings", {})
    morphology = str(settings.get("quality_morphology", row.get("quality_profile", "unknown"))) if isinstance(settings, dict) else "unknown"
    profile = str(row.get("quality_profile", settings.get("quality_profile", "unknown") if isinstance(settings, dict) else "unknown"))
    flags = list(row.get("quality_flags", []))
    visibility = float(row.get("defect_visibility_score", 0.0))
    background = float(row.get("background_preservation_l1", 0.0))
    outside = float(row.get("outside_refined_change_fraction", 0.0))
    inpaint_area = float(row.get("inpaint_mask_area_fraction", 0.0))
    quality = float(row.get("generation_quality_score", 0.0))
    tfidg = row.get("tfidg_lite", {}) if isinstance(row.get("tfidg_lite"), dict) else {}
    tfidg_settings = selector.get("tfidg_lite", {}) if isinstance(selector.get("tfidg_lite"), dict) else {}
    tfidg_enabled = bool(tfidg_settings.get("enabled", False))
    tfidg_score = float(tfidg.get("tfidg_lite_score", 0.0)) if tfidg else 0.0
    tfidg_coverage = float(tfidg.get("adaptive_mask_coverage_score", 0.0)) if tfidg else 0.0
    tfidg_alignment = float(tfidg.get("feature_alignment_score", 0.0)) if tfidg else 0.0
    tfidg_texture = float(tfidg.get("texture_preservation_score", 0.0)) if tfidg else 0.0
    tfidg_leakage = float(tfidg.get("leakage_score", 0.0)) if tfidg else 0.0
    reject_reasons: list[str] = []

    if not selector_enabled:
        if quality_filter_enabled and not _passes_phase5_quality_filter(row, phase5):
            reject_reasons.append("quality_filtered_training_failed")
        return {
            "accepted": not reject_reasons,
            "reject_reasons": reject_reasons,
            "selector_score": quality,
        }

    if quality_filter_enabled and not _passes_phase5_quality_filter(row, phase5):
        reject_reasons.append("quality_filtered_training_failed")
    if flags and bool(selector.get("reject_quality_flags", True)):
        reject_reasons.append("quality_flags")
    if visibility < float(selector.get("min_defect_visibility_score", phase5.get("min_defect_visibility_score", 0.015))):
        reject_reasons.append("low_defect_visibility")
    if background > float(selector.get("max_background_l1", phase5.get("max_background_l1", 0.05))):
        reject_reasons.append("high_background_change")
    if outside > float(selector.get("max_outside_refined_change_fraction", phase5.get("max_outside_refined_change_fraction", 0.25))):
        reject_reasons.append("high_outside_refined_change")
    if inpaint_area > float(selector.get("max_inpaint_area_fraction", phase5.get("max_inpaint_area_fraction", 0.20))):
        reject_reasons.append("overbroad_inpaint_mask")
    if tfidg_enabled:
        if not tfidg and bool(tfidg_settings.get("reject_missing_metrics", True)):
            reject_reasons.append("missing_tfidg_lite_metrics")
        min_coverage = float(tfidg_settings.get("min_adaptive_mask_coverage_score", 0.35))
        if tfidg and tfidg_coverage < min_coverage:
            reject_reasons.append("low_tfidg_mask_coverage")
        min_score = tfidg_settings.get("min_tfidg_lite_score")
        if tfidg and min_score is not None and tfidg_score < float(min_score):
            reject_reasons.append("low_tfidg_lite_score")

    preferred = _preferred_profiles_for_morphology(selector, morphology)
    profile_bonus = 0.0
    if preferred:
        if profile in preferred:
            profile_bonus = float(selector.get("preferred_profile_bonus", 0.04))
        else:
            profile_bonus = -float(selector.get("nonpreferred_profile_penalty", 0.03))
            if bool(selector.get("reject_nonpreferred_profiles", False)):
                reject_reasons.append("nonpreferred_morphology_profile")

    score = (
        quality
        + float(selector.get("visibility_weight", 1.5)) * visibility
        - float(selector.get("background_weight", 1.0)) * background
        - float(selector.get("outside_change_weight", 0.20)) * outside
        - float(selector.get("inpaint_area_weight", 0.05)) * inpaint_area
        - float(selector.get("flag_penalty", 0.05)) * len(flags)
        + profile_bonus
        + (float(tfidg_settings.get("selector_score_weight", 0.35)) * tfidg_score if tfidg_enabled and tfidg else 0.0)
    )
    return {
        "accepted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "selector_score": float(score),
        "tfidg_lite_score": tfidg_score,
        "tfidg_adaptive_mask_coverage_score": tfidg_coverage,
        "tfidg_feature_alignment_score": tfidg_alignment,
        "tfidg_texture_preservation_score": tfidg_texture,
        "tfidg_leakage_score": tfidg_leakage,
    }


def _synthetic_selector_settings(phase5: dict[str, Any]) -> dict[str, Any]:
    raw = phase5.get("synthetic_selector", {})
    settings = dict(raw) if isinstance(raw, dict) else {"enabled": bool(raw)}
    settings.setdefault("enabled", bool(phase5.get("quality_filtered_training", {}).get("enabled", False)) if isinstance(phase5.get("quality_filtered_training", {}), dict) else False)
    settings.setdefault("group_by", ["category", "defect_type", "morphology", "variant", "quality_profile"])
    settings.setdefault("max_per_group", 999999)
    settings.setdefault(
        "preferred_profiles",
        {
            "multi_scuff": ["scuff_soft_low_strength"],
            "scratch_band": ["scratch_thin_detail", "scratch_ridge_balanced"],
            "single_stroke": ["single_stroke_clean"],
            "crack_band": ["scratch_thin_detail", "scratch_ridge_balanced"],
        },
    )
    return settings


def _preferred_profiles_for_morphology(selector: dict[str, Any], morphology: str) -> list[str]:
    raw = selector.get("preferred_profiles", {})
    if not isinstance(raw, dict):
        return []
    values = raw.get(morphology, [])
    return [str(value) for value in values] if isinstance(values, list) else [str(values)]


def _phase4_candidate_key(row: dict[str, Any]) -> str:
    return str(row.get("output_path", ""))


def _selected_candidate_keys(candidates: list[tuple[dict[str, Any], dict[str, Any]]], phase5: dict[str, Any]) -> set[str]:
    selector = _synthetic_selector_settings(phase5)
    enabled = bool(selector.get("enabled", False))
    accepted = [(row, decision) for row, decision in candidates if bool(decision.get("accepted", False))]
    if not enabled:
        return {_phase4_candidate_key(row) for row, _ in accepted}
    max_per_group = int(selector.get("max_per_group", 999999))
    if max_per_group <= 0:
        return set()
    group_by = [str(value) for value in selector.get("group_by", [])]
    grouped: dict[tuple[str, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, decision in accepted:
        grouped.setdefault(_selector_group_key(row, group_by), []).append((row, decision))
    selected: set[str] = set()
    for values in grouped.values():
        ranked = sorted(values, key=lambda item: float(item[1].get("selector_score", 0.0)), reverse=True)
        selected.update(_phase4_candidate_key(row) for row, _ in ranked[:max_per_group])
    return selected


def _selector_group_key(row: dict[str, Any], group_by: list[str]) -> tuple[str, ...]:
    settings = row.get("settings", {})
    values = {
        "category": str(row.get("category", "")),
        "defect_type": str(row.get("defect_type", "")),
        "variant": str(row.get("variant", "")),
        "quality_profile": str(row.get("quality_profile", settings.get("quality_profile", "")) if isinstance(settings, dict) else row.get("quality_profile", "")),
        "morphology": str(settings.get("quality_morphology", row.get("quality_profile", "unknown"))) if isinstance(settings, dict) else "unknown",
    }
    return tuple(values.get(key, "") for key in group_by)


def _write_synthetic_selector_reports(report_dir: Path, selector_logs: list[dict[str, Any]], phase5: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = report_dir / "synthetic_selector_report.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in selector_logs) + ("\n" if selector_logs else ""),
        encoding="utf-8",
    )
    summary = _synthetic_selector_summary(selector_logs)
    selector = _synthetic_selector_settings(phase5)
    tfidg_settings = selector.get("tfidg_lite", {}) if isinstance(selector.get("tfidg_lite"), dict) else {}
    md_path = report_dir / "synthetic_selector_report.md"
    lines = [
        "# Synthetic Selector Report",
        "",
        f"Selector enabled: `{bool(selector.get('enabled', False))}`",
        f"TF-IDG-lite gate enabled: `{bool(tfidg_settings.get('enabled', False))}`",
        f"TF-IDG-lite min adaptive mask coverage: `{tfidg_settings.get('min_adaptive_mask_coverage_score', 'n/a')}`",
        f"Candidate rows: `{len(selector_logs)}`",
        f"Accepted rows: `{summary['accepted']}`",
        f"Rejected rows: `{summary['rejected']}`",
        f"JSONL: `{jsonl_path}`",
        "",
        "This selector is used before Phase 5 synthetic mixing. It rejects low-quality synthetic samples and keeps the best rows per configured group.",
        "",
        "## Rejection Reasons",
        "",
    ]
    reason_counts: dict[str, int] = summary["reject_reasons"]
    if reason_counts:
        lines.extend(f"- `{reason}`: `{count}`" for reason, count in sorted(reason_counts.items()))
    else:
        lines.append("- none")
    lines.extend(["", "## Accepted By Morphology/Profile", ""])
    accepted_by_group: dict[tuple[str, str], int] = summary["accepted_by_morphology_profile"]
    if accepted_by_group:
        for (morphology, profile), count in sorted(accepted_by_group.items()):
            lines.append(f"- `{morphology}` / `{profile}`: `{count}`")
    else:
        lines.append("- none")
    best = sorted((row for row in selector_logs if row.get("accepted")), key=lambda row: float(row.get("selector_score", 0.0)), reverse=True)[:10]
    if best:
        lines.extend(
            [
                "",
                "## Top Accepted Samples",
                "",
                "| Rank | Variant | Profile | Morphology | Selector Score | TF-IDG Score | Coverage | Output |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for index, row in enumerate(best, start=1):
            lines.append(
                "| "
                f"{index} | "
                f"{row.get('variant')} | "
                f"{row.get('quality_profile')} | "
                f"{row.get('morphology')} | "
                f"{float(row.get('selector_score', 0.0)):.6f} | "
                f"{float(row.get('tfidg_lite_score', 0.0)):.4f} | "
                f"{float(row.get('tfidg_adaptive_mask_coverage_score', 0.0)):.4f} | "
                f"`{Path(str(row.get('output_path', ''))).name}` |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _synthetic_selector_summary(selector_logs: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    accepted_by_group: dict[tuple[str, str], int] = {}
    accepted = 0
    for row in selector_logs:
        if row.get("accepted"):
            accepted += 1
            key = (str(row.get("morphology", "unknown")), str(row.get("quality_profile", "unknown")))
            accepted_by_group[key] = accepted_by_group.get(key, 0) + 1
        for reason in row.get("reject_reasons", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "accepted": accepted,
        "rejected": len(selector_logs) - accepted,
        "reject_reasons": reason_counts,
        "accepted_by_morphology_profile": accepted_by_group,
    }


def _phase4_metadata_path(config: AppConfig, provider: str, variant: str | None = None) -> Path:
    if provider == "qwen" and variant:
        return config.output_dir / "phase4" / provider / variant / "metadata.jsonl"
    return config.output_dir / "phase4" / provider / "metadata.jsonl"


def _phase5_variants(config: AppConfig, provider: str) -> list[str]:
    phase5 = _phase5_config(config)
    phase4 = dict(config.data.get("phase4", {}))
    raw = phase5.get("variants", phase4.get("variants"))
    if raw is None:
        return []
    variants = [str(value) for value in raw] if isinstance(raw, list) else [str(raw)]
    if provider != "qwen":
        return []
    allowed = {
        "qwen_mask_only",
        "full_qwen_hybrid",
        "fixed_mask_adapter",
        "clone_harmonized",
        "ip_adapter_hybrid",
        "latent_blend_harmonized",
    }
    unknown = [value for value in variants if value not in allowed]
    if unknown:
        raise ValueError(f"Unsupported Phase 5 variant(s): {', '.join(unknown)}")
    return variants


def _phase5_variant_specs(config: AppConfig, provider: str, variants: list[str]) -> list[dict[str, str | None]]:
    if provider != "qwen" or not variants:
        return []
    phase5 = _phase5_config(config)
    if not bool(phase5.get("group_by_quality_profile", False)):
        return [{"label": variant, "variant": variant, "quality_profile": None} for variant in variants]
    raw_profiles = phase5.get("quality_profiles", "all")
    if raw_profiles == "all":
        profiles = list(PHASE4_QUALITY_PROFILES)
    elif isinstance(raw_profiles, list):
        profiles = [str(value) for value in raw_profiles]
    else:
        profiles = [str(raw_profiles)]
    unknown = [profile for profile in profiles if profile not in PHASE4_QUALITY_PROFILES]
    if unknown:
        raise ValueError(f"Unsupported Phase 5 quality_profile(s): {', '.join(unknown)}")
    return [
        {"label": f"{variant}:{profile}", "variant": variant, "quality_profile": profile}
        for variant in variants
        for profile in profiles
    ]


def _mix_samples(
    real: list[SegmentationSample],
    synthetic: list[SegmentationSample],
    synthetic_ratio: float,
    seed: int,
    class_balanced: bool = False,
) -> list[SegmentationSample]:
    if synthetic_ratio <= 0.0 or not synthetic:
        return list(real)
    if synthetic_ratio >= 1.0:
        return list(real) + list(synthetic)
    target_synth = max(1, round(len(real) * synthetic_ratio / max(1e-6, 1.0 - synthetic_ratio)))
    rng = random.Random(seed + int(synthetic_ratio * 10_000))
    if class_balanced:
        selected = _class_balanced_synthetic(real, synthetic, target_synth, rng)
    else:
        selected = [synthetic[index % len(synthetic)] for index in range(target_synth)]
    rng.shuffle(selected)
    mixed = list(real) + selected
    rng.shuffle(mixed)
    return mixed


def _class_balanced_synthetic(
    real: list[SegmentationSample],
    synthetic: list[SegmentationSample],
    target_count: int,
    rng: random.Random,
) -> list[SegmentationSample]:
    real_keys = [(sample.category, sample.defect_type) for sample in real]
    key_order = sorted(set(real_keys))
    by_key: dict[tuple[str, str], list[SegmentationSample]] = {}
    for sample in synthetic:
        by_key.setdefault((sample.category, sample.defect_type), []).append(sample)
    selected: list[SegmentationSample] = []
    while len(selected) < target_count:
        made_progress = False
        for key in key_order:
            bucket = by_key.get(key)
            if not bucket:
                continue
            selected.append(bucket[len(selected) % len(bucket)])
            made_progress = True
            if len(selected) >= target_count:
                break
        if not made_progress:
            selected.extend(synthetic[index % len(synthetic)] for index in range(target_count - len(selected)))
    rng.shuffle(selected)
    return selected[:target_count]


def _patchcore_lite_evaluate(
    config: AppConfig,
    normal_train: list[SegmentationSample],
    calibration: list[SegmentationSample],
    held_out: list[SegmentationSample],
    seed: int,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    phase5 = _phase5_config(config)
    if not normal_train:
        raise ValueError("patchcore_lite needs at least one normal train/good image")
    image_size = int(phase5.get("patchcore_image_size", phase5.get("image_size", 96)))
    max_memory = int(phase5.get("patchcore_max_memory_patches", 5_000))
    chunk_size = int(phase5.get("patchcore_distance_chunk_size", 512))
    memory = _patchcore_lite_memory(normal_train, image_size, max_memory, seed)
    calibration_scores = _patchcore_lite_score_samples(calibration, memory, image_size, chunk_size)
    held_out_scores = _patchcore_lite_score_samples(held_out, memory, image_size, chunk_size)
    threshold_grid = _threshold_grid(phase5)
    threshold_selection = _select_threshold(calibration_scores, threshold_grid, phase5)
    selected_threshold = float(threshold_selection["selected_threshold"])
    selected_metrics = _evaluate_scores(held_out_scores, selected_threshold)
    fixed_threshold = float(phase5.get("threshold", 0.5))
    fixed_metrics = _evaluate_scores(held_out_scores, fixed_threshold)
    metric_rows = [segmentation_metrics(item["score"], item["truth"], threshold=selected_threshold) for item in held_out_scores]
    morphology_metrics = _evaluate_scores_by_morphology(held_out_scores, selected_threshold)
    prediction_contact_sheet = _write_prediction_examples(prediction_dir, held_out_scores, selected_threshold)
    return {
        "synthetic_ratio": 0.0,
        "train_real_count": len(normal_train),
        "train_synthetic_count": 0,
        "train_clean_negative_count": 0,
        "held_out_count": len(held_out),
        "epochs": 0,
        "image_size": image_size,
        "pos_weight": 1.0,
        "dice_loss_weight": 0.0,
        "threshold": selected_threshold,
        "threshold_policy": "adaptation_dice_on_patchcore_scores",
        "fixed_threshold": fixed_threshold,
        "calibration_iou": float(threshold_selection["calibration_iou"]),
        "calibration_dice": float(threshold_selection["calibration_dice"]),
        "loss_first": math.nan,
        "loss_final": math.nan,
        "predicted_positive_rate": selected_metrics["predicted_positive_rate"],
        "truth_positive_rate": selected_metrics["truth_positive_rate"],
        "pixel_auroc": _mean_metric(metric_rows, "pixel_auroc"),
        "aupro": _mean_metric(metric_rows, "aupro"),
        "iou": selected_metrics["iou"],
        "dice": selected_metrics["dice"],
        "fixed_predicted_positive_rate": fixed_metrics["predicted_positive_rate"],
        "fixed_iou": fixed_metrics["iou"],
        "fixed_dice": fixed_metrics["dice"],
        "score_mean": selected_metrics["score_mean"],
        "score_p95": selected_metrics["score_p95"],
        "morphology_metrics": morphology_metrics,
        "prediction_contact_sheet": str(prediction_contact_sheet) if prediction_contact_sheet else "",
        "note": "patchcore_lite_normal_memory_bank_handcrafted_features",
    }


def _patchcore_lite_memory(
    normal_train: list[SegmentationSample],
    image_size: int,
    max_memory: int,
    seed: int,
) -> np.ndarray:
    features = [_patchcore_lite_features(_load_rgb_array(sample.image_path, image_size)) for sample in normal_train]
    memory = np.concatenate([item.reshape(-1, item.shape[-1]) for item in features], axis=0).astype(np.float32)
    if len(memory) > max_memory:
        rng = np.random.default_rng(seed + 91_337)
        indices = rng.choice(len(memory), size=max_memory, replace=False)
        memory = memory[indices]
    return memory


def _patchcore_lite_score_samples(
    samples: list[SegmentationSample],
    memory: np.ndarray,
    image_size: int,
    chunk_size: int,
) -> list[dict[str, np.ndarray]]:
    scored: list[dict[str, np.ndarray]] = []
    for sample in samples:
        features = _patchcore_lite_features(_load_rgb_array(sample.image_path, image_size))
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
        distances: list[np.ndarray] = []
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start : start + chunk_size]
            # Squared Euclidean distance to the nearest normal patch feature.
            d2 = ((chunk[:, None, :] - memory[None, :, :]) ** 2).sum(axis=2)
            distances.append(np.sqrt(d2.min(axis=1)))
        score = np.concatenate(distances).reshape(image_size, image_size)
        score = _normalize_score_map(score)
        _, truth = _load_tensor_pair(sample, image_size, "cpu")
        scored.append(
            {
                "score": score.astype(np.float32),
                "truth": truth.squeeze().detach().cpu().numpy(),
                "morphology": sample.morphology,
                "image_path": sample.image_path,
                "mask_path": sample.mask_path,
                "category": sample.category,
                "defect_type": sample.defect_type,
            }
        )
    return scored


def _patchcore_lite_features(image: np.ndarray) -> np.ndarray:
    gray = image.mean(axis=2)
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    local_mean = _box_blur(gray, 5)
    residual = np.abs(gray - local_mean)
    return np.stack(
        [
            image[..., 0],
            image[..., 1],
            image[..., 2],
            gray,
            gx,
            gy,
            grad_mag,
            residual,
        ],
        axis=2,
    ).astype(np.float32)


def _load_rgb_array(path: str, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 1:
        return values
    pad = radius // 2
    padded = np.pad(values, pad, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[radius:, radius:]
        - integral[:-radius, radius:]
        - integral[radius:, :-radius]
        + integral[:-radius, :-radius]
    ) / float(radius * radius)


def _normalize_score_map(score: np.ndarray) -> np.ndarray:
    low = float(np.quantile(score, 0.01))
    high = float(np.quantile(score, 0.99))
    if high <= low:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip((score - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _patchcore_resnet_evaluate(
    config: AppConfig,
    normal_train: list[SegmentationSample],
    calibration: list[SegmentationSample],
    held_out: list[SegmentationSample],
    seed: int,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    phase5 = _phase5_config(config)
    if not normal_train:
        raise ValueError("patchcore_resnet needs at least one normal train/good image")
    image_size = int(phase5.get("patchcore_resnet_image_size", phase5.get("patchcore_image_size", phase5.get("image_size", 224))))
    max_memory = int(phase5.get("patchcore_resnet_max_memory_patches", phase5.get("patchcore_max_memory_patches", 10_000)))
    chunk_size = int(phase5.get("patchcore_resnet_distance_chunk_size", phase5.get("patchcore_distance_chunk_size", 256)))
    extractor = _PatchCoreResNetExtractor(phase5, _patchcore_device(phase5))
    memory = _patchcore_resnet_memory(normal_train, extractor, image_size, max_memory, seed)
    calibration_scores = _patchcore_resnet_score_samples(calibration, extractor, memory, image_size, chunk_size)
    held_out_scores = _patchcore_resnet_score_samples(held_out, extractor, memory, image_size, chunk_size)
    threshold_grid = _threshold_grid(phase5)
    threshold_selection = _select_threshold(calibration_scores, threshold_grid, phase5)
    selected_threshold = float(threshold_selection["selected_threshold"])
    selected_metrics = _evaluate_scores(held_out_scores, selected_threshold)
    fixed_threshold = float(phase5.get("threshold", 0.5))
    fixed_metrics = _evaluate_scores(held_out_scores, fixed_threshold)
    metric_rows = [segmentation_metrics(item["score"], item["truth"], threshold=selected_threshold) for item in held_out_scores]
    morphology_metrics = _evaluate_scores_by_morphology(held_out_scores, selected_threshold)
    prediction_contact_sheet = _write_prediction_examples(prediction_dir, held_out_scores, selected_threshold)
    return {
        "synthetic_ratio": 0.0,
        "train_real_count": len(normal_train),
        "train_synthetic_count": 0,
        "train_clean_negative_count": 0,
        "held_out_count": len(held_out),
        "epochs": 0,
        "image_size": image_size,
        "pos_weight": 1.0,
        "dice_loss_weight": 0.0,
        "threshold": selected_threshold,
        "threshold_policy": "adaptation_dice_on_patchcore_resnet_scores",
        "fixed_threshold": fixed_threshold,
        "calibration_iou": float(threshold_selection["calibration_iou"]),
        "calibration_dice": float(threshold_selection["calibration_dice"]),
        "loss_first": math.nan,
        "loss_final": math.nan,
        "predicted_positive_rate": selected_metrics["predicted_positive_rate"],
        "truth_positive_rate": selected_metrics["truth_positive_rate"],
        "pixel_auroc": _mean_metric(metric_rows, "pixel_auroc"),
        "aupro": _mean_metric(metric_rows, "aupro"),
        "iou": selected_metrics["iou"],
        "dice": selected_metrics["dice"],
        "fixed_predicted_positive_rate": fixed_metrics["predicted_positive_rate"],
        "fixed_iou": fixed_metrics["iou"],
        "fixed_dice": fixed_metrics["dice"],
        "score_mean": selected_metrics["score_mean"],
        "score_p95": selected_metrics["score_p95"],
        "morphology_metrics": morphology_metrics,
        "prediction_contact_sheet": str(prediction_contact_sheet) if prediction_contact_sheet else "",
        "note": f"patchcore_resnet_normal_memory_bank_{extractor.weights_label}_wide_resnet50_2_layer2_layer3",
    }


class _PatchCoreResNetExtractor:
    def __init__(self, phase5: dict[str, Any], device: str) -> None:
        self.device = device
        self.weights_label = str(phase5.get("patchcore_resnet_weights", "default")).lower()
        try:
            from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("patchcore_resnet requires torchvision with wide_resnet50_2 support") from exc
        if self.weights_label in {"none", "false", "random", "untrained"}:
            weights = None
            self.weights_label = "untrained"
        else:
            weights = Wide_ResNet50_2_Weights.DEFAULT
            self.weights_label = "imagenet_default"
        try:
            self.model = wide_resnet50_2(weights=weights).to(device).eval()
        except Exception as exc:
            raise RuntimeError(
                "Unable to load WideResNet50-2 for patchcore_resnet. "
                "If pretrained weights are unavailable, set phase5.patchcore_resnet_weights: none "
                "for a non-research smoke test, or pre-cache torchvision weights for final evidence."
            ) from exc
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._features: dict[str, torch.Tensor] = {}
        self.model.layer2.register_forward_hook(self._capture("layer2"))
        self.model.layer3.register_forward_hook(self._capture("layer3"))

    def _capture(self, name: str):
        def hook(_module, _inputs, output):
            self._features[name] = output.detach()

        return hook

    def features(self, image_path: str, image_size: int) -> np.ndarray:
        tensor = _resnet_input_tensor(image_path, image_size, self.device)
        self._features = {}
        with torch.no_grad():
            self.model(tensor)
            layer2 = self._features["layer2"]
            layer3 = torch.nn.functional.interpolate(
                self._features["layer3"],
                size=layer2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            features = torch.cat([layer2, layer3], dim=1)
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        return features.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _patchcore_device(phase5: dict[str, Any]) -> str:
    configured = str(phase5.get("patchcore_device", phase5.get("device", "cpu")))
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 5 PatchCore config requested CUDA, but torch.cuda.is_available() is false.")
    return configured


def _resnet_input_tensor(image_path: str, image_size: int, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)
    mean_tensor = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return ((tensor - mean_tensor) / std_tensor).to(device)


def _patchcore_resnet_memory(
    normal_train: list[SegmentationSample],
    extractor: _PatchCoreResNetExtractor,
    image_size: int,
    max_memory: int,
    seed: int,
) -> np.ndarray:
    features = [extractor.features(sample.image_path, image_size) for sample in normal_train]
    memory = np.concatenate([item.reshape(-1, item.shape[-1]) for item in features], axis=0).astype(np.float32)
    if len(memory) > max_memory:
        rng = np.random.default_rng(seed + 122_113)
        indices = rng.choice(len(memory), size=max_memory, replace=False)
        memory = memory[indices]
    return memory


def _patchcore_resnet_score_samples(
    samples: list[SegmentationSample],
    extractor: _PatchCoreResNetExtractor,
    memory: np.ndarray,
    image_size: int,
    chunk_size: int,
) -> list[dict[str, np.ndarray]]:
    scored: list[dict[str, np.ndarray]] = []
    for sample in samples:
        features = extractor.features(sample.image_path, image_size)
        grid_h, grid_w = features.shape[:2]
        flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
        distances: list[np.ndarray] = []
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start : start + chunk_size]
            d2 = ((chunk[:, None, :] - memory[None, :, :]) ** 2).sum(axis=2)
            distances.append(np.sqrt(d2.min(axis=1)))
        score_grid = np.concatenate(distances).reshape(grid_h, grid_w)
        score_grid = _normalize_score_map(score_grid)
        score_tensor = torch.from_numpy(score_grid)[None, None, :, :]
        score = (
            torch.nn.functional.interpolate(score_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
            .squeeze()
            .numpy()
        )
        score = _normalize_score_map(score)
        _, truth = _load_tensor_pair(sample, image_size, "cpu")
        scored.append(
            {
                "score": score.astype(np.float32),
                "truth": truth.squeeze().detach().cpu().numpy(),
                "morphology": sample.morphology,
                "image_path": sample.image_path,
                "mask_path": sample.mask_path,
                "category": sample.category,
                "defect_type": sample.defect_type,
            }
        )
    return scored


def _train_and_evaluate(
    config: AppConfig,
    train_samples: list[SegmentationSample],
    held_out: list[SegmentationSample],
    synthetic_ratio: float,
    seed: int,
    prediction_dir: Path | None = None,
) -> dict[str, Any]:
    phase5 = _phase5_config(config)
    device = _phase5_device(config)
    image_size = int(phase5.get("image_size", 96))
    epochs = int(phase5.get("epochs", 2))
    learning_rate = float(phase5.get("learning_rate", 1e-3))
    model = TinyUNet(base_channels=int(phase5.get("base_channels", 8))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    pos_weight = _positive_weight(train_samples, image_size, device, float(phase5.get("max_pos_weight", 25.0)))
    pos_weight_tensor = torch.tensor([pos_weight], device=device).view(1, 1, 1, 1)
    dice_loss_weight = float(phase5.get("dice_loss_weight", 0.5))
    focal_loss_weight = float(phase5.get("focal_loss_weight", 0.0))
    focal_gamma = float(phase5.get("focal_gamma", 2.0))
    uncertainty_loss_weight = _phase5_uncertainty_loss_weight(phase5)
    rng = random.Random(seed + int(synthetic_ratio * 1000))
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        shuffled = list(train_samples)
        rng.shuffle(shuffled)
        for sample in shuffled:
            image, mask = _load_tensor_pair(sample, image_size, device)
            loss_weight = _load_loss_weight(sample, image_size, device, phase5)
            logits = model(image.unsqueeze(0))
            target = mask.unsqueeze(0)
            weight = loss_weight.unsqueeze(0)
            loss = _weighted_bce_with_logits(logits, target, weight, pos_weight_tensor) + dice_loss_weight * _soft_dice_loss(
                logits,
                target,
                weight=weight,
            )
            if focal_loss_weight > 0.0:
                loss = loss + focal_loss_weight * _weighted_focal_loss(logits, target, weight, gamma=focal_gamma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.eval()
    metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        calibration_samples = [sample for sample in train_samples if sample.source == "real_adaptation"] or train_samples
        calibration_scores = _score_samples(model, calibration_samples, image_size, device)
        held_out_scores = _score_samples(model, held_out, image_size, device)
    fixed_threshold = float(phase5.get("threshold", 0.5))
    threshold_grid = _threshold_grid(phase5)
    threshold_selection = _select_threshold(calibration_scores, threshold_grid, phase5)
    selected_threshold = float(threshold_selection["selected_threshold"])
    selected_metrics = _evaluate_scores(held_out_scores, selected_threshold)
    fixed_metrics = _evaluate_scores(held_out_scores, fixed_threshold)
    for item in held_out_scores:
        metric_rows.append(segmentation_metrics(item["score"], item["truth"], threshold=selected_threshold))
    morphology_metrics = _evaluate_scores_by_morphology(held_out_scores, selected_threshold)
    prediction_contact_sheet = _write_prediction_examples(prediction_dir, held_out_scores, selected_threshold)
    clean_negative_count = sum(1 for sample in train_samples if sample.source == "sd_clean_inpaint_negative")
    supervised_normal_count = sum(1 for sample in train_samples if sample.source == "supervised_normal_negative")
    synthetic_count = sum(
        1
        for sample in train_samples
        if sample.source not in {"real_adaptation", "sd_clean_inpaint_negative", "supervised_normal_negative"}
    )
    real_count = sum(1 for sample in train_samples if sample.source == "real_adaptation")
    return {
        "synthetic_ratio": synthetic_ratio,
        "train_real_count": real_count,
        "train_synthetic_count": synthetic_count,
        "train_clean_negative_count": clean_negative_count,
        "train_supervised_normal_count": supervised_normal_count,
        "held_out_count": len(held_out),
        "epochs": epochs,
        "image_size": image_size,
        "pos_weight": pos_weight,
        "dice_loss_weight": dice_loss_weight,
        "focal_loss_weight": focal_loss_weight,
        "focal_gamma": focal_gamma,
        "uncertainty_loss_weight": uncertainty_loss_weight,
        "uncertainty_weighted_samples": sum(1 for sample in train_samples if sample.uncertainty_mask_path),
        "threshold": selected_threshold,
        "threshold_policy": str(phase5.get("threshold_policy", "adaptation_dice")),
        "fixed_threshold": fixed_threshold,
        "calibration_iou": float(threshold_selection["calibration_iou"]),
        "calibration_dice": float(threshold_selection["calibration_dice"]),
        "calibration_predicted_positive_rate": float(threshold_selection.get("predicted_positive_rate", math.nan)),
        "threshold_constraint_satisfied": int(float(threshold_selection.get("threshold_constraint_satisfied", 0.0)) >= 1.0),
        "loss_first": losses[0] if losses else math.nan,
        "loss_final": losses[-1] if losses else math.nan,
        "predicted_positive_rate": selected_metrics["predicted_positive_rate"],
        "truth_positive_rate": selected_metrics["truth_positive_rate"],
        "pixel_auroc": _mean_metric(metric_rows, "pixel_auroc"),
        "aupro": _mean_metric(metric_rows, "aupro"),
        "iou": selected_metrics["iou"],
        "dice": selected_metrics["dice"],
        "fixed_predicted_positive_rate": fixed_metrics["predicted_positive_rate"],
        "fixed_iou": fixed_metrics["iou"],
        "fixed_dice": fixed_metrics["dice"],
        "score_mean": selected_metrics["score_mean"],
        "score_p95": selected_metrics["score_p95"],
        "morphology_metrics": morphology_metrics,
        "prediction_contact_sheet": str(prediction_contact_sheet) if prediction_contact_sheet else "",
        "note": "tiny_unet_smoke",
    }


class TinyUNet(torch.nn.Module):
    def __init__(self, base_channels: int = 8) -> None:
        super().__init__()
        self.enc1 = _conv_block(3, base_channels)
        self.pool = torch.nn.MaxPool2d(2)
        self.enc2 = _conv_block(base_channels, base_channels * 2)
        self.up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = _conv_block(base_channels * 3, base_channels)
        self.out = torch.nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        up = self.up(e2)
        if up.shape[-2:] != e1.shape[-2:]:
            up = torch.nn.functional.interpolate(up, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.dec(torch.cat([e1, up], dim=1)))


def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
    )


def _load_tensor_pair(sample: SegmentationSample, image_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.open(sample.image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    preserve_soft_mask = sample.mask_mode == "soft"
    resample = Image.Resampling.BILINEAR if preserve_soft_mask else Image.Resampling.NEAREST
    if sample.mask_path:
        mask = Image.open(sample.mask_path).convert("L").resize((image_size, image_size), resample)
        raw_mask = np.asarray(mask, dtype=np.float32) / 255.0
    else:
        raw_mask = np.zeros((image_size, image_size), dtype=np.float32)
    image_arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if sample.positive_core_path and Path(sample.positive_core_path).exists():
        core = Image.open(sample.positive_core_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        core_arr = (np.asarray(core, dtype=np.float32) / 255.0) > 0.0
        raw_mask = np.maximum(raw_mask, core_arr.astype(np.float32))
    mask_arr = raw_mask[None, :, :] if preserve_soft_mask else (raw_mask > 0.0).astype(np.float32)[None, :, :]
    return torch.from_numpy(image_arr).to(device), torch.from_numpy(mask_arr).to(device)


def _load_loss_weight(sample: SegmentationSample, image_size: int, device: str, phase5: dict[str, Any]) -> torch.Tensor:
    min_weight = _phase5_uncertainty_loss_weight(phase5)
    core_weight = float(phase5.get("positive_core_loss_weight", 1.0))
    has_core = bool(sample.positive_core_path and Path(sample.positive_core_path).exists() and core_weight > 1.0)
    if (min_weight >= 1.0 or not sample.uncertainty_mask_path) and not has_core:
        return torch.ones((1, image_size, image_size), device=device, dtype=torch.float32)
    if sample.uncertainty_mask_path:
        path = Path(sample.uncertainty_mask_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 5 uncertainty mask: {path}")
        uncertainty = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        uncertainty_arr = np.asarray(uncertainty, dtype=np.float32) / 255.0
        weight_arr = 1.0 - uncertainty_arr * (1.0 - min_weight)
    else:
        weight_arr = np.ones((image_size, image_size), dtype=np.float32)
    if has_core:
        core = Image.open(sample.positive_core_path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        core_arr = np.asarray(core, dtype=np.float32) > 0.0
        weight_arr[core_arr] *= core_weight
    weight_arr = np.clip(weight_arr, min_weight, max(1.0, core_weight)).astype(np.float32)[None, :, :]
    return torch.from_numpy(weight_arr).to(device)


def _positive_weight(samples: list[SegmentationSample], image_size: int, device: str, max_weight: float) -> float:
    positives = 0.0
    total = 0.0
    for sample in samples:
        _, mask = _load_tensor_pair(sample, image_size, device)
        positives += float(mask.sum().detach().cpu())
        total += float(mask.numel())
    if positives <= 0.0:
        return 1.0
    negatives = max(0.0, total - positives)
    return float(min(max_weight, max(1.0, negatives / positives)))


def _weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    denominator = weight.sum().clamp_min(1e-6)
    return (raw * weight).sum() / denominator


def _weighted_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * target + (1.0 - probs) * (1.0 - target)
    focal = bce * (1.0 - pt).clamp_min(0.0).pow(gamma)
    denominator = weight.sum().clamp_min(1e-6)
    return (focal * weight).sum() / denominator


def _soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    if weight is None:
        weight = torch.ones_like(target)
    intersection = (probs * target * weight).sum(dim=(1, 2, 3))
    denominator = ((probs + target) * weight).sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _phase5_uncertainty_loss_weight(phase5: dict[str, Any]) -> float:
    raw = phase5.get("uncertainty_loss_weight", 1.0)
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError("phase5.uncertainty_loss_weight must be between 0.0 and 1.0")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "evaluator",
        "variant",
        "quality_profile",
        "run_seed",
        "synthetic_ratio",
        "train_real_count",
        "train_synthetic_count",
        "train_clean_negative_count",
        "train_supervised_normal_count",
        "held_out_count",
        "epochs",
        "image_size",
        "pos_weight",
        "dice_loss_weight",
        "focal_loss_weight",
        "focal_gamma",
        "uncertainty_loss_weight",
        "uncertainty_weighted_samples",
        "threshold",
        "threshold_policy",
        "fixed_threshold",
        "calibration_iou",
        "calibration_dice",
        "calibration_predicted_positive_rate",
        "threshold_constraint_satisfied",
        "loss_first",
        "loss_final",
        "predicted_positive_rate",
        "truth_positive_rate",
        "pixel_auroc",
        "aupro",
        "iou",
        "dice",
        "fixed_predicted_positive_rate",
        "fixed_iou",
        "fixed_dice",
        "score_mean",
        "score_p95",
        "morphology_metrics",
        "prediction_contact_sheet",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            copied = dict(row)
            if isinstance(copied.get("morphology_metrics"), dict):
                copied["morphology_metrics"] = json.dumps(copied["morphology_metrics"], sort_keys=True)
            writer.writerow(copied)


def _write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    real_count: int,
    normal_count: int,
    clean_negative_count: int,
    supervised_normal_count: int,
    synthetic_counts: dict[str, int],
    held_out_count: int,
    provider: str,
    audit_manifest: Path,
    evaluators: list[str],
    mask_policy_report: Path,
    selector_report: Path,
) -> None:
    variants = sorted({str(row.get("variant", provider)) for row in rows})
    uncertainty_weights = sorted({_format_optional_float(row.get("uncertainty_loss_weight")) for row in rows})
    uncertainty_weights = [value for value in uncertainty_weights if value != "n/a"]
    lines = [
        f"# Phase 5 Evaluation Report: {provider}",
        "",
        f"Evaluators: {', '.join(evaluators)}",
        f"Real adaptation samples: {real_count}",
        f"Normal train/good samples for normal-memory evaluators: {normal_count}",
        f"Clean SD-inpaint negative samples: {clean_negative_count}",
        f"Supervised normal zero-mask samples: {supervised_normal_count}",
        f"Phase 4 synthetic samples: {synthetic_counts}",
        f"Held-out real anomaly samples: {held_out_count}",
        f"Variants: {', '.join(variants)}",
        f"Uncertainty loss weights: {', '.join(uncertainty_weights) if uncertainty_weights else 'n/a'}",
        f"Human-light audit manifest: `{audit_manifest}`",
        f"Mask policy validation report: `{mask_policy_report}`",
        f"Synthetic selector report: `{selector_report}`",
        "",
        "IoU/Dice use a threshold selected on adaptation samples only; held-out anomalies are never used for threshold selection.",
        "Automatic masks are pseudo-labels. The audit manifest is intended for final research validation, not normal user operation.",
        "",
        "| Evaluator | Variant | Quality profile | Synthetic ratio | Real train | Synthetic train | Clean negatives | Supervised normals | Selected threshold | Pixel AUROC | AUPRO | IoU | Dice |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('evaluator', 'tiny_unet')} | "
            f"{row.get('variant', provider)} | "
            f"{row.get('quality_profile', 'all')} | "
            f"{float(row['synthetic_ratio']):.2f} | "
            f"{int(row['train_real_count'])} | "
            f"{int(row['train_synthetic_count'])} | "
            f"{int(row.get('train_clean_negative_count', 0))} | "
            f"{int(row.get('train_supervised_normal_count', 0))} | "
            f"{float(row['threshold']):.3f} | "
            f"{float(row['pixel_auroc']):.4f} | "
            f"{float(row['aupro']):.4f} | "
            f"{float(row['iou']):.4f} | "
            f"{float(row['dice']):.4f} |"
        )
    if rows:
        lines.extend(
            [
                "",
                "| Evaluator | Variant | Quality profile | Synthetic ratio | Predicted positive rate | Truth positive rate |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{row.get('evaluator', 'tiny_unet')} | "
                f"{row.get('variant', provider)} | "
                f"{row.get('quality_profile', 'all')} | "
                f"{float(row['synthetic_ratio']):.2f} | "
                f"{float(row['predicted_positive_rate']):.4f} | "
                f"{float(row['truth_positive_rate']):.4f} |"
            )
        lines.extend(
            [
                "",
                "| Evaluator | Variant | Quality profile | Synthetic ratio | Fixed 0.5 positive rate | Fixed 0.5 IoU | Fixed 0.5 Dice | Score mean | Score p95 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                f"{row.get('evaluator', 'tiny_unet')} | "
                f"{row.get('variant', provider)} | "
                f"{row.get('quality_profile', 'all')} | "
                f"{float(row['synthetic_ratio']):.2f} | "
                f"{float(row['fixed_predicted_positive_rate']):.4f} | "
                f"{float(row['fixed_iou']):.4f} | "
                f"{float(row['fixed_dice']):.4f} | "
                f"{float(row['score_mean']):.4f} | "
                f"{float(row['score_p95']):.4f} |"
            )
    lines.extend(["", _provider_note(provider)])
    if rows and all(float(row["fixed_predicted_positive_rate"]) == 0.0 for row in rows):
        lines.append(
            "Diagnostic: every ratio predicted zero positive pixels at the fixed threshold 0.5; "
            "selected-threshold IoU/Dice should be used for this smoke report."
        )
    decision = _ablation_decision(rows)
    if decision:
        lines.extend(["", "## Ablation Decision", "", decision])
    best_profiles = _best_profile_lines(rows)
    if best_profiles:
        lines.extend(["", "## Best Quality Profiles", "", *best_profiles])
    morphology_lines = _morphology_summary_lines(rows)
    if morphology_lines:
        lines.extend(["", "## Morphology Summary", "", *morphology_lines])
    ratio_guidance = _ratio_guidance(rows)
    if ratio_guidance:
        lines.extend(["", "## Synthetic Ratio Guidance", "", ratio_guidance])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ablation_decision(rows: list[dict[str, Any]]) -> str | None:
    by_key = {(str(row.get("variant", "")), float(row["synthetic_ratio"])): row for row in rows}
    mask = by_key.get(("qwen_mask_only", 0.5))
    full = by_key.get(("full_qwen_hybrid", 0.5))
    ratio = 0.5
    if mask is None or full is None:
        for candidate_ratio in (0.5, 0.65):
            mask_candidates = [
                row
                for row in rows
                if str(row.get("variant", "")).startswith("qwen_mask_only:")
                and abs(float(row["synthetic_ratio"]) - candidate_ratio) < 1e-6
            ]
            full_candidates = [
                row
                for row in rows
                if str(row.get("variant", "")).startswith("full_qwen_hybrid:")
                and abs(float(row["synthetic_ratio"]) - candidate_ratio) < 1e-6
            ]
            if mask_candidates and full_candidates:
                mask = max(mask_candidates, key=lambda row: (float(row["pixel_auroc"]), float(row["aupro"])))
                full = max(full_candidates, key=lambda row: (float(row["pixel_auroc"]), float(row["aupro"])))
                ratio = candidate_ratio
                break
    if mask is None or full is None:
        return None
    auroc_delta = float(full["pixel_auroc"]) - float(mask["pixel_auroc"])
    aupro_delta = float(full["aupro"]) - float(mask["aupro"])
    verdict = "passes" if auroc_delta > 0.0 or aupro_delta > 0.0 else "does not pass"
    return (
        f"At synthetic ratio `{ratio:.2f}`, `full_qwen_hybrid` {verdict} the Phase 8 success criterion "
        f"against `qwen_mask_only`: AUROC delta `{auroc_delta:.4f}`, AUPRO delta `{aupro_delta:.4f}`. "
        "Background preservation and mask-change diagnostics are reported in the Phase 4 variant summaries."
    )


def _best_profile_lines(rows: list[dict[str, Any]]) -> list[str]:
    candidates = [row for row in rows if float(row.get("synthetic_ratio", 0.0)) > 0.0 and ":" in str(row.get("variant", ""))]
    if not candidates:
        return []
    by_base: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        base = str(row["variant"]).split(":", 1)[0]
        by_base.setdefault(base, []).append(row)
    lines: list[str] = []
    for base, values in sorted(by_base.items()):
        best = max(values, key=lambda row: (float(row.get("pixel_auroc", 0.0)), float(row.get("aupro", 0.0)), float(row.get("dice", 0.0))))
        lines.append(
            f"- `{base}` best observed profile: `{best.get('quality_profile', 'all')}` at ratio "
            f"`{float(best['synthetic_ratio']):.2f}` "
            f"(AUROC `{float(best['pixel_auroc']):.4f}`, AUPRO `{float(best['aupro']):.4f}`, Dice `{float(best['dice']):.4f}`)."
        )
    return lines


def _morphology_summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    by_morphology: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metrics = row.get("morphology_metrics")
        if not isinstance(metrics, dict):
            continue
        for morphology, values in metrics.items():
            if isinstance(values, dict):
                item = {
                    "morphology": str(morphology),
                    "evaluator": row.get("evaluator", "tiny_unet"),
                    "variant": row.get("variant", ""),
                    "quality_profile": row.get("quality_profile", "all"),
                    "synthetic_ratio": row.get("synthetic_ratio", 0.0),
                    **values,
                }
                by_morphology.setdefault(str(morphology), []).append(item)
    lines: list[str] = []
    for morphology, values in sorted(by_morphology.items()):
        best = max(
            values,
            key=lambda row: (
                float(row.get("pixel_auroc", 0.0)),
                float(row.get("aupro", 0.0)),
                float(row.get("dice", 0.0)),
            ),
        )
        lines.append(
            f"- `{morphology}` best observed row: evaluator `{best['evaluator']}`, variant `{best['variant']}`, "
            f"profile `{best['quality_profile']}`, ratio `{float(best['synthetic_ratio']):.2f}` "
            f"(AUROC `{float(best.get('pixel_auroc', math.nan)):.4f}`, "
            f"AUPRO `{float(best.get('aupro', math.nan)):.4f}`, Dice `{float(best.get('dice', math.nan)):.4f}`, "
            f"samples `{int(best.get('count', 0))}`)."
        )
    return lines


def _ratio_guidance(rows: list[dict[str, Any]]) -> str | None:
    synthetic_rows = [
        row
        for row in rows
        if row.get("evaluator", "tiny_unet") == "tiny_unet" and float(row.get("synthetic_ratio", 0.0)) > 0.0
    ]
    if not synthetic_rows:
        return None
    best = max(
        synthetic_rows,
        key=lambda row: (
            float(row.get("pixel_auroc", 0.0)),
            float(row.get("aupro", 0.0)),
            float(row.get("dice", 0.0)),
        ),
    )
    best_ratio = float(best["synthetic_ratio"])
    low_ratio_rows = [row for row in synthetic_rows if float(row["synthetic_ratio"]) <= 0.5]
    low_ratio_best = max(low_ratio_rows, key=lambda row: float(row.get("pixel_auroc", 0.0))) if low_ratio_rows else best
    guidance = (
        f"Best observed row is `{best.get('variant')}` at ratio `{best_ratio:.2f}` "
        f"(AUROC `{float(best['pixel_auroc']):.4f}`, AUPRO `{float(best['aupro']):.4f}`). "
    )
    if best_ratio > 0.5 and float(low_ratio_best.get("pixel_auroc", 0.0)) >= float(best.get("pixel_auroc", 0.0)) - 0.02:
        guidance += (
            f"A lower ratio remains competitive: `{low_ratio_best.get('variant')}` at "
            f"`{float(low_ratio_best['synthetic_ratio']):.2f}`. Prefer the lower ratio unless visual QC clearly supports heavier synthetic mixing."
        )
    elif best_ratio <= 0.5:
        guidance += "This supports using a conservative synthetic mix before trying heavier synthetic replacement."
    else:
        guidance += "This favors a heavier synthetic mix for this run, but visual QC should be checked before adopting it."
    return guidance


def _provider_note(provider: str) -> str:
    if provider == "qwen":
        return (
            "This is a tiny U-Net downstream smoke scaffold using real Qwen/SD1.5 Phase 4 synthetic outputs; "
            "it preserves no-leakage split discipline. When enabled, `patchcore_resnet` is a WideResNet50-2 "
            "normal-memory PatchCore-style baseline; `patchcore_lite` is a local handcrafted fallback."
        )
    return (
        "This is a tiny U-Net downstream scaffold using heuristic Phase 4 synthetic outputs; "
        "it preserves no-leakage split discipline. When enabled, `patchcore_resnet` is a WideResNet50-2 "
        "normal-memory PatchCore-style baseline; `patchcore_lite` is a local handcrafted fallback."
    )


def _phase5_config(config: AppConfig) -> dict[str, Any]:
    return dict(config.data.get("phase5", {}))


def _format_optional_float(value: object) -> str:
    if value in {None, ""}:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _phase5_fingerprint(config: AppConfig, provider: str) -> str:
    return fingerprint(
        {
            "split": config.split_spec_fingerprint(),
            "phase4": config.data.get("phase4", {}),
            "phase5": config.data.get("phase5", {}),
            "provider": provider,
            "schema_version": PHASE5_SCHEMA_VERSION,
        }
    )


def _phase5_device(config: AppConfig) -> str:
    configured = str(config.data.get("phase5", {}).get("device", "cpu"))
    if configured == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 5 config requested CUDA, but torch.cuda.is_available() is false.")
    return configured


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if not math.isnan(float(row[key]))]
    return mean(values) if values else math.nan


def _score_samples(
    model: torch.nn.Module,
    samples: list[SegmentationSample],
    image_size: int,
    device: str,
) -> list[dict[str, np.ndarray]]:
    scored: list[dict[str, np.ndarray]] = []
    for sample in samples:
        image, mask = _load_tensor_pair(sample, image_size, device)
        score = torch.sigmoid(model(image.unsqueeze(0))).squeeze().detach().cpu().numpy()
        truth = mask.squeeze().detach().cpu().numpy()
        scored.append(
            {
                "score": score,
                "truth": truth,
                "morphology": sample.morphology,
                "image_path": sample.image_path,
                "mask_path": sample.mask_path,
                "category": sample.category,
                "defect_type": sample.defect_type,
            }
        )
    return scored


def _threshold_grid(phase5: dict[str, Any]) -> list[float]:
    raw = phase5.get("threshold_grid")
    if raw is not None:
        values = [float(value) for value in raw]
    else:
        values = [round(value, 2) for value in np.linspace(0.05, 0.95, 19)]
    return [min(1.0, max(0.0, value)) for value in values]


def _select_threshold(
    score_rows: list[dict[str, np.ndarray]],
    thresholds: list[float],
    phase5: dict[str, Any] | None = None,
) -> dict[str, float]:
    settings = phase5 or {}
    min_rate = float(settings.get("min_predicted_positive_rate", 0.0))
    max_rate = float(settings.get("max_predicted_positive_rate", 1.0))
    target_rate = float(settings.get("target_predicted_positive_rate", 0.05))
    best: dict[str, float] | None = None
    fallback: dict[str, float] | None = None
    for threshold in _adaptive_thresholds(score_rows, thresholds):
        metrics = _evaluate_scores(score_rows, threshold)
        candidate = {
            "selected_threshold": float(threshold),
            "calibration_iou": float(metrics["iou"]),
            "calibration_dice": float(metrics["dice"]),
            "predicted_positive_rate": float(metrics["predicted_positive_rate"]),
            "threshold_constraint_satisfied": float(min_rate <= float(metrics["predicted_positive_rate"]) <= max_rate),
        }
        fallback = _better_threshold_candidate(candidate, fallback, target_rate)
        if min_rate <= candidate["predicted_positive_rate"] <= max_rate:
            best = _better_threshold_candidate(candidate, best, target_rate)
    if best is None:
        best = fallback
    if best is None:
        return {
            "selected_threshold": 0.5,
            "calibration_iou": math.nan,
            "calibration_dice": math.nan,
            "predicted_positive_rate": math.nan,
            "threshold_constraint_satisfied": 0.0,
        }
    return best


def _better_threshold_candidate(
    candidate: dict[str, float],
    current: dict[str, float] | None,
    target_positive_rate: float,
) -> dict[str, float]:
    if current is None:
        return candidate
    if candidate["calibration_dice"] > current["calibration_dice"]:
        return candidate
    if candidate["calibration_dice"] == current["calibration_dice"]:
        if abs(candidate["predicted_positive_rate"] - target_positive_rate) < abs(
            current["predicted_positive_rate"] - target_positive_rate
        ):
            return candidate
    return current


def _adaptive_thresholds(score_rows: list[dict[str, np.ndarray]], base_thresholds: list[float]) -> list[float]:
    score_values = [np.asarray(item["score"], dtype=np.float32).reshape(-1) for item in score_rows]
    if not score_values:
        return sorted(set(base_thresholds))
    scores = np.concatenate(score_values)
    quantiles = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    values = list(base_thresholds) + [float(value) for value in quantiles]
    values.extend([float(scores.min()) - 1e-6, float(scores.max()) + 1e-6])
    return sorted({min(1.0, max(0.0, round(value, 6))) for value in values})


def _evaluate_scores(score_rows: list[dict[str, np.ndarray]], threshold: float) -> dict[str, float]:
    intersections = 0
    unions = 0
    predicted = 0
    positives = 0
    score_values: list[np.ndarray] = []
    predicted_positive_rates: list[float] = []
    truth_positive_rates: list[float] = []
    for item in score_rows:
        score = np.asarray(item["score"], dtype=np.float32)
        truth = np.asarray(item["truth"]) > 0
        binary = score >= threshold
        intersections += int(np.logical_and(binary, truth).sum())
        unions += int(np.logical_or(binary, truth).sum())
        predicted += int(binary.sum())
        positives += int(truth.sum())
        predicted_positive_rates.append(float(binary.mean()))
        truth_positive_rates.append(float(truth.mean()))
        score_values.append(score.reshape(-1))
    all_scores = np.concatenate(score_values) if score_values else np.asarray([], dtype=np.float32)
    return {
        "predicted_positive_rate": mean(predicted_positive_rates) if predicted_positive_rates else math.nan,
        "truth_positive_rate": mean(truth_positive_rates) if truth_positive_rates else math.nan,
        "iou": intersections / unions if unions else math.nan,
        "dice": (2 * intersections) / (predicted + positives) if predicted + positives else math.nan,
        "score_mean": float(all_scores.mean()) if all_scores.size else math.nan,
        "score_p95": float(np.quantile(all_scores, 0.95)) if all_scores.size else math.nan,
    }


def _evaluate_scores_by_morphology(score_rows: list[dict[str, np.ndarray]], threshold: float) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, np.ndarray]]] = {}
    for item in score_rows:
        morphology = str(item.get("morphology", "unknown"))
        grouped.setdefault(morphology, []).append(item)
    result: dict[str, dict[str, float]] = {}
    for morphology, values in grouped.items():
        aggregate = _evaluate_scores(values, threshold)
        metric_rows = [segmentation_metrics(item["score"], item["truth"], threshold=threshold) for item in values]
        result[morphology] = {
            "count": float(len(values)),
            "pixel_auroc": _mean_metric(metric_rows, "pixel_auroc"),
            "aupro": _mean_metric(metric_rows, "aupro"),
            "iou": aggregate["iou"],
            "dice": aggregate["dice"],
            "predicted_positive_rate": aggregate["predicted_positive_rate"],
            "truth_positive_rate": aggregate["truth_positive_rate"],
        }
    return result


def _write_prediction_examples(
    prediction_dir: Path | None,
    score_rows: list[dict[str, Any]],
    threshold: float,
    max_samples: int = 6,
) -> Path | None:
    if prediction_dir is None or not score_rows:
        return None
    prediction_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(
        score_rows,
        key=lambda item: float(np.asarray(item["score"], dtype=np.float32).max()),
        reverse=True,
    )[:max_samples]
    thumb, label_h = 128, 24
    columns: list[tuple[str, Any]] = [
        ("image", lambda item: Image.open(str(item["image_path"])).convert("RGB")),
        ("truth", lambda item: _overlay_truth(Image.open(str(item["image_path"])).convert("RGB"), np.asarray(item["truth"]))),
        ("score", lambda item: _score_heatmap(np.asarray(item["score"], dtype=np.float32))),
        ("pred", lambda item: _prediction_overlay(Image.open(str(item["image_path"])).convert("RGB"), np.asarray(item["score"], dtype=np.float32), threshold)),
    ]
    sheet = Image.new("RGB", (thumb * len(columns), (thumb + label_h) * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    rows: list[dict[str, Any]] = []
    for row_index, item in enumerate(selected):
        for col_index, (label, getter) in enumerate(columns):
            x = col_index * thumb
            y = row_index * (thumb + label_h)
            panel = getter(item).resize((thumb, thumb), Image.Resampling.BILINEAR)
            sheet.paste(panel, (x, y + label_h))
            draw.text((x + 4, y + 4), f"{Path(str(item.get('image_path', 'image'))).name} {label}", fill="black")
        rows.append(
            {
                "image_path": str(item.get("image_path", "")),
                "mask_path": str(item.get("mask_path", "")),
                "morphology": str(item.get("morphology", "unknown")),
                "category": str(item.get("category", "")),
                "defect_type": str(item.get("defect_type", "")),
                "score_max": float(np.asarray(item["score"], dtype=np.float32).max()),
                "score_mean": float(np.asarray(item["score"], dtype=np.float32).mean()),
                "threshold": float(threshold),
            }
        )
    contact_sheet = prediction_dir / "prediction_contact_sheet.png"
    sheet.save(contact_sheet)
    (prediction_dir / "prediction_examples.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contact_sheet


def _overlay_truth(image: Image.Image, truth: np.ndarray) -> Image.Image:
    mask = np.asarray(truth, dtype=np.float32) > 0
    return _overlay_array_mask(image, mask, np.asarray([255.0, 40.0, 40.0], dtype=np.float32))


def _prediction_overlay(image: Image.Image, score: np.ndarray, threshold: float) -> Image.Image:
    mask = np.asarray(score, dtype=np.float32) >= threshold
    return _overlay_array_mask(image, mask, np.asarray([40.0, 180.0, 255.0], dtype=np.float32))


def _overlay_array_mask(image: Image.Image, mask: np.ndarray, color: np.ndarray) -> Image.Image:
    base = np.asarray(image.resize(mask.shape[::-1], Image.Resampling.BILINEAR).convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask] = overlay[mask] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).convert("RGB")


def _score_heatmap(score: np.ndarray) -> Image.Image:
    score = np.asarray(score, dtype=np.float32)
    low = float(np.nanmin(score)) if score.size else 0.0
    high = float(np.nanmax(score)) if score.size else 1.0
    norm = np.zeros_like(score, dtype=np.float32) if high <= low else np.clip((score - low) / (high - low), 0.0, 1.0)
    heat = np.zeros((*norm.shape, 3), dtype=np.float32)
    heat[..., 0] = norm * 255.0
    heat[..., 1] = (1.0 - np.abs(norm - 0.5) * 2.0) * 180.0
    heat[..., 2] = (1.0 - norm) * 130.0
    return Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8)).convert("RGB")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _count_by_category(samples: list[SegmentationSample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.category] = counts.get(sample.category, 0) + 1
    return counts
