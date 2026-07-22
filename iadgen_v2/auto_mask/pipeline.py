from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from iadgen_v2.auto_mask.contracts import AutoMaskContext, CandidateProposal, EvidenceProvider
from iadgen_v2.auto_mask.evidence.base import fuse_evidence_maps
from iadgen_v2.auto_mask.mask_roles import posterior_mask_roles
from iadgen_v2.auto_mask.proposals import (
    SamRefiner,
    generate_generic_proposals,
    proposal_from_mask,
    refine_proposals,
)
from iadgen_v2.auto_mask.refinement import EdgeAwareRefiner, edge_align_field
from iadgen_v2.auto_mask.selection import GenericCandidateSelector


def run_generic_evidence_pipeline(
    context: AutoMaskContext,
    providers: Iterable[EvidenceProvider],
    *,
    output_dir: Path,
    variant_dir: Path,
    artifact_stem: str,
    selector: GenericCandidateSelector,
    sam_refiner: SamRefiner | None = None,
    edge_refine: bool = True,
    min_component_area: int = 8,
    specialist_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    failures: dict[str, str] = {}
    provider_seconds: dict[str, float] = {}
    for provider in providers:
        name = getattr(provider, "name", type(provider).__name__)
        start = time.monotonic()
        try:
            item = provider.compute(context)
            evidence.append(item)
            _save_float_mask(item.values, output_dir / f"{artifact_stem}_{item.source}_evidence.png")
        except Exception as exc:
            failures[name] = str(exc)
        finally:
            provider_seconds[name] = round(time.monotonic() - start, 3)
    if not evidence:
        raise ValueError(f"Generic evidence pipeline produced no evidence: {failures}")
    fused, disagreement, fusion_metadata = fuse_evidence_maps(evidence)
    fused_path = output_dir / f"{artifact_stem}_fused_evidence.png"
    disagreement_path = output_dir / f"{artifact_stem}_source_disagreement.png"
    _save_float_mask(fused, fused_path)
    _save_float_mask(disagreement, disagreement_path)
    edge_refiner = None
    edge_aligned_path: Path | None = None
    if edge_refine:
        try:
            aligned = edge_align_field(Image.open(context.image_path).convert("RGB"), fused)
            edge_refiner = EdgeAwareRefiner(aligned)
            edge_aligned_path = output_dir / f"{artifact_stem}_edge_aligned_evidence.png"
            _save_float_mask(aligned, edge_aligned_path)
        except Exception as exc:  # refinement is additive; never block the pipeline
            failures["edge_refiner"] = str(exc)
            edge_refiner = None
    proposals = generate_generic_proposals(
        fused,
        disagreement,
        evidence,
        context.primary_region,
        foreground=context.foreground,
        min_area=min_component_area,
        sam_refiner=sam_refiner,
        edge_refiner=edge_refiner,
    )
    specialist_proposals: list[CandidateProposal] = []
    for mode, mask in (specialist_masks or {}).items():
        try:
            specialist_proposals.append(
                proposal_from_mask(
                    mode,
                    mask,
                    fused,
                    disagreement,
                    evidence,
                    foreground=context.foreground,
                    region=context.primary_region,
                )
            )
        except ValueError:
            continue
    proposals.extend(specialist_proposals)
    # Edge-snap the structural specialists too, so specialist geometry benefits
    # from the same boundary refinement as the generic candidate family.
    if edge_refiner is not None and specialist_proposals:
        proposals.extend(
            refine_proposals(
                edge_refiner,
                specialist_proposals,
                fused,
                disagreement,
                evidence,
                foreground=context.foreground,
                region=context.primary_region,
                prefix="edge",
            )
        )
    proposals.sort(key=lambda proposal: proposal.score, reverse=True)
    decision, predictions = selector.select(proposals)
    if decision.selected_mode is None:
        raise ValueError("Generic selector abstained without a usable positive core")
    selected = next(proposal for proposal in proposals if proposal.mode == decision.selected_mode)
    candidate_paths: dict[str, str] = {}
    candidate_scores: dict[str, float] = {}
    candidate_measurements: dict[str, dict[str, float]] = {}
    for proposal in proposals:
        path = output_dir / f"{artifact_stem}_{proposal.mode}_refined.png"
        _save_binary_mask(proposal.mask, path)
        candidate_paths[proposal.mode] = str(path)
        candidate_scores[proposal.mode] = float(proposal.score)
        candidate_measurements[proposal.mode] = dict(proposal.measurements)

    roles = posterior_mask_roles(fused, disagreement, evidence, selected, decision)
    variant_paths: dict[str, str] = {}
    for name, values in roles.items():
        path = variant_dir / f"{artifact_stem}_{name}.png"
        _save_float_mask(values, path, binary=name not in {"training_soft", "uncertainty_map", "inpaint_soft"})
        variant_paths[name] = str(path)

    box_path = output_dir / f"{artifact_stem}_box.png"
    box = Image.new("L", context.image_size, 0)
    left, top, right, bottom = context.primary_region
    ImageDraw.Draw(box).rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), fill=255)
    box.save(box_path)
    refined_path = output_dir / f"{artifact_stem}_refined.png"
    _save_binary_mask(selected.mask, refined_path)
    disposition = decision.disposition
    training_variant = "training_medium" if disposition == "hard_mask_ok" else "training_soft"
    prediction_rows = [asdict(item) for item in predictions]
    selected_prediction = next(item for item in prediction_rows if item["mode"] == selected.mode)
    selected_pixels = int(selected.mask.sum())
    region_area = max(1, (right - left) * (bottom - top))
    label_policy = {
        "label_policy": disposition,
        "quality_morphology": "generic_posterior",
        "adapter_training_mask_variant": training_variant,
        "selected_refinement": selected.mode,
    }
    return {
        "box_mask_path": str(box_path),
        "refined_mask_path": str(refined_path),
        "generation_core_mask_path": variant_paths["generation_core"],
        "benchmark_eval_mask_path": variant_paths["eval_mvtec"],
        "inpaint_mask_path": variant_paths["inpaint_soft"],
        "eval_mask_path": variant_paths["eval_tight"],
        "training_mask_path": variant_paths[training_variant],
        "uncertainty_mask_path": variant_paths["uncertainty_map"],
        "mask_variant_paths": variant_paths,
        "mask_variant_overlay_paths": {},
        "selected_refinement": selected.mode,
        "candidate_scores": candidate_scores,
        "candidate_qc": {mode: values for mode, values in candidate_measurements.items()},
        "candidate_measurements": candidate_measurements,
        "candidate_predictions": prediction_rows,
        "candidate_failures": failures,
        "candidate_modes": [proposal.mode for proposal in proposals],
        "candidate_refined_paths": candidate_paths,
        "candidate_heatmap_paths": {item.source: str(output_dir / f"{artifact_stem}_{item.source}_evidence.png") for item in evidence},
        "policy_scores": {row["mode"]: row["conformal_iou_lower_bound"] for row in prediction_rows},
        "selection_policy": "generic_calibrated_reliability",
        "selection_arbitration": {
            "applied": True,
            "reason": "highest_conformal_iou_lower_bound",
            "initial_selected_refinement": proposals[0].mode,
            "final_selected_refinement": selected.mode,
        },
        "selection_decision": asdict(decision),
        "scratch_morphology_class": "generic",
        "structure_profile": str(context.semantic_attributes.get("structure_profile", "unknown")),
        "candidate_rejections": {},
        "label_policy": label_policy,
        "parameters": {
            "kind": "generic_evidence",
            "architecture": "v3-generic-evidence",
            "fused_evidence_path": str(fused_path),
            "source_disagreement_path": str(disagreement_path),
            "provider_seconds": provider_seconds,
            "edge_refinement": {
                "enabled": bool(edge_refiner is not None),
                "edge_aligned_evidence_path": str(edge_aligned_path) if edge_aligned_path is not None else None,
                "edge_candidate_modes": [proposal.mode for proposal in proposals if proposal.mode.startswith("edge_")],
                "selected_is_edge_refined": bool(selected.mode.startswith("edge_")),
            },
            "fusion": fusion_metadata,
            "evidence": {
                item.source: {
                    "reliability": item.reliability,
                    "calibration": item.calibration,
                    "augmentation_consistency": item.augmentation_consistency,
                    "metadata": item.metadata,
                }
                for item in evidence
            },
            "selected_prediction": selected_prediction,
            "mask_variants": {name: {"purpose": _role_purpose(name)} for name in variant_paths},
        },
        "qc": {
            "status": "pass" if disposition == "hard_mask_ok" else "warning",
            "reasons": list(decision.reasons),
            "mask_area_fraction": selected_pixels / max(1, context.image_size[0] * context.image_size[1]),
            "mask_to_qwen_box_fraction": selected_pixels / region_area,
            "component_count": int(selected.measurements.get("component_count", 0.0)),
        },
    }


def _save_binary_mask(values: np.ndarray, path: Path) -> None:
    _save_float_mask(np.asarray(values, dtype=np.float32), path, binary=True)


def _save_float_mask(values: np.ndarray, path: Path, *, binary: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    if binary:
        values = values > 0.5
    Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255), mode="L").save(path)


def _role_purpose(name: str) -> str:
    return {
        "generation_core": "high-confidence generation core",
        "eval_tight": "conservative compatibility pseudo-mask",
        "eval_mvtec": "generic evaluation mask without benchmark-specific priors",
        "training_medium": "hard pseudo-label training support",
        "training_wide": "expanded hard training support",
        "training_soft": "uncertainty-weighted soft pseudo-label",
        "positive_core": "majority-supported high-confidence pixels",
        "possible_region": "broad plausible anomaly support",
        "uncertainty_map": "source, registration, and boundary disagreement",
        "inpaint_soft": "soft generation envelope",
    }.get(name, name)
