# V2.2 Pipeline Overview

This document describes the current `v2` end-to-end defect-mask and synthetic-generation pipeline. The system is automatic and designed for scarce-data workflows, but its masks are pseudo-labels unless they come from official or human annotation.

## 1. Defect Localization

For custom data, `auto-masks` sends each defect image plus an optional user description to `Qwen/Qwen2.5-VL-3B-Instruct`. Qwen returns a rough `bbox_xyxy`, confidence, defect label, and short evidence. The folder name remains the trusted defect label.

The Qwen box is a search region, not a final mask. It may be padded or clipped according to config, then passed to candidate generators.

## 2. Mask Candidate Generation

The pipeline produces multiple candidate masks inside the Qwen region. Current candidates include evidence-based anomaly maps, soft patch masks, SAM/SAM2 heatmap masks, residual/pixel masks, line candidates, procedural fallbacks, optional DINOv2 memory/fusion candidates, and optional `delta_deno`.

`delta_deno` and DINOv2 candidates are research options, not guaranteed winners. They are selected only when configured and when QC metadata shows that they are valid.

## 3. Scoring And Selection

Each candidate receives QC metadata and a policy score. The selector favors evidence-based masks for multiple scratches or scuffed patches, favors line/ridge masks for long scratches, and penalizes masks that are empty, overly broad, fragmented, blob-like, or strongly off-policy for the description.

When `ensemble_consensus` is available, it is treated as one candidate among others, not an automatic winner. The metadata records candidate scores, QC warnings, rejection reasons, and selected policy.

## 4. Purpose-Specific Mask Variants

The selected mask is exported into several variants:

| Variant | Use |
| --- | --- |
| `eval_tight` | Tight pseudo-label for split/evaluation compatibility |
| `training_medium` | Phase 3 adapter reconstruction target |
| `training_wide` | Wider training/edit ablation |
| `inpaint_soft` | Soft SD inpainting envelope |

`inpaint_soft` is intentionally wider than `eval_tight`. It gives diffusion room to blend boundaries, while `eval_tight` avoids labeling too much background as defect.

## 5. Synthetic Generation Ablations

Phase 4 now supports these Qwen variants:

| Variant | Meaning |
| --- | --- |
| `qwen_mask_only` | Qwen/refined mask with SD1.5 text conditioning only |
| `full_qwen_hybrid` | Qwen/refined mask with CLIP plus gated projected Qwen tokens |
| `fixed_mask_adapter` | Same masks/seeds as mask-only, adapter tokens enabled |
| `clone_harmonized` | Real adaptation defect cloned with OpenCV `NORMAL_CLONE`, then low-strength SD harmonization |

The clone path is an ablation, not a guaranteed improvement. It can preserve defect geometry, but it may transfer source-background artifacts or mismatch target material.

## 6. Quality Diagnostics

Every Phase 4 row records:

- background preservation outside `inpaint_soft`,
- changed-pixel fraction inside the inpaint mask,
- defect visibility inside `eval_tight` / refined mask,
- changed-pixel fraction outside the refined mask,
- inpaint-mask area fraction,
- a simple generation quality score,
- quality flags such as `weak_visible_defect`, `high_background_change`, `high_outside_refined_change`, `overbroad_inpaint_mask`, and `low_mask_edit_fraction`.

Phase 4 also writes `quality_contact_sheet.png` per variant. It shows clean image, generated image, mask overlay, and difference heatmap for high/low ranked examples.

## 7. Downstream Evaluation

Phase 5 evaluates real-only, SD/text-only, full hybrid, fixed-mask adapter, and clone-harmonized outputs with identical held-out splits. Synthetic ratios should be treated as a tuning variable. In current runs, lower ratios such as `0.25` or `0.5` are safer than heavy synthetic replacement.

The tiny U-Net Phase 5 report is a decision scaffold, not a final research claim. Final claims should use stronger segmentation training, repeated seeds, and official or human-verified masks when possible.
