# V2 Auto-Mask Generation Process

This document describes the current automatic mask-generation path used by the `v2` project. The masks are automatic pseudo-labels for bootstrapping scarce industrial defect data. They are not human ground truth, and final research claims should still prefer official or manually verified masks when available.

## Goal

The user provides an MVTec-style dataset with normal images in `train/good` and a few defect images in `test/<defect_type>`. The pipeline automatically creates masks under `ground_truth/<defect_type>` so Phase 3 adapter training and Phase 5 segmentation evaluation can run without hand-drawn masks or user-drawn bounding boxes.

## Step 1: Qwen BBox Localization

`Qwen/Qwen2.5-VL-3B-Instruct` reads each defect image plus the optional user description and returns strict JSON with:

- `bbox_xyxy`
- `defect_type`
- `confidence`
- `evidence`

The defect folder name remains authoritative. For example, an image under `test/scratch` is treated as `scratch` even if Qwen reports another label. The Qwen box is used only as a search region; raw rectangles are not used as final segmentation masks except as debug artifacts.

## Step 2: Candidate Mask Generation

When `mask_refinement: auto`, the pipeline evaluates the configured `auto_candidate_modes` in order. The default reliable set is:

```yaml
auto_candidate_modes: [normal_anomaly, soft_patch, scuff_cluster, multi_scuff_fusion, fft_texture_suppression, sam2_heatmap, residual, scratch_band_clean, multi_linear, pixel, procedural]
```

Each candidate produces a refined binary mask, soft inpaint mask, QC metadata, and a score. The final mask is the highest-scoring valid candidate, not simply the first non-empty candidate.

Candidate roles:

- `normal_anomaly`: compares the defect crop against available normal images and is usually the best first choice for real scratches on textured surfaces.
- `soft_patch`: keeps multiple scratch/scuff evidence as a broader local damage region when the description says there are many scratches or a scuffed area.
- `scuff_cluster`: merges nearby scratch/scuff evidence into a patch-aware pseudo-label for multi-scuff cases.
- `multi_scuff_fusion`: preferred for difficult scuffs; uses FFT as a broad support map, intersects it with fine evidence from `normal_anomaly` and `soft_patch`, rejects grain-aligned clutter, and caps area so the result does not become a filled blob.
- `fft_texture_suppression`: suppresses dominant periodic texture frequencies inside the Qwen region, then masks the residual disruption. This is aimed at wood-like low-contrast scuffs where the defect breaks grain periodicity rather than forming a clean dark line.
- `sam2_heatmap`: uses heatmap-derived point prompts with SAM/SAM2 when available.
- `residual`, `scratch_band_clean`, `pixel`, `multi_linear`, `linear`: lower-cost or geometry-specific alternatives for contrast-driven or line-like defects.
- `procedural`: final fallback that creates a plausible scratch/crack-shaped mask inside the Qwen box.

`delta_deno` is an optional research candidate. It is not in the default list because it is heavier and can fail when cross-attention does not localize the target token clearly. If enabled, it must pass heatmap validity checks before it can be selected.

## Step 3: QC And Scoring

Every candidate is checked for empty masks, excessive area, many components, image-border contact, and scratch geometry. Scratch masks are penalized when they become large blobs or cover too much of the Qwen box. Descriptions such as `multiple scratches`, `many fine scratches`, or `scuffed area` favor `normal_anomaly` and `soft_patch` over single-line procedural masks.

Metadata records:

- selected refinement mode
- all candidate scores
- candidate QC
- candidate failures or rejection reasons
- final mask paths and overlays

This is important because bad masks should be diagnosable from the JSONL report instead of only from visual inspection.

## Step 4: Purpose-Specific Mask Outputs

The selected binary mask is exported into separate forms:

- `eval_tight`: tight pseudo-label used for generated `ground_truth/*_mask.png` and downstream segmentation evaluation when no human masks exist.
- `training_medium`: moderately dilated mask used by Phase 3 adapter training.
- `inpaint_soft`: wider, feathered mask used for Stable Diffusion inpainting to reduce hard boundaries.

These variants intentionally serve different purposes. A good inpaint mask is usually wider and softer than a good evaluation pseudo-label.

## Reliability Notes

Automatic masks are useful for a `31 normal + 3 defect` workflow, but they are still approximations. The most reliable operating mode is:

1. provide a short defect description for each target or image;
2. run `auto-masks`;
3. inspect overlays for the few real defect images;
4. use the selected masks for adapter training and synthetic generation;
5. avoid making final benchmark claims unless masks are human-verified or official.

## Phase 4 Quality Feedback

The synthetic-generation stage now records quality diagnostics for every generated image:

- defect visibility inside the refined/evaluation mask;
- background preservation outside the inpaint mask;
- changed-pixel fraction outside the refined mask;
- inpaint-mask area fraction;
- quality flags for weak defects, overbroad masks, excessive background changes, and low edit strength.

Each Phase 4 variant also writes a `quality_contact_sheet.png` with clean image, generated image, mask overlay, and difference heatmap panels. Use this sheet to decide whether mask refinement, prompt profile, or synthetic ratio should be adjusted.

`clone_harmonized` is available as a comparison variant. It clones a real adaptation defect into the selected target region and then runs low-strength SD harmonization. This can preserve structure, but it is still an ablation because source-background artifacts can transfer with the cloned patch.
