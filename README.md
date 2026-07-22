# IADGen v2

This directory contains the independent Phase 1 and Phase 2 foundation for the
VLM-guided surface-defect experiment described in `plan.md`. It writes
experiment assets only under `v2`; `v1` is a design reference, not a runtime
dependency. Model loading may read a shared Hugging Face cache outside this
directory.

For the current end-to-end automatic mask and synthetic-generation flow, see
`pipeline_v2_2_overview.md`. The short version: v2 can run without manual boxes
or masks, but automatically generated masks are pseudo-labels unless they are
officially or human verified.

For the next reliability-focused improvement plan after Phase 8, see
`phase9_reliability_plan.md`.

## Environment

The environment is intentionally created inside this directory:

```bash
conda env create --prefix .conda/envs/iadgen-v2 -f environment.yml
conda activate ./.conda/envs/iadgen-v2
```

## Phase 1 Commands

```bash
python -m iadgen_v2.cli feasibility --config configs/phase1_surface.yaml
python -m iadgen_v2.cli prepare --config configs/phase1_surface.yaml
python -m iadgen_v2.cli generate --config configs/phase1_surface.yaml --model mock
python -m iadgen_v2.cli evaluate --config configs/phase1_surface.yaml --model mock

# CUDA and cached/access-authorized model weights are required:
python -m iadgen_v2.cli generate --config configs/phase1_surface.yaml --model sd15
python -m iadgen_v2.cli evaluate --config configs/phase1_surface.yaml --model sd15
```

`prepare` downloads the configured MVTec AD categories into `data/mvtec_ad`
when absent, filters the experiment to `metal_nut/scratch`, `tile/crack`, and
`wood/scratch`, and writes a deterministic no-leakage split manifest below
`outputs/phase1_surface/prepared`.

The downloader attempts MVTec's advertised category links first. If those
links are unavailable, it downloads a public mirror of the original MVTec AD
archive, extracts only configured categories, records provenance in
`data/mvtec_ad/download_source.json`, and removes the large fallback archive
after extraction. MVTec AD remains subject to its CC BY-NC-SA 4.0 license.

The `mock` backend validates split, mask, metadata, and metrics plumbing
without making a realism claim. The `sd15` backend is the Phase 1 text-only
ControlNet inpainting baseline and intentionally has no VLM conditioning.
The configured baseline uses `placement_mode: surface_constrained`: placement
searches for mask locations covered by the product foreground for object
categories and avoids image borders for texture categories. `random_smoke`
placement remains available for plumbing tests only.

The checked-in config generates ten images per category for an initial
baseline batch. Lower `generation.samples_per_category` to `1` for quick
execution smoke validation.

## Split Rule

Real anomalies assigned to `held_out` are used only for final evaluation.
They are excluded from generation references, prompt examples, calibration,
and baseline source masks. Baseline insertion masks copy only adaptation-split
ground-truth morphology onto clean target images.

Generated metadata stores split and generation fingerprints. If data-split or
generation settings change, rerun `prepare` or `generate` rather than
combining stale artifacts with a new configuration.

## Custom Data With Auto Masks

For a small custom dataset, arrange images in the same MVTec-style layout used
by the rest of v2. The `auto-masks` command can bootstrap masks from defect
images by asking local Qwen for a rough box, then converting that box into a
description-guided refined mask:

```text
data/custom_mvtec/custom_part/
  train/good/*.png
  test/scratch/*.png
```

```bash
python -m iadgen_v2.cli auto-masks --config configs/custom_auto_masks_template.yaml
python -m iadgen_v2.cli prepare --config configs/custom_auto_masks_template.yaml --no-download
```

Edit `auto_masks.description_by_target` in the config before running. For
example, describe what you can see rather than drawing a box:

```yaml
auto_masks:
  description_by_target:
    custom_part/scratch: "a thin dark horizontal scratch near the center"
  mask_refinement: auto
```

`auto-masks` writes paired masks to
`ground_truth/<defect_type>/<image_stem>_mask.png` so `prepare` and later
phases can consume the dataset normally. It also writes Qwen boxes, refined
masks, inpaint masks, overlays, and a report under the configured
`outputs/.../auto_masks/qwen` and `reports/.../auto_masks/qwen` directories.
The default `auto` refinement starts with the normal-texture anomaly
pseudo-labeler: it fits robust local texture statistics from `train/good`,
scores defect-image pixels by how unusual they are inside the Qwen-proposed
region, suppresses components that look like repeated normal texture, and then
builds an adaptive inpaint envelope. Elongated evidence becomes a connected
scratch-band envelope, while scuffed or compact evidence remains a patch
envelope. It then falls back to SAM2/heatmap, pixel, residual, single-line
scratch, multi-line scratch, and procedural candidates. This keeps the user
workflow automatic: provide normal images, defect images, and an optional text
description; no manual boxes or masks are required.

Optional `dinov2_memory` and `dinov2_fusion` candidates are implemented for
research comparison, following the direction of AnomalyDINO and SuperAD:
extract patch tokens from normal `train/good` images, compare defect-image
patch tokens against that normal memory bank, and turn nearest-neighbor anomaly
distances into a heatmap. They are not enabled by default because DINO patch
maps can be too coarse for thin scratches and may produce blob-like masks. If
DINOv2 or SAM2 is unavailable, that candidate is clearly labeled as a fallback
in metadata rather than silently making a research claim.

For fine scratches, especially wood scratches, the generated masks are
pseudo-labels rather than true annotations. `auto-masks` flags suspicious masks
as `warning` when they are tiny, overly broad, fragmented, or otherwise
geometrically odd. Use the contact-sheet overlays in the report directory as a
quick quality check before relying on them for adapter training.
Each selected mask now also produces purpose-specific variants:
`eval_tight` is written to the MVTec-style `ground_truth` folder for split and
segmentation compatibility, while `training_medium`, `training_wide`, and
`inpaint_soft` are saved under the auto-mask output directory. The split
manifest records `training_mask_path`, and Phase 3 prefers that wider training
mask when it exists, so adapter training is not forced to use the tight
pseudo-label.
These masks are labeled `qwen_auto_refined_masks`; they are useful for
bootstrapping scarce data, but official or human masks are still preferred for
final research claims.

## Phase 2 Commands

Phase 2 currently has a runnable `heuristic` provider that validates the
spatial-grounding artifact flow without loading Qwen. It creates prompt
records, surface-valid region proposals, crude box masks, defect-specific
refined masks, inpaint masks, overlays, and placeholder `.pt` feature caches.
Those cache tensors are labeled explicitly as placeholders and are not Qwen
hidden states.

```bash
python -m iadgen_v2.cli phase2-propose --config configs/phase1_surface.yaml --provider heuristic
python -m iadgen_v2.cli phase2-evaluate --config configs/phase1_surface.yaml --provider heuristic
```

The current `qwen` provider intentionally fails fast until
`Qwen/Qwen2.5-VL-3B-Instruct`, `qwen-vl-utils`, and enough local storage are
available. The latest local check recorded that the Qwen model is not cached
and `qwen-vl-utils` is not installed.

The first heuristic run for `configs/phase1_surface.yaml` wrote 30 placement
records under `outputs/phase1_surface/phase2/heuristic` and a placement report
under `reports/phase1_surface/phase2/heuristic`. This is a plumbing and mask
compatibility milestone, not a VLM placement result.

## Phase 3 Commands

Phase 3 currently validates the offline-cache and gated-adapter contract. The
`heuristic` provider builds adaptation-only cache tensors, trains the projection
adapter against a surrogate shape objective, and verifies that SD1.5-style
`77 x 768` CLIP conditioning becomes `(77 + K) x 768` after appending gated
adapter tokens.

```bash
python -m iadgen_v2.cli phase3-cache --config configs/phase1_surface.yaml --provider heuristic
python -m iadgen_v2.cli phase3-train --config configs/phase1_surface.yaml --provider heuristic
python -m iadgen_v2.cli phase3-validate --config configs/phase1_surface.yaml --provider heuristic
```

The current run wrote 15 adaptation cache records, a checkpoint at
`outputs/phase1_surface/phase3/heuristic/checkpoints/adapter_surrogate.pt`, and
validation output under `reports/phase1_surface/phase3/heuristic`. This is not
yet diffusion denoising training and does not use real Qwen features.

## Phase 4 Commands

Phase 4 currently runs an integrated contract check: it consumes Phase 2
proposal metadata and cached prompt/image tokens, loads the Phase 3 adapter
checkpoint, appends projected tokens to `77 x 768` CLIP-shaped conditioning,
writes generated placeholder images, and records runtime/memory telemetry.

```bash
python -m iadgen_v2.cli phase4-generate --config configs/phase1_surface.yaml --provider heuristic
```

The current heuristic run wrote 30 records under
`outputs/phase1_surface/phase4/heuristic` and a profiling summary under
`reports/phase1_surface/phase4/heuristic`. This validates artifact integration
and CPU compute/reporting plumbing; it is not real Qwen-guided SD1.5 diffusion
inference. The current report also checks that pixels outside the inpaint mask
are preserved exactly by the heuristic renderer.

## Phase 5 Commands

Phase 5 currently runs a downstream segmentation scaffold using the fixed
adaptation and held-out split. It trains a small U-Net on real-only adaptation
data versus real-plus-Phase4-synthetic mixtures and evaluates only on held-out
real anomaly images.

```bash
python -m iadgen_v2.cli phase5-evaluate --config configs/phase1_surface.yaml --provider heuristic
```

The current heuristic run wrote
`reports/phase1_surface/phase5/heuristic/segmentation_results.csv` and
`summary.md`. This validates the no-leakage segmentation protocol and
real/synthetic ratio plumbing. The tiny CPU U-Net results are smoke-test
numbers, not final downstream evidence for the proposed Qwen/SD hybrid.
Phase 5 reports threshold-free AUROC/AUPRO and selects the IoU/Dice threshold
from adaptation samples only, never from held-out anomalies. The fixed `0.5`
threshold metrics remain in the CSV as collapse diagnostics.

## Phase 6 Real-Model Enablement

Phase 6 adds real `qwen` provider code paths while keeping `heuristic` for
smoke tests. Run the preflight first:

```bash
python -m iadgen_v2.cli phase6-preflight --config configs/phase1_surface.yaml
```

For a minimal real-model smoke run, use the dedicated one-sample-per-category
config:

```bash
python -m iadgen_v2.cli prepare --config configs/phase6_qwen_smoke.yaml
python -m iadgen_v2.cli phase6-preflight --config configs/phase6_qwen_smoke.yaml
python -m iadgen_v2.cli phase2-propose --config configs/phase6_qwen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase2-evaluate --config configs/phase6_qwen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase3-cache --config configs/phase6_qwen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase3-train --config configs/phase6_qwen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase3-validate --config configs/phase6_qwen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase4-generate --config configs/phase6_qwen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate --config configs/phase6_qwen_smoke.yaml --provider qwen
```

Implemented real-provider pieces:

- Phase 2 can preflight Qwen, request a normalized defect box, extract
  hidden-state tokens, and cache `[1, K, D]` tensors.
- Phase 3 can build Qwen adaptation caches and train the projection/gate
  adapter with SD1.5 inpainting denoising loss.
- Phase 4 can load the denoising adapter and run SD1.5 inpainting with
  appended Qwen adapter tokens.
- Phase 5 can consume `qwen` Phase 4 outputs once they exist.

Current workspace status: `qwen-vl-utils` is installed, CUDA/SD1.5 are ready,
and `Qwen/Qwen2.5-VL-3B-Instruct` is cached under
`model_cache/huggingface/hub`. The smoke config has completed Phase 2 through
Phase 5 with real Qwen token caches, SD1.5 denoising adapter training, real
hybrid inpainting, and tiny U-Net downstream evaluation. These are smoke
artifacts, not final research claims.

The latest smoke outputs are under `outputs/phase6_qwen_smoke`; reports are
under `reports/phase6_qwen_smoke`. The real Qwen caches use tensors shaped
`[1, 16, 2048]`, and the adapter validation reports combined conditioning
shape `[1, 93, 768]`.

## Phase 7 Ablation Commands

Phase 7 runs the moderate Qwen ablation study with shared masks/seeds across
`qwen_mask_only`, `full_qwen_hybrid`, and `fixed_mask_adapter`:

```bash
python -m iadgen_v2.cli prepare --config configs/phase7_qwen_ablation.yaml
python -m iadgen_v2.cli phase2-propose --config configs/phase7_qwen_ablation.yaml --provider qwen
python -m iadgen_v2.cli phase2-evaluate --config configs/phase7_qwen_ablation.yaml --provider qwen
python -m iadgen_v2.cli phase3-cache --config configs/phase7_qwen_ablation.yaml --provider qwen
python -m iadgen_v2.cli phase3-train --config configs/phase7_qwen_ablation.yaml --provider qwen
python -m iadgen_v2.cli phase3-validate --config configs/phase7_qwen_ablation.yaml --provider qwen
python -m iadgen_v2.cli phase4-generate --config configs/phase7_qwen_ablation.yaml --provider qwen --variant all
python -m iadgen_v2.cli phase5-evaluate --config configs/phase7_qwen_ablation.yaml --provider qwen
```

The current Phase 7 run wrote 45 generated samples per variant and a combined
segmentation comparison under `reports/phase7_qwen_ablation/phase5/qwen`.
