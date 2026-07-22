# Phase 9 Reliability-Max Plan

## Summary

Phase 8 showed that Qwen-guided placement plus adapter conditioning can improve
downstream segmentation, but the result is still fragile. The strongest row was
`fixed_mask_adapter:scratch_thin_detail` at a `0.25` synthetic ratio, while the
custom wood scuff case `000` remains the clearest mask-quality failure. Phase 9
therefore shifts from "more variants" to reliability controls:

- prevent the U-Net from learning Stable Diffusion seam or blur artifacts;
- increase real defect morphology diversity before synthetic mixing is raised;
- improve automatic scuff masks using frequency-domain texture suppression;
- replace one-run tiny U-Net evidence with repeated and stronger evaluators.

The user workflow remains automatic: normal images, defect images, and optional
text descriptions only. No manual boxes or masks are required.

## Current Evidence

Phase 8 generated `1080` Qwen-guided synthetic samples and evaluated `25`
segmentation rows.

| Result | Observation |
| --- | --- |
| Real-only baseline | AUROC `0.5833`, AUPRO `0.1834`, Dice `0.1369` |
| Best overall | `fixed_mask_adapter:scratch_thin_detail`, ratio `0.25`, AUROC `0.6866`, AUPRO `0.3185`, Dice `0.2045` |
| Best full hybrid | `full_qwen_hybrid:scratch_thin_detail`, ratio `0.25`, AUROC `0.6579`, Dice `0.1945` |
| Mask-only comparison | Full hybrid beats mask-only at ratio `0.50` for `scratch_ridge_balanced` |
| Clone harmonization | Useful but not the default; lower visibility and weaker AUROC/AUPRO |
| Custom wood `000` | Still weak because broad, low-contrast scuffs disrupt texture rather than forming a clean line |

The next improvement should not chase visual realism alone. The pipeline must
also prevent downstream models from exploiting synthetic artifacts unrelated to
the defect.

## Goals

1. Improve automatic masks for low-contrast multi-scuff defects, especially
   wood `000`, without degrading clear scratch cases `001` and `002`.
2. Make Phase 5 robust against SD inpainting artifacts by adding clean
   inpainted negative samples.
3. Expand morphology diversity so synthetic ratios above `0.25` are tested
   fairly instead of repeating the same few defects.
4. Add stronger and repeated downstream evaluation before making research
   claims.
5. Keep all artifacts under `v2/outputs/phase9_reliability` and
   `v2/reports/phase9_reliability`.

## Key Changes

### 1. Frequency-Domain Scuff Mask Candidate

Add a new auto-mask candidate, `fft_texture_suppression`, aimed at textured
materials such as wood.

The candidate should:

- run only inside the Qwen padded region;
- estimate dominant periodic texture frequencies with a 2D FFT;
- suppress the dominant wood-grain frequencies;
- invert back to a residual heatmap where scuffed or rubbed regions remain
  bright;
- combine the residual with `soft_patch` and `scuff_cluster` evidence;
- reject candidates with low residual contrast, excessive area, or strong
  border leakage.

This candidate should be preferred for `multi_scuff` morphology, especially
when the description includes terms such as `scuffed`, `rubbed`, `many faint
scratches`, or `worn patch`.

Expected custom-data behavior:

- `000`: improve broad scuff support while avoiding a giant blob.
- `001`: preserve the current clean single/vertical scratch result.
- `002`: preserve the current diagonal scratch-band result.

### 2. Symmetric Clean SD-Inpaint Negatives

Add a Phase 5 training option called `sd_clean_inpaint_negatives`.

For normal clean images:

1. sample random surface-valid boxes or masks;
2. run the same SD1.5 inpainting settings used by synthetic positive samples;
3. assign a zero anomaly mask;
4. include these images as normal examples during segmentation training.

Purpose:

- if SD creates a microscopic blur or harmonization signature, the segmenter
  sees that signature in normal images too;
- the model is forced to learn scratch/crack morphology instead of the SD
  patch boundary.

Diagnostics to report:

- false-positive rate on clean original images;
- false-positive rate on clean SD-inpainted negatives;
- predicted positive rate inside random inpainted boxes;
- downstream metrics with and without clean inpaint negatives.

### 3. Synthetic Morphology Diversity

Add controlled diversity before increasing synthetic ratios.

For clone/harmonized and source-defect variants:

- rotation jitter within a small range;
- elastic deformation for scratches/scuffs;
- local brightness, contrast, and color jitter;
- mask thickness jitter;
- random endpoint taper and small broken-stroke gaps.

For adapter-based variants:

- optional Gaussian noise on projected VLM tokens at inference;
- optional interpolation between compatible VLM token caches from the same
  defect class;
- keep the adapter gate active so token perturbation does not overpower CLIP
  prompt conditioning.

All diversity must be recorded in Phase 4 metadata:

- source defect id;
- geometry jitter parameters;
- color jitter parameters;
- token noise scale;
- token interpolation source, if used;
- generation seed.

### 4. Quality Filtering Before Phase 5

Phase 8 quality flags were too lenient. Add a `quality_filtered` Phase 5 mode
that can train only on synthetic samples passing stricter per-run thresholds.

Filtering should consider:

- defect visibility inside `eval_tight`;
- background preservation outside `inpaint_soft`;
- outside-refined changed-pixel fraction;
- inpaint-mask area fraction;
- obvious low-edit or over-edit cases;
- morphology-specific constraints, such as thinness for scratch bands and
  support coverage for multi-scuffs.

Reports must show:

- number of accepted/rejected samples per variant/profile/category;
- reject reasons;
- downstream metrics for unfiltered vs filtered synthetic pools.

### 5. Stronger Downstream Evaluation

Keep the current tiny U-Net as a fast scaffold, but stop treating it as final
evidence.

Add:

- repeated tiny U-Net runs with multiple seeds;
- a stronger segmentation model for supervised synthetic-augmentation
  evaluation;
- PatchCore or FastFlow as industrial anomaly-detection baselines.

Important caveat:

PatchCore is normally trained on normal images only, so it is not a direct
replacement for the supervised real-plus-synthetic U-Net experiment. Use it as
industrial evidence that synthetic generation does not damage normal-feature
modeling, and use supervised segmentation for the direct synthetic-defect
augmentation claim.

## Proposed Config

Create `configs/phase9_reliability.yaml` with:

```yaml
project:
  output_dir: outputs/phase9_reliability
  report_dir: reports/phase9_reliability

auto_masks:
  provider: qwen
  mask_refinement: auto
  auto_candidate_modes:
    - normal_anomaly
    - soft_patch
    - scuff_cluster
    - fft_texture_suppression
    - sam2_heatmap
    - residual
    - scratch_band_clean
    - multi_linear
    - pixel
    - procedural

phase4:
  variants:
    - qwen_mask_only
    - full_qwen_hybrid
    - fixed_mask_adapter
    - clone_harmonized
  seeds_per_record: 3
  diversity:
    enabled: true
    geometry_jitter: true
    color_jitter: true
    elastic_deform: true
    adapter_token_noise_std: [0.0, 0.01, 0.03]
  quality_filtering:
    write_diagnostics: true

phase5:
  synthetic_ratios: [0.0, 0.25, 0.5, 0.65]
  sd_clean_inpaint_negatives:
    enabled: true
    samples_per_clean: 1
    use_same_sd_settings_as_positive: true
  quality_filtered_training:
    enabled: true
  repeated_seeds: [1337, 2027, 3001]
  evaluators:
    - tiny_unet
    - stronger_unet
    - patchcore
```

## Milestones

### Phase 9.1: Scuff Mask Recovery

- Implement `fft_texture_suppression`.
- Integrate it into auto-mask candidate scoring.
- Add morphology policy so `multi_scuff` can prefer FFT/soft/scuff candidates
  while `scratch_band` keeps ridge-cleaning behavior.
- Rerun custom wood samples `000`, `001`, and `002`.
- Produce a side-by-side report comparing Phase 8 masks with Phase 9 masks.

### Phase 9.2: Artifact-Resistant Negatives

- Add clean SD-inpainted negative generation.
- Store them separately from synthetic positive anomalies.
- Train Phase 5 with and without these negatives.
- Report clean false-positive diagnostics.

### Phase 9.3: Diversity Controls

- Add source-defect geometry and color jitter.
- Add adapter token noise/interpolation as config-controlled options.
- Ensure fixed-mask comparisons still reuse identical masks and seeds.
- Report diversity parameters per generated image.

### Phase 9.4: Quality-Filtered Training

- Add strict synthetic quality filtering.
- Compare unfiltered vs filtered synthetic training at each ratio.
- Report accepted/rejected sample counts and reject reasons.

### Phase 9.5: Stronger Evaluation

- Add repeated-seed tiny U-Net results.
- Add one stronger supervised segmentation evaluator.
- Add PatchCore or FastFlow baseline report.
- Produce a final comparison table with mean and standard deviation.

## Test Plan

### Unit Tests

- `fft_texture_suppression` returns a non-empty non-rectangular mask for a
  synthetic periodic-texture scuff case.
- FFT candidate rejects all-zero, low-contrast, and overbroad residual maps.
- Multi-scuff policy prefers `fft_texture_suppression`, `soft_patch`, or
  `scuff_cluster` over `linear` when the description says scuffed/many
  scratches.
- Scratch-band policy still preserves clear line-like candidates for `001`
  and `002` style cases.
- Clean SD-inpaint negatives have zero masks and are labeled as normal in
  Phase 5.
- Quality filtering records accepted/rejected counts and reject reasons.
- Adapter token noise keeps conditioning shape `[1, 77 + K, 768]`.
- Fixed-mask comparisons preserve identical mask paths and seeds.

### Smoke Tests

Run on the custom wood set:

```bash
python -m iadgen_v2.cli auto-masks --config configs/auto_mask_smoke.yaml
```

Expected:

- `000` improves scuff coverage without selecting a single artificial line.
- `001` remains a clean vertical scratch.
- `002` remains a clean diagonal scratch band.
- The report writes a current contact sheet and candidate-score JSONL.

Run a one-sample Phase 9 generation/evaluation smoke:

```bash
python -m iadgen_v2.cli phase4-generate --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_reliability.yaml --provider qwen
```

Expected:

- positive synthetic images are written;
- clean SD-inpaint negatives are written;
- quality-filtered and unfiltered result rows are both present;
- all metadata records include diversity and quality-filter fields.

### Full Validation

- Rerun Qwen Phase 2-5 under `phase9_reliability`.
- Compare against Phase 8 best rows.
- Include repeated-seed statistics.
- Include custom wood `000/001/002` visual comparison.

## Acceptance Criteria

Phase 9 succeeds if:

1. custom wood `000` is visibly improved without degrading `001` and `002`;
2. clean SD-inpaint negatives reduce synthetic-artifact false positives;
3. filtered synthetic training improves or stabilizes AUROC/AUPRO/Dice versus
   unfiltered training;
4. full hybrid or fixed-mask adapter remains better than Qwen-mask-only at
   `0.25` or `0.5` synthetic ratio;
5. repeated-seed results are reported, not only a single lucky run;
6. reports clearly label automatic masks as pseudo-labels, not human ground
   truth.

## Priority Order

1. Implement `fft_texture_suppression` and prove the wood `000` mask improves.
2. Add clean SD-inpaint negatives and measure clean false positives.
3. Add diversity controls and quality filtering.
4. Add repeated/stronger evaluation.
5. Only then revisit higher synthetic ratios such as `0.65` or `0.8`.

