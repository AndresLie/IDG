# V2 End-To-End Pipeline Review

## Executive Review

The current `v2` system is a credible scarce-data industrial defect generation
prototype. It has moved beyond one-off mask heuristics into a closed-loop
research pipeline:

```text
normal images + few defect images
-> automatic pseudo-mask generation
-> morphology-aware mask policy
-> Qwen/SD1.5 synthetic generation
-> synthetic quality selection
-> downstream evaluation
-> visual evidence report
```

The strongest historical quantitative result still comes from the Phase 8/10
tiny-U-Net evaluation:

| Row | Pixel AUROC | AUPRO | Dice |
| --- | ---: | ---: | ---: |
| Real-only baseline | `0.5833` | `0.1834` | `0.1369` |
| Best synthetic row: `fixed_mask_adapter:scratch_thin_detail`, ratio `0.25` | `0.6866` | `0.3185` | `0.2045` |
| Best full hybrid: `full_qwen_hybrid:scratch_thin_detail`, ratio `0.25` | `0.6579` | `0.1583` | `0.1945` |

This remains promising: synthetic augmentation improved the real held-out
tiny-U-Net result in the best historical row. However, it is not final research
evidence.

Since that result, Phase 9 has been partially executed with real Qwen/SD1.5
artifacts:

| Phase 9 Targeted Row | Pixel AUROC | AUPRO | Dice |
| --- | ---: | ---: | ---: |
| `tiny_unet` real-only targeted baseline | `0.3430` | `0.0518` | `0.0984` |
| Best targeted synthetic row: `fixed_mask_adapter:scratch_thin_detail`, ratio `0.25` | `0.5895` | `0.0338` | `0.1187` |
| Best targeted AUPRO row: `fixed_mask_adapter:scratch_thin_detail`, ratio `0.50` | `0.4280` | `0.1006` | `0.1179` |
| `patchcore_lite` normal-only | `0.7828` | `0.3557` | `0.1761` |

The latest TF-IDG-inspired generation critic also ran on existing Phase 9
wood/scratch synthetic samples:

| Phase 11 Wood Critic Row | Value |
| --- | ---: |
| Generated wood/scratch records scored | `540` |
| Accepted by TF-IDG-lite thresholds | `74` |
| Mean TF-IDG-lite score | `0.7452` |
| Best mean group | `clone_harmonized:scratch_ridge_balanced`, `0.7843` |
| Dominant reject reason | `low_mask_coverage`, `466` samples |

The component-level diagnosis is important:

| Phase 11 Component | Accepted Avg | Rejected Avg |
| --- | ---: | ---: |
| Feature alignment | `0.9125` | `0.9010` |
| Adaptive mask coverage | `0.6721` | `0.0691` |
| Texture preservation | `0.8993` | `0.9034` |
| Leakage score | `0.9813` | `0.9920` |

The accepted and rejected sets are similarly reference-aligned and similarly
texture-safe. The main gap is mask occupancy: rejected samples are usually too
clean inside the selected defect region.

The targeted run proves that the Phase 2-5 loop can execute end-to-end with
fresh artifacts and visual reports. It does not yet prove the final research
claim because the tiny-U-Net scaffold remains poorly calibrated and the full
Phase 9 matrix was too broad for an interactive run.

## Professional Assessment

### What Is Strong

The project has several research-grade design decisions:

- The user workflow remains convenient: normal images, defect images, and
  optional descriptions; no manual bbox or mask drawing is required.
- Qwen localization is used as a semantic search-region generator, not as a
  pixel-perfect mask oracle.
- The pipeline now separates mask purposes:
  - `eval_tight` for pseudo-evaluation and MVTec compatibility;
  - `training_medium` for hard line-like defect training;
  - `training_soft` for uncertainty-aware fuzzy scuff/scratch training;
  - `inpaint_soft` for diffusion blending and broad/fuzzy scuff training.
- Broad low-contrast scuffs such as wood `000` are treated as `soft_mask_only`,
  which is technically more honest than forcing a crisp binary boundary. The
  upgraded auto-mask path now trains these cases with `training_soft` by default
  instead of using the inpainting mask as the training target.
- The latest auto-mask upgrade adds `structure_tensor_ridge`, Qwen sub-box
  parsing/fusion, candidate-agreement uncertainty maps, and purpose-specific
  `positive_core` / `possible_region` outputs.
- The latest wood auto-mask run also adds `patchcore_guided`, a clean-normal
  patch-memory candidate. It selects wood `000`, while `001` and `002` remain
  ensemble hard-mask scratch-band cases.
- Multi-scuff `training_soft` is now produced by a calibrated heatmap ensemble
  when available, combining PatchCore-guided, nearest-normal residual, FFT
  texture, and normal-anomaly evidence instead of relying only on binary
  candidate votes.
- Candidate score caps prevent warning-quality masks from being reported as
  perfect `1.0` candidates.
- The latest training upgrade uses `uncertainty_map` as a low-weight region in
  Phase 3 denoising loss and Phase 5 tiny-U-Net BCE/Dice loss, instead of
  treating every automatic pseudo-label pixel as equally certain.
- Phase 4 records generation diagnostics: visibility, background preservation,
  leakage, inpaint area, quality flags, and quality score.
- Phase 5 now supports stronger evidence paths:
  - `tiny_unet` for supervised synthetic-augmentation smoke testing;
  - `patchcore_lite` as a local normal-memory baseline;
  - `patchcore_resnet` as a WideResNet50-2 PatchCore-style evaluator.
- A synthetic selector now filters and ranks generated samples before Phase 5
  training.
- A TF-IDG-lite Phase 11 critic now ranks generated samples with transparent
  training-free diagnostics: visual reference alignment, adaptive mask
  coverage, texture preservation outside the mask, leakage, and morphology fit.
- Visual reports expose incomplete evidence instead of hiding it. Missing
  prediction panels are shown as missing until Phase 5 is rerun with the new
  prediction-output hooks.

### What Is Still Weak

The main weakness is not architecture anymore. The main weakness is still
evidence quality.

The targeted Phase 9 visual report now shows:

```text
mask -> generation -> leakage -> segmentation prediction
```

but the supervised tiny-U-Net rows are unstable. Several rows predict far too
many positive pixels, so the model can appear to improve one metric while
remaining weakly calibrated.

For wood `000`, the auto-mask policy is now much better aligned with the real
defect type: it uses `soft_mask_only` and `training_soft`. The latest code now
also propagates `uncertainty_mask_path` into the prepared split and uses
`uncertainty_map` as a low-weight loss region in Phase 3 and Phase 5 when
`uncertainty_loss_weight` is below `1.0`.

The full Phase 9 matrix still has not completed. A compact targeted run has
completed and should be treated as evidence of pipeline function, not final
research proof.

The Phase 11 wood critic reveals a specific generation weakness: most rejected
wood samples fail because the generated defect does not occupy enough of the
selected mask. That is more actionable than a generic "bad generation" label.
It points toward adaptive SD strength, regeneration, or selector filtering when
`adaptive_mask_coverage_score` is low.

There is also one interpretation caveat: `ip_adapter_hybrid` ties
`clone_harmonized` in the current critic output, but the targeted config has
IP-Adapter disabled. That row should be read as fallback/harmonized behavior,
not proof that true IP-Adapter visual injection is already improving the result.

### Current Interpretation

The correct claim today is:

```text
The pipeline can automatically produce uncertainty-aware morphology-specific
pseudo-labels, generate Qwen/SD1.5 synthetic defects, select synthetic samples,
and evaluate them with visual prediction evidence.
```

The strongest careful interpretation is:

```text
Historical Phase 8/10 results suggest selected synthetic augmentation can help;
the fresh Phase 9 targeted run confirms the architecture executes end-to-end
but also shows tiny-U-Net is not reliable enough for final claims.
```

The claim that is not yet proven is:

```text
The final Phase 9 uncertainty-aware mask plus synthetic-selector pipeline
improves held-out segmentation under stronger evaluators such as
patchcore_resnet and DRAEM.
```

## End-To-End Pipeline

### Inputs

The expected dataset layout is MVTec-style:

```text
dataset_root/
  category_or_custom_part/
    train/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png
```

For custom scarce-data use, the user supplies:

- normal images in `train/good`;
- a few real defect images in `test/<defect_type>`;
- optional text descriptions.

The `ground_truth` masks can be generated automatically by `auto-masks`.

### Stage 1: Auto-Mask Generation

Command:

```bash
python -m iadgen_v2.cli auto-masks --config configs/<config>.yaml
```

Qwen localizes each real defect and returns a rough bbox. If Qwen returns
multiple sub-boxes, the pipeline validates and fuses them into the search
region. That region is used only as a search fence. Candidate mask methods then
run inside the region, including normal-reference anomaly scoring,
PatchCore-style clean-normal patch memory scoring, soft patch generation, scuff
fusion, FFT texture suppression, structure-tensor ridge detection, SAM/SAM2
heatmap candidates when available, and geometry fallbacks.

The output includes:

```text
outputs/<exp>/auto_masks/qwen/metadata.jsonl
outputs/<exp>/auto_masks/qwen/masks/...
outputs/<exp>/auto_masks/qwen/mask_variants/...
reports/<exp>/auto_masks/qwen/contact_sheet_mask_variants_current.png
reports/<exp>/auto_masks/qwen/contact_sheet_candidate_comparison.png
reports/<exp>/auto_masks/qwen/visual_result_comparison.png   # when before/after comparison is generated
```

For current wood samples:

```text
000 -> patchcore_guided, soft_mask_only, training uses training_soft
001 -> hard_mask_ok, training uses training_medium
002 -> hard_mask_ok, training uses training_medium
```

For `multi_scuff`, `training_soft` can now be built from calibrated heatmap
evidence rather than only from candidate-mask voting. In the current wood run,
`000` uses PatchCore-guided, nearest-normal residual, FFT texture, and
normal-anomaly heatmaps to produce a broader soft training target while keeping
`eval_tight` compact.

Each automatic mask now also writes uncertainty-aware variants:

```text
eval_tight
training_medium
training_wide
training_soft
positive_core
possible_region
uncertainty_map
inpaint_soft
```

`uncertainty_map` marks candidate disagreement and weak pseudo-label boundary
regions. These are pseudo-labels, not human ground truth.

Latest wood comparison artifacts:

```text
reports/auto_mask_phase9_wood/auto_masks/qwen/contact_sheet_mask_variants_current.png
reports/auto_mask_phase9_wood/auto_masks/qwen/contact_sheet_candidate_comparison.png
reports/auto_mask_phase9_wood/auto_masks/qwen/visual_result_comparison.png
reports/auto_mask_phase9_wood/auto_masks/qwen/visual_result_comparison.md
reports/auto_mask_phase9_wood/phase10_mask_quality/qwen/mask_quality_ablation_report.md
reports/auto_mask_phase9_wood/phase10_mask_quality/qwen/mask_quality_ablation_contact_sheet.png
reports/auto_mask_phase9_wood/phase5/qwen/mask_policy_validation.md
```

The latest custom wood mask-policy validation confirms the intended split:

```text
soft_mask_only / multi_scuff -> training_soft: 1
hard_mask_ok / scratch_band -> training_medium: 2
status: pass for all 3 custom wood rows
```

### Stage 2: Prepare Splits

Command:

```bash
python -m iadgen_v2.cli prepare --config configs/<config>.yaml
```

The split manifest records:

- adaptation real defect samples;
- held-out real defect samples;
- clean target images;
- `mask_path`;
- `eval_mask_path`;
- `training_mask_path`;
- `uncertainty_mask_path`, when available;
- `label_policy`.

This preserves no-leakage discipline: held-out anomalies are not used for
generation references, prompt examples, adapter training, or calibration inputs.

### Stage 3: Qwen Feature Cache And Adapter Training

Commands:

```bash
python -m iadgen_v2.cli phase3-cache --config configs/<config>.yaml --provider qwen
python -m iadgen_v2.cli phase3-train --config configs/<config>.yaml --provider qwen
```

Phase 3 caches Qwen visual/spatial tokens offline. The adapter then projects
Qwen tokens into SD1.5 cross-attention width while preserving CLIP prompt
conditioning.

Backbones are frozen:

- Qwen frozen;
- CLIP text encoder frozen;
- VAE frozen;
- U-Net frozen;
- only adapter and gate parameters train.

For `soft_mask_only` samples, Phase 3 preserves grayscale alpha instead of
binarizing the mask. This matters for fuzzy scuffs. Phase 3 can now down-weight
ambiguous pseudo-label areas by resizing `uncertainty_map` to the SD latent
resolution and applying it to the denoising MSE. Phase 9 configs currently set:

```yaml
phase3:
  uncertainty_loss_weight: 0.25
```

### Stage 4: Synthetic Generation

Command:

```bash
python -m iadgen_v2.cli phase4-generate --config configs/<config>.yaml --provider qwen
```

Supported variants include:

- `qwen_mask_only`;
- `full_qwen_hybrid`;
- `fixed_mask_adapter`;
- `clone_harmonized`;
- `ip_adapter_hybrid`;
- `latent_blend_harmonized`.

Generation is morphology-aware through quality profiles:

| Morphology | Preferred Profile |
| --- | --- |
| `multi_scuff` | `scuff_soft_low_strength` |
| `scratch_band` | `scratch_thin_detail` or `scratch_ridge_balanced` |
| `single_stroke` | `single_stroke_clean` |
| `crack_band` | `scratch_thin_detail` or `scratch_ridge_balanced` |

Phase 4 writes:

```text
outputs/<exp>/phase4/qwen/<variant>/metadata.jsonl
reports/<exp>/phase4/qwen/<variant>/summary.md
reports/<exp>/phase4/qwen/<variant>/quality_contact_sheet.png
```

Each row stores:

- generation seed;
- variant and quality profile;
- conditioning shape;
- mask paths;
- background preservation;
- mask changed fraction;
- defect visibility;
- outside-refined change;
- inpaint area;
- quality flags;
- generation quality score.

### Stage 4.5: TF-IDG-Lite Generation Critic

Command:

```bash
python -m iadgen_v2.cli phase11-tfidg-critic --config configs/<config>.yaml --provider qwen
```

This stage is inspired by TF-IDG's training-free industrial defect generation
principle. It does not train a new model. Instead, it scores generated Phase 4
samples using:

- visual alignment to real defect exemplars from the adaptation split;
- adaptive changed-pixel coverage inside the selected mask;
- texture preservation outside the mask;
- leakage around the refined mask;
- morphology fit for scratch-like versus scuff-like defects.

It writes:

```text
reports/<exp>/phase11_tfidg_critic/qwen/tfidg_lite_report.md
reports/<exp>/phase11_tfidg_critic/qwen/tfidg_lite_metrics.csv
reports/<exp>/phase11_tfidg_critic/qwen/tfidg_lite_metrics.jsonl
reports/<exp>/phase11_tfidg_critic/qwen/tfidg_lite_contact_sheet.png
```

Current wood/scratch result:

```text
540 generated records scored
74 accepted
466 rejected for low_mask_coverage
best mean group: clone_harmonized:scratch_ridge_balanced
```

The result review is:

```text
wood generation is texture-safe and reference-aligned,
but too many samples are under-edited inside the mask.
```

Accepted samples average `0.6721` adaptive mask coverage, while rejected
samples average only `0.0691`. Texture preservation and leakage are both strong
in accepted and rejected groups, so the immediate issue is not seam damage. It
is weak defect visibility/occupancy inside the selected pseudo-mask.

This critic is now connected to Phase 5 selection, so training can prefer
generated defects that both look reference-aligned and actually occupy the
intended mask.

For wood/scratch, the recommended rule is:

```text
if adaptive_mask_coverage_score < 0.35:
  reject or regenerate with higher strength / stronger defect prompt
```

### Stage 5: Synthetic Selector

The synthetic selector runs inside Phase 5 before synthetic mixing.

It performs:

```text
generate -> score -> reject bad samples -> keep top samples per group
```

Current Phase 9 selector config:

```yaml
synthetic_selector:
  enabled: true
  max_per_group: 24
  group_by: [category, defect_type, morphology, variant, quality_profile]
  min_defect_visibility_score: 0.015
  max_background_l1: 0.05
    max_outside_refined_change_fraction: 0.25
    max_inpaint_area_fraction: 0.20
    reject_quality_flags: true
    tfidg_lite:
      enabled: true
      require_metrics_file: true
      min_adaptive_mask_coverage_score: 0.35
      selector_score_weight: 0.35
```

It writes:

```text
reports/<exp>/phase5/<provider>/synthetic_selector_report.md
reports/<exp>/phase5/<provider>/synthetic_selector_report.jsonl
reports/<exp>/phase5/<provider>/mask_policy_validation.md
```

This is a key research improvement because Phase 5 no longer blindly trains on
every synthetic image. The latest implementation connects the Phase 11
TF-IDG-lite critic into this selector: generated samples with
`adaptive_mask_coverage_score < 0.35` are rejected when critic metrics are
available, and `tfidg_lite_score` contributes to selector ranking.

### Stage 6: Downstream Evaluation

Command:

```bash
python -m iadgen_v2.cli phase5-evaluate --config configs/<config>.yaml --provider qwen
```

Phase 5 evaluates:

- real-only adaptation data;
- real plus selected synthetic data;
- multiple synthetic ratios;
- repeated seeds;
- clean SD-inpaint negative examples when enabled;
- morphology-level metrics.

For supervised tiny-U-Net rows, Phase 5 now supports uncertainty-aware BCE/Dice:
`positive_core` and confident background keep full weight, while pixels marked
by `uncertainty_map` are reduced to the configured loss weight. Phase 9 configs
currently set:

```yaml
phase5:
  uncertainty_loss_weight: 0.25
```

Evaluators:

| Evaluator | Role |
| --- | --- |
| `tiny_unet` | supervised synthetic-augmentation smoke scaffold |
| `patchcore_lite` | local normal-memory baseline with handcrafted patch features |
| `patchcore_resnet` | WideResNet50-2 PatchCore-style normal-memory evaluator |

Metrics:

- pixel AUROC;
- AUPRO;
- IoU;
- Dice;
- predicted positive rate;
- truth positive rate;
- fixed-threshold diagnostics;
- morphology-specific metrics.

Phase 5 now also writes prediction contact sheets:

```text
reports/<exp>/phase5/<provider>/prediction_examples/.../prediction_contact_sheet.png
```

These are required for the final visual evidence report.

### Stage 7: Visual Evidence Report

Command:

```bash
python -m iadgen_v2.cli phase9-visual-report --config configs/<config>.yaml --provider qwen
```

The report combines:

```text
input/background
selected mask
training-used mask
generated image
leakage map
segmentation prediction
```

Current generated visual reports:

```text
reports/phase9_reliability_targeted/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/phase10_visual_ablation/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/auto_mask_phase9_wood/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
```

Current limitation: the targeted Phase 9 visual report contains prediction
evidence, but the full Phase 9 matrix has not completed. Older visual reports
may still show missing prediction panels if they were generated before Phase 5
prediction contact sheets were added.

### Stage 8: Mask Quality Ablation

Command:

```bash
python -m iadgen_v2.cli phase10-mask-quality --config configs/<config>.yaml --provider qwen
```

Phase 10 compares automatic mask-training policies before committing to a full
generation/evaluation run:

```text
hard_binary -> crisp eval/selected pseudo-label
vote_soft -> binary candidate-agreement soft label
calibrated_soft -> current heatmap-calibrated soft label
```

The critic is diagnostic, not human ground truth. It measures heatmap support,
candidate agreement, area sanity, fragmentation, soft-label entropy, and grain
alignment risk. The current wood ablation shows hard masks remain best for
clear scratch-band cases `001` and `002`, while `000` is close between hard,
vote-soft, and calibrated-soft policies. That means the soft-scuff path is
reasonable but still needs downstream Phase 5 validation.

## Research-Grade Acceptance Criteria

The pipeline should be considered research-credible only when a fresh Phase 9
run shows:

1. selected synthetic data improves real held-out tiny-U-Net metrics over
   real-only;
2. `patchcore_resnet` does not reveal obvious artifact-driven failure;
3. morphology-specific results are reported, not only aggregate metrics;
4. `multi_scuff` cases use soft labels rather than fake crisp masks;
5. visual evidence shows reasonable generation, low leakage, and meaningful
   segmentation prediction overlays;
6. reports clearly label automatic masks as pseudo-labels, not human ground
   truth.

## Recommended Command Sequences

### Full Phase 9 Research Run

From `v2/`:

```bash
python -m iadgen_v2.cli auto-masks --config configs/phase9_reliability.yaml
python -m iadgen_v2.cli prepare --config configs/phase9_reliability.yaml
python -m iadgen_v2.cli phase2-propose --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase2-evaluate --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase3-cache --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase3-train --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase4-generate --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase11-tfidg-critic --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase9-visual-report --config configs/phase9_reliability.yaml --provider qwen
```

This is the complete intended run, but Phase 5 is broad and expensive.

### Targeted Evidence Run

The compact targeted run already completed and is the current practical
evidence artifact:

```bash
python -m iadgen_v2.cli phase11-tfidg-critic --config configs/phase9_reliability_targeted.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_reliability_targeted.yaml --provider qwen
python -m iadgen_v2.cli phase9-visual-report --config configs/phase9_reliability_targeted.yaml --provider qwen
```

The most important files to review now are:

```text
reports/phase9_reliability_targeted/execution_report.md
reports/phase9_reliability_targeted/phase5/qwen/summary.md
reports/phase9_reliability_targeted/phase5/qwen/segmentation_results.csv
reports/phase9_reliability_targeted/phase5/qwen/synthetic_selector_report.md
reports/phase9_reliability_targeted/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/phase9_reliability_targeted/phase11_tfidg_critic/qwen/tfidg_lite_report.md
reports/phase9_reliability_targeted/phase11_tfidg_critic/qwen/tfidg_lite_result_review.md
reports/phase9_reliability_targeted/phase11_tfidg_critic/qwen/tfidg_lite_contact_sheet.png
reports/phase9_reliability_targeted/wood_visual_result/wood_end_to_end_visual_result.png
reports/phase9_reliability_targeted/wood_visual_result/wood_end_to_end_visual_result.md
reports/auto_mask_phase9_wood/auto_masks/qwen/visual_result_comparison.png
reports/auto_mask_phase9_wood/auto_masks/qwen/auto_mask_upgrade_review.md
reports/auto_mask_phase9_wood/phase10_mask_quality/qwen/mask_quality_ablation_report.md
```

## Final Judgment

The current architecture is strong enough to support a serious experiment, and
the targeted Phase 9 run proves the real Qwen/SD1.5 pipeline can execute through
generation, selection, evaluation, and visual reporting.

The current auto-mask generator is also meaningfully better than before:
`000` is treated as an uncertainty-aware soft pseudo-label instead of a fake
hard mask, while `001` and `002` remain hard line-like scratch cases.

The remaining gap is evaluator reliability. The tiny-U-Net scaffold is useful
for smoke testing, but its calibration is too weak for final research claims.
The uncertainty-aware Phase 3/5 loss path is now implemented and test-covered.
The TF-IDG-lite generation critic is also implemented and has identified
under-edited wood masks as the main synthetic-generation failure. The
critic-to-selector bridge is implemented and test-covered, but a fresh Phase 5
rerun requires regenerating Phase 4 outputs because the existing Phase 4
metadata fingerprint is stale relative to the active Phase 2/3/4 config. The
next research-grade step is a fresh validation run with:

```text
regenerated Phase 4 outputs, uncertainty-aware Phase 3/5 losses enabled,
TF-IDG-lite filtering active in the Phase 5 selector, then patchcore_resnet
and/or DRAEM used as stronger final evidence
```

The best current characterization is:

```text
research-grade pipeline architecture implemented;
uncertainty-aware auto-mask path implemented and visually validated;
uncertainty-aware training/evaluation loss path implemented and test-covered;
mask-quality ablation runner implemented for hard/vote-soft/calibrated-soft policy comparison;
TF-IDG-lite generation critic implemented and run on wood/scratch samples;
TF-IDG-lite critic-to-selector bridge implemented and test-covered;
final research evidence pending stronger evaluator validation.
```
