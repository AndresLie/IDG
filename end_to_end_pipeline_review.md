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
| `patchcore_guided_tiny_unet` real-only, latest sprint | `0.7832` | `0.3609` | `0.1761` |
| Best guided synthetic row: `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.7798` | `0.3646` | `0.1971` |
| `patchcore_distilled_tiny_unet` one-epoch real-only smoke | `0.3253` | `0.0407` | `0.0179` |
| Distillation ablation raw `tiny_unet` synthetic row, `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.4426` | `0.1832` | `0.0919` |
| Distillation ablation `patchcore_guided_tiny_unet` real-only | `0.7821` | `0.3575` | `0.1864` |
| Distillation ablation `patchcore_distilled_tiny_unet` synthetic row, ratio `0.50` | `0.4254` | `0.0407` | `0.0482` |
| Teacher-refined ablation synthetic row, `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.5562` | `0.1208` | `0.0464` |
| Teacher-refined capped ablation synthetic row, ratio `0.50` | `0.5722` | `0.0839` | `0.0550` |
| Teacher-refined ResNet18 U-Net random-weight smoke, real-only, one epoch | `0.6268` | `0.0984` | `0.0128` |
| Teacher-refined ResNet18 U-Net pretrained real-only row | `0.6425` | `0.2754` | `0.0772` |
| Teacher-refined ResNet18 U-Net pretrained synthetic row: `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.6691` | `0.2175` | `0.2174` |
| Teacher-refined ResNet18 U-Net area-only synthetic ablation | `0.6450` | `0.2340` | `0.2116` |
| Teacher-refined ResNet18 U-Net boundary+area synthetic ablation | `0.6195` | `0.2133` | `0.2092` |
| PatchCore-fused teacher-refined ResNet18 U-Net compact validation, real-only | `0.6793` | `0.3297` | `0.1766` |
| PatchCore-fused teacher-refined ResNet18 U-Net compact validation, `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.6509` | `0.3105` | `0.1772` |
| PatchCore-ResNet-fused teacher-refined ResNet18 U-Net validation, real-only | `0.6759` | `0.3330` | `0.1518` |
| PatchCore-ResNet-fused teacher-refined ResNet18 U-Net validation, `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.8456` | `0.5155` | `0.2061` |
| Calibrated PatchCore-ResNet-fused teacher-refined ResNet18 U-Net validation, real-only | `0.6903` | `0.3332` | `0.2117` |
| Calibrated PatchCore-ResNet-fused teacher-refined ResNet18 U-Net validation, `qwen_mask_only:scratch_thin_detail`, ratio `0.50` | `0.8413` | `0.5177` | `0.2766` |
| Current-best calibrated validation, real-only | `0.8101` | `0.4751` | `0.2378` |
| Current-best calibrated validation, `qwen_mask_only:scratch_thin_detail`, ratio `0.25` | `0.8256` | `0.5008` | `0.2491` |
| Current-best calibrated validation, `fixed_mask_adapter:scratch_thin_detail`, ratio `0.50` | `0.8301` | `0.4536` | `0.2520` |
| Repeated-seed mean, real-only | `0.7861` | `0.4419` | `0.2204` |
| Repeated-seed mean, `qwen_mask_only:scratch_thin_detail`, ratio `0.25` | `0.8107` | `0.4578` | `0.2370` |
| Repeated-seed mean, `fixed_mask_adapter:scratch_thin_detail`, ratio `0.25` | `0.8185` | `0.4743` | `0.2331` |
| `patchcore_resnet` normal-only, latest sprint | `0.9558` | `0.8549` | `0.4439` |

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

### Locked VisA Generalization Result

The later category-agnostic `v3-generic-evidence-rc2-operational-20260726`
architecture completed a governed 1,200-image VisA run and one sealed locked
evaluation. This is now the strongest evidence about auto-mask versatility:

| Locked VisA Metric | Result |
| --- | ---: |
| Category-macro Dice | `0.1593` |
| 95% category-bootstrap interval | `[0.0941, 0.2503]` |
| Macro precision | `0.1419` |
| Macro recall | `0.7488` |
| Macro search-region recall | `0.5898` |
| Macro pixel AP | `0.2085` |
| Macro AUPRO | `0.8717` |
| Accepted-mask coverage | `0.5267` |

Only chewing-gum exceeds `0.30` Dice. The architecture retains useful ranked
anomaly evidence but over-segments: selected masks cover `5.24%` of pixels on
average versus an estimated `0.96%` true defect area. Selector expected-IoU
Spearman correlation is only `0.0851`. Therefore the current generic auto-mask
path is not a versatile zero-shot mask generator, and generation Sprint 6 is
blocked until a new calibration-first architecture passes a new locked test.

### V4.1 Cross-Dataset Calibration Result

The first post-VisA improvement sprint is implemented and evaluated on 1,344
exposed development images using leave-dataset-out predictions. It learns a
category-agnostic pixel posterior from robust fused-score rank, local contrast,
soft Qwen spatial support, and stable normal-reference boundaries. A compact
component extractor and a broad-mask override guard are integrated but remain
disabled by default.

| V4.1 Development Row | Macro Dice | Macro Precision | Aggregate Area Ratio | MVTec Dice | VisA Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| V3 input masks | `0.3741` | `0.3647` | `3.1548` | `0.5889` | `0.1593` |
| Best area-compliant guarded posterior | `0.3506` | `0.3805` | `1.9240` | `0.5244` | `0.1769` |
| Non-regressing `8x` guard | `0.3674` | `0.3720` | `2.7630` | `0.5702` | `0.1645` |

The posterior materially reduces over-segmentation and slightly improves VisA,
but it does not preserve the stronger MVTec result while meeting the area gate.
V4.1 therefore fails promotion. Its main scientific result is that pixel-field
calibration is insufficient for safe arbitration; V4.2 must operate on real
candidate quality and localization reliability.

### V4.2 Candidate Calibration Result

V4.2 reconstructs category-agnostic candidate pools from retained fused fields
and applies dataset-excluded candidate calibration with a guarded V3 fallback.
It also builds an evidence-proposal localization envelope.

| Dataset | V3 Dice | V4.2 Dice | Oracle Dice | Spearman | MAE | Coverage | Search Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MVTec development | `0.5889` | `0.5786` | `0.6539` | `0.7481` | `0.0835` | `0.9194` | `0.9191` |
| VisA exposed | `0.1593` | `0.1593` | `0.3583` | `0.5650` | `0.0854` | `0.7601` | `0.9682` |

The candidate ceiling and localization recall are substantially better than the
selected-mask result, but the calibrator does not convert them into mask
utility. Global candidate ranking is reasonably calibrated while within-image
pair ranking fails to transfer: no VisA alternatives pass the safe gain guard.
V4.2 remains default-off and does not justify another locked run.

## Professional Assessment

### What Is Strong

The project has several research-grade design decisions:

- The user workflow remains convenient: normal images, defect images, and
  optional descriptions; no manual bbox or mask drawing is required.
- Qwen localization is used as a semantic search-region generator, not as a
  pixel-perfect mask oracle.
- Localization now has an automatic zero-shot policy router. It can keep
  generic cases on direct Qwen bbox localization, route bottle rim/surface
  defects through object-geometry priors, and route zipper tooth-chain defects
  through normal-guided Qwen verification while keeping zipper fabric-border
  defects on the direct border-localization path.
- Candidate selection is now category-aware for non-scratch industrial cases:
  bottle contamination prefers FFT/SAM2 texture candidates, bottle rim chips
  prefer normal-residual/SAM/PatchCore candidates, zipper tooth-chain defects
  prefer PatchCore/normal-residual candidates, and zipper fabric-border defects
  prefer SAM/PatchCore border support.
- The latest selector refinement adds physical validation gates: bottle
  `broken_small` uses a SAM2-vs-normal-residual gate to avoid oversized SAM
  masks, and zipper `fabric_border` penalizes tiny full-frame SAM masks before
  choosing the final candidate.
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
  - `patchcore_guided_tiny_unet` as a PatchCore-lite-prior-stabilized supervised diagnostic;
  - `patchcore_distilled_tiny_unet` as a PatchCore teacher-map distillation student;
  - `teacher_refined_tiny_unet` as a PatchCore-guided pseudo-label refinement student;
  - `teacher_refined_resnet18_unet` as a stronger skip-connected student with
    optional ImageNet-pretrained ResNet18 encoder;
  - `patchcore_guided_teacher_refined_resnet18_unet` as a teacher-refined
    ResNet18 student whose inference score is fused with a PatchCore prior;
  - fused-score calibration for PatchCore-guided ResNet18 rows, using
    quantile normalization, gamma sharpening, and an area-aware threshold
    objective to reduce over-segmentation;
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

The `teacher_refined_resnet18_unet` evaluator is now implemented and validated
with ImageNet-pretrained ResNet18 weights. The synthetic row passes the planned
acceptance target: Dice `0.2174`, predicted-positive rate `0.0536` against a
truth-positive rate of `0.0518`, and AUROC `0.6691`. This is a real improvement
over the capped `teacher_refined_tiny_unet` synthetic row, which had Dice
`0.0550` and predicted-positive rate `0.1378`.

Boundary/area regularizers are also implemented and test-covered, but the first
ablation does not justify enabling them by default. Area-only improves AUPRO
from `0.2175` to `0.2340`, but slightly lowers Dice and AUROC. Boundary+area
lowers the main metrics. This suggests the automatic pseudo-label boundary is
not yet reliable enough to supervise directly; the current best balanced
student remains the pretrained ResNet18 row without those extra losses.

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

The latest localization claim is:

```text
The auto-mask front end can now choose localization policy automatically for
wood, bottle, and zipper-style defects instead of relying only on manually
configured per-target bbox rules.
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
heatmap candidates when available, a PCA-aligned repeated-chain specialist for
zipper teeth, a polar rim-residual specialist for small bottle chips, and
geometry fallbacks.

The output includes:

```text
outputs/<exp>/auto_masks/qwen/metadata.jsonl
outputs/<exp>/auto_masks/qwen/current_run.json
outputs/<exp>/auto_masks/qwen/last_run_status.json
outputs/<exp>/auto_masks/qwen/runs/<run_id>/metadata.jsonl
outputs/<exp>/auto_masks/qwen/runs/<run_id>/masks/...
outputs/<exp>/auto_masks/qwen/runs/<run_id>/mask_variants/...
reports/<exp>/auto_masks/qwen/contact_sheet_mask_variants_current.png
reports/<exp>/auto_masks/qwen/contact_sheet_candidate_comparison.png
reports/<exp>/auto_masks/qwen/visual_result_comparison.png   # when before/after comparison is generated
```

Auto-mask generation is now checkpoint-safe. New artifacts are written inside
an immutable run directory, interrupted runs retain partial diagnostics without
replacing the previous stable metadata, and `overwrite: false` carries existing
records into the next completed manifest.

Selector-only changes can reuse cached immutable candidates:

```bash
python -m iadgen_v2.cli auto-masks-reselect --config configs/<config>.yaml
```

This reruns policy scoring and production arbitration, rebuilds all
purpose-specific variants, and publishes a new transactional checkpoint without
rerunning Qwen, SAM, or PatchCore.

Latest MVTec bottle/zipper pilot results after the structural upgrade:

| Scope | Previous Dice | Current Dice |
| --- | ---: | ---: |
| Bottle overall | `0.7092` | `0.7361` |
| Bottle small chip | `0.6092` | `0.6899` |
| Zipper overall | `0.5648` | `0.6229` |
| Zipper tooth defects | `0.5719` | `0.6393` |

The repeated-chain production gate activates for two of six tooth samples. The
polar rim specialist is selected only for the smallest bottle chip, while
credible broader SAM/consensus masks remain selected for larger rim damage.

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
| `patchcore_guided_tiny_unet` | supervised tiny-U-Net score fused with a PatchCore-lite normal-memory prior |
| `patchcore_distilled_tiny_unet` | tiny-U-Net trained with an additional PatchCore teacher-map loss |
| `teacher_refined_tiny_unet` | tiny-U-Net trained on pseudo-labels refined by confident PatchCore teacher regions |
| `teacher_refined_resnet18_unet` | ResNet18 U-Net student trained on PatchCore-refined pseudo-labels |
| `patchcore_guided_teacher_refined_resnet18_unet` | teacher-refined ResNet18 student fused with a PatchCore prior at scoring time |
| calibrated fused ResNet18 | fused ResNet18 score with quantile/gamma calibration and area-aware thresholding |
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

## External-Domain Wood Pilot

A compact real production-line wood pilot has now been imported from the
Kodytek et al. wood surface defect dataset:

```text
data/kodytek_wood_pilot
```

The pilot contains:

```text
18 clean wood crops
4 crack crops
4 resin crops
4 knot-with-crack crops
```

The source is CC BY 4.0. Published bounding boxes were used only for
deterministic crop selection and weak-reference auditing. They were not placed
in the MVTec-style `ground_truth` folder and were not supplied to Qwen or the
automatic mask candidate ensemble.

The pipeline completed:

```text
external record import
-> Qwen localization
-> automatic candidate ensemble
-> purpose-specific mask variants
-> no-leakage split preparation
-> Phase 10 mask-policy ablation
-> weak-box visual audit
```

Current result:

| Item | Result |
| --- | ---: |
| Qwen valid localization | `11 / 12` |
| Full-image localization fallback | `1 / 12` |
| Auto-mask QC pass | `11 / 12` |
| Auto-mask QC warning | `1 / 12` |
| Auto-mask QC reject | `0 / 12` |

Weak-box diagnostics:

| Defect | Mask pixels inside published box | Published box covered by mask |
| --- | ---: | ---: |
| `crack` | `1.0000` | `0.1312` |
| `knot_with_crack` | `0.8959` | `0.1220` |
| `resin` | `0.8226` | `0.1347` |

These are not segmentation metrics because the source annotations are
rectangular boxes. They indicate that the automatic masks are usually compact
and contained within the weak defect region.

The external data revealed two architecture gaps:

- Qwen full-frame localization previously aborted the whole batch. The
  auto-mask runner now supports an explicit recorded full-image fallback.
- Phase 10 previously grouped samples only by filename, which collapsed rows
  when different defect folders reused names such as `000.png`. It now uses
  `category/defect_type/filename` sample identities.

The main domain-transfer failure is resin near dark conveyor or image borders.
The next mask upgrade should add clean-normal border suppression and richer
morphology classes such as `linear_crack`, `resin_streak`, and
`compact_knot_crack`.

Review artifacts:

```text
reports/kodytek_wood_pilot/external_data_review/kodytek_wood_pilot_review.md
reports/kodytek_wood_pilot/external_data_review/kodytek_wood_pilot_visual_review.png
reports/kodytek_wood_pilot/external_data_review/weak_box_audit.csv
reports/kodytek_wood_pilot/phase10_mask_quality/qwen/mask_quality_ablation_report.md
```

## MVTec Bottle And Zipper Prompt-Stress Test

A second external-domain test now evaluates automatic masks against official
MVTec AD pixel masks for `bottle` and `zipper`.

The benchmark copy is intentionally isolated:

```text
data/mvtec_bottle_zipper_auto_mask
```

Official masks are stored only under:

```text
data/mvtec_bottle_zipper_auto_mask/official_reference_masks
```

They are not supplied to Qwen or the auto-mask generator. The auto-mask command
writes pseudo-labels into the isolated MVTec-style `ground_truth` folder, then
the review script compares those pseudo-labels against the official reference
masks afterward.

Prompt adjustments were important:

- bottle prompts tell Qwen to ignore cap, reflections, transparency,
  background, and normal rim highlights;
- zipper prompts tell Qwen to ignore intact repeating teeth and normal woven
  fabric texture, and to focus on the tooth chain or fabric border depending
  on defect type.

Current subset:

```text
bottle: broken_large, broken_small, contamination
zipper: broken_teeth, fabric_border, split_teeth
3 samples per defect type
18 total defect samples
```

Latest bbox/localization router result:

| Category / defect | Policy | Old BBox Recall | New BBox Recall | Old BBox IoU | New BBox IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| `bottle/broken_large` | `bottle_rim_guided_qwen` | `0.8412` | `1.0000` | `0.1644` | `0.1901` |
| `bottle/broken_small` | `bottle_rim_guided_qwen` | `0.7449` | `1.0000` | `0.2990` | `0.0606` |
| `bottle/contamination` | `bottle_surface_guided` | `0.6628` | `1.0000` | `0.3815` | `0.3871` |
| `zipper/broken_teeth` | `normal_guided_qwen_verify` | `0.0140` | `0.9416` | `0.0084` | `0.0567` |
| `zipper/fabric_border` | `qwen_bbox` | `0.8728` | `0.6756` | `0.1466` | `0.1214` |
| `zipper/split_teeth` | `normal_guided_qwen_verify` | `0.3333` | `1.0000` | `0.0087` | `0.0886` |
| **Bottle overall** | mixed auto policy | `0.7497` | `1.0000` | `0.2816` | `0.2126` |
| **Zipper overall** | mixed auto policy | `0.4067` | `0.8724` | `0.0545` | `0.0889` |

This is a bbox/search-region result, not final segmentation. It shows that the
router closes the severe zipper tooth-chain localization failure while
preserving a separate direct-Qwen route for zipper fabric-border defects. The
main tradeoff is broader search boxes, especially for small bottle chips, so
downstream candidate refinement remains necessary.

Current selected-mask result from the official-mask review:

| Category / defect | Qwen Recall | Dice | Precision | Recall | Oracle Candidate Dice |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bottle/broken_large` | `1.0000` | `0.4151` | `0.5425` | `0.3388` | `0.4175` |
| `bottle/broken_small` | `1.0000` | `0.4822` | `0.3471` | `0.8639` | `0.5634` |
| `bottle/contamination` | `1.0000` | `0.7254` | `0.7551` | `0.7042` | `0.7413` |
| `zipper/broken_teeth` | `0.9416` | `0.5379` | `0.4313` | `0.8685` | `0.5379` |
| `zipper/fabric_border` | `0.6653` | `0.3115` | `0.4969` | `0.3035` | `0.3315` |
| `zipper/split_teeth` | `1.0000` | `0.5697` | `0.5371` | `0.7310` | `0.6121` |
| **Bottle overall** | `1.0000` | `0.5409` | `0.5483` | `0.6356` | `0.5740` |
| **Zipper overall** | `0.8690` | `0.4730` | `0.4885` | `0.6343` | `0.4939` |

Interpretation:

- Auto-routed localization now works strongly for this subset. Bottle reaches
  `1.0000` bbox recall and zipper remains useful at `0.8690` bbox recall.
- The category-aware selector closes most of the oracle gap. Bottle
  contamination improves from `0.0399` to `0.7254` Dice, bottle `broken_small`
  improves to `0.4822` Dice, and bottle overall improves from `0.1863` to
  `0.5409` Dice.
- Zipper tooth-chain masks also improve: `broken_teeth` reaches `0.5379` Dice
  and `split_teeth` reaches `0.5697` Dice.
- The remaining weakness is now narrower: `zipper/fabric_border` still has
  reduced bbox recall, but selection regret is down to `0.0201`.

This test identifies the next professional refinements:

```text
fabric-border routing/selector tuning and small-chip SAM-vs-normal residual
selection, not broad bbox or generic ensemble changes
```

Review artifacts:

```text
reports/mvtec_bottle_zipper_qwen_bbox_v2/qwen_bbox_eval/qwen/qwen_bbox_prompt_v2_review.md
reports/mvtec_bottle_zipper_qwen_bbox_v2/qwen_bbox_eval/qwen/qwen_bbox_visual_review.png
reports/mvtec_bottle_zipper_auto_mask/official_mask_evaluation/bottle_zipper_auto_mask_review.md
reports/mvtec_bottle_zipper_auto_mask/official_mask_evaluation/bottle_zipper_auto_mask_visual_review.png
reports/mvtec_bottle_zipper_auto_mask/official_mask_evaluation/auto_mask_metrics.csv
reports/mvtec_bottle_zipper_auto_mask/auto_masks/qwen/summary.md
```

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
reports/phase9_resnet18_patchcore_fusion_validation/phase5/qwen/summary.md
reports/phase9_resnet18_patchcore_fusion_validation/phase5/qwen/segmentation_results.csv
reports/phase9_resnet18_patchcore_resnet_fusion_validation/phase5/qwen/summary.md
reports/phase9_resnet18_patchcore_resnet_fusion_validation/phase5/qwen/segmentation_results.csv
reports/phase9_resnet18_patchcore_resnet_fusion_validation/phase5/qwen/result_review.md
reports/phase9_resnet18_patchcore_resnet_fusion_validation/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/phase9_resnet18_patchcore_resnet_calibrated_validation/phase5/qwen/summary.md
reports/phase9_resnet18_patchcore_resnet_calibrated_validation/phase5/qwen/segmentation_results.csv
reports/phase9_resnet18_patchcore_resnet_calibrated_validation/phase5/qwen/result_review.md
reports/phase9_resnet18_patchcore_resnet_calibrated_validation/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/phase9_current_best_calibrated_validation/phase5/qwen/summary.md
reports/phase9_current_best_calibrated_validation/phase5/qwen/segmentation_results.csv
reports/phase9_current_best_calibrated_validation/phase5/qwen/result_review.md
reports/phase9_current_best_calibrated_validation/phase5/qwen/variant_ratio_arbitration_report.md
reports/phase9_current_best_calibrated_validation/phase5/qwen/variant_ratio_arbitration.csv
reports/phase9_current_best_calibrated_validation/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/phase9_current_best_repeated_seed_validation/phase5/qwen/summary.md
reports/phase9_current_best_repeated_seed_validation/phase5/qwen/segmentation_results.csv
reports/phase9_current_best_repeated_seed_validation/final_research_evidence_report.md
reports/phase9_current_best_repeated_seed_validation/evidence_manifest.json
reports/phase9_current_best_repeated_seed_validation/auto_mask_evidence_linkage.md
reports/phase9_current_best_repeated_seed_validation/phase5/qwen/repeated_seed_result_review.md
reports/phase9_current_best_repeated_seed_validation/phase5/qwen/variant_ratio_arbitration_report.md
reports/phase9_current_best_repeated_seed_validation/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png
reports/auto_mask_phase9_wood/auto_masks/qwen/visual_result_comparison.png
reports/auto_mask_phase9_wood/auto_masks/qwen/auto_mask_upgrade_review.md
reports/auto_mask_phase9_wood/phase10_mask_quality/qwen/mask_quality_ablation_report.md
```

## Final Judgment

The completed locked VisA evaluation supersedes the earlier architecture-level
optimism for general zero-shot use. The pipeline remains a credible and well-
governed research prototype, but its current generic auto-mask output is not
release-ready: macro Dice is `0.1593`, precision is `0.1419`, and all five
predeclared generalization gates fail. High AUPRO does not rescue the broad,
poorly calibrated binary masks.

The pipeline infrastructure is strong enough to support serious controlled
experiments, and the targeted Phase 9 run proves the real Qwen/SD1.5 path can
execute through generation, selection, evaluation, and visual reporting. This
engineering readiness must not be read as evidence that the current generic
auto-mask architecture generalizes.

The current auto-mask generator is also meaningfully better than before:
`000` is treated as an uncertainty-aware soft pseudo-label instead of a fake
hard mask, while `001` and `002` remain hard line-like scratch cases.

The remaining gap is evaluator reliability. The tiny-U-Net scaffold is useful
for smoke testing, but its calibration is too weak for final research claims.
The uncertainty-aware Phase 3/5 loss path is now implemented and test-covered.
The TF-IDG-lite generation critic is also implemented and has identified
under-edited wood masks as the main synthetic-generation failure. The
critic-to-selector bridge is implemented and test-covered. Phase 5 now includes
two PatchCore-stabilized supervised paths: `patchcore_guided_tiny_unet`, which
fuses inference scores with a PatchCore-lite normal-memory prior, and
`patchcore_distilled_tiny_unet`, which trains the tiny-U-Net student with an
additional confidence-gated PatchCore teacher-map loss. The guided row is
already validated. The distilled row is implemented and test-covered, and a
one-epoch real-only smoke proves the architecture runs. A later controlled
ablation confirms the current direct teacher-map distillation recipe is still
weak: it does not approach `patchcore_lite`, while `patchcore_guided_tiny_unet`
remains stable. The latest `teacher_refined_tiny_unet` sprint converts teacher
scores into pseudo-label refinement targets and improves AUROC over direct
distillation. The capped refinement patch improves Dice slightly and reduces
over-mask behavior slightly, but the model still predicts too broadly and
remains below raw synthetic tiny-U-Net on Dice. The evidence now suggests the
remaining bottleneck is not only target construction; the tiny-U-Net student is
too weak. The next student upgrade is now validated:
`teacher_refined_resnet18_unet` with ImageNet weights improves the synthetic
row to Dice `0.2174` with predicted-positive rate `0.0536`, nearly matching the
held-out truth-positive rate `0.0518`. The subsequent PatchCore-fused ResNet18
validation improves the same student's ranking metrics in a compact run:
real-only AUROC/AUPRO/Dice moves from `0.5726`/`0.1841`/`0.1577` to
`0.6793`/`0.3297`/`0.1766`, and the synthetic ratio `0.50` row moves from
`0.5934`/`0.2340`/`0.1075` to `0.6509`/`0.3105`/`0.1772`.
The stronger PatchCore-ResNet teacher confirms the direction more clearly:
the fused synthetic row reaches AUROC `0.8456`, AUPRO `0.5155`, and Dice
`0.2061`, substantially above the unfused synthetic row
(`0.4761`/`0.1462`/`0.0331`).
The calibrated fused-score run improves segmentation usability further:
AUROC/AUPRO/Dice for the synthetic ratio `0.50` row becomes
`0.8413`/`0.5177`/`0.2766`, while predicted-positive rate drops from `0.1095`
to `0.0816`.
The latest current-best focused run is more conservative but more stable across
variants: `qwen_mask_only` at ratio `0.25` gives the best AUPRO (`0.5008`) and
`fixed_mask_adapter` at ratio `0.50` gives the best Dice (`0.2520`), while
real-only remains competitive (`0.2378` Dice). This means synthetic
augmentation helps, but variant/ratio selection is now the main remaining
evidence issue.
The new arbitration layer turns that observation into an explicit policy:
`qwen_mask_only` ratio `0.25` and `fixed_mask_adapter` ratio `0.50` are
recommended, while `qwen_mask_only` ratio `0.50` is demoted because it inflates
predicted area without Dice gain.
The repeated-seed validation makes the final recommendation more conservative:
ratio `0.25` is the stable synthetic setting. `fixed_mask_adapter` ratio `0.25`
has the best mean AUROC/AUPRO (`0.8185`/`0.4743`), while `qwen_mask_only` ratio
`0.25` has the best mean Dice (`0.2370`). Ratio `0.50` no longer looks
research-safe because it loses AUROC/AUPRO on average.

```text
keep teacher_refined_resnet18_unet as the supervised student path, keep
patchcore_guided_teacher_refined_resnet18_unet as the ranking-stabilized path,
keep fused-score calibration enabled for that path, then validate stability
with ratio `0.25` as the conservative synthetic default before making the final
research claim
```

The best current characterization is:

```text
research-grade execution and governance architecture implemented;
uncertainty-aware auto-mask path implemented and visually validated;
uncertainty-aware training/evaluation loss path implemented and test-covered;
mask-quality ablation runner implemented for hard/vote-soft/calibrated-soft policy comparison;
TF-IDG-lite generation critic implemented and run on wood/scratch samples;
TF-IDG-lite critic-to-selector bridge implemented and test-covered;
patchcore-guided tiny-U-Net evaluator implemented, tested, and run;
patchcore-distilled tiny-U-Net evaluator implemented and test-covered;
teacher-refined tiny-U-Net evaluator implemented, tested, and ablated;
teacher-refined ResNet18 U-Net evaluator implemented and pretrained-validated;
PatchCore-guided teacher-refined ResNet18 U-Net evaluator implemented, tested, and validated with PatchCore-lite and PatchCore-ResNet teachers;
fused-score calibration implemented, tested, and validated;
variant/ratio arbitration report implemented, tested, and run on current-best validation;
repeated-seed current-best validation run and reviewed;
final repeated-seed evidence package created with a compact artifact manifest;
auto-mask evidence co-located by reference for wood, bottle, and zipper;
boundary/area regularizers implemented, tested, and kept as experimental switches;
locked VisA generalization evaluated and rejected at 0.1593 macro Dice;
generation promotion blocked by the failed mask gate;
next architecture must be calibration-first and validated leave-dataset-out
before one evaluation on a new untouched benchmark.
```
