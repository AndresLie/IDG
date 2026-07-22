# Auto-Mask Generation Checkpoint

Checkpoint date: `2026-06-07`  
Workspace: `/home/p76147019/ImgGen/v2`  
Status: `implementation checkpoint; final post-gate research run pending`

## Executive State

The automatic pseudo-mask pipeline is implemented end to end for the current
wood defect study:

```text
normal images + few defect images
-> Qwen search-region localization
-> multi-method pixel evidence
-> candidate scoring and fusion
-> morphology-aware label policy
-> eval / training / uncertainty / inpaint masks
-> uncertainty-aware Phase 3 and Phase 5 losses
-> synthetic generation
-> TF-IDG-lite criticism
-> Phase 5 synthetic selection
-> downstream evaluation and visual reporting
```

The current architecture no longer assumes that one binary mask can serve every
purpose. It produces separate pseudo-label roles and routes them by defect
morphology.

## Current Wood Mask Result

| Image | Selected method | Morphology | Policy | Training mask | QC |
| --- | --- | --- | --- | --- | --- |
| `000.png` | `patchcore_guided` | `multi_scuff` | `soft_mask_only` | `training_soft` | warning |
| `001.png` | `ensemble_consensus` | `scratch_band` | `hard_mask_ok` | `training_medium` | pass |
| `002.png` | `ensemble_consensus` | `scratch_band` | `hard_mask_ok` | `training_medium` | pass |

Phase 5 manifest validation passes all current rows:

```text
soft_mask_only / multi_scuff -> training_soft: 1
hard_mask_ok / scratch_band -> training_medium: 2
validation status: 3/3 pass
```

## Implemented Features

### Automatic Mask Generation

- Qwen bounding-box and sub-box localization used as a search fence.
- Normal-reference residual and anomaly candidates.
- PatchCore-style clean-normal patch-memory candidate.
- FFT texture evidence and structure-tensor ridge evidence.
- Multi-scuff fusion, SAM/SAM2 hooks, geometry fallbacks, and candidate fusion.
- Candidate score caps for warning-quality masks.
- Morphology-aware hard/soft label routing.

### Purpose-Specific Outputs

- `eval_tight`
- `training_medium`
- `training_wide`
- `training_soft`
- `positive_core`
- `possible_region`
- `uncertainty_map`
- `inpaint_soft`

### Uncertainty-Aware Learning

- Phase 3 latent denoising loss uses `uncertainty_map`.
- Phase 5 tiny-U-Net BCE/Dice loss uses `uncertainty_map`.
- Current uncertainty loss weight is `0.25` in both phases.
- Ambiguous pixels are down-weighted rather than treated as certain labels.

### Generation Quality Control

- TF-IDG-lite critic scores reference alignment, adaptive mask coverage,
  outside-mask texture preservation, leakage, and morphology fit.
- Phase 5 selector reads TF-IDG-lite metrics.
- Samples with `adaptive_mask_coverage_score < 0.35` are rejected.
- TF-IDG-lite score contributes to selector ranking with weight `0.35`.

### Reporting

- Auto-mask variants and candidate comparison sheets.
- Wood before/after visual comparison.
- Mask-policy ablation report.
- Mask-policy manifest validation.
- Prediction contact sheets.
- Wood end-to-end visual result.
- Detailed 23-slide auto-mask progress presentation.

## Verified Results

### Focused Regression Suite

Run on `2026-06-07`:

```text
68 passed in 12.44s
```

Covered:

- automatic mask generation;
- uncertainty-aware Phase 3 loss;
- uncertainty-aware Phase 5 loss;
- mask-policy validation;
- Phase 10 mask-quality ablation;
- Phase 11 TF-IDG-lite critic;
- TF-IDG-lite Phase 5 selector decisions.

### Historical Phase 9 Targeted Metrics

These metrics were produced before the final TF-IDG-lite selector bridge was
enabled and therefore remain historical baselines:

| Evaluator / row | Pixel AUROC | AUPRO | Dice |
| --- | ---: | ---: | ---: |
| tiny-U-Net real-only | `0.3430` | `0.0518` | `0.0984` |
| fixed-mask adapter, ratio `0.25` | `0.5895` | `0.0338` | `0.1187` |
| fixed-mask adapter, ratio `0.50` | `0.4280` | `0.1006` | `0.1179` |
| PatchCore-lite normal-only | `0.7828` | `0.3557` | `0.1761` |

### TF-IDG-Lite Diagnostic

| Item | Result |
| --- | ---: |
| Generated wood/scratch records scored | `540` |
| Accepted by critic thresholds | `74` |
| Rejected for low mask coverage | `466` |
| Mean score | `0.7452` |
| Best mean group | `clone_harmonized:scratch_ridge_balanced`, `0.7843` |

Interpretation:

```text
The dominant generation failure is under-editing inside the intended mask,
not excessive texture damage or leakage.
```

### Mask-Policy Ablation

| Image | Diagnostic winner |
| --- | --- |
| `000.png` | `vote_soft`, narrowly |
| `001.png` | `hard_binary` |
| `002.png` | `hard_binary` |

This supports morphology-specific routing, although the automatic critic is not
human ground truth.

## Important Limitations

1. The full Phase 9 matrix has not completed after enabling the TF-IDG-lite
   selector gate.
2. The existing targeted Phase 5 report predates the gate and must not be cited
   as a post-gate result.
3. Tiny-U-Net remains poorly calibrated and is a smoke-test evaluator.
4. PatchCore-resNet and DRAEM-level validation remain pending.
5. Wood `000` remains a warning-quality soft pseudo-label with uncertain
   boundaries.
6. The `ip_adapter_hybrid` critic row currently reflects fallback behavior when
   true IP-Adapter injection is disabled.
7. Automatic masks are pseudo-labels, not official human ground truth.

## Resume Sequence

Run from `/home/p76147019/ImgGen/v2`.

### Fast Post-Gate Validation

```bash
python -m iadgen_v2.cli phase11-tfidg-critic \
  --config configs/phase9_reliability_targeted.yaml --provider qwen

python -m iadgen_v2.cli phase5-evaluate \
  --config configs/phase9_reliability_targeted.yaml --provider qwen

python -m iadgen_v2.cli phase9-visual-report \
  --config configs/phase9_reliability_targeted.yaml --provider qwen
```

Confirm in the new selector report:

```text
TF-IDG-lite gate enabled: true
minimum adaptive mask coverage: 0.35
rejections include low_tfidg_mask_coverage
```

### Full Research Run

```bash
python -m iadgen_v2.cli auto-masks \
  --config configs/phase9_reliability.yaml
python -m iadgen_v2.cli prepare \
  --config configs/phase9_reliability.yaml
python -m iadgen_v2.cli phase2-propose \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase2-evaluate \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase3-cache \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase3-train \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase4-generate \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase11-tfidg-critic \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate \
  --config configs/phase9_reliability.yaml --provider qwen
python -m iadgen_v2.cli phase9-visual-report \
  --config configs/phase9_reliability.yaml --provider qwen
```

## Acceptance Criteria For The Next Checkpoint

- TF-IDG-lite gate is visible in a newly generated selector report.
- Selected synthetic augmentation beats real-only under repeated seeds.
- Improvement is consistent in AUROC and AUPRO, not only Dice.
- PatchCore-resNet or DRAEM corroborates the result.
- Morphology-specific rows are reported separately.
- Wood `000` remains soft-label trained; `001` and `002` remain hard-mask
  trained.
- Prediction overlays do not show broad false-positive saturation.

## Checkpoint Contents

- `snapshot/`: copies of the key source, config, test, report, and presentation
  artifacts at checkpoint time.
- `SHA256SUMS`: integrity hashes for every snapshot file.
- `checkpoint_manifest.json`: machine-readable status and result summary.
- `auto_mask_generation_checkpoint_2026-06-07.tar.gz`: portable archive.

