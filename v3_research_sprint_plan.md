# V3 Research Sprint Plan

## Objective

Move the current pipeline from a strong auto-mask prototype into a research-grade
closed-loop industrial defect generation system.

Current diagnosis:

```text
auto-mask architecture: credible
candidate selection: low-regret on bottle/zipper
main bottleneck: generated defects are often under-edited inside the mask
final evidence bottleneck: tiny-U-Net is too unstable for final claims
```

Current MVTec bottle/zipper result:

| Scope | Qwen Recall | Dice | IoU | Precision | Recall | Oracle Dice | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle overall | `1.0000` | `0.7361` | `0.5872` | `0.6835` | `0.8206` | `0.7473` | `0.0112` |
| Zipper overall | `0.9385` | `0.6229` | `0.4545` | `0.6989` | `0.6030` | `0.6364` | `0.0136` |

Current generation critic result:

```text
Phase 11 TF-IDG-lite baseline:
540 generated samples scored
74 accepted
466 rejected for low_mask_coverage
```

The next version should focus on:

```text
foundation-feature anomaly fields
TF-IDG-style closed-loop generation
structure-aware candidate generation
stronger final evaluators
broader MVTec validation
```

## Sprint 1: Unify The Generation Critic

### Status

Implemented on the current checkpoint.

What changed:

- Added shared critic module:

```text
iadgen_v2/generation_critic.py
```

- Phase 11 TF-IDG-lite now calls the shared scorer.
- Phase 4 critic-guided regeneration now calls the same scorer during retry
  selection.
- Phase 4 records the shared score components in each attempt:

```text
tfidg_lite_score
feature_alignment_score
adaptive_mask_coverage_score
adaptive_mask_coverage_fraction
texture_preservation_score
leakage_score
morphology_fit_score
outside_change_fraction
shell_change_fraction
reject_reasons
```

- Phase 11 CSV now writes the same coverage/leakage diagnostic fields, so Phase
  4 and Phase 11 are explainable with one score vocabulary.
- Added pure shared-critic unit coverage:

```text
tests/test_generation_critic.py
```

Validation completed in the current environment:

```bash
python -m py_compile iadgen_v2/generation_critic.py iadgen_v2/phase4.py iadgen_v2/phase11_tfidg_critic.py
python -m pytest -q tests/test_generation_critic.py tests/test_phase11_tfidg_critic.py
```

Result:

```text
2 passed
```

Environment limitation:

```text
tests/test_phase4.py cannot run in the current Python environment because torch
is not installed.
iadgen_v2.cli cannot run here because skimage is not installed.
```

Direct Phase 11 smoke validation was still run against existing Phase 4
metadata:

```text
reports/phase9_regen_smoke/phase11_tfidg_critic/qwen/tfidg_lite_report.md
```

The existing Phase 4 metadata is from before Sprint 1, and it shows the exact
old mismatch Sprint 1 fixes for future generation runs:

```text
old embedded Phase 4 coverage: 0.6716 -> accepted
new shared Phase 11 coverage: 0.2705 -> rejected low_mask_coverage
```

### Goal

Make Phase 4 retry decisions use the same mask-coverage and leakage logic as
Phase 11 TF-IDG-lite, so Phase 4 does not accept samples that Phase 11 later
rejects.

### Motivation

The smoke validation exposed a scoring mismatch:

```text
Phase 4 internal adaptive coverage: 0.6716
Phase 11 TF-IDG-lite adaptive coverage: 0.2705
Phase 11 decision: reject low_mask_coverage
```

This means the retry controller is currently too lenient.

### Implementation Tasks

- Create a shared module, for example:

```text
iadgen_v2/generation_critic.py
```

- Move shared scoring functions into it:

```text
adaptive_mask_coverage_score
leakage_score
texture_preservation_score
morphology_fit_score
critic_reject_reasons
```

- Refactor Phase 11 to call the shared module.
- Refactor Phase 4 critic-guided regeneration to call the same shared module.
- Make Phase 4 stop retrying only when the shared critic accepts the sample.
- If all attempts fail, save the best attempt but mark:

```json
"critic_guided_generation": {
  "accepted": false,
  "reject_reasons": [...]
}
```

- Add metadata fields:

```text
shared_critic_score
shared_adaptive_mask_coverage_score
shared_leakage_score
selected_attempt_index
attempt_count
```

### Validation

Run:

```bash
python -m pytest -q tests/test_phase4.py tests/test_phase11_tfidg_critic.py tests/test_phase5.py
python -m iadgen_v2.cli phase4-generate --config configs/phase9_regen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase11-tfidg-critic --config configs/phase9_regen_smoke.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_regen_smoke.yaml --provider qwen
```

### Acceptance Criteria

- Phase 4 and Phase 11 coverage scores agree within a small tolerance.
- Phase 4 does not mark a sample accepted if Phase 11 would reject it for
  `low_mask_coverage`.
- Phase 5 rejects samples with missing or failing TF-IDG-lite metrics.
- Full unit test suite passes.

## Sprint 2: Strict TF-IDG-Style Regeneration

### Status

Implemented and smoke-tested on the current checkpoint.

What changed:

- Phase 4 retry attempts are now reason-aware. If the previous attempt fails
  with `low_mask_coverage`, the next attempt receives:

```text
extra strength
extra guidance
extra inference steps
low-coverage prompt suffix
mask-local input noise boost
mask-local visibility/contrast boost
```

- Attempt metadata now records:

```text
retry_reason
local_mask_noise_boost
postprocess_mask_contrast_boost
num_inference_steps
strength
guidance_scale
shared critic scores
```

- The retry remains bounded and auditable. The output is still passed through
  the same shared TF-IDG-lite critic before acceptance.

Validation completed:

```bash
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m py_compile iadgen_v2/phase4.py iadgen_v2/generation_critic.py iadgen_v2/phase11_tfidg_critic.py
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m pytest -q tests/test_phase4.py tests/test_generation_critic.py tests/test_phase11_tfidg_critic.py
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m iadgen_v2.cli phase4-generate --config configs/phase9_regen_smoke.yaml --provider qwen
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m iadgen_v2.cli phase11-tfidg-critic --config configs/phase9_regen_smoke.yaml --provider qwen
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_regen_smoke.yaml --provider qwen
```

Unit result:

```text
23 passed
```

Smoke result:

```text
attempt 0: rejected low_mask_coverage
  coverage_score: 0.2705
  coverage_fraction: 0.1028
  strength/guidance/steps: 0.62 / 8.0 / 22

attempt 1: accepted
  coverage_score: 0.3791
  coverage_fraction: 0.1441
  strength/guidance/steps: 0.78 / 8.8 / 26
  local_mask_noise_boost: 0.035
  postprocess_mask_contrast_boost: 0.08
```

Phase 11 and Phase 5 selector both accepted the regenerated sample:

```text
TF-IDG-lite score: 0.8027
feature alignment: 0.9054
texture preservation: 0.9027
leakage: 0.9747
adaptive mask coverage: 0.3791
selector accepted rows: 1 / 1
```

Downstream tiny-U-Net smoke result is not final research evidence:

```text
real-only AUROC/Dice: 0.6232 / 0.0132
synthetic ratio 0.25 AUROC/Dice: 0.5352 / 0.0273
```

Interpretation:

```text
Sprint 2 fixed the under-editing gate for the targeted smoke sample, but the
tiny-U-Net scaffold remains unstable and overpredicts positives. The next
validation must scale beyond one generated sample and include stronger
normal-memory evaluators.
```

### Goal

Turn Phase 4 from one-pass generation into a real closed-loop generator:

```text
generate -> score -> adapt strength/prompt/noise -> regenerate -> accept/reject
```

### Implementation Tasks

- Add morphology-specific retry schedules:

| Morphology | Retry Strategy |
| --- | --- |
| `micro_chip` | higher strength, sharper prompt, small-mask local noise boost |
| `tooth_break` | stronger local tooth-chain damage prompt, medium-high strength |
| `contamination` | moderate strength, broad but texture-preserving edit |
| `multi_scuff` | low-to-medium strength, soft mask, texture preservation priority |
| `scratch_band` | medium strength, thin line prompt, local contrast boost |

- Add local in-mask noise boost before SD inpainting.
- Add retry metadata:

```text
attempt_index
strength
guidance_scale
num_inference_steps
prompt_suffix
coverage_score
leakage_score
accepted
```

- Add a Phase 4 report section:

```text
critic-guided rows
accepted rows
mean selected attempt
low-coverage final rejects
```

### Validation

Run targeted Phase 9 after Sprint 1:

```bash
python -m iadgen_v2.cli phase4-generate --config configs/phase9_reliability_targeted.yaml --provider qwen
python -m iadgen_v2.cli phase11-tfidg-critic --config configs/phase9_reliability_targeted.yaml --provider qwen
```

### Acceptance Criteria

Baseline:

```text
74 / 540 accepted
466 low_mask_coverage rejects
```

Target:

```text
accepted samples >= 200 / 540
low_mask_coverage rejects reduced by at least 50%
texture_preservation_score remains >= 0.85 mean
leakage_score remains >= 0.90 mean
```

## Sprint 3: Foundation-Feature Anomaly Field

### Status

Implemented as an experimental candidate on the current checkpoint, but not
accepted into the primary bottle/zipper benchmark path yet.

What changed:

- Added `foundation_anomaly_field` candidate generation in:

```text
iadgen_v2/auto_masks.py
```

- The candidate fuses dependency-light zero-shot evidence from:

```text
PatchCore-style normal patch memory
nearest-normal RGB residual
normal-texture residual
FFT texture residual
optional MUSC-style mutual patch score when enabled
```

- The field is robust-calibrated inside the Qwen/search region and writes:

```text
foundation_anomaly_heatmap.png
foundation_anomaly_candidate.png
foundation_anomaly_overlay.png
```

- Added unit coverage for artifact writing and selector routing:

```bash
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m pytest -q tests/test_auto_masks.py
```

Result:

```text
79 passed
```

- Added an isolated smoke config:

```text
configs/mvtec_bottle_zipper_sprint3_foundation.yaml
```

- Added path arguments to the official-mask evaluator so sprint pilots can be
  reviewed without overwriting the primary benchmark report.

Validation run:

```bash
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m iadgen_v2.cli auto-masks --config configs/mvtec_bottle_zipper_sprint3_foundation.yaml
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m iadgen_v2.cli auto-masks-reselect --config configs/mvtec_bottle_zipper_sprint3_foundation.yaml
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python scripts/evaluate_mvtec_bottle_zipper_auto_masks.py --metadata-path outputs/mvtec_bottle_zipper_sprint3_foundation/auto_masks/qwen/metadata.jsonl --report-dir reports/mvtec_bottle_zipper_sprint3_foundation/official_mask_evaluation
```

Smoke result:

| Scope | Qwen Recall | Dice | IoU | Precision | Recall | Oracle Dice | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle overall | `1.0000` | `0.6063` | `0.4692` | `0.6302` | `0.6123` | `0.6388` | `0.0325` |
| Zipper overall | `0.8912` | `0.5085` | `0.3550` | `0.6259` | `0.4780` | `0.5493` | `0.0407` |

Interpretation:

```text
The foundation field is functional and explainable, but the isolated fast
smoke run does not meet Sprint 3 acceptance. It underperforms the primary
cached benchmark because the smoke config intentionally prunes SAM/MUSC-heavy
candidates to keep validation interactive.
```

Professional decision:

```text
Do not promote foundation_anomaly_field into configs/mvtec_bottle_zipper_auto_mask.yaml yet.
Keep it in the sprint config and use it for ablation/offline validation.
Do not execute Sprint 4 as if Sprint 3 passed acceptance.
```

Main issues found:

- Runtime is still high when multiple normal-memory candidates recompute patch
  features independently.
- The foundation candidate is selected on only a small number of samples.
- It is useful diagnostically: for one `zipper/split_teeth` sample it was the
  oracle-best candidate, but selector routing is not reliable enough yet.
- The remaining weak cases are still `bottle/broken_small`,
  `zipper/fabric_border`, and `zipper/split_teeth`.

Recommended fix before the next sprint:

```text
Add shared patch-feature caching across PatchCore, repeated-chain, SAM-regularized,
and foundation candidates; then rerun the full candidate set offline and only
promote foundation if it improves the primary metrics.
```

Cache fix status:

```text
implemented
```

What changed:

- Added a per-sample `PatchFeatureCache` in `iadgen_v2/auto_masks.py`.
- Shared cached handcrafted patch features across:

```text
patchcore_guided
musc_mutual_score
foundation_anomaly_field
repeated_chain_refiner
sam_prompt_regularized
zipper_fabric_border_layout
```

- Cache keys include:

```text
image/path identity
resize target size
patch size
stride
text-derived polarity
```

- Candidate metadata now records cache diagnostics such as:

```text
patchcore_guided_feature_cache_enabled
patchcore_guided_feature_cache_hits
patchcore_guided_feature_cache_misses
musc_feature_cache_hits
```

Validation:

```bash
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m py_compile iadgen_v2/auto_masks.py
/home/p76147019/ImgGen/v2/.conda/envs/iadgen-v2/bin/python -m pytest -q tests/test_auto_masks.py
```

Result:

```text
80 passed
```

Manual one-sample cache smoke:

```text
selected refinement: repeated_chain_refiner
feature-cache hits: 20
feature-cache misses: 5
```

Interpretation:

```text
The cache now works in the real candidate path. It reduces duplicate feature
extraction across PatchCore-family candidates, but a full benchmark rerun is
still needed before promoting foundation_anomaly_field into the primary
bottle/zipper config.
```

### Goal

Reduce category-specific heuristic dependence by creating a unified anomaly
field from modern pretrained features.

### Research Basis

Use normal-memory and foundation features inspired by:

- PatchCore-style normal patch memory;
- DINOv2 visual features;
- CLIP/AnomalyCLIP text-image abnormality cues;
- SAM2 boundary proposals.

### Implementation Tasks

- Add a new candidate source:

```text
foundation_anomaly_field
```

- Fuse:

```text
PatchCore residual
DINOv2 patch residual
nearest-normal RGB residual
texture residual
optional CLIP anomaly prompt score
```

- Calibrate the fused field per image using normal-image statistics.
- Use SAM/SAM2 as boundary proposer over high-field regions.
- Write candidate heatmaps and overlays:

```text
foundation_anomaly_heatmap.png
foundation_anomaly_candidate.png
foundation_anomaly_overlay.png
```

### Validation

Run MVTec bottle/zipper reselection:

```bash
python -m iadgen_v2.cli auto-masks-reselect --config configs/mvtec_bottle_zipper_auto_mask.yaml
python scripts/evaluate_mvtec_bottle_zipper_auto_masks.py
```

### Acceptance Criteria

- Bottle overall Dice stays `>= 0.7361`.
- Zipper overall Dice improves from `0.6229` to `>= 0.65`.
- Selector regret remains low:

```text
bottle regret <= 0.012
zipper regret <= 0.014
```

- No official masks are used during generation or selection.

## Sprint 4: Structure-Aware MVTec Specialists

### Goal

Improve the remaining hard MVTec cases without turning the pipeline into
dataset-specific overfitting.

### Current Weak Cases

```text
zipper/fabric_border Dice: 0.5899
zipper/broken_teeth recall: 0.6136
zipper/split_teeth recall: 0.5522
```

### Implementation Tasks

#### Zipper Fabric Border Candidate

- Estimate zipper axis using PCA.
- Define left/right fabric side bands.
- Suppress central tooth-chain region.
- Score side bands with normal residual and foundation anomaly field.
- Allow true edge-touching masks without over-penalizing QC.

#### Zipper Tooth Recall Expansion

- Start from high-confidence PatchCore/repeated-chain core.
- Expand along the tooth-chain axis, not into fabric.
- Gate expansion by residual/foundation heat support.
- Preserve precision using max-area and side-band suppression.

#### Bottle Small Chip Arbitration

- Allow `polar_rim_residual` to win more often when:

```text
area is sane
rim-sector support is high
SAM candidate is broad or low-contrast
```

### Validation

Run:

```bash
python -m iadgen_v2.cli auto-masks-reselect --config configs/mvtec_bottle_zipper_auto_mask.yaml
python scripts/evaluate_mvtec_bottle_zipper_auto_masks.py
python scripts/evaluate_repeated_chain_refiner.py
```

### Acceptance Criteria

```text
zipper/fabric_border Dice >= 0.65
zipper/broken_teeth recall >= 0.68
zipper/split_teeth recall >= 0.62
bottle/broken_small Dice >= 0.72
overall selector regret < 0.010
```

## Sprint 5: Strong Evaluator Stack

### Goal

Demote tiny-U-Net to smoke testing and establish stronger final evidence.

### Implementation Tasks

- Keep:

```text
tiny_unet = smoke scaffold only
patchcore_lite = local fallback
patchcore_resnet = main normal-memory baseline
```

- Add:

```text
EfficientAD evaluator
WinCLIP or AnomalyCLIP zero/few-shot evaluator
```

- Report:

```text
pixel AUROC
AUPRO
image AUROC
Dice/IoU at adaptation-selected threshold
predicted positive rate
morphology-specific metrics
bootstrap confidence intervals
```

### Validation

Run Phase 5 on targeted and full configs:

```bash
python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_reliability_targeted.yaml --provider qwen
python -m iadgen_v2.cli phase5-evaluate --config configs/phase9_reliability.yaml --provider qwen
```

### Acceptance Criteria

- `patchcore_resnet` appears in the result table.
- EfficientAD result appears in the result table.
- Synthetic augmentation improves at least one strong evaluator, not only
  tiny-U-Net.
- Predicted positive rate is not degenerate:

```text
0.001 <= predicted_positive_rate <= 0.15
```

## Sprint 6: Broader Zero-Shot MVTec Validation

### Goal

Test whether the architecture generalizes beyond wood, bottle, and zipper.

### Proposed Categories

```text
cable
capsule
hazelnut
leather
metal_nut
tile
wood
bottle
zipper
```

### Implementation Tasks

- Build isolated pilots with official masks stored outside training paths.
- Freeze thresholds before evaluation.
- Run auto-mask generation/reselection.
- Run official-mask evaluation only after generation.
- Produce per-category review sheets.

### Acceptance Criteria

- At least 6 categories evaluated.
- No official masks used before evaluation.
- Per-category failure modes documented.
- Report includes:

```text
mean Dice
mean IoU
precision
recall
Qwen/search-region recall
selector regret
QC warning/reject rate
```

## Sprint 7: Paper-Ready Evidence Pack

### Goal

Convert the engineering pipeline into thesis/paper evidence.

### Implementation Tasks

- Create a frozen experiment manifest:

```text
config hash
dataset hash
candidate modes
thresholds
model versions
official-mask isolation policy
```

- Produce final visual evidence sheets:

```text
input
Qwen/search region
generation core
eval mask
uncertainty map
generated synthetic image
TF-IDG critic map
downstream prediction
```

- Add bootstrap confidence intervals.
- Add ablations:

```text
without foundation anomaly field
without structure specialists
without TF-IDG regeneration
without uncertainty loss
without synthetic selector
```

### Acceptance Criteria

- Final report distinguishes:

```text
implemented architecture
validated evidence
remaining limitations
```

- Claims are limited to validated results.
- All visual evidence and CSVs are reproducible from commands.

## Recommended Immediate Order

Do these first:

```text
Sprint 1: shared TF-IDG critic
Sprint 2: strict regeneration
Sprint 4: zipper fabric/tooth candidate improvements
Sprint 5: EfficientAD / stronger evaluator
```

The highest-impact immediate fix is Sprint 1 because the latest smoke test
proved that Phase 4 and Phase 11 currently disagree about mask coverage.
