# Generalization-First End-To-End Improvement Plan

## Implementation Checkpoint: 2026-07-22

The `v3-generic-evidence` production path is implemented behind
`auto_masks.architecture: generic_evidence`. The category-specialized path
remains available as the frozen `v2-specialist-baseline`.

Implemented integrity and compatibility work:

- every CLI command is covered by an explicit runtime, official-preparation,
  or official-evaluation policy;
- only `locked-evaluate` can score against isolated official locked masks;
- lifecycle manifests record status, elapsed time, parent hashes, output
  hashes, content-based dataset fingerprints, and runtime resource usage;
- failed auto-mask runs retain diagnostic partial metadata without replacing
  the last successful stable metadata;
- generic Phase 4/5/11 execution is blocked until the development gate passes;
- architecture freezing hashes behavioral configuration, code, selector, and
  configured SAM checkpoints;
- all `18` bottle/zipper legacy modes and binary-mask SHA-256 hashes reproduce
  exactly after cached reselection;
- legacy official-mask metrics remain bottle Dice `0.7468` and zipper Dice
  `0.6180`;
- the complete regression suite currently passes: `201 passed`.

Implemented generic architecture work:

- production execution now constructs `AutoMaskContext`, `EvidenceMap`,
  `CandidateProposal`, and `SelectionDecision` contracts;
- multi-scale DINOv2-small evidence uses layers `-4` and `-1`, scales `448`
  and `672`, and content-addressed caching;
- DINO correspondence plus RANSAC affine registration falls back to an
  unregistered residual below the configured confidence thresholds;
- MuSc-style mutual rarity and texture residuals share the evidence-provider
  interface;
- evidence sources use leave-one-normal-out median/MAD calibration and
  reliability-weighted logit fusion;
- Qwen regions are soft priors with `1.0 / 0.5 / 0.2` inside, margin, and
  outside weights;
- Qwen localization responses use a content-addressed, integrity-checked cache
  keyed by image pixels, prompt, model revision, dtype, software version, and
  explicit greedy decoding settings;
- proposal generation uses fused quantiles, component combinations, and
  anomaly-supported SAM2 prompts;
- three leave-category-out HistGradientBoosting regressors predict candidate
  IoU, precision, and recall from normal-only synthetic corruption data;
- isotonic calibration, a 90% conformal lower bound, mathematically consistent
  precision/recall-to-IoU projection, and abstention are active;
- posterior-derived core, possible, uncertainty, training, evaluation, and
  inpainting mask roles are serialized without changing legacy consumers;
- expensive normal-null calibration and evidence maps now have deterministic
  disk caches.

Selector training completed on `5,097` candidate rows spanning five
development categories and six corruption families. Its 90% conformal residual
is `0.03738`.

Current fixed three-category smoke result:

| Category | Dice | Precision | Recall | Search Recall | Regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| `metal_nut` | `0.0010` | `0.0054` | `0.0005` | `0.1390` | `0.0085` |
| `tile` | `0.5654` | `0.4356` | `0.8055` | `1.0000` | `0.3528` |
| `wood` | `0.5430` | `0.9993` | `0.3728` | `0.5666` | `0.0000` |

Macro Dice is `0.3698`, worst-category Dice is `0.0010`, accepted coverage is
`33.33%`, mean search-region recall is `0.5685`, and selector regret is
`0.1204`. The live localization-cache population run took `150.01` seconds for
three images. Its immediate replay took `16.93` seconds (`8.86x` faster), with
all three Qwen localizations served from cache. Qwen response hashes, regions,
selected modes, and final evaluation-mask SHA-256 hashes matched exactly. The
result is below every release gate except individual tile/wood mask quality,
so `v3-generic-evidence-rc1` has not been frozen.

The main observed failures are Qwen localization accuracy and domain shift
between simple synthetic corruption training and real structural defects. The
new cache removes run-to-run drift but intentionally preserves the first
content-addressed response; it does not turn a poor box into a good one.
Metal-nut localization currently misses most of the defect, while wood
localization remains too narrow. These failures must be fixed with generic
localization/evidence/selector improvements and full five-category
leave-one-out validation, not category-name rules.

Sprint 4 architecture comparison, Sprint 5 locked evaluation, and Sprint 6
generation remain gated. No locked official mask has been opened by this path,
and Phase 4 generation remains frozen as required.

One modularity caveat remains: the generic execution path is modular, but the
frozen legacy implementation is still physically retained in the large
`auto_masks.py` compatibility module. Removing it would violate the exact
legacy reproduction requirement unless it is first moved mechanically.

## Executive Decision

The current pipeline is a strong research prototype, but it is not yet proven
to be generally good on unseen industrial data.

The immediate priority should not be another bottle- or zipper-specific
refinement. Bottle and zipper have been useful development categories, but
their official-mask evaluations have repeatedly influenced the architecture.
They should now be treated as a development benchmark rather than evidence of
zero-shot generalization.

The next major version should replace the default category-aware path with a
general anomaly-evidence architecture:

```text
few normal references + a defect image + optional description
-> category-agnostic object/component canonicalization
-> multi-scale foundation-feature anomaly fields
-> proposal masks from anomaly evidence and SAM2 boundaries
-> reliability-calibrated candidate fusion
-> core / possible / uncertainty mask roles
-> critic-controlled synthetic generation
-> independent downstream evaluation
```

Category-specific methods such as bottle polar unwrapping and zipper-chain
geometry can remain as optional specialists. They must not define the default
result for an unseen category.

## Current Result Review

### Auto-Mask Evidence

The latest full-candidate MVTec development result is encouraging:

| Scope | Images | Search Recall | Dice | Precision | Recall | Oracle Dice | Selection Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | `9` | `1.0000` | `0.7468` | `0.7105` | `0.8110` | `0.7482` | `0.0014` |
| Zipper | `9` | `0.8912` | `0.6180` | `0.6917` | `0.6099` | `0.6291` | `0.0111` |

This demonstrates that the current candidate pool and selector can produce
useful pseudo-labels. It does not establish category-general performance:

- only 18 pixel-labeled pilot images are represented;
- bottle and zipper have dedicated localization, selection, and cleanup code;
- official masks were isolated from runtime generation, but repeated benchmark
  reviews still influenced engineering decisions;
- the external Kodytek wood pilot has weak bounding-box references rather than
  pixel-accurate masks.

The external wood result is still useful transfer evidence: Qwen returned a
valid search region for `11/12` samples. It shows that localization can transfer,
but it does not measure mask boundary accuracy.

### Synthetic Generation Evidence

The TF-IDG-lite critic found a clear generation failure:

| Measure | Result |
| --- | ---: |
| Generated samples scored | `540` |
| Accepted | `74` |
| Rejected for low mask coverage | `466` |
| Accepted mean mask coverage | `0.6721` |
| Rejected mean mask coverage | `0.0691` |

The generator usually preserves texture and avoids leakage, but many outputs do
not create enough visible defect evidence inside the intended mask. This means
the current generation problem is primarily under-editing, not background
damage.

### Downstream Evidence

The repeated-seed supervised result supports a modest benefit at synthetic
ratio `0.25`:

| Row | AUROC | AUPRO | Dice |
| --- | ---: | ---: | ---: |
| Real-only | `0.7861` | `0.4419` | `0.2204` |
| `qwen_mask_only`, ratio `0.25` | `0.8107` | `0.4578` | `0.2370` |
| `fixed_mask_adapter`, ratio `0.25` | `0.8185` | `0.4743` | `0.2331` |

The gain is directionally positive but small: approximately `+0.0246` to
`+0.0324` AUROC, `+0.0159` to `+0.0324` AUPRO, and at most `+0.0166` Dice.
Only three seeds and one targeted data regime are represented.

The normal-only `patchcore_resnet` result remains substantially stronger:

```text
AUROC 0.9558, AUPRO 0.8549, Dice 0.4439
```

Therefore, the current research claim should be:

```text
selected synthetic augmentation can modestly improve one calibrated supervised
student on the current targeted experiment;
general downstream benefit is not yet proven.
```

## Architecture Review

### Advantages

- The workflow requires no manual bounding boxes or masks.
- Held-out anomaly samples are separated from adaptation and threshold
  calibration.
- Qwen is used as a semantic search aid instead of a pixel-mask oracle.
- The mask-role design is sound: `eval_tight`, `training_medium`,
  `training_soft`, `positive_core`, `possible_region`, `uncertainty_map`, and
  `inpaint_soft` serve different purposes.
- Normal-reference evidence, PatchCore-style memory, SAM2 proposals, texture
  residuals, and morphology checks provide complementary signals.
- Candidate artifacts and selection regret are recorded, making failures
  diagnosable.
- The TF-IDG-lite critic creates a closed feedback loop for generation quality.
- Stronger normal-memory and supervised evaluator paths now coexist.

### Disadvantages

- `iadgen_v2/auto_masks.py` is approximately `10,690` lines and combines
  localization, feature extraction, candidate construction, category policy,
  postprocessing, artifact writing, and selection. This makes controlled
  generalization experiments difficult.
- The production path contains direct checks for `bottle`, `zipper`, scratch,
  teeth, rim, and fabric-border cases. These improve the development benchmark
  but create hidden domain assumptions.
- Candidate ranking uses hand-tuned score offsets and thresholds rather than a
  category-independent estimate of mask reliability.
- Current normal-memory features are dominated by handcrafted patches and
  WideResNet PatchCore. DINOv2 support exists experimentally, but a unified
  multi-scale foundation field is not the default.
- SAM2 is treated as one candidate among many. It should instead refine
  anomaly-supported prompts and contribute boundary uncertainty.
- The pipeline mixes three different scientific settings: normal-only anomaly
  detection, few-defect pseudo-labeling, and synthetic supervised augmentation.
  Results from these settings need separate tables and claims.
- The strongest guided student fuses a PatchCore teacher at inference. That is
  a useful system, but it is not independent evidence that synthetic training
  alone caused the improvement.
- Variant and ratio choices were reviewed on the same targeted task. A locked
  validation/test protocol is needed before making a general claim.
- There is no abstention policy. A system intended for general data should be
  allowed to report `needs_review` when evidence is inconsistent.

## Correct Task Definitions

The project should stop using one broad `zero-shot` label for every experiment.

| Track | Available target data | Correct description |
| --- | --- | --- |
| A | No target normal or defect images | strict zero-shot anomaly segmentation |
| B | A few target normal images only | training-free few-shot / normal-reference anomaly segmentation |
| C | Normal images plus a few target defect images and text | scarce-defect automatic pseudo-labeling |
| D | Track C plus generated defects | synthetic-augmented supervised segmentation |

The current main pipeline is Track C followed by Track D. PatchCore and similar
normal-memory baselines are Track B. Reporting these separately will make the
research contribution clearer and prevent unfair comparisons.

## Target Architecture

### 1. Object And Component Canonicalization

Build a category-agnostic coordinate system before anomaly scoring:

- estimate product foreground from normal-image consensus and SAM2;
- align defect and normal images using coarse foundation features followed by
  local geometric registration;
- cluster recurring product components and preserve their relative positions;
- express the Qwen output as generic attributes:
  `appearance`, `shape`, `scale`, `location`, `boundary_contact`,
  `repetition`, and `confidence`;
- use multi-crop and flip consistency to estimate localization uncertainty.

Qwen should propose semantic regions and attributes. It should not directly
select a category-specific algorithm.

This follows the general direction of
[UniVAD](https://openaccess.thecvf.com/content/CVPR2025/html/Gu_UniVAD_A_Training-free_Unified_Model_for_Few-shot_Visual_Anomaly_Detection_CVPR_2025_paper.html),
which uses foundation-model component clustering, component-aware patch
matching, and graph-level component reasoning across domains.

### 2. Unified Multi-Scale Anomaly Evidence

Replace the collection of loosely calibrated heatmaps with one evidence API:

```text
E_local   = multi-scale DINOv2 or RADIO patch-memory distance
E_normal  = registered nearest-normal residual
E_mutual  = MuSc-style rarity across the image batch
E_text    = object-agnostic normal/abnormal vision-language score
E_logic   = component presence, count, and relative-position inconsistency
```

Each source must return:

```text
normalized anomaly map
normal-derived calibration statistics
spatial resolution
reliability score
augmentation-consistency score
```

Use robust normalization derived only from normal references. Fuse sources by
reliability, not fixed category offsets.

Research support:

- [AnomalyDINO](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html)
  shows that training-free DINOv2 nearest-neighbor patch matching is a strong
  few-shot industrial baseline.
- [MuSc](https://proceedings.iclr.cc/paper_files/paper/2024/hash/096b1019463f34eb241e87cfce8dfe16-Abstract-Conference.html)
  shows that mutual rarity across unlabeled images and multiple neighborhood
  scales can strongly improve zero-shot anomaly segmentation.
- [AnomalyCLIP](https://arxiv.org/abs/2310.18961) motivates object-agnostic
  normal/abnormal prompts rather than product-name-specific prompt rules.
- [ReMP-AD](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_ReMP-AD_Retrieval-enhanced_Multi-modal_Prompt_Fusion_for_Few-Shot_Industrial_Visual_Anomaly_ICCV_2025_paper.html)
  motivates retrieval that filters noisy normal references before multimodal
  fusion.

### 3. Generic Proposal And Boundary Refinement

Generate proposals from the fused field at multiple operating points. Prompt
SAM2 with positive anomaly peaks, negative normal anchors, and the semantic
search region. Reject SAM masks that are unsupported by anomaly evidence.

Describe every proposal using category-independent measurements:

```text
evidence coverage
foreground containment
normal-reference contrast
component count
compactness and elongation
boundary contact
periodicity disruption
cross-augmentation stability
cross-source agreement
```

The existing polar-rim and repeated-chain methods can remain optional proposal
operators activated by measured topology such as ring geometry or periodic
structure. Activation must not depend on the strings `bottle` or `zipper`.

### 4. Reliability-Calibrated Selection With Abstention

Replace hand-written candidate score bonuses with a reliability model trained
without target official masks.

Training data can be created from normal images by injecting diverse synthetic
corruptions with known masks. Candidate-quality targets are the IoU between a
candidate and the injected mask. Use leave-category-out training so the
selector never learns the evaluated product category.

The selector should predict:

```text
expected mask IoU
expected precision
expected recall
calibrated confidence
```

Use ensemble or conformal calibration to produce an abstention decision:

```text
hard_mask_ok
soft_mask_only
needs_review
```

For automatic operation, `needs_review` should still write a conservative
`positive_core` and broad `possible_region`, but should not silently create a
hard pseudo-ground-truth mask.

### 5. Uncertainty-First Mask Roles

Construct mask roles from posterior evidence rather than morphological dilation
alone:

- `positive_core`: high-confidence intersection across evidence and transforms;
- `possible_region`: calibrated union of plausible support;
- `uncertainty_map`: disagreement, registration uncertainty, and boundary
  instability;
- `training_soft`: anomaly probability with uncertain areas down-weighted;
- `eval_tight`: conservative binary mask for compatibility only;
- `inpaint_soft`: generation envelope independent of evaluation policy.

This retains one of the strongest parts of the existing architecture while
making its inputs more general.

### 6. Generation That Responds To Evidence

Keep the shared TF-IDG-lite critic, but turn it into a controller rather than a
fixed rejection threshold:

- choose denoising strength from defect scale and evidence confidence;
- regenerate low-coverage samples with stronger local conditioning;
- reduce strength or mask envelope when leakage or texture damage rises;
- stop when coverage enters a morphology-dependent interval rather than forcing
  every mask to exceed one global occupancy threshold;
- measure diversity in foundation-feature space to avoid keeping near-duplicates;
- rank synthetic samples by both realism and usefulness to an independent
  anomaly evaluator.

The design should borrow the separation principle from
[SeaS](https://openaccess.thecvf.com/content/ICCV2025/html/Dai_SeaS_Few-shot_Industrial_Anomaly_Image_Generation_with_Separation_and_Sharing_ICCV_2025_paper.html):
normal-product attributes and anomaly attributes should have separate
conditioning/loss paths. A full SeaS replacement should be evaluated as an
ablation, not assumed to be better without an equal-budget comparison.

### 7. Independent Downstream Evidence

Use at least two evaluator families that do not share the same teacher:

```text
normal-memory: PatchCore-ResNet and AnomalyDINO/UniVAD
supervised: ResNet18 U-Net or a stronger segmentation student
vision-language: AnomalyCLIP or a current object-agnostic equivalent
```

Report synthetic benefit against the same evaluator trained or calibrated
without synthetic samples. Keep PatchCore-fused students as a practical system,
but do not use them as the only evidence for generation quality.

## Generalization Protocol

### Dataset Governance

Declare all previously inspected categories as development data:

```text
development: bottle, zipper, wood, metal_nut, tile
```

Use untouched MVTec AD categories for locked evaluation:

```text
carpet, cable, capsule, grid, hazelnut, leather,
pill, screw, toothbrush, transistor
```

After freezing the architecture, run external-domain validation on VisA and
the public portion of
[MVTec AD 2](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2).
MVTec AD 2 is particularly valuable because it includes lighting variation and
an evaluation-server test partition.

Treat logical anomalies as a separate extension. The current local-pixel
pipeline should not claim logical anomaly coverage until it is tested on
[MVTec LOCO AD](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad),
which contains both structural and logical anomalies.

### Leakage Firewall

- Official masks for locked categories live outside the runtime dataset tree.
- Configuration and model hashes are frozen before official evaluation.
- The locked test script runs once per tagged architecture version.
- No per-category threshold, prompt, candidate bonus, or cleanup rule may be
  added after viewing locked results.
- Failed categories become future development data only after the current
  benchmark result is archived.

### Required Metrics

Report macro averages so large categories do not dominate:

```text
pixel AUROC, pixel AP, AUPRO
Dice and IoU at a normal/adaptation-calibrated threshold
precision, recall, and predicted-positive rate
search-region recall
selector regret during development only
abstention coverage and risk
per-category and worst-quartile performance
runtime and peak GPU memory
```

Use five seeds for trainable evaluators and hierarchical bootstrap confidence
intervals over categories and images. Never select a synthetic ratio from the
locked test set.

## Sprint Plan

### Sprint 0: Evidence Firewall And Baseline Freeze

Deliverables:

- label bottle/zipper/wood/metal_nut/tile as development categories;
- define locked category manifests and official-mask isolation checks;
- create separate Track A-D result tables;
- freeze the current architecture as `v2-specialist-baseline`;
- add config, dataset, model, and code fingerprints to every report.

Exit criteria:

```text
locked masks cannot be resolved by auto-mask or training code
all current development results reproduce from a manifest
```

### Sprint 1: Modularize The Auto-Mask Core

Split `auto_masks.py` into stable interfaces:

```text
auto_mask/localization.py
auto_mask/registration.py
auto_mask/evidence/base.py
auto_mask/evidence/normal_memory.py
auto_mask/evidence/mutual_rarity.py
auto_mask/evidence/vision_language.py
auto_mask/proposals.py
auto_mask/selection.py
auto_mask/mask_roles.py
auto_mask/specialists/
```

Preserve existing behavior through regression tests before changing metrics.

Exit criteria:

```text
current bottle/zipper outputs remain within deterministic tolerance
default core has no category-name conditionals
specialists are optional plugins with explicit activation evidence
```

### Sprint 2: Foundation Evidence Baseline

Implement cached multi-scale DINOv2 patch memory, registered normal residuals,
MuSc-style mutual rarity, and object-agnostic vision-language cues behind the
common evidence API.

Run leave-one-development-category-out ablations. Do not use the locked set.

Exit criteria:

```text
macro development Dice >= current generic baseline
worst-category Dice improves
feature extraction runtime is measured and cached
no category-specific score offsets are used
```

### Sprint 3: Generic Selector And Uncertainty

Train the category-agnostic candidate reliability estimator on synthetic
normal-image corruptions. Add transformation consistency and calibrated
abstention. Build posterior-derived mask roles.

Exit criteria:

```text
leave-one-category-out selector regret <= 0.05
confidence expected-calibration error <= 0.10
high-confidence accepted masks have higher Dice than low-confidence masks
needs_review rate <= 25% on development data
```

### Sprint 4: Closed-Loop Generation Upgrade

Connect mask confidence, scale, and uncertainty to the Phase 4 retry controller.
Add feature-space diversity and independent utility scoring. Compare the current
SD1.5 adapter path with one separation-based generation baseline under the same
sample and compute budget.

Exit criteria:

```text
critic acceptance >= 50%
low-mask-coverage rejects reduced by at least 60%
mean texture preservation >= 0.85
mean leakage score >= 0.90
no more than 20% near-duplicate accepted samples
```

### Sprint 5: Locked MVTec Generalization Run

Freeze all settings, run the ten untouched MVTec AD categories, then unlock
official masks only for evaluation.

Target acceptance criteria:

```text
search-region recall >= 0.90 macro
auto-mask Dice >= 0.55 macro
auto-mask Dice >= 0.30 for every category
development-to-locked Dice gap <= 0.10 absolute
accepted-mask coverage >= 75%
```

These are research targets, not guaranteed outcomes. Results below target must
be reported rather than repaired with locked-category rules.

### Sprint 6: External Generalization And Synthetic Utility

Run the frozen system on VisA and MVTec AD 2. Evaluate synthetic augmentation
with two independent evaluator families and five seeds.

Synthetic augmentation is accepted only if:

```text
macro AUPRO or pixel AP improves by >= 2 absolute points
95% bootstrap interval for the mean gain excludes zero
no more than 20% of categories regress by > 1 point
ratio is selected on development/validation data, never test data
```

### Sprint 7: Logical And Production Extension

Only after structural generalization succeeds:

- add component-graph reasoning for missing, extra, misplaced, or wrong-count
  parts;
- evaluate separately on MVTec LOCO AD;
- test calibration under lighting and position shifts;
- define human-review and failure escalation for production deployment.

## Stop/Go Decisions

| Observation | Decision |
| --- | --- |
| Foundation field improves mean but hurts worst categories | improve reliability calibration before generation work |
| Selector has low regret but localization recall is low | improve canonicalization/Qwen consistency |
| Masks improve but synthetic utility does not | focus on generation controller and diversity |
| Synthetic samples pass critic but do not improve independent evaluators | critic is optimizing appearance rather than task utility |
| Locked-category performance collapses | remove specialist assumptions; do not tune locked categories |
| Normal-only baseline remains dominant | position synthesis as optional augmentation, not the main detector |

## Recommended Immediate Work

The next implementation should be Sprint 0 followed by Sprint 1. This creates
the experimental firewall and modular boundary required for every later claim.

The first algorithmic implementation after that should be Sprint 2: a
category-agnostic, multi-scale DINOv2 + normal-residual + mutual-rarity evidence
field. Further zipper-specific tuning should pause until the frozen general
baseline has been measured.

## Final Research Position

Current status:

```text
strong development prototype;
credible bottle/zipper pseudo-label quality;
promising but modest synthetic augmentation gain;
generalization not yet demonstrated.
```

Target status after this plan:

```text
category-agnostic default architecture;
specialists isolated from the general core;
calibrated uncertainty and abstention;
locked unseen-category evidence;
synthetic benefit confirmed by independent evaluators and external datasets.
```
