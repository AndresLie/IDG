# IADGen v2 — Current Weaknesses and Improvement Plan (Merged)

**Assessment date:** 2026-07-23
**Reviewed branch:** `paired-edge-selector` at `a7171af`
**Scope:** the full picture — auto-mask/selector path, runtime caching, *and* the
generation → downstream half. This document merges two reviews:

- **Track A audit** — matched-cohort execution of the heavy generic-evidence
  auto-mask path, selector calibration, and runtime (ran the pipeline at
  `e992272`).
- **Track B audit** — the generation proxy-gaming and downstream-augmentation
  findings (visual audit + repeated-seed results).

Measured claims are tied to artifacts where available; §3.5 records the exact
lineage and identifies interactive checks that still need durable logs. Deltas on
the tiny development cohorts are diagnostic, not confirmatory.

---

## 1. Executive summary

> **Superseded in part — read §1a/§1b first.** This section is the *initial
> merged audit* (pre-A-S1b), written before real-candidate recalibration. Its
> conclusion that the branch shows "no quality improvement" and that the paired
> gate doesn't fix the selector reflects the pre-recalibration state and is
> superseded by the validated A-S1b/A-S2 results in §1a and the PCA probe in §1b.
> §5.3 is the canonical status tracker. The audit below is retained for the
> baseline framing and the Track-B findings, which still hold.

The branch is functional, deterministic, and far faster than the earlier heavy
pipeline (207 tests pass; selector artifact byte-reproducible; identical mask
hashes on replay). The **runtime mechanism is validated**, but the implementation
is not merge-ready until the device and cache-correctness contracts in A7/A-S5
are fixed or the independently safe changes are split out.

It is **not** ready to be promoted as a quality improvement, on two independent
grounds:

- **Mask/selector (Track A):** on the matched six-image cohort the heavy config
  reaches Dice `0.3737`, *below* the edge-off baseline `0.3943`; the paired-V3
  variant `0.3773` is also not distinguishable from edge-off. Adding
  edge/paired candidates *raised* the oracle ceiling but *raised selection
  regret* too, so output quality did not improve. The dominant bottleneck is
  **absolute selector calibration**, which the paired gate does not fix.
- **Generation/downstream (Track B):** the generator satisfies the coverage
  proxy while producing **near-invisible defects** (visibility ~0.055; audited
  images show no visible crack / a faint smudge), and the synthetic downstream
  gain is within noise and dominated ~2× by a normal-only PatchCore baseline.

**Correction to a prior claim.** An earlier note framed the paired selector
(`on_v3`) as a validated improvement (bottle 0.479→0.508, zipper 0.254→0.258).
That was the 18-image per-category split and is within noise; the matched
six-image cohort shows it does **not** beat edge-off. The defensible claim is
narrower: the paired gate **prevents the catastrophic regression** the crude
edge variants caused (zipper 0.189–0.211), but is **not** a demonstrated Dice
gain.

**Recommendation:** keep the branch experimental; retain the runtime work but
merge only after A-S5 contract hardening (or split out a demonstrably safe
subset); hold selector/edge behavior behind an experimental flag. Track A and
Track B may be developed in parallel only when Track B consumes a frozen,
content-hashed mask manifest; otherwise sequence Track B after Track A stabilizes.

---

## 1a. Validated baseline and freeze (2026-07-23)

Sprint outcome: two stacking, leave-category-out-validated mask/selector gains
(A-S1b real-candidate selector recalibration + A-S2 repeated-structure widening),
deployed. The widened-pool refit passed a **predeclared** keep/discard gate
(LCO macro Dice +0.084, regret −0.084, no category regression, calibration MAE
not worse, coverage up) and is kept and deployed.

**Three figures kept strictly separate (do not conflate):**

| Figure | macro Dice | Meaning |
| --- | ---: | --- |
| Full-dev-trained **diagnostic** | `0.4723` | deployed selector scored on categories inside its own training — in-distribution, NOT generalization |
| **Leave-category-out estimate** | `0.5088` | defensible: train on 3 dev categories, evaluate on 2 held out; the honest generalization number |
| **Locked-category result** | not measured | the real test; blocked (see below) |

Baseline for reference: pre-A-S1b synthetic selector `0.3885`. The widened refit
*raised* the LCO estimate while *lowering* the in-distribution diagnostic
(0.5014→0.4723) — less overfitting, healthier generalization.

**Caveats:** all development-category, n=18 pilot, light-evidence config. These
are pseudo-label mask-quality gains; they do **not** touch the downstream
synthetic-utility (Track B) claim, which remains dominated by the normal-only
PatchCore baseline.

**Confirmed blockers — the scientifically decisive remaining work:**
- **Locked-category generalization (X1):** the 10 untouched MVTec categories are
  not present and the environment has no network. This is the real generalization
  test and cannot run here.
- **Downstream synthetic utility (B-S1 re-audit):** SD1.5 inpainting weights are
  not cached offline.

Development-category mask tuning is **stopped**; further minor refinements are
not the priority.

**Freeze status — hashes sealed; formal release-freeze correctly gate-blocked.**
`freeze-architecture` intentionally refuses because it requires a *passing*
development mask gate (macro Dice ≥ 0.55, worst-category ≥ 0.30, search-recall
≥ 0.90, accepted coverage ≥ 0.75, regret ≤ 0.05), and the current deployed masks
(macro ≈ 0.47–0.51 diagnostic / 0.51 LCO) are **below that release bar** — the
governance firewall working as designed, not a bug. So this is a *validated
checkpoint*, not a release-grade freeze. Reproducibility hashes are sealed via
the experiment manifest (`configs/v3_generic_evidence_frozen.yaml`,
`outputs/v3_generic_evidence_frozen/experiment_manifests/`):

```
architecture_core_fingerprint: c64a9830a2afffb5d3203bf122d1da97
config effective_fingerprint:  b8157e80833df6eb5eb2e2d113fc3b6c
code content_fingerprint:      23040dae26156b686e3e909c83c3661
dataset inventory_fingerprint: 403e039e6c894f7d3209463beb0a7ea4 (5 dev categories, 2120 files)
models_fingerprint:            277df7fc96a381206919722af4a9aa6a
```

The formal `freeze-architecture` release will be issued only when the mask gate
passes or after the locked-category evaluation — whichever the project chooses;
it must not be forced by lowering the gate.

## 1b. Post-freeze evidence exploration — PCA residual (2026-07-23)

Commit `a7171af` adds a **SubspaceAD-inspired**, default-off PCA-residual
provider: fit a subspace to normal DINOv2 patch features and score orthogonal
reconstruction residual. It is useful research infrastructure, but it is not a
faithful reproduction of [SubspaceAD](https://arxiv.org/abs/2602.23013). The
official recipe uses DINOv2-with-registers giant, 672-pixel input, 30
augmentations and 0.99 explained variance; this provider uses DINOv2-small,
448-pixel input, no augmentation and 0.90 explained variance. Any paper claim
must use the term *SubspaceAD-inspired adaptation* unless the official protocol
is reproduced separately.

Exploratory A/B on the 18-image bottle/zipper pilot:

| Scope | Oracle Dice | Selected Dice | Regret |
| --- | ---: | ---: | ---: |
| Zipper | 0.5750 → 0.5884 (`+0.0134`) | **0.3832 → 0.5290 (`+0.1458`)** | 0.1919 → 0.0594 |
| Bottle | 0.7630 → 0.7591 (`−0.0039`) | 0.5615 → 0.5318 (`−0.0297`) | 0.2015 → 0.2273 |
| Macro | 0.6690 → 0.6738 (`+0.0048`) | **0.4724 → 0.5304 (`+0.0580`)** | 0.1967 → 0.1434 |

**Professional interpretation:** this is a mixed, development-only result, not
yet evidence that PCA raises the proposal ceiling. Adding a provider
renormalizes fusion weights and replaces the fused map; the enabled pool is not
a strict superset of the baseline pool. Therefore an oracle change cannot
cleanly separate new PCA signal from dilution of existing evidence. The selected
gain is concentrated in zipper, bottle regresses, and the selector was not
calibrated specifically on PCA-perturbed candidates.

The runtime claim also needs correction. Normal DINO tokens are reused, but the
PCA basis plus leave-one-normal-out calibration are fitted again per target
image. Live provider timings during the pilot were roughly 73–78 seconds per
image; the run artifacts were later cleaned, so this timing must be reproduced
in a durable report before a performance claim.

**Decision:** keep the provider default-off. Do not immediately gate it using
the bottle/zipper-separating repetition threshold; that would turn an observed
category split into a rule without category-held-out evidence. Permit one
bounded PCA follow-up only after the provider (1) caches the fitted basis and
calibration per content-hashed normal set, (2) removes the fabricated
`augmentation_consistency=0.75` value, (3) preserves the complete baseline
candidate family as a strict subset, and (4) evaluates the gate with nested
leave-category-out validation. Archive the route if macro oracle gain remains
below `0.01` or any held-out category regresses by more than `0.01` selected
Dice.

## 2. Confirmed strengths (protect these)

- Qwen used as a search aid, not a pixel oracle.
- **Abstention works** — end-to-end smoke flagged 2/3 hard cases `needs_review`
  instead of publishing hard masks.
- **Governance firewall works** — generation is frozen behind a development
  mask-validation gate; the smoke correctly returned `go:false`.
- **Runtime mechanism + determinism validated** — DINO normal-token cache + GPU
  cosine KNN (1.2e-7 equivalent to CPU); reproducible selector artifact;
  identical replay mask hashes. Device/cache contracts remain open.
- On the 18-image development cohort, the paired gate **removed the crude
  variants' observed catastrophic regression**; this is not yet a general claim.

---

## 3. Measured baseline

### 3.1 End-to-end execution (Track A)

| Stage | Result | Measurement |
| --- | --- | ---: |
| Preflight | Pass | RTX A6000; 43.24 GiB free; Qwen/DINOv2/SAM2/selector present |
| Automated tests | Pass | 207 passed |
| Paired-selector training | Pass | 274.7 s; 5 dev categories; 5,756 paired rows |
| Selector reproducibility | Pass | SHA-256 reproduced exactly |
| Heavy inference (cold cache) | Pass | 450.7 s / 6 images ≈ 75.1 s/image |
| Heavy inference (warm cache) | Pass | 19.93 s / 6 images ≈ 3.32 s/image (replay) |
| Official-mask evaluation | Pass | 6 records; metrics + error sheet |
| Mask replay determinism | Pass | all 6 mask hashes match |

### 3.2 Matched six-image quality (Track A) — one image per defect type

| Variant | Dice | IoU | Precision | Recall | Oracle Dice | Regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge off | **0.3943** | **0.2934** | **0.4255** | 0.5020 | 0.5397 | **0.1454** |
| Paired V3 | 0.3773 | 0.2873 | 0.3911 | 0.5313 | **0.5691** | 0.1918 |
| Heavy stack | 0.3737 | 0.2672 | 0.3231 | **0.5672** | 0.5545 | 0.1808 |

Adding candidates raised oracle but raised regret; net Dice did not improve. No
variant demonstrates an improvement on this six-image diagnostic cohort. A claim
of equivalence or statistical indistinguishability requires the paired interval
analysis specified in A-S6.

### 3.3 Heavy-stack per-sample (Track A)

| Sample | Selected | Oracle | Regret | Primary failure |
| --- | ---: | ---: | ---: | --- |
| bottle/broken_large | 0.3033 | 0.8048 | 0.5015 | over-segmentation + selector mis-rank |
| bottle/broken_small | 0.7983 | 0.8022 | 0.0039 | strong mask, selector under-confident |
| bottle/contamination | 0.6348 | 0.7262 | 0.0914 | excessive area |
| zipper/broken_teeth | 0.1646 | 0.5603 | 0.3957 | scattered FPs + mis-rank |
| zipper/fabric_border | 0.3414 | 0.4335 | 0.0921 | precision low |
| zipper/split_teeth | 0.0000 | 0.0000 | 0.0000 | **proposal-generation failure** |

### 3.4 Downstream + generation (Track B)

Repeated-seed (n=3):

| Configuration | AUROC | AUPRO | Dice |
| --- | ---: | ---: | ---: |
| Real-only | 0.7861 ± 0.0531 | 0.4419 ± 0.0684 | 0.2204 ± 0.0262 |
| `qwen_mask_only`, r0.25 | 0.8107 ± 0.0232 | 0.4578 ± 0.0434 | **0.2370 ± 0.0159** |
| `fixed_mask_adapter`, r0.25 | **0.8185 ± 0.0378** | **0.4743 ± 0.0563** | 0.2331 ± 0.0319 |
| **normal-only PatchCore-ResNet** | **0.9558** | **0.8549** | **0.4439** |

There is no single synthetic configuration that owns all three bold synthetic
metrics: `fixed_mask_adapter` leads AUROC/AUPRO, while `qwen_mask_only` leads
Dice.

Generation critic (current, 1620 samples): coverage median 0.3082, **1620/1620
accepted**, but mean `defect_visibility_score` is 0.0551 (max 0.1033). The cited
"540/466 rejected" figure is a **stale pre-Sprint-2 baseline**.

### 3.5 Evidence index and lineage

Paths below are repository-relative.

| Claim family | Primary artifact(s) | Lineage note |
| --- | --- | --- |
| Edge off/on/on-v2/on-v3, 18 images | `reports/ab_edge_refine/{off,on,on_v2,on_v3}/official_mask_evaluation/auto_mask_metrics.csv` | Development cohort; current worktree configs include untracked files |
| Heavy six-image run | `reports/ab_edge_refine/heavy/official_mask_evaluation/auto_mask_metrics.csv`, `reports/ab_edge_refine/heavy_run.log` | Executed on `paired-edge-selector` at `e992272` |
| Selector retraining | `reports/ab_edge_refine/retrain_paired.log` | Pairing/training provenance must also be linked from the inference manifest |
| Repeated-seed downstream results | `reports/phase9_current_best_repeated_seed_validation/evidence_manifest.json`, `phase5/qwen/segmentation_results.csv`, `phase5/qwen/repeated_seed_result_review.md` | Three seeds; artifacts do not embed `e992272`, so they are historical Track-B evidence rather than proof about this branch |
| Current generation critic | `reports/phase9_reliability_targeted/phase11_tfidg_critic/qwen/tfidg_lite_metrics.jsonl`, `tfidg_lite_report.md` | 1620 records; 1620 accepted |
| Visual audit | `reports/phase9_current_best_repeated_seed_validation/phase9_visual/qwen/phase9_visual_evidence_contact_sheet.png`, `summary.md` | Qualitative evidence; retain stratified sample identifiers in the next audit |
| Tests and byte/hash replay | Interactive execution at assessment time | Verified during review, but not yet persisted as a release artifact; A-S6 must save command, environment, and output log |

---

## 4. Weaknesses

### Track A — mask / selector / runtime

**A1. The selector bottleneck improved but is not closed.** Real-candidate
recalibration reduced leave-category-out regret `0.1985 → 0.1074`, but the
result still narrowly misses the original `≤0.10` engineering gate. More
important, all evidence is development-category evidence; no locked-category
result exists.

**A2. The main gain is not yet externally confirmed.** A-S1b and A-S2 improved
category-held-out development metrics, but the untouched ten-category MVTec run
is unavailable. The full-dev diagnostic (`0.4723`) must not be presented as
generalization; the leave-category-out estimate (`0.5088`) is the strongest
current figure.

**A3. PCA evidence is mixed and experimentally confounded.** The pilot improves
macro selected Dice by `0.0580`, driven by zipper, while bottle regresses
`0.0297`. Macro oracle changes only `+0.0048`; provider fusion replaces the
baseline fused map rather than adding a strict candidate superset. The current
provider also refits PCA/LOO calibration per image at roughly 73–78 s/image.

**A4. QC is still not actionable.** In the PCA A/B, every one of the 18 outputs
was marked `warning`, so the status does not separate strong masks from risky
ones. Risk-coverage/selective-Dice evidence is missing.

**A5. The evidence stack remains too costly and unevenly cached.** The validated
DINO cache/GPU KNN mechanism is sound, but MuSc remains the heavy cold path and
the new PCA model/calibration has no per-normal-set fit cache.

**A6. Experiment packaging is incomplete.** Selector parent lineage is not
consistently declared in inference manifests; temporary A/B artifacts were
cleaned after summary metrics were copied into the report; the working tree
still contains unrelated untracked/deleted files. Confirmatory runs need durable
configs, manifests, logs and per-image metrics.

### Track B — generation / downstream

**B1. Synthetic-augmentation benefit is within noise and baseline-dominated.**
Best mean gain is +0.0324 AUROC for `fixed_mask_adapter`, but its paired per-seed
deltas are -0.0311, +0.0316, and +0.0966 (paired t ≈ 0.88, n=3). The best mean
Dice gain is +0.0166 for `qwen_mask_only`, with paired deltas +0.0018, -0.0132,
and +0.0612 (paired t ≈ 0.73, n=3). Neither is confirmatory. PatchCore dominates
~2× on Dice; the evaluated student also fuses a PatchCore teacher (confound).

**B2. The generator games the coverage proxy.** 1620/1620 accepted at median
coverage 0.3082, but mean visibility is 0.0551; audited crack shows no crack and
the scratch is a smudge.

**B3. The replacement visibility metric is not yet valid on structured
backgrounds.** Its contrast term compares absolute generated-region intensity
with a surrounding ring. An unchanged high-contrast structure can therefore
score as visible despite zero edit magnitude. Existing unit tests use a uniform
background and do not cover this failure mode.

**B4. The empirical controller audit is blocked.** The mechanism and plumbing
exist, but the pinned SD1.5 inpainting weights are not cached in the offline
environment.

**B5. Downstream evaluation lacks rigor.** No PatchCore-free ablation; no matched
compute; no preregistration; no hierarchical bootstrap CIs; tiny-U-Net is
unstable.

### Cross-cutting

**C1.** Tiny, development-only evaluation surfaces: 18 auto-mask pilot images and
30 held-out downstream images repeated across three seeds; no locked/external
evidence. **C2.** `auto_masks.py` ~10.9k-line monolith blocks controlled
experiments. **C3.** Track B consumes masks produced by Track A; without a frozen
mask manifest, parallel changes confound generation/downstream comparisons.
**C4.** Multiple roadmap documents contain stale, contradictory status; §5 of
this report is now the canonical execution plan.

---

## 5. Updated research plan — canonical as of 2026-07-23

This section supersedes the execution order and go/no-go language in the
historical appendix below, as well as the immediate-order sections in
`v3_research_sprint_plan.md` and `generalization_first_improvement_plan.md`.
Those documents remain useful architectural history, but their status tables
predate the validated A-S1b/A-S2 results.

### 5.1 Research position

The most defensible primary contribution is now:

> **Real-candidate calibration reduces synthetic-to-real selector error and
> converts proposal improvements that were previously unusable into selected
> pseudo-mask gains, with abstention for residual risk.**

This is stronger and more specific than “a large multimodel stack improves
industrial segmentation.” The evidence already supports the mechanism:
synthetic selector augmentation failed, real-candidate recalibration improved
MAE/correlation/regret, and the same widening operator changed from harmful to
helpful after recalibration.

The research questions should be frozen as:

1. **RQ1 — calibration:** Does development-supervised real-candidate
   calibration generalize to unseen product categories better than
   synthetic-corruption training?
2. **RQ2 — conversion:** Does calibrated selection convert proposal-ceiling
   gains into selected-Dice gains without sacrificing risk coverage?
3. **RQ3 — synthesis:** After visible-defect quality control, does synthetic
   data improve an independent student over a matched normal-only control?

Label the method accurately. Because development official masks are used after
candidate generation to calibrate the selector, the selector is
**development-supervised / label-efficient with category-held-out
generalization**, not fully unsupervised.

### 5.2 What the newest research changes

- [SubspaceAD](https://arxiv.org/abs/2602.23013) justifies PCA residual as a
  strong *standalone few-shot baseline*, not automatic inclusion in a fused
  stack. Reproduce its official protocol separately if making a comparison.
- [RadioCore](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html)
  reinforces the value of multi-scale foundation features, but its public
  repository currently says “coming soon.” Treat it as a literature comparator,
  not a dependency or another provider to implement now.
- [Boxes2Pixels](https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/html/Lendering_Boxes2Pixels_Learning_Defect_Segmentation_from_Noisy_SAM_Masks_CVPRW_2026_paper.html)
  supports treating SAM pseudo-masks as a noisy teacher. If a student is trained
  later, uncertainty pixels should be ignored/down-weighted and background
  supervision should permit one-sided correction; pseudo-masks must not be
  treated as clean ground truth.
- [MIRAGE](https://openaccess.thecvf.com/content/CVPR2026W/VAND/html/Hu_MIRAGE_Model-agnostic_Industrial_Realistic_Anomaly_Generation_and_Evaluation_for_Visual_CVPRW_2026_paper.html)
  evaluates anomaly generation on two independent axes: downstream utility and
  perceptual quality/human judgment. Track B should adopt that split because the
  current coverage proxy already demonstrated that a scalar critic can be
  gamed.

The practical consequence is a **feature freeze**: no new evidence family,
backbone, prompt module or category specialist enters the default pipeline
before the locked-category run. New papers become external baselines or
post-confirmation ablations, not automatic implementation tasks.

### 5.3 Canonical sprint tracker

| Sprint | Vertical slice | Status | Evidence / contract | Exit decision |
| --- | --- | --- | --- | --- |
| **R0** | Validated checkpoint + PCA closure | 🟡 In progress | `99b7000` is the sealed checkpoint; `a7171af` adds default-off PCA infrastructure. Record the mixed result and retain the baseline candidate pool unchanged. | Close when the PCA result, runtime limitation and non-faithful-protocol caveat are durable. Do not enable PCA by default. |
| **R1** | Selector contribution evidence pack | ⬜ Next | Freeze candidate/config/model hashes; compare synthetic selector, real-calibrated selector, and real-calibrated+widening with leave-category-out predictions. | Proceed to a primary selector claim only if hierarchical paired intervals support lower regret and higher selected Dice. |
| **R2** | Locked data + runtime environment unlock | 🚧 External setup | Acquire the 10 untouched MVTec categories, an external dataset, and the pinned SD1.5 cache; hash inventories before any run. | No thresholds or architecture changes after locked masks become readable. |
| **R3** | Locked-category mask confirmation | ⬜ Blocked by R2 | Run the frozen baseline and validated checkpoint once; official masks are evaluation-only. PCA remains off. | A generalization claim requires a positive paired effect across categories, not only a high development score. Report a null or collapse without tuning locked categories. |
| **R4** | Visibility critic validity + generation re-audit | ⬜ Blocked by R2 | First repair/test the critic contract on structured unchanged backgrounds; then run baseline vs visibility controller with blind human review. | Keep the controller only if metric change, human judgment, leakage and texture preservation agree. |
| **R5** | Independent synthetic-utility ablation | ⬜ After R4 | Same student ±synthetic, PatchCore fusion removed, matched steps, five seeds, fixed ratios, hierarchical bootstrap; include strong normal-only references. | Promote synthesis only if a preregistered regime has a positive interval. Otherwise report the null and make synthesis secondary. |
| **R6** | External baselines + paper package | ⬜ After R3/R5 | VisA/MVTec AD 2, official SubspaceAD reproduction, calibration-size curves, risk-coverage plots, manifests and failure sheets. | Claims must match the strongest completed evidence tier. |

### 5.4 Sprint R0 — close the PCA probe correctly

The current selected-Dice gain is large enough to record, but not clean enough
to deploy. The immediate engineering contract is:

- keep `pca_subspace_enabled: false` in every production/frozen config;
- rename it “SubspaceAD-inspired” in reports and code-facing documentation;
- replace `augmentation_consistency=0.75` with `None` unless consistency is
  measured;
- cache the PCA basis and leave-one-out calibration once per content-hashed
  normal set before any rerun;
- add unit-normalized subtle-anomaly, one-normal, cache
  reuse/invalidation, deterministic-LOO, and default-off/on assembly tests;
- make a future PCA test **strictly additive**: preserve every baseline candidate
  and append PCA-only/PCA-augmented candidates instead of replacing the fused
  map.

This is not the highest-priority experiment. If locked data remain unavailable,
one nested leave-category-out periodicity-gate experiment is allowed as bounded
fallback work. Predeclare the gate and stop if macro oracle gain is `<0.01`, the
selected-Dice interval includes zero, or any held-out category regresses
`>0.01`. Do not iterate the threshold on bottle/zipper.

### 5.5 Sprint R1 — package the main selector contribution

Build one reproducible command/report that emits:

- candidate-pool identity and selector parent hashes;
- leave-category-out MAE, Pearson correlation, mean regret and selected Dice;
- per-category and per-morphology paired deltas with hierarchical bootstrap
  intervals;
- calibration curves and calibration-set-size curves;
- risk-coverage/selective-Dice curves for abstention;
- the pre-recalibration, A-S1b, and A-S1b+A-S2 variants on identical candidate
  pools wherever the comparison requires identical pools.

Use `0.1074` regret as the measured result, not “low regret” without
qualification. It is the **144-image leave-category-out OOF calibration regret**;
the deployed 18-image pilot regret is higher (~0.16–0.20). It narrowly misses the
original `≤0.10` engineering gate.
Do not tune further on bottle/zipper merely to cross that round number. The
scientific gate is a reproducible category-held-out improvement with uncertainty
reported.

### 5.6 Sprints R2–R3 — locked confirmation before more architecture

Before data acquisition, write and hash a preregistration containing the
category split, primary endpoint, bootstrap unit, exclusions and failure policy.
Then run:

1. frozen pre-recalibration selector;
2. real-candidate-calibrated selector;
3. calibrated selector plus measured widening;
4. strong standalone normal-only baselines, including the official SubspaceAD
   implementation if its model/cache is available.

Report search-region recall, oracle Dice, selected Dice, regret, calibration,
coverage, selective risk, runtime and peak memory. The existing absolute targets
(macro Dice `≥0.55`, worst category `≥0.30`) remain engineering aspirations, not
publication filters. The research success criterion is a positive
category-aware paired interval for the validated checkpoint versus the frozen
baseline, with failure categories disclosed.

### 5.7 Sprint R4 — validate the critic before spending SD compute

The current visibility score still includes absolute contrast between the
generated mask region and its surrounding ring. On a structured but unchanged
object this can be high even when generated and background images are identical.
Before the SD1.5 re-audit:

- define visibility entirely from **generated-minus-background change** inside
  the mask, added/removed gradient energy, and change relative to a ring;
- add an identity test on a high-contrast structured background that must score
  near zero;
- use morphology-specific lower **and upper** visibility/fidelity bands so a
  destructive edit cannot win merely by being stronger;
- validate critic–human alignment on a stratified sample and report rank
  correlation plus disagreement cases;
- run a blinded two-reviewer study alongside automated metrics.

The keep gate is joint: visibility improves with a stratified 95% interval
excluding zero, at least 80% of accepted samples are judged visibly
defect-like by both reviewers, and leakage/texture preservation regress by no
more than `0.01`. If critic and human judgments disagree materially, stop and
repair the metric rather than tune the generator against it.

### 5.8 Sprint R5 — make or break the synthesis claim

Run the clean ablation already identified:

- identical student initialization family and matched optimization steps;
- normal-only versus normal+synthetic, with PatchCore teacher fusion disabled;
- five seeds, fixed synthetic ratios selected only on development data;
- hierarchical bootstrap over categories/images and seed-wise paired deltas;
- report pixel AP/AUPRO, image AUROC, Dice at a development-fixed threshold,
  predicted-positive rate and per-category regressions;
- evaluate perceptual quality separately from downstream utility.

If no preregistered regime beats normal-only with a 95% interval excluding zero,
the honest conclusion is valuable: synthesis is an optional/niche augmentation
path, while calibrated pseudo-mask selection remains the primary contribution.

### 5.9 Execution order and stop conditions

1. Finish R0 documentation; do not turn it into another tuning loop.
2. Build R1 now—it uses assets already present and packages the strongest
   contribution.
3. In parallel operationally, acquire/hash the locked datasets and pinned model
   caches for R2.
4. Run R3 before enabling any new default evidence or specialist.
5. Fix the visibility metric contract, then run R4 when SD1.5 is available.
6. Run R5 only on a generation corpus that passes R4.
7. Build R6 from completed evidence, including null results.

| Observation | Required decision |
| --- | --- |
| PCA strict-additive/LCO oracle gain `<0.01` | Archive PCA as a negative/mixed ablation; keep default off. |
| Selector gain disappears on locked categories | Narrow the claim to development calibration; do not tune the locked set. |
| Locked failures correlate with specialist/category cues | Remove those cues and define a new future split; do not reuse the exposed locked set for confirmation. |
| Visibility metric disagrees with blind reviewers | Invalidate the critic gate and repair the metric before generation tuning. |
| Synthetic utility interval includes zero | Reposition synthesis as secondary; do not claim that synthetic data improves detection. |
| Strong standalone baseline dominates the fused system | Report it and refocus the contribution on calibration/selection where supported. |

---

## Appendix A. Historical improvement plan — superseded

The checklist below is retained for provenance. Its open/blocked states are not
the current execution order; use §5.3 for status.

### A.1 Former improvement plan — sprint checkpoints

Contract-first vertical slices. A slice is `Done` only when every acceptance box
is checked. Track A and Track B may touch mostly separate code, but they are
data-dependent. They run in parallel only if Track B pins a frozen mask manifest,
including content hashes, generating config, and source commit.

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done
**Sprint status:** ⬜ not started · 🟡 in progress · ✅ complete

### Landed this cycle (foundation — done)

- [x] Guarded edge-aware mask refinement (`refinement.py`) added to the pool
- [x] Paired edge-vs-parent selector gate + provenance (`parent_mode`)
- [x] DINO normal-token cache + GPU cosine KNN (1.2e-7 equivalent, 3.7× warm)
- [x] Per-provider timing recorded in run metadata
- [x] Abstention (`needs_review`) + governance generation gate verified end-to-end
- [x] A7 cache/device hardening: device-honoring KNN, content-hash key, atomic writes (`70c5150`)
- [x] A-S1 best-non-edge edge gate + provenance features (`792b4e3`) — safe, but regret gate open (→ A-S1b)
- [x] B-S1 critic visibility term, validated on 150 real samples (`c4ccc37`) — controller half open
- [x] 211 automated tests passing; selector artifact byte-reproducible

> These are infrastructure/safety wins, **not** a demonstrated mask-quality gain
> (see §1 correction). The dominant levers — absolute selector calibration
> (A-S1b) and the Phase-4 visibility controller (B-S1 second half) — are open.

---

### Track A — mask / selector / runtime

#### 🟡 A-S1 · Make the edge gate baseline-preserving  *(fixes A1 — edge half; commit `792b4e3`)*
Edge-vs-baseline gate implemented and safe; the regret gate is blocked on the
absolute selector (see A-S1b).

**Contract**
- [x] Every refined candidate records `parent_mode`, refinement family, and the best non-edge baseline used for comparison
- [x] Selector returns predicted edge-vs-baseline gain, one-sided uncertainty bound, eligibility decision, and fallback reason

**Tasks**
- [x] Train paired differences against the **best non-edge candidate** per corruption (not only the parent)
- [x] Add refinement-family / provenance features + parent-relative measurement deltas
- [x] Calibrate residuals out-of-fold by development category (paired residual 0.084)
- [~] Abstain when expected-IoU, precision/recall projection, and paired predictions disagree materially *(inconsistency computed; abstention not yet wired)*

**Acceptance / exit gate**
- [x] Regression test: edge candidate beats weak parent but loses to a stronger non-edge candidate → non-edge chosen
- [x] No development category regresses > 0.01 Dice vs edge-off *(on_v4 bottle 0.479→0.516, zipper 0.254→0.261)*
- [ ] Mean selection regret `0.1808 → ≤ 0.10` — **NOT met (still ~0.20–0.25)**; dominated by absolute-selector mis-ranking → moved to A-S1b
- [ ] Expected-IoU calibration MAE `≤ 0.15` on development data → A-S1b

#### ✅ A-S1b · Fix selector calibration via real-candidate recalibration  *(fixes A1 — VALIDATED, deployed)*
**Result (the first validated, generalizing, output-metric win of the plan).**
Built a real-candidate calibration set (candidates generated without masks on 5
dev categories, 4,966 candidates over 144 images; official masks read only
afterward for IoU labels — `scripts/build_real_selector_calibration.py`,
`configs/as1b_calib.yaml`). Refit the selector heads on real rows.

Leave-category-out (each category scored by a model trained on the other four):
- IoU calibration MAE `0.091 → 0.056`, Pearson `0.591 → 0.699`
- Mean selection regret `0.1985 → 0.1074` (144 images)
- **End-to-end, bottle/zipper HELD OUT of training**, selected Dice
  `bottle 0.5155→0.5350`, `zipper 0.2614→0.2996` (macro `0.3885→0.4173`, +0.029);
  regret `bottle 0.2475→0.2280`, `zipper 0.1950→0.1569`; better on 10/18 samples.

Three consistent signals across sample sizes (calibration on 4,966 candidates,
regret on 144 images, LCO end-to-end Dice on 18) — robust, unlike isolated n=18
deltas. The full 5-category real selector is **deployed** to the dev selector
path (synthetic backed up as `generic_selector.synthetic_backup.joblib`).

**Corrections this established:** (1) the earlier "selector near-random, Pearson
0.086" was a wrong-slice artifact (selected-only, n=18); full-pool Pearson is
0.59. (2) The synthetic over-seg augmentation route was a proven **negative**
(regret 0.1808→0.2702; reverted). Real-candidate recalibration is the fix.

**Also ruled out — heuristic-swap shortcut (experiment 1a):** on the identical
candidate pool, disabling the learned bundle (→ hand-designed heuristic) is far
*worse*, not better: Dice 0.3885→0.1869, regret 0.221→0.423, recall 0.576→0.207,
worse on 11/18 samples (pools verified identical, oracle Δ=0). So despite its
poor point-calibration (Pearson 0.086 on selected samples), the learned selector
**ranks meaningfully better than the heuristic and is net-positive** — it must be
*improved* via real-data recalibration, not replaced. There is no free
selection boost; 1b is the only route to the ~0.22 selected-vs-oracle headroom.

**Firewall contract (must hold for the calibration set — do not violate):**
- [ ] Generate candidates **without** official masks (masks touched only after generation).
- [ ] Use **development** official masks only afterward to compute IoU labels.
- [ ] Train and evaluate with **leave-category-out** splits.
- [ ] Record candidate / config / model hashes for every calibration artifact.
- [ ] **Never** inspect or tune against locked-category masks.

**Contract**
- [ ] Selector exposes a calibrated reliability + an "unreliable prediction" flag per candidate

**Tasks**
- [x] Diagnose the inversion: absolute model rates a broad (area 0.59, cov 0.57) SAM mask at 0.61 while compact good candidates (area 0.1–0.19, cov 0.9, actual ~0.80) get ~0.20. Root cause: synthetic training contains no broad over-segmentation candidates → model learned "bigger → higher IoU"
- [x] ~~Inject synthetic over-segmentation negatives (dilated-truth + low-quantile broad, true low IoU)~~ **TRIED → FAILED**: mean regret 0.1808→0.2702; broke the previously-good `broken_small` (0.798→0.246). Reverted. The synthetic corruptions still don't match real candidate statistics, so more synthetic negatives don't close the gap.
- [ ] **Next: recalibrate on dev-category REAL candidates** — fit an isotonic/quantile map from a small held-out dev-category candidate set with official-mask IoU (allowed on development categories), rather than more synthetic data
- [ ] Consider abstention when direct vs precision/recall IoU disagree — but note the inversion is *confident* (low inconsistency), so abstention alone won't recover regret
- [ ] Leave-category-out regret from OOF predictions as the validation signal

**Acceptance / exit gate**
- [ ] Mean selection regret `≤ 0.10`; `broken_large`/`broken_teeth` regret `< 0.20`
- [ ] Expected-IoU calibration MAE `≤ 0.15`; leave-category-out regret `≤ 0.05`

#### ✅ A-S2 · Restore proposal recall for repeated-structure defects  *(fixes A2; VALIDATED, enabled after A-S1b)*
> **Diagnosis:** the oracle-zero cases are **confidently wrong Qwen localization**
> (`qwen_region_recall = 0`, `loc_status = valid`) whose box the soft prior fences
> to, suppressing the true defect (the *fallback*/full-image sample scored oracle
> 0.60 vs 0.0 for confident-wrong). A confident-wrong box is worse than no box.
>
> **Fix:** a measured, category-agnostic trigger (`widen_on_repeated_texture`;
> repeated-texture ≥ 0.17 — bottle ≤0.11 vs zipper ≥0.23) widens to full image so
> evidence is not fenced. Recovers zipper oracle 0.4565→0.5750 (dead samples
> `broken_teeth/001` 0→0.658, `split_teeth/000` 0→0.429); bottle untouched.
>
> **Gate resolved by A-S1b:** with the *synthetic* selector it regressed selected
> Dice (zipper 0.2614→0.1906); with the deployed **real-recalibrated** selector it
> now **improves** it (zipper 0.3083→0.3823, bottle unchanged). Enabled by default.
> Combined A-S1b + A-S2: zipper selected Dice **0.2614 → 0.3823 (+0.121)**.

**Tasks**
- [x] Diagnose where signal is lost → localization fencing (confident-wrong Qwen + soft prior), not thresholding/filtering
- [x] Add a measured-repetition localization-widening trigger (default off); verified oracle recovery 0→0.43–0.66
- [ ] Re-enable once the selector is recalibrated (A-S1b); prefer widening to the **periodic extent**, not the full image, to avoid feeding broad candidates to selection
- [ ] Emit `no_recall_candidate` diagnostic when all candidates lack evidence support

**Acceptance / exit gate**
- [x] `split_teeth`/`broken_teeth` oracle Dice `> 0.25` (0.429 / 0.658 with widening)
- [ ] Selected Dice non-regressing on the widened cohort — **blocked on A-S1b**
- [ ] No existing bottle proposal family loses oracle Dice
- [ ] Zero-oracle sample triggers abstention, not a pseudo-label

#### ⬜ A-S3 · Control over-segmentation without sacrificing recall  *(fixes A3)*
> **Tried → null (do not repeat as an additive fix).** An additive proposal-time
> SAM2 evidence-trim (intersect the SAM mask with fused ≥ 0.5) left oracle Dice
> **unchanged** (macro 0.6097 → 0.6097) and selected essentially flat
> (0.3885 → 0.3887); reverted. Reason: over-expansion is a **selection** error,
> not candidate availability — a good candidate already exists (`broken_large`
> oracle ~0.80), but the miscalibrated selector still over-rates the broad SAM
> mask. So A-S3 is gated by A-S1b (real-candidate recalibration); a *reject*
> variant would need a normal-only threshold and is overfit-prone until the
> selector is calibrated.

**Contract**
- [ ] Each candidate reports component count, parent-relative area growth, concentration, and normal-memory support; suppression is measurement-driven

**Tasks**
- [ ] Precision-risk estimates for large-area / fragmented masks
- [ ] Parent-relative area-growth penalties for SAM and edge refinements
- [ ] Split diffuse evidence from connected cores; keep uncertainty as soft mask
- [ ] Require stronger multi-provider agreement outside the Qwen region when localization is valid

**Acceptance / exit gate**
- [ ] Heavy precision `0.3231 → ≥ 0.4255` (edge-off)
- [ ] Recall retained `≥ 0.50`
- [ ] `broken_large` selection regret `< 0.20`

#### ⬜ A-S4 · Make QC actionable  *(fixes A5)*
**Contract**
- [ ] Three outcomes: accept hard mask / retain soft only / abstain-retry, with calibrated failure probabilities and concrete reasons

**Tasks**
- [ ] Fit QC thresholds on held-out development categories using real mask-quality outcomes
- [ ] Incorporate localization fallback, selector inconsistency, predicted precision, area, fragmentation, oracle-unavailable proxies
- [ ] Morphology-aware retry routing (localization retry / proposal expansion / precision tighten / abstain)

**Acceptance / exit gate**
- [ ] Strong `broken_small` and zero-Dice `split_teeth` get different dispositions
- [ ] Accepted masks have materially higher Dice than warned/abstained
- [ ] No zero-evidence mask published as a hard pseudo-label

#### 🟡 A-S5 · Remove cold-runtime bottlenecks + harden contracts  *(fixes A6, A7; A7 done in `70c5150`)*
Cache/device contracts (A7) hardened and tested; MuSc caching (A6) remains.

**Contract**
- [x] Provider timing recorded per source in metadata *(prep/inference split still coarse)*
- [x] Cache identity includes normal-set content, processor identity, model-config identity, scale/layers, dtype

**Tasks**
- [ ] Cache MuSc normal-patch memory + LOO calibration per normal set *(A6 — MuSc is now ~80% of cold cost)*
- [ ] Reuse cached DINO normal tokens inside registered residual
- [x] GPU KNN honors the resolved provider device (no CUDA when CPU configured)
- [x] Atomic cache writes + corruption recovery *(bounded GPU-OOM fallback still open)*

**Acceptance / exit gate**
- [ ] Cold `< 30 s/image` (currently ~76 s — needs MuSc caching); warm replay `< 5 s/image` (≈3.3 s ✓)
- [x] GPU/NumPy distances agree `1e-6`
- [x] Explicit CPU mode performs no CUDA allocation
- [x] Same-path content change invalidates in-memory and disk caches

#### ⬜ A-S6 · Package + run the confirmatory development experiment  *(fixes A8, A9, C1-partial)*
**Contract**
- [ ] All configs tracked; inference manifest declares selector-training manifest + selector hash as parents; reports derive sample counts from metadata

**Tasks**
- [ ] Commit the V3 / heavy experiment configs intentionally
- [ ] Fix the non-writable-array warning (materialize writable arrays at the boundary)
- [ ] Run all 18 bottle/zipper images with identical off/on/heavy cohorts
- [ ] Report per-category / per-morphology results with paired bootstrap CIs
- [ ] Leave locked categories untouched

**Acceptance / exit gate**
- [ ] Full-cohort result meets §7 go/no-go

---

### Track B — generation / downstream (parallel)

#### 🟡 B-S1 · Visibility/fidelity critic + visibility-targeted controller  *(fixes B2; critic `c4ccc37`, controller `36a6afd`)*
Critic + controller mechanism implemented and tested; empirical tuning (bands,
threshold) needs an SD re-audit run.

**Contract**
- [x] Add a visibility term to `iadgen_v2/generation_critic.py` (in-mask deviation magnitude, added edge/gradient energy, local contrast vs a surrounding normal ring)
- [x] Controller drives visibility (rank tiebreak + visibility-triggered retry); coverage demoted to a guard (weight 0.22→0.12 + opt-in reject gate)
- [ ] Freeze the exact Track-A mask manifest/hash consumed by this experiment

**Tasks**
- [x] Implement the visibility/fidelity metric + unit tests against known visible/invisible edits
- [x] Validate on real corpus: genuine defect 0.6–0.79 vs current corpus median 0.25 (150 samples); gate@0.35 flags 97%
- [x] Retarget the Phase-4 controller objective to visibility (rank + retry); per-attempt reason logging
- [ ] Preregister morphology-specific bands from real-defect reference patches and labeled fixtures
- [ ] SD re-audit: set `target_defect_visibility` / `min_defect_visibility_score`, regenerate, two-reviewer blind check

**Status: 🚧 BLOCKED on the pinned SD1.5 cache.** The empirical re-audit needs
SD1.5 inpainting weights available offline; `phase3-train` fails with
`LocalEntryNotFoundError` (network disabled). Mechanism + config plumbing are in
place and the gate defaults off, so nothing regresses. Resume when SD1.5 is
cached: run the two configs (baseline vs `target_defect_visibility` on) through
prepare→phase2→phase3→phase4 and compare visibility + blind review.

**Acceptance / exit gate**
- [ ] Mean visibility improves by `≥ 0.020` over the frozen 0.0551 baseline, with a morphology-stratified 95% bootstrap CI excluding zero
- [ ] `≥ 80%` of the stratified accepted sample is judged visibly defect-like by both reviewers
- [ ] Leakage and texture-preservation means each regress by no more than `0.01`

#### ⬜ B-S2 · Preregistered downstream ablation  *(fixes B1, B3)*
**Contract**
- [ ] Same student, ±synthetic, **PatchCore fusion removed**, matched training compute, 5 seeds, hierarchical bootstrap CIs; regimes + primary metric preregistered

**Tasks**
- [ ] Build the ablation harness (fusion-off student, matched steps)
- [ ] Preregister regimes/metric; select regime on development
- [ ] Run once on locked evaluation data

**Acceptance / exit gate**
- [ ] A regime where synthetic beats normal-only with 95% CI excluding zero — or the null reported and synthesis repositioned as optional augmentation

---

### Cross-cutting (after both tracks stabilize)

#### ⬜ X1 · Locked-category + external generalization  *(fixes C1)*
- [ ] Freeze thresholds/config/model; run 10 untouched MVTec categories, then VisA + MVTec AD 2; unlock masks only for evaluation
- [ ] Before unlocking data, preregister the initial engineering gates: macro Dice `≥ 0.55`; every category `≥ 0.30`; dev→locked gap `≤ 0.10`
- [ ] Record that these are proposed product/research gates, not thresholds inferred from the locked data; justify or revise them only before evaluation

#### ⬜ X2 · Finish modularizing the auto-mask core  *(fixes C2)*
- [ ] Move MuSc / registration / residual builders into `auto_mask/evidence/` providers behind the common API; freeze behavior with regression tests
- [ ] Default core has no category-name conditionals; specialists are opt-in; pilot metrics reproduce within tolerance

---

### A.2 Former execution order

Conditional parallelism: freeze and hash the Track-A mask artifact consumed by
Track B. If masks must change, rerun affected Track-B stages after Track A
stabilizes.

Both dominant levers are now **blocked on data/environment, not code** — the
correct next work is setup + data acquisition, then build each harness and run
it **together** (avoid another "implemented but not demonstrated" landing).

1. [~] **A-S1** edge-baseline gate done (`792b4e3`); **A-S1b** 🚧 blocked — synthetic route failed (negative result); needs a dev-category real-candidate calibration set
2. [~] **B-S1** critic + controller done (`c4ccc37`, `36a6afd`); 🚧 empirical re-audit blocked on the pinned SD1.5 cache
3. [ ] **A-S2** split-tooth proposal recall
4. [ ] **A-S3 / A-S4** over-segmentation control + actionable QC
5. [~] **A-S5** A7 cache/device hardening done (`70c5150`); MuSc caching (A6) open
6. [ ] **A-S6 + B-S2** full 18-image paired mask experiment + preregistered ablation
7. [ ] **X1 / X2** locked/external evaluation + modularization (architecture frozen first)

### A.3 Former go/no-go before merging or enabling edge refinement by default

- [ ] Overall selected Dice exceeds edge-off by `≥ 0.015` on the full 18-image cohort, with the improvement's 95% bootstrap CI excluding zero
- [ ] No development category regresses by more than `0.01` Dice
- [ ] Mean selection regret `≤ 0.10`; expected-IoU calibration MAE `≤ 0.15`
- [ ] No sample has zero oracle Dice without an explicit abstention outcome
- [ ] QC acceptance is meaningfully correlated with actual quality
- [ ] Cold `< 30 s/image`, warm `< 5 s/image`; runs warning-free, manifest-linked, byte-reproducible
- [ ] (Track B) accepted synthetic sample shows visible defect structure; downstream benefit confirmed with a CI excluding zero, or the null reported

### A.4 Former final recommendation

Do not merge or market the current branch as a mask-quality improvement. Retain
the runtime changes because their performance mechanism is validated, but merge
them only after A-S5 correctness tests pass or after splitting out an
independently safe subset. Keep the selector and edge-refinement behavior behind
an experimental flag until A-S1–A-S4, A-S6, and §7 pass. Treat B-S1 and B-S2 as
the gate on any Track-D ("synthetic augmentation helps") claim; run them in
parallel only against a frozen mask manifest. The mask work alone does not make
that claim defensible.

### A.5 Former stop/go honesty

- If B-S2 shows synthetic never beats normal-only on these categories, that is a
  legitimate result: reposition synthesis as niche augmentation, not the main
  detector.
- If locked-category masks collapse (X1), remove specialist assumptions rather
  than tune locked categories — the firewall forbids the latter.
