# IADGen v2 — Current Weaknesses and Improvement Plan (Merged)

**Assessment date:** 2026-07-22
**Reviewed branch:** `paired-edge-selector` at `e992272`
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

**A1. Selector calibration is the dominant quality bottleneck.**
Heavy oracle Dice `0.5545` vs selected `0.3737`; regret concentrated in
`broken_large` (0.5015) and `broken_teeth` (0.3957). Expected-IoU is not merely
mis-scaled, it is **inverted**: `broken_large` predicted 0.6115 / actual 0.1787;
`broken_small` predicted 0.0733 / actual 0.6643. The paired gate governs only
edge-vs-parent; once eligible, an edge candidate still competes against a
stronger non-edge candidate through this unreliable absolute selector.

**A2. Proposal recall is zero for `zipper/split_teeth`.** Selected and oracle
Dice both 0.0 — selection cannot repair it.

**A3. Heavy evidence raises recall by over-segmenting.** Recall 0.5672 but
precision 0.3231; `broken_large` recall 0.97 / precision 0.18.

**A4. Qwen localization is unreliable on half the cohort.** 3/6 valid; the rest
fall back to full-image.

**A5. Internal QC does not discriminate usable from unusable masks.** All six
outputs are `warning`, including the strong `broken_small` (0.798) and the total
`split_teeth` failure (0.0).

**A6. MuSc is the cold-runtime bottleneck.** DINO ~2.2–3.68 s (cached); MuSc
18.9–111.3 s/image; registered residual ~8.7–9.7 s.

**A7. Cache and device contracts are not hardened.** GPU KNN ignores an explicit
CPU device; the in-process token key is path-based (stale on same-path content
change); the disk key omits resolved model revision / processor identity;
token-cache writes are non-atomic. These are merge blockers for the affected
runtime changes because they can silently select the wrong backend or reuse stale
evidence.

**A8. Experiment packaging is incomplete.** `configs/ab_edge_heavy.yaml` and
`configs/ab_edge_refine_on_v3.yaml` are untracked; the inference manifest does
not declare the selector-training manifest / hash as a parent input.

**A9. Reporting/runtime warnings reduce confidence.** Hardcoded "three
deterministic samples per defect type"; torchvision non-writable-array warning;
warm-cache timing mistaken for throughput.

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

**B3. Downstream evaluation lacks rigor.** No PatchCore-free ablation; no matched
compute; no preregistration; no bootstrap CIs; tiny-U-Net unstable.

### Cross-cutting

**C1.** Tiny, development-only evaluation surfaces: 18 auto-mask pilot images and
30 held-out downstream images repeated across three seeds; no locked/external
evidence. **C2.** `auto_masks.py` ~10.9k-line monolith blocks controlled
experiments. **C3.** Track B consumes masks produced by Track A; without a frozen
mask manifest, parallel changes confound generation/downstream comparisons.

---

## 5. Improvement plan — sprint checkpoints

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

#### ⬜ A-S1b · Fix absolute selector calibration  *(fixes A1 — dominant half)*
Revealed by A-S1: regret is dominated by the absolute IoU model mis-ranking
NON-edge candidates (inversions: `broken_large` pred 0.61/actual 0.18;
`broken_small` pred 0.07/actual 0.66). The edge gate cannot touch this.

**Contract**
- [ ] Selector exposes a calibrated reliability + an "unreliable prediction" flag per candidate

**Tasks**
- [ ] Diagnose the synthetic→real domain gap driving the inversions (feature drift, corruption realism)
- [ ] Recalibrate the absolute IoU head (richer/real-defect-like corruptions and/or isotonic recalibration on held-out dev categories)
- [ ] Rank/abstain on the conformal lower bound when direct vs precision/recall IoU disagree materially
- [ ] Leave-category-out regret from OOF predictions as the validation signal

**Acceptance / exit gate**
- [ ] Mean selection regret `≤ 0.10`; `broken_large`/`broken_teeth` regret `< 0.20`
- [ ] Expected-IoU calibration MAE `≤ 0.15`; leave-category-out regret `≤ 0.05`

#### ⬜ A-S2 · Restore proposal recall for split-tooth defects  *(fixes A2)*
**Contract**
- [ ] Recall-safe, Qwen-independent candidate family for repeated / interrupted linear structures, activated by measured repetition & continuity breaks (no category names)

**Tasks**
- [ ] Diagnose where `split_teeth` signal is lost (pre-fusion / thresholding / component filtering)
- [ ] Add multi-threshold connected-chain candidates over the full evidence field when Qwen region is narrow/invalid
- [ ] Preserve small aligned components forming a coherent broken sequence
- [ ] Emit `no_recall_candidate` diagnostic when all candidates lack evidence support

**Acceptance / exit gate**
- [ ] `split_teeth` oracle Dice `> 0.25` on development samples
- [ ] No existing bottle proposal family loses oracle Dice
- [ ] Zero-oracle sample triggers abstention, not a pseudo-label

#### ⬜ A-S3 · Control over-segmentation without sacrificing recall  *(fixes A3)*
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

#### 🟡 B-S1 · Visibility/fidelity critic + visibility-targeted controller  *(fixes B2; critic half in `c4ccc37`)*
Critic term implemented and empirically validated; controller retarget remains.

**Contract**
- [x] Add a visibility term to `iadgen_v2/generation_critic.py` (in-mask deviation magnitude, added edge/gradient energy, local contrast vs a surrounding normal ring)
- [~] Controller drives visibility into a morphology-specific band; coverage demoted to a min/max guard *(critic weight demoted 0.22→0.12 + opt-in gate added; Phase-4 controller retarget pending)*
- [ ] Freeze the exact Track-A mask manifest/hash consumed by this experiment

**Tasks**
- [x] Implement the visibility/fidelity metric + unit tests against known visible/invisible edits
- [x] Validate on real corpus: genuine defect 0.6–0.79 vs current corpus median 0.25 (150 samples); gate@0.35 flags 97%
- [ ] Preregister morphology-specific bands from real-defect reference patches and labeled fixtures
- [ ] Retarget the Phase-4 controller objective to visibility; per-attempt reason logging
- [ ] Re-audit a stratified sample with two blinded reviewers; tune `min_defect_visibility_score`

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

## 6. Execution order

Conditional parallelism: freeze and hash the Track-A mask artifact consumed by
Track B. If masks must change, rerun affected Track-B stages after Track A
stabilizes.

1. [~] **A-S1** edge-baseline gate done (`792b4e3`); **A-S1b** absolute-selector calibration is the open regret lever
2. [~] **B-S1** critic term done (`c4ccc37`); Phase-4 visibility controller + re-audit is the open half
3. [ ] **A-S2** split-tooth proposal recall
4. [ ] **A-S3 / A-S4** over-segmentation control + actionable QC
5. [~] **A-S5** A7 cache/device hardening done (`70c5150`); MuSc caching (A6) open
6. [ ] **A-S6 + B-S2** full 18-image paired mask experiment + preregistered ablation
7. [ ] **X1 / X2** locked/external evaluation + modularization (architecture frozen first)

## 7. Go/no-go before merging or enabling edge refinement by default

- [ ] Overall selected Dice exceeds edge-off by `≥ 0.015` on the full 18-image cohort, with the improvement's 95% bootstrap CI excluding zero
- [ ] No development category regresses by more than `0.01` Dice
- [ ] Mean selection regret `≤ 0.10`; expected-IoU calibration MAE `≤ 0.15`
- [ ] No sample has zero oracle Dice without an explicit abstention outcome
- [ ] QC acceptance is meaningfully correlated with actual quality
- [ ] Cold `< 30 s/image`, warm `< 5 s/image`; runs warning-free, manifest-linked, byte-reproducible
- [ ] (Track B) accepted synthetic sample shows visible defect structure; downstream benefit confirmed with a CI excluding zero, or the null reported

## 8. Final recommendation

Do not merge or market the current branch as a mask-quality improvement. Retain
the runtime changes because their performance mechanism is validated, but merge
them only after A-S5 correctness tests pass or after splitting out an
independently safe subset. Keep the selector and edge-refinement behavior behind
an experimental flag until A-S1–A-S4, A-S6, and §7 pass. Treat B-S1 and B-S2 as
the gate on any Track-D ("synthetic augmentation helps") claim; run them in
parallel only against a frozen mask manifest. The mask work alone does not make
that claim defensible.

## 9. Stop/go honesty

- If B-S2 shows synthetic never beats normal-only on these categories, that is a
  legitimate result: reposition synthesis as niche augmentation, not the main
  detector.
- If locked-category masks collapse (X1), remove specialist assumptions rather
  than tune locked categories — the firewall forbids the latter.
