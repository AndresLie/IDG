# IADGen v2 — Current Weaknesses and Improvement Plan (Merged)

**Assessment date:** 2026-07-26
**Reviewed branch:** `paired-edge-selector` through `8b26b98`, including the
committed R4 visibility, R5 utility, R6 evidence, and VisA retention work
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

## 1c. R4 visibility-controller re-audit (2026-07-24)

The previously blocked SD1.5 audit is now executable from a pinned offline fp16
snapshot:

```text
stable-diffusion-v1-5/stable-diffusion-inpainting
revision 8a4288a76071f7280aedbdb3253bdb9e9d5d84bb
snapshot inventory f1196169cd575ffdf23309264dd39ad7391633a817f94fcaa9331b72b0fa4db7
```

A matched 18-sample pilot compared one-attempt generation against a
morphology-banded controller with up to three attempts. Inputs, masks, prompts,
profiles, and base seeds were identical.

| R4 metric | Paired delta | Morphology-stratified 95% CI |
| --- | ---: | ---: |
| Critic visibility | `+0.0117` | `[+0.0018, +0.0220]` |
| Adaptive mask coverage | `+0.0665` | `[+0.0281, +0.1142]` |
| Texture preservation | `-0.0000` | `[-0.0024, +0.0026]` |
| Leakage score | `-0.0204` | `[-0.0262, -0.0149]` |
| Critic score | `+0.0039` | `[-0.0011, +0.0100]` |
| Latency | `+1.9478 s/sample` | `[+1.8008, +2.1165]` |

Acceptance moved only from `10/18` to `11/18`, while attempts increased from
`18` to `45`. The controller helped metal-nut visibility, regressed tile
visibility/leakage, and left wood at `0/6` accepted. The automated keep gate
therefore **failed**: visibility did not reach the required `+0.020`, and
leakage regressed by more than `0.01`.

The result is a useful negative finding. Global strength/guidance escalation is
not the correct remedy for under-edited wood masks. The next bounded R4
experiment should use mask-local high-resolution generation or latent
reinjection, with the same leakage constraint; do not proceed to R5 on this
corpus.

That mask-local crop experiment was then executed as a strict one-attempt
ablation. It also failed:

| Mask-local metric | Paired delta | Morphology-stratified 95% CI |
| --- | ---: | ---: |
| Critic visibility | `-0.0510` | `[-0.0642, -0.0385]` |
| Adaptive mask coverage | `-0.2065` | `[-0.2786, -0.1400]` |
| Texture preservation | `+0.1040` | `[+0.1014, +0.1066]` |
| Leakage score | `+0.0234` | `[+0.0178, +0.0288]` |

Strict soft-envelope compositing solved leakage by attenuating the desired edit
too. Acceptance fell `10/18→6/18`, and wood remained `0/6`. Keep
`masked_crop_generation` default-off.

Two existing conditioning families were then tested:

- `fixed_mask_adapter` was positive but negligible: visibility `+0.0020`,
  coverage `+0.0063`, acceptance `10/18→12/18`, and wood remained `0/6`.
  The trained adapter gate is only `0.1185`.
- `clone_harmonized` failed as a global replacement but exposed a useful
  interaction. It regressed metal nut and tile, while wood improved visibility
  `+0.0868`, coverage `+0.3899`, and acceptance `0/6→4/6`.

A category-agnostic critic arbitration over text-only and clone-harmonized
outputs therefore selected text-only for all 12 metal/tile cases and clone for
all 6 wood cases. It is the first R4 candidate to pass every automated gate:

| Arbitrated metric | Paired delta | Morphology-stratified 95% CI |
| --- | ---: | ---: |
| Critic visibility | `+0.0289` | `[+0.0110, +0.0487]` |
| Adaptive mask coverage | `+0.1300` | `[+0.0510, +0.2126]` |
| Texture preservation | `+0.0005` | `[-0.0002, +0.0016]` |
| Leakage score | `-0.0022` | `[-0.0043, -0.0004]` |

Acceptance is `14/18` from 36 generated candidates and total latency rises
`+0.9618 s/sample`. This is **promising diagnostic routing evidence, not an
independent validation**: the same critic selects and scores the output. R4
remains open until two blind reviewers agree and a held-out utility test
confirms the selected corpus.

The current candidate was rerun from the same frozen inputs on 2026-07-24.
All 18 text-only images, all 18 clone-harmonized images, and all 18 arbitrated
outputs were byte-identical to the checkpoint. Every critic metric, acceptance
decision, and arbitration decision reproduced exactly; the routing remained
`12` text-only and `6` clone-harmonized. The automated gate therefore passes
again with identical quality deltas. Only wall-clock timing moved:
two-arm compute changed from `2.5911` to `2.5307 s/sample`, while peak CUDA
memory was unchanged. This validates deterministic replay, not independent
quality, so the blind and downstream gates remain open.

Artifacts:

```text
reports/r4_visibility_reaudit/comparison/r4_visibility_reaudit.md
reports/r4_visibility_reaudit/comparison/r4_visibility_comparison.png
reports/r4_visibility_reaudit/comparison/r4_visibility_blind_audit.png
reports/r4_visibility_reaudit/comparison/paired_metrics.csv
reports/r4_visibility_reaudit/mask_local_comparison/r4_visibility_reaudit.md
reports/r4_visibility_reaudit/mask_local_comparison/r4_visibility_comparison.png
reports/r4_visibility_reaudit/fixed_adapter_comparison/r4_visibility_reaudit.md
reports/r4_visibility_reaudit/clone_harmonized_comparison/r4_visibility_reaudit.md
reports/r4_visibility_reaudit/arbitrated_comparison/r4_visibility_reaudit.md
reports/r4_visibility_reaudit/arbitrated_comparison/r4_visibility_blind_audit.png
reports/r4_visibility_reaudit/rerun_comparison_20260724/r4_current_vs_previous.md
```

## 1d. R5 independent synthetic-utility ablation (2026-07-26)

R5 is now complete as a bounded development experiment. The run used an
independent, fusion-free ResNet18 U-Net student and compared real-only training
against the same real data plus the 14 R4 critic-accepted synthetic samples at
ratio `0.25`.

The fairness contract passed:

- the two arms used paired initialization seeds;
- both arms performed exactly `120` optimizer steps;
- architecture and held-out anomaly/normal sets were identical;
- no PatchCore label refinement, teacher loss, or score fusion was active;
- five seeds were evaluated over `metal_nut`, `tile`, and `wood`.

The primary endpoint and gate were frozen before execution. The primary metric
was category-macro pixel AP with a seed/category/image hierarchical bootstrap.
The preregistered promotion gate required a mean gain of at least `0.02` and a
95% interval whose lower bound exceeded zero.

| R5 metric | Paired delta | Hierarchical 95% CI |
| --- | ---: | ---: |
| Pixel AP | `-0.0200` | `[-0.0591, +0.0171]` |
| AUPRO | `-0.0676` | `[-0.1493, +0.0027]` |
| Pixel AUROC | `-0.0426` | `[-0.1133, +0.0041]` |
| Dice | `+0.0060` | `[-0.0069, +0.0212]` |
| Image AUROC | `+0.0420` | `[-0.0826, +0.1627]` |
| Predicted-positive rate | `+0.0118` | `[-0.0472, +0.0880]` |

The joint gate **fails**. Synthetic augmentation does not improve the primary
ranking metric, and every interval includes zero. The small Dice increase is
not confirmatory: it occurs alongside a higher predicted-positive rate and
lower pixel AP/AUPRO, while prediction sheets still show broad texture and
border responses rather than consistently tighter defect localization.

An immediate rerun from identical code, config, data, model, split, and
synthetic-metadata fingerprints reproduced the failed decision but not the exact
metrics:

| R5 rerun metric | Previous delta | Current delta | Current 95% CI |
| --- | ---: | ---: | ---: |
| Pixel AP | `-0.0200` | `-0.0066` | `[-0.0573, +0.0380]` |
| AUPRO | `-0.0676` | `-0.0322` | `[-0.1275, +0.0602]` |
| Pixel AUROC | `-0.0426` | `-0.0167` | `[-0.0634, +0.0288]` |
| Dice | `+0.0060` | `+0.0002` | `[-0.0158, +0.0179]` |

Every first training loss was identical, while later losses and thresholds
drifted. This localizes the remaining reproducibility weakness to the neural
training trajectory, most likely nondeterministic CUDA operations: the seed
helper seeds all random-number generators but does not enforce deterministic
Torch/cuDNN algorithms. The exact R5 effect size is therefore not ready for a
single-number claim. The decision is stable across both executions: the primary
effect is negative, every interval includes zero, and the promotion gate fails.

That correctness gap is now fixed behind an opt-in Phase 5 contract. Strict
mode enables deterministic Torch algorithms, deterministic cuDNN, the
`:4096:8` cuBLAS workspace, disables cuDNN benchmarking and TF32, and records
all settings plus a content hash of each realized training schedule.

Two fresh CUDA executions of the bounded replay config passed every predeclared
identity check:

```text
15/15 report artifacts byte-identical
segmentation CSV byte-identical
per-image metrics byte-identical
prediction contact sheets byte-identical
run status byte-identical
```

This proves the new deterministic mechanism on a one-seed, 12-step replay. It
does not make synthesis effective. The full deterministic five-seed R5
re-estimation was subsequently frozen and executed under a new architecture
tag:

| Deterministic R5 metric | Paired delta | Hierarchical 95% CI |
| --- | ---: | ---: |
| Pixel AP | `-0.0015` | `[-0.0414, +0.0403]` |
| AUPRO | `-0.0436` | `[-0.1504, +0.0594]` |
| Pixel AUROC | `-0.0302` | `[-0.1326, +0.0636]` |
| Dice | `+0.0011` | `[-0.0098, +0.0137]` |
| Image AUROC | `-0.0393` | `[-0.1435, +0.0723]` |

The deterministic fairness contract passes, but the utility gate fails again.
The preferred exact estimate is now effectively zero, with an interval spanning
meaningful harm and benefit. This strengthens the null conclusion: execution
drift is fixed, yet there is still no evidence that the selected synthetic
corpus improves independent segmentation.

This result closes the immediate Track-B question without hiding a negative
outcome:

```text
the R4-selected synthetic corpus is not independently useful under the
preregistered R5 regime; synthesis remains optional and cannot support a
"synthetic data improves detection" claim.
```

The locked R5 evaluation is not justified because the development promotion
gate failed. Do not tune ratios, thresholds, or generator parameters against
this completed cohort. R4 blind review remains useful for assessing perceptual
critic validity, but even a passing human review would not reverse the
downstream null measured here.

Artifacts:

```text
reports/r5_independent_synthetic_utility/preregistration.md
reports/r5_independent_synthetic_utility/phase5/qwen/summary.md
reports/r5_independent_synthetic_utility/phase5/qwen/segmentation_results.csv
reports/r5_independent_synthetic_utility/phase5/qwen/synthetic_selector_report.md
reports/r5_independent_synthetic_utility/utility_review/r5_synthetic_utility_review.md
reports/r5_independent_synthetic_utility/utility_review/r5_synthetic_utility_review.json
reports/r5_independent_synthetic_utility/rerun_comparison_20260726/r5_current_vs_previous.md
reports/r5_deterministic_replay/replay_contract.md
reports/r5_deterministic_replay/replay_result.md
reports/r5_independent_synthetic_utility_deterministic/preregistration.md
reports/r5_independent_synthetic_utility_deterministic/comparison/r5_deterministic_vs_historical.md
```

## 1e. R6 evidence package and external-baseline readiness (2026-07-26)

R6 now consolidates the sealed R1-R5 evidence through one governed,
hash-verifying command:

```bash
python -m iadgen_v2.cli r6-evidence-package \
  --config configs/r6_evidence_package.yaml
```

All six configured source artifacts passed their pinned SHA-256 checks. The
package is deterministic, and its 13 generated evidence artifacts pass their
own `artifact_hashes.sha256` verification.

The claim ledger reaches a deliberately bounded conclusion:

| Claim | Status | Strongest evidence |
| --- | --- | --- |
| Real-candidate selector calibration improves development regret | supported | `+0.1943 [0.1005,0.3007]` reduction on the widened LCO pool |
| Selector gain transfers to ten locked MVTec categories | supported | macro Dice `+0.0952 [0.0450,0.1440]` |
| Candidate generation contains better masks than selection publishes | supported diagnostic | locked oracle-selected gap `0.2455` Dice |
| Generic auto-mask system is release-ready | rejected | locked macro Dice `0.3171`; search recall `0.7162` |
| Critic arbitration improves perceptual generation | provisional | automated visibility `+0.0289 [0.0110,0.0487]`; human review pending |
| Synthetic augmentation improves independent segmentation | rejected | deterministic pixel AP `-0.0015 [-0.0414,+0.0403]` |
| External-dataset generalization / state of the art | not tested | VisA runtime ready; full external inference/evaluation pending |

R6 also makes the failure structure explicit:

- locked category-macro precision is only `0.2889` against recall `0.7357`, so
  over-segmentation remains the dominant published-mask error;
- oracle Dice is `0.5615` against selected Dice `0.3161`, so candidate ranking
  leaves `0.2455` Dice unavailable at runtime;
- `74/839` locked images have zero Dice;
- cable is localization-limited (`0.4443` search recall);
- leather, screw, and toothbrush are dominated by excess predicted area;
- screw, toothbrush, and hazelnut retain large selector regret.

The risk-coverage comparison adds a separate calibration warning. At 25%
coverage, development mean regret is `0.0720` on the non-widened pool and
`0.0999` on the widened pool, but locked mean regret is `0.2363`. Confidence
therefore does not transfer well enough for abstention to turn the current
system into a reliable high-confidence product mode.

External reproduction is now partially unblocked:

- official SubspaceAD code and protocol were identified, but the local
  environment lacks `anomalib` and the required
  `facebook/dinov2-with-registers-giant` cache;
- the official VisA archive and one-class split are now content-hash verified;
  all `10,821` runtime images are isolated from `1,200` evaluation-only masks;
- the exact frozen `v3-generic-evidence-rc1` commit completed a 12-category,
  one-anomaly-per-category smoke run without opening those masks;
- the smoke run produced all 12 metadata rows with no execution errors, but
  `9/12` were `soft_mask_only` and `3/12` were `needs_review`; this proves
  compatibility, not external mask quality;
- projected full VisA inference is roughly 11-12 hours at the observed smoke
  rate, followed by the one-shot locked evaluation;
- an immediate warm-cache rerun reproduced all selected modes and every
  published mask variant exactly; elapsed time fell from `413.181 s` to
  `168.198 s`, but raw metadata exposed a cache-provenance schema mismatch;
- the current implementation now versions evidence caches and persists complete
  producer metadata for functional texture, DINO memory, and registered
  residual providers. Cold/warm regression tests verify identical evidence
  values, calibration, reliability, and metadata keys; only `cache_hit`
  changes. The frozen VisA commit remains untouched because this is a
  non-scoring provenance correction;
- a current-code integration rerun confirms the fix on VisA candle: cold and
  warm metadata are identical after excluding only run identity, timing, and
  cache state; all mask pixels match, and the current output also matches the
  frozen candle output exactly. Warm evidence reuse reduces elapsed time
  `19.351 s -> 17.152 s` without changing the decision;
- a bounded locked-evaluation retention profile now removes only disposable
  visual diagnostics. On the 12-category VisA smoke cohort, selected modes,
  selection decisions, fused score maps, and all 10 mask roles remain
  byte-identical to the previous full-retention run. Per-run artifacts fall
  `86.59 MiB -> 34.57 MiB` (`-60.1%`), with candidate/provider PNG paths
  explicitly empty rather than dangling. Contact sheets are now category-
  balanced and capped at 64 rows by default, or can be disabled entirely;
- this operational patch changes the package content hash even though it does
  not change inference. It therefore cannot silently replace the exact frozen
  RC. Before a locked run uses it, the behavior-equivalent implementation must
  be re-sealed as a new RC, or the original frozen worktree must be run with
  additional storage;
- a one-shot cache policy now resolves that remaining storage bottleneck.
  Target DINO/texture evidence is memory-only while the expensive shared DINO
  normal tokens and normal-only calibration remain persistent. The repeated
  12-category run writes `0` target `.npz` files, retains all `72` normal-token
  files, and again has zero differences in selected modes, decisions, fused
  maps, or the 120 mask-role files. The full projection falls from roughly
  `12 GiB` to `3.6 GiB`;
- the read-only NumPy warning exposed by the real run is fixed at both DINO and
  SAM image boundaries by materializing writable RGB arrays. A subsequent
  real candle integration run completed warning-free. A full 12-category
  warning-free rerun then reproduced all regions, candidate scores,
  measurements, calibrated predictions, fused maps, and 120 mask roles exactly;
  elapsed time improved `164.761 s -> 156.508 s`;
- the committed RC2 implementation was rerun once more on the identical cohort.
  It again produced zero differences in regions, candidate data, selection,
  fused maps, and all 120 role masks. Elapsed time was `173.081 s`, which is
  `+10.6%` slower than the immediately previous run. This is one timing pair,
  not evidence of a stable throughput regression, but it must not be reported
  as a speedup;
- MVTec AD 2 is not present locally and requires its official access path.

The frozen-RC, storage, and interruption gates are now resolved:

- `auto-masks` can resume a failed, interrupted, orphaned, or publish-failed
  run from its per-sample JSONL checkpoint;
- resume requires exact auto-mask and architecture fingerprints, including the
  code, selector, checkpoints, and model identities;
- recovered mask artifacts must exist inside the original run directory;
- malformed trailing checkpoint JSON is discarded before continuation;
- a live process using the same output directory blocks a second writer;
- changed configurations start a fresh run instead of mixing cohorts.

The current implementation was resealed as
`v3-generic-evidence-rc2-operational-20260726` only after the governed
operational-equivalence command compared it with the original frozen VisA
smoke output. All 12 samples matched with zero differences in Qwen regions,
candidate scores, candidate measurements, calibrated predictions, selection
decisions, fused evidence maps, and all 120 mask-role files. Official masks
remained unopened.

The full VisA preflight now passes:

```text
runtime anomaly images: 1,200
normal-reference images: 8,659
free storage: 16.21 GiB
projected retained footprint: about 3.6 GiB
Qwen/model/selector/freeze validation: passed
resume mode: enabled
```

The full frozen run has now begun. Nine bounded executions have completed
`355/1,200` runtime images under the same run ID. The eight resume operations
reused `55`, `60`, `115`, `159`, `201`, `231`, `261`, and then `291`
checkpoint rows without duplication. Exact cumulative throughput is
`22.282 s/image`, projecting about `5.23` additional hours and a `5.50 GiB`
final run footprint. The checkpoint contains complete candle, capsule, and
cashew categories plus 55 chewing-gum rows, with 288 `needs_review` and 67
`soft_mask_only` decisions. Chewing gum is the strongest external acceptance
slice so far: 29 of 55 rows are `soft_mask_only`, Qwen is valid on 53, and SAM2
is selected on 29. Mean source disagreement is nevertheless `0.8225`, so this
remains an unscored confidence-transfer warning, not a quality result, and no
official masks have been opened.

The fifth interruption also exposed a bounded production weakness: PNG role
artifacts are written directly rather than atomically. One corrupt file was
left for the next uncheckpointed cashew sample. Because the JSONL row had not
been committed, all six partial artifacts for that sample were safely removed
and no checkpoint row was affected. Do not change this during the frozen run;
add atomic temporary-file publication in the next operational revision.

This does not reopen mask or generation tuning. The next R6 work is to resume
and complete the approximately 5.23-hour remaining VisA inference, run the
one-shot locked evaluation, then acquire dependencies for the remaining
external reproductions.

Artifacts:

```text
reports/r6_evidence_package/preregistration.md
reports/r6_evidence_package/r6_evidence_package.json
reports/r6_evidence_package/evidence_index.json
reports/r6_evidence_package/claim_ledger.md
reports/r6_evidence_package/headline_results.md
reports/r6_evidence_package/locked_failure_sheet.md
reports/r6_evidence_package/risk_coverage.md
reports/r6_evidence_package/external_baseline_status.md
reports/r6_evidence_package/paper_package.md
reports/r6_evidence_package/artifact_hashes.sha256
reports/v3_generic_evidence_visa_smoke/visa_smoke_review.md
reports/v3_generic_evidence_visa_smoke/rerun_comparison_20260726/visa_current_vs_previous.md
reports/v3_generic_evidence_visa_cache_provenance_smoke/rerun_comparison_20260726/current_vs_previous.md
reports/v3_generic_evidence_visa_retention_smoke/retention_review.md
```

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

## 1c. Strictly-additive PCA probe (2026-07-24)

All four preconditions from §1b are now met (basis/calibration cached,
`augmentation_consistency=None`, strict-superset candidate family via
`pca_subspace_additive`, LOCO refit). The PCA residual is kept **out of fusion**
and only *appends* candidates, so baseline candidates are byte-preserved — the
oracle can only rise. Pool: 144 dev images, 7243 candidates (2028 PCA-appended),
gate `always`. Empirical additivity check: **oracle never regresses** (0/144
violations; PCA raised the oracle on 68/144). Selector refit leave-category-out
on baseline-only vs full pool:

| | Baseline-only | Additive full |
| --- | ---: | ---: |
| Category-macro selected Dice | `0.4892` | `0.5303` |
| Macro Dice regret | `0.1747` | `0.1650` |

Per-category selected-Dice delta (full − baseline): metal_nut **`+0.174`**,
zipper `+0.047`, tile `+0.011`, bottle `−0.006`, wood `−0.019`. Paired
hierarchical bootstrap of the macro delta: **`+0.0411`, 95% CI
`[−0.0110, +0.1161]`**, P(≤0) ≈ `0.096`, wins/ties/losses `50/57/37`.

**Read:** strict additivity removes the fused-PCA oracle-regression confound and
lifts the point estimate (macro `+0.041`, P(≤0)≈10%), but the **CI still crosses
zero** — PCA stays **default-off**. The sharper finding: because the oracle is
monotone here, the residual bottle/wood regressions are pure **selection errors**
(the selector sometimes picks a worse PCA candidate), not evidence problems. That
localizes the next lever to selection quality on the union pool, not the PCA
provider — and confirms the macro result is **power-limited on 5 categories**, so
R2/R3 (locked categories) remains the decisive unlock. Reproduce:
`scripts/evaluate_additive_pca_loco.py` on `configs/as1b_calib_widen_pca_additive.yaml`.

## 1d. Locked-category confirmation (2026-07-24)

The network/data blocker was removed and the preregistered ten-category MVTec
locked run completed:

```text
839 anomaly images
29,584 sealed candidates
specialists disabled
PCA disabled
official masks opened only after runtime outputs were finalized
```

The frozen runtime result is below the architecture release targets:

| Metric | Locked result |
| --- | ---: |
| Category-macro Dice | `0.3171` |
| Worst category Dice (`screw`) | `0.1170` |
| Category-macro search-region recall | `0.7162` |
| Accepted-mask coverage | `0.7306` |
| Category-macro pixel AUROC | `0.9481` |
| Category-macro AUPRO | `0.8548` |
| Category-macro pixel AP | `0.4032` |

The primary selector claim passes on the identical locked candidate pool:

| Fixed selector | Macro Dice | Oracle Dice | Regret | Candidate MAE | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic-corruption baseline | `0.2208` | `0.5615` | `0.3407` | `0.1267` | `0.2472` |
| Real-candidate nonwidened | **`0.3292`** | `0.5615` | **`0.2323`** | **`0.0964`** | **`0.5426`** |
| Real-candidate widened/deployed | `0.3161` | `0.5615` | `0.2455` | `0.0968` | `0.5224` |

Preregistered widened-real minus synthetic selector macro-Dice delta:
**`+0.0952`, 95% CI `[+0.0450, +0.1440]`**, P(≤0) `0.0001`,
wins/ties/losses `479/176/184`.

**Interpretation:** real-candidate calibration generalizes and is now the
supported primary contribution. The full architecture does not generalize at
release quality. The `0.5615` candidate oracle versus `0.3161` selected Dice
leaves `0.2455` regret, so selection remains the largest recoverable gap.
Search-region recall `0.7162` is a second independent bottleneck. High recall
and low precision, especially screw (`0.8027` recall, `0.0929` precision), show
that over-segmentation is the dominant binary-mask error.

The nonwidened selector outperforming the widened selector on the same widened
pool means the widening-specific development win did not transfer cleanly.
Do not tune that policy on the exposed locked categories.

Primary artifacts:

```text
locked_generalization_preregistration.yaml
reports/v3_generic_evidence_locked/locked_evaluation/v3-generic-evidence-frozen-20260723/
reports/r3_locked_confirmation/locked_candidate_pool_analysis.md
reports/r3_locked_confirmation/locked_result_review.md
```

## 1e. Current frozen development rerun (2026-07-24)

The current widened selector was rerun on the identical 144-image,
five-category development candidate pool. Candidate generation was exactly
reproducible:

```text
5,215 / 5,215 candidate masks byte-identical
all candidate measurements exact
all Qwen regions exact
candidate oracle Dice unchanged
```

Only selection changed. The previous metadata exactly replays the retained
nonwidened selector, while the current metadata exactly replays the deployed
widened selector:

| Metric | Previous nonwidened | Current widened | Delta |
| --- | ---: | ---: | ---: |
| Category-macro Dice | `0.5558` | `0.6078` | `+0.0520` |
| Category-macro regret | `0.1081` | `0.0561` | `-0.0520` |
| Category-macro oracle Dice | `0.6639` | `0.6639` | `0.0000` |

The hierarchical paired 95% interval for macro Dice is
`[+0.0258, +0.0844]`. All five development category means improve, led by
wood (`+0.1071`).

This does not overturn the locked result. On the ten exposed locked categories,
the nonwidened selector remains slightly better than widened (`0.3292` versus
`0.3161`). The combined evidence indicates development-specific benefit and
imperfect transfer, not a general release win.

The rerun also exposed a provenance defect: prior manifests hashed the config
but did not record the mutable selector file behind `selector_model_path`.
Future manifests now record the architecture-core fingerprint and explicit
SHA-256 records for configured selector/SAM artifacts.

Full review:

```text
reports/current_pipeline_checkpoint/frozen_current_rerun/current_vs_previous_review.md
reports/current_pipeline_checkpoint/frozen_current_rerun/selected_mask_comparison.png
```

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
| **R0** | Validated checkpoint + PCA closure + correctness | ✅ Done (`e34a433`) | `99b7000` sealed checkpoint; `a7171af` default-off PCA. `e34a433` makes visibility contrast change-based, sets PCA `augmentation_consistency=None`, adds acceptance tests. PCA stays off by default. | Closed: PCA result, runtime limitation and change-based visibility contract are durable. |
| **R1** | Selector contribution evidence pack | ✅ Done (`755e14a`, `5b1850c`) | `scripts/build_selector_evidence_pack.py` (reporting only; deployed selector untouched). Hash-verified identical pools; synthetic vs real-LCO on non-widened (delta +0.0911 [0.0277,0.1757]) and widened (+0.1943 [0.1005,0.3007], all 5 categories exclude zero); selected Dice, per-category/per-morphology hierarchical intervals, calibration-size + reliability + risk-coverage curves. | Primary selector claim supported: hierarchical paired intervals show lower regret and higher selected Dice on both pools. |
| **R2** | Locked data + runtime environment unlock | 🟡 MVTec + SD1.5 unlocked | Ten locked MVTec categories acquired and hash-inventoried; runtime/reference isolation verified. Pinned SD1.5 fp16 cache is now local and recorded in Phase 4 manifests. External dataset remains open. | Locked MVTec and generation-runtime slices complete without post-access mask behavior changes. |
| **R3** | Locked-category mask confirmation | ✅ Primary confirmation complete | 839 images, 29,584 candidates. Locked macro Dice `0.3171`; oracle `0.5615`. Real-widened selector beats synthetic selector `+0.0952`, CI `[+0.0450,+0.1440]`; full release targets fail. | Selector contribution confirmed; architecture release rejected. No tuning on exposed categories. |
| **R4** | Visibility critic validity + generation re-audit | 🟡 Automated gate passed; independent gate pending | Global retry and mask-local crop rejected. Fixed adapter negligible. Text/clone critic arbitration: visibility `+0.0289 [0.0110,0.0487]`, coverage `+0.1300`, leakage `-0.0022`, acceptance `10/18→14/18`. | Keep rejected mechanisms default-off. Do not promote arbitration until two blind reviewers and independent downstream evaluation agree. |
| **R5** | Independent synthetic-utility ablation | ✅ Complete; promotion gate failed | Fusion-free ResNet18 U-Net, paired seeds, 120 matched steps, five seeds, fixed ratio, and hierarchical bootstrap. Historical pixel AP was `-0.0200` then `-0.0066`; strict deterministic replay reproduces `15/15` artifacts, and the full deterministic estimate is `-0.0015 [-0.0414,+0.0403]`. | Null confirmed under deterministic execution. Do not promote the synthetic corpus or run locked R5. |
| **R6** | External baselines + paper package | 🟡 VisA runtime ready; quantitative external run pending | Governed hash-pinned package includes the R1 calibration curves, development/locked risk-coverage, R3 failure sheet, R4 provisional result, R5 deterministic null, and claim ledger. Official VisA data is hash-pinned and mask-isolated; the exact frozen RC completed a 12-category smoke run. SubspaceAD and MVTec AD 2 remain blocked. | Run full frozen VisA inference and one-shot locked evaluation without reopening tuning. |

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

The metric-contract work is complete:

- visibility is defined from **generated-minus-background change** inside the
  mask, added/removed gradient energy, and change relative to a ring;
- structured unchanged and ring-only identity tests score near zero;
- morphology-specific lower, target, and upper visibility bands are supported;
- over-visible and fidelity-regressing attempts reduce strength rather than
  escalating it;
- the pinned fp16/safetensors SD snapshot is load-tested offline and recorded in
  finalized experiment manifests.

The first SD1.5 pilot is also complete and fails the joint promotion gate. The
controller's visibility delta is positive but too small, and the leakage-score
regression is too large. The blinded two-reviewer sheet has been generated but
has not been scored by two independent reviewers.

The mask-local crop sub-sprint was run and stopped after its first matched
cohort because visibility and coverage regressed significantly. The existing
conditioning-family comparison is also complete: fixed adapter is negligible,
while clone harmonization is useful only on the cases where the critic finds
support.

The next R4 sub-sprint is validation, not another generator:

- collect two independent blind judgments on the arbitrated A/B sheet;
- report inter-reviewer agreement and critic/reviewer disagreements;
- require both reviewers to judge at least 80% of accepted samples defect-like;
- only then integrate arbitration into Phase 4/5 and run independent utility.

The keep gate is joint: visibility improves with a stratified 95% interval
excluding zero, at least 80% of accepted samples are judged visibly
defect-like by both reviewers, and leakage/texture preservation regress by no
more than `0.01`. If critic and human judgments disagree materially, stop and
repair the metric rather than tune the generator against it.

### 5.8 Sprint R5 — make or break the synthesis claim

The clean ablation is complete:

- identical student initialization family and matched optimization steps;
- normal-only versus normal+synthetic, with PatchCore teacher fusion disabled;
- five seeds, fixed synthetic ratios selected only on development data;
- hierarchical bootstrap over categories/images and seed-wise paired deltas;
- report pixel AP/AUPRO, image AUROC, Dice at a development-fixed threshold,
  predicted-positive rate and per-category regressions;
- evaluate perceptual quality separately from downstream utility.

The preregistered regime did not beat normal-only. Pixel AP moved by
`-0.0200 [-0.0591,+0.0171]`; AUPRO and pixel AUROC were also negative, while
the small Dice increase was uncertain. The stop rule is therefore active:
synthesis is an optional/niche augmentation path, while calibrated pseudo-mask
selection remains the primary contribution. The next work is R6 evidence
packaging and external baselines, not post-hoc R5 tuning.

The identical-input rerun also failed (`pixel AP -0.0066
[-0.0573,+0.0380]`) but exposed non-byte-reproducible CUDA training
trajectories. Any future Track-B neural experiment must first pass a
deterministic replay test; this correctness work does not reopen R5 tuning.

That replay gate now passes. The new deterministic Phase 5 contract reproduces
all `15` bounded quality artifacts byte-for-byte across two command executions.
Use `configs/r5_independent_synthetic_utility_deterministic.yaml` for any future
full-scale re-estimation, with a new frozen preregistration and architecture
tag. The original R5 null and stop decision remain unchanged.

The full-scale deterministic re-estimation is now complete. Its primary pixel
AP delta is `-0.0015 [-0.0414,+0.0403]`; AUPRO and pixel AUROC remain negative,
and every interval includes zero. This replaces the drifting historical runs as
the preferred exact estimate without changing the decision. R5 is closed.

### 5.9 Sprint R6 — package the evidence before external expansion

The internal evidence package is complete. It verifies the exact R1-R5 source
hashes and emits deterministic machine-readable and paper-facing outputs:

- an evidence index and output hash manifest;
- a supported/provisional/rejected/not-tested claim ledger;
- one headline table spanning selector calibration, locked quality, critic
  validity, and downstream utility;
- a ten-category locked failure sheet;
- development and post-lock risk-coverage diagnostics;
- an external-baseline readiness matrix.

The package confirms the selector-calibration contribution and rejects both
release-readiness and downstream synthetic-utility claims. It also shows that
abstention confidence shifts badly: selecting the top 25% lowers development
regret below `0.10`, but locked regret remains `0.2363`.

External execution is the remaining R6 slice:

1. run the full frozen VisA inference and one-shot locked evaluation using the
   completed content-hashed runtime/reference split;
2. obtain MVTec AD 2 through its official access process;
3. install the official SubspaceAD environment and cache its specified
   DINOv2-with-registers giant backbone;
4. reproduce the official baseline without changing the frozen IADGen method;
5. add only locally measured external rows to the claim ledger.

Do not substitute the existing DINOv2-small/PCA-inspired provider for official
SubspaceAD, and do not quote paper numbers as project measurements.

### 5.10 Execution order and stop conditions

1. Finish R0 documentation; do not turn it into another tuning loop.
2. Build R1 now—it uses assets already present and packages the strongest
   contribution.
3. In parallel operationally, acquire/hash the locked datasets and pinned model
   caches for R2.
4. Run R3 before enabling any new default evidence or specialist.
5. Keep global retry, mask-local crop, and fixed-adapter promotion disabled;
   complete blind review of text/clone arbitration.
6. Preserve R5 as the preregistered development null; do not run its locked
   extension or tune against the completed cohort.
7. Complete the independent R4 blind review only as critic-validity evidence.
8. Keep the completed internal R6 package sealed and include the R5 null.
9. Acquire external prerequisites, then run official baselines and datasets
   against the frozen method without further development-category tuning.

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
Critic + controller mechanism implemented and tested. The first pinned-SD
re-audit is complete; the global controller failed its joint keep gate.

**Contract**
- [x] Add a visibility term to `iadgen_v2/generation_critic.py` (in-mask deviation magnitude, added edge/gradient energy, local contrast vs a surrounding normal ring)
- [x] Controller drives visibility (rank tiebreak + visibility-triggered retry); coverage demoted to a guard (weight 0.22→0.12 + opt-in reject gate)
- [x] Freeze/hash the exact Phase 2 parent metadata consumed by both arms

**Tasks**
- [x] Implement the visibility/fidelity metric + unit tests against known visible/invisible edits
- [x] Validate on real corpus: genuine defect 0.6–0.79 vs current corpus median 0.25 (150 samples); gate@0.35 flags 97%
- [x] Retarget the Phase-4 controller objective to visibility (rank + retry); per-attempt reason logging
- [x] Implement morphology-specific lower/target/upper bands
- [x] Run matched SD re-audit with `target_defect_visibility` enabled
- [x] Generate randomized blind-review sheet and separate answer key
- [x] Run and reject the strict mask-local crop ablation (`visibility -0.0510`,
  `coverage -0.2065`)
- [x] Run existing conditioning-family comparison: fixed adapter negligible;
  clone harmonization improves wood but regresses other categories
- [x] Build category-agnostic text/clone arbitration; automated R4 gate passes
  (`visibility +0.0289`, `leakage -0.0022`, `10/18→14/18` accepted)
- [x] Reproduce the current arbitration run: all generated images, critic
  metrics, acceptance outcomes, and `12/6` routing decisions are identical
- [ ] Obtain independent judgments from two reviewers
- [ ] Integrate arbitration into Phase 4/5 only after blind-review agreement

**Status: automated gate passed; independent gate pending.** The fp16 SD1.5 snapshot is now
available offline and hash-inventoried in both finalized manifests. The
controller increased visibility by `+0.0117 [0.0018,0.0220]` and coverage by
`+0.0665 [0.0281,0.1142]`, but leakage score regressed by
`-0.0204 [-0.0262,-0.0149]`; acceptance improved only `10/18→11/18` at 2.5×
attempt count. Keep that controller off. Text/clone arbitration subsequently
passed all automated gates, but the critic is not an independent judge and the
blind human gate remains open.

**Acceptance / exit gate**
- [x] Mean visibility improves by `≥ 0.020` over the matched one-attempt
  baseline, with a morphology-stratified 95% bootstrap CI excluding zero
  *(automated arbitration result; not independent validation)*
- [ ] `≥ 80%` of the stratified accepted sample is judged visibly defect-like by both reviewers
- [x] Leakage and texture-preservation means each regress by no more than `0.01`

#### ✅ B-S2 · Preregistered downstream ablation  *(fixes B1, B3)*
**Contract**
- [x] Same student, ±synthetic, **PatchCore fusion removed**, matched training compute, 5 seeds, hierarchical bootstrap CIs; regimes + primary metric preregistered

**Tasks**
- [x] Build the ablation harness (fusion-off student, matched steps)
- [x] Preregister the development regime, primary metric, interval, and gate
- [x] Run the development experiment once
- [x] Stop before locked evaluation because the development promotion gate failed

**Acceptance / exit gate**
- [x] Null reported and synthesis repositioned as optional augmentation:
  pixel AP `-0.0200 [-0.0591,+0.0171]`

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
