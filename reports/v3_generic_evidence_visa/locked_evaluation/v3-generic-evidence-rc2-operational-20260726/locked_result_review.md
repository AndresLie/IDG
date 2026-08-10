# VisA Locked Result Review

## Executive Decision

The frozen `v3-generic-evidence-rc2-operational-20260726` architecture completed
all 1,200 VisA runtime images and the one permitted locked evaluation. The
execution and governance architecture worked; the mask-quality generalization
claim did not.

```text
runtime completion: 1,200 / 1,200
category-macro Dice: 0.1593
95% category-bootstrap interval: [0.0941, 0.2503]
accepted-mask coverage: 0.5267
categories at or above 0.30 Dice: 1 / 12
release decision: reject
generation Sprint 6: blocked
```

This is a decisive negative result, not an ambiguous near miss. Every
preregistered locked acceptance target failed.

## Integrity

The result is methodologically valid:

- runtime inference completed before official masks were opened;
- the stable and run-local metadata files are byte-identical;
- all 1,200 sample identities are unique;
- all 16,800 retained PNG artifacts are referenced and readable;
- architecture and auto-mask fingerprints remained unchanged;
- `locked-evaluate` wrote a seal before resolving official masks;
- rerunning the locked evaluation is forbidden by the sealed configuration.

Key hashes:

```text
runtime metadata:
7a086751bbc4609fdfc9a01096d2b6258c173809738d30f889f2a436d99489e3

locked metrics:
d18499a70a5864e58cfd269b1a09a4170e969fe39ba9e05f91190d7b6f868d83

reference manifest:
cb19d9e379b1bcfc4546e5103a2452fa85d83e9f758f501e9c431a2ff43ad693
```

## Locked Metrics

| Category | Dice | Precision | Recall | Search Recall | Pixel AP | AUPRO | Accepted Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candle | `0.0903` | `0.0523` | `0.7525` | `0.7723` | `0.0721` | `0.8707` | `0.00` |
| capsules | `0.2472` | `0.2192` | `0.6393` | `0.3926` | `0.3247` | `0.9095` | `0.15` |
| cashew | `0.1451` | `0.1303` | `0.8316` | `0.9396` | `0.2513` | `0.9043` | `0.23` |
| chewinggum | `0.5897` | `0.6674` | `0.6944` | `0.8774` | `0.5725` | `0.9063` | `0.42` |
| fryum | `0.1786` | `0.1825` | `0.8240` | `0.5920` | `0.2480` | `0.8930` | `0.95` |
| macaroni1 | `0.0271` | `0.0140` | `0.7811` | `0.5531` | `0.1933` | `0.8742` | `0.17` |
| macaroni2 | `0.0266` | `0.0139` | `0.5954` | `0.4372` | `0.0810` | `0.9088` | `0.31` |
| pcb1 | `0.1052` | `0.0748` | `0.8571` | `0.3411` | `0.0809` | `0.8406` | `0.73` |
| pcb2 | `0.1342` | `0.0862` | `0.6240` | `0.3985` | `0.1309` | `0.8289` | `0.42` |
| pcb3 | `0.0720` | `0.0414` | `0.8434` | `0.3625` | `0.1493` | `0.8384` | `1.00` |
| pcb4 | `0.1562` | `0.1039` | `0.5702` | `0.5239` | `0.1164` | `0.7169` | `0.94` |
| pipe_fryum | `0.1389` | `0.1172` | `0.9724` | `0.8873` | `0.2816` | `0.9687` | `1.00` |
| **Macro** | **`0.1593`** | **`0.1419`** | **`0.7488`** | **`0.5898`** | **`0.2085`** | **`0.8717`** | **`0.5267`** |

## Gate Assessment

| Acceptance criterion | Target | Result | Decision |
| --- | ---: | ---: | --- |
| Locked macro Dice | `>= 0.55` | `0.1593` | fail |
| Every category Dice | `>= 0.30` | minimum `0.0266` | fail |
| Search-region recall | `>= 0.90` macro | `0.5898` | fail |
| Development-to-locked gap | `<= 0.10` | `0.3495` | fail |
| Accepted-mask coverage | `>= 0.75` | `0.5267` | fail |

The development LCO estimate was `0.5088`. Its `0.3495` drop to VisA is too
large to attribute to ordinary sampling noise.

## What Worked

1. **Governed execution.** Resume, fingerprint, artifact-retention, and locked
   evaluation contracts survived a long 1,200-image run without contaminating
   inference with official labels.
2. **Anomaly ranking.** Macro AUPRO is `0.8717`, and chewing-gum reaches
   `0.5897` Dice. The evidence field can localize useful anomaly signal when
   object texture and defect contrast align with its assumptions.
3. **High recall.** Macro recall is `0.7488`. The system usually includes the
   defect somewhere in its selected support.
4. **Soft localization fallback.** Qwen failure does not erase anomaly evidence.
   Full-image fallback preserves recall, although precision becomes poor.

## What Failed

### 1. Binary Masks Are Too Large

The mean predicted positive rate is `0.0524`; the estimated true positive rate
is only `0.0096`. Published masks therefore cover roughly 5.5 times the true
defect area. Macro precision collapses to `0.1419` while recall remains high.

The visual sheet shows repeated normal-boundary selection: candle rims, cashew
edges, macaroni contours, PCB boards/components, and pipe-fryum object bodies.
This is the primary Dice failure.

### 2. Selector Calibration Does Not Transfer

```text
expected-IoU MAE: 0.2183
expected-IoU bias: +0.1468
expected-IoU Spearman: 0.0851
confidence-vs-Dice Spearman: 0.0840
conformal lower-bound coverage: 0.3458
```

The selector is systematically optimistic and almost unable to rank real VisA
mask quality. Its conformal lower bound does not behave like a 90% lower bound
under dataset shift.

Accepted masks average `0.1830` Dice versus `0.1329` for `needs_review`. That
direction is correct, but a `0.0501` separation is far too weak for reliable
abstention. Pipe-fryum is the clearest failure: 100% coverage and very high
predicted quality produce only `0.1389` Dice.

### 3. Localization Is Not General Enough

Macro search-region recall is `0.5898`. Qwen-valid rows average only `0.4904`
search recall. Full-image fallbacks report `1.0` search recall by construction,
but their Dice falls to `0.0665`. The current soft-prior design prevents total
recall failure, yet it does not recover spatial precision.

### 4. Structure Recognition Is Collapsed

`1,018/1,200` samples are classified as `ring_sector`, including every
pipe-fryum row and almost every PCB row. Specialists were disabled, so this did
not route inference, but the topology signal is not trustworthy enough for a
future structural policy.

### 5. Ranking Metrics Do Not Rescue Publication

High AUPRO should not be read as release readiness. Pixel AP is only `0.2085`,
and selected binary masks remain poor. The evidence field contains useful local
ordering, but its absolute calibration and component extraction are wrong.

## Visual Evidence

The category-balanced sheet intentionally shows the lowest- and highest-Dice
selected mask for every category:

```text
locked_best_worst_contact_sheet.png
```

Columns are input, official mask, selected mask, and fused evidence. Cyan areas
outside the red official defect expose the dominant boundary and object-body
false positives.

## Next Architecture: V4 Calibration-First

VisA is now exposed and may only be used as development data for a future
version. It cannot be reused as locked confirmation for V4.

### Sprint V4.1: Score-To-Mask Calibration

- learn a category-agnostic pixel posterior from fused score, source agreement,
  local gradient, foreground boundary distance, and normal-reference rarity;
- train with leave-dataset-out folds across MVTec development and VisA;
- calibrate thresholds by expected precision/recall rather than fixed fused
  quantiles;
- reject masks whose area is inconsistent with posterior mass;
- target predicted/true area ratio below `2.0` and macro precision above `0.30`.

### Sprint V4.2: Real-Candidate Selector Refit

- train candidate IoU, precision, and recall heads on real development
  candidates from multiple datasets;
- retain category-name blindness and use only geometric/evidence features;
- use grouped leave-category-and-dataset-out calibration;
- require expected-IoU Spearman `>= 0.40`, MAE `<= 0.12`, and conformal coverage
  between `0.85` and `0.95` before deployment.

### Sprint V4.3: Boundary Suppression And Compact Components

- model normal object boundaries from the normal-reference memory;
- penalize evidence that follows stable contours without interior anomaly
  contrast;
- select sparse connected components with posterior support instead of broad
  object envelopes;
- preserve thin-defect recall through uncertainty-aware soft labels.

### Sprint V4.4: Multi-Proposal Localization

- combine Qwen boxes, evidence maxima, and foreground geometry as independent
  proposals;
- score region reliability rather than treating one Qwen box as the primary
  search region;
- target macro search recall `>= 0.85` without full-image fallback inflation.

### Sprint V4.5: New Locked Benchmark

- freeze V4 only after leave-dataset-out macro Dice improves materially over
  `0.1593` and no development dataset collapses;
- acquire a genuinely untouched external benchmark;
- run one locked evaluation with the same firewall and sealing protocol;
- keep generation frozen until the new locked mask gate passes.

## Final Judgment

The architecture is research-useful as an evidence-ranking and governance
prototype, but it is not a general zero-shot pseudo-mask generator. The most
valuable result is now clear: adding more evidence providers is not the next
move. V4 must convert existing evidence into compact, calibrated masks and must
validate that calibration across datasets before synthetic generation resumes.
