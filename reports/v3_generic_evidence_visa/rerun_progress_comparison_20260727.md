# VisA Current Rerun Versus Previous Checkpoint

## Decision

The rerun is operationally successful and quality-inconclusive.

It resumed the exact frozen RC2 run, added 55 records without duplication, and
held throughput within one percent of the prior checkpoint. It did not open
official VisA masks, so this is not evidence that selected-mask Dice improved.

## Matched Execution Comparison

| Measure | Previous checkpoint | Current checkpoint | Change |
| --- | ---: | ---: | ---: |
| Completed rows | `60` | `115` | `+55` |
| Cohort progress | `5.00%` | `9.58%` | `+4.58 pp` |
| Segment processing rate | `17.825 s/image` | `17.700 s/image` | `-0.70%` |
| Run artifacts | about `235 MiB` | `459.9 MiB` | about `+225 MiB` |
| Unique sample identities | `60` | `115` | `+55` |
| Duplicate identities | `0` | `0` | no regression |
| Resume count | `1` | `2` | expected |

The run ID, auto-mask fingerprint, and architecture-core fingerprint are
unchanged. The stable output manifest remains unpublished because the cohort is
incomplete.

## Unscored Result Comparison

| Selector diagnostic | Previous 60 | New 55 | Interpretation |
| --- | ---: | ---: | --- |
| `needs_review` | `60/60` | `53/55` | confidence remains poor |
| `soft_mask_only` | `0/60` | `2/55` | first non-review dispositions |
| Mean expected IoU | `0.1630` | `0.1505` | lower in mixed-category segment |
| Mean IoU lower bound | `0.0275` | `0.0281` | effectively unchanged |
| Mean source disagreement | `0.3792` | `0.4643` | worse in mixed-category segment |
| Mean mask area | `0.0322` | `0.0329` | stable |
| Qwen full-image fallback | `66.7%` | `43.6%` | fewer fallbacks |

The new segment is not composition-matched: it contains 40 candle and 15
capsule images. The new candle slice has mean expected IoU `0.1665`, close to
the previous candle checkpoint. The first capsule slice is weaker:

```text
expected IoU: 0.1078
IoU lower bound: 0.0202
source disagreement: 0.6291
needs_review: 13/15
```

This is the most important new warning. The selector's development confidence
does not transfer cleanly to the first external capsule images.

## Integrity Checks

```text
same run ID: yes
same frozen architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 60
records added: 55
duplicate identities: 0
missing run-local eval/training/uncertainty masks: 0
auto-mask tests: 98 passed
official masks opened: no
```

The `mask_path` compatibility targets are intentionally absent while the run is
partial. They are published atomically only when all 1,200 records complete.

## Professional Review

The resumable execution architecture is effective. The current segment shows
no throughput or checkpoint-integrity regression and should be retained.

The model-quality picture is not yet positive. Almost every row abstains, the
conformal lower bounds remain near zero, and all 115 samples are classified as
`ring_sector`. Because structural specialists are disabled, the last issue does
not alter this frozen run, but it indicates that the structure profile is not a
trustworthy generic descriptor on VisA.

Do not tune on these unscored diagnostics. Resume the unchanged run, finish all
1,200 runtime records, seal the stable manifest, then invoke the one-shot
locked evaluation. Only that comparison can determine whether the current
external result is better or worse than the previous MVTec evidence.
