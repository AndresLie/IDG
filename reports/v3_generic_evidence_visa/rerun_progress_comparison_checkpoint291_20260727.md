# VisA Checkpoint 291 Versus Checkpoint 261

## Decision

Continue the frozen run. The third matched cashew slice regresses selector
confidence primarily alongside a sharp rise in Qwen full-image fallback.
Evidence disagreement improves, so the result points to localization validity
rather than general evidence instability.

## Execution Comparison

| Measure | Checkpoint 261 segment | Checkpoint 291 segment | Change |
| --- | ---: | ---: | ---: |
| New cashew rows | `30` | `30` | matched |
| Segment runtime | `974.748 s` | `998.798 s` | `+24.050 s` |
| Segment rate | `32.492 s/image` | `33.293 s/image` | `+2.47%` |
| Cumulative progress | `21.75%` | `24.25%` | `+2.50 pp` |
| Run artifacts | `1,292.2 MiB` | `1,484.0 MiB` | `+191.8 MiB` |
| Resume count | `6` | `7` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cashew runtime remains broadly stable. The small segment-rate increase is not
large enough to identify a throughput regression.

## Matched Cashew Comparison

| Diagnostic | Previous 30 | New 30 | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2112` | `0.1723` | `-0.0389` |
| Median expected IoU | `0.1677` | `0.1677` | unchanged |
| Mean conformal IoU lower bound | `0.0758` | `0.0370` | `-0.0388` |
| Mean source disagreement | `0.7170` | `0.6445` | `-0.0725` |
| Median source disagreement | `0.6172` | `0.6051` | `-0.0122` |
| Mean selected-mask area | `0.0360` | `0.0400` | `+0.0040` |
| Qwen full-image fallback | `1/30` | `8/30` | `+7` |
| `soft_mask_only` | `9/30` | `3/30` | `-6` |
| `needs_review` | `21/30` | `27/30` | `+6` |

The selector becomes less confident even though source disagreement improves.
The strongest accompanying change is localization fallback, which rises from
`3.3%` to `26.7%`. This supports the architecture's use of Qwen as a soft prior,
but also shows that localization confidence remains a major external transfer
variable.

## Interruption Integrity

The interrupt occurred while writing the overlay for the next uncheckpointed
sample, `cashew_bad_091`. All 14 files created for that incomplete sample were
removed so the next resume recomputes it cleanly.

```text
checkpoint rows affected: 0
duplicate sample identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
official masks opened: no
```

This is the second observed direct-PNG interruption residue. Atomic artifact
publication remains a real post-cohort operational fix, but package code must
not change while the frozen checkpoint is active.

## Professional Interpretation

The result is not an architecture regression: code and fingerprints are
unchanged, runtime is stable, and source agreement improves. It is a
within-category sample/localization shift.

The result strengthens two concerns:

1. Qwen localization reliability strongly influences confidence transfer.
2. Direct image writes can leave uncheckpointed residue after interruption.

Neither concern should be tuned against the sealed VisA cohort. Finish runtime
inference, then use locked labels to quantify whether fallback predicts lower
Dice and whether source disagreement is calibrated.
