# VisA Checkpoint 201 Versus Checkpoint 159

## Decision

Continue the frozen run. The new capsule slice improves several unscored
confidence diagnostics, but external quality remains unmeasured and abstention
remains dominant.

## Execution Comparison

| Measure | Checkpoint 159 | Checkpoint 201 | Segment change |
| --- | ---: | ---: | ---: |
| Completed rows | `159` | `201` | `+42` |
| Cohort progress | `13.25%` | `16.75%` | `+3.50 pp` |
| Segment runtime | `959.831 s` | `966.605 s` | `+6.774 s` |
| Segment rate | `21.814 s/image` | `23.014 s/image` | `+5.50%` slower |
| Run artifacts | `682.8 MiB` | `905.2 MiB` | `+222.4 MiB` |
| Resume count | `3` | `4` | expected |
| Duplicate identities | `0` | `0` | no regression |

The fifth segment contains 41 capsule rows and the first cashew row. Cumulative
runtime is `19.748 s/image`, projecting approximately `5.48` remaining hours.

## Matched Capsule Comparison

| Diagnostic | Previous 44 capsules | New 41 capsules | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.1289` | `0.1432` | `+0.0143` |
| Median expected IoU | `0.0885` | `0.1215` | `+0.0331` |
| Mean conformal IoU lower bound | `0.0300` | `0.0315` | `+0.0015` |
| Mean source disagreement | `0.6436` | `0.6183` | `-0.0253` |
| Median source disagreement | `0.6609` | `0.6253` | `-0.0356` |
| Mean selected-mask area | `0.0317` | `0.0231` | `-0.0086` |
| `soft_mask_only` coverage | `8/44` | `5/41` | `18.2% -> 12.2%` |
| Qwen full-image fallback | `12/44` | `7/41` | `27.3% -> 17.1%` |

The later capsule slice has better expected quality, lower evidence
disagreement, and fewer localization fallbacks. However, only five rows reach
`soft_mask_only`; 36 remain `needs_review`. These diagnostics therefore suggest
heterogeneous sample difficulty, not a validated architecture improvement.

## Structure Comparison

| Profile | Previous 44 capsules | New 41 capsules |
| --- | ---: | ---: |
| `ring_sector` | `33` | `26` |
| `repeated_chain` | `11` | `15` |

The profile distribution is less collapsed in the later capsule slice.
Structural specialists remain disabled, so the profile does not affect the
frozen mask decision.

## Interruption Integrity

The interrupt occurred while writing `cashew_bad_001_training_wide.png`.
That sample had not been appended to the checkpoint. Its six partially written
artifacts were removed so the next resume recomputes it from scratch.

```text
checkpointed records affected: 0
checkpoint duplicate identities: 0
missing checkpoint eval masks: 0
missing checkpoint training masks: 0
missing checkpoint uncertainty masks: 0
official masks opened: no
```

This exposes a bounded operational weakness: per-sample image files are not
written atomically. The current JSONL checkpoint still prevents incomplete rows
from being accepted. Atomic PNG writes should be considered after the frozen
external cohort completes, because changing package code now would invalidate
the active checkpoint.

## Professional Interpretation

The current segment is directionally better than the previous capsule segment,
but it does not change the research conclusion. The external selector remains
highly conservative, and there is still no official Dice, precision, recall, or
calibration measurement.

Complete the unchanged runtime cohort before interpreting confidence transfer.
After locked evaluation, separate true mask failure from confidence
miscalibration and add atomic artifact publication to the next operational
architecture revision.
