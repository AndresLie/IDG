# VisA Checkpoint 231 Versus Checkpoint 201

## Decision

Continue the frozen run. The first substantial cashew slice has stronger
predicted quality and localization validity than the preceding capsule slice,
but high source disagreement still drives conservative abstention.

## Execution Comparison

| Measure | Checkpoint 201 | Checkpoint 231 | Segment change |
| --- | ---: | ---: | ---: |
| Completed rows | `201` | `231` | `+30` |
| Cohort progress | `16.75%` | `19.25%` | `+2.50 pp` |
| Segment runtime | `966.605 s` | `981.600 s` | `+14.995 s` |
| Segment rate | `23.014 s/image` | `32.720 s/image` | `+42.17%` slower |
| Run artifacts | `905.2 MiB` | `1,098.7 MiB` | `+193.5 MiB` |
| Resume count | `4` | `5` | expected |
| Duplicate identities | `0` | `0` | no regression |

The cashew segment is materially slower. Its first window also included a
longer model initialization, so `32.720 s/image` is an upper-bound operational
estimate rather than a pure steady-state category rate.

## Category Transition

The previous segment contained 41 capsule rows and one cashew row. The new
segment contains 30 cashew rows.

| Diagnostic | Previous 41 capsules | New 30 cashews |
| --- | ---: | ---: |
| Mean expected IoU | `0.1432` | `0.1862` |
| Median expected IoU | `0.1215` | `0.1677` |
| Mean conformal IoU lower bound | `0.0315` | `0.0507` |
| Mean source disagreement | `0.6183` | `0.6626` |
| Mean selected-mask area | `0.0231` | `0.0330` |
| Qwen full-image fallback | `7/41` | `2/30` |
| `soft_mask_only` | `5/41` | `5/30` |
| `needs_review` | `36/41` | `25/30` |

The cashew selector predictions are more optimistic and Qwen localization is
more often valid. Source disagreement is worse, however, so five-sixths of the
rows still abstain. Because this is a category transition rather than a matched
cohort, the differences must not be described as an architecture gain.

## Structure Diagnostics

All 30 new cashew rows are labeled `ring_sector`, compared with a mix of
`ring_sector` and `repeated_chain` in capsules. Structural specialists are
disabled and therefore do not alter the frozen selected masks.

## Integrity

```text
same run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 201
records completed: 231
new records: 30
duplicate sample identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 3,234
corrupt retained PNGs: 0
official masks opened: no
```

## Professional Interpretation

Operational resumability remains effective across a second category boundary.
The runtime and storage projections remain feasible, although disk usage has
reached 97% and must be monitored on every checkpoint.

Confidence transfer remains the scientific weakness. The current selector
assigns better expected IoU to cashew than capsule, but disagreement prevents
hard acceptance. Locked evaluation is required to determine whether this is
appropriate caution or severe miscalibration.

Do not tune on this unscored external cohort. Complete the frozen runtime,
publish its stable manifest, and open official references only through the
one-shot locked evaluator.
