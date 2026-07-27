# VisA Checkpoint 261 Versus Checkpoint 231

## Decision

Continue the frozen run. This exact 30-versus-30 cashew comparison shows stable
runtime and stronger selector predictions, but source disagreement worsens and
official quality remains unmeasured.

## Execution Comparison

| Measure | Checkpoint 231 segment | Checkpoint 261 segment | Change |
| --- | ---: | ---: | ---: |
| New cashew rows | `30` | `30` | matched |
| Segment runtime | `981.600 s` | `974.748 s` | `-6.852 s` |
| Segment rate | `32.720 s/image` | `32.492 s/image` | `-0.70%` |
| Cumulative progress | `19.25%` | `21.75%` | `+2.50 pp` |
| Run artifacts | `1,098.7 MiB` | `1,292.2 MiB` | `+193.5 MiB` |
| Resume count | `5` | `6` | expected |
| Duplicate identities | `0` | `0` | no regression |

The cashew steady-state runtime is reproducible across the two segments. It is
slower than candle and capsule, but it is not drifting further.

## Matched Cashew Comparison

| Diagnostic | Previous 30 | New 30 | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.1862` | `0.2112` | `+0.0250` |
| Median expected IoU | `0.1677` | `0.1677` | unchanged |
| Mean conformal IoU lower bound | `0.0507` | `0.0758` | `+0.0251` |
| Mean source disagreement | `0.6626` | `0.7170` | `+0.0544` |
| Median source disagreement | `0.6038` | `0.6172` | `+0.0134` |
| Mean selected-mask area | `0.0330` | `0.0360` | `+0.0030` |
| Qwen full-image fallback | `2/30` | `1/30` | improved |
| `soft_mask_only` | `5/30` | `9/30` | `+4` |
| `needs_review` | `25/30` | `21/30` | `-4` |

The later slice is more often accepted as a soft pseudo-label and has higher
predicted IoU. The simultaneous disagreement increase means the prediction
gain should not be interpreted as ground-truth improvement. The selector may
be responding to stronger anomaly contrast while evidence providers disagree
about boundaries.

## Structure Diagnostics

All 60 matched cashew rows are labeled `ring_sector`. Structural specialists
remain disabled, so this has no effect on frozen selection. It remains a
post-evaluation diagnostic weakness because the profile lacks diversity on
cashew.

## Integrity

```text
same run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 231
records completed: 261
new records: 30
duplicate sample identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 3,655
corrupt retained PNGs: 0
official masks opened: no
```

## Professional Interpretation

The production path is stable and the confidence system distinguishes easier
and harder cashew slices. It still does not establish calibration transfer:
`21/30` new rows abstain, source disagreement rises, and no official labels have
been consulted.

Complete the unchanged runtime cohort. Locked evaluation must determine whether
the higher expected-IoU/lower-bound predictions correspond to real Dice and
whether disagreement is a useful risk signal on VisA.
