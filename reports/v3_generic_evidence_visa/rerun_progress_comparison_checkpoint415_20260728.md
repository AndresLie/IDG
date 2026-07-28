# VisA Checkpoint 415 Versus Checkpoint 355

## Decision

Continue the frozen run. The completed chewing-gum category is heterogeneous,
while the first fryum slice has the strongest acceptance, calibration, and
structure-profile diagnostics observed so far.

## Execution Comparison

| Measure | Checkpoint 355 segment | Checkpoint 415 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `64` | `60` | mixed category |
| Segment runtime | `985.655 s` | `1,019.593 s` | `+33.938 s` |
| Segment rate | `15.401 s/image` | `16.993 s/image` | `+10.34%` |
| Cumulative progress | `29.58%` | `34.58%` | `+5.00 pp` |
| Run artifacts | `1,665.1 MiB` | `1,837.5 MiB` | `+172.4 MiB` |
| Resume count | `8` | `9` | expected |
| Duplicate identities | `0` | `0` | no regression |

Runtime remains much faster than cashew. The ten-percent segment difference is
small relative to the category and initialization mixture.

## Completed Chewing-Gum Category

| Diagnostic | First 55 | Final 45 |
| --- | ---: | ---: |
| Mean expected IoU | `0.2415` | `0.1990` |
| Median expected IoU | `0.2558` | `0.1877` |
| Mean conformal IoU lower bound | `0.1117` | `0.0697` |
| Mean source disagreement | `0.8225` | `0.8815` |
| Mean selected-mask area | `0.0174` | `0.0082` |
| Qwen valid localization | `53/55` | `43/45` |
| `soft_mask_only` | `29/55` | `13/45` |
| SAM2 selected | `29/55` | `19/45` |

The first half overstates whole-category acceptance. The completed category has:

```text
soft_mask_only: 42/100
needs_review: 58/100
SAM2 selected: 48/100
Qwen valid localization: 96/100
```

The lower acceptance in the final slice coincides with smaller masks, lower
predicted IoU, and higher disagreement. SAM2 is frequent but does not guarantee
acceptance.

## Fryum Transition

| Diagnostic | First 15 fryum rows |
| --- | ---: |
| Mean expected IoU | `0.3307` |
| Median expected IoU | `0.2921` |
| Mean conformal IoU lower bound | `0.1953` |
| Mean source disagreement | `0.5184` |
| Mean selected-mask area | `0.0818` |
| Qwen valid localization | `14/15` |
| `soft_mask_only` | `15/15` |
| `repeated_chain` profile | `15/15` |

Fryum is the first external category slice with complete soft acceptance and a
consistent non-ring structural profile. The result is promising because
confidence, lower bound, Qwen validity, and source agreement move together.

It is still not quality evidence. Locked labels must verify that the larger
selected masks and repeated-chain interpretation correspond to defect pixels.

## Integrity

```text
same run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 355
records completed: 415
new records: 60
duplicate sample identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 5,810
corrupt retained PNGs: 0
official masks opened: no
```

## Professional Interpretation

The frozen architecture now shows three distinct external regimes:

1. Candle/capsule/cashew mostly abstain.
2. Chewing gum is mixed and strongly SAM2-dependent.
3. Early fryum is accepted consistently with lower disagreement and a
   repeated-chain profile.

This heterogeneity is precisely why one macro external evaluation is necessary.
Do not promote fryum behavior or tune weaker categories before the locked run
is complete.
