# VisA Checkpoint 540 Versus Checkpoint 506

## Decision

Continue the frozen run. The larger macaroni1 slice confirms a low-acceptance
confidence regime and exposes both localization fallback and selection
concentration. Official VisA masks remain sealed, so this is not a Dice result.

## Execution Comparison

| Measure | Checkpoint 506 segment | Checkpoint 540 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `43` | `34` | macaroni1-only |
| Segment runtime | `922.372 s` | `926.776 s` | `+4.403 s` |
| Segment rate | `21.451 s/image` | `27.258 s/image` | `+27.07%` |
| Cumulative progress | `42.17%` | `45.00%` | `+2.83 pp` |
| Run artifacts | `2,257.4 MiB` | `2,418.2 MiB` | `+160.8 MiB` |
| Resume count | `11` | `12` | expected |
| Duplicate identities | `0` | `0` | no regression |

Macaroni1 is more expensive than the preceding fryum-heavy segment. Cumulative
throughput is now `21.781 s/image`, leaving approximately `3.99` hours.

## Macaroni1 Comparison

| Diagnostic | First 6 | Next 34 | First 40 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2351` | `0.2206` | `0.2228` |
| Mean conformal IoU lower bound | `0.0996` | `0.0851` | `0.0873` |
| Mean source disagreement | `0.3954` | `0.4743` | `0.4625` |
| Mean selected-mask area | `0.0351` | `0.0460` | `0.0443` |
| Qwen full-image fallback | `0/6` | `17/34` | `17/40` |
| `soft_mask_only` | `2/6` | `5/34` | `7/40` |
| `needs_review` | `4/6` | `29/34` | `33/40` |
| `ring_sector` profile | `6/6` | `34/34` | `40/40` |

The initial six rows overstate category acceptance. The larger slice has lower
predicted quality, higher disagreement, and a 50% localization fallback rate.
Those factors move together and plausibly explain the selector's conservative
disposition.

## Selection Concentration

```text
next 34 selected modes:
fused_q950: 28
fused_q975: 4
fused_q900: 2
```

This concentration may mean q950 is consistently appropriate, or that the
candidate pool and calibrated selector lack useful discrimination for this
external structure. Locked candidate regret and oracle measurements are needed
to distinguish those possibilities. The frozen run must not be tuned now.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 506
records completed: 540
new records: 34
unique sample identities: 540
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 7,560
expected retained PNGs: 7,560
corrupt retained PNGs: 0
official masks opened: no
```

One fused-evidence image for uncheckpointed macaroni1 `040` was removed after
the graceful interruption. No durable row was changed.

## Professional Interpretation

Macaroni1 currently looks like a difficult external category for confidence
transfer, but the architecture is behaving conservatively rather than
publishing hard masks. The important locked questions are:

1. Are the `needs_review` masks actually low quality?
2. Is `fused_q950` selection causing regret relative to existing candidates?
3. Does Qwen fallback reduce search-region recall or merely remove a weak prior?

Those questions can only be answered after completing and sealing all 1,200
runtime rows.
