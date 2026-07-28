# VisA Checkpoint 463 Versus Checkpoint 415

## Decision

Continue the frozen run. The next 48 fryum rows reproduce the first 15-row
confidence pattern under a materially higher Qwen fallback rate. Official VisA
masks remain sealed, so this is runtime and confidence-transfer evidence only.

## Execution Comparison

| Measure | Checkpoint 415 segment | Checkpoint 463 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `60` | `48` | fryum-only |
| Segment runtime | `1,019.593 s` | `982.629 s` | `-36.963 s` |
| Segment rate | `16.993 s/image` | `20.471 s/image` | `+20.47%` |
| Cumulative progress | `34.58%` | `38.58%` | `+4.00 pp` |
| Run artifacts | `1,837.5 MiB` | `2,058.1 MiB` | `+220.7 MiB` |
| Resume count | `9` | `10` | expected |
| Duplicate identities | `0` | `0` | no regression |

The segment is slower than the previous mixed chewing-gum/fryum segment, but
cumulative throughput remains stable at `21.409 s/image`.

## Fryum Replication

| Diagnostic | First 15 | Next 48 | Direction |
| --- | ---: | ---: | --- |
| Mean expected IoU | `0.3307` | `0.3339` | stable |
| Median expected IoU | `0.2921` | `0.2805` | stable |
| Mean conformal IoU lower bound | `0.1953` | `0.1984` | stable |
| Mean source disagreement | `0.5184` | `0.5004` | slightly better |
| Mean selected-mask area | `0.0818` | `0.0767` | similar |
| Qwen full-image fallback | `1/15` | `13/48` | substantially higher |
| `soft_mask_only` | `15/15` | `46/48` | persistent |
| `repeated_chain` profile | `15/15` | `48/48` | persistent |

The architecture does not collapse when Qwen localization becomes less
reliable. This is consistent with the intended soft-prior design: fused generic
evidence retains influence outside the localized region.

Two rows fall to `needs_review`, which is preferable to forcing hard masks.
The selected proposal distribution also broadens across fused quantiles,
components, and edge-aware candidates instead of collapsing to one mode.

## Cumulative State

```text
completed: 463 / 1,200
categories: candle 100, capsules 100, cashew 100, chewinggum 100, fryum 63
needs_review: 322
soft_mask_only: 141
hard_mask_ok: 0
Qwen valid localization: 350
Qwen full-image fallback: 113
unique sample identities: 463
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 6,482
corrupt retained PNGs: 0
official masks opened: no
```

## Professional Interpretation

Fryum is now a replicated external confidence regime rather than a promising
15-image transition artifact. Its predicted quality, lower bound, acceptance,
and repeated-chain classification remain stable across 63 images.

The result specifically supports graceful degradation of localization. It does
not establish segmentation accuracy, calibration against real IoU, or category
generalization. Those claims remain reserved for the one-shot locked evaluation
after all 1,200 runtime rows are sealed.
