# VisA Checkpoint 506 Versus Checkpoint 463

## Decision

Continue the frozen run. Fryum is now complete and retains high soft-mask
coverage, but its final 37 rows are more conservative than the first 63.
Macaroni1 has only six rows and is not yet interpretable. Official VisA masks
remain sealed.

## Execution Comparison

| Measure | Checkpoint 463 segment | Checkpoint 506 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `48` | `43` | fryum/macaroni1 transition |
| Segment runtime | `982.629 s` | `922.372 s` | `-60.257 s` |
| Segment rate | `20.471 s/image` | `21.451 s/image` | `+4.78%` |
| Cumulative progress | `38.58%` | `42.17%` | `+3.59 pp` |
| Run artifacts | `2,058.1 MiB` | `2,257.4 MiB` | `+199.3 MiB` |
| Resume count | `10` | `11` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative throughput remains stable at `21.413 s/image`, leaving an estimated
`4.13` hours of inference.

## Completed Fryum Category

| Diagnostic | First 63 | Final 37 | Full 100 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.3331` | `0.2969` | `0.3197` |
| Mean conformal IoU lower bound | `0.1976` | `0.1614` | `0.1842` |
| Mean source disagreement | `0.5047` | `0.4883` | `0.4986` |
| Mean selected-mask area | `0.0779` | `0.0510` | `0.0679` |
| Qwen full-image fallback | `14/63` | `2/37` | `16/100` |
| `soft_mask_only` | `61/63` | `34/37` | `95/100` |
| `needs_review` | `2/63` | `3/37` | `5/100` |
| `repeated_chain` profile | `63/63` | `37/37` | `100/100` |

Fryum is the strongest completed external confidence category so far. The
final slice does not collapse, but its predicted IoU, lower bound, and mask area
are lower. Because localization improves and disagreement also falls, Qwen
failure does not explain the drift. A plausible unscored explanation is that
the final images contain weaker or smaller evidence.

That explanation must not be promoted to a quality claim before locked labels
are opened. The smaller masks could represent either improved precision or
under-segmentation.

## Macaroni1 Transition

```text
rows: 6
soft_mask_only: 2
needs_review: 4
mean expected IoU: 0.2351
mean lower bound: 0.0996
mean source disagreement: 0.3954
Qwen valid localization: 6/6
ring_sector profile: 6/6
```

Six rows are insufficient for a category conclusion. The lower acceptance
despite valid localization and relatively low disagreement shows that neither
signal alone determines selector disposition.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 463
records completed: 506
new records: 43
unique sample identities: 506
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 7,084
expected retained PNGs: 7,084
corrupt retained PNGs: 0
official masks opened: no
```

The interruption left one fused-evidence image for uncheckpointed macaroni1
`006`; it was removed without changing any durable row.

## Professional Interpretation

The new checkpoint strengthens two operational findings:

1. Generic evidence supports a full external category with high abstention-safe
   coverage and a stable non-ring structure profile.
2. Confidence is heterogeneous within that category, so early-slice metrics
   must not be generalized to the whole category.

The correct next step remains completion of the frozen runtime cohort, followed
by the single preregistered locked evaluation.
