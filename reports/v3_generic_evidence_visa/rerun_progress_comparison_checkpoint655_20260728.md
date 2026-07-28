# VisA Checkpoint 655 Versus Checkpoint 611

## Decision

Continue the frozen run. The expanded macaroni2 slice has healthier confidence
diagnostics than macaroni1: structure routing and candidate selection remain
diverse, source disagreement improves, and soft-mask coverage rises. This is
not yet evidence of better mask overlap because official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 611 segment | Checkpoint 655 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `36` | `44` | macaroni2-only |
| Segment runtime | `927.848 s` | `918.526 s` | `-9.321 s` |
| Segment rate | `25.774 s/image` | `20.876 s/image` | `-19.00%` |
| Cumulative progress | `50.92%` | `54.58%` | `+3.67 pp` |
| Run artifacts | `2,743.0 MiB` | `2,903.8 MiB` | `+160.8 MiB` |
| Resume count | `14` | `15` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative throughput improves to `22.178 s/image`, leaving approximately
`3.36` hours.

## Macaroni2 Comparison

| Diagnostic | First 11 | Next 44 | First 55 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2198` | `0.2186` | `0.2188` |
| Mean conformal IoU lower bound | `0.0843` | `0.0831` | `0.0833` |
| Mean source disagreement | `0.4969` | `0.3732` | `0.3979` |
| Mean selected-mask area | `0.0429` | `0.0331` | `0.0351` |
| Qwen full-image fallback | `4/11` | `17/44` | `21/55` |
| `soft_mask_only` | `3/11` | `16/44` | `19/55` |
| `needs_review` | `8/11` | `28/44` | `36/55` |
| `repeated_chain` | `6/11` | `21/44` | `27/55` |
| `ring_sector` | `5/11` | `23/44` | `28/55` |

Expected quality and localization fallback remain stable, while disagreement
falls substantially and soft coverage rises from 27.3% to 36.4%.

## Proposal Diversity

```text
first 55 selected modes:
fused_q975: 17
fused_q950: 14
fused_q900_component_1: 6
edge_fused_q850_component_1_1: 5
fused_q900: 4
fused_q950_component_1: 3
edge_fused_q900_component_1_2: 2
fused_q900_component_2: 2
edge_fused_q900_component_2_2: 1
fused_q900_component_3: 1
```

No selected mode dominates the category. This contrasts with macaroni1, where
q950 accounted for 70/100 rows. The generic proposal path is therefore capable
of expressing heterogeneous external evidence when the selector supports it.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 611
records completed: 655
new records: 44
unique sample identities: 655
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 9,170
expected retained PNGs: 9,170
corrupt retained PNGs: 0
official masks opened: no
```

The interruption occurred inside Qwen inference and left no incomplete image
artifact.

## Professional Interpretation

Macaroni2 provides a useful counterexample to the macaroni1 warning. Similar
predicted quality does not force the same structure or proposal mode. The
generic architecture remains expressive; the locked evaluation must test
whether that diversity corresponds to lower selector regret and better masks.
