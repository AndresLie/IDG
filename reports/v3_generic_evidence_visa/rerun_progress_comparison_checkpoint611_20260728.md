# VisA Checkpoint 611 Versus Checkpoint 575

## Decision

Continue the frozen run. Macaroni1 is complete and confirms a strong
abstention signal. Its final decline is not explained by Qwen fallback or
source disagreement alone. Macaroni2 has only 11 rows. Official masks remain
sealed.

## Execution Comparison

| Measure | Checkpoint 575 segment | Checkpoint 611 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `35` | `36` | macaroni1/macaroni2 transition |
| Segment runtime | `918.397 s` | `927.848 s` | `+9.451 s` |
| Segment rate | `26.240 s/image` | `25.774 s/image` | `-1.78%` |
| Cumulative progress | `47.92%` | `50.92%` | `+3.00 pp` |
| Run artifacts | `2,583.2 MiB` | `2,743.0 MiB` | `+159.8 MiB` |
| Resume count | `13` | `14` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative throughput is `22.271 s/image`, leaving approximately `3.64` hours.

## Completed Macaroni1 Category

| Diagnostic | Previous 35 | Final 25 | Full 100 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2186` | `0.1954` | `0.2145` |
| Mean conformal IoU lower bound | `0.0831` | `0.0599` | `0.0790` |
| Mean source disagreement | `0.4585` | `0.4148` | `0.4492` |
| Mean selected-mask area | `0.0401` | `0.0375` | `0.0411` |
| Qwen full-image fallback | `15/35` | `4/25` | `36/100` |
| `soft_mask_only` | `9/35` | `1/25` | `17/100` |
| `needs_review` | `26/35` | `24/25` | `83/100` |
| `ring_sector` profile | `35/35` | `25/25` | `100/100` |

The final slice is less confident even though Qwen localization and source
agreement improve. This rules out a simple explanation that localization
fallback is the only macaroni1 weakness.

## Selection Distribution

```text
final 25:
fused_q950: 13
fused_q975: 11
fused_q900: 1

full 100:
fused_q950: 70
fused_q975: 26
fused_q900: 3
edge_fused_q900_component_1_1: 1
```

The final slice is less concentrated than the preceding rows, yet acceptance
falls. Candidate measurements or calibration therefore contribute materially
to disposition. Locked oracle-candidate and selector-regret analysis is needed.

## Macaroni2 Transition

```text
rows: 11
soft_mask_only: 3
needs_review: 8
mean expected IoU: 0.2198
mean lower bound: 0.0843
mean source disagreement: 0.4969
Qwen fallback: 4/11
repeated_chain: 6/11
ring_sector: 5/11
```

Macaroni2 starts with confidence similar to the macaroni1 aggregate but a mixed
structural profile and broader proposal selection. The sample is too small for
a category claim.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 575
records completed: 611
new records: 36
unique sample identities: 611
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 8,554
expected retained PNGs: 8,554
corrupt retained PNGs: 0
official masks opened: no
```

One fused-evidence image for uncheckpointed macaroni2 `011` was removed. No
durable row was changed.

## Professional Interpretation

Macaroni1 is the clearest external abstention warning so far. That is not
automatically a failure: if the corresponding masks are genuinely poor,
abstention is correct. If high-IoU candidates exist, the weakness is selector
calibration or ranking. The sealed evaluation must report both selected quality
and candidate oracle gap to resolve this distinction.
