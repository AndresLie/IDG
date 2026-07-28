# VisA Checkpoint 575 Versus Checkpoint 540

## Decision

Continue the frozen run. A second substantial macaroni1 slice reproduces the
low-confidence regime without further degradation. Acceptance improves
modestly, but candidate selection remains concentrated. Official masks remain
sealed.

## Execution Comparison

| Measure | Checkpoint 540 segment | Checkpoint 575 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `34` | `35` | macaroni1-only |
| Segment runtime | `926.776 s` | `918.397 s` | `-8.379 s` |
| Segment rate | `27.258 s/image` | `26.240 s/image` | `-3.74%` |
| Cumulative progress | `45.00%` | `47.92%` | `+2.92 pp` |
| Run artifacts | `2,418.2 MiB` | `2,583.2 MiB` | `+165.0 MiB` |
| Resume count | `12` | `13` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative throughput is `22.052 s/image`, leaving approximately `3.83` hours.

## Macaroni1 Replication

| Diagnostic | Previous 34 | New 35 | First 75 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2206` | `0.2186` | `0.2208` |
| Mean conformal IoU lower bound | `0.0851` | `0.0831` | `0.0854` |
| Mean source disagreement | `0.4743` | `0.4585` | `0.4606` |
| Mean selected-mask area | `0.0460` | `0.0401` | `0.0424` |
| Qwen full-image fallback | `17/34` | `15/35` | `32/75` |
| `soft_mask_only` | `5/34` | `9/35` | `16/75` |
| `needs_review` | `29/34` | `26/35` | `59/75` |
| `ring_sector` profile | `34/34` | `35/35` | `75/75` |

The new slice is not materially worse. Expected quality and lower bound are
flat, while disagreement, localization fallback, and review rate improve
slightly. Macaroni1 nevertheless remains a low-coverage category.

## Selection Concentration

```text
new 35 selected modes:
fused_q950: 26
fused_q975: 8
edge_fused_q900_component_1_1: 1

first 75 selected modes:
fused_q950: 57
fused_q975: 15
fused_q900: 2
edge_fused_q900_component_1_1: 1
```

The candidate distribution broadens slightly, but q950 remains dominant.
Without official candidate IoU, this cannot be classified as either correct
consistency or selector regret.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 540
records completed: 575
new records: 35
unique sample identities: 575
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 8,050
expected retained PNGs: 8,050
corrupt retained PNGs: 0
official masks opened: no
```

The interruption occurred inside Qwen inference and left no incomplete image
artifact.

## Professional Interpretation

Macaroni1 now has enough unscored rows to establish a stable external
confidence warning. The architecture abstains on most samples rather than
forcing hard masks, which is operationally appropriate. The decisive locked
analysis remains candidate oracle gap, selector regret, and Qwen search-region
recall.
