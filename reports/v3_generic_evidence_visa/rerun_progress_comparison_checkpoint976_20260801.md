# VisA Checkpoint 976 Versus Checkpoint 925

## Decision

Continue the unchanged frozen run. The next 51 pcb3 rows reproduce `100%`
soft-mask acceptance, even though Qwen fallback increases substantially. This
supports the generic soft-prior path, but the absence of any pcb3 abstention is
also a risk-coverage and calibration warning. Official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 925 segment | Checkpoint 976 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `47` | `51` | all new rows are pcb3 |
| Productive segment runtime | `898.044 s` | `898.265 s` | `+0.221 s` |
| Productive segment rate | `19.107 s/image` | `17.613 s/image` | `-7.82%` |
| Cumulative progress | `77.08%` | `81.33%` | `+4.25 pp` |
| Run artifacts | `4,193.5 MiB` | `4,384.9 MiB` | `+191.4 MiB` |
| Resume count | `22` | `23` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput improves to `21.322 s/image`, leaving
approximately `1.33` hours. The segment is faster than the preceding mixed
pcb2/pcb3 segment, but timing remains an operational measurement rather than an
architecture-quality result.

## Matched Pcb3 Comparison

| Diagnostic | First 25 pcb3 | Next 51 pcb3 | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.3735` | `0.3518` | `-0.0217` |
| Median expected IoU | `0.3863` | `0.3412` | `-0.0451` |
| Mean conformal IoU lower bound | `0.2380` | `0.2163` | `-0.0217` |
| Mean source disagreement | `0.4061` | `0.3803` | `-0.0257` |
| Mean selected-mask area | `0.1200` | `0.1122` | `-0.0078` |
| Qwen valid localization | `23/25` | `37/51` | `92.0%` to `72.5%` |
| Qwen full-image fallback | `2/25` | `14/51` | `8.0%` to `27.5%` |
| `soft_mask_only` | `25/25` | `51/51` | unchanged `100%` |
| `needs_review` | `0/25` | `0/51` | unchanged |
| `ring_sector` | `25/25` | `51/51` | unchanged collapse |

The new slice is slightly less confident, but evidence providers agree more
closely and masks are modestly smaller. Complete acceptance survives a
3.4-times higher Qwen fallback rate. That is consistent with Qwen acting as a
soft prior rather than a hard fence.

However, disposition does not separate risk anywhere in pcb3. The selector
accepts all 76 rows despite meaningful variation in localization, expected IoU,
mask area, and proposal family. If locked pcb3 quality is heterogeneous, the
current confidence threshold will be unable to abstain selectively.

## Pcb3 Aggregate

```text
rows: 76
soft_mask_only: 76
needs_review: 0
mean expected IoU: 0.3589
mean conformal IoU lower bound: 0.2235
mean source disagreement: 0.3888
mean selected-mask area: 0.1148
Qwen valid localization: 60/76
Qwen full-image fallback: 16/76
structure profile: ring_sector 76/76
```

Selected modes remain concentrated but not singular:

```text
fused_q850_component_1: 26
edge_fused_q850_2: 23
fused_q900_component_1: 10
edge_fused_q850_component_1_2: 8
edge_fused_q900_component_1_2: 5
fused_q900: 1
fused_q950_component_1: 1
fused_q900_component_3: 1
edge_fused_q900_component_2_2: 1
```

The locked comparison must report precision, recall, positive rate, selected
Dice, oracle Dice, calibration error, and risk coverage together. Broad masks
could reflect useful PCB coverage or systematic over-segmentation.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 925
records completed: 976
new durable records: 51
unique sample identities: 976
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 13,664
expected retained PNGs: 13,664
corrupt retained PNGs: 0
uncheckpointed pcb3 row-076 residue: 0
official masks opened: no
```

Partial metadata SHA-256:

```text
1d2719f36b4b9ff152f24d06fc153d7a6ba4bca67aee22842bed0a872c7c4a3b
```

## Professional Interpretation

The current rerun is operationally faster and behaviorally consistent. It
replicates pcb3 acceptance under weaker localization, which is encouraging for
zero-shot robustness. It simultaneously shows that current confidence does not
rank risk within pcb3. Neither interpretation should trigger runtime tuning;
finish the frozen cohort and resolve the ambiguity with the one-shot locked
evaluation.
