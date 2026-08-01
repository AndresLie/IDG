# VisA Checkpoint 833 Versus Checkpoint 791

## Decision

Continue the unchanged frozen run. Pcb1 is now complete, and the first 33 pcb2
rows show a more conservative confidence regime with smaller masks and lower
source disagreement. This is a category-transfer difference, not yet a quality
ranking. Official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 791 segment | Checkpoint 833 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `46` | `42` | nine pcb1 plus 33 pcb2 |
| Productive segment runtime | `894.673 s` | `898.380 s` | `+3.708 s` |
| Productive segment rate | `19.449 s/image` | `21.390 s/image` | `+9.98%` |
| Cumulative progress | `65.92%` | `69.42%` | `+3.50 pp` |
| Run artifacts | `3,528.0 MiB` | `3,749.2 MiB` | `+221.3 MiB` |
| Resume count | `19` | `20` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput is `21.748 s/image`, leaving approximately
`2.22` hours. Category-transition loading and differing image content make the
9.98% segment slowdown operational rather than an architecture regression.

## Confidence Comparison

| Diagnostic | Previous pcb1 46 | New mixed 42 | Pcb2 first 33 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2400` | `0.2306` | `0.2215` |
| Mean conformal IoU lower bound | `0.1045` | `0.0951` | `0.0860` |
| Mean source disagreement | `0.5788` | `0.4550` | `0.4277` |
| Mean selected-mask area | `0.0945` | `0.0591` | `0.0417` |
| Qwen valid localization | `38/46` | `40/42` | `31/33` |
| Qwen full-image fallback | `8/46` | `2/42` | `2/33` |
| `soft_mask_only` | `32/46` | `25/42` | `16/33` |
| `needs_review` | `14/46` | `17/42` | `17/33` |
| `ring_sector` | `46/46` | `42/42` | `33/33` |

Pcb2 is less frequently accepted than pcb1 despite better localization and
lower evidence disagreement. The selector also chooses masks less than half
the pcb1 area. This could be appropriate precision for smaller PCB defects or
systematic under-segmentation; confidence diagnostics alone cannot decide.

## Category Completion

Completed pcb1:

```text
soft_mask_only: 73/100
needs_review: 27/100
mean expected IoU: 0.2436
mean conformal IoU lower bound: 0.1081
mean source disagreement: 0.5800
mean selected-mask area: 0.0950
Qwen valid localization: 84/100
structure profile: ring_sector 100/100
```

Pcb2 proposal selection remains diverse:

```text
fused_q975: 6
fused_q950: 4
fused_q950_component_1: 4
edge_fused_q900_1: 4
fused_q850_component_1: 3
fused_q900: 3
fused_q900_component_2: 3
fused_q900_component_1: 2
edge_fused_q850_component_1_1: 1
edge_fused_q850_component_1_2: 1
fused_q900_component_3: 1
fused_q975_component_1: 1
```

The structural profile remains uniform even though selected proposal families
are varied. Specialists are disabled, so this does not alter frozen inference,
but it remains a correctness risk for future structure-aware routing.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 791
records completed: 833
new durable records: 42
unique sample identities: 833
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 11,662
expected retained PNGs: 11,662
corrupt retained PNGs: 0
uncheckpointed pcb2 row-033 residue: 0
official masks opened: no
```

## Professional Interpretation

Checkpoint 833 closes pcb1 with reproducibly high frozen confidence and opens
pcb2 with more conservative masks. The contrast is useful external evidence of
category-sensitive behavior in a category-agnostic architecture, but neither
regime should be promoted or repaired from confidence proxies. The locked
evaluation must compare precision, recall, positive rate, selector regret, and
risk coverage for both PCB categories before any post-freeze design decision.
