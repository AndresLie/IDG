# VisA Checkpoint 745 Versus Checkpoint 696

## Decision

Continue the unchanged frozen run. Macaroni2 is now complete, and the first 45
pcb1 rows show substantially higher predicted quality and soft-mask coverage.
This apparent improvement is category-compositional and remains unscored:
pcb1 also has higher evidence disagreement, larger masks, and a fully collapsed
`ring_sector` profile. Official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 696 segment | Checkpoint 745 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `41` | `49` | four macaroni2 plus 45 pcb1 |
| Productive segment runtime | `898.173 s` | `898.364 s` | `+0.192 s` |
| Productive segment rate | `21.907 s/image` | `18.334 s/image` | `-16.31%` |
| Cumulative progress | `58.00%` | `62.08%` | `+4.08 pp` |
| Run artifacts | `3,053.1 MiB` | `3,296.1 MiB` | `+243.0 MiB` |
| Resume count | `17` | `18` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput improves to `21.910 s/image`, leaving
approximately `2.77` hours. The segment includes one-time category-transition
loading, so its faster row rate should not be interpreted as a code-level
performance improvement.

## Confidence Comparison

| Diagnostic | Previous macaroni2 41 | New mixed 49 | Pcb1 first 45 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2156` | `0.2408` | `0.2432` |
| Mean conformal IoU lower bound | `0.0801` | `0.1053` | `0.1077` |
| Mean source disagreement | `0.4261` | `0.5736` | `0.5863` |
| Mean selected-mask area | `0.0358` | `0.0846` | `0.0897` |
| Qwen valid localization | `41/41` | `41/49` | `37/45` |
| Qwen full-image fallback | `0/41` | `8/49` | `8/45` |
| `soft_mask_only` | `11/41` | `33/49` | `32/45` |
| `needs_review` | `30/41` | `16/49` | `13/45` |
| `repeated_chain` | `24/41` | `2/49` | `0/45` |
| `ring_sector` | `17/41` | `47/49` | `45/45` |

The confidence increase is real in the frozen selector outputs, but it is not
yet validated against mask overlap. Pcb1 accepts more rows despite higher
source disagreement and some Qwen fallbacks. Its masks are also approximately
2.5 times larger than the previous macaroni2 slice.

## Category Completion

Completed macaroni2:

```text
soft_mask_only: 31/100
needs_review: 69/100
mean expected IoU: 0.2173
mean conformal IoU lower bound: 0.0818
mean source disagreement: 0.4108
Qwen valid localization: 79/100
structure profile: repeated_chain 53, ring_sector 47
```

Pcb1 proposal selection remains diverse:

```text
fused_q850_component_1: 8
edge_fused_q850_component_1_2: 6
fused_q850_component_2: 5
edge_fused_q850_1: 5
edge_fused_q900_2: 5
edge_fused_q850_2: 4
fused_q900_component_1: 3
edge_fused_q900_1: 3
fused_q850: 2
fused_q900_component_2: 2
edge_fused_q900_component_1_2: 1
fused_q900: 1
```

Candidate diversity argues against a single-mode collapse. The structural
profile collapse is still a correctness warning for future profile-aware
routing, although specialists are disabled in this frozen run.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 696
records completed: 745
new durable records: 49
unique sample identities: 745
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 10,430
expected retained PNGs: 10,430
corrupt retained PNGs: 0
uncheckpointed interruption residue: 0
official masks opened: no
```

## Professional Interpretation

Checkpoint 745 is operationally stronger and diagnostically mixed. The generic
pipeline transfers to pcb1 with high predicted acceptance and varied proposal
selection, but the larger masks, higher source disagreement, and uniform
structure label create a plausible over-segmentation risk. The one-shot locked
evaluation must compare accepted-mask risk, selector regret, and positive rate
for pcb1 before this confidence shift can be called an improvement.
