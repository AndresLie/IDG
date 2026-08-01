# VisA Checkpoint 791 Versus Checkpoint 745

## Decision

Continue the unchanged frozen run. The second pcb1 slice closely reproduces
the first slice's predicted quality, localization mix, and soft-mask coverage.
This supports deterministic confidence transfer within pcb1, but not mask
quality: all 91 pcb1 rows still route as `ring_sector`, selected masks remain
large, and official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 745 segment | Checkpoint 791 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `49` | `46` | pcb1-only continuation |
| Productive segment runtime | `898.364 s` | `894.673 s` | `-3.691 s` |
| Productive segment rate | `18.334 s/image` | `19.449 s/image` | `+6.08%` |
| Cumulative progress | `62.08%` | `65.92%` | `+3.84 pp` |
| Run artifacts | `3,296.1 MiB` | `3,528.0 MiB` | `+231.9 MiB` |
| Resume count | `18` | `19` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput improves to `21.767 s/image`, leaving
approximately `2.47` hours. The matched pcb1 segment is 6.08% slower per row
than the preceding transition segment, but remains within normal
category-level runtime variation.

## Matched Pcb1 Comparison

| Diagnostic | First 45 | Next 46 | First 91 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2432` | `0.2400` | `0.2416` |
| Mean conformal IoU lower bound | `0.1077` | `0.1045` | `0.1061` |
| Mean source disagreement | `0.5863` | `0.5788` | `0.5825` |
| Mean selected-mask area | `0.0897` | `0.0945` | `0.0922` |
| Qwen valid localization | `37/45` | `38/46` | `75/91` |
| Qwen full-image fallback | `8/45` | `8/46` | `16/91` |
| `soft_mask_only` | `32/45` | `32/46` | `64/91` |
| `needs_review` | `13/45` | `14/46` | `27/91` |
| `ring_sector` | `45/45` | `46/46` | `91/91` |

The two slices differ by only `0.0032` in expected IoU and conformal lower
bound. Their localization fallback counts and acceptance counts are nearly
identical. This is the strongest external evidence so far that the frozen
selector's pcb1 confidence regime is stable rather than a transient batch
effect.

## Proposal Diversity

The next 46 pcb1 rows select:

```text
edge_fused_q900_2: 13
fused_q850_component_1: 10
fused_q850_component_2: 4
edge_fused_q850_component_1_2: 4
edge_fused_q850_2: 3
edge_fused_q850_1: 2
edge_fused_q900_1: 2
fused_q975: 2
fused_q850: 1
fused_q900_component_1: 1
fused_q900_component_2: 1
fused_q950: 1
edge_fused_q900_component_1_2: 1
edge_fused_q900_component_2_2: 1
```

Proposal selection remains diverse and shifts toward edge-aware modes. The
uniform structure label is therefore not caused by a single selected proposal
family. Structural specialists are disabled, so the label does not alter this
frozen inference run.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 745
records completed: 791
new durable records: 46
unique sample identities: 791
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 11,074
expected retained PNGs: 11,074
corrupt retained PNGs: 0
uncheckpointed row-091 residue: 0
official masks opened: no
```

## Professional Interpretation

Checkpoint 791 converts the initial pcb1 observation into a reproducible
within-category confidence result. It does not establish high-quality masks.
Stable large areas and `91/91 ring_sector` routing could represent consistent
PCB defect capture or consistent over-segmentation. The one-shot locked
evaluation must resolve that ambiguity through Dice, precision, positive rate,
selector regret, and risk-coverage rather than another threshold adjustment.
