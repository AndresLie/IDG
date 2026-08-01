# VisA Checkpoint 925 Versus Checkpoint 878

## Decision

Continue the unchanged frozen run. Pcb2 is complete, and its final slice
rebounds from the preceding confidence decline. The first 25 pcb3 rows are all
accepted as soft masks, but their substantially larger area makes precision
and positive-rate the decisive locked checks. Official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 878 segment | Checkpoint 925 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `45` | `47` | 22 pcb2 plus 25 pcb3 |
| Productive segment runtime | `898.134 s` | `898.044 s` | `-0.090 s` |
| Productive segment rate | `19.959 s/image` | `19.107 s/image` | `-4.27%` |
| Cumulative progress | `73.17%` | `77.08%` | `+3.91 pp` |
| Run artifacts | `3,989.1 MiB` | `4,193.5 MiB` | `+204.4 MiB` |
| Resume count | `21` | `22` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput is `21.527 s/image`, leaving approximately
`1.64` hours. Free storage is `8.85 GiB`, still above the `5.31 GiB` projected
final run footprint.

## Completed Pcb2

| Diagnostic | First 78 pcb2 | Final 22 pcb2 | Completed 100 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2106` | `0.2327` | `0.2154` |
| Mean conformal IoU lower bound | `0.0751` | `0.0972` | `0.0799` |
| Mean source disagreement | `0.4274` | `0.3949` | `0.4203` |
| Mean selected-mask area | `0.0437` | `0.0406` | `0.0430` |
| Qwen valid localization | `73/78` | `19/22` | `92/100` |
| `soft_mask_only` | `30/78` | `12/22` | `42/100` |
| `needs_review` | `48/78` | `10/22` | `58/100` |
| `ring_sector` | `78/78` | `22/22` | `100/100` |

The final slice improves expected IoU and lower-bound confidence while source
disagreement falls. This reverses the previous slice's decline and argues
against progressive selector collapse. It does not establish actual mask
quality; the entire pcb2 category remains unscored.

Pcb2 proposal selection is diverse: no mode exceeds 16/100. The largest are
`fused_q950_component_1` (`16`), `fused_q975` (`16`), `fused_q900` (`10`),
`edge_fused_q900_1` (`10`), and `fused_q850_component_1` (`9`).

## Pcb3 Opening

```text
rows: 25
soft_mask_only: 25
needs_review: 0
mean expected IoU: 0.3735
mean conformal IoU lower bound: 0.2380
mean source disagreement: 0.4061
mean selected-mask area: 0.1200
Qwen valid localization: 23/25
structure profile: ring_sector 25/25
```

Pcb3 is more confident than completed pcb1 and pcb2, including under two Qwen
fallbacks. Its masks are also much larger: mean area `0.1200` versus `0.0950`
for pcb1 and `0.0430` for pcb2. The same pattern can represent better coverage
or over-segmentation, so the locked review must report precision, recall,
positive rate, selected Dice, oracle Dice, and selector regret together.

Selected modes in the first 25 pcb3 rows are concentrated in two families:

```text
edge_fused_q850_2: 9
fused_q850_component_1: 9
fused_q900_component_1: 3
edge_fused_q900_component_1_2: 1
edge_fused_q900_component_2_2: 1
edge_fused_q850_component_1_2: 1
fused_q900: 1
```

This mode concentration is another post-evaluation diagnostic, not a basis for
changing the frozen selector.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 878
records completed: 925
new durable records: 47
unique sample identities: 925
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 12,950
expected retained PNGs: 12,950
corrupt retained PNGs: 0
uncheckpointed pcb3 row-025 residue: 0
official masks opened: no
```

Partial metadata SHA-256:

```text
34eec435132476e9d8efcc2ddf3dd4f4f79b65a793a75d68baff9ac6ec696b88
```

## Professional Interpretation

Checkpoint 925 is operationally valid and exposes two useful external
generalization behaviors: confidence varies substantially inside pcb2, and
pcb3 receives uniformly high confidence with broad masks. Neither observation
should be optimized from runtime proxies. Complete the frozen cohort, then use
the one-shot locked evaluation to distinguish calibration, selection, and
proposal-ceiling failures.
