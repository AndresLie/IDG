# VisA Checkpoint 1070 Versus Checkpoint 1025

## Decision

Continue the unchanged frozen run. The next 45 pcb4 rows introduce four
`needs_review` decisions under uniformly valid Qwen localization, providing the
first within-PCB risk separation. Whether those abstentions are correct remains
unknown because official masks are sealed.

## Execution Comparison

| Measure | Checkpoint 1025 primary segment | Checkpoint 1070 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `48` before recovery | `45` | all new rows are pcb4 |
| Runtime | `893.095 s` | `897.043 s` | `+3.948 s` |
| Segment rate | `18.606 s/image` | `19.934 s/image` | `+7.14%` |
| Cumulative progress | `85.42%` | `89.17%` | `+3.75 pp` |
| Run artifacts | `4,593.9 MiB` | `4,813.3 MiB` | `+219.4 MiB` |
| Resume count | `25` | `26` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput including the previous recovery finalizer is
`21.185 s/image`, leaving approximately `0.77` hours. This pcb4-only segment is
7.14% slower than the prior mixed transition segment; one pair is insufficient
to infer a stable performance regression.

## Matched Pcb4 Comparison

| Diagnostic | First 25 pcb4 | Next 45 pcb4 | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2915` | `0.2823` | `-0.0092` |
| Median expected IoU | `0.2805` | `0.2628` | `-0.0177` |
| Mean conformal IoU lower bound | `0.1560` | `0.1468` | `-0.0092` |
| Mean source disagreement | `0.4295` | `0.4207` | `-0.0088` |
| Mean selected-mask area | `0.0712` | `0.0797` | `+0.0085` |
| Qwen valid localization | `25/25` | `45/45` | unchanged `100%` |
| `soft_mask_only` | `25/25` | `41/45` | `100%` to `91.1%` |
| `needs_review` | `0/25` | `4/45` | first selective abstentions |
| `ring_sector` | `25/25` | `44/45` | one `repeated_chain` |

The later slice is only slightly less confident, while evidence disagreement
improves and masks become larger. Since localization is valid for every image,
the four abstentions are attributable to candidate measurements and calibrated
selection rather than Qwen fallback.

The review cases are:

```text
032  fused_q900_component_1  expected=0.1677  lower=0.0322  area=0.0179
039  fused_q850_component_2  expected=0.2385  lower=0.1030  area=0.0224
041  fused_q950              expected=0.1677  lower=0.0322  area=0.0473
043  fused_q950              expected=0.1677  lower=0.0322  area=0.0462
```

The selector is therefore not simply rejecting large masks. Locked evaluation
must test whether these lower-confidence candidates actually have worse Dice,
precision, or recall.

## Pcb4 Aggregate

```text
rows: 70
soft_mask_only: 66
needs_review: 4
mean expected IoU: 0.2856
mean conformal IoU lower bound: 0.1501
mean source disagreement: 0.4238
mean selected-mask area: 0.0767
Qwen valid localization: 70/70
structure profile: ring_sector 69, repeated_chain 1
```

Selected modes remain diverse. `fused_q850_component_1` leads at `21/70`,
followed by `edge_fused_q850_component_1_2` (`10`),
`fused_q850_component_2` (`8`), and two six-row fused component families. This
is not a single-mode collapse.

## Shutdown And Integrity

This segment used a single manual `Ctrl-C` rather than timeout escalation.
`last_run_status.json` finalized correctly as `interrupted`, so the stale
`running` problem did not recur. The interrupt landed during overlay encoding
for uncheckpointed pcb4 `070`, leaving 14 row artifacts and a corrupt overlay.
All 14 were removed together before validation.

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 1,025
records completed: 1,070
unique sample identities: 1,070
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 14,980
expected retained PNGs: 14,980
corrupt retained PNGs: 0
uncheckpointed pcb4 row-070 residue: 0
official masks opened: no
```

Partial metadata SHA-256:

```text
826511948b8a389285843a68cfd4c9fe8cfead069da2c47ced20db14c9478ed0
```

## Professional Interpretation

The current result is more informative than universal acceptance: pcb4 now
contains selective abstentions under stable localization. That is directionally
consistent with useful calibration, but it is not proof until the locked masks
show lower quality on the abstained cases. Finish the frozen cohort before any
selector or threshold change.
