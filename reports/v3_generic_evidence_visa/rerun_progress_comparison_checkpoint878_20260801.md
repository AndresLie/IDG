# VisA Checkpoint 878 Versus Checkpoint 833

## Decision

Continue the unchanged frozen run. The next 45 pcb2 rows reproduce the same
localization and evidence-agreement regime as the first 33, but selector
confidence and soft-mask acceptance decline. This is a selector-transfer
warning to test after the cohort is sealed, not a reason to tune against locked
runtime data. Official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 833 segment | Checkpoint 878 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `42` | `45` | all new rows are pcb2 |
| Productive segment runtime | `898.380 s` | `898.134 s` | `-0.246 s` |
| Productive segment rate | `21.390 s/image` | `19.959 s/image` | `-6.69%` |
| Cumulative progress | `69.42%` | `73.17%` | `+3.75 pp` |
| Run artifacts | `3,749.2 MiB` | `3,989.1 MiB` | `+239.9 MiB` |
| Resume count | `20` | `21` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput is `21.656 s/image`, leaving approximately
`1.94` hours. The segment is modestly faster than its mixed pcb1/pcb2
predecessor, but one timing pair is operational evidence only.

## Matched Pcb2 Comparison

| Diagnostic | First 33 pcb2 | Next 45 pcb2 | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2215` | `0.2026` | `-0.0189` |
| Median expected IoU | `0.2040` | `0.2040` | unchanged |
| Mean conformal IoU lower bound | `0.0860` | `0.0671` | `-0.0189` |
| Mean source disagreement | `0.4277` | `0.4272` | `-0.0005` |
| Mean selected-mask area | `0.0417` | `0.0452` | `+0.0035` |
| Qwen valid localization | `31/33` | `42/45` | `93.9%` vs `93.3%` |
| Qwen full-image fallback | `2/33` | `3/45` | stable |
| `soft_mask_only` | `16/33` | `14/45` | `48.5%` to `31.1%` |
| `needs_review` | `17/33` | `31/45` | `51.5%` to `68.9%` |
| `ring_sector` | `33/33` | `45/45` | unchanged collapse |

The confidence decline is not accompanied by worse localization, greater
provider disagreement, or smaller masks. That makes within-category content,
candidate-pool composition, and selector calibration/transfer the leading
post-evaluation hypotheses. It does not establish that the later masks are
worse: expected IoU is a model prediction, and official overlap remains
unknown.

Across all 78 pcb2 rows:

```text
soft_mask_only: 30
needs_review: 48
mean expected IoU: 0.2106
mean conformal IoU lower bound: 0.0751
mean source disagreement: 0.4274
mean selected-mask area: 0.0437
Qwen valid localization: 73/78
structure profile: ring_sector 78/78
```

Proposal selection remains diverse. The largest families are
`fused_q950_component_1` (`15`), `fused_q975` (`14`), `fused_q900` (`10`), and
`edge_fused_q900_1` (`8`). The confidence change is therefore not a complete
collapse to one proposal mode.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 833
records completed: 878
new durable records: 45
unique sample identities: 878
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 12,292
expected retained PNGs: 12,292
corrupt retained PNGs: 0
uncheckpointed pcb2 row-078 residue: 0
official masks opened: no
```

Partial metadata SHA-256:

```text
a1f5a946b89e6526b21e39931722f32300d704038e55b68cb7c900e3394432fa
```

## Professional Interpretation

Checkpoint 878 is operationally healthy and scientifically useful. It
replicates the earlier pcb2 evidence regime while exposing a real weakness in
confidence transfer inside one external category. The correct response is to
finish the frozen run and quantify selected Dice, oracle gap, calibration, and
risk coverage once. Changing thresholds or providers now would contaminate the
external-generalization experiment.
