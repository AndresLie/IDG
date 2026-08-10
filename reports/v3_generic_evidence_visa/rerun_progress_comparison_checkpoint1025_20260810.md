# VisA Checkpoint 1025 Versus Checkpoint 976

## Decision

Continue the unchanged frozen run. Pcb3 is complete with `100/100` soft-mask
acceptance, and pcb4 opens at `25/25` despite lower confidence and smaller
masks. The result replicates permissive PCB dispositions but strengthens the
risk-coverage warning. Official masks remain sealed.

## Execution Comparison

| Measure | Checkpoint 976 segment | Checkpoint 1025 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `51` | `49` | 24 pcb3 plus 25 pcb4 |
| Primary bounded runtime | `898.265 s` | `893.095 s` | `-5.170 s` |
| Primary bounded rows | `51` | `48` | before recovery |
| Primary segment rate | `17.613 s/image` | `18.606 s/image` | `+5.64%` |
| Recovery finalizer | none | `67.524 s`, one row | not rate-comparable |
| Cumulative progress | `81.33%` | `85.42%` | `+4.09 pp` |
| Run artifacts | `4,384.9 MiB` | `4,593.9 MiB` | `+209.0 MiB` |
| Resume count | `23` | `25` | bounded run plus recovery |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput including recovery is `21.240 s/image`,
leaving approximately `1.03` hours. The main segment is 5.64% slower per row
than the previous pcb3-only segment, consistent with category-transition and
startup overhead rather than evidence of an inference regression.

## Completed Pcb3

| Diagnostic | First 76 pcb3 | Final 24 pcb3 | Completed 100 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.3589` | `0.3470` | `0.3561` |
| Mean conformal IoU lower bound | `0.2235` | `0.2115` | `0.2206` |
| Mean source disagreement | `0.3888` | `0.3561` | `0.3810` |
| Mean selected-mask area | `0.1148` | `0.1029` | `0.1119` |
| Qwen valid localization | `60/76` | `21/24` | `81/100` |
| `soft_mask_only` | `76/76` | `24/24` | `100/100` |
| `needs_review` | `0/76` | `0/24` | `0/100` |
| `ring_sector` | `76/76` | `24/24` | `100/100` |

The final slice is modestly less confident and smaller, while source agreement
improves. Uniform acceptance persists through the entire category, so current
disposition cannot rank risk within pcb3.

Completed pcb3 proposal selection remains concentrated in two families:

```text
fused_q850_component_1: 33
edge_fused_q850_2: 29
fused_q900_component_1: 14
edge_fused_q850_component_1_2: 10
edge_fused_q900_component_1_2: 8
fused_q950_component_1: 2
fused_q900_component_3: 2
fused_q900: 1
edge_fused_q900_component_2_2: 1
```

## Pcb4 Opening

```text
rows: 25
soft_mask_only: 25
needs_review: 0
mean expected IoU: 0.2915
mean conformal IoU lower bound: 0.1560
mean source disagreement: 0.4295
mean selected-mask area: 0.0712
Qwen valid localization: 25/25
structure profile: ring_sector 25/25
```

Pcb4 masks are smaller and less confident than pcb3, with higher evidence
disagreement, yet all are accepted. This could indicate that both categories
are genuinely easy, or that the calibrated threshold is too permissive for
PCB-like structures. Locked precision, positive rate, selected Dice, oracle
gap, calibration error, and risk coverage must resolve the ambiguity.

## Recovery Audit

The bounded process flushed 1,024 rows before interruption during proposal
refinement. Cleanup itself was interrupted, leaving a stale `running` status
and `metadata.jsonl.tmp`; no process remained. The next unchanged command used
the built-in orphan-resume contract to:

```text
validate all 1,024 sample identities and fingerprints
verify run-local mask artifacts
recover the same run ID
complete one additional pcb4 row
publish metadata.partial.jsonl with 1,025 rows
finalize status as interrupted
```

This validates data recovery but exposes a next-RC operational weakness:
checkpoint status publication should tolerate a second signal during exception
cleanup. No manual metadata promotion or runtime-content edit was performed.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records completed: 1,025
unique sample identities: 1,025
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 14,350
expected retained PNGs: 14,350
corrupt retained PNGs: 0
uncheckpointed pcb4 row-025 residue: 0
official masks opened: no
```

Partial metadata SHA-256:

```text
4cdd5abab093c84339d677147202ac032323e34b2765d949f24398490f5e72f2
```

## Professional Interpretation

The current architecture remains operationally recoverable and behaviorally
consistent across PCB categories. Its emerging weakness is not localization:
pcb3 and early pcb4 receive universal soft acceptance despite material shifts
in expected IoU, disagreement, and mask area. Finish the frozen cohort, then
use the one-shot locked evaluation to determine whether this is robust
generalization or overconfident over-segmentation.
