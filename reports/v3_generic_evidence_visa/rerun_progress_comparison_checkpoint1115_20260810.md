# VisA Checkpoint 1115 Versus Checkpoint 1070

## Decision

Continue the unchanged frozen run. Pcb4 is now complete, pipe_fryum has opened,
and the official VisA masks remain sealed. The remaining 85 runtime images are
an execution task, not an invitation to tune the architecture.

## Execution Comparison

| Measure | Checkpoint 1070 segment | Checkpoint 1115 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `45` | `45` | matched count |
| Runtime | `897.043 s` | `922.569 s` | `+25.526 s` |
| Segment rate | `19.934 s/image` | `20.502 s/image` | `+2.85%` |
| Cumulative progress | `89.17%` | `92.92%` | `+3.75 pp` |
| Run artifacts | `4,813.3 MiB` | `5,014.4 MiB` | `+201.1 MiB` |
| Resume count | `26` | `27` | expected |
| Duplicate identities | `0` | `0` | no regression |

Cumulative productive throughput including the earlier recovery finalizer is
`21.158 s/image`, leaving approximately `0.50` hours. The 2.85% segment-rate
change is operational noise at this sample size, not evidence of a stable
throughput regression.

## Pcb4 Completion

| Diagnostic | Previous 45 pcb4 | Final 30 pcb4 | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2823` | `0.2891` | `+0.0068` |
| Mean conformal IoU lower bound | `0.1468` | `0.1536` | `+0.0069` |
| Mean source disagreement | `0.4207` | `0.4003` | `-0.0204` |
| Mean selected-mask area | `0.0797` | `0.0632` | `-0.0165` |
| Qwen valid localization | `45/45` | `30/30` | unchanged `100%` |
| `soft_mask_only` | `41/45` | `28/30` | `91.1%` to `93.3%` |
| `needs_review` | `4/45` | `2/30` | sparse abstention persists |
| `ring_sector` | `44/45` | `29/30` | same profile mix |

The final slice is slightly more confident, has lower disagreement, and selects
smaller masks. Its two review rows are pcb4 `085` and `097`. Completed pcb4 is:

```text
rows: 100
soft_mask_only: 94
needs_review: 6
mean expected IoU: 0.2866
mean conformal IoU lower bound: 0.1511
mean source disagreement: 0.4168
mean selected-mask area: 0.0726
Qwen valid localization: 100/100
structure profile: ring_sector 98, repeated_chain 2
```

This is meaningful risk-coverage behavior but not yet proof of useful
calibration. Locked evaluation must establish whether the six review cases
actually have lower Dice, precision, or recall.

## Pipe Fryum Opening

The first 15 pipe_fryum rows are uniformly accepted:

```text
soft_mask_only: 15/15
mean expected IoU: 0.4766
mean conformal IoU lower bound: 0.3411
mean source disagreement: 0.4897
mean selected-mask area: 0.0661
Qwen valid localization: 14/15
Qwen full-image fallback: 1/15
structure profile: ring_sector 15/15
```

Ten rows select `fused_q900_component_1`. The high confidence survives the one
Qwen fallback, which is directionally consistent with localization acting as a
soft prior. However, a 15-row opening slice with concentrated selection cannot
support a quality or generalization claim before the locked masks are opened.

## Shutdown And Integrity

The run received exactly one manual `Ctrl-C` and finalized
`last_run_status.json` as `interrupted`. The interrupt landed during proposal
construction for uncheckpointed pipe_fryum `015`, leaving one fused-evidence
PNG. That orphan was removed before the retained-artifact audit.

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 1,070
records completed: 1,115
unique sample identities: 1,115
duplicate identities: 0
missing referenced PNGs: 0
retained PNGs: 15,610
expected retained PNGs: 15,610
orphan retained PNGs: 0
corrupt retained PNGs: 0
uncheckpointed pipe_fryum row-015 residue: 0
official masks opened: no
```

Partial metadata SHA-256:

```text
ad3e05bf745f76a35b68834915e0ab2146c96681d889f9f6d91295056a497bb5
```

## Professional Interpretation

The execution remains healthy and the category transition adds useful
unscored transfer diagnostics. Pcb4 demonstrates sparse abstention under
perfect localization; pipe_fryum opens with uniformly high calibrated
confidence despite one localization fallback. Neither pattern can be judged as
correct until all 1,200 runtime rows are sealed and the preregistered locked
evaluation is run once. The correct next action is therefore completion, not
architecture modification.
