# VisA Checkpoint 696 Versus Checkpoint 655

## Decision

Continue the unchanged frozen run. The new macaroni2 slice shows that valid
Qwen localization alone does not guarantee confident selection: all 41 new
rows have valid localization, but evidence disagreement rises and soft-mask
coverage falls. Official masks remain sealed, so this is a confidence-transfer
finding rather than a Dice or IoU result.

## Execution Comparison

| Measure | Checkpoint 655 segment | Checkpoint 696 segment | Change |
| --- | ---: | ---: | ---: |
| New durable rows | `44` | `41` | macaroni2-only |
| Productive segment runtime | `918.526 s` | `898.173 s` | `-20.354 s` |
| Productive segment rate | `20.876 s/image` | `21.907 s/image` | `+4.94%` |
| Cumulative progress | `54.58%` | `58.00%` | `+3.42 pp` |
| Run artifacts | `2,903.8 MiB` | `3,053.1 MiB` | `+149.3 MiB` |
| Resume count | `15` | `17` | includes one zero-row environment failure |
| Duplicate identities | `0` | `0` | no regression |

The productive cumulative rate is `22.162 s/image`, leaving approximately
`3.10` hours. A restricted-sandbox attempt added zero rows because GPU access
was unavailable; it is finalized as a failed environment manifest and excluded
from productive timing. A defensive relaunch then rejected the stale
namespace-local PID before any inference state changed.

## Macaroni2 Comparison

| Diagnostic | Previous 44 | New 41 | First 96 |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.2186` | `0.2156` | `0.2174` |
| Mean conformal IoU lower bound | `0.0831` | `0.0801` | `0.0819` |
| Mean source disagreement | `0.3732` | `0.4261` | `0.4100` |
| Mean selected-mask area | `0.0331` | `0.0358` | `0.0354` |
| Qwen valid localization | `27/44` | `41/41` | `75/96` |
| Qwen full-image fallback | `17/44` | `0/41` | `21/96` |
| `soft_mask_only` | `16/44` | `11/41` | `30/96` |
| `needs_review` | `28/44` | `30/41` | `66/96` |
| `repeated_chain` | `21/44` | `24/41` | `51/96` |
| `ring_sector` | `23/44` | `17/41` | `45/96` |

The selected disposition becomes more conservative despite perfect
localization. This weakens a Qwen-box explanation for the confidence drop and
places more weight on evidence-source disagreement and selector calibration.

## Proposal Diversity

```text
first 96 selected modes:
fused_q975: 29
fused_q950: 26
fused_q900_component_1: 10
edge_fused_q850_component_1_1: 9
fused_q900: 7
fused_q900_component_2: 6
fused_q950_component_1: 4
edge_fused_q900_component_1_2: 2
edge_fused_q900_component_2_2: 1
edge_fused_q850_component_2_1: 1
fused_q900_component_3: 1
```

No single mode dominates macaroni2, and the structure profile remains mixed.
The generic proposal path is therefore still expressive even when acceptance
is conservative.

## Integrity

```text
same frozen run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed by productive run: 655
records completed: 696
new durable records: 41
unique sample identities: 696
duplicate identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 9,744
expected retained PNGs: 9,744
corrupt retained PNGs: 0
uncheckpointed row-096 residue removed: yes
official masks opened: no
```

## Professional Interpretation

Checkpoint 696 strengthens the diagnosis rather than the quality claim.
Macaroni2 has broad proposal and structure diversity, but perfect Qwen validity
does not improve predicted quality or acceptance in the latest slice. The
locked evaluation should therefore partition selector regret by localization
status, source disagreement, structure profile, and selected proposal family.
No threshold or architecture change should be made before that evaluation.
