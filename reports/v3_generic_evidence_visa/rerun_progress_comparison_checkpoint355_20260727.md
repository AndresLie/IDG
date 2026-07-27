# VisA Checkpoint 355 Versus Checkpoint 291

## Decision

Continue the frozen run. The cashew-to-chewing-gum transition substantially
improves throughput and soft-mask acceptance, while introducing a strong SAM2
selection pattern and very high source disagreement.

## Execution Comparison

| Measure | Checkpoint 291 segment | Checkpoint 355 segment | Change |
| --- | ---: | ---: | ---: |
| New rows | `30` | `64` | mixed category |
| Segment runtime | `998.798 s` | `985.655 s` | `-13.143 s` |
| Segment rate | `33.293 s/image` | `15.401 s/image` | `-53.74%` |
| Cumulative progress | `24.25%` | `29.58%` | `+5.33 pp` |
| Run artifacts | `1,484.0 MiB` | `1,665.1 MiB` | `+181.1 MiB` |
| Resume count | `7` | `8` | expected |
| Duplicate identities | `0` | `0` | no regression |

The segment completes cashew and enters chewing gum. Its throughput improvement
is category-dependent, not a code or cache change.

## Final Cashew Slice

The final nine cashew rows are easier according to the frozen selector:

| Diagnostic | Final nine cashews |
| --- | ---: |
| Mean expected IoU | `0.3288` |
| Median expected IoU | `0.3477` |
| Mean conformal IoU lower bound | `0.1933` |
| Mean source disagreement | `0.8829` |
| Qwen valid localization | `8/9` |
| `soft_mask_only` | `6/9` |
| `needs_review` | `3/9` |

Expected quality and acceptance are high, but disagreement is also extreme.
This small terminal slice should not be generalized to all cashew samples.

## Chewing-Gum Transition

| Diagnostic | First 55 chewing-gum rows |
| --- | ---: |
| Mean expected IoU | `0.2415` |
| Median expected IoU | `0.2558` |
| Mean conformal IoU lower bound | `0.1117` |
| Mean source disagreement | `0.8225` |
| Mean selected-mask area | `0.0174` |
| Qwen valid localization | `53/55` |
| `soft_mask_only` | `29/55` |
| `needs_review` | `26/55` |
| `sam2_fused_q850_1` selected | `29/55` |

This is the highest soft-mask acceptance observed in the full VisA run so far.
The selected mode distribution changes materially: SAM2 refinement accounts for
exactly the same number of rows as `soft_mask_only`.

That association is diagnostic, not causal. Locked evaluation must compare
SAM2-selected and non-SAM2 rows before claiming that refinement improves mask
quality.

## Structure Diagnostics

Chewing gum remains classified almost entirely as `ring_sector` (`54/55`), with
one `unknown`. Structural specialists are disabled, but the profile remains too
collapsed to support a generic structure-understanding claim.

## Integrity

```text
same run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 291
records completed: 355
new records: 64
duplicate sample identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
retained PNGs: 4,971
corrupt retained PNGs: 0
official masks opened: no
```

## Professional Interpretation

The transition demonstrates that frozen architecture behavior is strongly
category-dependent:

- cashew is slow and usually abstains;
- chewing gum is fast, frequently uses SAM2, and reaches near-even
  soft-mask/review coverage;
- high disagreement persists even on accepted rows.

The next locked analysis should stratify Dice and calibration by category,
Qwen fallback, SAM2 selection, and source disagreement. Until then, the
improved acceptance rate is promising but unverified.
