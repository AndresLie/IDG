# VisA Completion Versus Checkpoint 1115

## Execution Comparison

| Measure | Checkpoint 1115 | Final cohort | Change |
| --- | ---: | ---: | ---: |
| Durable rows | `1,115` | `1,200` | `+85` |
| Progress | `92.92%` | `100.00%` | `+7.08 pp` |
| Segment runtime | `922.569 s / 45` | `1,732.696 s / 85` | matched rate |
| Segment rate | `20.502 s/image` | `20.385 s/image` | `-0.57%` |
| Retained artifacts | `5,014.4 MiB` | `5,338.9 MiB` | `+324.5 MiB` |
| Resume count | `27` | `28` | expected |

The final continuation completed naturally and published stable metadata. It
did not require interruption cleanup.

## Final Segment

The remaining 85 pipe_fryum rows contain:

```text
soft_mask_only: 84
hard_mask_ok: 1
mean expected IoU: 0.5255
mean conformal lower bound: 0.3900
mean source disagreement: 0.4775
mean selected area: 0.0851
Qwen valid localization: 67/85
Qwen fallback: 18/85
```

Completed pipe_fryum reaches 100% accepted coverage and predicted expected IoU
`0.5181`. The locked evaluation shows that this confidence was badly
miscalibrated: category Dice is only `0.1389`.

## Final Integrity

```text
stable rows: 1,200
unique identities: 1,200
stable/run metadata byte-identical: yes
retained PNGs: 16,800
expected PNGs: 16,800
missing PNGs: 0
orphan PNGs: 0
corrupt PNGs: 0
partial checkpoint: absent
temporary checkpoint: absent
```

Stable runtime SHA-256:

```text
7a086751bbc4609fdfc9a01096d2b6258c173809738d30f889f2a436d99489e3
```

## Locked Outcome

The one-shot evaluation opened official masks only after the runtime cohort was
sealed.

```text
macro Dice: 0.1593 [0.0941, 0.2503]
macro precision: 0.1419
macro recall: 0.7488
macro search recall: 0.5898
macro pixel AP: 0.2085
macro AUPRO: 0.8717
accepted coverage: 0.5267
```

All five Sprint 5 acceptance gates fail. Generation Sprint 6 must not proceed
with this mask corpus. The next architecture must be calibration-first and use
a new untouched dataset for locked confirmation.
