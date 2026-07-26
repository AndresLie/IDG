# VisA Checkpoint 159 Versus Checkpoint 115

## Decision

Continue the frozen run. The fourth segment passes operational integrity, but
external mask quality remains unmeasured.

The segment adds a category-matched capsule slice, which makes the confidence
comparison more useful than the preceding mixed candle/capsule segment. It
shows modestly better predicted quality but persistent source disagreement and
abstention.

## Execution Comparison

| Measure | Checkpoint 115 | Checkpoint 159 | Segment change |
| --- | ---: | ---: | ---: |
| Completed rows | `115` | `159` | `+44` |
| Cohort progress | `9.58%` | `13.25%` | `+3.67 pp` |
| Segment runtime | `973.517 s` | `959.831 s` | different row counts |
| Segment rate | `17.700 s/image` | `21.814 s/image` | `+23.24%` slower |
| Run artifacts | `459.9 MiB` | `682.8 MiB` | `+222.9 MiB` |
| Resume count | `2` | `3` | expected |
| Duplicate identities | `0` | `0` | no regression |

The slowdown is associated with the all-capsule segment. It is not an
architecture-fingerprint change or failed cache resume.

## Capsule Confidence Comparison

| Diagnostic | First 15 capsules | Next 44 capsules | Change |
| --- | ---: | ---: | ---: |
| Mean expected IoU | `0.1078` | `0.1289` | `+0.0212` |
| Median expected IoU | `0.0808` | `0.0885` | `+0.0077` |
| Mean conformal IoU lower bound | `0.0202` | `0.0300` | `+0.0098` |
| Mean source disagreement | `0.6291` | `0.6436` | `+0.0144` |
| Mean selected-mask area | `0.0404` | `0.0317` | `-0.0087` |
| `soft_mask_only` coverage | `2/15` | `8/44` | `13.3% -> 18.2%` |
| Qwen full-image fallback | `4/15` | `12/44` | `26.7% -> 27.3%` |

The selector is slightly more optimistic on the later capsule slice, but the
confidence interval remains weak and source disagreement does not improve.
This does not satisfy a quality gate.

## Structure Diagnostics

The previous 115 rows were all labeled `ring_sector`. The next 44 capsule rows
contain:

```text
ring_sector: 33
repeated_chain: 11
```

This disproves a total structure-profile collapse, but `ring_sector` still
dominates. Specialists are disabled, so the profile does not change selected
masks in this frozen external run.

## Integrity

```text
same run ID: yes
same architecture fingerprint: yes
same auto-mask fingerprint: yes
records resumed: 115
records completed: 159
new records: 44
duplicate sample identities: 0
missing eval masks: 0
missing training masks: 0
missing uncertainty masks: 0
official masks opened: no
```

## Professional Interpretation

The resumable production architecture continues to work correctly. The capsule
runtime cost is higher and should be reported, but it is manageable within the
remaining disk and compute budget.

The quality warning remains the dominant result: `149/159` cumulative rows are
`needs_review`, and no row is `hard_mask_ok`. This may represent conservative
calibration rather than universally bad masks, but it confirms that development
confidence thresholds do not transfer cleanly to VisA.

Do not alter thresholds or providers during the sealed cohort. Complete all
1,200 runtime records, finalize the manifest, and use the one-shot locked
evaluation to distinguish calibration failure from mask-generation failure.
