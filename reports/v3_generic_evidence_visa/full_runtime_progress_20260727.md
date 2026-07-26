# VisA Full Runtime Inference Progress

## Scope

This is an execution checkpoint for the frozen
`v3-generic-evidence-rc2-operational-20260726` architecture. It is not an
evaluation result. Official VisA masks remain sealed.

Runtime run:

```text
run_id: 20260726T160703Z-fcd6745c
runtime samples expected: 1,200
runtime samples completed: 115
progress: 9.58%
```

## Resume Validation

The first bounded segment was intentionally interrupted after 55 complete
records. The second command:

- matched the original auto-mask and architecture fingerprints;
- reused run ID `20260726T160703Z-fcd6745c`;
- reported `records_resumed: 55` and `resume_count: 1`;
- continued to 60 records without recomputing the recovered rows;
- wrote a new valid partial checkpoint after a second graceful interruption.

A third bounded command then:

- matched the same auto-mask and architecture fingerprints;
- reused the same run ID;
- reported `records_resumed: 60` and `resume_count: 2`;
- added 55 unique records and stopped at 115;
- preserved all run-local evaluation, training, and uncertainty-mask artifacts;
- produced no duplicate sample identity keys.

Current checkpoint:

```text
outputs/v3_generic_evidence_visa/auto_masks/qwen/runs/
  20260726T160703Z-fcd6745c/metadata.partial.jsonl
```

The stable runtime manifest has not been published, which is correct for an
incomplete locked run.

## Runtime Projection

| Observation | Value |
| --- | ---: |
| Completed rows | `115 / 1,200` |
| Current run artifacts | `459.9 MiB` |
| Previous exact rate, first 60 rows | `17.825 s/image` |
| Latest exact rate, next 55 rows | `17.700 s/image` |
| Latest-vs-previous rate change | `-0.70%` |
| Cumulative exact rate | `17.765 s/image` |
| Projected remaining compute | approximately `5.35 hours` |
| Projected total compute | approximately `5.92 hours` |
| Linear artifact projection | approximately `4.69 GiB` |
| Free storage after checkpoint | approximately `16 GiB` |

The projection is operational only. Category transitions and cache reuse may
change the final rate and footprint.

## Unscored Diagnostics

The first checkpoint contained 60 `candle` rows. The new segment contains 40
additional `candle` rows and the first 15 `capsules` rows:

```text
candle: 100
capsules: 15

needs_review: 113
soft_mask_only: 2
hard_mask_ok: 0
```

Selected proposal modes over all 115 rows:

```text
fused_q975: 70
fused_q950: 24
fused_q900_component_1: 8
fused_q900: 5
fused_q850_component_1: 4
fused_q950_component_1: 2
fused_q850: 1
edge_fused_q900_component_1_2: 1
```

The latest segment's exact selector diagnostics are:

| Diagnostic | Previous 60 | New 55 |
| --- | ---: | ---: |
| Mean expected IoU | `0.1630` | `0.1505` |
| Mean conformal IoU lower bound | `0.0275` | `0.0281` |
| Mean source disagreement | `0.3792` | `0.4643` |
| Mean selected-mask area | `0.0322` | `0.0329` |
| Qwen full-image fallback | `40/60` | `24/55` |

The aggregate expected-IoU drop is mostly category-composition drift, not a
same-category regression. The 40 new candle rows average `0.1665` expected IoU,
while the first 15 capsule rows average only `0.1078` and source disagreement
`0.6291`. All 115 rows are still labeled `ring_sector`, which is an additional
structure-profile transfer warning. Structural specialists are disabled, so
that label does not route the frozen inference path.

These are serious confidence-transfer warnings, but they are not official mask
quality measurements. The preregistered run must complete before opening
official masks or changing selector thresholds.

Checkpoint integrity:

```text
partial metadata SHA-256:
e1e245b1565dca52ea81fbf7807e318c6dcbff071342a8f7eeb21ab5799db4f4

architecture fingerprint:
f946c2ea91eaa6e3727f1f0c2d13fc5518e0daf113404638c1288b90721daf23

auto-mask fingerprint:
8aceb1c68c1f8c76b256252938208e4292863a4a7dedf035878a52a700b61fd2
```

## Next Execution

Resume with the unchanged code and configuration:

```bash
python -m iadgen_v2.cli auto-masks \
  --config configs/v3_generic_evidence_visa.yaml
```

Any package-code, selector, checkpoint, model, or behavioral-config change will
correctly invalidate this checkpoint and start a fresh cohort.
