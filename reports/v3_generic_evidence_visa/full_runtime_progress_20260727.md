# VisA Full Runtime Inference Progress

## Scope

This is an execution checkpoint for the frozen
`v3-generic-evidence-rc2-operational-20260726` architecture. It is not an
evaluation result. Official VisA masks remain sealed.

Runtime run:

```text
run_id: 20260726T160703Z-fcd6745c
runtime samples expected: 1,200
runtime samples completed: 201
progress: 16.75%
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

A fourth bounded command:

- reused all 115 checkpoint rows under the same fingerprints and run ID;
- incremented `resume_count` from 2 to 3;
- added 44 capsule records;
- stopped cleanly at 159 unique records;
- again left the stable runtime manifest unpublished.

A fifth bounded command:

- reused all 159 checkpoint rows and incremented `resume_count` to 4;
- added the final 41 capsule rows and the first cashew row;
- stopped at 201 unique records with no checkpoint duplicates;
- left one corrupt, uncheckpointed `cashew_bad_001` PNG after interruption
  during a file write; all artifacts for that incomplete sample were removed,
  while all 201 checkpoint rows remained intact.

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
| Completed rows | `201 / 1,200` |
| Current run artifacts | `905.2 MiB` |
| Previous exact rate, first 60 rows | `17.825 s/image` |
| Latest exact rate, next 55 rows | `17.700 s/image` |
| Fourth-segment rate, 44 capsule rows | `21.814 s/image` |
| Fourth-vs-third segment rate change | `+23.24%` |
| Fifth-segment rate, 42 rows | `23.014 s/image` |
| Fifth-vs-fourth segment rate change | `+5.50%` |
| Cumulative exact rate | `19.748 s/image` |
| Projected remaining compute | approximately `5.48 hours` |
| Projected total compute | approximately `6.58 hours` |
| Linear artifact projection | approximately `5.28 GiB` |
| Free storage after checkpoint | approximately `16 GiB` |

The projection is operational only. Category transitions and cache reuse may
change the final rate and footprint.

## Unscored Diagnostics

The checkpoint now contains all candle and capsule anomalies plus the first
cashew anomaly:

```text
candle: 100
capsules: 100
cashew: 1

needs_review: 186
soft_mask_only: 15
hard_mask_ok: 0
```

Selected proposal modes over all 201 rows:

```text
fused_q975: 111
fused_q950: 29
fused_q900: 15
fused_q950_component_1: 14
fused_q975_component_1: 14
fused_q900_component_1: 9
fused_q850_component_1: 4
fused_q850: 2
edge_fused_q900_component_1_2: 1
edge_fused_q850_component_1_1: 1
edge_fused_q900_component_1_1: 1
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
same-category regression. The 40-row candle continuation averaged `0.1665`
expected IoU. The first 15 capsule rows averaged only `0.1078`, while the next
44 capsule rows improved modestly to `0.1289`. Capsule source disagreement
remains high at `0.6436`.

The structure profile is no longer completely collapsed: 11 of the latest 44
capsule rows are `repeated_chain`; the other 33 remain `ring_sector`. Structural
specialists are disabled, so these labels do not route the frozen inference
path.

The final 41 capsule rows strengthen that observation: 15 are
`repeated_chain`, and 26 are `ring_sector`. Relative to the preceding 44
capsules, mean expected IoU improves from `0.1289` to `0.1432`, mean source
disagreement falls from `0.6436` to `0.6183`, and Qwen full-image fallback
falls from `27.3%` to `17.1%`. Despite those directional improvements, 36 of 41
rows remain `needs_review`.

These are serious confidence-transfer warnings, but they are not official mask
quality measurements. The preregistered run must complete before opening
official masks or changing selector thresholds.

Checkpoint integrity:

```text
partial metadata SHA-256:
b0da2fbbc8bc18a54095a31b3f36bd59d146b5a82aa91497cc89c1e00e29b21f

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
