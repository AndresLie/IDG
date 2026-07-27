# VisA Full Runtime Inference Progress

## Scope

This is an execution checkpoint for the frozen
`v3-generic-evidence-rc2-operational-20260726` architecture. It is not an
evaluation result. Official VisA masks remain sealed.

Runtime run:

```text
run_id: 20260726T160703Z-fcd6745c
runtime samples expected: 1,200
runtime samples completed: 291
progress: 24.25%
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

A sixth bounded command:

- reused all 201 checkpoint rows and incremented `resume_count` to 5;
- added 30 cashew rows;
- stopped at 231 unique records;
- interrupted during Qwen inference, leaving no partial image artifact;
- preserved all checkpoint-local masks and readable retained PNGs.

A seventh bounded command:

- reused all 231 checkpoint rows and incremented `resume_count` to 6;
- added a category-matched 30-row cashew segment;
- stopped at 261 unique records;
- interrupted during proposal construction without corrupting retained images;
- preserved the same frozen fingerprints and sealed reference boundary.

An eighth bounded command:

- reused all 261 checkpoint rows and incremented `resume_count` to 7;
- added another category-matched 30-row cashew segment;
- stopped at 291 unique records;
- interrupted while writing the next uncheckpointed sample overlay;
- removed all 14 artifacts for incomplete `cashew_bad_091`, leaving every
  checkpointed row and retained image intact.

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
| Completed rows | `291 / 1,200` |
| Current run artifacts | `1,484.0 MiB` |
| Previous exact rate, first 60 rows | `17.825 s/image` |
| Latest exact rate, next 55 rows | `17.700 s/image` |
| Fourth-segment rate, 44 capsule rows | `21.814 s/image` |
| Fourth-vs-third segment rate change | `+23.24%` |
| Fifth-segment rate, 42 rows | `23.014 s/image` |
| Fifth-vs-fourth segment rate change | `+5.50%` |
| Sixth-segment rate, 30 cashew rows | `32.720 s/image` |
| Sixth-vs-fifth segment rate change | `+42.17%` |
| Seventh-segment rate, 30 cashew rows | `32.492 s/image` |
| Seventh-vs-sixth segment rate change | `-0.70%` |
| Eighth-segment rate, 30 cashew rows | `33.293 s/image` |
| Eighth-vs-seventh segment rate change | `+2.47%` |
| Cumulative exact rate | `23.796 s/image` |
| Projected remaining compute | approximately `6.01 hours` |
| Projected total compute | approximately `7.93 hours` |
| Linear artifact projection | approximately `5.98 GiB` |
| Free storage after checkpoint | approximately `15 GiB` |

The projection is operational only. Category transitions and cache reuse may
change the final rate and footprint.

## Unscored Diagnostics

The checkpoint now contains all candle and capsule anomalies plus the first 91
cashew anomalies:

```text
candle: 100
capsules: 100
cashew: 91

needs_review: 259
soft_mask_only: 32
hard_mask_ok: 0
```

Selected proposal modes over all 291 rows:

```text
fused_q975: 121
fused_q950: 60
fused_q900: 39
fused_q950_component_1: 24
fused_q975_component_1: 17
fused_q900_component_1: 13
fused_q850_component_1: 6
fused_q850: 2
edge_fused_q900_component_1_2: 2
edge_fused_q850_component_1_1: 4
edge_fused_q900_component_1_1: 1
edge_fused_q850_component_1_2: 1
fused_q900_component_2: 1
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

The first 30-row cashew segment is more expensive and more optimistic:

```text
mean expected IoU: 0.1862
mean conformal IoU lower bound: 0.0507
mean source disagreement: 0.6626
Qwen valid localization: 28/30
needs_review: 25/30
soft_mask_only: 5/30
```

The higher expected quality does not produce a reliable acceptance rate because
source disagreement remains high. All 30 cashew rows are labeled `ring_sector`;
specialists remain disabled.

The next category-matched 30-row cashew slice is directionally stronger:

```text
mean expected IoU: 0.2112 versus 0.1862
mean conformal IoU lower bound: 0.0758 versus 0.0507
mean source disagreement: 0.7170 versus 0.6626
Qwen valid localization: 29/30 versus 28/30
soft_mask_only: 9/30 versus 5/30
```

Expected quality and non-review coverage improve, but source disagreement also
worsens. This is evidence of heterogeneous cashew difficulty and selector
confidence variation, not a mask-quality result.

The following matched 30-row cashew slice reverses that confidence gain:

```text
mean expected IoU: 0.1723 versus 0.2112
mean conformal IoU lower bound: 0.0370 versus 0.0758
mean source disagreement: 0.6445 versus 0.7170
Qwen full-image fallback: 8/30 versus 1/30
soft_mask_only: 3/30 versus 9/30
```

Evidence providers agree more closely, but Qwen localization falls back much
more often and selector confidence drops. This makes localization validity the
strongest unscored explanation for the within-category shift.

These are serious confidence-transfer warnings, but they are not official mask
quality measurements. The preregistered run must complete before opening
official masks or changing selector thresholds.

Checkpoint integrity:

```text
partial metadata SHA-256:
c89c5374239d18497bd0369460f395b89273e476990c609be3819133898b4cbe

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
