# VisA Full Runtime Inference Progress

## Scope

This is an execution checkpoint for the frozen
`v3-generic-evidence-rc2-operational-20260726` architecture. It is not an
evaluation result. Official VisA masks remain sealed.

Runtime run:

```text
run_id: 20260726T160703Z-fcd6745c
runtime samples expected: 1,200
runtime samples completed: 60
progress: 5.0%
```

## Resume Validation

The first bounded segment was intentionally interrupted after 55 complete
records. The second command:

- matched the original auto-mask and architecture fingerprints;
- reused run ID `20260726T160703Z-fcd6745c`;
- reported `records_resumed: 55` and `resume_count: 1`;
- continued to 60 records without recomputing the recovered rows;
- wrote a new valid partial checkpoint after a second graceful interruption.

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
| Completed rows | `60 / 1,200` |
| Current run artifacts | `235 MiB` |
| Observed processing rate | approximately `16.7 s/image` |
| Projected compute time | approximately `5.6 hours` |
| Linear artifact projection | approximately `4.6 GiB` |
| Free storage at preflight | `16.21 GiB` |

The projection is operational only. Category transitions and cache reuse may
change the final rate and footprint.

## Unscored Diagnostics

All 60 completed rows belong to `candle`:

```text
needs_review: 60
soft_mask_only: 0
hard_mask_ok: 0
```

Selected proposal modes:

```text
fused_q975: 36
fused_q950: 21
fused_q850_component_1: 2
fused_q950_component_1: 1
```

This is a serious confidence-transfer warning, but not evidence that all masks
are wrong. The preregistered run must complete before opening official masks or
changing selector thresholds.

## Next Execution

Resume with the unchanged code and configuration:

```bash
python -m iadgen_v2.cli auto-masks \
  --config configs/v3_generic_evidence_visa.yaml
```

Any package-code, selector, checkpoint, model, or behavioral-config change will
correctly invalidate this checkpoint and start a fresh cohort.
