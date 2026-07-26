# VisA RC2 Current-vs-Previous Rerun

## Scope

This comparison uses the same 12-category, one-anomaly-per-category,
mask-isolated VisA smoke cohort. It tests implementation reproducibility and
runtime behavior only. Official VisA masks were not opened.

Compared runs:

```text
previous: 20260726T112627Z-a5349ba3
current:  20260726T160008Z-03249972
current code checkpoint: ed8ac89
```

## Artifact Equivalence

| Check | Difference |
| --- | ---: |
| Sample keys | `0 / 12` |
| Qwen regions | `0 / 12` |
| Selected refinements | `0 / 12` |
| Candidate scores | `0 / 12` |
| Candidate measurements | `0 / 12` |
| Calibrated candidate predictions | `0 / 12` |
| Candidate modes/rejections | `0 / 12` |
| Selection decisions | `0 / 12` |
| Fused evidence maps | `0 / 12` |
| Mask-role files | `0 / 120` |

Disposition counts are also identical:

```text
hard_mask_ok: 0
soft_mask_only: 9
needs_review: 3
```

The only normalized metadata differences are run-specific paths, run IDs, and
provider timing. Run artifacts changed from `36,250,017` to `36,250,014`
bytes; the three-byte difference is confined to `metadata.jsonl`.

## Runtime

| Metric | Previous | Current | Delta |
| --- | ---: | ---: | ---: |
| End-to-end elapsed time | `156.508 s` | `173.081 s` | `+16.573 s` (`+10.6%`) |
| DINO evidence mean | `0.4713 s/image` | `0.6248 s/image` | `+0.1535 s/image` |
| Texture evidence mean | `1.2710 s/image` | `1.1661 s/image` | `-0.1049 s/image` |
| Peak CUDA memory | not durably recorded | `714,032,128 bytes` | not comparable |
| Peak process RSS | not durably recorded | `9,202,384,896 bytes` | not comparable |

This single paired execution does not establish a stable throughput regression,
but the current run is measurably slower and should not be described as a
speedup. Most of the `16.573 s` difference is outside the two timed evidence
providers, consistent with startup, model-loading, governance hashing, SAM, or
system-load variation.

## Judgment

The RC2 operational sprint is behavior-preserving: the selected masks,
confidence/disposition decisions, fused evidence, and every retained mask role
match the previous implementation exactly. Resume and reseal support therefore
did not alter inference.

It also did not improve mask quality, which was expected. The unchanged
`0 / 9 / 3` disposition split remains weak external diagnostic evidence and
cannot support a VisA quality claim until the full runtime inference and locked
evaluation complete.
