# VisA Locked-Evaluation Retention Review

## Scope

This is a runtime and artifact-retention validation on the mask-isolated VisA
smoke tree. It does not open or score against official VisA masks.

Compared runs:

```text
previous full retention:
outputs/v3_generic_evidence_visa_smoke/auto_masks/qwen/runs/20260726T083031Z-11f6833f

current locked-evaluation retention:
outputs/v3_generic_evidence_visa_retention_smoke/auto_masks/qwen/runs/20260726T110435Z-d885573b
```

## Equivalence Result

| Check | Result |
| --- | ---: |
| Runtime samples | `12 / 12` matched |
| Selected-mode differences | `0` |
| Selection-decision differences | `0` |
| Fused-score-map differences | `0` |
| Mask-role differences | `0 / 120` |
| Previous dispositions | `0` hard, `9` soft, `3` review |
| Current dispositions | `0` hard, `9` soft, `3` review |

The retained roles are:

```text
generation_core
eval_tight
eval_mvtec
training_medium
training_wide
training_soft
positive_core
possible_region
uncertainty_map
inpaint_soft
```

The locked evaluator still receives `eval_tight` and the fused score map, so
Dice/IoU/precision/recall and ranking metrics remain computable.

## Storage Result

| Run artifact group | Full | Locked-evaluation |
| --- | ---: | ---: |
| Masks and score maps | `45.03 MiB` | `10.02 MiB` |
| Mask roles | `23.76 MiB` | `23.76 MiB` |
| Overlays | `16.93 MiB` | `0` |
| Metadata | `0.88 MiB` | `0.79 MiB` |
| Total per-run artifacts | `86.59 MiB` | `34.57 MiB` |

Reduction:

```text
52.02 MiB per 12 images
60.1% of per-run artifact storage
approximately 5.1 GiB saved over 1,200 images
```

The profile omits:

```text
provider evidence preview PNGs
source-disagreement preview PNGs
edge-aligned preview PNGs
unselected candidate-mask PNGs
full-resolution overlays
```

Candidate scores, measurements, calibrated predictions, selection decisions,
provider metadata, and timing remain serialized in metadata.

## One-Shot Cache Result

The follow-up one-shot profile keeps reusable normal-only work and disables
target evidence persistence:

```text
72 shared DINO normal-token files retained
0 target DINO evidence .npz files
0 target texture-residual evidence .npz files
about 217 MiB persistent shared cache
```

The one-shot run again has zero selected-mode, decision, fused-map, or mask-role
differences from full retention. Full 1,200-image storage now projects to:

```text
about 3.4 GiB lean run artifacts
about 0.2 GiB shared normal cache
about 3.6 GiB total, excluding small manifests/reports
```

This trades target-evidence warm replay for a safe locked-run footprint. Shared
normal encoding remains reusable. A later production hardening sprint should
add resumable per-image checkpoints before using this mode for unattended runs.

The real run also exposed a read-only NumPy warning in torchvision. DINO and SAM
boundaries now materialize writable RGB arrays, and a subsequent candle
integration run completed without the warning.

## Scientific Gate

This change is operational and mask-equivalent, but source changes alter the
architecture package hash. It must not silently replace the existing frozen
commit. The valid next choices are:

1. revalidate and seal this behavior-equivalent implementation as a new RC;
2. run the exact frozen worktree with additional storage.

No VisA mask-quality conclusion is made by this report.

## Current Rerun Comparison

The warning-free current implementation was rerun on the same 12-category
mask-isolated smoke cohort:

```text
previous: 20260726T111959Z-50507ce8
current:  20260726T112627Z-a5349ba3
```

| Check | Previous | Current | Difference |
| --- | ---: | ---: | ---: |
| Samples | `12` | `12` | `0` |
| Selected modes changed | - | - | `0` |
| Selection decisions changed | - | - | `0` |
| Qwen regions changed | - | - | `0` |
| Candidate-score rows changed | - | - | `0` |
| Candidate-measurement rows changed | - | - | `0` |
| Calibrated-prediction rows changed | - | - | `0` |
| Fused score maps changed | - | - | `0 / 12` |
| Mask roles changed | - | - | `0 / 120` |
| Hard / soft / review | `0 / 9 / 3` | `0 / 9 / 3` | `0` |
| Run artifact bytes | `36,250,017` | `36,250,017` | `0` |
| Elapsed time | `164.761 s` | `156.508 s` | `-8.253 s` (`-5.0%`) |
| Target evidence cache files | `0` | `0` | `0` |
| Shared normal-token files | `72` | `72` | `0` |

The current run emitted no read-only NumPy warning.

The experiment architecture fingerprint changed from
`3bf31ec03e8ae97e3601d6561ae4b66d1e829e549b9d99ef7aaeabd05dc47bf9`
to
`81512bc8cc6b3fe801b3bfffbc5e698293fe1bfb515607ac2b150107fa5b7607`
because the package source now includes the SAM writable-array boundary fix.
This is a provenance change, not an inference change; the strict artifact
comparison above verifies behavioral equivalence on this cohort.

Official VisA masks remained unopened.
