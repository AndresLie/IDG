# V4.2e Reproducibility Rerun Comparison

## Verdict

The committed V4.2e rerun is numerically and artifact reproducible. The candidate
model, candidate-row cache, leave-dataset-out metrics, and generated report are
byte-identical to the first run. The scientific decision is unchanged: V4.2e
improves oracle capacity but fails deployment selection gates.

## Artifact Reproduction

| Artifact | SHA-256 | Comparison |
| --- | --- | --- |
| Candidate ranker | `3ee0a782d11ba0e8e829e58e6c732046cf6ded540f3c14b8b71f958bb1ad952f` | Byte-identical |
| Candidate-row cache | `671e78bf2c221bec7b737091a7dde246c7f5ce6ce5fa29216875440e26d3bcb2` | Byte-identical |
| Metrics JSON | `f0b7772a24dc3ffa44e762faaf253f6cfa56948ffed44ca5ba856c35c6a973bd` | Byte-identical |
| Generated report | `06a4f4ff24cd6794f821d053d45ad3644a04d3034a0f9bf00c1e03bb0e6ef332` | Byte-identical |

The calibration manifest intentionally differs because `candidate_row_cache`
changes from `cache_hit: false` to `cache_hit: true`. Its rerun SHA-256 is
`db5d55156a1396782da7f461e108557c82cb669b29889de38974a598509856af`.

## Runtime

| Run | Run ID | Cache | Elapsed |
| --- | --- | --- | ---: |
| First | `20260811T045309Z-df59592d` | Miss | `3,587.50 s` |
| Rerun | `20260811T055725Z-8f953f3f` | Hit | `3,236.32 s` |

The cache saves `351.18 s` (`9.79%`, `1.11x`). Nested all-pairs model fitting
still dominates, so row caching is useful but not a complete performance fix.

## Metric Comparison

The rerun exactly reproduces V4.2e:

| Metric | V4.2d | Current V4.2e | Delta |
| --- | ---: | ---: | ---: |
| Macro selected Dice | `0.3722` | `0.3595` | `-0.0126` |
| Macro oracle Dice | `0.4887` | `0.5029` | `+0.0142` |
| Selection regret | `0.1165` | `0.1433` | `+0.0268` |
| Area ratio | `2.5394` | `2.5105` | `-0.0289` |
| Expected-IoU MAE | `0.0857` | `0.1021` | `+0.0165` |
| Expected-IoU Spearman | `0.6217` | `0.6400` | `+0.0183` |

## Provenance Caveat

Package-code fingerprints and every configured input artifact are identical.
The started-manifest architecture-core fingerprint differs because the training
configuration points `auto_masks.candidate_calibrator_model_path` at its own
output. That file did not exist when the first run started, but existed and was
hashed when the rerun started. This should be corrected in a governance-only
sprint by separating training outputs from deployed architecture inputs or by
recording both pre-run and finalized architecture fingerprints.

## Decision

Keep V4.2e default-off. Reproducibility is confirmed, but reproducible failure of
the deployment gates is still failure. The next quality sprint remains bounded,
uncertainty-aware within-component contour refinement rather than further
component-union expansion or decision-threshold tuning.
