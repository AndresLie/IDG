# V4.2b Within-Image Candidate Ranking Review

## Decision

V4.2b is implemented and validated as a development prototype, but it does not
pass promotion. Keep it default-off and preserve V3 as the production fallback.

The final run used `1,356` exposed development images and `25,537` retained
candidates from three independent sources:

```text
MVTec development: 144 images, 2,582 candidates
VisA exposed:      1,200 images, 22,723 candidates
Kodytek exposed:      12 images, 232 candidates
```

Every metric row was predicted by a model that excluded the complete source
dataset. Kodytek runtime inference used an isolated tree containing only normal
and defect images; its masks were used afterward as development labels.

## Architecture Change

V4.2b replaces candidate-versus-baseline-only regression with category-free
within-image list ranking:

- every candidate is paired with every other candidate from the same image;
- directional feature and rank deltas encode the proposed replacement;
- orientation-invariant pair means and list mean/std features encode absolute
  candidate-pool context;
- antisymmetric inference guarantees that reversing a pair reverses its score;
- an override requires both a positive pairwise lower bound and at least
  `0.01` expected-IoU improvement from the independent absolute head;
- otherwise, the original V3 mask is returned unchanged.

No category name, defect name, or official-mask-derived feature enters runtime
selection.

## Final Leave-Dataset-Out Result

| Dataset | V3 Dice | V4.2b Dice | Delta | Oracle | List Spearman | Pair coverage | Search recall | Override rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kodytek exposed | `0.3164` | `0.3164` | `+0.0000` | `0.4951` | `0.6470` | `0.9182` | `1.0000` | `0.0000` |
| MVTec development | `0.5889` | `0.5934` | `+0.0045` | `0.6539` | `0.7667` | `0.3462` | `0.9191` | `0.0208` |
| VisA exposed | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.6160` | `0.9631` | `0.9682` | `0.0000` |

Dataset-macro Dice changes from `0.3549` to `0.3564`, a gain of only
`+0.0015`. This is below the preregistered `+0.01` requirement.

Relative to V4.2, MVTec selected Dice improves from `0.5786` to `0.5934`
(`+0.0148`), while VisA remains unchanged. The dual-head guard is therefore a
real safety improvement, but not a general utility improvement.

## What Worked

- Absolute list context raises MVTec within-image rank Spearman from the first
  V4.2b prototype's `0.7349` to `0.7667`.
- The dual-head guard removes the observed catastrophic tile/wood replacements.
- The final row has no dataset regression and retains the high localization
  recall of V4.2.
- The runtime contract is conservative: only three MVTec masks are overridden;
  VisA and Kodytek fall back entirely to V3.

## What Failed

The ranking signal transfers better than its uncertainty estimate. Pairwise
coverage is only `0.3462` on MVTec and `0.9631` on VisA, outside the required
`0.85-0.95` interval in opposite directions. A single residual calibrated on
the other exposed datasets is therefore not exchangeable across these domains.

The candidate ceiling is still large, especially on VisA (`0.3583` oracle
versus `0.1593` selected), but the current model cannot identify safe overrides
without either allowing harmful MVTec replacements or abstaining on VisA.

Kodytek is a legitimate third source but is too small and too homogeneous to
calibrate cross-domain tail risk by itself. Further threshold tuning on these
same three sources would be development overfitting, not evidence of generality.

## Gate Result

```text
expected_iou_spearman = pass
expected_iou_mae = pass
pairwise_conformal_coverage = fail
search_region_recall = pass
dataset_non_regression = pass
utility_gain = fail
```

## Reproducibility

```text
run_id: 20260810T102815Z-670a15ce
elapsed: 1036.174514 seconds
peak RSS: 2,681,552,896 bytes
model SHA-256: b5860d003f44bb093b6074dcb6fcd486081e58a9aed94d12292738f5621904fc
metrics SHA-256: 1d3722c445a352f095ed7925df9b1cddff58b0981787eb2287471c62607a0bc7
report SHA-256: 841f9b57d6458872c274c773bc9701b641e53c914dc78f6c7088dbb9a18ea39d
calibration manifest SHA-256: 6d9a35e3512297c7020a805e6b3f33e43c09dd7b3b7f177b92f6058f1b5089c9
Kodytek metadata SHA-256: f6e964ba95eb80ee2b1b27c9ca32973b5a8a71e14a20e4cf4b579053c8387af7
tests: 280 passed in 46.83 seconds
```

## Next Gate

Do not tune another threshold on this cohort. The next learned-selector attempt
requires broader exposed calibration support, with at least two additional
independent labeled industrial datasets. Until then, V3 remains the honest
default and V4.2b remains a research ablation.
