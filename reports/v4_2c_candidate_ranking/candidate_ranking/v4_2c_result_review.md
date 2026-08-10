# V4.2c Broader Cross-Domain Risk Review

## Decision

V4.2c is complete, but it does not pass promotion. Keep V3 as the production
fallback and retain V4.2c as exposed-development evidence.

The sprint added two independent industrial sources, BTAD and KSDD2, through a
pinned and governed preparation path. The resulting five-source study contains
`1,476` images and `27,816` retained candidates. Every reported fold excludes
the complete evaluated source, and the selector sees no category or defect
name.

## Result

| Dataset | Baseline Dice | Selected Dice | Delta | Oracle | Pair Coverage | Search Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTAD exposed | `0.2209` | `0.2209` | `+0.0000` | `0.3218` | `0.9393` | `0.6315` |
| Kodytek exposed | `0.3164` | `0.3164` | `+0.0000` | `0.4951` | `0.9091` | `1.0000` |
| KSDD2 exposed | `0.5274` | `0.5274` | `+0.0000` | `0.6144` | `0.8504` | `0.9607` |
| MVTec development | `0.5889` | `0.5889` | `+0.0000` | `0.6539` | `0.5747` | `0.9191` |
| VisA exposed | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.9585` | `0.9682` |

Dataset-macro Dice remains `0.3626`; the override rate is zero for all five
sources. The source-balanced 90% pair residual is approximately `0.14-0.17`,
which is larger than every actionable predicted gain. Pair coverage still
fails on MVTec, despite this complete abstention.

## Interpretation

Broader data did not rescue the global all-pairs residual contract. The ranker
remains informative (dataset-macro within-image Spearman `0.7455`), but one
additive uncertainty radius is both too conservative to act and insufficiently
exchangeable to cover MVTec. The correct follow-up is decision-conditioned
calibration, not a looser post-hoc residual threshold.

BTAD exposes a separate localization failure. Its aggregate search recall is
only `0.6315`, driven by fragmented defects in `btad_01`. Selection changes
cannot repair candidates whose search region misses the anomaly.

## Gates

```text
expected_iou_spearman = pass
expected_iou_mae = pass
pairwise_conformal_coverage = fail
search_region_recall = fail
dataset_non_regression = pass
utility_gain = fail
```

## Reproducibility

```text
run_id: 20260810T113128Z-fcc300d0
elapsed: 784.301004 seconds
peak RSS: 2,903,552,000 bytes
model SHA-256: 3243c14ea85e78796ad6b90810d00ce4b7af02ae59794125e9215841603cadb6
metrics SHA-256: 7a887e1d9857ec90008ebb5542aca9af88fbd6036224825653a89a609c2e6428
report SHA-256: 26b524003251e74890ab48932f4741c0540471faafbae5188b018134a5dad603
manifest SHA-256: 6ae1b30da1852ceda406fcb6751b46491446bab37b8db15265cc93352b656773
```
