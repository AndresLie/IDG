# V4.2d Decision-Conditioned Selection Review

## Decision

V4.2d is the strongest five-source candidate-selection result so far, but it
misses the preregistered promotion threshold. Keep the implementation and
research artifacts, leave it default-off, and do not retune the threshold grid
on these five exposed sources.

The selector now calibrates the final list-score override margin through an
inner source-jackknife. Each outer leave-dataset-out model learns its decision
threshold only from predictions that excluded each remaining source in turn.
When no threshold satisfies the fixed inner utility and regression constraints,
the policy returns the exact V3 mask.

## Result

| Dataset | V3 Dice | V4.2d Dice | Delta | Oracle | Override Rate | Learned Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTAD exposed | `0.2209` | `0.2305` | `+0.0096` | `0.3218` | `0.4000` | `0.005` |
| Kodytek exposed | `0.3164` | `0.3625` | `+0.0460` | `0.4951` | `0.5833` | `0.005` |
| KSDD2 exposed | `0.5274` | `0.5274` | `+0.0000` | `0.6144` | `0.0000` | disabled |
| MVTec development | `0.5889` | `0.5811` | `-0.0078` | `0.6539` | `0.5069` | `0.005` |
| VisA exposed | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.0000` | disabled |

Dataset-macro Dice improves from `0.3626` to `0.3722`, a gain of `+0.00957`.
This is substantially better than V4.2c (`+0.0000`) and V4.2b's shared-source
gain (`+0.0015`), but it is `0.00043` below the preregistered `+0.0100` gate.
Selection regret falls from `0.1261` to `0.1165`; macro precision rises from
`0.3228` to `0.3396` while macro recall decreases from `0.6969` to `0.6852`.

On the three V4.2b sources, V4.2d changes macro selected Dice from `0.3564` to
`0.3676`. This supports the decision-conditioned contract, but it is exposed
development evidence rather than a locked generalization result.

## What Worked

- The nested policy converts transferable rank information into useful actions
  on BTAD and Kodytek.
- The MVTec regression is limited to `0.0078`, within the allowed `0.02`.
- KSDD2 and VisA fail closed instead of accepting an unsupported threshold.
- The absolute IoU head remains well behaved: all five folds pass Spearman and
  MAE gates.
- The protocol, all five source manifests, and the posterior model are hashed
  in the finalized experiment manifest.

## What Failed

- The primary utility gain is just below threshold and must be recorded as a
  failure, not rounded to a pass.
- Two outer folds cannot find an eligible inner threshold, so the decision
  calibration gate fails.
- MVTec still loses `0.0078`; ranking errors remain concentrated in the strong
  legacy domain.
- BTAD search recall remains `0.6315`. The decision selector cannot repair this
  independent proposal/localization bottleneck.
- Runtime rises to `2,154.15` seconds because every outer fold contains an
  inner source-jackknife. Candidate-row caching is needed before another model
  iteration.

## Gates

```text
expected_iou_spearman = pass
expected_iou_mae = pass
decision_calibration = fail
search_region_recall = fail
dataset_non_regression = pass
utility_gain = fail (0.00957 < 0.01000)
```

## Next Sprint

Do not tune another selector threshold on this cohort. The next exposed sprint
should address the orthogonal localization failure with a category-free,
evidence-mass region proposal for fragmented or distributed defects. It should
also cache reconstructed candidates so nested evaluations do not repeat image
loading and feature extraction.

Required gates for that sprint:

```text
BTAD search recall >= 0.85
every-source search recall >= 0.85
dataset-macro search area <= 0.60
V4.2d selection Dice does not regress by more than 0.005 macro
```

## Reproducibility

```text
run_id: 20260810T115530Z-80d0c716
elapsed: 2154.147577 seconds
peak RSS: 2,949,005,312 bytes
model SHA-256: c8dc779ef496a27de0ccd619c458bf303a5092f10f953d674a0281a5627c652f
metrics SHA-256: 98d13657abdaa23bdcc9745b5b131db05755a26e0dfd475229a19649410e95af
report SHA-256: 678f7bf9ee5a349e99f97ed35836c8089c0339db13ae28847d4ca8664cf376a1
manifest SHA-256: 139df58cbdde3d758a0764493bd47f72804c485ec7a11e3ea27abb598ff9514c
tests: 288 passed in 54.89 seconds
```
