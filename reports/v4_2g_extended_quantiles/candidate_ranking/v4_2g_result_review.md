# V4.2g Extended Quantile Family Review

## Decision

The proposal hypothesis is confirmed; promotion fails. Keep the extended quantile
family default-off and preserve V4.2f/V4.2d as the production fallback. Do not run
another selection or threshold sprint on these five exposed sources.

Executed against `research_protocols/v4_2g_extended_quantiles.yaml`, sealed before
execution. Only `candidate_quantiles` changed
(`[0.85,0.90,0.95,0.975]` -> `[0.85,0.90,0.95,0.975,0.99,0.995,0.999]`).

## Result

| Source | Oracle V4.2f | Oracle V4.2g | Δ oracle | Selected V4.2f | Selected V4.2g | Δ selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| btad_exposed | `0.3488` | `0.3575` | `+0.0087` | `0.2344` | `0.2195` | `-0.0148` |
| kodytek_exposed | `0.5139` | `0.5139` | `+0.0000` | `0.3164` | `0.3164` | `+0.0000` |
| ksdd2_exposed | `0.6297` | `0.6501` | `+0.0205` | `0.5274` | `0.5274` | `+0.0000` |
| mvtec_development | `0.6576` | `0.6584` | `+0.0008` | `0.5948` | `0.5948` | `+0.0000` |
| visa_exposed | `0.3601` | `0.4206` | `+0.0605` | `0.1593` | `0.1593` | `+0.0000` |
| Dataset macro | `0.5020` | `0.5201` | `+0.0181` | `0.3664` | `0.3635` | `-0.0030` |

Candidates `32,244 -> 44,091` (`+36.7%`).

## Gates

```text
per_source_oracle_regression   <= 0 regression   PASS
dataset_macro_oracle_gain      >= +0.0100        PASS (+0.0181)
visa_oracle_gain               >= +0.0300        PASS (+0.0605)
selected_dice_regression       <= 0.0050         PASS (0.0030)
candidate_pool_growth          <= 100%           PASS (+36.7%)
utility_gain (promotion)       >= +0.0100        FAIL (-0.0030)
decision_calibration                             FAIL
expected_iou_spearman                            FAIL (0.5988)
search_region_recall                             FAIL (0.8959 macro; BTAD 0.6315)
```

## What Worked

The truncated quantile family was a real architectural cap. The VisA oracle gate
that failed in V4.2f at `+0.0018` passes here at `+0.0605`, the largest
single-source oracle gain in the project. All five sources are non-regressive,
confirming strict additivity.

## What Failed

The selector converts none of the gain. VisA, KSDD2, Kodytek and MVTec selected
Dice are bit-identical to V4.2f because the decision policy fails closed rather
than acting on candidates it cannot validate. Remaining oracle-minus-selected
headroom is macro `0.1566` and VisA `0.2613`. Published area ratio is unchanged
(macro `2.639`, VisA `5.247`) precisely because the tighter candidates are never
selected. Adding extreme-tight candidates also lowered absolute-IoU rank
correlation (`expected_iou_spearman` `0.5988`).

## Next

Do not iterate selection a sixth time. Correct-quality masks provably exist in the
pool (VisA oracle `0.4206` against a published `0.1593`), and a fixed p99.5
threshold with no selector already beat the deployed pipeline on VisA
(`0.183` vs `0.153`). The indicated change is to remove the learned selector from
the critical path for the segmentation claim and publish a dense, area-calibrated
thresholded map, retaining the selector and the Qwen region only for generation,
where a single region is genuinely required.
