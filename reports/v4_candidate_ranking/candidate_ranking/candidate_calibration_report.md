# V4.2 Cross-Dataset Candidate Calibration Report

Every metric row is predicted by a model that excluded the complete source dataset. Official masks label retained candidates only after generation.

- Development candidate model: `/home/p76147019/ImgGen/v2/outputs/v4_candidate_ranking/auto_masks/selector/cross_dataset_candidate_ranker.joblib`
- Selection strategy: `all_pairs_list_rank`
- Minimum absolute-head IoU gain: `0.01`
- Coverage gate metric: `pairwise_conformal_coverage`
- Promotion passed: `False`

| Dataset | Images | Baseline Dice | Selected Dice | Delta | Oracle Dice | Regret | Precision | Recall | Area Ratio | Override Rate | IoU Spearman | List Spearman | MAE | IoU Coverage | Pair Coverage | Search Recall | Search Area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kodytek_exposed | 12 | `0.3164` | `0.3164` | `+0.0000` | `0.4951` | `0.1786` | `0.2367` | `0.8559` | `3.2069` | `0.0000` | `0.5301` | `0.6470` | `0.1080` | `0.8836` | `0.9182` | `1.0000` | `0.3342` |
| mvtec_development | 144 | `0.5889` | `0.5934` | `+0.0045` | `0.6539` | `0.0606` | `0.5921` | `0.7013` | `1.0555` | `0.0208` | `0.7635` | `0.7667` | `0.0993` | `0.9423` | `0.3462` | `0.9191` | `0.4634` |
| visa_exposed | 1200 | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.1990` | `0.1420` | `0.7487` | `5.2475` | `0.0000` | `0.5396` | `0.6160` | `0.1035` | `0.6915` | `0.9631` | `0.9682` | `0.4916` |

## Promotion Gates

- expected_iou_spearman: `True`
- expected_iou_mae: `True`
- conformal_coverage: `False`
- search_region_recall: `True`
- dataset_non_regression: `True`
- utility_gain: `False`

This is exposed multi-dataset development evidence, not a locked-generalization claim.
