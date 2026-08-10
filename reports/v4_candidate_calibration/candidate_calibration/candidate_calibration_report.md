# V4.2 Cross-Dataset Candidate Calibration Report

Every metric row is predicted by a model that excluded the complete source dataset. Official masks label retained candidates only after generation.

- Development candidate model: `/home/p76147019/ImgGen/v2/outputs/v4_candidate_calibration/auto_masks/selector/cross_dataset_candidate_calibrator.joblib`
- Promotion passed: `False`

| Dataset | Images | Baseline Dice | Selected Dice | Delta | Oracle Dice | Precision | Recall | Area Ratio | Override Rate | Spearman | MAE | Coverage | Search Recall | Search Area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mvtec_development | 144 | `0.5889` | `0.5786` | `-0.0103` | `0.6539` | `0.5925` | `0.6895` | `1.0478` | `0.0278` | `0.7481` | `0.0835` | `0.9194` | `0.9191` | `0.4634` |
| visa_exposed | 1200 | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.1420` | `0.7487` | `5.2475` | `0.0000` | `0.5650` | `0.0854` | `0.7601` | `0.9682` | `0.4916` |

## Promotion Gates

- expected_iou_spearman: `True`
- expected_iou_mae: `True`
- conformal_coverage: `False`
- search_region_recall: `True`
- dataset_non_regression: `True`
- utility_gain: `False`

This is exposed multi-dataset development evidence, not a locked-generalization claim.
