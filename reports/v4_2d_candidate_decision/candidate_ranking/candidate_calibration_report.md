# V4.2 Cross-Dataset Candidate Calibration Report

Every metric row is predicted by a model that excluded the complete source dataset. Official masks label retained candidates only after generation.

- Development candidate model: `/home/p76147019/ImgGen/v2/outputs/v4_2d_candidate_decision/auto_masks/selector/cross_dataset_candidate_ranker.joblib`
- Selection strategy: `source_calibrated_list_rank`
- Risk calibration: `source_jackknife_equal_weight`
- Minimum absolute-head IoU gain: `0.01`
- Coverage gate metric: `nested_source_decision_calibration`
- Promotion passed: `False`

| Dataset | Images | Baseline Dice | Selected Dice | Delta | Oracle Dice | Regret | Precision | Recall | Area Ratio | Override Rate | Decision Threshold | Inner Gain | Inner Worst | Gain Residual | IoU Spearman | List Spearman | MAE | IoU Coverage | Pair Coverage | Search Recall | Search Area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| btad_exposed | 60 | `0.2209` | `0.2305` | `+0.0096` | `0.3218` | `0.0912` | `0.2402` | `0.3822` | `1.8926` | `0.4000` | `0.005` | `0.0187` | `-0.0086` | `0.1674` | `0.4025` | `0.7364` | `0.0820` | `0.8591` | `0.9393` | `0.6315` | `0.4662` |
| kodytek_exposed | 12 | `0.3164` | `0.3625` | `+0.0460` | `0.4951` | `0.1326` | `0.2895` | `0.8114` | `2.8634` | `0.5833` | `0.005` | `0.0119` | `0.0053` | `0.1370` | `0.4871` | `0.6534` | `0.1154` | `0.8922` | `0.9091` | `1.0000` | `0.3342` |
| ksdd2_exposed | 60 | `0.5274` | `0.5274` | `+0.0000` | `0.6144` | `0.0870` | `0.4341` | `0.7839` | `1.7391` | `0.0000` | `0.05` | `0.0000` | `0.0000` | `0.1650` | `0.8387` | `0.8614` | `0.0749` | `0.9823` | `0.8504` | `0.9607` | `0.4863` |
| mvtec_development | 144 | `0.5889` | `0.5811` | `-0.0078` | `0.6539` | `0.0728` | `0.5922` | `0.6999` | `0.9542` | `0.5069` | `0.005` | `0.0210` | `0.0024` | `0.1645` | `0.8515` | `0.8575` | `0.0711` | `0.9845` | `0.5747` | `0.9191` | `0.4634` |
| visa_exposed | 1200 | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.1990` | `0.1420` | `0.7487` | `5.2475` | `0.0000` | `0.05` | `0.0000` | `0.0000` | `0.1690` | `0.5287` | `0.6188` | `0.0851` | `0.8363` | `0.9585` | `0.9682` | `0.4916` |

## Promotion Gates

- expected_iou_spearman: `True`
- expected_iou_mae: `True`
- decision_calibration: `False`
- search_region_recall: `False`
- dataset_non_regression: `True`
- utility_gain: `False`

This is exposed multi-dataset development evidence, not a locked-generalization claim.
