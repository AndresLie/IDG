# V4.1 Score-To-Mask Calibration Report

The posterior is trained only on exposed development labels. Every reported row is predicted by a model that excluded its entire dataset.

- Frozen model: `/home/p76147019/ImgGen/v2/outputs/v4_score_to_mask_calibration/auto_masks/posterior/score_to_mask.joblib`
- Development images: `1344`
- Selected threshold: `0.020`
- Broad-mask override ratio: `4.000`
- Selected row passes all promotion constraints: `False`

| Dataset | Images | V3 Dice | V4 Dice | Delta | Precision | Recall | Predicted/True Area | Override Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mvtec_development | 144 | `0.5889` | `0.5244` | `-0.0646` | `0.5904` | `0.6219` | `0.7098` | `0.1181` |
| visa_exposed | 1200 | `0.1593` | `0.1769` | `+0.0176` | `0.1705` | `0.6607` | `3.1383` | `0.2425` |

This is development evidence, not a new locked-generalization claim.
