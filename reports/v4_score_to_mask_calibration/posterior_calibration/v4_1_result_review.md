# V4.1 Cross-Dataset Score-To-Mask Review

## Decision

V4.1 is implemented and reproducible, but it is **not promoted** into the
default auto-mask architecture. The posterior path remains opt-in and its
configuration keeps `posterior_override_eval: false`.

## Contract

The sprint uses 1,344 exposed development images:

- 144 MVTec bottle/zipper/wood/metal-nut/tile images;
- 1,200 previously locked, now exposed VisA images.

Every reported prediction comes from a model that excluded the complete source
dataset. No category name or defect name is a model feature. The model uses:

- robust and within-image rank normalization of fused evidence;
- local rank context and gradient magnitude;
- the Qwen region as a soft spatial prior;
- stable normal-reference boundary evidence;
- adaptive Otsu thresholding and seeded compact components;
- a broad-mask area-ratio guard against unnecessary replacement.

## Result

| Row | Macro Dice | Macro Precision | Macro Recall | Aggregate Area Ratio | MVTec Dice | VisA Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 input masks | `0.3741` | `0.3647` | `0.7242` | `3.1548` | `0.5889` | `0.1593` |
| Forced posterior, before guard | `0.3247` | `0.3655` | `0.6022` | `1.5135` | `0.4596` | `0.1898` |
| Best area-compliant guard (`4x`) | `0.3506` | `0.3805` | `0.6413` | `1.9240` | `0.5244` | `0.1769` |
| Non-regressing guard (`8x`) | `0.3674` | `0.3720` | not selected | `2.7630` | `0.5702` | `0.1645` |

The area-compliant row improves VisA Dice by `+0.0176` and reduces the macro
area ratio from `3.1548` to `1.9240`, but MVTec Dice falls by `-0.0646`. The
`8x` guard limits MVTec regression to `-0.0187`, but fails the area target.
There is no tested operating point that satisfies all three requirements:

```text
aggregate area ratio <= 2.0
macro precision >= 0.30
per-dataset Dice regression <= 0.02
```

## Interpretation

The fused field carries useful anomaly ranking, and stable-normal boundary
evidence improves precision. However, field calibration cannot determine when a
broad selector mask is a legitimate large defect versus over-segmentation. A
single area-ratio guard therefore trades one dataset against the other.

The next sprint must move from blanket pixel-mask replacement to candidate-level
arbitration. V4.2 should predict real candidate precision/recall/IoU using
normal-boundary burden, posterior support, spatial containment, selector
disagreement, and localization confidence. The same leave-dataset-out and
non-regression gates remain mandatory.

## Artifacts

```text
configs/v4_score_to_mask_calibration.yaml
outputs/v4_score_to_mask_calibration/auto_masks/posterior/score_to_mask.joblib
reports/v4_score_to_mask_calibration/posterior_calibration/posterior_calibration_manifest.json
reports/v4_score_to_mask_calibration/posterior_calibration/score_to_mask_calibration_report.md
reports/v4_score_to_mask_calibration/posterior_calibration/leave_dataset_out_metrics.json
```
