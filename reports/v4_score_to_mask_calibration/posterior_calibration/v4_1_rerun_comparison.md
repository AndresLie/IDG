# V4.1 Reproducibility Comparison

## Scope

The committed V4.1 calibration command was rerun on the identical exposed
development cohorts:

```bash
python -m iadgen_v2.cli auto-mask-train-posterior \
  --config configs/v4_score_to_mask_calibration.yaml
```

The run completed successfully as
`20260810T082529Z-8c20612e` in `353.34` seconds. It used 144 MVTec images and
1,200 exposed VisA images. This is a reproducibility check, not new locked
generalization evidence.

## Exact Comparison

| Artifact | Previous sealed SHA-256 | Current rerun SHA-256 | Result |
| --- | --- | --- | --- |
| Posterior model | `e0645f155e544e92fbd3f551d32dfb11536132b5de0a80b97e1b9e583ba747e5` | `e0645f155e544e92fbd3f551d32dfb11536132b5de0a80b97e1b9e583ba747e5` | Exact |
| Leave-dataset-out metrics | `6ffd5542d010964fb0ad3a406376feeda169eee8efe19d5ab4af1693f3887085` | `6ffd5542d010964fb0ad3a406376feeda169eee8efe19d5ab4af1693f3887085` | Exact |
| Calibration report | `cba3a210fadea6d725c22a276753707945a1c3967f8198f6b66586131a5fe851` | `cba3a210fadea6d725c22a276753707945a1c3967f8198f6b66586131a5fe851` | Exact |
| Calibration manifest | `1800c77e97bd4c6837c51363d8f047074b088353e331ec1c025026df8ec1f078` | `1800c77e97bd4c6837c51363d8f047074b088353e331ec1c025026df8ec1f078` | Exact |

The selected operating point and all reported metrics are unchanged:

| Row | Macro Dice | Macro Precision | Macro Recall | Aggregate Area Ratio | MVTec Dice | VisA Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 input masks | `0.3741` | `0.3647` | `0.7242` | `3.1548` | `0.5889` | `0.1593` |
| V4.1 selected guard (`4x`) | `0.3506` | `0.3805` | `0.6413` | `1.9240` | `0.5244` | `0.1769` |

## Decision

V4.1 is deterministic, but the reproducible result still fails promotion. It
improves VisA Dice by `+0.0176` and controls aggregate mask area, while reducing
MVTec Dice by `-0.0646`. The posterior override therefore remains disabled.

The reproducibility check strengthens confidence that the failure is
architectural rather than random training drift. V4.2 should continue with
candidate-level, multi-dataset selector and localization recalibration instead
of further tuning the global posterior area guard.
