# V4.2 Reproducibility Rerun Comparison

## Scope

The committed V4.2 candidate-calibration command was rerun against the same
cached MVTec-development and exposed-VisA candidate cohorts:

```bash
.conda/envs/iadgen-v2/bin/python -m iadgen_v2.cli \
  auto-mask-train-candidate-calibrator \
  --config configs/v4_candidate_calibration.yaml
```

This is a reproducibility check. It does not add data, tune thresholds, or
change the V4.2 promotion decision.

| Run | Run ID | Status | Elapsed | Peak RSS |
| --- | --- | --- | ---: | ---: |
| Previous sealed run | `20260810T085933Z-3263f819` | succeeded | `435.549419 s` | `1,040,277,504 B` |
| Current committed rerun | `20260810T091345Z-3524261a` | succeeded | `367.093792 s` | `1,038,925,824 B` |
| Difference | - | - | `-68.455627 s` (`-15.7171%`) | `-1,351,680 B` (`-0.1299%`) |

The runtime difference is observational only. No controlled performance change
was made, so it must not be interpreted as an optimization claim.

## Artifact Reproducibility

All substantive V4.2 artifacts reproduced byte-for-byte.

| Artifact | Previous SHA-256 | Current SHA-256 | Match |
| --- | --- | --- | --- |
| Candidate model | `477b838048681144465f69fc569a2b15db69e43bde2705bf97ae685e3be88451` | `477b838048681144465f69fc569a2b15db69e43bde2705bf97ae685e3be88451` | yes |
| Full metric matrix | `f71cbb576c109745b3daf2e1c77180b6bb6eb2ff9e750e8917e7c964411924a4` | `f71cbb576c109745b3daf2e1c77180b6bb6eb2ff9e750e8917e7c964411924a4` | yes |
| Markdown report | `f9677cba9d788e78688f90e9c39043fb733317082e9bf816a8918c6c93f556a4` | `f9677cba9d788e78688f90e9c39043fb733317082e9bf816a8918c6c93f556a4` | yes |
| Calibration manifest | `6cb9e25e47fc695f40ee68d5f56efd99c6fff24e9db8f13ca87688a7fdc2670c` | `6cb9e25e47fc695f40ee68d5f56efd99c6fff24e9db8f13ca87688a7fdc2670c` | yes |

The current experiment manifest also records the finalized calibration-manifest
hash above. The prior experiment manifest captured an earlier workspace
fingerprint before the V4.2 implementation was committed; the committed rerun
nevertheless reproduces every substantive output exactly.

## Metric Comparison

| Dataset | Previous selected Dice | Current selected Dice | Delta vs previous | Oracle Dice | Spearman | MAE | Coverage | Search recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MVTec development | `0.5786` | `0.5786` | `+0.0000` | `0.6539` | `0.7481` | `0.0835` | `0.9194` | `0.9191` |
| VisA exposed | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.5650` | `0.0854` | `0.7601` | `0.9682` |

Baseline Dice also remains exactly `0.5889` for MVTec and `0.1593` for VisA.
Therefore V4.2 still changes MVTec by `-0.0103` and VisA by `+0.0000`.

## Promotion Decision

The gate vector is unchanged:

```text
expected_iou_spearman = pass
expected_iou_mae = pass
conformal_coverage = fail
search_region_recall = pass
dataset_non_regression = pass
utility_gain = fail
```

V4.2 remains a deterministic, default-off development prototype. The rerun
strengthens confidence in the implementation and reporting, but it does not
improve candidate selection. V4.2b remains the correct next sprint: optimize
within-image pair/list ranking on broader exposed development data while
retaining the V3 selection as a fallback.
