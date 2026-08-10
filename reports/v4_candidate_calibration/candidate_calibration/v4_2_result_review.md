# V4.2 Candidate Calibration Result Review

## Decision

V4.2 is implemented and evaluated, but it is **not promoted** into the runtime
auto-mask path. The candidate calibrator and evidence-augmented localization
remain default-off.

## Experiment Contract

The run reconstructed the same category-agnostic candidate family for 144
MVTec development images and 1,200 exposed VisA images. Candidate generation
used only retained fused evidence, uncertainty, normal references, Qwen regions,
and the default V3 selected mask. Official masks were opened afterward to label
candidates.

Every reported prediction was produced by a model that excluded the complete
source dataset. The candidate model used 22 category-free features covering:

- fused-evidence support and contrast;
- uncertainty and candidate geometry;
- calibrated-posterior support;
- stable normal-boundary burden;
- semantic-prior and augmented-region containment;
- candidate consensus and overlap with the V3 fallback;
- category-independent candidate-family indicators.

## Result

| Dataset | V3 Dice | V4.2 Dice | Delta | Oracle Dice | Spearman | MAE | Coverage | Search Recall | Search Area |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MVTec development | `0.5889` | `0.5786` | `-0.0103` | `0.6539` | `0.7481` | `0.0835` | `0.9194` | `0.9191` | `0.4634` |
| VisA exposed | `0.1593` | `0.1593` | `+0.0000` | `0.3583` | `0.5650` | `0.0854` | `0.7601` | `0.9682` | `0.4916` |

The reconstructed pool substantially raises the available ceiling, especially
on VisA. The model also predicts absolute candidate IoU reasonably well across
datasets. However, those global calibration metrics do not translate into
useful within-image choices. The paired gain lower bound applies no safe VisA
overrides and changes only `2.78%` of MVTec masks.

Relaxing the guard is not a valid fix. A zero threshold on raw predicted gain
improves VisA by about `+0.0119` Dice but reduces MVTec by `-0.0439`. Ridge-based
pairwise extrapolation also regresses the held-out dataset. This is a genuine
cross-dataset ranking failure, not an overly conservative scalar threshold.

## Localization Finding

The evidence-proposal envelope raises search-region recall above `0.90` on both
datasets, but covers roughly `46-49%` of each image. It is useful as a soft
evidence prior, not yet as a precise localization claim. Future work must report
coverage risk together with recall so full-image expansion cannot game the
metric.

## Next Slice

V4.2b should train directly on within-image candidate pairs or lists and add a
third exposed dataset before another cross-dataset promotion attempt. The gate
must require:

```text
positive dataset-macro Dice gain >= 0.01
every-dataset Dice regression <= 0.02
every-dataset conformal coverage in [0.85, 0.95]
every-dataset search recall >= 0.85 with search area reported
```

Until those conditions pass, the production architecture remains V3 and no new
locked evaluation is justified.
