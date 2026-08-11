# V4.2f Bounded Uncertainty-Contour Result Review

## Decision

V4.2f is a useful proposal mechanism but **does not pass promotion**. Keep the
implementation and evidence, leave it disabled in production, and retain V4.2d
as the deployment fallback.

The category-free contour family is more efficient than V4.2e: it recovers
almost the same oracle ceiling with a bounded pool and materially better
selection/calibration behavior. It nevertheless misses the preregistered VisA
endpoint and narrowly exceeds the selected-Dice regression guard.

## Contract

The experiment was preregistered in
`research_protocols/v4_2f_uncertainty_contours.yaml` before execution. It keeps
the complete V4.2d candidate pool, disables V4.2e component unions, and adds at
most three category-free candidates per image. Each candidate is a seeded
within-component contour derived from fused-evidence and posterior mid-ranks,
with source disagreement suppressing uncertain boundaries.

Official masks were used only to label and evaluate exposed-development
candidates. They do not enter proposal construction.

## Five-Source Result

| Dataset | V4.2d oracle | V4.2f oracle | Oracle gain | V4.2d selected | V4.2f selected | Selected gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTAD | `0.3218` | `0.3488` | `+0.0271` | `0.2305` | `0.2344` | `+0.0039` |
| Kodytek | `0.4951` | `0.5139` | `+0.0189` | `0.3625` | `0.3164` | `-0.0460` |
| KSDD2 | `0.6144` | `0.6297` | `+0.0153` | `0.5274` | `0.5274` | `+0.0000` |
| MVTec development | `0.6539` | `0.6576` | `+0.0037` | `0.5811` | `0.5948` | `+0.0136` |
| VisA exposed | `0.3583` | `0.3601` | `+0.0018` | `0.1593` | `0.1593` | `+0.0000` |
| **Dataset macro** | **`0.4887`** | **`0.5020`** | **`+0.0134`** | **`0.3722`** | **`0.3664`** | **`-0.0057`** |

V4.2f adds exactly `4,428` candidates across `1,476` images, reaching `32,244`
total candidates. Pool growth over V4.2d is `15.92%`, below the preregistered
`20%` maximum. The new contours are strict oracle winners on `125` images:
`84` VisA, `16` BTAD, `12` MVTec, `11` KSDD2, and `2` Kodytek.

Only `12` new contours are actually selected. Seven are low-Dice VisA masks,
while the three selected KSDD2 contours are strong. The larger deployment
regression is not simply caused by those 12 masks: refitting on the enlarged
pool also changes fold-level thresholds and suppresses all Kodytek overrides,
returning that source to its weaker baseline.

## Gate Audit

| Gate | Target | Result | Pass |
| --- | ---: | ---: | :---: |
| VisA oracle gain | `>= +0.0300` | `+0.0018` | No |
| Dataset-macro oracle gain | `>= +0.0100` | `+0.0134` | Yes |
| Per-source oracle regression | `<= 0` | no regressions | Yes |
| Candidate-pool growth | `<= 20%` | `15.92%` | Yes |
| Macro selected-Dice regression | `<= 0.0050` | `0.00571` | No |
| Expected-IoU MAE | `<= 0.12` per source | Kodytek `0.1264` | No |

The macro MAE is acceptable (`0.0908`), but the production evaluator applies
the threshold to every held-out source, so Kodytek correctly fails the stricter
gate. Search-region recall and decision-calibration gates also remain failed
for pre-existing reasons; V4.2f was not designed to change localization.

## Comparison With V4.2e

| Metric | V4.2e | V4.2f | Difference |
| --- | ---: | ---: | ---: |
| Candidates | `41,638` | `32,244` | `-9,394` (`-22.6%`) |
| Macro oracle Dice | `0.5029` | `0.5020` | `-0.0008` |
| Macro selected Dice | `0.3595` | `0.3664` | `+0.0069` |
| Macro expected-IoU MAE | `0.1021` | `0.0908` | `-0.0113` |
| Initial elapsed time | `3,587.5 s` | `2,502.9 s` | `-30.2%` |

This is the main positive result: uncertainty contours are a substantially
better bounded proposal family than exhaustive component unions. The remaining
failure is that they improve easy compact defects more than the intended VisA
domain-shift cases.

## Professional Interpretation

V4.2f confirms that disagreement-aware contouring can improve boundary
precision without category rules and without a combinatorial candidate pool.
It does **not** establish a general mask-quality improvement because the primary
cross-domain endpoint is effectively flat and selection remains unstable.

Do not add more fixed contour quantiles. The fact that all three candidates are
emitted for every image and only `125/1,476` become oracle winners shows the
next bottleneck is **candidate admission and selector stability**, not another
uniform threshold sweep. A future study should predict whether a sample needs
a contour and spend the candidate budget only where disagreement is localized;
that study requires a new preregistration and must preserve the V4.2d fallback.

## Reproducibility

- Run status: `succeeded`
- Elapsed time: `2,502.87 s`
- Peak process RSS: `3,553,366,016` bytes
- Candidate cache SHA-256:
  `21184761905958b5a3a572b51391dafaa029e306358fc9c34e6aac816756598b`
- Model SHA-256:
  `c5a6c9de23a7720c79ec4740bdfab70aa897b33203eadd966f78ce48f5d0a2ad`
- Metrics SHA-256:
  `06717691fc2ec0bd94866fbbbf8b17e01c656f5661a5cc9b8532ca6fd3b0e4c8`
- Automated tests: `292 passed`

A cache-hit rerun completed in `2,072.77 s` and reproduced the candidate cache,
model, metrics JSON, generated Markdown report, architecture fingerprint,
package fingerprint, model fingerprint, config fingerprints, and split
fingerprint exactly. It is `17.18%` faster than the initial `2,502.87 s` run.
The promotion decision remains unchanged.
