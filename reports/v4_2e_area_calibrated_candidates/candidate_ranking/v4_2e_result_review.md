# V4.2e Area-Calibrated Candidate Review

## Verdict

V4.2e validates one architectural hypothesis but fails promotion. Category-free
component unions raise candidate oracle capacity across every source, yet the
gain is too small on VisA and the larger pool destabilizes leave-dataset-out
selection. The feature must remain disabled.

## Comparison To V4.2d

| Dataset | Oracle Before | Oracle Current | Oracle Delta | Selected Before | Selected Current | Selected Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTAD exposed | `0.3218` | `0.3353` | `+0.0136` | `0.2305` | `0.2349` | `+0.0043` |
| Kodytek exposed | `0.4951` | `0.5063` | `+0.0112` | `0.3625` | `0.3252` | `-0.0373` |
| KSDD2 exposed | `0.6144` | `0.6315` | `+0.0171` | `0.5274` | `0.5274` | `+0.0000` |
| MVTec development | `0.6539` | `0.6695` | `+0.0156` | `0.5811` | `0.5509` | `-0.0303` |
| VisA exposed | `0.3583` | `0.3717` | `+0.0134` | `0.1593` | `0.1593` | `+0.0000` |
| **Dataset macro** | **`0.4887`** | **`0.5029`** | **`+0.0142`** | **`0.3722`** | **`0.3595`** | **`-0.0126`** |

## Gate Review

| Gate | Target | Result | Status |
| --- | ---: | ---: | --- |
| Macro oracle gain | `>= +0.010` | `+0.0142` | Pass |
| VisA oracle gain | `>= +0.030` | `+0.0134` | Fail |
| Per-source oracle regression | `<= 0.000` | `0.000` worst | Pass |
| Macro selected regression | `<= 0.005` | `0.0126` | Fail |

The strict-superset design behaves correctly: `263/1476` samples use a new
oracle mode and no source loses oracle Dice. The deployment path does not. The
expanded candidate context changes list-ranking features and increases MVTec
override rate from `0.5069` to `0.6528`, producing smaller but less complete
masks. VisA still fails closed, so its improved candidates never reach output.

## Efficiency And Reproducibility

- Candidate count: `27,816 -> 41,638` (`+49.7%`).
- First governed runtime: `3,587.50 s` versus V4.2d `2,154.15 s`.
- Reconstructed-row cache: `5,321,898` bytes, SHA-256
  `671e78bf2c221bec7b737091a7dde246c7f5ce6ce5fa29216875440e26d3bcb2`.
- Cache fingerprint:
  `f7c39ca22207c38fc37bcd72db7fe2d079aabbb1854faa5e94381d2e341f3da2`.
- Test suite: `290 passed` under Python `3.10.20`.

The cache removes repeated image loading, posterior inference, and candidate
feature reconstruction. It does not remove the quadratic all-pairs ranking
cost, which now dominates the run.

## Next Improvement

Do not add another component-subset grid or tune the decision threshold on this
cohort. The next study should generate a small number of precision-oriented
contours inside each broad component using source-consensus cores,
uncertainty-aware geodesic growth, and a hard per-image candidate budget. Its
primary target should remain VisA oracle Dice, with pool growth capped at `20%`
and the V4.2d selector fallback preserved.
