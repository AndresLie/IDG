# Pseudo-Label Student Review

## Decision

Negative result. A segmentation student trained on the pipeline's pseudo-masks
does **not** beat those masks, in-distribution or across categories. The
published mask stays the pseudo-label. Keep the script as research
infrastructure; do not promote a student stage.

## Motivation

Box-supervised pipelines learn defect segmentation from noisy SAM masks and
obtain a model whose output is cleaner than its own training labels. This sprint
tested the same idea without the bounding box: train only on our pseudo-masks and
score the student against official masks.

## Protocol

```text
model              ResNet18-encoder U-Net (same family as the R5 evaluator)
training signal    pseudo-masks only (eval_tight); official masks never trained on
scoring            official masks, held-out samples only
steps              300, batch 8, 256 px, AdamW 3e-4
seeds              0, 1, 2 (deterministic torch algorithms, :4096:8 workspace)
pool               144 MVTec development images, 5 categories
```

Reproduce:

```bash
python scripts/evaluate_pseudo_label_student.py \
  --metadata outputs/as1b_calib_widen/auto_masks/qwen/metadata.jsonl \
  --data-root data/mvtec_ad --mode image \
  --out reports/pseudo_label_student/result_image_mode.json
```

## Result

| Split | Pseudo-label Dice | Student Dice | Delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Leave-category-out (unseen category) | `0.6075` | `0.2122` | `-0.3953` | `[-0.5199, -0.2543]` |
| Stratified image folds (in-distribution) | `0.5890` | `0.4519` | `-0.1371` | `[-0.1898, -0.0843]` |

Per split, in-distribution mode: fold_a `0.6108 -> 0.4210`, fold_b
`0.5671 -> 0.4828`. Leave-category-out mode: bottle `0.5295 -> 0.2245`,
metal_nut `0.7106 -> 0.2619`, tile `0.7741 -> 0.1564`, wood `0.6653 -> 0.2237`,
zipper `0.3581 -> 0.1947`.

Both deltas are negative with intervals excluding zero.

## Interpretation

The student neither denoises nor transfers. Per-seed spread on wood
(`0.1217, 0.4096, 0.1398`) and zipper (`0.2824, 0.2200, 0.0817`) shows the fit is
also unstable at this budget.

The leave-category-out collapse is consistent with every other cross-domain
result in this project: the pipeline's weakness is generalization, not
within-domain fitting.

## Scope limit

This used 144 images, 72 per training fold, 300 steps at 256 px. That is a small
budget for a segmentation network. The result therefore refutes
"train a student on the development pool" and **not** the denoising idea at
publication scale. Any future attempt needs a substantially larger pseudo-label
corpus and a preregistered gate.
