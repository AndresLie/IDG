# VisA Full Runtime Inference Progress

## Scope

This is an execution checkpoint for the frozen
`v3-generic-evidence-rc2-operational-20260726` architecture. It is not an
evaluation result. Official VisA masks remain sealed.

Runtime run:

```text
run_id: 20260726T160703Z-fcd6745c
runtime samples expected: 1,200
runtime samples completed: 878
progress: 73.17%
```

## Resume Validation

The first bounded segment was intentionally interrupted after 55 complete
records. The second command:

- matched the original auto-mask and architecture fingerprints;
- reused run ID `20260726T160703Z-fcd6745c`;
- reported `records_resumed: 55` and `resume_count: 1`;
- continued to 60 records without recomputing the recovered rows;
- wrote a new valid partial checkpoint after a second graceful interruption.

A third bounded command then:

- matched the same auto-mask and architecture fingerprints;
- reused the same run ID;
- reported `records_resumed: 60` and `resume_count: 2`;
- added 55 unique records and stopped at 115;
- preserved all run-local evaluation, training, and uncertainty-mask artifacts;
- produced no duplicate sample identity keys.

A fourth bounded command:

- reused all 115 checkpoint rows under the same fingerprints and run ID;
- incremented `resume_count` from 2 to 3;
- added 44 capsule records;
- stopped cleanly at 159 unique records;
- again left the stable runtime manifest unpublished.

A fifth bounded command:

- reused all 159 checkpoint rows and incremented `resume_count` to 4;
- added the final 41 capsule rows and the first cashew row;
- stopped at 201 unique records with no checkpoint duplicates;
- left one corrupt, uncheckpointed `cashew_bad_001` PNG after interruption
  during a file write; all artifacts for that incomplete sample were removed,
  while all 201 checkpoint rows remained intact.

A sixth bounded command:

- reused all 201 checkpoint rows and incremented `resume_count` to 5;
- added 30 cashew rows;
- stopped at 231 unique records;
- interrupted during Qwen inference, leaving no partial image artifact;
- preserved all checkpoint-local masks and readable retained PNGs.

A seventh bounded command:

- reused all 231 checkpoint rows and incremented `resume_count` to 6;
- added a category-matched 30-row cashew segment;
- stopped at 261 unique records;
- interrupted during proposal construction without corrupting retained images;
- preserved the same frozen fingerprints and sealed reference boundary.

An eighth bounded command:

- reused all 261 checkpoint rows and incremented `resume_count` to 7;
- added another category-matched 30-row cashew segment;
- stopped at 291 unique records;
- interrupted while writing the next uncheckpointed sample overlay;
- removed all 14 artifacts for incomplete `cashew_bad_091`, leaving every
  checkpointed row and retained image intact.

A ninth bounded command:

- reused all 291 checkpoint rows and incremented `resume_count` to 8;
- completed the final nine cashew rows;
- added the first 55 chewing-gum rows;
- stopped at 355 unique records during selector inference;
- left no corrupt retained image or incomplete checkpoint row.

A tenth bounded command:

- reused all 355 checkpoint rows and incremented `resume_count` to 9;
- completed the final 45 chewing-gum rows;
- added the first 15 fryum rows;
- stopped at 415 unique records during Qwen inference;
- preserved all checkpoint artifacts without interruption residue.

An eleventh bounded command:

- reused all 415 checkpoint rows and incremented `resume_count` to 10;
- added the next 48 fryum rows;
- stopped at 463 unique records during evidence normalization;
- preserved all checkpoint artifacts without corrupt or incomplete retained
  images.

A twelfth bounded command:

- reused all 463 checkpoint rows and incremented `resume_count` to 11;
- completed the final 37 fryum rows and added the first six macaroni1 rows;
- stopped at 506 unique records during proposal construction;
- removed the single uncheckpointed fused-evidence image for macaroni1 `006`;
- preserved all durable checkpoint artifacts without corruption.

A thirteenth bounded command:

- reused all 506 checkpoint rows and incremented `resume_count` to 12;
- added the next 34 macaroni1 rows;
- stopped at 540 unique records during proposal construction;
- removed the single uncheckpointed fused-evidence image for macaroni1 `040`;
- preserved every durable checkpoint row and mask-role artifact.

A fourteenth bounded command:

- reused all 540 checkpoint rows and incremented `resume_count` to 13;
- added another 35 macaroni1 rows;
- stopped at 575 unique records during Qwen inference;
- left no uncheckpointed image artifact;
- preserved the frozen fingerprints and all durable checkpoint outputs.

A fifteenth bounded command:

- reused all 575 checkpoint rows and incremented `resume_count` to 14;
- completed the final 25 macaroni1 rows and added the first 11 macaroni2 rows;
- stopped at 611 unique records during proposal construction;
- removed the single uncheckpointed fused-evidence image for macaroni2 `011`;
- preserved all durable masks and the frozen execution identity.

A sixteenth bounded command:

- reused all 611 checkpoint rows and incremented `resume_count` to 15;
- added the next 44 macaroni2 rows;
- stopped at 655 unique records during Qwen inference;
- left no uncheckpointed image artifact;
- preserved all frozen fingerprints and durable mask-role outputs.

A seventeenth bounded command:

- resumed all 655 checkpoint rows and incremented `resume_count` to 16;
- ran inside a restricted sandbox without GPU access and added zero rows;
- was terminated after the lack of progress was confirmed;
- left the durable and pending 655-row checkpoint files byte-identical;
- left a namespace-local PID in `last_run_status.json`, which was repaired only
  after confirming that no inference process remained.

A defensive GPU relaunch then refused to run because the stale namespace-local
PID appeared live. It added no rows and did not change the checkpoint. Its
failure manifest and the zero-row environment-isolation manifest are finalized
separately from productive inference timing.

An eighteenth bounded command:

- resumed all 655 durable rows and incremented `resume_count` to 17;
- used the same frozen configuration with GPU access restored;
- added 41 macaroni2 rows and stopped at 696 unique records;
- was interrupted during proposal construction for macaroni2 `096`;
- removed the single uncheckpointed fused-evidence artifact for that sample;
- preserved the run ID, both frozen fingerprints, and every durable mask role.

A nineteenth bounded command:

- resumed all 696 checkpoint rows and incremented `resume_count` to 18;
- completed the final four macaroni2 rows and entered pcb1;
- added 45 pcb1 rows and stopped at 745 unique records;
- was interrupted during Qwen inference for pcb1 `045`;
- left no uncheckpointed mask or image artifact;
- preserved the same run ID, frozen fingerprints, and sealed reference
  boundary.

A twentieth bounded command:

- resumed all 745 checkpoint rows and incremented `resume_count` to 19;
- added the next 46 pcb1 rows and stopped at 791 unique records;
- was interrupted during texture-evidence computation for pcb1 `091`;
- left no uncheckpointed image or mask artifact;
- preserved the same run ID, frozen fingerprints, and sealed reference
  boundary.

A twenty-first bounded command:

- resumed all 791 checkpoint rows and incremented `resume_count` to 20;
- completed the final nine pcb1 rows and entered pcb2;
- added the first 33 pcb2 rows and stopped at 833 unique records;
- was interrupted during Qwen inference for pcb2 `033`;
- left no uncheckpointed image or mask artifact;
- preserved the same run ID, frozen fingerprints, and sealed reference
  boundary.

A twenty-second bounded command:

- resumed all 833 checkpoint rows and incremented `resume_count` to 21;
- added the next 45 pcb2 rows and stopped at 878 unique records;
- was interrupted during proposal cleanup for pcb2 `078`;
- removed the single uncheckpointed fused-evidence image for that sample;
- preserved the run ID, both frozen fingerprints, every durable mask role, and
  the sealed reference boundary.

Current checkpoint:

```text
outputs/v3_generic_evidence_visa/auto_masks/qwen/runs/
  20260726T160703Z-fcd6745c/metadata.partial.jsonl
```

The stable runtime manifest has not been published, which is correct for an
incomplete locked run.

## Runtime Projection

| Observation | Value |
| --- | ---: |
| Completed rows | `878 / 1,200` |
| Current run artifacts | `3,989.1 MiB` |
| Previous exact rate, first 60 rows | `17.825 s/image` |
| Latest exact rate, next 55 rows | `17.700 s/image` |
| Fourth-segment rate, 44 capsule rows | `21.814 s/image` |
| Fourth-vs-third segment rate change | `+23.24%` |
| Fifth-segment rate, 42 rows | `23.014 s/image` |
| Fifth-vs-fourth segment rate change | `+5.50%` |
| Sixth-segment rate, 30 cashew rows | `32.720 s/image` |
| Sixth-vs-fifth segment rate change | `+42.17%` |
| Seventh-segment rate, 30 cashew rows | `32.492 s/image` |
| Seventh-vs-sixth segment rate change | `-0.70%` |
| Eighth-segment rate, 30 cashew rows | `33.293 s/image` |
| Eighth-vs-seventh segment rate change | `+2.47%` |
| Ninth-segment rate, 64 transition rows | `15.401 s/image` |
| Ninth-vs-eighth segment rate change | `-53.74%` |
| Tenth-segment rate, 60 transition rows | `16.993 s/image` |
| Tenth-vs-ninth segment rate change | `+10.34%` |
| Eleventh-segment rate, 48 fryum rows | `20.471 s/image` |
| Eleventh-vs-tenth segment rate change | `+20.47%` |
| Twelfth-segment rate, 43 transition rows | `21.451 s/image` |
| Twelfth-vs-eleventh segment rate change | `+4.78%` |
| Thirteenth-segment rate, 34 macaroni1 rows | `27.258 s/image` |
| Thirteenth-vs-twelfth segment rate change | `+27.07%` |
| Fourteenth-segment rate, 35 macaroni1 rows | `26.240 s/image` |
| Fourteenth-vs-thirteenth segment rate change | `-3.74%` |
| Fifteenth-segment rate, 36 transition rows | `25.774 s/image` |
| Fifteenth-vs-fourteenth segment rate change | `-1.78%` |
| Sixteenth-segment rate, 44 macaroni2 rows | `20.876 s/image` |
| Sixteenth-vs-fifteenth segment rate change | `-19.00%` |
| Latest productive rate, 41 macaroni2 rows | `21.907 s/image` |
| Latest-vs-previous productive rate change | `+4.94%` |
| Nineteenth-segment rate, 49 transition rows | `18.334 s/image` |
| Nineteenth-vs-eighteenth segment rate change | `-16.31%` |
| Twentieth-segment rate, 46 pcb1 rows | `19.449 s/image` |
| Twentieth-vs-nineteenth segment rate change | `+6.08%` |
| Twenty-first-segment rate, 42 transition rows | `21.390 s/image` |
| Twenty-first-vs-twentieth segment rate change | `+9.98%` |
| Twenty-second-segment rate, 45 pcb2 rows | `19.959 s/image` |
| Twenty-second-vs-twenty-first rate change | `-6.69%` |
| Cumulative productive rate | `21.656 s/image` |
| Projected remaining compute | approximately `1.94 hours` |
| Projected total compute | approximately `7.22 hours` |
| Linear artifact projection | approximately `5.32 GiB` |
| Free storage after checkpoint | approximately `13.17 GiB` |

The zero-row restricted-sandbox attempt is excluded from productive throughput.
The projection is operational only. Category transitions and cache reuse may
change the final rate and footprint.

## Unscored Diagnostics

The checkpoint now contains eight complete categories plus the first 78 pcb2
anomalies:

```text
candle: 100
capsules: 100
cashew: 100
chewinggum: 100
fryum: 100
macaroni1: 100
macaroni2: 100
pcb1: 100
pcb2: 78

needs_review: 552
soft_mask_only: 326
hard_mask_ok: 0
```

Selected proposal modes over all 878 rows:

```text
fused_q975: 205
fused_q950: 182
fused_q900: 71
fused_q950_component_1: 55
fused_q975_component_1: 19
fused_q900_component_1: 41
fused_q850_component_1: 69
fused_q850: 16
sam2_fused_q850_1: 49
edge_fused_q900_component_1_2: 12
edge_fused_q850_component_1_1: 22
edge_fused_q900_component_1_1: 4
edge_fused_q850_component_1_2: 22
fused_q900_component_2: 16
edge_fused_q850_1: 13
edge_fused_q850_2: 17
edge_fused_q900_1: 17
edge_fused_q900_2: 23
fused_q850_component_2: 15
edge_fused_q900_component_2_2: 2
fused_q900_component_3: 3
edge_fused_q850_component_2_1: 1
fused_q950_component_2: 2
```

An earlier transition segment's exact selector diagnostics are retained for
lineage:

| Diagnostic | Previous 60 | New 55 |
| --- | ---: | ---: |
| Mean expected IoU | `0.1630` | `0.1505` |
| Mean conformal IoU lower bound | `0.0275` | `0.0281` |
| Mean source disagreement | `0.3792` | `0.4643` |
| Mean selected-mask area | `0.0322` | `0.0329` |
| Qwen full-image fallback | `40/60` | `24/55` |

The aggregate expected-IoU drop is mostly category-composition drift, not a
same-category regression. The 40-row candle continuation averaged `0.1665`
expected IoU. The first 15 capsule rows averaged only `0.1078`, while the next
44 capsule rows improved modestly to `0.1289`. Capsule source disagreement
remains high at `0.6436`.

The structure profile is no longer completely collapsed: 11 of the latest 44
capsule rows are `repeated_chain`; the other 33 remain `ring_sector`. Structural
specialists are disabled, so these labels do not route the frozen inference
path.

The final 41 capsule rows strengthen that observation: 15 are
`repeated_chain`, and 26 are `ring_sector`. Relative to the preceding 44
capsules, mean expected IoU improves from `0.1289` to `0.1432`, mean source
disagreement falls from `0.6436` to `0.6183`, and Qwen full-image fallback
falls from `27.3%` to `17.1%`. Despite those directional improvements, 36 of 41
rows remain `needs_review`.

The first 30-row cashew segment is more expensive and more optimistic:

```text
mean expected IoU: 0.1862
mean conformal IoU lower bound: 0.0507
mean source disagreement: 0.6626
Qwen valid localization: 28/30
needs_review: 25/30
soft_mask_only: 5/30
```

The higher expected quality does not produce a reliable acceptance rate because
source disagreement remains high. All 30 cashew rows are labeled `ring_sector`;
specialists remain disabled.

The next category-matched 30-row cashew slice is directionally stronger:

```text
mean expected IoU: 0.2112 versus 0.1862
mean conformal IoU lower bound: 0.0758 versus 0.0507
mean source disagreement: 0.7170 versus 0.6626
Qwen valid localization: 29/30 versus 28/30
soft_mask_only: 9/30 versus 5/30
```

Expected quality and non-review coverage improve, but source disagreement also
worsens. This is evidence of heterogeneous cashew difficulty and selector
confidence variation, not a mask-quality result.

The following matched 30-row cashew slice reverses that confidence gain:

```text
mean expected IoU: 0.1723 versus 0.2112
mean conformal IoU lower bound: 0.0370 versus 0.0758
mean source disagreement: 0.6445 versus 0.7170
Qwen full-image fallback: 8/30 versus 1/30
soft_mask_only: 3/30 versus 9/30
```

Evidence providers agree more closely, but Qwen localization falls back much
more often and selector confidence drops. This makes localization validity the
strongest unscored explanation for the within-category shift.

The final nine cashew rows then shift sharply upward:

```text
mean expected IoU: 0.3288
mean conformal IoU lower bound: 0.1933
Qwen valid localization: 8/9
soft_mask_only: 6/9
```

The first 55 chewing-gum rows are the strongest external acceptance slice so
far:

```text
mean expected IoU: 0.2415
mean conformal IoU lower bound: 0.1117
mean source disagreement: 0.8225
Qwen valid localization: 53/55
soft_mask_only: 29/55
SAM2-selected proposal: 29/55
```

The acceptance increase coincides with valid localization and SAM2 selection,
but source disagreement is extremely high. Only locked labels can determine
whether SAM2 is resolving useful boundaries or publishing overconfident masks.

The remaining 45 chewing-gum rows are harder:

```text
mean expected IoU: 0.1990 versus 0.2415
mean conformal IoU lower bound: 0.0697 versus 0.1117
mean source disagreement: 0.8815 versus 0.8225
soft_mask_only: 13/45 versus 29/55
SAM2-selected proposal: 19/45
```

Across all 100 chewing-gum rows, 42 are `soft_mask_only` and 48 select SAM2.
SAM2 selection therefore does not determine acceptance by itself.

The first 15 fryum rows are substantially stronger:

```text
mean expected IoU: 0.3307
mean conformal IoU lower bound: 0.1953
mean source disagreement: 0.5184
Qwen valid localization: 14/15
soft_mask_only: 15/15
structure profile: repeated_chain 15/15
```

This is the first category slice with complete soft-mask acceptance and a
non-ring structure profile. It remains unverified until locked evaluation.

The next 48 fryum rows replicate that confidence-transfer pattern:

```text
soft_mask_only: 46/48
needs_review: 2/48
mean expected IoU: 0.3339 versus 0.3307
mean conformal IoU lower bound: 0.1984 versus 0.1953
mean source disagreement: 0.5004 versus 0.5184
mean selected-mask area: 0.0767 versus 0.0818
Qwen full-image fallback: 13/48 versus 1/15
structure profile: repeated_chain 48/48
```

The persistent confidence despite substantially more Qwen fallbacks supports
the architectural choice to treat localization as a soft prior: generic
evidence can preserve a coherent decision when localization weakens. This is
still an unscored diagnostic, not evidence that the masks overlap official
defects.

The final 37 fryum rows are more conservative than the first 63:

```text
soft_mask_only: 34/37 versus 61/63
mean expected IoU: 0.2969 versus 0.3331
mean conformal IoU lower bound: 0.1614 versus 0.1976
mean source disagreement: 0.4883 versus 0.5047
mean selected-mask area: 0.0510 versus 0.0779
Qwen full-image fallback: 2/37 versus 14/63
structure profile: repeated_chain 37/37
```

The completed fryum category therefore contains 95 `soft_mask_only` and five
`needs_review` decisions. Confidence and mask area drift downward in the final
slice despite better localization and slightly lower source disagreement. That
pattern points to image-content variation rather than Qwen failure, but only
locked evaluation can establish whether the smaller masks are more precise or
under-segmented.

The first six macaroni1 rows are preliminary:

```text
soft_mask_only: 2/6
needs_review: 4/6
mean expected IoU: 0.2351
mean conformal IoU lower bound: 0.0996
mean source disagreement: 0.3954
Qwen valid localization: 6/6
structure profile: ring_sector 6/6
```

This sample is too small for a category conclusion.

The next 34 macaroni1 rows establish a weaker confidence-transfer regime:

```text
soft_mask_only: 5/34 versus 2/6
needs_review: 29/34 versus 4/6
mean expected IoU: 0.2206 versus 0.2351
mean conformal IoU lower bound: 0.0851 versus 0.0996
mean source disagreement: 0.4743 versus 0.3954
mean selected-mask area: 0.0460 versus 0.0351
Qwen full-image fallback: 17/34 versus 0/6
structure profile: ring_sector 34/34
selected mode: fused_q950 28/34
```

Across the first 40 macaroni1 rows, only seven are `soft_mask_only`. The
confidence decline coincides with both increased localization fallback and
higher evidence disagreement, unlike fryum where the soft-prior path remained
stable under fallback. Selection also concentrates strongly on one fused
quantile. These are warnings about external confidence transfer and candidate
diversity, not official mask-quality measurements.

The next 35 macaroni1 rows confirm that the difficult regime is stable rather
than progressively collapsing:

```text
soft_mask_only: 9/35 versus 5/34
needs_review: 26/35 versus 29/34
mean expected IoU: 0.2186 versus 0.2206
mean conformal IoU lower bound: 0.0831 versus 0.0851
mean source disagreement: 0.4585 versus 0.4743
mean selected-mask area: 0.0401 versus 0.0460
Qwen full-image fallback: 15/35 versus 17/34
structure profile: ring_sector 35/35
selected mode: fused_q950 26/35
```

Acceptance improves modestly and disagreement falls, while calibrated quality
remains essentially flat. Across 75 macaroni1 rows, 59 are `needs_review`, 32
use Qwen fallback, and 57 select `fused_q950`. This strengthens the
preregistered need to inspect search-region recall and selector regret after
the runtime cohort is sealed.

The final 25 macaroni1 rows are the weakest category slice:

```text
soft_mask_only: 1/25 versus 9/35
needs_review: 24/25 versus 26/35
mean expected IoU: 0.1954 versus 0.2186
mean conformal IoU lower bound: 0.0599 versus 0.0831
mean source disagreement: 0.4148 versus 0.4585
mean selected-mask area: 0.0375 versus 0.0401
Qwen full-image fallback: 4/25 versus 15/35
structure profile: ring_sector 25/25
selected modes: fused_q950 13, fused_q975 11, fused_q900 1
```

Macaroni1 completes with 17 `soft_mask_only` and 83 `needs_review` decisions.
The final decline occurs despite better Qwen validity, lower disagreement, and
a broader selected-mode mix, so localization fallback and q950 concentration
cannot fully explain it. Locked evaluation must separate genuinely weak
candidates from selector calibration error.

The first 11 macaroni2 rows are preliminary:

```text
soft_mask_only: 3/11
needs_review: 8/11
mean expected IoU: 0.2198
mean conformal IoU lower bound: 0.0843
mean source disagreement: 0.4969
Qwen full-image fallback: 4/11
structure profile: repeated_chain 6, ring_sector 5
```

The mixed structural profile and broader selected modes differ from macaroni1,
but 11 rows are insufficient for a category conclusion.

The next 44 macaroni2 rows preserve that mixed structure and improve coverage:

```text
soft_mask_only: 16/44 versus 3/11
needs_review: 28/44 versus 8/11
mean expected IoU: 0.2186 versus 0.2198
mean conformal IoU lower bound: 0.0831 versus 0.0843
mean source disagreement: 0.3732 versus 0.4969
mean selected-mask area: 0.0331 versus 0.0429
Qwen full-image fallback: 17/44 versus 4/11
structure profile: ring_sector 23, repeated_chain 21
```

Macaroni2 now contains 19 `soft_mask_only` and 36 `needs_review` decisions
across 55 rows. Calibrated quality remains flat while disagreement and review
rate improve. Proposal selection is also diverse: no mode exceeds 17/55.
Unlike macaroni1, this category does not show a q950 collapse or a single
structure profile.

The next 41 macaroni2 rows isolate localization from confidence transfer:

```text
soft_mask_only: 11/41 versus 16/44
needs_review: 30/41 versus 28/44
mean expected IoU: 0.2156 versus 0.2186
mean conformal IoU lower bound: 0.0801 versus 0.0831
mean source disagreement: 0.4261 versus 0.3732
mean selected-mask area: 0.0358 versus 0.0331
Qwen valid localization: 41/41 versus 27/44
structure profile: repeated_chain 24, ring_sector 17
```

Macaroni2 now contains 30 `soft_mask_only` and 66 `needs_review` decisions
across 96 rows. Its mean expected IoU is `0.2174`, mean conformal lower bound is
`0.0819`, and mean disagreement is `0.4100`. The latest acceptance decline
occurs despite perfect Qwen validity, so localization fallback does not explain
this slice. Evidence agreement and selector calibration remain the more likely
confidence bottlenecks. This is an unscored diagnostic, not a mask-quality
conclusion.

The final four macaroni2 rows do not change the category diagnosis:

```text
soft_mask_only: 1/4
needs_review: 3/4
mean expected IoU: 0.2135
mean conformal IoU lower bound: 0.0780
mean source disagreement: 0.4313
Qwen valid localization: 4/4
structure profile: repeated_chain 2, ring_sector 2
```

Completed macaroni2 contains 31 `soft_mask_only` and 69 `needs_review`
decisions. Its mean expected IoU is `0.2173`, mean lower bound is `0.0818`,
and mean disagreement is `0.4108`. Its structure remains balanced at 53
`repeated_chain` and 47 `ring_sector`.

The first 45 pcb1 rows enter a distinctly more permissive confidence regime:

```text
soft_mask_only: 32/45
needs_review: 13/45
mean expected IoU: 0.2432
mean conformal IoU lower bound: 0.1077
mean source disagreement: 0.5863
mean selected-mask area: 0.0897
Qwen valid localization: 37/45
Qwen full-image fallback: 8/45
structure profile: ring_sector 45/45
```

The acceptance increase is promising, but it is not yet a quality result.
Pcb1 also has larger masks, higher evidence disagreement, and a completely
collapsed structure profile. Structural specialists are disabled, so the
profile does not alter frozen inference, but locked evaluation must determine
whether the selector is capturing PCB defects or over-accepting broad anomaly
regions.

The next 46 pcb1 rows replicate that confidence regime:

```text
soft_mask_only: 32/46 versus 32/45
needs_review: 14/46 versus 13/45
mean expected IoU: 0.2400 versus 0.2432
mean conformal IoU lower bound: 0.1045 versus 0.1077
mean source disagreement: 0.5788 versus 0.5863
mean selected-mask area: 0.0945 versus 0.0897
Qwen valid localization: 38/46 versus 37/45
Qwen full-image fallback: 8/46 versus 8/45
structure profile: ring_sector 46/46
```

Across 91 pcb1 rows, 64 are `soft_mask_only` and 27 are `needs_review`. Mean
expected IoU is `0.2416`, mean lower bound is `0.1061`, mean disagreement is
`0.5825`, and mean selected area is `0.0922`. The near-identical acceptance,
localization, and confidence values across two contiguous slices make the
regime reproducible within pcb1. The persistent `91/91 ring_sector` routing
and large masks remain unresolved over-segmentation and structure-calibration
risks rather than reasons to promote the result.

The final nine pcb1 rows strengthen rather than reverse that result:

```text
soft_mask_only: 9/9
mean expected IoU: 0.2640
mean conformal IoU lower bound: 0.1285
mean source disagreement: 0.5549
mean selected-mask area: 0.1232
Qwen valid localization: 9/9
structure profile: ring_sector 9/9
```

Completed pcb1 contains 73 `soft_mask_only` and 27 `needs_review` decisions.
Its mean expected IoU is `0.2436`, mean lower bound is `0.1081`, mean
disagreement is `0.5800`, and mean selected area is `0.0950`. Every row remains
`ring_sector`.

The first 33 pcb2 rows enter a more conservative regime:

```text
soft_mask_only: 16/33
needs_review: 17/33
mean expected IoU: 0.2215
mean conformal IoU lower bound: 0.0860
mean source disagreement: 0.4277
mean selected-mask area: 0.0417
Qwen valid localization: 31/33
Qwen full-image fallback: 2/33
structure profile: ring_sector 33/33
```

Relative to pcb1, pcb2 has lower predicted acceptance and substantially smaller
masks, but also much lower source disagreement and better localization. This is
category-transfer heterogeneity, not evidence that pcb2 is worse. Locked
precision, recall, positive rate, and selector regret must determine whether
pcb2 is appropriately conservative or under-segmented.

The next 45 pcb2 rows are a strictly category-matched continuation and become
more conservative again:

```text
soft_mask_only: 14/45 versus 16/33
needs_review: 31/45 versus 17/33
mean expected IoU: 0.2026 versus 0.2215
mean conformal IoU lower bound: 0.0671 versus 0.0860
mean source disagreement: 0.4272 versus 0.4277
mean selected-mask area: 0.0452 versus 0.0417
Qwen valid localization: 42/45 versus 31/33
structure profile: ring_sector 45/45
```

The acceptance drop from `48.5%` to `31.1%` occurs while localization validity
and source disagreement are nearly unchanged and mask area rises slightly.
This rules out a simple Qwen-fallback or provider-disagreement explanation for
the confidence drift. Content difficulty, candidate-pool composition, and
selector transfer remain the preregistered locked analyses.

Across all 78 pcb2 rows, 30 are `soft_mask_only` and 48 are `needs_review`.
Mean expected IoU is `0.2106`, mean conformal lower bound is `0.0751`, mean
source disagreement is `0.4274`, and mean selected area is `0.0437`. All 78
rows remain `ring_sector`; specialists are disabled.

These are serious confidence-transfer warnings, but they are not official mask
quality measurements. The preregistered run must complete before opening
official masks or changing selector thresholds.

Checkpoint integrity:

```text
partial metadata SHA-256:
a1f5a946b89e6526b21e39931722f32300d704038e55b68cb7c900e3394432fa

architecture fingerprint:
f946c2ea91eaa6e3727f1f0c2d13fc5518e0daf113404638c1288b90721daf23

auto-mask fingerprint:
8aceb1c68c1f8c76b256252938208e4292863a4a7dedf035878a52a700b61fd2
```

## Next Execution

Resume with the unchanged code and configuration:

```bash
python -m iadgen_v2.cli auto-masks \
  --config configs/v3_generic_evidence_visa.yaml
```

Any package-code, selector, checkpoint, model, or behavioral-config change will
correctly invalidate this checkpoint and start a fresh cohort.
