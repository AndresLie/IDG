# VLM-Guided Surface Defect Generation with a Trainable Diffusion Adapter

## 1. Objective and Hypothesis

Industrial defect synthesis must place anomalies on physically plausible surfaces
and render damage that matches the local material and structure. This study
extends the existing `v1` MVTec AD generation and evaluation pipeline with
vision-language model (VLM) guidance for both placement and appearance.

**Hypothesis:** VLM-derived spatial-semantic guidance can improve both defect
placement plausibility and generated defect appearance, leading to stronger
downstream anomaly segmentation than text-only or placement-only diffusion
baselines.

The study tests two separate claims:

1. VLM guidance selects more plausible defect locations on clean target images.
2. VLM-derived token conditioning improves defect rendering inside a selected
   region when location is held constant.

### Initial Scope

The first experiment is limited to surface damage in MVTec AD:

| Category | Defect Type | Motivation |
| --- | --- | --- |
| `metal_nut` | `scratch` | Reflective manufactured metal surface |
| `tile` | `crack` | Planar brittle surface with visible fracture patterns |
| `wood` | `scratch` | Textured natural surface requiring material consistency |

Structural defects such as broken or crushed zipper teeth are out of scope for
the initial experiment. They may be tested only after the surface-defect method
has been validated.

## 2. Primary Method

### 2.1 Model Stack

The first working system uses:

- **VLM:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Renderer:** the existing `v1` Stable Diffusion 1.5 ControlNet inpainting
  path, extended with adapter conditioning
- **Existing infrastructure:** the `v1` mask-based generation workflow,
  background-preservation metrics, and placement benchmark concepts

For each clean target image and defect instruction, the VLM is run once to
identify a plausible target region, expressed as coordinates or a bounding
box, and extract spatial-semantic visual features. The VLM proposal is not
treated as a pixel-accurate anomaly mask. A mask-refinement stage creates an
irregular insertion mask within that region before generation.

### 2.2 Location-to-Mask Refinement

The spatial generation path separates **where** a defect belongs from **what
shape** its pixels occupy:

1. The VLM proposes a surface-valid region and records its normalized
   coordinates, defect type, and placement confidence.
2. A defect-specific procedural mask generator samples a binary morphology
   within the proposed region:
   - `scratch` masks use a curved polyline or Bezier-like stroke with width
     jitter, tapered endpoints, and bounded dilation.
   - `crack` masks use a jagged random-walk centerline with sparse branches,
     tapered widths, and bounded dilation.
3. The binary evaluation mask is preserved exactly for leakage, visibility,
   and segmentation-label measurements. A lightly feathered copy may be used
   only as the inpainting blend mask and must be recorded separately.
4. Every sample records the VLM region, mask-generation seed, morphology
   parameters, binary mask, and inpainting mask so results are reproducible.

Morphology parameter ranges are estimated only from ground-truth masks in the
adaptation split. Masks from held-out anomalies are never used to set shape
parameters or generate synthetic labels. This design prevents rectangular VLM
boxes from becoming artificial square defect boundaries.

### 2.3 Trainable Appearance Adapter

The core contribution is a lightweight trainable adapter that aligns VLM
spatial-semantic tokens with diffusion appearance conditioning:

1. Select and pool a fixed number `K` of VLM tokens representing the proposed
   target region, surrounding material, and requested defect context.
2. Project those tokens from VLM hidden width into Stable Diffusion 1.5
   cross-attention width `768`.
3. Append projected tokens to the standard CLIP text-conditioning sequence,
   giving conditioning shape `(77 + K) \times 768`.
4. Apply a learnable gate that controls the contribution of projected VLM
   tokens during denoising.

Stable Diffusion 1.5 uses standard CLIP text conditioning of shape
`77 \times 768`; the adapter augments its cross-attention key/value
conditioning rather than replacing CLIP embeddings.

The token adapter is not expected to preserve pixel-level coordinates after
projection into a one-dimensional cross-attention sequence. In the primary
method, VLM influence on spatial layout is expressed through its target-region
proposal and the refined insertion mask supplied to SD1.5 inpainting; no claim
is made that appended tokens alone localize pixels. The existing
background/edge ControlNet path from `v1` is retained to preserve surrounding
surface geometry. The gated VLM tokens are evaluated as appearance and
material-context guidance inside that spatial constraint.

The VLM, CLIP text encoder, VAE, and diffusion U-Net remain frozen. Only the
projection adapter and gating parameters are trained.

### 2.4 Proposed Generation Contract

The future generation path accepts:

- clean target image,
- defect instruction, such as "add a narrow surface scratch on the metal face,"
- VLM-selected target region and placement metadata,
- procedurally refined binary insertion mask and optional feathered inpainting
  mask,
- cached VLM spatial-semantic token features,
- projected and gated adapter conditioning.

It produces a generated anomalous image plus metadata required for evaluation:
region, masks, mask-generation seed and parameters, prompt, placement score or
rationale output, generation seed, runtime, and adapter configuration.

## 3. Training and Data Protocol

### 3.1 Data Split

Use MVTec AD samples only from the initial surface-defect scope. For each
category/defect pair, establish before experiments:

- a fixed few-shot **adaptation split** of real anomalous images and masks for
  adapter training and eligible source references,
- a disjoint **held-out evaluation split** of real anomalous images and masks
  used only for final measurement,
- clean `good` images used as generation targets according to the experiment
  split configuration.

Held-out anomaly images must not be used as generation references,
adapter-training targets, prompt examples, placement calibration inputs, or
hyperparameter-selection evidence.

### 3.2 Offline VLM Feature Cache

The VLM must not be resident in memory during adapter optimization. Before
adapter training:

1. Materialize each adaptation input, its corruption mask, defect instruction,
   and the VLM-selected region/context used by the training example.
2. Run Qwen in inference mode with gradients disabled and save the selected
   token tensor for each example as a `.pt` feature-cache artifact.
3. Save accompanying JSON metadata containing model identifier/revision,
   prompt, image and mask identifiers, tensor shape, token-selection policy,
   region coordinates, and preprocessing settings.
4. Treat any image or prompt transformation that would affect VLM output as a
   new cache item; diffusion timestep/noise sampling remains online during
   adapter training.

At training time, load cached tensors from disk and load only the diffusion
pipeline plus adapter/gate. At inference time, compute and cache VLM features
before denoising and release or offload the VLM before diffusion rendering when
memory limits require it.

### 3.3 Adapter Training

For each adaptation example:

1. Use the ground-truth anomaly mask to hide or inpaint-corrupt the anomalous
   region in the training image.
2. Load the precomputed VLM spatial-semantic tensor for the corrupted image
   context and defect instruction from the offline feature cache.
3. Feed projected, gated tokens alongside CLIP prompt conditioning to the
   diffusion model, using the ground-truth mask as the spatial constraint.
4. Optimize only the adapter and gate parameters with diffusion denoising loss
   so that the original anomalous region is reconstructed.

### 3.4 Inference

For each clean target image:

1. Provide the clean image and requested surface defect instruction to the VLM.
2. Select a plausible target region and generate an irregular insertion mask
   within it using the relevant `scratch` or `crack` morphology policy.
3. Cache VLM token features once for that image/instruction pair and free or
   offload the VLM before rendering when necessary.
4. Generate the defect through masked SD1.5 inpainting with the retained
   ControlNet geometry path, using CLIP conditioning plus projected and gated
   VLM appearance conditioning.

## 4. Baselines and Ablations

Experiments must isolate placement quality from rendering quality:

| Experiment | Placement Source | Appearance Conditioning | Purpose |
| --- | --- | --- | --- |
| `SD1.5 text-only` | Existing/default mask policy | CLIP prompt only | Minimum diffusion baseline |
| `Existing placement + SD1.5` | Strongest applicable `v1` placement baseline | CLIP prompt only | Existing pipeline comparison |
| `VLM-box only` | Unrefined VLM bounding-box mask | CLIP prompt only | Measure the failure mode of crude region masks |
| `VLM-mask only` | VLM region plus refined morphology mask | CLIP prompt only | Isolate VLM placement and mask-refinement benefit |
| `Adapter with fixed masks` | Identical fixed masks across methods | CLIP plus gated VLM tokens | Isolate rendering benefit |
| `Full VLM hybrid` | VLM region plus refined morphology mask | CLIP plus gated VLM tokens | Full proposed method |

Required adapter ablations:

- Disable the learned gate or hold it to a fixed contribution.
- Remove projected VLM tokens while retaining the same masks and prompts.
- Compare unrefined bounding boxes with defect-specific refined masks while
  retaining the same VLM-selected regions and appearance conditioning.
- Report conditioning shape checks and behavior under identical generation
  seeds where comparisons share a mask.

A larger VLM or SDXL renderer is a follow-up ablation only after the primary
Qwen2.5-VL-3B plus SD1.5 experiment is functioning and measured.

## 5. Evaluation and Success Criteria

### 5.1 Placement Evaluation

Measure placement separately from rendering using:

- valid-surface placement rate,
- boundary clearance and material-continuity measures aligned with the existing
  `v1` placement benchmark,
- qualitative overlays of selected masks on clean images.

Placement comparisons must include the existing placement method and the
VLM-selected-region method without adapter rendering. Mask analysis must also
compare unrefined VLM bounding boxes against refined `scratch`/`crack`
morphologies to quantify boundary artifacts and labeling quality.

### 5.2 Rendering Evaluation

Measure generated-image behavior using:

- background preservation outside the insertion mask,
- mask leakage outside the intended defect region,
- defect visibility within the mask,
- distribution-level realism metrics only when evaluated on sufficiently sized,
  comparable sets of real and generated anomaly images.

Reference-image LPIPS must not be presented as a standalone realism claim when
the generated defect and reference defect occur on different backgrounds or at
different locations.

### 5.3 Downstream Segmentation Evaluation

The primary downstream test is anomaly segmentation on held-out real anomaly
images:

1. Train one fixed U-Net configuration using only real adaptation data.
2. Train otherwise identical U-Net runs for each generator under predetermined
   anomalous-sample mixes: real only, real:synthetic `1:1`, real:synthetic
   `1:4`, and synthetic only.
3. Keep the normal-sample policy, category balance, optimizer configuration,
   number of training updates, and random seeds fixed across generator/mix
   comparisons so increased synthetic volume is not itself the advantage.
4. Select no mix ratio using the held-out test anomalies: either report all
   fixed ratios or select a preferred ratio using adaptation-only
   cross-validation before final held-out evaluation.
5. Evaluate every trained model on the same held-out real split with the same
   reporting protocol.

Report pixel-level AUROC, AUPRO, and IoU or Dice for every method and
real-to-synthetic ratio. This identifies whether synthetic generation is useful
augmentation or introduces a domain-shift penalty.

### 5.4 Success Criteria

The proposed `Full VLM hybrid` method is successful if it:

- improves held-out downstream segmentation over both `SD1.5 text-only` and
  `VLM-mask only`,
- improves or supports the placement claim relative to the existing placement
  baseline,
- does not materially worsen background preservation or mask leakage compared
  with the strongest rendering baseline.

## 6. Implementation Milestones

### Phase 1: Baseline, Splits, and Feasibility (Weeks 1-2)

- [x] Validate execution of the `v1`-derived SD1.5 baseline on the three
  surface-defect targets using surface-constrained adaptation-mask placement.
  The initial baseline batch uses ten generated targets per category; larger
  comparison runs remain part of later experimental reporting.
- [x] Create fixed adaptation and held-out evaluation splits and record seeds.
- [ ] Establish placement, rendering, and downstream segmentation metric loops.
  Rendering reports, Phase 2 placement metrics, and segmentation metric
  utilities are implemented; downstream U-Net training/evaluation remains
  pending.
- [ ] Confirm target GPU capacity, model availability, peak memory, and
  training feasibility before combined VLM/diffusion experiments.
  SD1.5 baseline feasibility and peak memory are measured; combined
  VLM/diffusion feasibility remains pending.

### Phase 2: VLM Spatial Grounding (Weeks 3-4)

- [x] Implement Qwen2.5-VL-3B prompt templates for surface-defect location
  selection as region/coordinate proposals. The current runnable provider is
  heuristic because Qwen weights and `qwen-vl-utils` are not available locally.
- [x] Implement defect-specific mask refinement for `scratch` and `crack`
  regions with logged seeds and morphology parameters.
- [ ] Extract spatial-semantic VLM features and produce both crude-box and
  refined-mask placement variants for clean target images.
  The Phase 2 artifact contract is implemented with placeholder `.pt` feature
  caches for plumbing; real Qwen token extraction remains pending.
- [ ] Evaluate VLM placement against the applicable existing placement baseline
  and quantify bounding-box versus refined-mask behavior before training an
  appearance adapter.
  A heuristic placement report is implemented and has been run on 30 configured
  samples; VLM-vs-baseline placement evaluation remains pending.

### Phase 3: Gated Projection Adapter (Weeks 5-7)

- [ ] Precompute Qwen feature tensors for the materialized adaptation inputs
  into `.pt` cache files with reproducibility metadata; do not load Qwen in
  the adapter training loop.
  The offline adaptation-cache path is implemented with heuristic placeholder
  tensors for all adaptation samples; real Qwen feature extraction remains
  pending.
- [x] Implement the projection adapter from selected VLM features to diffusion
  cross-attention width `768`.
- [x] Append projected tokens to CLIP conditioning through a learnable gate
  while freezing the VLM and diffusion backbones.
- [ ] Train on adaptation examples using diffusion denoising loss.
  A surrogate adapter-shape training command is implemented and verified on
  adaptation-only records; true SD1.5 denoising-loss training remains pending.
- [ ] Validate token shapes, gate behavior, and fixed-mask rendering tests.
  Token-shape and gate validation are implemented; fixed-mask diffusion
  rendering tests remain pending.

### Phase 4: Integrated Generation and Compute Profiling (Weeks 8-9)

- [ ] Integrate VLM-selected regions, refined masks, retained spatial
  inpainting/ControlNet conditioning, and adapter appearance conditioning into
  the generation workflow.
  The heuristic integration contract now consumes Phase 2 regions/masks/cached
  tokens and the Phase 3 adapter checkpoint; real Qwen-selected regions and
  SD1.5 adapter inference remain pending.
- [ ] Cache VLM features once per clean-image/instruction pair and unload or
  offload the VLM before diffusion rendering when required.
  Cached feature artifacts are consumed once per prompt/image pair in the
  heuristic path; actual VLM offload behavior remains pending because Qwen is
  not loaded.
- [ ] Measure runtime and peak VRAM, then apply mixed precision, offloading, or
  attention optimizations as required by verified hardware limits.
  Runtime, max RSS, background-preservation, and CUDA-memory fields are
  reported for the current heuristic path; peak VRAM is `n/a` for the CPU run,
  and optimization decisions for real Qwen/SD inference remain pending.

### Phase 5: Evaluation and Reporting (Weeks 10-12)

- [ ] Run placement, rendering, and full-method baseline comparisons.
  Placement, Phase 4 integration, and downstream scaffold reports are
  materialized for the heuristic path; final full-method comparisons require
  real Qwen/SD outputs.
- [ ] Run mask-refinement and adapter ablations using controlled regions,
  masks, and seeds.
- [ ] Train and evaluate downstream U-Net segmentation models on held-out real
  anomalies at the predetermined real-to-synthetic training ratios.
  A tiny U-Net smoke scaffold now runs real-only versus real-plus-synthetic
  ratios on held-out real anomalies; final architecture, training budget, and
  real Qwen/SD synthetic data remain pending.
- [ ] Produce qualitative comparisons, quantitative tables, compute profiles,
  and limitations.
  Phase reports and CSV tables are generated under `v2/reports`; final paper
  tables still require the real model stack and ablations.

### Phase 6: Real-Model Enablement

- [x] Install and preflight Qwen runtime dependencies.
  `qwen-vl-utils` is installed and the Phase 6 preflight reports CUDA, Qwen,
  SD1.5, and storage readiness.
- [x] Add real `qwen` provider paths for Qwen token extraction, SD1.5
  denoising-loss adapter training, and SD1.5 inpainting generation.
  These paths fail clearly when Qwen weights are unavailable.
- [ ] Cache `Qwen/Qwen2.5-VL-3B-Instruct` and run one-image-per-category Qwen
  extraction.
  Current blocker: Qwen is not cached and the filesystem has less than the
  configured `30 GiB` free-space requirement.
- [ ] Run real denoising adapter training and real hybrid generation.
  These require the Qwen caches above and should be followed by Phase 5 reruns
  with `provider=qwen`.

## 7. Academic Deliverables

- **Methodology:** mathematical formulation of the projected and gated VLM
  appearance adapter, explicit mask-refinement procedure, retained spatial
  conditioning path, frozen/trainable components, and conditioning dimensions.
- **Experimental Protocol:** documented dataset split, no-leakage rule,
  cached-feature construction, baselines, controlled ablations, downstream
  data ratios, seeds, and compute environment.
- **Qualitative Results:** placement overlays and side-by-side defect rendering
  examples for placement-only and full-adapter variants.
- **Quantitative Results:** placement, preservation/leakage, compute, and
  held-out segmentation tables.
- **Limitations:** explicit treatment of surface-only initial scope, compute
  requirements, and follow-up validation needed for structural defects, SDXL,
  or alternate VLMs.

## 8. Assumptions and Constraints

- The first research target includes only `metal_nut/scratch`, `tile/crack`,
  and `wood/scratch`.
- The primary stack is `Qwen/Qwen2.5-VL-3B-Instruct` with Stable Diffusion 1.5
  inpainting.
- Conditioning retains CLIP prompt embeddings and appends projected VLM tokens
  through a learnable gate for appearance guidance; spatial placement is
  enforced by refined masks in the inpainting/ControlNet path.
- Qwen region coordinates are not used directly as binary anomaly masks;
  defect-specific procedural morphology creates synthetic training labels
  using parameters derived only from adaptation data.
- Adapter training consumes offline cached VLM tensors rather than running the
  VLM within the optimization loop.
- The current v2 code implements the baseline pipeline, heuristic spatial
  artifact flow, offline adapter-cache scaffolding, and gated projection
  adapter validation. Real Qwen extraction and SD1.5 denoising-loss adapter
  training remain pending.
- SD1.5 baseline feasibility has been measured on the available NVIDIA RTX
  A6000. Combined Qwen/diffusion feasibility remains mandatory before adapter
  training or integrated inference.
