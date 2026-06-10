# Data Augmentation Plan — Carbonate Thin-Section Segmentation

> Status: **in progress**. Phase 0 (class-list reconciliation to 18 classes + `--ignore_artifacts`)
> is **implemented** in `swin_training_pipeline_221.py`; remaining phases are still proposals for
> review. This document describes *what* we'd build and *why*, so we can agree on scope before
> implementing each phase.

## Context

We do pixel-level semantic segmentation of carbonate thin sections with a SwinV2-Tiny +
UPerNet model ([swin_training_pipeline_221.py](swin_training_pipeline_221.py)), SSL-pretrained
on unlabeled images and finetuned with **5-fold cross-validation** (stratified folds were just
added on branch `feat/stratified-kfold-cv`). The labeled set is ~71 images today, ~100 soon —
small enough that augmentation is load-bearing, not cosmetic.

The class statistics (71 images) expose three distinct problems that drive the whole plan:

| Pattern | Classes | Implication |
|---|---|---|
| Rare by image (<5 imgs) | Brachiopod (2), Aggregate Grain (3) | Cannot appear in every val fold; stratifier already warns. |
| Image-poor / instance-rich | Mollusk (6 img / 170 inst), Ooid (14/863), Calc. Algae (7/63) | Copy-paste / mosaic to redistribute is high-value. |
| Pixel-tiny objects | Ostracod (159k px), Watermark (255k), Gastropods (1.2M) | Random 512² crops on ~3840×3072 images rarely contain them. |
| Pixel-dominant matrix | Micrite (150M px), Cement (139M px) | ~85% of labeled pixels; needs class-weight/focal loss (already supported). |

## Goals

- Increase effective sample diversity and rare-class exposure without distorting petrographic reality.
- Keep every augmentation **mask-correct** and **CV-leakage-free**.
- Make augmented samples **visually inspectable** so we can verify "realistic."

## Non-goals (for now)

- Instance / panoptic segmentation (current model is semantic; see caveat below).
- PPL/XPL synthetic recoloring (high risk to diagnostic color; deferred).
- Changing the model architecture or the SSL pretraining stage.

## Correctness invariants (apply to every augmentation)

1. **Masks use nearest-neighbor; out-of-frame fill = `ignore_index` (255), not background.**
   Affine/elastic/rotation borders must pad the mask with ignore, or we teach "black corner =
   background." The dataset already wraps masks as `tv_tensors.Mask`, so torchvision v2 does
   nearest automatically — we just set the fill value.
2. **Cross-image augmentation must be fold-safe.** Mosaic, copy-paste, and matrix-swap pull
   content from *other* images; they must draw only from the **current fold's training
   indices**. This forces them to live inside `run_single_fold` (after the split), not in a
   global transform. Violating this leaks validation grains into training and inflates CV mIoU.
3. **Validation transforms stay deterministic** (center-crop only). Augment train folds only.
4. **Color is diagnostic** (XPL interference colors): photometric jitter stays mild.
5. **Magnification consistency**: mosaic/scale must not mix magnifications (filenames show
   "4x", "2x detail"), or grain sizes become physically impossible.
6. **Artifacts** (Scale Bar, Watermark) route to `ignore_index` and are never used as
   copy-paste/mosaic source content.

> **Semantic-vs-instance note:** the "blur merges touching instances" concern does not apply to
> the current semantic model — two touching ooids share the label "ooid," so there is no
> instance boundary to lose. Mild blur is safe. This caveat only returns if we move to instance
> segmentation later.

## Phase 0 — Reconcile the class list (prerequisite, small) — ✅ IMPLEMENTED

The code currently defines `NUM_CLASSES = 16` and a 16-entry `CLASS_NAMES` (background + 15).
The new stats table has **17 foreground classes including Sponge and Watermark**, which are not
in the code today. Before any augmentation:

- Confirm the authoritative **label-id → class** mapping from the current Supervisely export
  (does "17 classes" include background? what integer id is Sponge/Watermark?).
- Update `NUM_CLASSES`, `CLASS_NAMES`, and route **Watermark + Scale Bar → ignore_index**
  (generalize the existing `--ignore_scale_bar` to an `--ignore_artifacts` set).
- Without this, per-class IoU rows are mislabeled and metrics are wrong regardless of augmentation.

**Open question to resolve here:** exact id mapping and total label count.

## Phase 1 — Per-sample augmentations (cheap, safe, highest immediate value)

These slot into the `train_transforms` Compose built in `run_single_fold`; validation transforms
unchanged. All are single-image, mask-aware.

- **Geometric (image+mask):** RandomAffine (rotate ±, scale ~0.8–1.2 within a magnification
  band, shear small, translate small) with mask fill = ignore. Keep existing H/V flips.
- **Rare-class-aware crop** (new custom transform): with probability `p`, choose the crop
  window to contain a randomly selected rare-class pixel (from a configurable rare-class set);
  otherwise crop randomly. This directly addresses the "tiny objects never appear in crops"
  problem and is likely the single highest-impact item given the stats.
- **Photometric (image only, mild):** brightness/contrast, gamma, hue/saturation (bounded),
  Gaussian noise, Gaussian blur, CLAHE / histogram equalization.

**Where it lives:** `CarbonateSegmentationDataset.transforms` pipeline. The rare-class crop needs
the mask to pick a center, which the joint `transforms(img, mask)` signature already provides.

## Phase 2 — Multi-image composition (medium effort, fold-safe)

- **Mosaic (2–4 images) / quadrant-puzzle:** sample k training items, tile their images+masks
  into one canvas, then crop. The "split into quadrants and recombine across images" idea is a
  mosaic variant. Must enforce magnification consistency and fill seams/borders with ignore.
- **Elastic / grid distortion (mild):** grains are semi-rigid; small displacement only, mask nearest.

**Where it lives:** a fold-scoped dataset wrapper constructed inside `run_single_fold` from the
training Subset, so sources are train-only.

## Phase 3 — Synthetic compositing (high effort, high payoff for rare classes)

- **Copy-paste grain compositing:** cut instances of target rare classes (Brachiopod, Mollusk,
  Aggregate Grain, Gastropods) from training images and paste—with edge blending—onto other
  training matrices; update the mask accordingly. **Fold-safe sourcing is mandatory.**
- **Synthetic occlusion:** CoarseDropout-style patches filled with matrix/cement texture over
  grains; occluded mask region becomes the matrix/cement label (teaches partial-grain recognition).
- **Matrix/cement texture swap:** composite grains onto different matrix backgrounds; hardest to
  make seamless; lowest priority.

## Library decision (recommended default)

Use **torchvision `transforms.v2`** for Phases 0–1 (already integrated via `tv_tensors`, no new
dependency; covers everything except CLAHE, which we can implement or pull narrowly). Introduce
**Albumentations** only if/when we want its richer set (CLAHE, ElasticTransform, GridDistortion,
CoarseDropout) for Phases 2–3. Mosaic and copy-paste are custom in either library.

*(Open decision — can switch to Albumentations-first if you prefer fewer custom transforms.)*

## Proposed config surface

- `--aug_level {none,light,medium,heavy}` preset, plus granular overrides:
- `--rare_crop_prob`, `--rare_classes <ids>`, `--mosaic_prob`, `--mosaic_n {2,4}`,
  `--copypaste_prob`, `--occlusion_prob`, photometric magnitudes.
- `--dump_aug_samples N` — write N augmented image+mask overlays to `<output_dir>/aug_preview/`
  for visual QA (critical for verifying realism before a long run).

## Verification plan

1. **Visual QA:** `--dump_aug_samples` overlays — confirm masks track transforms and grains look
   plausible (no impossible scales, no seams through grains, artifacts excluded).
2. **Leakage check:** assert copy-paste/mosaic source indices ⊆ current fold's train indices.
3. **Mask-fill check:** confirm padded regions are `ignore_index`, not background.
4. **Smoke run:** `--epochs 1 --aug_level heavy` on Colab; confirm it trains and the per-fold
   class-coverage / per-class IoU logs look sane.
5. **A/B:** short 5-fold run with `--aug_level none` vs `light` to confirm rare-class IoU moves.

## Suggested order of work

Phase 0 → Phase 1 (esp. rare-class-aware crop) → visual QA → Phase 2 → Phase 3. Each phase is a
separate branch/PR so it can be reviewed and smoke-tested independently.

## Open questions for you

1. Exact 17(/18?)-class label-id mapping (Phase 0 blocker).
2. Library preference: torchvision-first (recommended) vs Albumentations-first.
3. Which rare classes to prioritize for copy-paste, and is fold-safe copy-paste worth the effort
   now or after Phases 0–2 land.
4. Are all images the same magnification, or do we need magnification metadata to bound
   scale/mosaic?
