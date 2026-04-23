# Swin Transformer + UPerNet Segmentation Model

This document walks through the full model architecture used in the carbonate-thin-section segmentation pipeline, from self-supervised pretraining on unlabeled petrographic imagery through supervised finetuning for 16-class semantic segmentation. It is written to be technically complete but readable without a computer-vision background; terms are defined where they first appear.

The pipeline has two training stages, implemented in two scripts:

1. **Masked self-supervised (SSL) pretraining** — [swin_ssl_pretrain_221.py](swin_ssl_pretrain_221.py). Trains the Swin backbone alone to reconstruct hidden parts of unlabeled carbonate images.
2. **Supervised finetuning** — [swin_training_pipeline_221.py](swin_training_pipeline_221.py). Attaches a UPerNet decoder on top of the pretrained Swin backbone and trains end-to-end on labeled masks (16 classes) with cross-entropy loss.

The two stages are connected by a weight-transfer step: the backbone learned in stage (1) is loaded into the encoder of stage (2) via `--backbone_checkpoint`, so stage (2) starts from a backbone that already "understands" carbonate textures rather than a generic ImageNet one.

---

## 1. Stage-1 — Masked self-supervised pretraining

### 1.1 Why pretrain at all?

In petrography, pixel-accurate class labels are expensive: a trained micromorphologist must trace every bivalve fragment, peloid, ooid, echinoderm, etc., in each thin section. Labeled tiles are therefore scarce. The unlabeled image corpus, however, is orders of magnitude larger — every thin section we have scanned is potential training material if we can learn from it *without* needing masks.

Masked self-supervised learning is how we do that. The idea is simple: if a model can look at a carbonate image with large patches blacked out and still correctly guess what is behind those patches, it must have internalized useful priors about carbonate texture, grain morphology, matrix appearance, and local context. Those priors are exactly what we want the backbone to bring into the downstream segmentation task.

This is the same recipe (masked image modeling, or MIM) behind models like MAE and SimMIM in the general computer-vision literature; here it is restricted to our domain so the priors are domain-specific rather than generic natural-image priors.

### 1.2 What the pretraining actually does

For each training step ([swin_ssl_pretrain_221.py:340-384](swin_ssl_pretrain_221.py)):

1. Sample an unlabeled carbonate image (from `cretaceous thin sections`, `Permian-Triassic`, and `T/J photomicrographs`), random-resized-crop it to 512×512, and normalize.
2. Divide the 512×512 image into a 32×32 grid of 16-pixel blocks, then randomly hide ~55% of those blocks by zeroing their pixels (`random_block_mask`, [swin_ssl_pretrain_221.py:133-144](swin_ssl_pretrain_221.py)). This is the *masked input*.
3. Feed the masked input through the SwinV2-Tiny backbone, then a small stack of transposed-convolution layers that upsamples the backbone's 16×16 feature map back to 512×512 RGB (`SwinMaskedPretrainModel`, [swin_ssl_pretrain_221.py:147-173](swin_ssl_pretrain_221.py)).
4. Compare the prediction to the *original* image, but only at pixels that were hidden (`masked_l1_loss`, [swin_ssl_pretrain_221.py:176-180](swin_ssl_pretrain_221.py)). The gradient from that L1 reconstruction error updates the backbone.

The decoder in step 3 is a throwaway: its only job is to turn features into pixels so the loss can be computed. After pretraining we discard it and keep just the backbone weights (`backbone_state`, [swin_ssl_pretrain_221.py:190-199](swin_ssl_pretrain_221.py)). Reconstructing pixels is a *pretext* task — we never care about the reconstructed images themselves; we care that the backbone had to learn rich representations to produce them.

### 1.3 What the backbone learns from this

To fill in a hidden ooid or a missing echinoderm ossicle, the backbone has to learn, from data alone:

- What carbonate textures look like at multiple scales (micrite matrix vs. sparry cement vs. skeletal material).
- How grains are shaped, oriented, and spatially arranged.
- What kinds of neighborhoods are consistent (e.g., cement rims around grains, laminated micrite, peloidal packstones).
- Color, brightness, and birefringence statistics under our imaging setup.

None of that requires a human label. The model discovers it by being forced to predict missing parts of real images. These priors are more useful for our downstream task than ImageNet priors (which are tuned to cats, cars, and people), because they come from exactly the visual world our segmenter will work in.

### 1.4 Pretraining configuration (defaults)

| Parameter | Default | Rationale |
|---|---|---|
| Backbone | `microsoft/swinv2-tiny-patch4-window8-256` | Smallest SwinV2, fits comfortably on one GPU. |
| Crop | 512×512 | Large enough to contain multi-grain context at typical magnifications. |
| Mask block size | 16 px | Blocks large enough that the model cannot trivially interpolate from neighboring pixels. |
| Mask ratio | 0.55 | Aggressive masking forces genuine semantic reasoning, not local inpainting. |
| Loss | Masked L1 on pixel values | Simple and stable; only hidden pixels contribute. |
| Optimizer | AdamW (β=0.9/0.95), weight decay 0.05 | Standard for transformer pretraining. |
| LR schedule | Cosine with 10-epoch warmup | Prevents early instability; smooth decay afterwards. |
| Checkpoints | `ssl_swinv2_best.pth` / `..._last.pth` / periodic | Best is tracked by training loss; all include `backbone_state` for transfer. |

Reconstruction previews (`recon_previews/epoch_XXX.png`) are written each epoch so you can visually confirm the model is learning to recover plausible carbonate structure rather than memorizing background color.

### 1.5 Weight transfer — how pretraining connects to finetuning

The SSL checkpoint stores the backbone weights under the key `backbone_state`, already stripped of the `backbone.` prefix so they can be loaded directly into a fresh UPerNet's backbone (`strip_backbone_prefix`, [swin_ssl_pretrain_221.py:190-199](swin_ssl_pretrain_221.py)).

At the start of finetuning, `load_ssl_backbone_checkpoint` ([swin_training_pipeline_221.py:175-201](swin_training_pipeline_221.py)) reads that dictionary and calls `model.backbone.load_state_dict(..., strict=False)`. The `strict=False` is intentional: the SSL backbone and the UPerNet backbone are the same architecture (SwinV2-Tiny), but the hierarchical-feature adapter that UPerNet inserts can add a few small keys; `strict=False` lets the compatible ones load cleanly and reports anything that did not match, rather than crashing.

After this step, only the *encoder* has been initialized from SSL. The UPerNet decoder still starts from random weights. That is by design: the decoder is task-specific (it outputs class logits for our 16 classes), so there is nothing in SSL for it to inherit. The next stage trains everything together.

---

## 2. Stage-2 — Supervised finetuning for 16-class segmentation

The goal of this stage is to produce a per-pixel class prediction for each image, across the following 16 labels:

```
0 Background     4 Echinoderms       8 Unid biota    12 Mollusk
1 Bivalves       5 Foraminifera      9 Ooid          13 Ostracod
2 Micrite        6 Calcareous Algae 10 Gastropods    14 Aggregate Grain
3 Cement         7 Peloid           11 Scale Bar     15 Brachiopod
```

The segmenter is `SwinV2-Tiny (encoder) → UPerNet (decoder) → per-pixel 16-way classifier`. We now explain each piece and why this combination is well-suited to carbonate imagery.

### 2.1 The Swin Transformer backbone, in plain terms

Transformers work by *attention*: every element in an input compares itself to every other element to decide what to pay attention to. For images, the naive version of this — "every pixel attends to every other pixel" — is prohibitively expensive, because the cost grows with the square of the image size. Swin was designed to fix that while preserving the representational power of attention.

**Hierarchy of feature maps.** A petrographic image contains information at many scales: a single ooid is a few hundred pixels across, but a skeletal grain's relationship to surrounding cement and micrite spans thousands of pixels. Swin processes the image in four *stages*, each coarser than the last:

- **Stage 0** — fine resolution, few channels. Captures local texture and edges (the look of individual crystals, grain boundaries).
- **Stage 1** — half the spatial resolution, twice the channels. Captures small-object structure (a single shell fragment, a peloid).
- **Stage 2** — quarter resolution, 4× channels. Captures grain-assemblage patterns.
- **Stage 3** — eighth resolution, 8× channels. Captures global scene composition (grain-rich vs. matrix-dominated, packstone vs. wackestone).

Between stages, a *patch-merging* operation concatenates 2×2 groups of neighboring feature vectors and linearly projects them, halving spatial resolution while increasing the channel dimension. This mirrors classical CNN pyramids (like ResNet) but within a transformer architecture.

**Windowed attention.** Inside each stage, Swin divides the feature map into non-overlapping windows (e.g., 8×8 tokens) and computes self-attention *only within each window*. Cost now grows linearly with image area instead of quadratically, which is what makes Swin feasible for high-resolution petrographic tiles.

**Shifted windows.** The obvious objection is that features near a window boundary can never talk to features just outside. Swin solves this by alternating two block types: a plain windowed block (W-MSA), then a *shifted* one (SW-MSA) in which the window grid is translated diagonally by half a window before attention is computed. Because the shift moves tokens across the old boundaries, information leaks between windows over successive blocks, and after two blocks every token has had the opportunity to influence every other token in its neighborhood. The shift is implemented as a cyclic roll of the feature map with an attention mask that prevents spurious interactions between regions that were not originally adjacent — a detail worth knowing because it is why Swin's receptive field grows with depth even though each individual block is local.

**SwinV2 tweaks.** SwinV2 (the version used here) adds a few stabilizers over the original Swin: a post-norm instead of pre-norm layout, scaled cosine attention (more stable at high resolution), and a log-spaced continuous position bias that transfers better across image sizes. For our purposes these are implementation details; they make training at 512×512 more reliable.

**What comes out.** When the backbone finishes, we have four feature maps — one per stage — with progressively lower spatial resolution and progressively higher semantic abstraction. This is the "hierarchical feature pyramid" that the UPerNet decoder will consume. It is requested explicitly in our config via `out_indices=[0, 1, 2, 3]` ([swin_training_pipeline_221.py:167](swin_training_pipeline_221.py)).

### 2.2 The UPerNet decoder

A backbone alone cannot produce a segmentation mask, because its deepest feature map is much smaller than the input image (for SwinV2-Tiny at 512 input, stage-3 is 16×16). We need a *decoder* that (a) fuses information across the four stages and (b) upsamples back toward input resolution. UPerNet does both with two well-established ideas plugged together:

**(a) Pyramid Pooling Module (PPM) on the deepest stage.** PPM is applied only to stage-3 features. It pools the feature map at several bin sizes (e.g., 1×1, 2×2, 3×3, 6×6 averages), projects each pooled map back to the feature dimension, upsamples them to the stage-3 size, and concatenates with the original stage-3 features. This gives the decoder, at its deepest point, a representation that literally contains "what the whole image looks like at several different grains of summary" — pure global context. In a thin section, that is what lets the model say things like *"this is a grain-rich zone dominated by echinoderms, so an ambiguous rounded grain here is more likely a crinoid ossicle than an ooid."*

**(b) Feature Pyramid Network (FPN)-style top-down pathway with lateral connections.** Starting from the PPM output (deepest, most global), the decoder progressively upsamples and adds in information from each shallower stage:

- Lateral 1×1 convs project stages 0–2 into a common channel dimension.
- The top-down stream is upsampled ×2 and added to each lateral at the matching resolution.
- A 3×3 conv refines each fused map.

The effect is that spatially-precise but semantically-thin features from stage 0 (*"there is a sharp grain boundary here"*) get combined with spatially-coarse but semantically-rich features from stage 3 (*"this whole region is skeletal-grain packstone"*). Neither alone is enough; the fusion is what lets the model draw clean edges around objects it recognizes at the correct scale.

**(c) The segmentation head.** The fused multi-scale maps are concatenated, reduced with a final conv, and passed to a 1×1 classifier that produces a 16-channel logit volume (one channel per class). We then bilinearly upsample the logits to the input resolution and take argmax per pixel to get the final mask (`evaluate`, [swin_training_pipeline_221.py:411-437](swin_training_pipeline_221.py); `predict_image_tiled` for inputs larger than the training crop, [swin_training_pipeline_221.py:440-482](swin_training_pipeline_221.py)).

### 2.3 Why this hierarchy is well-matched to carbonate petrography

Carbonate thin sections are *simultaneously* small-scale and large-scale problems. Consider what distinguishes a few of our classes:

- **Micrite vs. cement** is a local texture call — microcrystalline lime mud versus coarser sparry calcite. The discriminative signal is at the crystal-size scale: stage-0 features.
- **Peloids vs. ooids vs. aggregate grains** is a mid-scale call about grain internal structure and cortex concentric banding. Stage-1/2 features.
- **Bivalves vs. brachiopods vs. gastropods** depends on shell microstructure *and* overall shape. The microstructure is local; the shape — "is this a curved valve, a coiled gastropod cross-section, or an articulated fragment?" — is regional. Stages 1–3.
- **Scale bar** is a global, non-geological object that never co-occurs with surrounding grains. Essentially a stage-3 pattern.

A single-scale model (either a plain CNN operating at one resolution or a plain ViT attending globally to flat patches) has to compromise. Swin's hierarchy keeps all four scales alive; UPerNet's fusion ensures the final per-pixel decision uses all of them. In practice this means the model can draw a sharp boundary around a shell fragment (fine scale) *while* correctly classifying it as bivalve vs. brachiopod (regional scale) *while* respecting that the surrounding matrix is micrite and the cement-filled void is not (global scale).

A second practical benefit: because the backbone was pretrained to reconstruct carbonate imagery, each stage's features are already tuned to the visual statistics of our domain before any masks are seen. The finetuner's job is largely to teach the decoder how to map those features onto *our* 16 classes, rather than teaching the encoder what carbonates look like from scratch.

### 2.4 Training details (finetuning)

- **Data loading.** Pairs `img/` with `masks/` by filename stem ([swin_training_pipeline_221.py:224-309](swin_training_pipeline_221.py)). Masks are single-channel integer images with values 0–15. The scale-bar class (11) can optionally be remapped to an ignore index (`--ignore_scale_bar`) so the model is not penalized or rewarded for it — useful because scale bars are an artifact of image acquisition, not geology.
- **Augmentation.** `RandomCrop(512)` + horizontal and vertical flips for training; `CenterCrop(512)` for validation. Flips are safe because carbonate thin sections have no inherent orientation.
- **Loss.** Cross-entropy per pixel, with optional class weights (`--class_weights` from a `.npy` file or `--auto_class_weights` computed inverse-frequency on the training split, [swin_training_pipeline_221.py:204-221](swin_training_pipeline_221.py)). Weighting matters because micrite and cement occupy vastly more pixels than, say, brachiopods; unweighted loss would let the model ignore rare classes.
- **Optimizer.** AdamW with weight decay; optional cosine or step LR schedule.
- **Metric.** Validation mean Intersection-over-Union (mIoU), accumulated across the entire validation set as a single confusion matrix and then averaged over classes whose union is non-zero ([swin_training_pipeline_221.py:319-350](swin_training_pipeline_221.py)). This is more honest than averaging per-image IoUs, because it correctly weights classes by their actual frequency.
- **Checkpoint policy.** Save the model with the best validation mIoU so far to `best_upernet_swinv2.pth`, along with per-class IoU. A CSV (`val_per_class_iou.csv`) tracks per-class IoU across epochs so we can see, for example, that gastropod IoU is improving even if overall mIoU is flat.
- **High-resolution inference.** For thin-section images larger than the training crop, `predict_image_tiled` runs the model on overlapping 512×512 tiles and averages the logits in overlap zones before argmax. This avoids edge artifacts and lets us segment full-slide imagery at native resolution.

### 2.5 End-to-end command examples

Stage 1 (SSL pretraining):

```bash
python code/model_training_pipeline/swin_ssl_pretrain_221.py \
  --unlabeled_root data/carbonate_imgs_and_masks \
  --epochs 100 --batch_size 8 --crop 512 --mask_ratio 0.55 \
  --output_dir runs/ssl
```

Stage 2 (supervised finetuning, transferring the SSL backbone):

```bash
python code/model_training_pipeline/swin_training_pipeline_221.py \
  --data_root data/carbonate_imgs_and_masks \
  --backbone_checkpoint runs/ssl/ssl_swinv2_best.pth \
  --epochs 20 --batch_size 2 --crop 512 \
  --auto_class_weights --ignore_scale_bar \
  --output_dir runs/finetune
```

Omitting `--backbone_checkpoint` falls back to the ImageNet-pretrained SwinV2 backbone that ships with Hugging Face, which is a reasonable baseline but gives up the domain-specific priors earned in stage 1.

---

## 3. Putting the two stages together

The mental model for a non-specialist reader is:

1. **Pretraining teaches the backbone to *see* carbonates.** Using only unlabeled images and a fill-in-the-blanks task, the encoder learns what carbonate textures, grains, and matrices look like at multiple scales. No one tells it what any of these are.
2. **Transfer copies that visual understanding into the segmenter's encoder.** Because SwinV2-Tiny is the same architecture in both stages, the learned weights drop in directly.
3. **Finetuning teaches the decoder to *name* what the encoder sees.** The UPerNet decoder, starting from scratch, learns to take the encoder's multi-scale features and assign each pixel to one of 16 categories, using a relatively small labeled dataset.
4. **The hierarchy does the heavy lifting at inference.** Each prediction fuses evidence from four scales, which is what the biology actually requires — shell microstructure and regional fabric and global composition all matter, and no single-scale model can weigh them together.

The result is a segmenter that can be trained on a modest labeled set because it does not have to learn carbonate perception from those labels — it already has it.
