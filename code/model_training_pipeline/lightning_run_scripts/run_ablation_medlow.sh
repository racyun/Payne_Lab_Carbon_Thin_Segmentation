#!/usr/bin/env bash
# Medium/low-priority augmentation ablation sweep, run sequentially with no stoppage.
#   rgb_shift (4), gamma (4), clahe (3), blur (4), noise (4), hflip, vflip, grayscale
#   settings: 20 epochs, 3-fold stratified CV, merge bivalves+mollusk+gastropods (12:1,10:1),
#             on-the-fly single augmentation on ALL_LABELS originals, cross-fold-mean W&B logging.
# Baseline is NOT repeated here (use abl_baseline from the high-priority sweep).
# Paths are hard-coded to the Studio layout so this works regardless of shell env vars.
#
# Usage (inside a tmux session):   bash code/model_training_pipeline/lightning_run_scripts/run_ablation_medlow.sh
set -u

REPO_ROOT="$HOME/Payne_Lab_Carbon_Thin_Segmentation"
IMG_DIR=/teamspace/studios/this_studio/petro_data/labeled/img
MASK_DIR=/teamspace/studios/this_studio/petro_data/labeled/masks_machine
OUT_ROOT=/teamspace/studios/this_studio/petro_data/model_outputs/ablation
SSL_CKPT=/teamspace/studios/this_studio/petro_data/model_outputs/ssl_full/ssl_swinv2_best.pth
MASTER_LOG="$OUT_ROOT/_sweep_progress_medlow.log"

mkdir -p "$OUT_ROOT"
cd "$REPO_ROOT" || { echo "repo not found: $REPO_ROOT"; exit 1; }

n_img=$(ls "$IMG_DIR" 2>/dev/null | wc -l)
if [ "$n_img" -lt 3 ]; then echo "ERROR: only $n_img images in $IMG_DIR — sync ALL_LABELS first."; exit 1; fi
if [ ! -f "$SSL_CKPT" ]; then echo "ERROR: SSL backbone missing at $SSL_CKPT — sync it first."; exit 1; fi
echo "preflight OK: $n_img images | SSL backbone present"

# "aug_type level"   (level '-' = no magnitude, e.g. flips / grayscale)
EXPERIMENTS=(
  "rgb_shift 5"
  "rgb_shift 10"
  "rgb_shift 20"
  "rgb_shift 40"
  "gamma 0.15"
  "gamma 0.30"
  "gamma 0.50"
  "gamma 0.80"
  "clahe 1.0"
  "clahe 2.0"
  "clahe 4.0"
  "blur 0.5"
  "blur 1.0"
  "blur 1.5"
  "blur 2.0"
  "noise 0.01"
  "noise 0.02"
  "noise 0.04"
  "noise 0.08"
  "hflip -"
  "vflip -"
  "grayscale -"
)

echo "=== med/low ablation sweep started $(date) | ${#EXPERIMENTS[@]} experiments ===" | tee -a "$MASTER_LOG"
i=0
for spec in "${EXPERIMENTS[@]}"; do
  i=$((i + 1))
  set -- $spec
  AUG=$1; LEVEL=$2
  if [ "$LEVEL" = "-" ]; then NAME="abl_${AUG}"; LEVEL_ARG=""; else NAME="abl_${AUG}_${LEVEL}"; LEVEL_ARG="--aug_level $LEVEL"; fi
  OUT="$OUT_ROOT/$NAME"; mkdir -p "$OUT"

  echo ">>> [$(date +%H:%M)] ($i/${#EXPERIMENTS[@]}) START $NAME" | tee -a "$MASTER_LOG"
  WANDB_DIR="$OUT" python -u code/model_training_pipeline/swin_training_pipeline_221.py \
    --img_dir "$IMG_DIR" --mask_dir "$MASK_DIR" \
    --n_folds 3 --cv_strategy stratified \
    --merge_class_map 12:1,10:1 \
    --ablation_aug "$AUG" $LEVEL_ARG --aug_frac 0.2 \
    --epochs 20 --batch_size 2 --crop 512 --num_workers 4 --lr 3e-4 \
    --warmup_epochs 5 --scheduler cosine \
    --ignore_artifacts \
    --loss_type focal --focal_gamma 2.0 \
    --backbone_checkpoint "$SSL_CKPT" \
    --output_dir "$OUT" \
    --wandb_mean_only \
    --wandb_project payne-carbonate-segmentation \
    --wandb_run_name "$NAME" \
    > "$OUT/run.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "<<< [$(date +%H:%M)] ($i/${#EXPERIMENTS[@]}) DONE   $NAME" | tee -a "$MASTER_LOG"
  else
    echo "!!! [$(date +%H:%M)] ($i/${#EXPERIMENTS[@]}) FAILED $NAME (rc=$rc) — see $OUT/run.log" | tee -a "$MASTER_LOG"
  fi
done
echo "=== med/low ablation sweep COMPLETE $(date) ===" | tee -a "$MASTER_LOG"
