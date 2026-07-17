# -*- coding: utf-8 -*-
"""
Render GLNet 4-panel prediction figures (Original | Ground Truth Mask | Prediction | Overlay)
from an ALREADY-TRAINED checkpoint — for runs started before per-fold viz existed.

It rebuilds the exact validation split (so the panels are the fold's val images), loads the
checkpoint, and reuses the pipeline's whole-image tiled prediction + renderer. Output matches
the single-path pipeline's look and lands in <checkpoint_dir>/prediction_viz/ by default.

IMPORTANT: pass the SAME split-determining flags you used for the run (--n_folds, --cv_strategy,
--seed, --merge_class_map, --ignore_artifacts), or the val split won't match. global_hw / local_crop
/ fold are read from the checkpoint automatically.

Example (one fold):
    python code/model_training_pipeline/glnet_viz_from_checkpoint.py \
      --checkpoint /teamspace/.../model_outputs/glnet_g1024_l512/fold_0/best_glnet.pth \
      --img_dir  /teamspace/studios/this_studio/petro_data/labeled/img \
      --mask_dir /teamspace/studios/this_studio/petro_data/labeled/masks_machine \
      --n_folds 3 --cv_strategy stratified --seed 1337 \
      --merge_class_map 12:1,10:1 --ignore_artifacts

All folds (bash loop):
    for f in 0 1 2; do
      python code/model_training_pipeline/glnet_viz_from_checkpoint.py \
        --checkpoint /teamspace/.../glnet_g1024_l512/fold_$f/best_glnet.pth \
        --img_dir .../img --mask_dir .../masks_machine \
        --n_folds 3 --cv_strategy stratified --seed 1337 \
        --merge_class_map 12:1,10:1 --ignore_artifacts
    done
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from glnet_model_221 import build_glnet
from glnet_pipeline_221 import GlnetDataset, render_predictions_glnet
from swin_training_pipeline_221 import (
    BACKBONE_ID,
    IGNORE_INDEX,
    NUM_CLASSES,
    artifact_ignore_ids,
    augment_aware_kfold_indices,
    build_class_presence_matrix,
    kfold_train_val_indices,
    parse_merge_map,
    set_seed,
    stratified_kfold_indices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render GLNet prediction panels from a trained checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to a fold's best_glnet.pth.")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--out", default=None, help="Output dir (default: <checkpoint_dir>/prediction_viz).")
    # Split-determining flags — MUST match the training run.
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--cv_strategy", type=str, default="stratified", choices=["stratified", "random"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--merge_class_map", type=str, default=None, help="e.g. '12:1,10:1'.")
    p.add_argument("--ignore_artifacts", action="store_true")
    p.add_argument("--ignore_scale_bar", action="store_true")
    p.add_argument("--group_by_stem", action="store_true")
    p.add_argument("--group_pattern", type=str, default=r"_aug\d+$")
    p.add_argument("--num_augmentations_per_img", type=int, default=None)
    # Overrides (default: read from checkpoint).
    p.add_argument("--fold", type=int, default=None, help="Override fold (default: from checkpoint).")
    p.add_argument("--global_h", type=int, default=None)
    p.add_argument("--global_w", type=int, default=None)
    p.add_argument("--local_crop", type=int, default=None)
    p.add_argument("--backbone_id", type=str, default=BACKBONE_ID)
    p.add_argument("--eval_stride", type=int, default=None, help="Tile stride (default = local_crop).")
    p.add_argument("--eval_tile_batch", type=int, default=4)
    p.add_argument("--viz_samples", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    fold = args.fold if args.fold is not None else int(ckpt.get("fold", 0))
    gh = args.global_h or int(ckpt["global_hw"][0])
    gw = args.global_w or int(ckpt["global_hw"][1])
    local_crop = args.local_crop or int(ckpt["local_crop"])
    num_classes = int(ckpt.get("num_classes", NUM_CLASSES))
    eval_stride = args.eval_stride if args.eval_stride else local_crop
    print(f"[viz] checkpoint fold={fold} global={gh}x{gw} local={local_crop} classes={num_classes}")

    ignore_ids = artifact_ignore_ids(args)
    merge_map = parse_merge_map(args.merge_class_map)

    # Rebuild the SAME CV split the run used, then take this fold's val indices.
    probe = GlnetDataset(args.img_dir, args.mask_dir, (gh, gw), local_crop, ignore_ids, merge_map,
                         train=True, print_pair_count=True)
    n = len(probe)
    presence = None
    if args.cv_strategy == "stratified":
        presence = build_class_presence_matrix(probe.pairs, NUM_CLASSES, IGNORE_INDEX, ignore_ids, merge_map)
    if args.group_by_stem:
        splits = augment_aware_kfold_indices(probe.pairs, args.group_pattern, presence, args.n_folds,
                                             args.seed, max_augs_per_source=args.num_augmentations_per_img)
    elif args.cv_strategy == "stratified":
        splits = stratified_kfold_indices(presence, args.n_folds, args.seed)
    else:
        splits = kfold_train_val_indices(n, args.n_folds, args.seed)
    val_idx = splits[fold][1].tolist()
    print(f"[viz] fold {fold}: {len(val_idx)} val images")

    val_full = GlnetDataset(args.img_dir, args.mask_dir, (gh, gw), local_crop, ignore_ids, merge_map,
                            train=False)
    model = build_glnet(num_classes, IGNORE_INDEX, args.backbone_id).to(device)
    model.load_state_dict(ckpt["model_state"])

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent / "prediction_viz"
    render_predictions_glnet(model, val_full, val_idx, device, local_crop, eval_stride,
                             args.eval_tile_batch, out_dir, args.viz_samples)


if __name__ == "__main__":
    main()
