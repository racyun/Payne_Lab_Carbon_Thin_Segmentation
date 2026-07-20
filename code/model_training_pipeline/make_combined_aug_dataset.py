# -*- coding: utf-8 -*-
"""
Build a pre-augmented labeled dataset = all originals + N augmented copies from each of the
top-4 ablation augmentations (affine 30, rgb_shift 10, clahe 1.0, gamma 0.50).

Augmented files are named "<stem>_aug<k>.png" so the training pipeline's --group_by_stem /
--group_pattern '_aug\\d+$' keeps them TRAINING-ONLY and validates on clean originals.
Geometric aug (affine) transforms the mask too (nearest, fill=ignore); photometric augs
(rgb_shift, clahe, gamma) copy the original mask unchanged. Augs are applied to the FULL image,
so the training pipeline's 512 random crops still see full-resolution augmented context.

Run on the Studio:
    python code/model_training_pipeline/make_combined_aug_dataset.py \
      --img_dir  /teamspace/studios/this_studio/petro_data/labeled/img \
      --mask_dir /teamspace/studios/this_studio/petro_data/labeled/masks_machine \
      --out_dir  /teamspace/studios/this_studio/petro_data/labeled_aug_top4 \
      --n_per_aug 36
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import torch
from torchvision.io import read_image, write_png
from torchvision.transforms.v2 import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppl_augment as P

# The four top-ranked ablation augmentations (by mean mIoU) and their levels.
AUGS = [("affine", 30), ("rgb_shift", 10), ("clahe", 1.0), ("gamma", 0.50)]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_img(p: Path) -> torch.Tensor:
    img = read_image(str(p))
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3]
    return img.to(torch.uint8)


def load_mask(p: Path) -> torch.Tensor:
    m = read_image(str(p))
    if m.ndim == 3 and m.shape[0] > 1:
        m = m[0:1, ...]
    return m.to(torch.uint8)  # [1, H, W]


def apply_aug(aug_type: str, level, img: torch.Tensor, mask1: torch.Tensor):
    """Return (aug_img[3,H,W] uint8, aug_mask[1,H,W] uint8)."""
    cfg = P.single_aug_cfg(aug_type, level, frac=1.0, crop=512)
    if aug_type == "affine":
        out_img, out_mask, _ = P.affine(img, mask1, cfg)
        return out_img.to(torch.uint8), out_mask.to(torch.uint8)
    if aug_type == "rgb_shift":
        return P.rgb_shift(img, cfg)[0].to(torch.uint8), mask1
    if aug_type == "clahe":
        return P.clahe(img, cfg).to(torch.uint8), mask1
    if aug_type == "gamma":
        lo, hi = cfg["gamma_range"]
        return TF.adjust_gamma(img, random.uniform(lo, hi)).to(torch.uint8), mask1
    raise ValueError(f"unsupported aug_type {aug_type!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_aug", type=int, default=36, help="augmented images per aug type.")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    img_dir, mask_dir = Path(args.img_dir), Path(args.mask_dir)
    out = Path(args.out_dir)
    out_img, out_mask = out / "img", out / "masks_machine"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    imgs = {p.stem: p for p in img_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")}
    masks = {p.stem: p for p in mask_dir.iterdir()
             if p.suffix.lower() == ".png" and not p.name.startswith(".")}
    stems = sorted(imgs.keys() & masks.keys())
    if not stems:
        raise SystemExit("No paired (image, mask) originals found.")
    print(f"[gen] {len(stems)} paired originals")

    # 1) copy every original verbatim (these are the clean images CV validates on)
    for s in stems:
        shutil.copy(imgs[s], out_img / imgs[s].name)
        shutil.copy(masks[s], out_mask / masks[s].name)

    # 2) N augmented copies per aug type, named <stem>_aug<k> for group-by-stem
    counter = {s: 0 for s in stems}
    made = 0
    for aug_type, level in AUGS:
        chosen = random.sample(stems, min(args.n_per_aug, len(stems)))
        for s in chosen:
            img, mask1 = load_img(imgs[s]), load_mask(masks[s])
            aug_img, aug_mask = apply_aug(aug_type, level, img, mask1)
            counter[s] += 1
            name = f"{s}_aug{counter[s]}"
            write_png(aug_img, str(out_img / f"{name}.png"))
            write_png(aug_mask.to(torch.uint8), str(out_mask / f"{name}.png"))
            made += 1
        print(f"[gen]   {aug_type} {level}: wrote {len(chosen)} augmented images")

    total = len(stems) + made
    print(f"[gen] DONE: {len(stems)} originals + {made} augmented = {total} images "
          f"({made / total * 100:.0f}% augmented) in {out}")


if __name__ == "__main__":
    main()
