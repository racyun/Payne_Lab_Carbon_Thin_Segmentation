# -*- coding: utf-8 -*-
"""
Visualize what each augmentation (and each magnitude) does, using the SAME ppl_augment code
the ablation study trained with. Produces one figure per augmentation family: rows = magnitude
levels, columns = [original crop, example 1, example 2].

Run on the Studio:
    python code/model_training_pipeline/visualize_augmentations.py \
      --img_dir /teamspace/studios/this_studio/petro_data/labeled/img \
      --mask_dir /teamspace/studios/this_studio/petro_data/labeled/masks_machine \
      --out /teamspace/studios/this_studio/petro_data/aug_viz
Then download the PNGs in --out.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torchvision.io import read_image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as TF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppl_augment as P

# Same families/levels as the ablation sweep.
FAMILIES = {
    "rgb_shift":    [5, 10, 20, 40],
    "gamma":        [0.15, 0.30, 0.50, 0.80],
    "clahe":        [1.0, 2.0, 4.0],
    "noise":        [0.01, 0.02, 0.04, 0.08],
    "blur":         [0.5, 1.0, 1.5, 2.0],
    "affine":       [5, 15, 30, 45],
    "color_jitter": ["light", "medium", "strong", "very_strong"],
    "rare_crop":    [0.0, 0.25, 0.5, 0.75, 1.0],
    "hflip":        [None],
    "vflip":        [None],
    "grayscale":    [None],
}


def load_base(img_path: Path, mask_path: Path, crop: int = 512):
    img = read_image(str(img_path))
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.shape[0] == 4:
        img = img[:3]
    img = img.to(torch.uint8)
    mask = read_image(str(mask_path))
    if mask.ndim == 3 and mask.shape[0] > 1:
        mask = mask[0:1]
    mask = mask.squeeze(0).to(torch.long)

    _, H, W = img.shape
    scale = crop / min(H, W)
    if scale > 1:  # upsize so the short side >= crop, then center-crop a full tile
        nh, nw = int(H * scale) + 1, int(W * scale) + 1
        img = TF.resize(img, [nh, nw], antialias=True).to(torch.uint8)
        mask = TF.resize(mask.unsqueeze(0), [nh, nw], interpolation=InterpolationMode.NEAREST).squeeze(0)
    img = TF.center_crop(img, [crop, crop])
    mask = TF.center_crop(mask.unsqueeze(0), [crop, crop]).squeeze(0)
    return img, mask


def apply_one(img_u8: torch.Tensor, mask_long: torch.Tensor, aug_type: str, level):
    """Apply exactly one augmentation to a fixed crop (so only the aug differs from the original)."""
    cfg = P.single_aug_cfg(aug_type, level, frac=1.0, crop=int(img_u8.shape[-1]))
    img = img_u8.clone()
    mask_c = mask_long.clone().unsqueeze(0)
    if aug_type == "rare_crop":
        out, _, _ = P.rare_class_aware_crop(img, mask_c, cfg)
        return out
    if aug_type == "affine":
        out, _, _ = P.affine(img, mask_c, cfg)
        return out
    if aug_type == "hflip":
        return TF.hflip(img)
    if aug_type == "vflip":
        return TF.vflip(img)
    if aug_type == "color_jitter":
        return P.color_jitter(img, cfg)
    if aug_type == "rgb_shift":
        return P.rgb_shift(img, cfg)[0]
    if aug_type == "clahe":
        return P.clahe(img, cfg)
    if aug_type == "gamma":
        w = float(level)
        return TF.adjust_gamma(img, random.uniform(max(0.05, 1 - w), 1 + w))
    if aug_type == "blur":
        return TF.gaussian_blur(img, kernel_size=5, sigma=random.uniform(0.1, float(level)))
    if aug_type == "noise":
        return P.gaussian_noise(img, cfg)
    if aug_type == "grayscale":
        return TF.rgb_to_grayscale(img, num_output_channels=3)
    return img


def chw_to_hwc(t: torch.Tensor):
    return t.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    ap.add_argument("--img", default=None, help="Specific image filename to use (default: first found).")
    ap.add_argument("--out", default="./aug_viz")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    img_dir, mask_dir = Path(args.img_dir), Path(args.mask_dir)
    exts = {".jpg", ".jpeg", ".png"}
    if args.img:
        img_path = img_dir / args.img
    else:
        cands = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts and not p.name.startswith("."))
        img_path = cands[0]
    mask_path = mask_dir / (img_path.stem + ".png")
    if not mask_path.exists():
        raise SystemExit(f"No mask for {img_path.name} at {mask_path}")
    print(f"Using image: {img_path.name}")

    base_img, base_mask = load_base(img_path, mask_path, args.crop)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fam, levels in FAMILIES.items():
        nrows = len(levels)
        fig, axes = plt.subplots(nrows, 3, figsize=(9, 3 * nrows), squeeze=False)
        for r, level in enumerate(levels):
            title = fam if level is None else f"{fam} = {level}"
            panels = [("original", base_img),
                      ("example 1", apply_one(base_img, base_mask, fam, level)),
                      ("example 2", apply_one(base_img, base_mask, fam, level))]
            for c, (lbl, im) in enumerate(panels):
                ax = axes[r][c]
                ax.imshow(chw_to_hwc(im))
                ax.set_xticks([]); ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel(title, fontsize=11)
                if r == 0:
                    ax.set_title(lbl, fontsize=11)
        fig.suptitle(f"Augmentation: {fam}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = out / f"{fam}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)

    print(f"\nDone. {len(FAMILIES)} figures in {out}")


if __name__ == "__main__":
    main()
