# -*- coding: utf-8 -*-
"""
PPL (plane-polarized) data-augmentation library for carbonate thin-section segmentation.

Phase 1 per-sample augmentations from AUGMENTATION_PLAN.md, built on
torchvision ``transforms.v2.functional`` + cv2 (CLAHE). Single source of truth
imported by both the augmentation-preview and review-dashboard notebooks (and,
eventually, the training pipeline).

Conventions:
  - geometric ops transform image (bilinear, fill=0) AND mask (nearest, fill=IGNORE_INDEX)
  - photometric ops touch the IMAGE ONLY (mask passes through unchanged)
  - everything operates on uint8 CHW images, pre-normalization
  - ``augment(..., return_ops=True)`` also returns the list of augmentations that fired
"""

from __future__ import annotations

import random

import torch
import cv2
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2 import InterpolationMode

from swin_training_pipeline_221 import CLASS_NAMES, IGNORE_INDEX


DEFAULT_AUG = dict(
    crop=512,
    # rare-class-aware cropping: with prob rare_crop_prob, center the crop on a
    # randomly chosen rare-class pixel so small/rare grains actually appear.
    rare_crop_prob=0.5,
    rare_class_ids=(4, 6, 10, 12, 13, 14, 15, 16),
    hflip_p=0.5, vflip_p=0.2,
    affine_p=0.7, affine_deg=15.0, affine_translate=0.05, affine_scale=(0.85, 1.15), affine_shear=8.0,
    # photometric (pushed harder because PPL color is non-diagnostic)
    color_jitter_p=0.9, brightness=0.30, contrast=0.30, saturation=0.40, hue=0.08,
    rgbshift_p=0.5, rgbshift_max=20,
    clahe_p=0.3, clahe_clip=2.0, clahe_grid=8,
    gamma_p=0.4, gamma_range=(0.7, 1.5),
    blur_p=0.2, blur_sigma=(0.1, 1.5),
    noise_p=0.3, noise_sigma=0.04,
    grayscale_p=0.10,
)


def read_raw_image(path):
    """uint8 [3,H,W] RGB."""
    im = read_image(str(path))
    if im.shape[0] == 1:
        im = im.repeat(3, 1, 1)
    elif im.shape[0] == 4:
        im = im[:3]
    return im.to(torch.uint8)


def read_raw_mask(path):
    """long [H,W] of class ids (0..17, plus 255 ignore)."""
    m = read_image(str(path))
    if m.ndim == 3 and m.shape[0] > 1:
        m = m[0:1, ...]
    return m.squeeze(0).to(torch.long)


def _u8(t):
    return t.clamp(0, 255).to(torch.uint8)


# ----- geometric (image + mask jointly) -----
def rare_class_aware_crop(img, mask_c, cfg):
    _, H, W = img.shape
    crop = cfg["crop"]
    ch, cw = min(crop, H), min(crop, W)
    top = left = None
    label = "crop:random"
    rare = [c for c in cfg["rare_class_ids"] if bool((mask_c == c).any())]
    if rare and random.random() < cfg["rare_crop_prob"]:
        cid = random.choice(rare)
        ys, xs = torch.where(mask_c[0] == cid)
        j = random.randrange(len(ys))
        cy, cx = int(ys[j]), int(xs[j])
        top = min(max(cy - ch // 2, 0), max(H - ch, 0))
        left = min(max(cx - cw // 2, 0), max(W - cw, 0))
        label = f"crop:rare({CLASS_NAMES[cid]})"
    if top is None:
        top = random.randint(0, max(H - ch, 0))
        left = random.randint(0, max(W - cw, 0))
    img = TF.crop(img, top, left, ch, cw)
    mask_c = TF.crop(mask_c, top, left, ch, cw)
    if (ch, cw) != (crop, crop):  # smaller-than-crop image: resize up to crop
        img = TF.resize(img, [crop, crop], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_c = TF.resize(mask_c, [crop, crop], interpolation=InterpolationMode.NEAREST)
        label += "+resize"
    return img, mask_c, label


def affine(img, mask_c, cfg):
    angle = random.uniform(-cfg["affine_deg"], cfg["affine_deg"])
    tx = int(random.uniform(-cfg["affine_translate"], cfg["affine_translate"]) * img.shape[-1])
    ty = int(random.uniform(-cfg["affine_translate"], cfg["affine_translate"]) * img.shape[-2])
    sc = random.uniform(*cfg["affine_scale"])
    sh = random.uniform(-cfg["affine_shear"], cfg["affine_shear"])
    img = TF.affine(img, angle=angle, translate=[tx, ty], scale=sc, shear=[sh],
                    interpolation=InterpolationMode.BILINEAR, fill=0)
    mask_c = TF.affine(mask_c, angle=angle, translate=[tx, ty], scale=sc, shear=[sh],
                       interpolation=InterpolationMode.NEAREST, fill=IGNORE_INDEX)
    label = f"affine(rot={angle:+.0f}deg, scale={sc:.2f}, shear={sh:+.0f}deg)"
    return img, mask_c, label


# ----- photometric (image only) -----
def color_jitter(img, cfg):
    b, c, s, h = cfg["brightness"], cfg["contrast"], cfg["saturation"], cfg["hue"]
    ops = [
        lambda x: TF.adjust_brightness(x, 1 + random.uniform(-b, b)),
        lambda x: TF.adjust_contrast(x, 1 + random.uniform(-c, c)),
        lambda x: TF.adjust_saturation(x, 1 + random.uniform(-s, s)),
        lambda x: TF.adjust_hue(x, random.uniform(-h, h)),
    ]
    random.shuffle(ops)
    for op in ops:
        img = op(img)
    return img


def rgb_shift(img, cfg):
    m = cfg["rgbshift_max"]
    shift = torch.randint(-m, m + 1, (3, 1, 1))
    s = shift.flatten().tolist()
    return _u8(img.to(torch.int16) + shift), f"rgb_shift({s[0]:+d},{s[1]:+d},{s[2]:+d})"


def clahe(img, cfg):
    hwc = img.permute(1, 2, 0).cpu().numpy()                       # HWC uint8 RGB
    lab = cv2.cvtColor(hwc, cv2.COLOR_RGB2LAB)
    cla = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(cfg["clahe_grid"], cfg["clahe_grid"]))
    lab[:, :, 0] = cla.apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def gaussian_noise(img, cfg):
    noise = torch.randn(img.shape, dtype=torch.float32) * (cfg["noise_sigma"] * 255.0)
    return _u8(img.to(torch.float32) + noise)


def augment(img, mask, cfg=DEFAULT_AUG, return_ops=False):
    """img: uint8 [3,H,W]; mask: long [H,W].

    Returns (img_u8, mask_long), or (img_u8, mask_long, applied_ops) when
    return_ops=True. ``applied_ops`` is the list of augmentation labels that
    actually fired this call (the crop always runs).
    """
    mask_c = mask.unsqueeze(0)
    applied = []

    # --- geometric (image + mask) ---
    img, mask_c, lbl = rare_class_aware_crop(img, mask_c, cfg)
    applied.append(lbl)
    if random.random() < cfg["hflip_p"]:
        img, mask_c = TF.hflip(img), TF.hflip(mask_c)
        applied.append("hflip")
    if random.random() < cfg["vflip_p"]:
        img, mask_c = TF.vflip(img), TF.vflip(mask_c)
        applied.append("vflip")
    if random.random() < cfg["affine_p"]:
        img, mask_c, lbl = affine(img, mask_c, cfg)
        applied.append(lbl)

    # --- photometric (image only) ---
    if random.random() < cfg["color_jitter_p"]:
        img = color_jitter(img, cfg)
        applied.append("color_jitter")
    if random.random() < cfg["rgbshift_p"]:
        img, lbl = rgb_shift(img, cfg)
        applied.append(lbl)
    if random.random() < cfg["clahe_p"]:
        img = clahe(img, cfg)
        applied.append("clahe")
    if random.random() < cfg["gamma_p"]:
        g = random.uniform(*cfg["gamma_range"])
        img = TF.adjust_gamma(img, g)
        applied.append(f"gamma({g:.2f})")
    if random.random() < cfg["blur_p"]:
        s = random.uniform(*cfg["blur_sigma"])
        img = TF.gaussian_blur(img, kernel_size=5, sigma=s)
        applied.append(f"blur(sigma={s:.1f})")
    if random.random() < cfg["noise_p"]:
        img = gaussian_noise(img, cfg)
        applied.append("noise")
    if random.random() < cfg["grayscale_p"]:
        img = TF.rgb_to_grayscale(img, num_output_channels=3)
        applied.append("grayscale")

    img, mask = _u8(img), mask_c.squeeze(0)
    if return_ops:
        return img, mask, applied
    return img, mask
