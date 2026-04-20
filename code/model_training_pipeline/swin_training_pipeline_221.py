# -*- coding: utf-8 -*-
"""
SwinV2-Tiny (ImageNet) + UPerNet semantic segmentation for carbonate thin sections.

Pairs images under ``<data_root>/img`` with masks under ``<data_root>/masks`` (same stem).
Default: 16 classes (labels 0–15). Trains with AdamW; saves best checkpoint by validation mIoU.

Run locally::

    python swin_training_pipeline_221.py --data_root /path/to/carbonate_imgs_and_masks

Requires: torch, torchvision, transformers, tqdm, pillow, numpy.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.v2 import functional as V2F
from tqdm.auto import tqdm

from transformers import UperNetConfig, UperNetForSemanticSegmentation

# ---------------------------------------------------------------------------
# Constants (carbonate labeling: 0–15 => 16 logits)
# ---------------------------------------------------------------------------

NUM_CLASSES = 16
IGNORE_INDEX = 255
SCALE_BAR_CLASS_ID = 11
BACKBONE_ID = "microsoft/swinv2-tiny-patch4-window8-256"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UPerNet + SwinV2 on carbonate segmentation masks.")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root folder containing ``img/`` and ``masks/``. Default: repo data path when unset.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--crop", type=int, default=512, help="Train random crop and val center crop size.")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=".", help="Directory for checkpoints and viz.")
    p.add_argument("--no_train", action="store_true", help="Only build data loaders and exit (smoke test).")
    p.add_argument(
        "--ignore_scale_bar",
        action="store_true",
        help=f"Remap mask class {SCALE_BAR_CLASS_ID} (scale bar) to ignore_index so it is not learned.",
    )
    p.add_argument(
        "--class_weights",
        type=str,
        default=None,
        help="Optional path to a .npy vector of shape (num_classes,) for weighted cross-entropy "
        "(training uses manual loss when set).",
    )
    p.add_argument("--tile_size", type=int, default=512, help="Tile edge length for ``predict_image_tiled``.")
    p.add_argument("--tile_stride", type=int, default=None, help="Stride for tiling; default = tile_size // 2.")
    p.add_argument("--no_viz", action="store_true", help="Skip matplotlib overlay at end of training.")
    return p.parse_args()


def default_data_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent.parent / "data" / "carbonate_imgs_and_masks"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_semantic_segmentation(
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
    backbone_id: str = BACKBONE_ID,
) -> UperNetForSemanticSegmentation:
    cfg = UperNetConfig(
        backbone=backbone_id,
        use_pretrained_backbone=True,
        backbone_kwargs={"out_indices": [0, 1, 2, 3]},
        num_labels=num_classes,
        loss_ignore_index=ignore_index,
        use_auxiliary_head=False,
    )
    return UperNetForSemanticSegmentation(cfg)


class CarbonateSegmentationDataset(torch.utils.data.Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    MASK_EXTS = {".png"}

    def __init__(
        self,
        root: str | Path,
        transforms=None,
        normalize: bool = True,
        strict: bool = True,
        ignore_scale_bar: bool = False,
        ignore_index: int = IGNORE_INDEX,
        print_pair_count: bool = True,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.normalize = normalize
        self.ignore_scale_bar = ignore_scale_bar
        self.ignore_index = ignore_index
        self._print_pair_count = print_pair_count

        img_dir = self.root / "img"
        mask_dir = self.root / "masks"

        imgs = {
            p.stem: p
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMG_EXTS and not p.name.startswith(".")
        }
        masks = {
            p.stem: p
            for p in mask_dir.iterdir()
            if p.is_file() and p.suffix.lower() in self.MASK_EXTS and not p.name.startswith(".")
        }

        common = sorted(imgs.keys() & masks.keys())
        self.pairs = [(imgs[k], masks[k]) for k in common]

        missing_masks = sorted(imgs.keys() - masks.keys())
        missing_imgs = sorted(masks.keys() - imgs.keys())
        if strict:
            if missing_masks:
                print(f"[dataset] {len(missing_masks)} images have no mask. Examples:", missing_masks[:5])
            if missing_imgs:
                print(f"[dataset] {len(missing_imgs)} masks have no image. Examples:", missing_imgs[:5])
            if not self.pairs:
                raise RuntimeError("Found no (image, mask) pairs. Check folder names & extensions.")
        if self._print_pair_count:
            print(f"[dataset] Using {len(self.pairs)} paired samples.")

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img = read_image(str(img_path))
        mask = read_image(str(mask_path))

        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0:1, ...]

        mask = mask.squeeze(0).to(torch.long)

        if self.ignore_scale_bar:
            mask = mask.clone()
            mask[mask == SCALE_BAR_CLASS_ID] = self.ignore_index

        img = tv_tensors.Image(img)
        sem_mask = tv_tensors.Mask(mask)

        if self.transforms is not None:
            img, sem_mask = self.transforms(img, sem_mask)

        img_f = V2F.convert_image_dtype(img, dtype=torch.float32)
        if img_f.shape[0] == 1:
            img_f = img_f.repeat(3, 1, 1)

        if self.normalize:
            img_f = V2F.normalize(img_f, mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

        labels = torch.as_tensor(sem_mask, dtype=torch.long)
        return img_f, labels

    def __len__(self):
        return len(self.pairs)


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> float:
    valid = target != ignore_index
    correct = (pred == target) & valid
    denom = int(valid.sum().item())
    return float(correct.sum().item() / max(1, denom))


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """Single-image or batched [B,H,W] pred/target -> [K,K] float confusion matrix."""
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    if target.numel() == 0:
        return torch.zeros(num_classes, num_classes, device=pred.device, dtype=torch.float32)
    k = (target * num_classes + pred).to(torch.int64)
    return torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()


def miou_from_confusion(cm: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Mean IoU over classes with union > 0 (full accumulated matrix). Returns (mIoU, per_class_iou)."""
    num_classes = cm.shape[0]
    diag = torch.diag(cm)
    union = cm.sum(1) + cm.sum(0) - diag
    iou = torch.where(union > 0, diag / torch.clamp(union, min=1.0), torch.full_like(union, float("nan")))
    valid = union > 0
    miou = float(torch.nanmean(iou).item()) if bool(valid.any().item()) else 0.0
    return miou, iou


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch: int,
    num_classes: int,
    ignore_index: int,
    class_weights: torch.Tensor | None,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"epoch {epoch:02d}", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if class_weights is None:
            out = model(pixel_values=imgs, labels=labels)
            loss = out.loss
        else:
            out = model(pixel_values=imgs)
            logits = out.logits
            logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total += float(loss.item()) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total/max(1,n):.4f}")

    print(f"epoch {epoch:02d} | train_avg_loss {total/max(1,n):.4f}")
    return total / max(1, n)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes: int,
    ignore_index: int,
    class_weights: torch.Tensor | None,
) -> tuple[float, float, float, torch.Tensor]:
    """Returns val_loss, pixel_acc, mIoU (from full-val confusion matrix), per-class IoU vector."""
    model.eval()
    total, n = 0.0, 0
    acc_sum, m = 0.0, 0
    cm_total = torch.zeros(num_classes, num_classes, device=device, dtype=torch.float32)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if class_weights is None:
            out = model(pixel_values=imgs, labels=labels)
            loss = out.loss
            logits = out.logits
        else:
            out = model(pixel_values=imgs)
            logits = out.logits
            logits_up = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = F.cross_entropy(logits_up, labels, weight=class_weights, ignore_index=ignore_index)

        logits = F.interpolate(out.logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        preds = logits.argmax(dim=1)

        acc_sum += pixel_accuracy(preds, labels, ignore_index)
        cm_total += confusion_matrix(preds, labels, num_classes, ignore_index)

        bs = imgs.size(0)
        total += float(loss.item()) * bs
        n += bs
        m += 1

    miou, per_class = miou_from_confusion(cm_total)
    return total / max(1, n), acc_sum / max(1, m), miou, per_class


@torch.no_grad()
def predict_image_tiled(
    model: UperNetForSemanticSegmentation,
    pixel_values: torch.Tensor,
    device: torch.device,
    tile_size: int,
    tile_stride: int | None = None,
) -> torch.Tensor:
    """
    Full-resolution prediction by overlapping tiles (logits averaged).

    ``pixel_values``: [1, 3, H, W] in the same normalization as training.
    Returns: [H, W] int64 class map at input resolution.
    """
    if tile_stride is None:
        tile_stride = max(1, tile_size // 2)

    model.eval()
    _, _, h, w = pixel_values.shape
    pixel_values = pixel_values.to(device)
    num_classes = model.config.num_labels

    logits_acc = torch.zeros(1, num_classes, h, w, device=device, dtype=torch.float32)
    weight = torch.zeros(1, 1, h, w, device=device, dtype=torch.float32)

    for y0 in range(0, max(1, h - tile_size + 1), tile_stride):
        for x0 in range(0, max(1, w - tile_size + 1), tile_stride):
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            if y1 - y0 < tile_size or x1 - x0 < tile_size:
                y0n = max(0, y1 - tile_size)
                x0n = max(0, x1 - tile_size)
            else:
                y0n, x0n = y0, x0
            patch = pixel_values[:, :, y0n:y1, x0n:x1]
            out = model(pixel_values=patch)
            ph, pw = patch.shape[-2:]
            up = F.interpolate(out.logits, size=(ph, pw), mode="bilinear", align_corners=False)
            logits_acc[:, :, y0n:y1, x0n:x1] += up
            weight[:, :, y0n:y1, x0n:x1] += 1.0

    logits_acc = logits_acc / torch.clamp(weight, min=1.0)
    return logits_acc.argmax(dim=1)[0].cpu()


def denorm_to_uint8(img_3chw: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    x = img_3chw.detach().cpu().numpy().astype(np.float32)
    x = (x * std + mean) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return np.transpose(x, (1, 2, 0))


def colorize_mask(mask_np: np.ndarray, palette: list | None = None) -> Image.Image:
    if palette is None:
        rng = np.random.default_rng(123)
        k_max = int(mask_np.max()) + 1
        palette = [
            (int(rng.integers(0, 256)), int(rng.integers(0, 256)), int(rng.integers(0, 256)))
            for _ in range(k_max)
        ]
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    for k, (r, g, b) in enumerate(palette):
        rgb[mask_np == k] = (r, g, b)
    return Image.fromarray(rgb)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.data_root is None:
        data_root = default_data_root()
        if not (data_root / "img").is_dir():
            raise SystemExit(
                f"No --data_root given and default {data_root} is missing ``img/``. Pass --data_root explicitly."
            )
        print(f"[config] Using default data_root: {data_root}")
    else:
        data_root = Path(args.data_root)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = args.crop
    train_transforms = Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.2),
            RandomCrop((crop, crop)),
        ]
    )
    val_transforms = Compose([CenterCrop((crop, crop))])

    probe = CarbonateSegmentationDataset(
        data_root,
        transforms=None,
        normalize=True,
        strict=True,
        ignore_scale_bar=args.ignore_scale_bar,
    )
    n = len(probe)

    val_frac = args.val_frac
    n_val = max(1, int(val_frac * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    train_full = CarbonateSegmentationDataset(
        data_root,
        transforms=train_transforms,
        normalize=True,
        strict=False,
        ignore_scale_bar=args.ignore_scale_bar,
        print_pair_count=False,
    )
    val_full = CarbonateSegmentationDataset(
        data_root,
        transforms=val_transforms,
        normalize=True,
        strict=False,
        ignore_scale_bar=args.ignore_scale_bar,
        print_pair_count=False,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if on_gpu else "cpu")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=on_gpu,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=on_gpu,
    )

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if args.no_train:
        print("[config] --no_train set; exiting after dataloader smoke test.")
        return

    class_weights: torch.Tensor | None = None
    if args.class_weights:
        w = np.load(args.class_weights).astype(np.float32)
        if w.shape != (NUM_CLASSES,):
            raise ValueError(f"class_weights .npy must have shape ({NUM_CLASSES},), got {w.shape}")
        class_weights = torch.tensor(w, device=device)

    model = get_model_semantic_segmentation(NUM_CLASSES, IGNORE_INDEX, BACKBONE_ID)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_miou = -1.0
    ckpt_path = out_dir / "best_upernet_swinv2.pth"

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            NUM_CLASSES,
            IGNORE_INDEX,
            class_weights,
        )
        va_loss, va_acc, va_miou, per_iou = evaluate(
            model, val_loader, device, NUM_CLASSES, IGNORE_INDEX, class_weights
        )
        print(
            f"Epoch {epoch:02d} | train {tr:.4f} | val {va_loss:.4f} | acc {va_acc:.3f} | mIoU {va_miou:.3f}"
        )
        if va_miou > best_miou:
            best_miou = va_miou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "ignore_index": IGNORE_INDEX,
                    "backbone_id": BACKBONE_ID,
                    "per_class_val_iou": per_iou.cpu(),
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path}")

    if args.no_viz:
        return

    model.eval()
    with torch.no_grad():
        img, _ = val_ds[0]
        h, w = img.shape[-2:]
        if h > crop or w > crop:
            pred = predict_image_tiled(
                model,
                img.unsqueeze(0),
                device,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
            )
        else:
            logits = model(pixel_values=img.unsqueeze(0).to(device)).logits
            logits = F.interpolate(logits, size=img.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(dim=1)[0].cpu()

    pred_np = pred.numpy().astype(np.uint8)
    vis_path = out_dir / "prediction_vis.png"
    colorize_mask(pred_np).save(vis_path)

    try:
        import matplotlib.pyplot as plt

        orig = denorm_to_uint8(img)
        color_mask = np.array(Image.open(vis_path))
        alpha = 0.55
        overlay = (alpha * color_mask + (1 - alpha) * orig).astype(np.uint8)

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(orig)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Prediction (colorized)")
        plt.imshow(color_mask)
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(overlay)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "prediction_overlay.png", dpi=150)
        plt.close()
        print(f"[viz] Saved {vis_path} and {out_dir / 'prediction_overlay.png'}")
    except ImportError:
        print(f"[viz] Saved {vis_path} (matplotlib not installed; skipped overlay figure)")


if __name__ == "__main__":
    main()
