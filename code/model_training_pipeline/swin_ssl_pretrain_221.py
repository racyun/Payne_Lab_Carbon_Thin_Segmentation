# -*- coding: utf-8 -*-
"""
Masked self-supervised pretraining for SwinV2-Tiny on unlabeled carbonate images.

This script does NOT require segmentation masks. It trains a Swin encoder to
reconstruct masked image regions, then saves a checkpoint that can be loaded
into `UperNetForSemanticSegmentation(...).backbone` for supervised finetuning.

Example:
    python code/model_training_pipeline/swin_ssl_pretrain_221.py \
      --unlabeled_root data/carbonate_imgs_and_masks \
      --epochs 100 --batch_size 8 --crop 512 --mask_ratio 0.55
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomResizedCrop, RandomVerticalFlip
from torchvision.transforms.v2 import functional as V2F
from tqdm.auto import tqdm
from transformers import Swinv2Model
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BACKBONE_ID = "microsoft/swinv2-tiny-patch4-window8-256"
DEFAULT_GDRIVE_ROOT = "/content/drive/My Drive/Petrographic images_ML work"
# Top-level unlabeled regions under GDRIVE_ROOT (each scanned recursively).
DEFAULT_UNLABELED_SUBFOLDERS = (
    "cretaceous thin sections",
    "Permian-Triassic",
)
# Drive copies may vary in slash/spelling; try all common aliases.
T_J_UNLABELED_ALIASES = (
    "TJ photomicrographs",
    "T/J photomicrographs",
    "T/J photmicrographs",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Masked SSL pretraining for SwinV2 on unlabeled petrography images.")
    p.add_argument(
        "--unlabeled_root",
        type=str,
        default=None,
        help="Optional folder containing unlabeled images. If omitted, use Google Drive root + canonical subfolders.",
    )
    p.add_argument(
        "--gdrive_root",
        type=str,
        default=DEFAULT_GDRIVE_ROOT,
        help="Google Drive root containing the three unlabeled subfolders.",
    )
    p.add_argument("--output_dir", type=str, default=".", help="Directory for SSL checkpoints.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop", type=int, default=512, help="Random crop size for SSL training.")
    p.add_argument("--mask_patch", type=int, default=16, help="Masking block size in pixels.")
    p.add_argument("--mask_ratio", type=float, default=0.55, help="Fraction of mask blocks to hide.")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume SSL training.")
    p.add_argument("--amp", action="store_true", help="Enable AMP mixed precision.")
    p.add_argument("--max_steps_per_epoch", type=int, default=0, help="0 means full epoch.")
    p.add_argument("--no_strict_file_check", action="store_true", help="Skip empty dataset hard-fail.")
    p.add_argument(
        "--save_recon_every",
        type=int,
        default=1,
        help="Save reconstruction preview every N epochs (0 disables).",
    )
    p.add_argument(
        "--num_recon_samples",
        type=int,
        default=2,
        help="Number of batch items shown in each reconstruction preview.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnlabeledImageDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self, paths: list[Path], crop: int):
        self.paths = sorted(paths)
        self.transforms = Compose(
            [
                RandomResizedCrop((crop, crop), scale=(0.7, 1.0), antialias=True),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.2),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = read_image(str(self.paths[idx]))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] > 3:
            img = img[:3]

        img = self.transforms(img)
        img = V2F.convert_image_dtype(img, torch.float32)
        img = V2F.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img


def random_block_mask(batch: int, h: int, w: int, patch: int, ratio: float, device: torch.device) -> torch.Tensor:
    """Returns pixel-space mask [B,1,H,W], where 1 indicates masked pixels."""
    gh = h // patch
    gw = w // patch
    n = gh * gw
    k = int(round(ratio * n))
    mask_grid = torch.zeros(batch, n, device=device, dtype=torch.float32)
    for i in range(batch):
        idx = torch.randperm(n, device=device)[:k]
        mask_grid[i, idx] = 1.0
    mask_grid = mask_grid.view(batch, 1, gh, gw)
    return F.interpolate(mask_grid, size=(h, w), mode="nearest")


class SwinMaskedPretrainModel(nn.Module):
    def __init__(self, backbone_id: str = BACKBONE_ID):
        super().__init__()
        self.backbone = Swinv2Model.from_pretrained(backbone_id)
        hidden = self.backbone.config.hidden_size
        # 16x16 feature map at crop=512 for Swin-Tiny (stride 32).
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden, 256, kernel_size=2, stride=2),  # 16->32
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),     # 32->64
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),      # 64->128
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),       # 128->256
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),       # 256->512
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x_masked)
        tokens = out.last_hidden_state  # [B, L, C]
        b, l, c = tokens.shape
        side = int(round(l**0.5))
        feat = tokens.transpose(1, 2).reshape(b, c, side, side)  # [B,C,16,16] for 512-crop
        return self.decoder(feat)  # [B,3,H,W]


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Only masked pixels contribute.
    diff = (pred - target).abs() * mask
    denom = torch.clamp(mask.sum() * target.shape[1], min=1.0)
    return diff.sum() / denom


def cosine_lr(base_lr: float, epoch: int, max_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * t))


def strip_backbone_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module.backbone."):
            cleaned[k.replace("module.backbone.", "", 1)] = v
        elif k.startswith("backbone."):
            cleaned[k.replace("backbone.", "", 1)] = v
        else:
            cleaned[k] = v
    return cleaned


def collect_images_recursive(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in UnlabeledImageDataset.EXTS and not p.name.startswith(".")
    )


def resolve_unlabeled_paths(args: argparse.Namespace) -> list[Path]:
    if args.unlabeled_root:
        root = Path(args.unlabeled_root)
        return collect_images_recursive(root)

    base = Path(args.gdrive_root)
    all_paths: list[Path] = []
    for sub in DEFAULT_UNLABELED_SUBFOLDERS:
        subdir = base / sub
        if subdir.is_dir():
            all_paths.extend(collect_images_recursive(subdir))
        else:
            print(f"[ssl] Warning: unlabeled folder not found: {subdir}")

    tj_found: list[Path] = []
    for sub in T_J_UNLABELED_ALIASES:
        subdir = base / sub
        if subdir.is_dir():
            tj_found.append(subdir)
            all_paths.extend(collect_images_recursive(subdir))
    if tj_found:
        print("[ssl] T/J unlabeled folder(s) used:", ", ".join(str(p) for p in tj_found))
    else:
        print(
            "[ssl] Warning: T/J folder not found. Tried:\n  "
            + "\n  ".join(str(base / s) for s in T_J_UNLABELED_ALIASES)
        )

    # Deduplicate while preserving path objects.
    return sorted(set(all_paths))


def denorm_to_uint8(img_3chw: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=img_3chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img_3chw.device).view(3, 1, 1)
    x = img_3chw * std + mean
    x = x.clamp(0, 1).detach().cpu().numpy()
    return (np.transpose(x, (1, 2, 0)) * 255.0).astype(np.uint8)


def save_reconstruction_preview(
    epoch: int,
    output_dir: Path,
    imgs: torch.Tensor,
    masked_imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    max_samples: int,
) -> None:
    preview_dir = output_dir / "recon_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    n = min(max_samples, imgs.shape[0])
    fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(14, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        orig_np = denorm_to_uint8(imgs[i])
        masked_np = denorm_to_uint8(masked_imgs[i])
        recon_np = denorm_to_uint8(pred[i].clamp(-5, 5))

        m = mask[i, 0].detach().cpu().numpy()
        recon_mix = orig_np.copy()
        recon_mix[m > 0.5] = recon_np[m > 0.5]

        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(masked_np)
        axes[i, 1].set_title("Masked Input")
        axes[i, 2].imshow(recon_np)
        axes[i, 2].set_title("Reconstruction")
        axes[i, 3].imshow(recon_mix)
        axes[i, 3].set_title("Masked Region Replaced")
        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    out_file = preview_dir / f"epoch_{epoch + 1:03d}.png"
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"[ssl] saved reconstruction preview: {out_file}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = resolve_unlabeled_paths(args)
    ds = UnlabeledImageDataset(paths, crop=args.crop)
    if len(ds) == 0 and not args.no_strict_file_check:
        if args.unlabeled_root:
            raise RuntimeError(f"No images found under {args.unlabeled_root}.")
        _subs = list(DEFAULT_UNLABELED_SUBFOLDERS) + list(T_J_UNLABELED_ALIASES)
        raise RuntimeError(
            "No images found under Google Drive unlabeled subfolders: "
            + ", ".join(str(Path(args.gdrive_root) / s) for s in _subs)
        )
    print(f"[ssl] Found {len(ds)} unlabeled images")

    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if on_gpu else "cpu")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=on_gpu,
        drop_last=True,
    )

    model = SwinMaskedPretrainModel(BACKBONE_ID).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler(device="cuda", enabled=(args.amp and on_gpu))

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_loss = float(ckpt.get("best_loss", best_loss))
        print(f"[ssl] Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(args.lr, epoch, args.epochs, args.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        running = 0.0
        steps = 0
        preview_cache = None
        pbar = tqdm(loader, desc=f"ssl epoch {epoch + 1:03d}", leave=False)
        for imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)
            b, _, h, w = imgs.shape
            mask = random_block_mask(b, h, w, args.mask_patch, args.mask_ratio, device)
            masked_imgs = imgs * (1.0 - mask)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(args.amp and on_gpu)):
                pred = model(masked_imgs)
                loss = masked_l1_loss(pred, imgs, mask)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            running += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / max(1, steps):.4f}", lr=f"{lr:.2e}")
            if preview_cache is None:
                preview_cache = (
                    imgs.detach().cpu(),
                    masked_imgs.detach().cpu(),
                    pred.detach().cpu(),
                    mask.detach().cpu(),
                )

            if args.max_steps_per_epoch > 0 and steps >= args.max_steps_per_epoch:
                break

        epoch_loss = running / max(1, steps)
        print(f"[ssl] epoch {epoch + 1:03d} | loss {epoch_loss:.5f} | lr {lr:.2e}")

        if args.save_recon_every > 0 and ((epoch + 1) % args.save_recon_every == 0) and preview_cache is not None:
            p_imgs, p_masked, p_pred, p_mask = preview_cache
            save_reconstruction_preview(
                epoch=epoch,
                output_dir=out_dir,
                imgs=p_imgs,
                masked_imgs=p_masked,
                pred=p_pred,
                mask=p_mask,
                max_samples=max(1, args.num_recon_samples),
            )

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "backbone_state": strip_backbone_prefix(model.backbone.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": min(best_loss, epoch_loss),
            "config": {
                "backbone_id": BACKBONE_ID,
                "crop": args.crop,
                "mask_patch": args.mask_patch,
                "mask_ratio": args.mask_ratio,
            },
        }
        if scaler.is_enabled():
            ckpt["scaler_state"] = scaler.state_dict()

        last_path = out_dir / "ssl_swinv2_last.pth"
        torch.save(ckpt, last_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, out_dir / "ssl_swinv2_best.pth")
            print(f"[ssl] saved best checkpoint: {out_dir / 'ssl_swinv2_best.pth'}")

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save(ckpt, out_dir / f"ssl_swinv2_epoch_{epoch + 1:03d}.pth")

    print("[ssl] Done.")


if __name__ == "__main__":
    main()

