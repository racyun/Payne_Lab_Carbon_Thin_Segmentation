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

# Label ids 0–15 (must align with NUM_CLASSES).
CLASS_NAMES = (
    "background",
    "bivalves",
    "micrite",
    "cement",
    "echinoderms",
    "foraminifera",
    "calcareous algae",
    "peloid",
    "unid biota",
    "ooid",
    "gastropods",
    "scale bar",
    "mollusk",
    "ostracod",
    "aggregate grain",
    "brachiopod",
)
DEFAULT_GDRIVE_LABELED_IMG_DIR = (
    "/content/drive/My Drive/Petrographic images_ML work/labelled images_PS/labelled images_PS/my_dataset/img"
)
DEFAULT_GDRIVE_LABELED_MASK_DIR = (
    "/content/drive/My Drive/Petrographic images_ML work/labelled images_PS/labelled images_PS/my_dataset/masks_machine"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UPerNet + SwinV2 on carbonate segmentation masks.")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root folder containing ``img/`` and ``masks/``. Default: repo data path when unset.",
    )
    p.add_argument(
        "--img_dir",
        type=str,
        default=DEFAULT_GDRIVE_LABELED_IMG_DIR,
        help="Explicit labeled image directory. Overrides data_root/img when set.",
    )
    p.add_argument(
        "--mask_dir",
        type=str,
        default=DEFAULT_GDRIVE_LABELED_MASK_DIR,
        help="Explicit labeled mask directory. Overrides data_root/masks when set.",
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
    p.add_argument(
        "--auto_class_weights",
        action="store_true",
        help="Estimate inverse-frequency class weights from training masks.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine", "step"],
        help="Learning-rate scheduler for finetuning.",
    )
    p.add_argument("--step_size", type=int, default=10, help="StepLR step size when scheduler=step.")
    p.add_argument("--gamma", type=float, default=0.5, help="StepLR gamma when scheduler=step.")
    p.add_argument("--tile_size", type=int, default=512, help="Tile edge length for ``predict_image_tiled``.")
    p.add_argument("--tile_stride", type=int, default=None, help="Stride for tiling; default = tile_size // 2.")
    p.add_argument(
        "--viz_samples",
        type=int,
        default=4,
        help="Number of validation samples to visualize at end of training.",
    )
    p.add_argument(
        "--backbone_checkpoint",
        type=str,
        default=None,
        help="Optional SSL checkpoint path; loads checkpoint['backbone_state'] into model.backbone.",
    )
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


def load_ssl_backbone_checkpoint(model: UperNetForSemanticSegmentation, checkpoint_path: str | Path) -> None:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"backbone checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
        ssl_sd = ckpt["backbone_state"]
    elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        # Fallback if checkpoint was saved without a dedicated backbone_state.
        ssl_sd = {
            k.replace("backbone.", "", 1): v
            for k, v in ckpt["model_state"].items()
            if k.startswith("backbone.")
        }
    else:
        raise KeyError(
            "Checkpoint missing 'backbone_state' and compatible 'model_state'. "
            "Expected an SSL checkpoint from swin_ssl_pretrain_221.py."
        )

    missing, unexpected = model.backbone.load_state_dict(ssl_sd, strict=False)
    print(
        "[ssl->finetune] loaded backbone checkpoint:",
        ckpt_path,
        f"\n  missing={len(missing)} keys, unexpected={len(unexpected)} keys",
    )


def estimate_class_weights_from_dataset(
    dataset, num_classes: int, ignore_index: int, device: torch.device
) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for _, labels in dataset:
        flat = labels.reshape(-1)
        valid = (flat != ignore_index) & (flat >= 0) & (flat < num_classes)
        if valid.any():
            binc = torch.bincount(flat[valid], minlength=num_classes).to(torch.float64)
            counts += binc

    if counts.sum() == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return weights.to(dtype=torch.float32, device=device)


class CarbonateSegmentationDataset(torch.utils.data.Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_EXTS = {".jpg", ".jpeg", ".png"}
    MASK_EXTS = {".png"}

    def __init__(
        self,
        root: str | Path,
        img_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
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

        img_dir = Path(img_dir) if img_dir is not None else self.root / "img"
        mask_dir = Path(mask_dir) if mask_dir is not None else self.root / "masks"

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


def print_per_class_iou(per_iou: torch.Tensor, epoch: int) -> None:
    """Pretty-print per-class validation IoU (one line per class)."""
    print(f"  per-class val IoU (epoch {epoch:02d}):")
    vec = per_iou.detach().cpu()
    for i, name in enumerate(CLASS_NAMES[: len(vec)]):
        v = vec[i]
        if torch.isfinite(v).item():
            print(f"    {name} IoU: {float(v):.4f}")
        else:
            print(f"    {name} IoU: nan")


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
    scheduler=None,
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

    if scheduler is not None:
        scheduler.step()

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

    candidate_img_dir = Path(args.img_dir) if args.img_dir else None
    candidate_mask_dir = Path(args.mask_dir) if args.mask_dir else None
    use_explicit_dirs = bool(
        candidate_img_dir
        and candidate_mask_dir
        and candidate_img_dir.is_dir()
        and candidate_mask_dir.is_dir()
    )
    if use_explicit_dirs:
        data_root = Path(args.data_root) if args.data_root else Path(".")
        img_dir = candidate_img_dir
        mask_dir = candidate_mask_dir
        print(f"[config] Using explicit labeled dirs:\n  img={img_dir}\n  mask={mask_dir}")
    elif args.data_root is None:
        if candidate_img_dir or candidate_mask_dir:
            print(
                "[config] Explicit labeled dirs not found; falling back to --data_root/default layout.\n"
                f"  checked img_dir={candidate_img_dir}\n"
                f"  checked mask_dir={candidate_mask_dir}"
            )
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
        img_dir=img_dir if use_explicit_dirs else None,
        mask_dir=mask_dir if use_explicit_dirs else None,
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
        img_dir=img_dir if use_explicit_dirs else None,
        mask_dir=mask_dir if use_explicit_dirs else None,
        transforms=train_transforms,
        normalize=True,
        strict=False,
        ignore_scale_bar=args.ignore_scale_bar,
        print_pair_count=False,
    )
    val_full = CarbonateSegmentationDataset(
        data_root,
        img_dir=img_dir if use_explicit_dirs else None,
        mask_dir=mask_dir if use_explicit_dirs else None,
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
    elif args.auto_class_weights:
        class_weights = estimate_class_weights_from_dataset(train_ds, NUM_CLASSES, IGNORE_INDEX, device)
        print("[train] Using auto-estimated class weights:", class_weights.detach().cpu().numpy().round(3).tolist())

    model = get_model_semantic_segmentation(NUM_CLASSES, IGNORE_INDEX, BACKBONE_ID)
    if args.backbone_checkpoint:
        load_ssl_backbone_checkpoint(model, args.backbone_checkpoint)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.step_size), gamma=args.gamma)

    best_miou = -1.0
    ckpt_path = out_dir / "best_upernet_swinv2.pth"
    per_class_log_path = out_dir / "val_per_class_iou.csv"
    with per_class_log_path.open("w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(f"class_{i}" for i in range(NUM_CLASSES)) + "\n")

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
            scheduler,
        )
        va_loss, va_acc, va_miou, per_iou = evaluate(
            model, val_loader, device, NUM_CLASSES, IGNORE_INDEX, class_weights
        )
        print(
            f"Epoch {epoch:02d} | train {tr:.4f} | val {va_loss:.4f} | acc {va_acc:.3f} | mIoU {va_miou:.3f}"
        )
        print_per_class_iou(per_iou, epoch)
        with per_class_log_path.open("a", encoding="utf-8") as f:
            vals = ",".join(f"{float(v):.6f}" if torch.isfinite(v) else "nan" for v in per_iou.detach().cpu())
            f.write(f"{epoch},{vals}\n")
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
    n_viz = max(1, min(args.viz_samples, len(val_ds)))
    viz_dir = out_dir / "prediction_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        with torch.no_grad():
            for i in range(n_viz):
                img, gt_mask = val_ds[i]
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
                gt_np = gt_mask.cpu().numpy().astype(np.uint8)
                pred_color = np.array(colorize_mask(pred_np))
                gt_color = np.array(colorize_mask(gt_np))
                orig = denorm_to_uint8(img)
                alpha = 0.55
                overlay = (alpha * pred_color + (1 - alpha) * orig).astype(np.uint8)

                # Save raw predicted color mask for compatibility with previous output style.
                pred_path = viz_dir / f"prediction_{i:02d}.png"
                Image.fromarray(pred_color).save(pred_path)

                # Save four-panel figure: original, ground-truth mask, prediction, overlay.
                panel_path = viz_dir / f"panel_{i:02d}.png"
                plt.figure(figsize=(20, 6))
                plt.subplot(1, 4, 1)
                plt.title("Original")
                plt.imshow(orig)
                plt.axis("off")
                plt.subplot(1, 4, 2)
                plt.title("Ground Truth Mask")
                plt.imshow(gt_color)
                plt.axis("off")
                plt.subplot(1, 4, 3)
                plt.title("Prediction")
                plt.imshow(pred_color)
                plt.axis("off")
                plt.subplot(1, 4, 4)
                plt.title("Overlay")
                plt.imshow(overlay)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(panel_path, dpi=150)
                plt.close()

        print(f"[viz] Saved {n_viz} prediction masks and {n_viz} 4-panel figures under {viz_dir}")
    except ImportError:
        # Fallback: save only raw predicted masks if matplotlib is unavailable.
        with torch.no_grad():
            for i in range(n_viz):
                img, _ = val_ds[i]
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
                Image.fromarray(np.array(colorize_mask(pred_np))).save(viz_dir / f"prediction_{i:02d}.png")
        print(f"[viz] matplotlib not installed; saved {n_viz} predicted masks under {viz_dir}")


if __name__ == "__main__":
    main()
