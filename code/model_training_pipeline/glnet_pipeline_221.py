# -*- coding: utf-8 -*-
"""
Training pipeline for the Global-Local dual-branch model (glnet_model_221.GlobalLocalUperNet).

Each sample = the whole image downsampled to a full frame (global branch) + one full-resolution
crop (local branch); the model fuses them and predicts the crop. This gives whole-image context
while keeping full-resolution detail, addressing the "512 crops chop whole grains" problem.

Reuses from swin_training_pipeline_221: the CV split (augment_aware / stratified), focal loss,
confusion-matrix mIoU / per-class IoU, cross-fold-mean W&B logging, seed, and the merge/ignore
mask handling. Same flags (--n_folds, --merge_class_map, --ignore_artifacts, --loss_type,
--wandb_mean_only, --backbone_checkpoint, ...) plus --global_h/--global_w, --local_crop, --amp.

Evaluation (--eval_mode):
  - 'tiled' (default): faithful WHOLE-IMAGE metric. Each val image is tiled into full-res crops;
    each is predicted with the shared downsampled global image; overlapping logits are averaged,
    argmaxed, and scored against the full mask. Global features are encoded once per image.
  - 'center': single center crop + global (fast; for quick smoke runs).

At the end of each fold (unless --no_viz) it saves --viz_samples 4-panel figures
(Original | Ground Truth Mask | Prediction | Overlay) under fold_N/prediction_viz/, using the
whole-image tiled prediction of the best-mIoU model (same look as the single-path pipeline).

Run (smoke): python code/model_training_pipeline/glnet_pipeline_221.py --img_dir ... --mask_dir ... --no_train
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.io import read_image
from torchvision.transforms.v2 import functional as TF
from tqdm.auto import tqdm

try:
    import wandb  # noqa: F401

    _WANDB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _WANDB_AVAILABLE = False

from glnet_model_221 import build_glnet
from swin_training_pipeline_221 import (
    BACKBONE_ID,
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    artifact_ignore_ids,
    augment_aware_kfold_indices,
    build_class_presence_matrix,
    colorize_mask,
    confusion_matrix,
    focal_cross_entropy,
    kfold_train_val_indices,
    load_ssl_backbone_checkpoint,
    log_cv_mean_wandb,
    miou_from_confusion,
    parse_merge_map,
    set_seed,
    stratified_kfold_indices,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS = {".png"}


def _normalize(img_u8: torch.Tensor) -> torch.Tensor:
    return TF.normalize(img_u8.float().div(255.0), IMAGENET_MEAN, IMAGENET_STD)


class GlnetDataset(torch.utils.data.Dataset):
    """Returns (global_img, local_crop, bbox_norm, local_mask). Train=random crop, else center crop."""

    def __init__(self, img_dir, mask_dir, global_hw, local_crop, ignore_ids=(), merge_map=None,
                 train=True, print_pair_count=False):
        self.global_hw = (int(global_hw[0]), int(global_hw[1]))
        self.crop = int(local_crop)
        self.ignore_ids = tuple(ignore_ids)
        self.merge_map = dict(merge_map or {})
        self.train = train
        img_dir, mask_dir = Path(img_dir), Path(mask_dir)
        imgs = {p.stem: p for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")}
        masks = {p.stem: p for p in mask_dir.iterdir()
                 if p.is_file() and p.suffix.lower() in MASK_EXTS and not p.name.startswith(".")}
        self.pairs = [(imgs[k], masks[k]) for k in sorted(imgs.keys() & masks.keys())]
        if not self.pairs:
            raise RuntimeError("Found no (image, mask) pairs. Check folder names & extensions.")
        if print_pair_count:
            print(f"[dataset] Using {len(self.pairs)} paired samples.")

    def __len__(self):
        return len(self.pairs)

    def _load(self, idx):
        img_p, mask_p = self.pairs[idx]
        img = read_image(str(img_p))
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        elif img.shape[0] == 4:
            img = img[:3]
        img = img.to(torch.uint8)
        mask = read_image(str(mask_p))
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0:1]
        mask = mask.squeeze(0).to(torch.long)
        if self.ignore_ids or self.merge_map:
            mask = mask.clone()
            for cid in self.ignore_ids:
                mask[mask == cid] = IGNORE_INDEX
            for src, dst in self.merge_map.items():
                mask[mask == src] = dst
        oob = (mask >= NUM_CLASSES) & (mask != IGNORE_INDEX)
        if bool(oob.any()):
            mask[oob] = IGNORE_INDEX
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load(idx)
        c = self.crop
        # Pad up so a full-res crop always fits (mask border -> ignore).
        _, H, W = img.shape
        if H < c or W < c:
            ph, pw = max(0, c - H), max(0, c - W)
            img = F.pad(img, (0, pw, 0, ph), value=0)
            mask = F.pad(mask.unsqueeze(0), (0, pw, 0, ph), value=IGNORE_INDEX).squeeze(0)
            _, H, W = img.shape

        # Global branch: whole (padded) image resized to the full frame.
        gh, gw = self.global_hw
        global_img = TF.resize(img, [gh, gw], antialias=True)
        global_img = _normalize(global_img)

        # Local branch: a crop of the same (padded) image; bbox normalized to (H, W).
        if self.train:
            top = random.randint(0, H - c)
            left = random.randint(0, W - c)
        else:
            top = (H - c) // 2
            left = (W - c) // 2
        local = TF.crop(img, top, left, c, c)
        local_mask = TF.crop(mask.unsqueeze(0), top, left, c, c).squeeze(0)
        local = _normalize(local)
        bbox = torch.tensor([left / W, top / H, (left + c) / W, (top + c) / H], dtype=torch.float32)
        return global_img, local, bbox, local_mask

    def full_sample(self, idx):
        """For tiled whole-image eval: returns (global_img_norm[1,3,gh,gw-shaped],
        full_img_uint8[3,H,W], full_mask_long[H,W]) with the SAME pad/merge/ignore handling."""
        img, mask = self._load(idx)
        c = self.crop
        _, H, W = img.shape
        if H < c or W < c:
            ph, pw = max(0, c - H), max(0, c - W)
            img = F.pad(img, (0, pw, 0, ph), value=0)
            mask = F.pad(mask.unsqueeze(0), (0, pw, 0, ph), value=IGNORE_INDEX).squeeze(0)
        gh, gw = self.global_hw
        global_img = _normalize(TF.resize(img, [gh, gw], antialias=True))
        return global_img, img, mask


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def _loss(logits, mask, class_weights, loss_type, focal_gamma):
    if loss_type == "focal":
        return focal_cross_entropy(logits, mask, class_weights, IGNORE_INDEX, focal_gamma)
    return F.cross_entropy(logits, mask, weight=class_weights, ignore_index=IGNORE_INDEX)


def train_one_epoch(model, loader, optimizer, device, epoch, class_weights, loss_type, focal_gamma,
                    scheduler=None, scaler=None, max_steps=None) -> float:
    model.train()
    total, n = 0.0, 0
    use_amp = scaler is not None
    pbar = tqdm(loader, desc=f"epoch {epoch:02d}", leave=False)
    for i, (g, l, bbox, mask) in enumerate(pbar):
        if max_steps is not None and i >= max_steps:
            break
        g, l, bbox, mask = (g.to(device), l.to(device), bbox.to(device), mask.to(device))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(g, l, bbox)
            loss = _loss(logits, mask, class_weights, loss_type, focal_gamma)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = g.size(0)
        total += float(loss.item()) * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total/max(1,n):.4f}")
    if scheduler is not None:
        scheduler.step()
    print(f"epoch {epoch:02d} | train_avg_loss {total/max(1,n):.4f}")
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, class_weights, loss_type, focal_gamma):
    model.eval()
    total, n = 0.0, 0
    cm = torch.zeros(num_classes, num_classes, device=device, dtype=torch.float64)
    for g, l, bbox, mask in loader:
        g, l, bbox, mask = (g.to(device), l.to(device), bbox.to(device), mask.to(device))
        logits = model(g, l, bbox)
        total += float(_loss(logits, mask, class_weights, loss_type, focal_gamma).item()) * g.size(0)
        n += g.size(0)
        cm += confusion_matrix(logits.argmax(dim=1), mask, num_classes, IGNORE_INDEX).double()
    miou, per_iou = miou_from_confusion(cm)
    pixel_acc = (cm.diag().sum() / cm.sum().clamp(min=1)).item()
    return total / max(1, n), pixel_acc, miou, per_iou.cpu()


def _tile_positions(size: int, crop: int, stride: int):
    """Top-left offsets covering [0, size); the last tile is clamped to the edge (small overlap)."""
    if size <= crop:
        return [0]
    pos = list(range(0, size - crop + 1, stride))
    if pos[-1] != size - crop:
        pos.append(size - crop)
    return pos


@torch.no_grad()
def _tiled_logits(model, global_img, full_u8, device, crop, stride, tile_batch):
    """Encode the global image ONCE, predict every full-res tile reusing those features, and return
    the averaged full-image logits [C, H, W] on CPU (overlapping tiles are logit-averaged)."""
    _, H, W = full_u8.shape
    g_feats = model.encode_global(global_img.unsqueeze(0).to(device))
    tops, lefts = _tile_positions(H, crop, stride), _tile_positions(W, crop, stride)
    num_classes = model.num_classes
    logit_sum = torch.zeros(num_classes, H, W, dtype=torch.float32)  # CPU accumulator
    count = torch.zeros(H, W, dtype=torch.float32)
    locals_, bboxes, positions = [], [], []

    def _flush():
        if not locals_:
            return
        loc = torch.stack(locals_).to(device)
        bb = torch.stack(bboxes).to(device)
        logits = model.forward_local(g_feats, loc, bb).float().cpu()
        for k, (t, l) in enumerate(positions):
            logit_sum[:, t:t + crop, l:l + crop] += logits[k]
            count[t:t + crop, l:l + crop] += 1.0
        locals_.clear()
        bboxes.clear()
        positions.clear()

    for t in tops:
        for l in lefts:
            locals_.append(_normalize(TF.crop(full_u8, t, l, crop, crop)))
            bboxes.append(torch.tensor([l / W, t / H, (l + crop) / W, (t + crop) / H],
                                       dtype=torch.float32))
            positions.append((t, l))
            if len(locals_) >= tile_batch:
                _flush()
    _flush()
    return logit_sum / count.clamp(min=1).unsqueeze(0)  # [C, H, W]


@torch.no_grad()
def evaluate_tiled(model, dataset, indices, device, num_classes, crop, stride, tile_batch,
                   loss_type, focal_gamma):
    """Faithful WHOLE-IMAGE eval: tile each val image into full-res crops, predict each with the
    shared downsampled global image, average overlapping logits, argmax, and score the full mask."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    total_loss, n = 0.0, 0
    for idx in tqdm(indices, desc="tiled-eval", leave=False):
        global_img, full_u8, full_mask = dataset.full_sample(idx)
        avg = _tiled_logits(model, global_img, full_u8, device, crop, stride, tile_batch)
        pred = avg.argmax(0)
        cm += confusion_matrix(pred, full_mask, num_classes, IGNORE_INDEX).double()
        total_loss += float(_loss(avg.unsqueeze(0), full_mask.unsqueeze(0), None, loss_type, focal_gamma).item())
        n += 1
    miou, per_iou = miou_from_confusion(cm)
    pixel_acc = (cm.diag().sum() / cm.sum().clamp(min=1)).item()
    return total_loss / max(1, n), pixel_acc, miou, per_iou.cpu()


@torch.no_grad()
def render_predictions_glnet(model, dataset, indices, device, crop, stride, tile_batch,
                             viz_dir: Path, n_viz: int):
    """Save 4-panel figures (Original | Ground Truth Mask | Prediction | Overlay) for the first
    n_viz val items, matching the single-path pipeline's look. Predictions are the whole-image
    tiled output, so the panels show the real full-image segmentation."""
    model.eval()
    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[viz] matplotlib not installed; skipping GLNet panels.")
        return
    n = min(n_viz, len(indices))
    for i in range(n):
        global_img, full_u8, full_mask = dataset.full_sample(indices[i])
        avg = _tiled_logits(model, global_img, full_u8, device, crop, stride, tile_batch)
        pred_np = avg.argmax(0).numpy().astype(np.uint8)
        gt_np = full_mask.numpy().astype(np.uint8)
        orig = full_u8.permute(1, 2, 0).numpy().astype(np.uint8)
        pred_color = np.array(colorize_mask(pred_np))
        gt_color = np.array(colorize_mask(gt_np))
        overlay = (0.55 * pred_color + 0.45 * orig).astype(np.uint8)

        Image.fromarray(pred_color).save(viz_dir / f"prediction_{i:02d}.png")
        plt.figure(figsize=(20, 6))
        for j, (title, im) in enumerate([("Original", orig), ("Ground Truth Mask", gt_color),
                                         ("Prediction", pred_color), ("Overlay", overlay)]):
            plt.subplot(1, 4, j + 1)
            plt.title(title)
            plt.imshow(im)
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(viz_dir / f"panel_{i:02d}.png", dpi=150)
        plt.close()
    print(f"[viz] Saved {n} GLNet 4-panel figures under {viz_dir}")


def build_scheduler(optimizer, args):
    if args.scheduler == "none" and args.warmup_epochs <= 0:
        return None
    main_epochs = max(1, args.epochs - max(0, args.warmup_epochs))
    main_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs) if args.scheduler == "cosine" else None
    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs)
        if main_sched is None:
            return warmup
        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[args.warmup_epochs])
    return main_sched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Global-Local dual-branch (GLNet-style) segmentation with K-fold CV.")
    p.add_argument("--img_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--global_h", type=int, default=1024)
    p.add_argument("--global_w", type=int, default=1280)
    p.add_argument("--local_crop", type=int, default=512)
    p.add_argument("--backbone_id", type=str, default=BACKBONE_ID)
    p.add_argument("--backbone_checkpoint", type=str, default=None, help="SSL backbone checkpoint.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Train batch size. Must be >=2: UPerNet's PSP head has BatchNorm that "
                        "fails on a 1x1-pooled batch of 1.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--cv_strategy", type=str, default="stratified", choices=["stratified", "random"])
    p.add_argument("--group_by_stem", action="store_true", help="Augmentations training-only (pre-augmented dataset).")
    p.add_argument("--group_pattern", type=str, default=r"_aug\d+$")
    p.add_argument("--num_augmentations_per_img", type=int, default=None)
    p.add_argument("--merge_class_map", type=str, default=None, help="e.g. '12:1,10:1'.")
    p.add_argument("--ignore_artifacts", action="store_true")
    p.add_argument("--ignore_scale_bar", action="store_true")
    p.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--amp", action="store_true", help="Mixed precision (recommended at high resolution).")
    p.add_argument("--eval_mode", type=str, default="tiled", choices=["tiled", "center"],
                   help="'tiled' = faithful whole-image eval (tile+stitch); 'center' = single center crop (fast).")
    p.add_argument("--eval_stride", type=int, default=None,
                   help="Tile stride for tiled eval (default = --local_crop, i.e. non-overlapping).")
    p.add_argument("--eval_tile_batch", type=int, default=4, help="Tiles predicted per batch during tiled eval.")
    p.add_argument("--viz_samples", type=int, default=8,
                   help="Number of val images to save as 4-panel prediction figures per fold.")
    p.add_argument("--no_viz", action="store_true", help="Skip saving prediction panels at end of each fold.")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--no_train", action="store_true", help="Build model + one dummy forward, report memory, exit.")
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_mean_only", action="store_true")
    return p.parse_args()


def run_single_fold(fold, n_folds, train_idx, val_idx, cfg, args, device):
    ignore_ids, merge_map = cfg["ignore_ids"], cfg["merge_map"]
    global_hw = (args.global_h, args.global_w)
    common = dict(img_dir=args.img_dir, mask_dir=args.mask_dir, global_hw=global_hw,
                  local_crop=args.local_crop, ignore_ids=ignore_ids, merge_map=merge_map)
    if args.batch_size < 2:
        raise SystemExit("--batch_size must be >=2 (UPerNet's PSP-head BatchNorm fails on a batch of 1).")
    train_full = GlnetDataset(train=True, **common)
    val_full = GlnetDataset(train=False, **common)
    train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)

    on_gpu = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=on_gpu, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=on_gpu)

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[glnet] Fold {fold + 1}/{n_folds} | Train {len(train_ds)} | Val {len(val_ds)} | "
          f"global={global_hw} local={args.local_crop}")
    if args.no_train:
        g, l, bbox, mask = next(iter(train_loader))
        print(f"[no_train] batch shapes: global {tuple(g.shape)} | local {tuple(l.shape)} | "
              f"bbox {tuple(bbox.shape)} | mask {tuple(mask.shape)}")
        return {"fold": fold, "status": "no_train"}

    model = build_glnet(NUM_CLASSES, IGNORE_INDEX, args.backbone_id).to(device)
    if args.backbone_checkpoint:
        load_ssl_backbone_checkpoint(model, args.backbone_checkpoint)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and on_gpu) else None
    eval_stride = args.eval_stride if args.eval_stride else args.local_crop
    print(f"[train] global={global_hw} local={args.local_crop} loss={args.loss_type} amp={bool(scaler)} "
          f"eval={args.eval_mode}" + (f" (stride {eval_stride}, tile_batch {args.eval_tile_batch})"
                                       if args.eval_mode == "tiled" else ""))

    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled" and not args.wandb_mean_only:
        if not _WANDB_AVAILABLE:
            print("[wandb] wandb not installed; skipping.")
        else:
            import wandb as _wandb

            name = args.wandb_run_name
            if n_folds >= 2:
                name = f"{name}_fold{fold}" if name else f"fold{fold}"
            wandb_run = _wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name,
                                    mode=args.wandb_mode, group=args.wandb_run_name or None, reinit=True,
                                    config={"stage": "glnet_finetune", "global_hw": list(global_hw),
                                            "local_crop": args.local_crop, "num_classes": NUM_CLASSES,
                                            "epochs": args.epochs, "lr": args.lr, "loss_type": args.loss_type,
                                            "focal_gamma": args.focal_gamma, "amp": bool(scaler),
                                            "n_folds": n_folds, "fold": fold, "seed": args.seed})

    best_miou, best_row = -1.0, None
    ckpt_path = fold_dir / "best_glnet.pth"
    metric_history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, epoch, None, args.loss_type,
                             args.focal_gamma, scheduler, scaler, args.max_steps_per_epoch)
        if args.eval_mode == "tiled":
            va_loss, va_acc, va_miou, per_iou = evaluate_tiled(
                model, val_full, val_idx, device, NUM_CLASSES, args.local_crop, eval_stride,
                args.eval_tile_batch, args.loss_type, args.focal_gamma)
        else:
            va_loss, va_acc, va_miou, per_iou = evaluate(model, val_loader, device, NUM_CLASSES, None,
                                                         args.loss_type, args.focal_gamma)
        print(f"Fold {fold} | Epoch {epoch:02d} | train {tr:.4f} | val {va_loss:.4f} | "
              f"acc {va_acc:.3f} | mIoU {va_miou:.3f}")
        ious = [float(per_iou[i]) if torch.isfinite(per_iou[i]).item() else float("nan") for i in range(NUM_CLASSES)]
        print("  per-class val IoU (epoch %02d):" % epoch)
        for i, nm in enumerate(CLASS_NAMES[:NUM_CLASSES]):
            print(f"    {i:02d} {nm}: " + (f"{ious[i]:.4f}" if ious[i] == ious[i] else "nan"))

        em = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]), "train/loss": float(tr),
              "val/loss": float(va_loss), "val/pixel_acc": float(va_acc), "val/mIoU": float(va_miou)}
        for i, nm in enumerate(CLASS_NAMES[:NUM_CLASSES]):
            if ious[i] == ious[i]:
                em[f"val_iou/{i:02d}_{nm}"] = ious[i]
        metric_history.append(em)
        if wandb_run is not None:
            wandb_run.log(em, step=epoch)

        if va_miou > best_miou:
            best_miou = va_miou
            best_row = {"epoch": epoch, "miou": float(va_miou), "pixel_acc": float(va_acc), "val_loss": float(va_loss)}
            torch.save({"model_state": model.state_dict(), "num_classes": NUM_CLASSES, "fold": fold,
                        "global_hw": list(global_hw), "local_crop": args.local_crop,
                        "per_class_val_iou": per_iou}, ckpt_path)
            print(f"  saved {ckpt_path}")
            if wandb_run is not None:
                wandb_run.summary["best_val_mIoU"] = best_miou
                wandb_run.summary["best_epoch"] = epoch

    if not args.no_viz:
        best_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])  # panels reflect the best-mIoU model
        render_predictions_glnet(model, val_full, val_idx, device, args.local_crop, eval_stride,
                                 args.eval_tile_batch, fold_dir / "prediction_viz", args.viz_samples)

    if wandb_run is not None:
        wandb_run.finish()
    assert best_row is not None
    best_row.update(fold=fold, best_checkpoint=str(ckpt_path), history=metric_history)
    return best_row


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ignore_ids = artifact_ignore_ids(args)
    merge_map = parse_merge_map(args.merge_class_map)
    if ignore_ids:
        print(f"[config] Ignoring artifact classes (-> {IGNORE_INDEX}): {list(ignore_ids)}")
    if merge_map:
        pairs = ", ".join(f"{s}:{CLASS_NAMES[s]} -> {d}:{CLASS_NAMES[d] if d < len(CLASS_NAMES) else d}"
                          for s, d in merge_map.items())
        print(f"[config] Merging classes (masks + presence): {pairs}")
    cfg = {"ignore_ids": ignore_ids, "merge_map": merge_map}

    probe = GlnetDataset(args.img_dir, args.mask_dir, (args.global_h, args.global_w), args.local_crop,
                         ignore_ids, merge_map, train=True, print_pair_count=True)
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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_folds))
    print(f"[config] GLNet dual-branch | global={args.global_h}x{args.global_w} local={args.local_crop} | "
          f"{args.n_folds}-fold {args.cv_strategy} CV | seed={args.seed} | folds={folds_to_run} | n={n}")

    all_best = []
    for fold in folds_to_run:
        tr, va = splits[fold]
        result = run_single_fold(fold, args.n_folds, tr.tolist(), va.tolist(), cfg, args, device)
        if result.get("status") != "no_train":
            all_best.append(result)
    if args.no_train:
        return

    histories = [r.get("history", []) for r in all_best]
    summary_rows = [{k: v for k, v in r.items() if k != "history"} for r in all_best]
    miou_vals = [r["miou"] for r in all_best]
    summary = {"model": "glnet", "global_hw": [args.global_h, args.global_w], "local_crop": args.local_crop,
               "n_folds": args.n_folds, "per_fold_best": summary_rows,
               "mean_best_miou": float(np.mean(miou_vals)) if miou_vals else None,
               "std_best_miou": float(np.std(miou_vals)) if miou_vals else None}
    with (out_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== CV summary ===")
    print(json.dumps(summary, indent=2))
    log_cv_mean_wandb(args, histories, config_extra={"model": "glnet",
                      "global_hw": f"{args.global_h}x{args.global_w}", "local_crop": args.local_crop})


if __name__ == "__main__":
    main()
