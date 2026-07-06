# -*- coding: utf-8 -*-
"""
Baseline semantic-segmentation runner (U-Net / DeepLabV3+) for benchmarking against the
SwinV2+UPerNet method — on the SAME folds, validation images, loss, metric, and W&B logging.

Models come from ``segmentation_models_pytorch`` (smp) with ImageNet-pretrained encoders
(the "pretraining"); we then finetune on the carbonate data (the "finetuning"). Everything
else is reused from the main pipeline so results are directly comparable:

  - datasets: CarbonateSegmentationDataset (17-class) / BinaryCarbonateDataset (binary),
  - CV split: augment_aware_kfold_indices / stratified_kfold_indices (same seed, clean-original
    validation, --group_by_stem + --num_augmentations_per_img, --merge_class_map, --ignore_*),
  - loss: focal (or CE) with optional inverse-frequency class weights,
  - metrics: confusion_matrix -> mIoU + per-class IoU,
  - W&B: per-fold runs + a single cross-fold-averaged run (log_cv_mean_wandb).

Run (17-class U-Net)::

    python baseline_seg_pipeline_221.py --task multiclass --arch unet \
      --img_dir ... --mask_dir ... --ignore_artifacts --loss_type focal --wandb_mean_only ...

Requires: torch, torchvision, segmentation-models-pytorch, tqdm, pillow, numpy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors
from torchvision.transforms.v2 import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from tqdm.auto import tqdm

try:
    import wandb  # noqa: F401

    _WANDB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _WANDB_AVAILABLE = False

from swin_training_pipeline_221 import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    artifact_ignore_ids,
    augment_aware_kfold_indices,
    build_class_presence_matrix,
    confusion_matrix,
    estimate_class_weights_from_dataset,
    focal_cross_entropy,
    kfold_train_val_indices,
    log_cv_mean_wandb,
    miou_from_confusion,
    parse_merge_map,
    set_seed,
    stratified_kfold_indices,
    CarbonateSegmentationDataset,
)
from swin_binary_segmentation_221 import (
    ARTIFACT_CLASS_IDS,
    BINARY_CLASS_NAMES,
    NUM_BINARY_CLASSES,
    BinaryCarbonateDataset,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_baseline_model(arch: str, encoder: str, num_classes: int):
    import segmentation_models_pytorch as smp

    common = dict(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)
    if arch == "unet":
        return smp.Unet(**common)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**common)
    raise SystemExit(f"--arch must be 'unet' or 'deeplabv3plus', got {arch!r}")


# ---------------------------------------------------------------------------
# Train / eval (smp models return a logits tensor directly, unlike UPerNet)
# ---------------------------------------------------------------------------
def _seg_loss(logits, labels, class_weights, loss_type, focal_gamma):
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    if loss_type == "focal":
        return focal_cross_entropy(logits, labels, class_weights, IGNORE_INDEX, focal_gamma), logits
    return F.cross_entropy(logits, labels, weight=class_weights, ignore_index=IGNORE_INDEX), logits


def train_one_epoch(model, loader, optimizer, device, epoch, class_weights, loss_type,
                    focal_gamma, scheduler=None, max_steps=None) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"epoch {epoch:02d}", leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        if max_steps is not None and i >= max_steps:
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _seg_loss(model(imgs), labels, class_weights, loss_type, focal_gamma)
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
def evaluate(model, loader, device, num_classes, class_weights, loss_type, focal_gamma):
    model.eval()
    total, n = 0.0, 0
    cm = torch.zeros(num_classes, num_classes, device=device, dtype=torch.float64)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        loss, logits = _seg_loss(model(imgs), labels, class_weights, loss_type, focal_gamma)
        preds = logits.argmax(dim=1)
        total += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
        cm += confusion_matrix(preds, labels, num_classes, IGNORE_INDEX).double()
    miou, per_iou = miou_from_confusion(cm)
    pixel_acc = (cm.diag().sum() / cm.sum().clamp(min=1)).item()
    return total / max(1, n), pixel_acc, miou, per_iou.cpu()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def make_dataset(task, data_root, img_dir, mask_dir, transforms, ignore_ids, merge_map,
                 strict, print_pair_count):
    if task == "binary":
        return BinaryCarbonateDataset(
            data_root, img_dir=img_dir, mask_dir=mask_dir, transforms=transforms,
            normalize=True, strict=strict, print_pair_count=print_pair_count,
        )
    return CarbonateSegmentationDataset(
        data_root, img_dir=img_dir, mask_dir=mask_dir, transforms=transforms,
        normalize=True, strict=strict, ignore_class_ids=ignore_ids, merge_class_map=merge_map,
        print_pair_count=print_pair_count,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Net / DeepLabV3+ semantic-seg baselines with K-fold CV.")
    p.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    p.add_argument("--arch", choices=["unet", "deeplabv3plus"], default="unet")
    p.add_argument("--encoder", type=str, default="resnet50", help="smp encoder (ImageNet-pretrained).")
    p.add_argument("--data_root", type=str, default=".")
    p.add_argument("--img_dir", type=str, default=None)
    p.add_argument("--mask_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--n_folds", type=int, default=3)
    p.add_argument("--cv_strategy", type=str, default="stratified", choices=["stratified", "random"])
    p.add_argument("--group_by_stem", action="store_true",
                   help="Augmentations TRAINING-ONLY; validate on clean originals (pre-augmented dataset).")
    p.add_argument("--group_pattern", type=str, default=r"_aug\d+$")
    p.add_argument("--num_augmentations_per_img", type=int, default=None,
                   help="Cap augmentations per original used for training (only with --group_by_stem).")
    p.add_argument("--merge_class_map", type=str, default=None,
                   help="Combine/remove classes, e.g. '12:1,10:1' (multiclass only).")
    p.add_argument("--ignore_artifacts", action="store_true", help="Route scale bar + watermark to ignore.")
    p.add_argument("--ignore_scale_bar", action="store_true", help="Route only scale bar to ignore.")
    p.add_argument("--loss_type", type=str, default="focal", choices=["ce", "focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--auto_class_weights", action="store_true",
                   help="Inverse-frequency class weights estimated from the training fold.")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--fold", type=int, default=None, help="Run only this fold; otherwise all folds.")
    p.add_argument("--no_train", action="store_true", help="Build loaders and exit.")
    p.add_argument("--max_steps_per_epoch", type=int, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_mean_only", action="store_true",
                   help="Log only the cross-fold-averaged W&B run, not per-fold runs.")
    return p.parse_args()


def build_scheduler(optimizer, args):
    if args.scheduler == "none" and args.warmup_epochs <= 0:
        return None
    main_epochs = max(1, args.epochs - max(0, args.warmup_epochs))
    main_sched = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs)
        if args.scheduler == "cosine" else None
    )
    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                                             total_iters=args.warmup_epochs)
        if main_sched is None:
            return warmup
        return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched],
                                               milestones=[args.warmup_epochs])
    return main_sched


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def run_single_fold(fold, n_folds, train_idx, val_idx, cfg, args, device):
    task, num_classes, class_names = cfg["task"], cfg["num_classes"], cfg["class_names"]
    ignore_ids, merge_map = cfg["ignore_ids"], cfg["merge_map"]
    crop = args.crop

    train_tf = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.2),
        RandomCrop((crop, crop), pad_if_needed=True, fill={tv_tensors.Mask: IGNORE_INDEX, "others": 0}),
    ])
    val_tf = Compose([CenterCrop((crop, crop))])

    train_full = make_dataset(task, args.data_root, cfg["img_dir"], cfg["mask_dir"], train_tf,
                              ignore_ids, merge_map, strict=False, print_pair_count=False)
    val_full = make_dataset(task, args.data_root, cfg["img_dir"], cfg["mask_dir"], val_tf,
                            ignore_ids, merge_map, strict=False, print_pair_count=False)
    train_ds, val_ds = Subset(train_full, train_idx), Subset(val_full, val_idx)

    on_gpu = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=on_gpu, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=on_gpu)

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{args.arch}/{task}] Fold {fold + 1}/{n_folds} | Train {len(train_ds)} | Val {len(val_ds)}")
    if args.no_train:
        return {"fold": fold, "status": "no_train"}

    model = build_baseline_model(args.arch, args.encoder, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)

    class_weights = None
    if args.auto_class_weights:
        class_weights = estimate_class_weights_from_dataset(train_ds, num_classes, IGNORE_INDEX, device)
        print("[train] auto class weights:", class_weights.detach().cpu().numpy().round(3).tolist())
    print(f"[train] arch={args.arch} encoder={args.encoder} loss={args.loss_type} "
          f"gamma={args.focal_gamma} scheduler={args.scheduler} warmup={args.warmup_epochs}")

    # --- W&B per-fold run (suppressed with --wandb_mean_only) ---
    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled" and not args.wandb_mean_only:
        if not _WANDB_AVAILABLE:
            print("[wandb] wandb not installed; skipping. `pip install wandb`.")
        else:
            import wandb as _wandb

            name = args.wandb_run_name
            if n_folds >= 2:
                name = f"{name}_fold{fold}" if name else f"fold{fold}"
            wandb_run = _wandb.init(
                project=args.wandb_project, entity=args.wandb_entity, name=name,
                mode=args.wandb_mode, group=args.wandb_run_name or None, reinit=True,
                config={"stage": "baseline_finetune", "arch": args.arch, "encoder": args.encoder,
                        "task": task, "num_classes": num_classes, "epochs": args.epochs,
                        "lr": args.lr, "crop": args.crop, "loss_type": args.loss_type,
                        "focal_gamma": args.focal_gamma, "auto_class_weights": args.auto_class_weights,
                        "warmup_epochs": args.warmup_epochs, "scheduler": args.scheduler,
                        "n_folds": n_folds, "fold": fold, "cv_strategy": args.cv_strategy,
                        "seed": args.seed},
            )
            print(f"[wandb] logging to {args.wandb_project} run={wandb_run.name}")

    best_miou, best_row = -1.0, None
    ckpt_path = fold_dir / f"best_{args.arch}_{task}.pth"
    metrics_csv = fold_dir / "val_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,pixel_acc,miou," + ",".join(f"iou_{i}" for i in range(num_classes)) + "\n")
    metric_history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, device, epoch, class_weights,
                             args.loss_type, args.focal_gamma, scheduler, args.max_steps_per_epoch)
        va_loss, va_acc, va_miou, per_iou = evaluate(model, val_loader, device, num_classes,
                                                     class_weights, args.loss_type, args.focal_gamma)
        print(f"Fold {fold} | Epoch {epoch:02d} | train {tr:.4f} | val {va_loss:.4f} | "
              f"acc {va_acc:.3f} | mIoU {va_miou:.3f}")

        ious = [float(per_iou[i]) if torch.isfinite(per_iou[i]).item() else float("nan") for i in range(num_classes)]
        with metrics_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr:.6f},{va_loss:.6f},{va_acc:.6f},{va_miou:.6f}," +
                    ",".join(f"{v:.6f}" if v == v else "nan" for v in ious) + "\n")

        epoch_metrics = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]),
                         "train/loss": float(tr), "val/loss": float(va_loss),
                         "val/pixel_acc": float(va_acc), "val/mIoU": float(va_miou)}
        for i, nm in enumerate(class_names[:num_classes]):
            if ious[i] == ious[i]:  # finite
                epoch_metrics[f"val_iou/{i:02d}_{nm}"] = ious[i]
        metric_history.append(epoch_metrics)
        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=epoch)

        if va_miou > best_miou:
            best_miou = va_miou
            best_row = {"epoch": epoch, "miou": float(va_miou), "pixel_acc": float(va_acc),
                        "val_loss": float(va_loss)}
            torch.save({"model_state": model.state_dict(), "arch": args.arch, "encoder": args.encoder,
                        "task": task, "num_classes": num_classes, "fold": fold,
                        "per_class_val_iou": per_iou}, ckpt_path)
            print(f"  saved {ckpt_path}")
            if wandb_run is not None:
                wandb_run.summary["best_val_mIoU"] = best_miou
                wandb_run.summary["best_epoch"] = epoch

    if wandb_run is not None:
        wandb_run.finish()

    assert best_row is not None
    best_row["fold"] = fold
    best_row["best_checkpoint"] = str(ckpt_path)
    best_row["history"] = metric_history
    return best_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    use_explicit = args.img_dir is not None and args.mask_dir is not None
    img_dir = args.img_dir if use_explicit else None
    mask_dir = args.mask_dir if use_explicit else None

    # Task-specific config.
    if args.task == "binary":
        num_classes, class_names = NUM_BINARY_CLASSES, BINARY_CLASS_NAMES
        ignore_ids, merge_map = ARTIFACT_CLASS_IDS, {}
    else:
        num_classes, class_names = NUM_CLASSES, CLASS_NAMES
        ignore_ids = artifact_ignore_ids(args)
        merge_map = parse_merge_map(args.merge_class_map)
        if ignore_ids:
            print(f"[config] Ignoring artifact classes (-> {IGNORE_INDEX}): {list(ignore_ids)}")
        if merge_map:
            pairs = ", ".join(f"{s}:{CLASS_NAMES[s]} -> {d}:{CLASS_NAMES[d] if d < len(CLASS_NAMES) else d}"
                              for s, d in merge_map.items())
            print(f"[config] Merging classes (masks + presence): {pairs}")

    cfg = {"task": args.task, "num_classes": num_classes, "class_names": class_names,
           "ignore_ids": ignore_ids, "merge_map": merge_map, "img_dir": img_dir, "mask_dir": mask_dir}

    probe = make_dataset(args.task, args.data_root, img_dir, mask_dir, None, ignore_ids, merge_map,
                         strict=True, print_pair_count=True)
    n = len(probe)

    # CV split (presence is over the ORIGINAL 18-class scheme for both tasks).
    presence = None
    if args.cv_strategy == "stratified":
        p_ignore = ignore_ids if args.task == "multiclass" else ARTIFACT_CLASS_IDS
        presence = build_class_presence_matrix(probe.pairs, NUM_CLASSES, IGNORE_INDEX, p_ignore,
                                               merge_map if args.task == "multiclass" else None)

    if args.group_by_stem:
        splits = augment_aware_kfold_indices(probe.pairs, args.group_pattern, presence,
                                             args.n_folds, args.seed,
                                             max_augs_per_source=args.num_augmentations_per_img)
        n_orig = sum(len(va) for _, va in splits)
        cap = f"; capped to {args.num_augmentations_per_img} augs/original" if args.num_augmentations_per_img is not None else ""
        print(f"[cv] augment-aware split: {n} files -> {n_orig} originals; augmentations are "
              f"TRAINING-ONLY and never in val{cap}.")
    elif args.cv_strategy == "stratified":
        splits = stratified_kfold_indices(presence, args.n_folds, args.seed)
    else:
        splits = kfold_train_val_indices(n, args.n_folds, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_folds))
    if args.fold is not None and not (0 <= args.fold < args.n_folds):
        raise SystemExit(f"--fold must be in [0, {args.n_folds}), got {args.fold}")

    print(f"[config] Baseline {args.arch}/{args.encoder} | task={args.task} | {args.n_folds}-fold "
          f"{args.cv_strategy} CV | seed={args.seed} | folds={folds_to_run} | n={n}")

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
    summary = {"arch": args.arch, "encoder": args.encoder, "task": args.task, "n_folds": args.n_folds,
               "per_fold_best": summary_rows,
               "mean_best_miou": float(np.mean(miou_vals)) if miou_vals else None,
               "std_best_miou": float(np.std(miou_vals)) if miou_vals else None}
    with (out_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== CV summary ===")
    print(json.dumps(summary, indent=2))

    # Single W&B run with each metric averaged across folds (one curve per metric).
    log_cv_mean_wandb(args, histories,
                      config_extra={"arch": args.arch, "encoder": args.encoder, "task": args.task})


if __name__ == "__main__":
    main()
