# -*- coding: utf-8 -*-
"""
Binary grain vs background segmentation (SwinV2-Tiny + UPerNet).

Maps original multiclass masks (0=background, 1–15=grain components, 255=ignore) to:
  0 = non-grain (background)
  1 = grain (any original class 1–15)

5-fold cross-validation over image/mask pairs. Metrics: pixel accuracy, mean IoU, per-class IoU.

Run::

    python swin_binary_segmentation_221.py --img_dir ... --mask_dir ... --output_dir ...

Requires: torch, torchvision, transformers, tqdm, pillow, numpy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.transforms.v2 import CenterCrop, Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms.v2 import functional as V2F
from tqdm.auto import tqdm

from transformers import UperNetConfig, UperNetForSemanticSegmentation

from swin_training_pipeline_221 import evaluate, set_seed, train_one_epoch

NUM_BINARY_CLASSES = 2
IGNORE_INDEX = 255
BACKBONE_ID = "microsoft/swinv2-tiny-patch4-window8-256"
BINARY_CLASS_NAMES = ("non_grain_background", "grain")


def _json_sanitize(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return obj

DEFAULT_GDRIVE_LABELED_IMG_DIR = (
    "/content/drive/My Drive/Petrographic images_ML work/labelled images_PS/labelled images_PS/my_dataset/img"
)
DEFAULT_GDRIVE_LABELED_MASK_DIR = (
    "/content/drive/My Drive/Petrographic images_ML work/labelled images_PS/labelled images_PS/my_dataset/masks_machine"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Binary grain segmentation with K-fold CV (UPerNet + SwinV2).")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--img_dir", type=str, default=DEFAULT_GDRIVE_LABELED_IMG_DIR)
    p.add_argument("--mask_dir", type=str, default=DEFAULT_GDRIVE_LABELED_MASK_DIR)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If set, run only this fold index in [0, n_folds). Otherwise run all folds.",
    )
    p.add_argument("--no_train", action="store_true", help="Build loaders and exit.")
    p.add_argument(
        "--backbone_checkpoint",
        type=str,
        default=None,
        help="Optional SSL checkpoint (backbone_state) from swin_ssl_pretrain_221.py.",
    )
    p.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=None,
        help="If set, cap training batches per epoch (smoke test).",
    )
    return p.parse_args()


def default_data_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent.parent / "data" / "carbonate_imgs_and_masks"


def kfold_train_val_indices(n: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, val_idx) for each fold; shuffled, stratification-free."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    parts = np.array_split(perm, n_folds)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_folds):
        val_idx = parts[f]
        train_idx = np.concatenate([parts[j] for j in range(n_folds) if j != f])
        splits.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
    return splits


def multiclass_mask_to_binary(mask_long: torch.Tensor) -> torch.Tensor:
    """0 -> 0 (non-grain), 1–15 -> 1 (grain), 255 -> ignore. Other values -> ignore."""
    m = mask_long
    out = torch.full_like(m, IGNORE_INDEX)
    known = (m >= 0) & (m <= 15)
    out[known & (m == 0)] = 0
    out[known & (m >= 1)] = 1
    return out


def get_binary_model(ignore_index: int = IGNORE_INDEX) -> UperNetForSemanticSegmentation:
    cfg = UperNetConfig(
        backbone=BACKBONE_ID,
        use_pretrained_backbone=True,
        backbone_kwargs={"out_indices": [0, 1, 2, 3]},
        num_labels=NUM_BINARY_CLASSES,
        loss_ignore_index=ignore_index,
        use_auxiliary_head=False,
    )
    return UperNetForSemanticSegmentation(cfg)


def load_ssl_backbone_checkpoint(model: UperNetForSemanticSegmentation, checkpoint_path: str | Path) -> None:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"backbone checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "backbone_state" in ckpt and isinstance(ckpt["backbone_state"], dict):
        ssl_sd = ckpt["backbone_state"]
    elif "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        ssl_sd = {
            k.replace("backbone.", "", 1): v
            for k, v in ckpt["model_state"].items()
            if k.startswith("backbone.")
        }
    else:
        raise KeyError("Checkpoint missing 'backbone_state' or compatible 'model_state'.")

    missing, unexpected = model.backbone.load_state_dict(ssl_sd, strict=False)
    print(
        "[ssl->binary] loaded backbone:",
        ckpt_path,
        f"\n  missing={len(missing)} unexpected={len(unexpected)}",
    )


class BinaryCarbonateDataset(torch.utils.data.Dataset):
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
        print_pair_count: bool = True,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.normalize = normalize
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

        if strict:
            if not self.pairs:
                raise RuntimeError("Found no (image, mask) pairs.")
        if self._print_pair_count:
            print(f"[binary dataset] {len(self.pairs)} paired samples (multiclass -> binary in __getitem__).")

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        img = read_image(str(img_path))
        mask = read_image(str(mask_path))
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0:1, ...]
        mask = mask.squeeze(0).to(torch.long)
        mask = multiclass_mask_to_binary(mask)

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

    def __len__(self) -> int:
        return len(self.pairs)


class _CappedLoader:
    """Wrap a DataLoader to yield at most ``max_steps`` batches per epoch."""

    def __init__(self, loader: DataLoader, max_steps: int | None):
        self.loader = loader
        self.max_steps = max_steps

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if self.max_steps is not None and i >= self.max_steps:
                break
            yield batch

    def __len__(self) -> int:
        if self.max_steps is None:
            return len(self.loader)
        return min(self.max_steps, len(self.loader))


def print_binary_per_class_iou(per_iou: torch.Tensor, epoch: int) -> None:
    print(f"  per-class val IoU (epoch {epoch:02d}):")
    vec = per_iou.detach().cpu()
    for i, name in enumerate(BINARY_CLASS_NAMES[: len(vec)]):
        v = vec[i]
        if torch.isfinite(v).item():
            print(f"    {name} IoU: {float(v):.4f}")
        else:
            print(f"    {name} IoU: nan")


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    candidate_img_dir = Path(args.img_dir) if args.img_dir else None
    candidate_mask_dir = Path(args.mask_dir) if args.mask_dir else None
    use_explicit = bool(
        candidate_img_dir
        and candidate_mask_dir
        and candidate_img_dir.is_dir()
        and candidate_mask_dir.is_dir()
    )
    if use_explicit:
        data_root = Path(args.data_root) if args.data_root else Path(".")
        return data_root, candidate_img_dir, candidate_mask_dir
    if args.data_root is None:
        data_root = default_data_root()
        if not (data_root / "img").is_dir():
            raise SystemExit(f"Missing img/ under {data_root}; pass --img_dir/--mask_dir or --data_root.")
        print(f"[config] data_root: {data_root}")
        return data_root, None, None
    data_root = Path(args.data_root)
    return data_root, None, None


def run_single_fold(
    fold: int,
    train_idx: list[int],
    val_idx: list[int],
    data_root: Path,
    img_dir: Path | None,
    mask_dir: Path | None,
    use_explicit: bool,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    crop = args.crop
    train_tf = Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.2),
            RandomCrop((crop, crop)),
        ]
    )
    val_tf = Compose([CenterCrop((crop, crop))])

    train_full = BinaryCarbonateDataset(
        data_root,
        img_dir=img_dir if use_explicit else None,
        mask_dir=mask_dir if use_explicit else None,
        transforms=train_tf,
        normalize=True,
        strict=False,
        print_pair_count=False,
    )
    val_full = BinaryCarbonateDataset(
        data_root,
        img_dir=img_dir if use_explicit else None,
        mask_dir=mask_dir if use_explicit else None,
        transforms=val_tf,
        normalize=True,
        strict=False,
        print_pair_count=False,
    )
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    on_gpu = device.type == "cuda"
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

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Fold {fold} | train {len(train_ds)} | val {len(val_ds)} ===")

    if args.no_train:
        return {"fold": fold, "status": "no_train"}

    model = get_binary_model(IGNORE_INDEX)
    if args.backbone_checkpoint:
        load_ssl_backbone_checkpoint(model, args.backbone_checkpoint)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_miou = -1.0
    best_row: dict | None = None
    ckpt_path = fold_dir / "best_upernet_swinv2_binary.pth"
    metrics_csv = fold_dir / "val_metrics.csv"

    with metrics_csv.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,pixel_acc,miou,iou_bg,iou_grain\n")

    for epoch in range(1, args.epochs + 1):
        effective_loader = (
            _CappedLoader(train_loader, args.max_steps_per_epoch)
            if args.max_steps_per_epoch
            else train_loader
        )

        tr = train_one_epoch(
            model,
            effective_loader,
            optimizer,
            device,
            epoch,
            NUM_BINARY_CLASSES,
            IGNORE_INDEX,
            None,
            scheduler=None,
        )
        va_loss, va_acc, va_miou, per_iou = evaluate(
            model, val_loader, device, NUM_BINARY_CLASSES, IGNORE_INDEX, None
        )
        print(
            f"Fold {fold} | Epoch {epoch:02d} | train {tr:.4f} | val {va_loss:.4f} | "
            f"acc {va_acc:.4f} | mIoU {va_miou:.4f}"
        )
        print_binary_per_class_iou(per_iou, epoch)

        i0 = float(per_iou[0].item()) if torch.isfinite(per_iou[0]).item() else float("nan")
        i1 = float(per_iou[1].item()) if torch.isfinite(per_iou[1]).item() else float("nan")
        with metrics_csv.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr:.6f},{va_loss:.6f},{va_acc:.6f},{va_miou:.6f},{i0:.6f},{i1:.6f}\n")

        if va_miou > best_miou:
            best_miou = va_miou
            best_row = {
                "epoch": epoch,
                "val_loss": float(va_loss),
                "pixel_acc": float(va_acc),
                "miou": float(va_miou),
                "iou_non_grain_background": i0,
                "iou_grain": i1,
            }
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": NUM_BINARY_CLASSES,
                    "ignore_index": IGNORE_INDEX,
                    "backbone_id": BACKBONE_ID,
                    "fold": fold,
                    "per_class_val_iou": per_iou.cpu(),
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path}")

    assert best_row is not None
    best_row["fold"] = fold
    best_row["best_checkpoint"] = str(ckpt_path)
    return best_row


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root, img_dir, mask_dir = resolve_data_paths(args)
    use_explicit = img_dir is not None and mask_dir is not None
    if use_explicit:
        print(f"[config] img={img_dir}\n  mask={mask_dir}")

    probe = BinaryCarbonateDataset(
        data_root,
        img_dir=img_dir if use_explicit else None,
        mask_dir=mask_dir if use_explicit else None,
        transforms=None,
        normalize=True,
        strict=True,
    )
    n = len(probe)
    if n < args.n_folds:
        raise SystemExit(f"Need at least {args.n_folds} samples for {args.n_folds}-fold CV; found {n}.")

    split_folds = kfold_train_val_indices(n, args.n_folds, args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds_to_run: list[int]
    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.n_folds:
            raise SystemExit(f"--fold must be in [0, {args.n_folds}), got {args.fold}")
        folds_to_run = [args.fold]
    else:
        folds_to_run = list(range(args.n_folds))

    all_best: list[dict] = []
    for fold in folds_to_run:
        tr, va = split_folds[fold]
        train_idx, val_idx = tr.tolist(), va.tolist()
        result = run_single_fold(
            fold,
            train_idx,
            val_idx,
            data_root,
            img_dir,
            mask_dir,
            use_explicit,
            args,
            device,
        )
        if result.get("status") != "no_train":
            all_best.append(result)

    if args.no_train:
        return

    summary_path = out_dir / "cv_summary.json"
    miou_vals = [r["miou"] for r in all_best]
    acc_vals = [r["pixel_acc"] for r in all_best]
    summary = {
        "n_folds": args.n_folds,
        "folds_completed": folds_to_run,
        "per_fold_best_at_checkpoint": all_best,
        "mean_best_miou": float(np.mean(miou_vals)) if miou_vals else None,
        "std_best_miou": float(np.std(miou_vals)) if miou_vals else None,
        "mean_best_pixel_acc": float(np.mean(acc_vals)) if acc_vals else None,
        "std_best_pixel_acc": float(np.std(acc_vals)) if acc_vals else None,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(summary), f, indent=2)
    print("\n=== Cross-validation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
