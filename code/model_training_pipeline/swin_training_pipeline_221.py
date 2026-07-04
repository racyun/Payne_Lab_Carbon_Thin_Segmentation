# -*- coding: utf-8 -*-
"""
SwinV2-Tiny (ImageNet) + UPerNet semantic segmentation for carbonate thin sections.

Pairs images under ``<data_root>/img`` with masks under ``<data_root>/masks`` (same stem).
Default: 18 classes (labels 0–17). Trains with AdamW; saves best checkpoint by validation mIoU.
Classes 11 (scale bar) and 17 (watermark) are non-biological artifacts; pass
``--ignore_artifacts`` to remap them to ignore_index so they are excluded from the loss.

By default this runs 3-fold cross-validation (``--n_folds 3``): each fold trains a fresh
model on 2/3 of the data and validates on the held-out 1/3, writing per-fold checkpoints
under ``<output_dir>/fold_<i>/`` and an aggregate ``cv_summary.json`` (mean/std mIoU).
Folds are built by multi-label *stratification* over per-image class presence
(``--cv_strategy stratified``, the default) so each class is spread as evenly as possible
across folds; pass ``--cv_strategy random`` for a plain shuffled split.
Set ``--n_folds 1`` to fall back to a single train/val split governed by ``--val_frac``.

Run locally::

    python swin_training_pipeline_221.py --data_root /path/to/carbonate_imgs_and_masks

Requires: torch, torchvision, transformers, tqdm, pillow, numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants (carbonate labeling: 0–17 => 18 logits)
# ---------------------------------------------------------------------------

NUM_CLASSES = 18
IGNORE_INDEX = 255
SCALE_BAR_CLASS_ID = 11
WATERMARK_CLASS_ID = 17
# Non-biological annotation artifacts. With --ignore_artifacts these are remapped to
# ignore_index so they neither contribute to the loss nor appear in per-class metrics.
ARTIFACT_CLASS_IDS = (SCALE_BAR_CLASS_ID, WATERMARK_CLASS_ID)
BACKBONE_ID = "microsoft/swinv2-tiny-patch4-window8-256"

# Label ids 0–17 (must align with NUM_CLASSES).
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
    "sponge",
    "watermark",
)
DEFAULT_GDRIVE_LABELED_IMG_DIR = (
    "/content/drive/My Drive/Petrographic images_ML work/labelled images_PS/labelledDataset_02032026/my_dataset/img"
)
DEFAULT_GDRIVE_LABELED_MASK_DIR = (
    "/content/drive/My Drive/Petrographic images_ML work/labelled images_PS/labelledDataset_02032026/my_dataset/masks_machine"
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
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Linear LR warmup over the first N epochs (0 disables). Combines with --scheduler.",
    )
    p.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function. 'ce' = standard cross-entropy; 'focal' = focal CE for imbalance.",
    )
    p.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter for focal loss (only used when --loss_type=focal).",
    )
    p.add_argument("--crop", type=int, default=512, help="Train random crop and val center crop size.")
    p.add_argument(
        "--n_folds",
        type=int,
        default=3,
        help="K for K-fold cross-validation (default 3). Set to 1 (or 0) to use a single "
        "train/val split governed by --val_frac instead.",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="If set, run only this fold index in [0, n_folds); otherwise run all folds. "
        "Useful for parallelizing folds across separate sessions.",
    )
    p.add_argument(
        "--cv_strategy",
        type=str,
        default="stratified",
        choices=["stratified", "random"],
        help="How to build the K folds. 'stratified' = multi-label iterative stratification "
        "over per-image class presence, spreading each class as evenly as possible across folds "
        "(every class with >=2 images lands in all K training folds). 'random' = plain shuffled "
        "split (legacy). Only used when --n_folds >= 2.",
    )
    p.add_argument(
        "--group_by_stem",
        action="store_true",
        help="Train on a pre-augmented dataset correctly: augmentations (files matching "
        "--group_pattern, e.g. 00_TK-0.3_aug03) are TRAINING-ONLY. Folds are built over the "
        "original images, each validation fold holds only clean originals, and an image's "
        "augmentations follow it into training (augs of a val original are dropped). No leakage, "
        "no evaluation on distorted images, and val sets match a no-aug run on the same originals.",
    )
    p.add_argument(
        "--group_pattern",
        type=str,
        default=r"_aug\d+$",
        help="Regex stripped from each file stem to derive its group key (default matches the "
        "'_augNN' suffix written by the augmentation export). Only used with --group_by_stem.",
    )
    p.add_argument(
        "--num_augmentations_per_img",
        type=int,
        default=None,
        help="Cap how many augmentations of each original image are used for TRAINING (keeps the "
        "lowest-numbered, e.g. 5 -> aug00..aug04). Default None = use all. Only used with "
        "--group_by_stem; does not affect validation (which is always clean originals).",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Validation fraction for the single-split path (only used when --n_folds <= 1).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=".", help="Directory for checkpoints and viz.")
    p.add_argument("--no_train", action="store_true", help="Only build data loaders and exit (smoke test).")
    p.add_argument(
        "--ignore_scale_bar",
        action="store_true",
        help=f"Remap mask class {SCALE_BAR_CLASS_ID} (scale bar) to ignore_index so it is not learned. "
        "Subset of --ignore_artifacts; kept for backward compatibility.",
    )
    p.add_argument(
        "--ignore_artifacts",
        action="store_true",
        help=f"Remap all non-biological artifact classes {ARTIFACT_CLASS_IDS} "
        "(scale bar, watermark) to ignore_index so they are excluded from loss and metrics. "
        "Takes precedence over --ignore_scale_bar.",
    )
    p.add_argument(
        "--merge_class_map",
        type=str,
        default="",
        help="Combine classes by remapping mask pixel values at load time. Comma-separated "
        "'src:dst' pairs, e.g. '12:1' merges mollusk (12) into bivalves (1). NUM_CLASSES is "
        "unchanged, so a merged source id becomes an unused (empty) class in the metrics. "
        "Applied to both training masks and the stratification presence matrix.",
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
    # Weights & Biases logging.
    p.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name. If unset, W&B logging is disabled.",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name (auto-generated if unset). Per-fold runs get a '_fold<i>' suffix.",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity (team or user).",
    )
    p.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode (online uploads live; offline writes locally; disabled = no logging).",
    )
    return p.parse_args()


def default_data_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent.parent / "data" / "carbonate_imgs_and_masks"


def artifact_ignore_ids(args: argparse.Namespace) -> tuple[int, ...]:
    """Resolve which class ids to remap to ignore_index from the CLI flags.

    --ignore_artifacts (all artifacts) takes precedence over --ignore_scale_bar (scale bar only).
    """
    if getattr(args, "ignore_artifacts", False):
        return ARTIFACT_CLASS_IDS
    if getattr(args, "ignore_scale_bar", False):
        return (SCALE_BAR_CLASS_ID,)
    return ()


def parse_merge_map(spec: str) -> dict[int, int]:
    """Parse a '--merge_class_map' string ('12:1,15:8') into a {src: dst} dict.

    Each src/dst must be a valid class id in [0, NUM_CLASSES) (dst may also be IGNORE_INDEX
    to drop a class), and src != dst. Returns {} for an empty/None spec.
    """
    if not spec:
        return {}
    merge: dict[int, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        try:
            src_s, dst_s = pair.split(":")
            src, dst = int(src_s), int(dst_s)
        except ValueError as exc:
            raise SystemExit(f"--merge_class_map: bad pair {pair!r}; expected 'src:dst'.") from exc
        if not (0 <= src < NUM_CLASSES):
            raise SystemExit(f"--merge_class_map: src {src} out of range [0, {NUM_CLASSES}).")
        if not (0 <= dst < NUM_CLASSES or dst == IGNORE_INDEX):
            raise SystemExit(f"--merge_class_map: dst {dst} must be in [0, {NUM_CLASSES}) or {IGNORE_INDEX}.")
        if src == dst:
            raise SystemExit(f"--merge_class_map: src == dst ({src}) is a no-op.")
        merge[src] = dst
    return merge


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


def build_class_presence_matrix(
    pairs: list[tuple[Path, Path]],
    num_classes: int,
    ignore_index: int,
    ignore_class_ids: tuple[int, ...] = (),
    merge_class_map: dict[int, int] | None = None,
) -> np.ndarray:
    """Binary [n_samples, num_classes] matrix: 1 where class c is present in image i's mask.

    Reads masks at full resolution (no crop transforms), so presence reflects the whole
    image. Mirrors the artifact remapping AND the --merge_class_map remapping in
    CarbonateSegmentationDataset.__getitem__ so folds stratify on the same (merged) labels.
    """
    n = len(pairs)
    presence = np.zeros((n, num_classes), dtype=np.uint8)
    for i, (_, mask_path) in enumerate(pairs):
        mask = read_image(str(mask_path))
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask[0:1, ...]
        mask = mask.squeeze(0).to(torch.long)
        if ignore_class_ids or merge_class_map:
            mask = mask.clone()
            for cid in ignore_class_ids:
                mask[mask == cid] = ignore_index
            for src, dst in (merge_class_map or {}).items():
                mask[mask == src] = dst
        for v in torch.unique(mask).tolist():
            if 0 <= v < num_classes:
                presence[i, v] = 1
    return presence


def report_class_fold_feasibility(presence: np.ndarray, n_folds: int) -> None:
    """Print per-class image counts and warn about classes too rare to cover every fold."""
    counts = presence.sum(axis=0).astype(int)
    num_classes = presence.shape[1]
    print(f"[cv] Stratified {n_folds}-fold over multi-label class presence ({presence.shape[0]} images).")
    print(
        "[cv] Per-class image counts: "
        + ", ".join(
            f"{(CLASS_NAMES[c] if c < len(CLASS_NAMES) else c)}={int(counts[c])}"
            for c in range(num_classes)
            if counts[c] > 0
        )
    )
    rare = [c for c in range(num_classes) if 0 < counts[c] < n_folds]
    absent = [c for c in range(num_classes) if counts[c] == 0]
    if rare:
        print(
            f"[cv][warn] {len(rare)} class(es) appear in fewer than {n_folds} images, so they "
            "cannot be present in every validation fold (continuing best-effort):"
        )
        for c in rare:
            name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            print(f"    - {c:02d} {name}: in {int(counts[c])} image(s)")
    if absent:
        names = ", ".join((CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)) for c in absent)
        print(f"[cv] Note: {len(absent)} class(es) absent from the dataset entirely: {names}")


def stratified_kfold_indices(
    presence: np.ndarray, n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Multi-label iterative stratification (Sechidis et al. 2011).

    Assigns each sample to one validation fold so that, for every class, its images are
    spread as evenly as possible across the K folds. Returns (train_idx, val_idx) per fold,
    matching the kfold_train_val_indices API. Deterministic given ``seed`` (used only for
    tie-breaking).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    n, num_classes = presence.shape
    rng = np.random.RandomState(seed)

    # Desired number of samples per fold, and desired count of each class per fold.
    fold_capacity = np.full(n_folds, n / n_folds, dtype=float)
    desired = np.outer(np.ones(n_folds), presence.sum(axis=0).astype(float)) / n_folds

    fold_of = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)

    def pick_fold(candidates: np.ndarray) -> int:
        """Among candidate folds, prefer most remaining capacity; break ties randomly."""
        caps = fold_capacity[candidates]
        best = candidates[caps == caps.max()]
        return int(best[rng.randint(best.size)])

    while remaining.any():
        rem_idx = np.where(remaining)[0]
        rem_label_counts = presence[rem_idx].sum(axis=0).astype(float)
        positive = np.where(rem_label_counts > 0)[0]

        if positive.size == 0:
            # Label-free leftovers (e.g. a mask that is all ignore): place by capacity.
            for i in rem_idx:
                j = pick_fold(np.arange(n_folds))
                fold_of[i] = j
                fold_capacity[j] -= 1
                remaining[i] = False
            break

        # Rarest still-unassigned label first (ties -> smallest class id).
        label = int(positive[np.argmin(rem_label_counts[positive])])
        samples = rem_idx[presence[rem_idx, label] > 0]
        for i in samples:
            if not remaining[i]:
                continue
            col = desired[:, label]
            cand = np.where(col == col.max())[0]
            j = pick_fold(cand)
            fold_of[i] = j
            remaining[i] = False
            present_labels = np.where(presence[i] > 0)[0]
            desired[j, present_labels] -= 1.0
            fold_capacity[j] -= 1.0

    all_idx = np.arange(n)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_folds):
        val_idx = np.sort(all_idx[fold_of == f])
        train_idx = np.sort(all_idx[fold_of != f])
        splits.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
    return splits


def group_members_by_stem(
    pairs: list[tuple[Path, Path]], pattern: str = r"_aug\d+$"
) -> list[np.ndarray]:
    """Group image indices by source stem (with ``pattern`` stripped).

    An original and its augmentations share a base stem once the ``_augNN`` suffix is
    removed, so e.g. ``00_TK-0.3`` and ``00_TK-0.3_aug03`` land in one group. Returns a list
    of int64 arrays of image indices, one per group, in first-seen order.
    """
    rx = re.compile(pattern)
    order: dict[str, int] = {}
    members: list[list[int]] = []
    for i, (img_path, _) in enumerate(pairs):
        key = rx.sub("", img_path.stem)
        if key not in order:
            order[key] = len(members)
            members.append([])
        members[order[key]].append(i)
    return [np.asarray(m, dtype=np.int64) for m in members]


def _expand_group_splits(
    group_splits: list[tuple[np.ndarray, np.ndarray]], members: list[np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert group-level (train, val) index arrays into image-level index arrays."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    empty = np.array([], dtype=np.int64)
    for tr_g, va_g in group_splits:
        tr = np.concatenate([members[int(g)] for g in tr_g]) if len(tr_g) else empty
        va = np.concatenate([members[int(g)] for g in va_g]) if len(va_g) else empty
        out.append((np.sort(tr).astype(np.int64), np.sort(va).astype(np.int64)))
    return out


def grouped_kfold_indices(
    members: list[np.ndarray], n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random K-fold over GROUPS (families), expanded to image indices."""
    return _expand_group_splits(kfold_train_val_indices(len(members), n_folds, seed), members)


def stratified_grouped_kfold_indices(
    presence: np.ndarray, members: list[np.ndarray], n_folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over GROUPS, expanded to image indices.

    Builds a group-level presence row (a class is 'present' in a group if any member has it),
    stratifies the groups, then expands to image indices — so each family stays in one fold.
    """
    gpres = np.zeros((len(members), presence.shape[1]), dtype=presence.dtype)
    for gi, mem in enumerate(members):
        gpres[gi] = (presence[mem].sum(axis=0) > 0).astype(presence.dtype)
    return _expand_group_splits(stratified_kfold_indices(gpres, n_folds, seed), members)


def augment_aware_kfold_indices(
    pairs: list[tuple[Path, Path]],
    pattern: str,
    presence: np.ndarray | None,
    n_folds: int,
    seed: int,
    max_augs_per_source: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold split where augmentations are TRAINING-ONLY and every validation fold is clean originals.

    Files whose stem matches ``pattern`` (default the ``_augNN`` suffix) are treated as augmented
    copies of a source ORIGINAL. The K folds are built over the original images only — stratified
    over ``presence`` when given, else random. For each fold:

      - validation = the original images assigned to that fold (clean, un-augmented),
      - training   = the remaining originals PLUS every augmentation whose source original is in
        training.

    Augmentations of a validation original are dropped entirely (never trained on, never
    evaluated). This removes both CV leakage AND evaluation on distorted images, and — because the
    split is computed over the originals only — a with-augmentation run and a no-augmentation run
    on the same originals (same seed) get identical validation folds, so they are directly
    comparable.

    ``max_augs_per_source`` caps how many augmentations of each original are used for training:
    the lowest-numbered ``_augNN`` are kept (e.g. 5 -> aug00..aug04), the rest ignored. ``None``
    uses all available. Validation is unaffected (it never contains augmentations), so changing
    this does not change the val sets.
    """
    rx = re.compile(pattern)
    digit_rx = re.compile(r"\d+")
    orig_pos: list[int] = []
    orig_stems: list[str] = []
    aug_by_src: dict[str, list[tuple[int, int]]] = {}  # src -> [(aug_index, global_index)]
    for i, (img_path, _) in enumerate(pairs):
        stem = img_path.stem
        m = rx.search(stem)
        if m:
            src = rx.sub("", stem)
            dm = digit_rx.search(m.group(0))
            aug_index = int(dm.group()) if dm else 0
            aug_by_src.setdefault(src, []).append((aug_index, i))
        else:
            orig_pos.append(i)
            orig_stems.append(stem)
    # Keep the lowest-numbered augmentations per original (cap), then reduce to plain index lists.
    for src, items in aug_by_src.items():
        ordered = sorted(items)  # by aug_index
        if max_augs_per_source is not None:
            ordered = ordered[:max_augs_per_source]
        aug_by_src[src] = [gi for _, gi in ordered]
    orig_pos_arr = np.asarray(orig_pos, dtype=np.int64)
    if len(orig_pos_arr) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} ORIGINAL (non-augmented) images for {n_folds}-fold CV; "
            f"found {len(orig_pos_arr)} (pattern {pattern!r})."
        )

    # Fold assignment over ORIGINALS only (local indices 0..len(orig)-1).
    if presence is not None:
        orig_splits = stratified_kfold_indices(presence[orig_pos_arr], n_folds, seed)
    else:
        orig_splits = kfold_train_val_indices(len(orig_pos_arr), n_folds, seed)

    all_local = np.arange(len(orig_pos_arr))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for _, val_local in orig_splits:
        val_local = np.asarray(val_local, dtype=np.int64)
        val_stems = {orig_stems[p] for p in val_local.tolist()}
        train_local = np.setdiff1d(all_local, val_local, assume_unique=False)
        train_global = list(orig_pos_arr[train_local])
        # Augs follow their source original into TRAIN; augs of a val original are dropped.
        for src, aug_list in aug_by_src.items():
            if src not in val_stems:
                train_global.extend(aug_list)
        splits.append(
            (
                np.sort(np.asarray(train_global, dtype=np.int64)),
                np.sort(orig_pos_arr[val_local].astype(np.int64)),
            )
        )
    return splits


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

    # weights_only=False: SSL checkpoints written by swin_ssl_pretrain_221.py contain
    # numpy scalars; PyTorch 2.6+ refuses to unpickle those under weights_only=True.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
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
        ignore_class_ids: tuple[int, ...] = (),
        ignore_index: int = IGNORE_INDEX,
        print_pair_count: bool = True,
        merge_class_map: dict[int, int] | None = None,
    ):
        self.root = Path(root)
        self.transforms = transforms
        self.normalize = normalize
        self.ignore_class_ids = tuple(ignore_class_ids)
        self.ignore_index = ignore_index
        self.merge_class_map = dict(merge_class_map or {})
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

        if self.ignore_class_ids or self.merge_class_map:
            mask = mask.clone()
            for cid in self.ignore_class_ids:
                mask[mask == cid] = self.ignore_index
            # Combine classes (e.g. mollusk 12 -> bivalves 1) by rewriting pixel values.
            for src, dst in self.merge_class_map.items():
                mask[mask == src] = dst

        # Defensive: any label outside the [0, NUM_CLASSES) scheme cannot be predicted by
        # the model's NUM_CLASSES logits and would crash the loss (CUDA assertion
        # "cur_target < n_classes"). Map such stray labels to ignore_index. NOTE: this
        # silently drops out-of-range labels — if a mask systematically uses a new class
        # id >= NUM_CLASSES, that's a labeling/scheme issue to resolve, not stray pixels.
        oob = (mask >= NUM_CLASSES) & (mask != self.ignore_index)
        if bool(oob.any()):
            mask[oob] = self.ignore_index

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


def focal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None,
    ignore_index: int,
    gamma: float,
) -> torch.Tensor:
    """Focal cross-entropy: (1 - p_t)^gamma * CE.

    Down-weights pixels the model already classifies confidently, focusing
    gradient on the hard / under-learned classes. Reduces to plain weighted
    CE when gamma == 0.
    """
    ce = F.cross_entropy(
        logits, labels, weight=weight, ignore_index=ignore_index, reduction="none"
    )
    pt = torch.exp(-ce)
    focal = (1.0 - pt).clamp(min=0.0).pow(gamma) * ce
    valid = labels != ignore_index
    if valid.any():
        return focal[valid].mean()
    return focal.mean()


def compute_seg_loss(
    out,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    ignore_index: int,
    loss_type: str,
    focal_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, logits_at_label_resolution)."""
    logits = out.logits
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    if loss_type == "focal":
        loss = focal_cross_entropy(logits, labels, class_weights, ignore_index, focal_gamma)
    else:
        loss = F.cross_entropy(
            logits, labels, weight=class_weights, ignore_index=ignore_index
        )
    return loss, logits


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
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc=f"epoch {epoch:02d}", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Use HF's built-in CE loss only when nothing custom is requested.
        if class_weights is None and loss_type == "ce":
            out = model(pixel_values=imgs, labels=labels)
            loss = out.loss
        else:
            out = model(pixel_values=imgs)
            loss, _ = compute_seg_loss(
                out, labels, class_weights, ignore_index, loss_type, focal_gamma
            )

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
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
) -> tuple[float, float, float, torch.Tensor]:
    """Returns val_loss, pixel_acc, mIoU (from full-val confusion matrix), per-class IoU vector."""
    model.eval()
    total, n = 0.0, 0
    acc_sum, m = 0.0, 0
    cm_total = torch.zeros(num_classes, num_classes, device=device, dtype=torch.float32)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if class_weights is None and loss_type == "ce":
            out = model(pixel_values=imgs, labels=labels)
            loss = out.loss
        else:
            out = model(pixel_values=imgs)
            loss, _ = compute_seg_loss(
                out, labels, class_weights, ignore_index, loss_type, focal_gamma
            )

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


def render_predictions(model, val_ds, device, crop: int, args, viz_dir: Path) -> None:
    """Save prediction masks + 4-panel figures for the first ``--viz_samples`` val items."""
    model.eval()
    n_viz = max(1, min(args.viz_samples, len(val_ds)))
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


def run_single_fold(
    fold: int,
    n_folds: int,
    train_idx: list[int],
    val_idx: list[int],
    data_root: Path,
    img_dir: Path | None,
    mask_dir: Path | None,
    use_explicit_dirs: bool,
    args: argparse.Namespace,
    device: torch.device,
    on_gpu: bool,
) -> dict:
    """Train + validate one fold (or the single split when n_folds <= 1).

    Returns a dict of best-checkpoint metrics for the cross-validation summary.
    """
    # When cross-validating, isolate each fold's artifacts under fold_<i>/.
    # For the single-split path (n_folds <= 1) keep the historical flat layout.
    fold_dir = Path(args.output_dir) / f"fold_{fold}" if n_folds >= 2 else Path(args.output_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    tag = f"Fold {fold + 1}/{n_folds} | " if n_folds >= 2 else ""

    ignore_ids = artifact_ignore_ids(args)
    merge_map = parse_merge_map(args.merge_class_map)
    crop = args.crop
    train_transforms = Compose(
        [
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.2),
            # pad_if_needed: some labeled images (e.g. in ALL_LABELS) are smaller than
            # `crop` in a dimension; pad up to the crop size, filling the mask with
            # ignore_index so padded borders are not learned as a real class.
            RandomCrop(
                (crop, crop),
                pad_if_needed=True,
                fill={tv_tensors.Mask: IGNORE_INDEX, "others": 0},
            ),
        ]
    )
    # CenterCrop already pads (with 0) when an image is smaller than the crop, so it
    # does not crash on sub-crop images; padded mask borders default to class 0.
    val_transforms = Compose([CenterCrop((crop, crop))])

    train_full = CarbonateSegmentationDataset(
        data_root,
        img_dir=img_dir if use_explicit_dirs else None,
        mask_dir=mask_dir if use_explicit_dirs else None,
        transforms=train_transforms,
        normalize=True,
        strict=False,
        ignore_class_ids=ignore_ids,
        merge_class_map=merge_map,
        print_pair_count=False,
    )
    val_full = CarbonateSegmentationDataset(
        data_root,
        img_dir=img_dir if use_explicit_dirs else None,
        mask_dir=mask_dir if use_explicit_dirs else None,
        transforms=val_transforms,
        normalize=True,
        strict=False,
        ignore_class_ids=ignore_ids,
        merge_class_map=merge_map,
        print_pair_count=False,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

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

    print(f"[train] {tag}Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    if args.no_train:
        print("[config] --no_train set; exiting after dataloader smoke test.")
        return {"fold": fold, "status": "no_train"}

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

    # Build the main scheduler (post-warmup), then optionally chain a linear-warmup
    # SequentialLR in front of it.
    main_sched: torch.optim.lr_scheduler._LRScheduler | None = None
    main_epochs = max(1, args.epochs - max(0, args.warmup_epochs))
    if args.scheduler == "cosine":
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=main_epochs)
    elif args.scheduler == "step":
        main_sched = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.step_size), gamma=args.gamma
        )

    if args.warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=args.warmup_epochs
        )
        if main_sched is None:
            scheduler = warmup_sched
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_sched],
                milestones=[args.warmup_epochs],
            )
    else:
        scheduler = main_sched

    # ------------------------------------------------------------------
    # Weights & Biases setup (logged once, then per-epoch metrics below).
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb_project and args.wandb_mode != "disabled":
        if not _WANDB_AVAILABLE:
            print("[wandb] wandb not installed; skipping logging. `pip install wandb` to enable.")
        else:
            import wandb as _wandb  # local re-import for clarity

            run_name = args.wandb_run_name
            if n_folds >= 2:
                run_name = f"{run_name}_fold{fold}" if run_name else f"fold{fold}"
            wandb_run = _wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                mode=args.wandb_mode,
                group=args.wandb_run_name or None,
                reinit=True,
                config={
                    "stage": "finetune",
                    "backbone_id": BACKBONE_ID,
                    "num_classes": NUM_CLASSES,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "scheduler": args.scheduler,
                    "loss_type": args.loss_type,
                    "focal_gamma": args.focal_gamma,
                    "auto_class_weights": args.auto_class_weights,
                    "class_weights_path": args.class_weights,
                    "ignore_class_ids": list(ignore_ids),
                    "merge_class_map": {str(k): v for k, v in merge_map.items()},
                    "crop": args.crop,
                    "n_folds": n_folds,
                    "fold": fold,
                    "val_frac": args.val_frac,
                    "seed": args.seed,
                    "ssl_backbone_checkpoint": args.backbone_checkpoint,
                },
            )
            print(f"[wandb] logging to project={args.wandb_project} run={wandb_run.name}")

    best_miou = -1.0
    best_acc = 0.0
    best_epoch = -1
    best_per_iou: torch.Tensor | None = None
    ckpt_path = fold_dir / "best_upernet_swinv2.pth"
    per_class_log_path = fold_dir / "val_per_class_iou.csv"
    with per_class_log_path.open("w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(f"class_{i}" for i in range(NUM_CLASSES)) + "\n")

    # In-memory history for end-of-run W&B composite chart + table.
    epoch_history: list[int] = []
    per_class_history: list[list[float]] = [[] for _ in range(NUM_CLASSES)]

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
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
        )
        va_loss, va_acc, va_miou, per_iou = evaluate(
            model,
            val_loader,
            device,
            NUM_CLASSES,
            IGNORE_INDEX,
            class_weights,
            loss_type=args.loss_type,
            focal_gamma=args.focal_gamma,
        )
        print(
            f"{tag}Epoch {epoch:02d} | train {tr:.4f} | val {va_loss:.4f} | acc {va_acc:.3f} | mIoU {va_miou:.3f}"
        )
        print_per_class_iou(per_iou, epoch)
        with per_class_log_path.open("a", encoding="utf-8") as f:
            vals = ",".join(f"{float(v):.6f}" if torch.isfinite(v) else "nan" for v in per_iou.detach().cpu())
            f.write(f"{epoch},{vals}\n")

        # Track per-class IoU history (used for the end-of-run composite chart + table).
        epoch_history.append(epoch)
        per_iou_cpu = per_iou.detach().cpu()
        for i in range(NUM_CLASSES):
            v = per_iou_cpu[i].item()
            per_class_history[i].append(float(v) if torch.isfinite(per_iou_cpu[i]).item() else float("nan"))

        # ------- Weights & Biases per-epoch metrics -------
        if wandb_run is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            log_payload = {
                "epoch": epoch,
                "lr": current_lr,
                "train/loss": tr,
                "val/loss": va_loss,
                "val/pixel_acc": va_acc,
                "val/mIoU": va_miou,
            }
            for i, name in enumerate(CLASS_NAMES[: per_iou.numel()]):
                v = per_iou[i].item()
                if torch.isfinite(per_iou[i]).item():
                    log_payload[f"val_iou/{i:02d}_{name}"] = v
            wandb_run.log(log_payload, step=epoch)

        if va_miou > best_miou:
            best_miou = va_miou
            best_acc = va_acc
            best_epoch = epoch
            best_per_iou = per_iou.detach().cpu()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "ignore_index": IGNORE_INDEX,
                    "backbone_id": BACKBONE_ID,
                    "fold": fold,
                    "n_folds": n_folds,
                    "per_class_val_iou": per_iou.cpu(),
                },
                ckpt_path,
            )
            print(f"  saved {ckpt_path}")
            if wandb_run is not None:
                wandb_run.summary["best_val_mIoU"] = best_miou
                wandb_run.summary["best_epoch"] = epoch

    # ------- End-of-run W&B summary artifacts: composite line chart + table -------
    if wandb_run is not None and epoch_history:
        import wandb as _wandb

        # Composite chart: all classes on one panel, one line per class.
        # NaN values are replaced with None so missing classes are dropped from the line.
        ys_per_class = [
            [None if (isinstance(v, float) and (v != v)) else v for v in series]
            for series in per_class_history
        ]
        try:
            line_chart = _wandb.plot.line_series(
                xs=epoch_history,
                ys=ys_per_class,
                keys=list(CLASS_NAMES[:NUM_CLASSES]),
                title="Per-class validation IoU across epochs",
                xname="epoch",
            )
            wandb_run.log({"val_iou/per_class_lines": line_chart})
        except Exception as exc:  # noqa: BLE001 — best-effort, don't kill the run.
            print(f"[wandb] line_series chart skipped: {exc}")

        # Sortable table: rows = epochs, columns = epoch + each class IoU.
        try:
            tbl = _wandb.Table(columns=["epoch"] + list(CLASS_NAMES[:NUM_CLASSES]))
            for row_idx, ep in enumerate(epoch_history):
                row = [ep] + [per_class_history[c][row_idx] for c in range(NUM_CLASSES)]
                tbl.add_data(*row)
            wandb_run.log({"val_iou/per_class_table": tbl})
        except Exception as exc:  # noqa: BLE001
            print(f"[wandb] per-class table skipped: {exc}")

        wandb_run.finish()

    if not args.no_viz:
        render_predictions(model, val_ds, device, crop, args, fold_dir / "prediction_viz")

    per_iou_list = (
        [None if not torch.isfinite(v).item() else float(v) for v in best_per_iou]
        if best_per_iou is not None
        else None
    )
    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "miou": float(best_miou),
        "pixel_acc": float(best_acc),
        "per_class_val_iou": per_iou_list,
    }


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
        img_dir = None
        mask_dir = None
    else:
        data_root = Path(args.data_root)
        img_dir = None
        mask_dir = None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ignore_ids = artifact_ignore_ids(args)
    if ignore_ids:
        names = ", ".join(f"{c}:{CLASS_NAMES[c]}" for c in ignore_ids if c < len(CLASS_NAMES))
        print(f"[config] Ignoring artifact classes (remapped to {IGNORE_INDEX}): {names}")

    merge_map = parse_merge_map(args.merge_class_map)
    if merge_map:
        def _name(c):
            return CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else str(c)
        pairs_str = ", ".join(f"{s}:{_name(s)} -> {d}:{_name(d)}" for s, d in merge_map.items())
        print(f"[config] Merging classes (remapped in masks + presence): {pairs_str}")

    probe = CarbonateSegmentationDataset(
        data_root,
        img_dir=img_dir if use_explicit_dirs else None,
        mask_dir=mask_dir if use_explicit_dirs else None,
        transforms=None,
        normalize=True,
        strict=True,
        ignore_class_ids=ignore_ids,
        merge_class_map=merge_map,
    )
    n = len(probe)

    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if on_gpu else "cpu")

    use_kfold = args.n_folds is not None and args.n_folds >= 2

    # ------------------------------------------------------------------
    # Single train/val split (legacy path: --n_folds 1).
    # ------------------------------------------------------------------
    if not use_kfold:
        n_val = max(1, int(args.val_frac * n))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()
        print(f"[config] Single split | val_frac={args.val_frac} | train={len(train_idx)} val={len(val_idx)}")
        run_single_fold(
            0, 1, train_idx, val_idx, data_root, img_dir, mask_dir, use_explicit_dirs, args, device, on_gpu
        )
        return

    # ------------------------------------------------------------------
    # K-fold cross-validation (default: --n_folds 5).
    # ------------------------------------------------------------------
    # When training on a pre-augmented dataset, keep augmentations TRAINING-ONLY: fold over the
    # original images, validate on clean originals, and route each image's augmentations into
    # training (dropping augs whose source original is in validation). No leakage, and validation
    # is never done on distorted images.
    if not args.group_by_stem and n < args.n_folds:
        raise SystemExit(f"Need at least {args.n_folds} samples for {args.n_folds}-fold CV; found {n}.")

    presence: np.ndarray | None = None
    if args.cv_strategy == "stratified":
        presence = build_class_presence_matrix(
            probe.pairs, NUM_CLASSES, IGNORE_INDEX, ignore_ids, merge_map
        )
        report_class_fold_feasibility(presence, args.n_folds)

    if args.group_by_stem:
        splits = augment_aware_kfold_indices(
            probe.pairs, args.group_pattern, presence, args.n_folds, args.seed,
            max_augs_per_source=args.num_augmentations_per_img,
        )
        n_orig = sum(len(va) for _, va in splits)
        cap = (
            f"; capped to {args.num_augmentations_per_img} augs/original"
            if args.num_augmentations_per_img is not None
            else ""
        )
        print(
            f"[cv] augment-aware split: {n} files -> {n_orig} originals (held out for validation); "
            f"augmentations (pattern {args.group_pattern!r}) are TRAINING-ONLY and never in val{cap}."
        )
    elif args.cv_strategy == "stratified":
        splits = stratified_kfold_indices(presence, args.n_folds, args.seed)
    else:
        splits = kfold_train_val_indices(n, args.n_folds, args.seed)

    # Show how many classes each validation fold actually covers (stratified only).
    if presence is not None:
        present_total = int((presence.sum(axis=0) > 0).sum())
        for f, (_, va) in enumerate(splits):
            covered = int((presence[va].sum(axis=0) > 0).sum())
            print(f"[cv] fold {f}: val n={len(va)} covers {covered}/{present_total} present classes")

    if args.fold is not None:
        if args.fold < 0 or args.fold >= args.n_folds:
            raise SystemExit(f"--fold must be in [0, {args.n_folds}), got {args.fold}")
        folds_to_run = [args.fold]
    else:
        folds_to_run = list(range(args.n_folds))

    grp = " | augment-aware (val = clean originals only)" if args.group_by_stem else ""
    print(
        f"[config] Multiclass UPerNet+SwinV2 | {args.n_folds}-fold cross-validation | "
        f"strategy={args.cv_strategy} | seed={args.seed} | folds_to_run={folds_to_run} | n={n}{grp}"
    )

    all_best: list[dict] = []
    for fold in folds_to_run:
        tr, va = splits[fold]
        result = run_single_fold(
            fold,
            args.n_folds,
            tr.tolist(),
            va.tolist(),
            data_root,
            img_dir,
            mask_dir,
            use_explicit_dirs,
            args,
            device,
            on_gpu,
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
        "per_fold_best": all_best,
        "mean_best_miou": float(np.mean(miou_vals)) if miou_vals else None,
        "std_best_miou": float(np.std(miou_vals)) if miou_vals else None,
        "mean_best_pixel_acc": float(np.mean(acc_vals)) if acc_vals else None,
        "std_best_pixel_acc": float(np.std(acc_vals)) if acc_vals else None,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Cross-validation summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
