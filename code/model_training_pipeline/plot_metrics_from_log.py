# -*- coding: utf-8 -*-
"""
Rebuild loss / accuracy / mIoU / per-class-IoU charts for a BINARY run from its
training log (or from the per-fold val_metrics.csv files) — no Weights & Biases needed.

The binary runner prints one block per epoch, e.g.::

    Fold 0 | Epoch 01 | train 0.0967 | val 0.0759 | acc 0.375 | mIoU 0.229
      per-class val IoU (epoch 01):
        background IoU: 0.1632
        grain IoU: 0.2939
      val pixel split (epoch 01):
        prediction   -> background: 71.54% | grain: 28.46%

This scrapes those lines into a tidy CSV and a 2x2 chart, one line per fold.

Usage::

    # from a run log (recommended — separates CE / focal / focal_cw runs cleanly)
    python plot_metrics_from_log.py --log .../seg_with_aug_binary_3fold/binary_run_focal_cw.log

    # or from the per-fold CSVs in an output dir (last writer wins per fold)
    python plot_metrics_from_log.py --csv_dir .../seg_with_aug_binary_3fold

Writes <stem>_metrics.csv and <stem>_charts.png next to the input (or --out_prefix).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EPOCH_RE = re.compile(
    r"Fold\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*train\s+([\d.]+)\s*\|\s*val\s+([\d.]+)"
    r"\s*\|\s*acc\s+([\d.]+)\s*\|\s*mIoU\s+([\d.]+)"
)
BG_IOU_RE = re.compile(r"background IoU:\s*([\d.]+|nan)")
GRAIN_IOU_RE = re.compile(r"grain IoU:\s*([\d.]+|nan)")
PRED_RE = re.compile(r"prediction\s*->\s*background:\s*([\d.]+)%\s*\|\s*grain:\s*([\d.]+)%")

FIELDS = [
    "fold", "epoch", "train_loss", "val_loss", "pixel_acc", "miou",
    "iou_bg", "iou_grain", "pred_bg_pct", "pred_grain_pct",
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def parse_log(path: Path) -> list[dict]:
    """Scrape per-epoch records from a run log."""
    rows: list[dict] = []
    cur: dict | None = None
    for line in path.read_text(errors="ignore").splitlines():
        m = EPOCH_RE.search(line)
        if m:
            if cur is not None:
                rows.append(cur)
            cur = {
                "fold": int(m.group(1)), "epoch": int(m.group(2)),
                "train_loss": _f(m.group(3)), "val_loss": _f(m.group(4)),
                "pixel_acc": _f(m.group(5)), "miou": _f(m.group(6)),
                "iou_bg": float("nan"), "iou_grain": float("nan"),
                "pred_bg_pct": float("nan"), "pred_grain_pct": float("nan"),
            }
            continue
        if cur is None:
            continue
        m = BG_IOU_RE.search(line)
        if m:
            cur["iou_bg"] = _f(m.group(1))
            continue
        m = GRAIN_IOU_RE.search(line)
        if m:
            cur["iou_grain"] = _f(m.group(1))
            continue
        m = PRED_RE.search(line)
        if m:
            cur["pred_bg_pct"] = _f(m.group(1))
            cur["pred_grain_pct"] = _f(m.group(2))
    if cur is not None:
        rows.append(cur)
    return rows


def parse_csv_dir(csv_dir: Path) -> list[dict]:
    """Read every fold_*/val_metrics.csv under csv_dir."""
    rows: list[dict] = []
    for csv_path in sorted(csv_dir.glob("fold_*/val_metrics.csv")):
        fold = int(csv_path.parent.name.split("_")[-1])
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                rows.append({
                    "fold": fold, "epoch": int(r["epoch"]),
                    "train_loss": _f(r.get("train_loss")), "val_loss": _f(r.get("val_loss")),
                    "pixel_acc": _f(r.get("pixel_acc")), "miou": _f(r.get("miou")),
                    "iou_bg": _f(r.get("iou_bg")), "iou_grain": _f(r.get("iou_grain")),
                    "pred_bg_pct": _f(r.get("pred_bg_pct")), "pred_grain_pct": _f(r.get("pred_grain_pct")),
                })
    return rows


def write_csv(rows: list[dict], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot(rows: list[dict], out_png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = sorted({r["fold"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=13)

    def series(fold, key):
        pts = sorted((r["epoch"], r[key]) for r in rows if r["fold"] == fold)
        return [e for e, _ in pts], [v for _, v in pts]

    panels = [
        (axes[0, 0], "Loss", [("train_loss", "train"), ("val_loss", "val")]),
        (axes[0, 1], "Pixel accuracy", [("pixel_acc", "acc")]),
        (axes[1, 0], "mIoU", [("miou", "mIoU")]),
        (axes[1, 1], "Per-class val IoU", [("iou_bg", "background"), ("iou_grain", "grain")]),
    ]
    for ax, ttl, keys in panels:
        for fold in folds:
            for key, lbl in keys:
                xs, ys = series(fold, key)
                ax.plot(xs, ys, marker=".", ms=3, label=f"fold{fold} {lbl}")
        ax.set_title(ttl)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axes[1, 1].set_ylim(-0.02, 1.0)
    axes[0, 1].set_ylim(0, 1.0)
    axes[1, 0].set_ylim(0, 1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print("wrote", out_png)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", type=str, help="Path to a run .log file.")
    src.add_argument("--csv_dir", type=str, help="Output dir containing fold_*/val_metrics.csv.")
    p.add_argument("--out_prefix", type=str, default=None, help="Output path prefix (default: next to input).")
    args = p.parse_args()

    if args.log:
        in_path = Path(args.log)
        rows = parse_log(in_path)
        default_prefix = in_path.with_suffix("")
    else:
        in_path = Path(args.csv_dir)
        rows = parse_csv_dir(in_path)
        default_prefix = in_path / "combined"

    if not rows:
        raise SystemExit(f"No epoch records found in {in_path}. Is it the right file/dir?")

    prefix = Path(args.out_prefix) if args.out_prefix else default_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, prefix.with_name(prefix.name + "_metrics.csv"))
    print("wrote", prefix.with_name(prefix.name + "_metrics.csv"), f"({len(rows)} epoch-rows)")
    try:
        plot(rows, prefix.with_name(prefix.name + "_charts.png"), title=prefix.name)
    except ImportError:
        print("matplotlib not installed; wrote the CSV only. `pip install matplotlib` to also get charts.")


if __name__ == "__main__":
    main()
