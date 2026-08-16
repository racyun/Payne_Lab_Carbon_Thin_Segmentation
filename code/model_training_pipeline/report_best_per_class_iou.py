#!/usr/bin/env python3
"""Print best per-class validation IoU for the most recent training runs.

Reads the per-epoch metric CSVs that both pipelines already write, so it needs no
checkpoints and no GPU:

  swin_training_pipeline_221.py  ->  <run>/fold_<i>/val_per_class_iou.csv
                                     (epoch, class_0 ... class_N)
  baseline_seg_pipeline_221.py   ->  <run>/fold_<i>/val_metrics.csv
                                     (epoch, train_loss, val_loss, pixel_acc, miou, iou_0 ... iou_N)

Two different things can be called "the best per-class IoU", so both are printed:

  BEST-EVER   max over epochs of each class's IoU, taken independently per class.
              This is what "best IoU reached per class" literally means; the maxima
              generally land on different epochs, so the column is not a snapshot of
              any single model.
  CHECKPOINT  per-class IoU at the epoch with the highest mIoU, i.e. the weights that
              were actually saved to best_*.pth. This is the row to quote in the paper.

Per-fold values are combined with a nanmean across folds (classes absent from a fold
are nan and drop out), matching how the pipelines average mIoU across CV folds.

Usage on the Studio::

    python code/model_training_pipeline/report_best_per_class_iou.py \
        --root /teamspace/studios/this_studio/petro_data/model_outputs --last 6

    # or name the runs explicitly, in the order you want the columns
    python code/model_training_pipeline/report_best_per_class_iou.py \
        --runs .../seg_with_aug_3fold .../seg_no_aug_3fold ...
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# Mirrors CLASS_NAMES in swin_training_pipeline_221.py (labels 0-17).
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


def nanmean(values: list[float]) -> float:
    finite = [v for v in values if v == v]
    return sum(finite) / len(finite) if finite else float("nan")


def fmt(v: float, width: int = 7) -> str:
    return f"{'--':>{width}}" if v != v else f"{v:{width}.4f}"


# ---------------------------------------------------------------------------
# Reading one fold
# ---------------------------------------------------------------------------

def read_fold_csv(path: Path) -> tuple[list[int], list[list[float]], list[float]]:
    """Return (epochs, per_epoch_class_ious, per_epoch_miou) from either CSV format.

    mIoU is recomputed as the nanmean over classes rather than read from the file, so
    both formats agree: that is exactly miou_from_confusion()'s definition (mean over
    classes whose union > 0; absent classes are nan and excluded).
    """
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return [], [], []

    header = rows[0]
    iou_cols = [i for i, h in enumerate(header) if h.startswith(("class_", "iou_"))]
    epoch_col = header.index("epoch") if "epoch" in header else 0

    epochs, per_epoch, mious = [], [], []
    for row in rows[1:]:
        if len(row) <= max(iou_cols, default=0):
            continue  # truncated final line from a run that was killed mid-write
        ious = []
        for i in iou_cols:
            try:
                ious.append(float(row[i]))
            except ValueError:
                ious.append(float("nan"))
        try:
            epochs.append(int(float(row[epoch_col])))
        except ValueError:
            epochs.append(len(epochs) + 1)
        per_epoch.append(ious)
        mious.append(nanmean(ious))
    return epochs, per_epoch, mious


def summarize_fold(path: Path) -> dict | None:
    """Best-ever per-class IoU and the best-mIoU-epoch snapshot for one fold."""
    epochs, per_epoch, mious = read_fold_csv(path)
    if not per_epoch:
        return None
    n_cls = max(len(r) for r in per_epoch)

    best_ever = [float("nan")] * n_cls
    best_ever_epoch = [-1] * n_cls
    for e, ious in zip(epochs, per_epoch):
        for c, v in enumerate(ious):
            if v == v and (best_ever[c] != best_ever[c] or v > best_ever[c]):
                best_ever[c], best_ever_epoch[c] = v, e

    finite_miou = [(m, i) for i, m in enumerate(mious) if m == m]
    bi = max(finite_miou)[1] if finite_miou else 0
    return {
        "n_classes": n_cls,
        "best_ever": best_ever,
        "best_ever_epoch": best_ever_epoch,
        "ckpt_epoch": epochs[bi],
        "ckpt_miou": mious[bi],
        "ckpt_per_class": per_epoch[bi] + [float("nan")] * (n_cls - len(per_epoch[bi])),
        "n_epochs": len(epochs),
    }


# ---------------------------------------------------------------------------
# Reading one run (all folds)
# ---------------------------------------------------------------------------

def fold_csvs(run_dir: Path) -> list[Path]:
    out = []
    for fold_dir in sorted(run_dir.glob("fold_*")):
        for name in ("val_per_class_iou.csv", "val_metrics.csv"):
            p = fold_dir / name
            if p.is_file():
                out.append(p)
                break
    return out


def summarize_run(run_dir: Path) -> dict | None:
    folds = [s for s in (summarize_fold(p) for p in fold_csvs(run_dir)) if s]
    if not folds:
        return None
    n_cls = max(f["n_classes"] for f in folds)

    def col(key: str) -> list[float]:
        return [
            nanmean([f[key][c] for f in folds if c < len(f[key])])
            for c in range(n_cls)
        ]

    return {
        "name": run_dir.name,
        "path": run_dir,
        "n_folds": len(folds),
        "n_epochs": max(f["n_epochs"] for f in folds),
        "n_classes": n_cls,
        "best_ever": col("best_ever"),
        "ckpt_per_class": col("ckpt_per_class"),
        "ckpt_miou": nanmean([f["ckpt_miou"] for f in folds]),
        "best_ever_miou": nanmean([nanmean(f["best_ever"]) for f in folds]),
        "ckpt_epochs": [f["ckpt_epoch"] for f in folds],
        "folds": folds,
    }


def discover_runs(root: Path, last: int) -> list[Path]:
    """Most recently modified run directories (a run = a dir containing fold_*/)."""
    runs = {p.parent for p in root.rglob("fold_*/val_per_class_iou.csv")}
    runs |= {p.parent for p in root.rglob("fold_*/val_metrics.csv")}
    runs = {p.parent for p in runs}  # fold_* -> run dir

    def mtime(d: Path) -> float:
        return max(
            (c.stat().st_mtime for c in d.rglob("*.csv")),
            default=d.stat().st_mtime,
        )

    return sorted(runs, key=mtime, reverse=True)[:last]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(title: str, note: str, runs: list[dict], key: str, label_w: int = 22) -> None:
    col_w = 12
    print()
    print(title)
    print(note)
    heads = [r["name"][: col_w - 1] for r in runs]
    print(f"{'class':<{label_w}}" + "".join(f"{h:>{col_w}}" for h in heads))
    print("-" * (label_w + col_w * len(runs)))
    n_cls = max(r["n_classes"] for r in runs)
    for c in range(n_cls):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
        vals = [r[key][c] if c < len(r[key]) else float("nan") for r in runs]
        if all(v != v for v in vals):
            continue  # class absent everywhere (merged away or ignored)
        print(f"{c:02d} {name:<{label_w - 3}}" + "".join(fmt(v, col_w) for v in vals))
    print("-" * (label_w + col_w * len(runs)))
    miou_key = "best_ever_miou" if key == "best_ever" else "ckpt_miou"
    print(f"{'mIoU':<{label_w}}" + "".join(fmt(r[miou_key], col_w) for r in runs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--root",
        default="/teamspace/studios/this_studio/petro_data/model_outputs",
        help="Directory to search for run folders (default: the Studio model_outputs root).",
    )
    ap.add_argument("--last", type=int, default=6, help="How many most-recent runs to report (default: 6).")
    ap.add_argument("--runs", nargs="*", default=None, help="Explicit run directories; overrides --root/--last.")
    ap.add_argument(
        "--table",
        choices=("best", "ckpt", "both"),
        default="best",
        help="best = max over epochs per class (default); ckpt = best-mIoU epoch; both = print each.",
    )
    ap.add_argument("--csv", default=None, help="Also write a tidy CSV here.")
    args = ap.parse_args()

    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
    else:
        root = Path(args.root)
        if not root.is_dir():
            raise SystemExit(f"--root not found: {root}")
        run_dirs = discover_runs(root, args.last)
        if not run_dirs:
            raise SystemExit(f"No runs with fold_*/ metric CSVs under {root}")

    runs = []
    for d in run_dirs:
        s = summarize_run(d)
        if s is None:
            print(f"[skip] no fold metric CSVs under {d}")
            continue
        runs.append(s)
    if not runs:
        raise SystemExit("No readable runs.")

    print("=== runs (most recent first) ===")
    for r in runs:
        print(
            f"  {r['name']:<28} folds={r['n_folds']}  epochs={r['n_epochs']}  "
            f"best epoch/fold={r['ckpt_epochs']}  {r['path']}"
        )

    if args.table in ("best", "both"):
        print_table(
            "=== BEST-EVER per-class val IoU (max over epochs, nanmean across folds) ===",
            "    Each class's own best epoch; not a single model snapshot, so the mIoU row\n"
            "    is an upper bound, not the mIoU of any saved checkpoint (see --table ckpt).",
            runs,
            "best_ever",
        )
    if args.table in ("ckpt", "both"):
        print_table(
            "=== per-class val IoU AT THE BEST-mIoU EPOCH (the saved checkpoint) ===",
            "    One consistent model per fold; matches the mIoU in Table 1.",
            runs,
            "ckpt_per_class",
        )
    print("\n'--' = class absent from every fold's validation set (merged away or ignored).")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["run", "class_id", "class_name", "best_ever_iou", "ckpt_iou"])
            for r in runs:
                for c in range(r["n_classes"]):
                    nm = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
                    w.writerow([r["name"], c, nm, r["best_ever"][c], r["ckpt_per_class"][c]])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
