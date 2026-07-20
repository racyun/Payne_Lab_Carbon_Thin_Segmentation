# -*- coding: utf-8 -*-
"""
Which augmentations help which classes — controlled for the per-class NOISE FLOOR.

The overall ablation mIoU is within run-to-run noise (~0.02 std). Per-class IoU is noisier
still, so a raw "aug X gave class Y +delta" table is mostly noise. This script:
  1. Computes each run's per-class IoU = mean over the 3 folds (ignoring null/absent classes),
     from that run's cv_summary.json.
  2. Uses the 5 identical noise reps to estimate each class's noise sigma (std across reps).
  3. For each augmentation, reports per-class delta vs the no-aug baseline, and flags a class as
     genuinely "helped" only if delta > NSIGMA * that class's noise sigma (default 2).

Run on the Studio:
    python code/model_training_pipeline/analyze_per_class_aug.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

ROOT = "/teamspace/studios/this_studio/petro_data/model_outputs"
NSIGMA = 2.0  # a delta must exceed this many per-class noise-sigmas to count as real

CLASS_NAMES = (
    "background", "bivalves", "micrite", "cement", "echinoderms", "foraminifera",
    "calcareous algae", "peloid", "unid biota", "ooid", "gastropods", "scale bar",
    "mollusk", "ostracod", "aggregate grain", "brachiopod", "sponge", "watermark",
)


def per_class_mean(cv_path: str) -> dict[int, float]:
    """Per-class IoU averaged across folds (skipping null/absent classes)."""
    with open(cv_path) as f:
        d = json.load(f)
    acc: dict[int, list[float]] = defaultdict(list)
    for fold in d["per_fold_best"]:
        for i, v in enumerate(fold.get("per_class_val_iou", [])):
            if v is not None:
                acc[i].append(float(v))
    return {i: sum(vs) / len(vs) for i, vs in acc.items() if vs}


def main() -> None:
    base = per_class_mean(f"{ROOT}/ablation/abl_baseline/cv_summary.json")

    # per-class noise sigma from the 5 identical reps
    rep_paths = sorted(glob.glob(f"{ROOT}/noise_baseline/*/cv_summary.json"))
    reps = [per_class_mean(p) for p in rep_paths]
    noise_sigma: dict[int, float] = {}
    for i in range(len(CLASS_NAMES)):
        vals = [r[i] for r in reps if i in r]
        if len(vals) >= 2:
            m = sum(vals) / len(vals)
            noise_sigma[i] = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
    print(f"[noise] per-class sigma from {len(reps)} reps at {ROOT}/noise_baseline/")
    for i in sorted(noise_sigma):
        print(f"    {CLASS_NAMES[i]:<18} baseline {base.get(i, float('nan')):.3f}  sigma {noise_sigma[i]:.3f}")

    # each augmentation run
    augs: dict[str, dict[int, float]] = {}
    for p in sorted(glob.glob(f"{ROOT}/ablation/abl_*/cv_summary.json")):
        name = os.path.basename(os.path.dirname(p)).replace("abl_", "")
        if name == "baseline":
            continue
        augs[name] = per_class_mean(p)

    # significant helps: delta > NSIGMA * sigma
    by_class: dict[int, list[tuple[float, str]]] = defaultdict(list)
    by_aug: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for name, pc in augs.items():
        for i, iou in pc.items():
            if i not in base or i not in noise_sigma or noise_sigma[i] == 0:
                continue
            delta = iou - base[i]
            if delta > NSIGMA * noise_sigma[i]:
                by_class[i].append((delta, name))
                by_aug[name].append((delta, i))

    print(f"\n=== Significant per-class HELPS (delta > {NSIGMA}x per-class noise sigma) ===")
    if not by_class:
        print("  NONE. No augmentation moved any class beyond its noise floor "
              "-> per-class 'effects' are noise, consistent with the mean-mIoU result.")
    else:
        for i in sorted(by_class, key=lambda k: -max(d for d, _ in by_class[k])):
            hits = sorted(by_class[i], reverse=True)
            print(f"\n  {CLASS_NAMES[i]}  (baseline {base[i]:.3f}, sigma {noise_sigma[i]:.3f}):")
            for delta, name in hits:
                print(f"      +{delta:.3f}   {name}")

    print("\n=== By augmentation: classes it significantly helps ===")
    for name in sorted(by_aug, key=lambda k: -len(by_aug[k])):
        cls = ", ".join(f"{CLASS_NAMES[i]} (+{d:.3f})" for d, i in sorted(by_aug[name], reverse=True))
        print(f"  {name:<26} {cls}")


if __name__ == "__main__":
    main()
