#!/usr/bin/env python3
"""
Consolidate all SSL pretraining images into a single (flat) Google Drive folder.

Recursively gathers every image under these subfolders of the dataset root:
    cretaceous thin sections
    Permian-Triassic
    TJ photomicrographs
and copies them into ONE flat folder, then (by default) uploads that folder to
Google Drive via rclone.

Why flatten by source path:
    The subfolders nest deeply (e.g. "TJ photomicrographs/ICA 40.3/img1.jpg") and
    different subfolders can reuse the same filename, so a naive flatten would
    overwrite files. Each image is therefore renamed to its source-relative path
    with separators replaced by "__", e.g.
        TJ photomicrographs/ICA 40.3/img1.jpg  ->  TJ photomicrographs__ICA 40.3__img1.jpg
    This guarantees unique names and keeps each image's provenance readable.
    (Filenames are irrelevant to SSL pretraining, which just reads all images.)

Assumptions (documented per request — override with the flags below):
    - Source images are already on the Studio's local disk at
      /teamspace/studios/this_studio/petro_data/unlabeled/<subfolder>/...
      (this is the copy synced for SSL). Use --sync_first to pull them fresh from
      Drive instead, or point --src_root somewhere else.
    - An rclone remote named "gdrive" is configured (run with RCLONE_CONFIG set if
      your config lives outside the default path, e.g.
      `export RCLONE_CONFIG=~/rclone-config/rclone.conf`).

Examples:
    # Flatten the already-synced local copy and upload to Drive (default behaviour):
    python collect_pretrain_images.py

    # Pull the 3 folders fresh from Drive first, then flatten + upload:
    python collect_pretrain_images.py --sync_first

    # Just build the local flat folder, don't touch Drive:
    python collect_pretrain_images.py --no_upload
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

SUBFOLDERS = ["cretaceous thin sections", "Permian-Triassic", "TJ photomicrographs"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Drive root (relative to the gdrive: rclone remote, i.e. "My Drive").
DRIVE_ROOT = "Petrographic images_ML work"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--src_root",
        default="/teamspace/studios/this_studio/petro_data/unlabeled",
        help="Local dir containing the 3 unlabeled subfolders (default: the SSL-synced copy).",
    )
    p.add_argument(
        "--out_dir",
        default="/teamspace/studios/this_studio/petro_data/all_pretrain_images",
        help="Local flat folder to collect images into.",
    )
    p.add_argument(
        "--rclone_dest",
        default=f"gdrive:{DRIVE_ROOT}/all_pretrain_images",
        help="rclone destination for the flat folder on Drive.",
    )
    p.add_argument("--subfolders", nargs="+", default=SUBFOLDERS, help="Source subfolders to scan.")
    p.add_argument("--sep", default="__", help="Separator used when flattening source paths into filenames.")
    p.add_argument(
        "--sync_first",
        action="store_true",
        help="rclone-pull the 3 subfolders from Drive into --src_root before flattening.",
    )
    p.add_argument("--no_upload", action="store_true", help="Build the local flat folder only; skip the Drive upload.")
    return p.parse_args()


def _rclone(*args: str) -> None:
    cmd = ["rclone", *args, "--progress", "--transfers=8", "--checkers=16"]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    out_dir = Path(args.out_dir)

    if args.sync_first:
        print("Syncing source folders from Drive -> local ...")
        for sub in args.subfolders:
            _rclone("copy", f"gdrive:{DRIVE_ROOT}/{sub}", str(src_root / sub))

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for sub in args.subfolders:
        base = src_root / sub
        if not base.is_dir():
            print(f"[warn] source folder missing, skipping: {base}")
            continue
        n = 0
        for p in sorted(base.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
                continue
            rel = p.relative_to(src_root)            # e.g. "TJ photomicrographs/ICA 40.3/img1.jpg"
            name = args.sep.join(rel.parts)          # -> "TJ photomicrographs__ICA 40.3__img1.jpg"
            dest = out_dir / name
            # Extremely unlikely with full-path flattening, but stay safe on collisions.
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                k = 1
                while (out_dir / f"{stem}_{k}{suf}").exists():
                    k += 1
                dest = out_dir / f"{stem}_{k}{suf}"
            # Hardlink when possible (same filesystem) to avoid duplicating ~tens of GB;
            # fall back to a real copy across filesystems.
            try:
                os.link(p, dest)
            except OSError:
                shutil.copy2(p, dest)
            n += 1
            total += 1
        print(f"  {sub}: {n} images")

    print(f"\nCollected {total} images -> {out_dir}")
    if total == 0:
        print("[error] No images found. Check --src_root or run with --sync_first.")
        return

    if args.no_upload:
        print("--no_upload set; leaving the flat folder local only.")
        print(f"To upload later: rclone copy '{out_dir}' '{args.rclone_dest}' --progress --transfers=8 --checkers=16")
        return

    print(f"\nUploading flat folder to {args.rclone_dest} ...")
    _rclone("copy", str(out_dir), args.rclone_dest)
    print(f"\nDone. {total} images are now in Drive at: {args.rclone_dest}")


if __name__ == "__main__":
    main()
