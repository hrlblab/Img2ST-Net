#!/usr/bin/env python3

import csv
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

PATCH_SIZE = 448  # patch size
Image.MAX_IMAGE_PIXELS = None


def is_patch_dark(patch, threshold=25):
    """Check if a patch is too dark (too many black pixels)."""
    patch_gray = patch.convert("L")
    arr = np.array(patch_gray)

    black_pixels = np.sum(arr < 10)
    total_pixels = arr.size
    black_ratio = black_pixels / total_pixels

    return black_ratio > (threshold / 100.0)


def cut_one_image(img_path: Path, out_root: Path, csv_dir: Path):
    with Image.open(img_path) as im:
        w, h = im.size

        # Patch output directory: <out_root>/<image_name>/
        patch_dir = out_root / img_path.stem
        patch_dir.mkdir(parents=True, exist_ok=True)

        # CSV output path: <csv_dir>/<image_name>.csv
        csv_path = csv_dir / f"{img_path.stem}.csv"
        with open(csv_path, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["patch_name", "center_x", "center_y", "i", "j"])

            skipped = 0
            total = 0

            # Sliding window patching
            for i, top in tqdm(enumerate(range(0, h - PATCH_SIZE + 1, PATCH_SIZE)), desc=f"{img_path.name}"):
                for j, left in enumerate(range(0, w - PATCH_SIZE + 1, PATCH_SIZE)):
                    box = (left, top, left + PATCH_SIZE, top + PATCH_SIZE)
                    patch = im.crop(box)

                    total += 1

                    # Skip if the black pixel ratio is too high
                    if is_patch_dark(patch, threshold=25):
                        skipped += 1
                        continue

                    patch_name = f"{img_path.stem}_{i}_{j}.png"
                    patch.save(patch_dir / patch_name)

                    cx = left + PATCH_SIZE // 2
                    cy = top + PATCH_SIZE // 2
                    writer.writerow([patch_name, cx, cy, i, j])

        print(f"✔ {img_path.name} finished: saved {total - skipped} patches, skipped {skipped} too-dark patches ➜ CSV: {csv_path.relative_to(out_root)}")


def main(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = dst_dir / "csv_index"
    csv_dir.mkdir(exist_ok=True)

    for png_path in tqdm(sorted(src_dir.glob("*.png")), desc="Overall progress"):
        if str(png_path).split('/')[-1] == 'D.png':
            cut_one_image(png_path, dst_dir, csv_dir)
        else:
            print(png_path)
            continue

    print("\n✅ All images processed!")


if __name__ == "__main__":
    datasets = ['BC']
    for data in datasets:
        src_dirs = Path(f'./datasets/{data}/WSI')
        dst_dirs = Path(f'./datasets/{data}/cropped_img')
        main(src_dirs, dst_dirs)
