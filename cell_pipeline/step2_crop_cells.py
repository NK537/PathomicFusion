# cell_pipeline/step2_crop_cells.py
#
# STEP 2 — Crop individual cell patches from section images.
#
# Reads JSON files from Step 1 (cell centroids), opens the corresponding
# section PNG, and crops a (cell_crop_size x cell_crop_size) region around
# each cell centroid. Saves one PNG per cell.
#
# Output naming:
#   cell_patches/{section_stem}/{section_stem}_{cell_id}.png
#
# Usage:
#   python3 cell_pipeline/step2_crop_cells.py

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell_pipeline.config_cell import CELL_CONFIG


def crop_cells_from_section(
    img_path:      str,
    mask_json:     dict,
    out_dir:       str,
    section_stem:  str,
    crop_size:     int = 64,
    skip_existing: bool = True,
) -> int:
    """
    Crops all cells from one section image.
    Returns number of cells saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    nuc   = mask_json.get("nuc", {})
    img   = None   # lazy load — only open if needed
    saved = 0

    for cell_id, cell_info in nuc.items():
        cx, cy  = cell_info["centroid"]          # (x, y) pixel coords
        cx, cy  = int(round(cx)), int(round(cy))
        out_path = os.path.join(out_dir, f"{section_stem}_{cell_id}.png")

        if skip_existing and os.path.exists(out_path):
            saved += 1
            continue

        # Lazy-load image on first cell that needs processing
        if img is None:
            img = np.array(Image.open(img_path).convert("RGB"))
            H, W = img.shape[:2]

        half = crop_size // 2
        # Clamp to image boundaries
        x1 = max(0, cx - half);  x2 = min(W, cx + half)
        y1 = max(0, cy - half);  y2 = min(H, cy + half)
        crop = img[y1:y2, x1:x2]

        # Pad to exact crop_size if near image border
        pad_h = crop_size - crop.shape[0]
        pad_w = crop_size - crop.shape[1]
        if pad_h > 0 or pad_w > 0:
            crop = np.pad(crop,
                          ((0, pad_h), (0, pad_w), (0, 0)),
                          mode="reflect")

        Image.fromarray(crop).save(out_path)
        saved += 1

    return saved


def crop_all_cells(
    section_dir:   str,
    cell_mask_dir: str,
    cell_patch_dir: str,
    crop_size:     int  = 64,
    skip_existing: bool = True,
):
    json_files = sorted(
        f for f in os.listdir(cell_mask_dir)
        if f.endswith(".json")
    )
    print(f"Found {len(json_files)} JSON mask files")

    total_cells = 0
    missing_img  = 0

    for jf in tqdm(json_files, desc="Cropping cells"):
        stem     = jf[:-5]                                   # remove .json
        img_path = os.path.join(section_dir, stem + ".png")
        out_dir  = os.path.join(cell_patch_dir, stem)

        if not os.path.exists(img_path):
            print(f"  [WARN] Section image not found: {img_path}")
            missing_img += 1
            continue

        with open(os.path.join(cell_mask_dir, jf)) as f:
            mask_json = json.load(f)

        n = crop_cells_from_section(
            img_path      = img_path,
            mask_json     = mask_json,
            out_dir       = out_dir,
            section_stem  = stem,
            crop_size     = crop_size,
            skip_existing = skip_existing,
        )
        total_cells += n

    print(f"\n=== DONE ===  total_cells={total_cells}  missing_sections={missing_img}")
    print(f"Cell patches saved to: {cell_patch_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    crop_all_cells(
        section_dir    = CELL_CONFIG["section_dir"],
        cell_mask_dir  = CELL_CONFIG["cell_mask_dir"],
        cell_patch_dir = CELL_CONFIG["cell_patch_dir"],
        crop_size      = CELL_CONFIG["cell_crop_size"],
        skip_existing  = not args.no_skip,
    )
