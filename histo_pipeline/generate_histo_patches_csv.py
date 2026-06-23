# histo_pipeline/generate_histo_patches_csv.py
#
# Scans the patch image directory and generates histo_patches.csv.
# Only needed if you want a manifest of all patches -- training scans
# the embedding dir directly and does NOT require this CSV.
#
# Usage:
#   python histo_pipeline/generate_histo_patches_csv.py
#
# Output: data/Picasso/histo/histo_patches.csv
#   columns: patient_id, section, patch_filename

import os, sys, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from histo_pipeline.config_histo import HISTO_CONFIG
from histo_pipeline.patient_histo_dataset import filename_to_pat_id


def main(cfg):
    patch_dir = cfg["histo_patch_dir"]
    out_csv   = os.path.join(os.path.dirname(cfg["fusion_label_xlsx"]), "histo_patches.csv")

    if not os.path.isdir(patch_dir):
        print(f"ERROR: patch_dir not found: {patch_dir}")
        print("Set histo_patch_dir in config_histo.py to the correct path.")
        return

    rows = []
    skipped = 0
    for fname in sorted(os.listdir(patch_dir)):
        if not fname.lower().endswith(".png"):
            continue
        pat_id = filename_to_pat_id(fname)
        if pat_id is None:
            skipped += 1
            continue
        # Extract section from filename: "02_003 Sigmoid 1E_16_146.png"
        parts   = fname.split()
        section = parts[1] if len(parts) > 1 else "Unknown"
        rows.append({
            "patient_id":     pat_id,
            "section":        section,
            "patch_filename": fname,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} patches ({skipped} skipped) to {out_csv}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(df.groupby("patient_id").size().describe().to_string())


if __name__ == "__main__":
    main(HISTO_CONFIG)
