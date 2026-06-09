# histo_pipeline/main_histo.py
#
# 5-fold cross-validation for the Picasso histology survival pipeline.
# Mirrors main_picasso.py (endoscopy) exactly.
#
# Usage:
#   cd /home/usama/Projects/PathomicFusion
#   python -m histo_pipeline.main_histo
#   -- OR --
#   python histo_pipeline/main_histo.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from histo_pipeline.config_histo import HISTO_CONFIG
from histo_pipeline.train_histo import train_histo


def get_patient_ids(cfg):
    """
    Read all valid patient codes from the shared label Excel file.
    Returns (codes, events) for stratified splitting.
    """
    raw  = pd.read_excel(cfg["label_xlsx"])
    cols = list(raw.columns)
    try:
        doi = cols.index("date_of_outcome"); cols[doi + 1] = "days_to_outcome"
    except (ValueError, IndexError):
        for i, c in enumerate(cols):
            if str(c).startswith("Unnamed:"): cols[i] = "days_to_outcome"; break
    raw.columns = cols

    raw["code"]        = raw["code"].astype(str).str.zfill(4)
    raw["ANY OUTCOME"] = pd.to_numeric(raw["ANY OUTCOME"], errors="coerce")
    raw = raw[raw["ANY OUTCOME"].notna()].copy()

    # Further restrict to patients that actually have histo patch embeddings
    patch_df     = pd.read_csv(cfg["histo_patches_csv"])
    patch_df["patient_id"] = patch_df["patient_id"].astype(str).str.zfill(4)
    histo_ids    = set(patch_df["patient_id"].unique())
    raw          = raw[raw["code"].isin(histo_ids)].copy()

    codes  = raw["code"].tolist()
    events = raw["ANY OUTCOME"].astype(int).tolist()

    print(f"Total histo patients : {len(codes)}")
    print(f"Events               : {sum(events)}")
    print(f"Censored             : {len(events) - sum(events)}")
    return codes, events


def print_results(results):
    print("\n" + "=" * 58)
    print("FINAL 5-Fold CV Results — Picasso Histology")
    print("=" * 58)
    print(f"{'Config':<20} {'Mean C-idx':>12} {'Std':>8}  {'Per-fold'}")
    print("-" * 58)
    for name, scores in results.items():
        arr   = np.array(scores)
        folds = "  ".join(f"{s:.4f}" for s in scores)
        print(f"{name:<20} {arr.mean():>12.4f} {arr.std():>8.4f}  {folds}")
    print("=" * 58)

    rows = [
        {"config": name, "fold": i, "c_index": s}
        for name, scores in results.items()
        for i, s in enumerate(scores)
    ]
    out = "histo_cv_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    codes, events = get_patient_ids(HISTO_CONFIG)

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds   = list(skf.split(codes, events))
    results = {"histo_only": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ids = [codes[i] for i in train_idx]
        val_ids   = [codes[i] for i in val_idx]

        print(f"\n{'='*58}")
        print(f"FOLD {fold_idx+1}/5  train={len(train_ids)}  val={len(val_ids)}")
        print("="*58)

        c_idx = train_histo(
            cfg       = HISTO_CONFIG,
            train_ids = train_ids,
            val_ids   = val_ids,
            fold_idx  = fold_idx,
        )
        results["histo_only"].append(c_idx)

    print_results(results)
