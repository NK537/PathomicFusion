"""
main_picasso.py
---------------
5-fold cross-validation for the Picasso endoscopy survival pipeline.

Patient IDs are read from PicassoOnly_Outcome_train.xlsx (code column).
Splits are performed at patient level using stratified KFold on event status.

Usage:
    python main_picasso.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config_picasso import PICASSO_CONFIG
from train_picasso import train_picasso


def get_patient_ids(cfg):
    """
    Read all valid patient codes from the label Excel file.
    Returns (codes, events) arrays for stratified splitting.
    """
    raw  = pd.read_excel(cfg["label_xlsx"])
    raw["code"] = raw["code"].astype(str).str.zfill(4)

    # Rename the unnamed days column
    cols = list(raw.columns)
    doi  = cols.index("date_of_outcome")
    cols[doi + 1] = "days_to_outcome"
    raw.columns = cols

    raw["ANY OUTCOME"] = pd.to_numeric(raw["ANY OUTCOME"], errors="coerce")

    # Keep only rows with valid outcome
    raw = raw[raw["ANY OUTCOME"].notna()].copy()

    codes  = raw["code"].tolist()
    events = raw["ANY OUTCOME"].astype(int).tolist()

    print(f"Total patients: {len(codes)}  |  Events: {sum(events)}  |  Censored: {len(events)-sum(events)}")
    return codes, events


def print_results(results):
    print("\n" + "="*55)
    print("FINAL 5-Fold CV Results — Picasso Endoscopy")
    print("="*55)
    print(f"{'Config':<20} {'Mean C-idx':>12} {'Std':>8}  {'Per-fold'}")
    print("-"*55)
    for name, scores in results.items():
        arr   = np.array(scores)
        folds = "  ".join(f"{s:.4f}" for s in scores)
        print(f"{name:<20} {arr.mean():>12.4f} {arr.std():>8.4f}  {folds}")
    print("="*55)

    rows = [
        {"config": name, "fold": i, "c_index": s}
        for name, scores in results.items()
        for i, s in enumerate(scores)
    ]
    pd.DataFrame(rows).to_csv("picasso_cv_results.csv", index=False)
    print("Saved to picasso_cv_results.csv")


if __name__ == "__main__":
    codes, events = get_patient_ids(PICASSO_CONFIG)

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds  = list(skf.split(codes, events))
    results = {"endo_only": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ids = [codes[i] for i in train_idx]
        val_ids   = [codes[i] for i in val_idx]

        print(f"\n{'='*55}")
        print(f"FOLD {fold_idx+1}/5  train={len(train_ids)}  val={len(val_ids)}")
        print('='*55)

        c_idx = train_picasso(
            cfg       = PICASSO_CONFIG,
            train_ids = train_ids,
            val_ids   = val_ids,
            fold_idx  = fold_idx,
        )
        results["endo_only"].append(c_idx)

    print_results(results)
