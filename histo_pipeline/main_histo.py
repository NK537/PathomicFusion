# histo_pipeline/main_histo.py
#
# 5-fold cross-validation for the Picasso histology survival pipeline.
#
# Split strategy:
#   Train_Outcome_rev2 = 0 or 1  -> training pool  (120 patients)
#   Train_Outcome_rev2 = 2       -> held-out test   (evaluated separately)
#   Train_Outcome_rev2 = -1      -> excluded
#
# Within the training pool, StratifiedKFold on event status gives 5 folds.
#
# Usage:
#   cd /home/usama/Projects/PathomicFusion
#   python histo_pipeline/main_histo.py

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from histo_pipeline.config_histo import HISTO_CONFIG
from histo_pipeline.patient_histo_dataset import normalize_pat_id
from histo_pipeline.train_histo import train_histo


def get_train_patients(cfg):
    """
    Read PICASSO_dataframe.xlsx and return all patients with valid outcomes.
    Aggregates per patient: outcome = max across WSIs (1 if any WSI has event).
    """
    df = pd.read_excel(cfg["histo_label_xlsx"])
    df["ID"]      = df["ID"].apply(normalize_pat_id)
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df = df[df["outcome"].isin([0.0, 1.0])].copy()

    # One row per patient
    pat_df = df.groupby("ID")["outcome"].max().reset_index()
    pat_df["outcome"] = pat_df["outcome"].astype(int)

    pat_ids = pat_df["ID"].tolist()
    events  = pat_df["outcome"].tolist()

    print(f"Training pool : {len(pat_ids)} patients")
    print(f"Events        : {sum(events)}")
    print(f"Censored      : {len(events) - sum(events)}")
    return pat_ids, events


def print_results(results):
    print("\n" + "=" * 60)
    print("FINAL 5-Fold CV Results -- Picasso Histology")
    print("=" * 60)
    print(f"{'Config':<20} {'Mean C-idx':>12} {'Std':>8}  Per-fold")
    print("-" * 60)
    for name, scores in results.items():
        arr   = np.array(scores)
        folds = "  ".join(f"{s:.4f}" for s in scores)
        print(f"{name:<20} {arr.mean():>12.4f} {arr.std():>8.4f}  {folds}")
    print("=" * 60)

    rows = [
        {"config": name, "fold": i, "c_index": s}
        for name, scores in results.items()
        for i, s in enumerate(scores)
    ]
    out = "histo_cv_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    pat_ids, events = get_train_patients(HISTO_CONFIG)

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds   = list(skf.split(pat_ids, events))
    results = {"histo_only": []}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ids = [pat_ids[i] for i in train_idx]
        val_ids   = [pat_ids[i] for i in val_idx]

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/5  train={len(train_ids)}  val={len(val_ids)}")
        print("="*60)

        c_idx = train_histo(
            cfg       = HISTO_CONFIG,
            train_ids = train_ids,
            val_ids   = val_ids,
            fold_idx  = fold_idx,
        )
        results["histo_only"].append(c_idx)

    print_results(results)
