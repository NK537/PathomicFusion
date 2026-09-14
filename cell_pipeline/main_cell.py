# cell_pipeline/main_cell.py
#
# 5-fold cross-validation for the cell-graph survival pipeline.
#
# Prerequisites (run in order):
#   python3 cell_pipeline/step1_segment_cells.py --fallback   # or with HoVer-Net
#   python3 cell_pipeline/step2_crop_cells.py
#   python3 cell_pipeline/step3_embed_cells.py [--spatiopath]
#   python3 cell_pipeline/step4_build_graphs.py
#   python3 cell_pipeline/main_cell.py
#
# Usage:
#   cd /home/admin1/PathomicFusion
#   python3 cell_pipeline/main_cell.py

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cell_pipeline.config_cell  import CELL_CONFIG
from cell_pipeline.train_cell   import train_cell
from histo_pipeline.patient_histo_dataset import normalize_pat_id, load_labels


def get_train_patients(cfg):
    """Returns (patient_ids, events) from the label file."""
    label_df = load_labels(cfg["histo_label_xlsx"], cfg["tte_label_xlsx"])

    # Only keep patients who have a graph built
    graph_dir = cfg["cell_graph_dir"]
    available = {
        os.path.splitext(f)[0]
        for f in os.listdir(graph_dir)
        if f.endswith(".pt")
    } if os.path.isdir(graph_dir) else set()

    label_df = label_df[label_df.index.isin(available)]

    pat_ids = label_df.index.tolist()
    events  = label_df["event"].tolist()

    print(f"Patients with graphs : {len(pat_ids)}")
    print(f"Events               : {sum(events)}")
    print(f"Censored             : {len(events) - sum(events)}")
    return pat_ids, events


def print_results(results):
    print("\n" + "=" * 60)
    print("FINAL 5-Fold CV Results -- Cell Graph Pipeline")
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
    out = "cell_cv_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    pat_ids, events = get_train_patients(CELL_CONFIG)

    if len(pat_ids) < 5:
        print(f"ERROR: only {len(pat_ids)} patients with graphs — need ≥ 5 for 5-fold CV.")
        print("Run Steps 1-4 first to build cell graphs.")
        sys.exit(1)

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds   = list(skf.split(pat_ids, events))
    results = {"cell_gnn": []}
    all_oof = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ids = [pat_ids[i] for i in train_idx]
        val_ids   = [pat_ids[i] for i in val_idx]

        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/5  train={len(train_ids)}  val={len(val_ids)}")
        print("="*60)

        c_idx, fold_preds = train_cell(
            cfg       = CELL_CONFIG,
            train_ids = train_ids,
            val_ids   = val_ids,
            fold_idx  = fold_idx,
        )
        results["cell_gnn"].append(c_idx)

        if fold_preds is not None:
            fold_preds["fold"] = fold_idx
            all_oof.append(fold_preds)

    print_results(results)

    if all_oof:
        oof_df = pd.concat(all_oof, ignore_index=True)
        oof_df.to_csv("cell_oof_predictions.csv", index=False)
        print(f"OOF predictions saved to cell_oof_predictions.csv  ({len(oof_df)} rows)")
