"""
main_picasso.py
---------------
5-fold cross-validation for the Picasso bimodal survival pipeline.

Runs four configurations and reports mean +/- std C-index:
  1. histo_only      - histopathology branch alone (ablation)
  2. endo_only       - endoscopy branch alone (ablation)
  3. concat_fusion   - both branches, simple concat + MLP head
  4. cross_attention - both branches, bidirectional cross-attention fusion  <-- primary

Usage:
    python main_picasso.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from config_picasso import PICASSO_CONFIG
from train_picasso import train_picasso


def get_all_patient_ids(cfg):
    """Return the full list of patient IDs present in all three CSVs."""
    histo_df = pd.read_csv(cfg["histo_patches_csv"])
    endo_df  = pd.read_csv(cfg["endo_frames_csv"])
    label_df = pd.read_csv(cfg["label_csv"])

    histo_ids = set(histo_df["patient_id"].unique())
    endo_ids  = set(endo_df["patient_id"].unique())
    label_ids = set(label_df["patient_id"].unique())

    valid_ids = sorted(histo_ids & endo_ids & label_ids)
    print(f"Total eligible patients: {len(valid_ids)}")
    return valid_ids


def run_cv(cfg, n_splits=5):
    """
    Run 5-fold CV for all four experimental configurations.

    Returns a dict: config_name -> list of per-fold C-indices
    """
    all_ids = get_all_patient_ids(cfg)
    kf      = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds   = list(kf.split(all_ids))

    # Four configs: name -> (fusion_type_override, ablation)
    configs = [
        ("histo_only",      "histo_only"),
        ("endo_only",       "endo_only"),
        ("concat_fusion",   None),          # cfg["fusion_type"] is overridden below
        ("cross_attention", None),
    ]

    results = {name: [] for name, _ in configs}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_ids = [all_ids[i] for i in train_idx]
        val_ids   = [all_ids[i] for i in val_idx]
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx+1}/{n_splits}  "
              f"train={len(train_ids)}  val={len(val_ids)}")
        print('='*60)

        for name, ablation in configs:
            # Clone config and set fusion_type for this run
            run_cfg = dict(cfg)
            if name == "concat_fusion":
                run_cfg["fusion_type"] = "concat"
            elif name == "cross_attention":
                run_cfg["fusion_type"] = "cross_attention"

            c_idx = train_picasso(
                cfg       = run_cfg,
                train_ids = train_ids,
                val_ids   = val_ids,
                fold_idx  = fold_idx,
                ablation  = ablation,
            )
            results[name].append(c_idx)

    return results


def print_results(results):
    print("\n" + "="*60)
    print("FINAL RESULTS — 5-Fold Cross-Validation")
    print("="*60)
    print(f"{'Config':<22} {'Mean C-idx':>12} {'Std':>8}  {'Per-fold'}")
    print("-"*60)
    for name, scores in results.items():
        arr   = np.array(scores)
        folds = "  ".join(f"{s:.4f}" for s in scores)
        print(f"{name:<22} {arr.mean():>12.4f} {arr.std():>8.4f}  {folds}")
    print("="*60)

    # Save CSV
    rows = []
    for name, scores in results.items():
        for fold_idx, s in enumerate(scores):
            rows.append({"config": name, "fold": fold_idx, "c_index": s})
    pd.DataFrame(rows).to_csv("picasso_cv_results.csv", index=False)
    print("Results saved to picasso_cv_results.csv")


if __name__ == "__main__":
    results = run_cv(PICASSO_CONFIG, n_splits=5)
    print_results(results)
