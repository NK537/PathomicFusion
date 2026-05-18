"""
km_analysis.py
==============
Kaplan-Meier survival analysis for all four model conditions:
  - old_mlp       : UNI + MLP + Attention
  - new_mlp       : UNI + MLP + Cross-Attention
  - old_geneformer: UNI + Geneformer + Attention
  - new_geneformer: UNI + Geneformer + Cross-Attention

How it works
------------
1. Loops over all 5 folds for each condition.
2. Loads the saved best model checkpoint for that fold.
3. Runs inference on the VALIDATION patients for that fold
   (same KFold split as training, so no data leakage).
4. Aggregates predictions across all 5 folds → whole-cohort risk scores.
5. Stratifies patients at median risk → high / low risk groups.
6. Plots Kaplan-Meier curves and runs log-rank test.
7. Saves one PNG per model + one 2x2 summary figure.

Usage
-----
    python km_analysis.py

Outputs (saved to km_results/)
-------------------------------
    km_old_mlp.png
    km_new_mlp.png
    km_old_geneformer.png
    km_new_geneformer.png
    km_summary_2x2.png
    km_results.csv          ← p-values and n for every model
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# ── your project imports (same as train_for_cancer.py) ──────────────────────
from models.MLP import MLPBranch
from models.uni_mil_branch import UNIMILBranch
from models.geneformer_branch import GeneformerBranch
from fusion.attention_fusion import AttentionFusion
from fusion.cross_attention_fusion import CrossAttentionFusion
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from pytorch_dataset_loader.patient_mil_dataset import PatientMILDataset
from pytorch_dataset_loader.patient_mil_geneformer_dataset import PatientMILGeneformerDataset
from pytorch_dataset_loader.geneformer_collate import geneformer_collate_fn
from geneformer_utils.gene_tokenizer import BulkGeneformerTokenizer, load_token_map

# ── configuration (must match main.py exactly) ───────────────────────────────
CFG = {
    "patch_csv":            "data/TCGA_GBMLGG/patches_with_labels.csv",
    "patch_dir":            "data/TCGA_GBMLGG/patches/",
    "gene_csv":             "data/TCGA_GBMLGG/gene data/clean_gene_expression.csv",
    "label_csv":            "data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv",
    "use_foundation":       True,
    "emb_dir":              "data/TCGA_GBMLGG/uni_embeddings/",
    "foundation_dim":       1024,
    "k_patches":            32,
    "use_geneformer":       True,
    "geneformer_token_map": "data/TCGA_GBMLGG/geneformer/gene_token_map.json",
    "geneformer_max_len":   2048,
    "geneformer_freeze":    True,
}

CANCER_NAME  = "GBMLGG"
CHECKPOINT_DIR = "Best_Model1"
OUTPUT_DIR     = "km_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── model configurations ─────────────────────────────────────────────────────
# Each entry: (model_type, use_geneformer, display_label)
MODEL_CONFIGS = [
    ("old", False, "M1: UNI + MLP + Attention"),
    ("new", False, "M2: UNI + MLP + Cross-Attention"),
    ("old", True,  "M3: UNI + Geneformer + Attention"),
    ("new", True,  "M4: UNI + Geneformer + Cross-Attention"),
]

# checkpoint filename pattern (must match train_for_cancer.py line 466)
# f"Best_Model/best_model_{cancer_name}_fold{fold_idx}_{model_type}{gene_tag}.pth"
def ckpt_path(fold_idx, model_type, use_geneformer):
    gene_tag = "_geneformer" if use_geneformer else "_mlp"
    return os.path.join(
        CHECKPOINT_DIR,
        f"best_model_{CANCER_NAME}_fold{fold_idx}_{model_type}{gene_tag}.pth"
    )


# ── build model from scratch (must match train_for_cancer.py) ────────────────
def build_model(model_type, use_geneformer, gene_input_dim):
    """Instantiate cnn_branch, mlp_branch, fusion_layer — no weights loaded yet."""
    cnn_branch = UNIMILBranch(uni_dim=CFG["foundation_dim"], out_dim=64)

    if use_geneformer:
        mlp_branch = GeneformerBranch(
            out_dim=64,
            freeze_backbone=CFG["geneformer_freeze"]
        )
    else:
        mlp_branch = MLPBranch(input_dim=gene_input_dim, feature_dim=64)

    if model_type == "old":
        fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128)
    else:
        fusion_layer = CrossAttentionFusion(d_model=64, n_heads=4, n_gene_tokens=8)

    return cnn_branch, mlp_branch, fusion_layer


def load_checkpoint(cnn_branch, mlp_branch, fusion_layer, fold_idx, model_type, use_geneformer):
    path = ckpt_path(fold_idx, model_type, use_geneformer)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Make sure CHECKPOINT_DIR is correct and training has completed."
        )
    ckpt = torch.load(path, map_location=DEVICE)
    cnn_branch.load_state_dict(ckpt["cnn"])
    mlp_branch.load_state_dict(ckpt["mlp"])
    fusion_layer.load_state_dict(ckpt["fusion"])
    cnn_branch.eval()
    mlp_branch.eval()
    fusion_layer.eval()
    return cnn_branch, mlp_branch, fusion_layer


# ── inference on one fold's validation set ───────────────────────────────────
@torch.no_grad()
def run_inference(cnn_branch, mlp_branch, fusion_layer,
                  val_loader, use_geneformer):
    """Return (risk_scores, surv_times, events) as numpy arrays."""
    all_scores, all_times, all_events = [], [], []

    for batch in val_loader:
        if use_geneformer:
            uni_embs, input_ids, attn_mask, surv_times, events, grades, pids = batch
            uni_embs    = uni_embs.to(DEVICE)
            input_ids   = input_ids.to(DEVICE)
            attn_mask   = attn_mask.to(DEVICE)
            surv_times  = surv_times.to(DEVICE)
            events      = events.to(DEVICE)
            cnn_feats = cnn_branch(uni_embs)
            mlp_feats = mlp_branch(input_ids, attn_mask)
        else:
            uni_embs, gene_vectors, surv_times, events, grades, pids = batch
            uni_embs     = uni_embs.to(DEVICE)
            gene_vectors = gene_vectors.to(DEVICE)
            surv_times   = surv_times.to(DEVICE)
            events       = events.to(DEVICE)
            cnn_feats = cnn_branch(uni_embs)
            mlp_feats = mlp_branch(gene_vectors)

        scores = fusion_layer(cnn_feats, mlp_feats)

        all_scores.extend(scores.cpu().numpy().flatten().tolist())
        all_times.extend(surv_times.cpu().numpy().flatten().tolist())
        all_events.extend(events.cpu().numpy().flatten().tolist())

    return (
        np.array(all_scores),
        np.array(all_times),
        np.array(all_events)
    )


# ── KM plot for one model ─────────────────────────────────────────────────────
def plot_km(risk_scores, surv_times, events, label, save_path, ax=None):
    """
    Stratify at median risk, fit KM curves, run log-rank test.
    Returns p_value (float).
    If ax is provided, draws into that axes (for the 2x2 summary).
    Otherwise creates its own figure.
    """
    median_risk = np.median(risk_scores)
    high_mask   = risk_scores >= median_risk
    low_mask    = ~high_mask

    n_high = int(high_mask.sum())
    n_low  = int(low_mask.sum())

    # log-rank test
    lr = logrank_test(
        surv_times[high_mask], surv_times[low_mask],
        event_observed_A=events[high_mask],
        event_observed_B=events[low_mask]
    )
    p_value = lr.p_value

    # ── plot ──────────────────────────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()

    kmf_high.fit(
        surv_times[high_mask],
        event_observed=events[high_mask],
        label=f"High Risk (n={n_high})"
    )
    kmf_low.fit(
        surv_times[low_mask],
        event_observed=events[low_mask],
        label=f"Low Risk (n={n_low})"
    )

    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#D62728", linewidth=2)
    kmf_low.plot_survival_function( ax=ax, ci_show=True, color="#1F77B4", linewidth=2)

    # format p-value nicely
    if p_value < 1e-4:
        p_str = f"p = {p_value:.2e}"
    else:
        p_str = f"p = {p_value:.4f}"

    ax.set_title(f"{label}\nLog-rank {p_str}", fontsize=11, pad=8)
    ax.set_xlabel("Time (Days)", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if own_fig:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved individual plot: {save_path}")

    return p_value, n_high, n_low


# ── main analysis loop ────────────────────────────────────────────────────────
def main():
    # ── load gene dataset once ───────────────────────────────────────────────
    gene_dataset = GeneDataset(
        gene_expression_csv=CFG["gene_csv"],
        patient_labels_csv=CFG["label_csv"]
    )
    gene_input_dim = gene_dataset.gene_df.shape[1]
    print(f"Gene input dim: {gene_input_dim}")

    # ── build the same patient list and KFold as main.py ────────────────────
    patch_df   = pd.read_csv(CFG["patch_csv"])
    patch_pids = set(patch_df["TCGA_ID"].unique())
    gene_pids  = set(gene_dataset.gene_df.index)
    patient_ids = np.array(sorted(list(patch_pids.intersection(gene_pids))))
    print(f"Total matched patients: {len(patient_ids)}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # MUST match main.py

    # ── geneformer tokenizer ─────────────────────────────────────────────────
    token_map      = load_token_map(CFG["geneformer_token_map"])
    gene_tokenizer = BulkGeneformerTokenizer(
        gene_names=list(gene_dataset.gene_df.columns),
        token_map=token_map,
        max_len=CFG.get("geneformer_max_len", 2048),
    )

    # ── results storage ──────────────────────────────────────────────────────
    summary_rows = []

    # ── 2x2 summary figure ───────────────────────────────────────────────────
    fig2x2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()  # M1=0, M2=1, M3=2, M4=3

    for model_idx, (model_type, use_geneformer, display_label) in enumerate(MODEL_CONFIGS):

        gene_tag  = "geneformer" if use_geneformer else "mlp"
        run_label = f"{model_type}_{gene_tag}"
        print(f"\n{'='*60}")
        print(f"Processing: {display_label}")
        print(f"{'='*60}")

        # accumulate predictions across all 5 folds
        all_scores, all_times, all_events = [], [], []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
            val_ids = patient_ids[val_idx]

            print(f"  Fold {fold_idx}: loading checkpoint and running inference ...")

            # ── build model ──────────────────────────────────────────────────
            cnn_branch, mlp_branch, fusion_layer = build_model(
                model_type, use_geneformer, gene_input_dim
            )
            cnn_branch    = cnn_branch.to(DEVICE)
            mlp_branch    = mlp_branch.to(DEVICE)
            fusion_layer  = fusion_layer.to(DEVICE)

            # ── load checkpoint ──────────────────────────────────────────────
            cnn_branch, mlp_branch, fusion_layer = load_checkpoint(
                cnn_branch, mlp_branch, fusion_layer,
                fold_idx, model_type, use_geneformer
            )

            # ── build val dataset for this fold ─────────────────────────────
            if use_geneformer:
                val_dataset = PatientMILGeneformerDataset(
                    patch_csv=CFG["patch_csv"],
                    emb_dir=CFG["emb_dir"],
                    gene_df=gene_dataset.gene_df,
                    gene_tokenizer=gene_tokenizer,
                    k_patches=CFG["k_patches"],
                    subset_ids=val_ids,
                    token_cache_path=(
                        f"data/TCGA_GBMLGG/geneformer/val_tokens_fold{fold_idx}.pt"
                    ),
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=4,
                    shuffle=False,
                    collate_fn=geneformer_collate_fn
                )
            else:
                val_dataset = PatientMILDataset(
                    patch_csv=CFG["patch_csv"],
                    emb_dir=CFG["emb_dir"],
                    gene_df=gene_dataset.gene_df,
                    k_patches=CFG["k_patches"],
                    subset_ids=val_ids
                )
                val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

            # ── run inference ────────────────────────────────────────────────
            scores, times, evts = run_inference(
                cnn_branch, mlp_branch, fusion_layer,
                val_loader, use_geneformer
            )

            all_scores.append(scores)
            all_times.append(times)
            all_events.append(evts)

            print(f"  Fold {fold_idx}: {len(scores)} patients, "
                  f"risk range [{scores.min():.3f}, {scores.max():.3f}]")

            # free GPU memory between folds
            del cnn_branch, mlp_branch, fusion_layer
            torch.cuda.empty_cache()

        # ── concatenate all-fold predictions ─────────────────────────────────
        all_scores = np.concatenate(all_scores)
        all_times  = np.concatenate(all_times)
        all_events = np.concatenate(all_events)

        print(f"\n  Total patients aggregated: {len(all_scores)}")
        print(f"  Events: {int(all_events.sum())} / {len(all_events)}")

        # ── individual KM plot ────────────────────────────────────────────────
        indiv_path = os.path.join(OUTPUT_DIR, f"km_{run_label}.png")
        p_val, n_high, n_low = plot_km(
            all_scores, all_times, all_events,
            label=display_label,
            save_path=indiv_path
        )

        # ── draw into 2x2 panel ───────────────────────────────────────────────
        plot_km(
            all_scores, all_times, all_events,
            label=display_label,
            save_path=None,
            ax=axes_flat[model_idx]
        )

        print(f"  Log-rank p-value: {p_val:.4e}")
        print(f"  High-risk n={n_high}, Low-risk n={n_low}")

        summary_rows.append({
            "Model":        display_label,
            "run_label":    run_label,
            "n_high_risk":  n_high,
            "n_low_risk":   n_low,
            "p_value":      p_val,
            "significant":  "Yes" if p_val < 0.05 else "No"
        })

    # ── save 2x2 figure ───────────────────────────────────────────────────────
    fig2x2.suptitle(
        "Kaplan-Meier Survival Curves — TCGA-GBMLGG (769 patients)\n"
        "Stratified at median predicted risk score",
        fontsize=13, y=1.01
    )
    fig2x2.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "km_summary_2x2.png")
    fig2x2.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig2x2)
    print(f"\nSaved 2x2 summary figure: {summary_path}")

    # ── save results CSV ─────────────────────────────────────────────────────
    df_results = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "km_results.csv")
    df_results.to_csv(csv_path, index=False)

    # ── print final summary table ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("KAPLAN-MEIER SUMMARY")
    print("="*70)
    print(f"{'Model':<45} {'n_high':>7} {'n_low':>7} {'p-value':>12} {'Sig?':>6}")
    print("-"*70)
    for row in summary_rows:
        print(
            f"{row['Model']:<45} "
            f"{row['n_high_risk']:>7} "
            f"{row['n_low_risk']:>7} "
            f"{row['p_value']:>12.4e} "
            f"{'✓' if row['significant']=='Yes' else '✗':>6}"
        )
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  km_old_mlp.png          — M1 individual KM plot")
    print(f"  km_new_mlp.png          — M2 individual KM plot")
    print(f"  km_old_geneformer.png   — M3 individual KM plot")
    print(f"  km_new_geneformer.png   — M4 individual KM plot")
    print(f"  km_summary_2x2.png      — 2x2 panel for paper (Figure 1)")
    print(f"  km_results.csv          — p-values and n for Table 4")


if __name__ == "__main__":
    main()