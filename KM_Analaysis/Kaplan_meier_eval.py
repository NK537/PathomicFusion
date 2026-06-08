"""
kaplan_meier_eval.py

Kaplan-Meier survival analysis for all model variants:
  - old_geneformer  (AttentionFusion  + GeneformerBranch)
  - new_geneformer  (CrossAttentionFusion + GeneformerBranch)

Run AFTER training is complete so the best model checkpoints exist.

Usage:
    python kaplan_meier_eval.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")           # headless – remove if running in a notebook

from torch.utils.data import DataLoader
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import KFold

# ── project imports ───────────────────────────────────────────────────────────
from models.CNN import CNNBranch
from models.MLP import MLPBranch
from models.uni_mil_branch import UNIMILBranch
from models.geneformer_branch import GeneformerBranch
from fusion.attention_fusion import AttentionFusion
from fusion.cross_attention_fusion import CrossAttentionFusion

from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from pytorch_dataset_loader.patient_mil_geneformer_dataset import PatientMILGeneformerDataset
from pytorch_dataset_loader.geneformer_collate import geneformer_collate_fn
from geneformer_utils.gene_tokenizer import BulkGeneformerTokenizer, load_token_map
# ─────────────────────────────────────────────────────────────────────────────


# ── cancer config (must match main.py) ───────────────────────────────────────
cancers = {
    "GBMLGG": {
        "patch_csv":    "data/TCGA_GBMLGG/patches_with_labels.csv",
        "patch_dir":    "data/TCGA_GBMLGG/patches/",
        "gene_csv":     "data/TCGA_GBMLGG/gene data/clean_gene_expression.csv",
        "label_csv":    "data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv",
        "use_foundation":   True,
        "emb_dir":          "data/TCGA_GBMLGG/uni_embeddings/",
        "foundation_dim":   1024,
        "k_patches":        32,
        "use_geneformer":   True,
        "geneformer_token_map": "data/TCGA_GBMLGG/geneformer/gene_token_map.json",
        "geneformer_max_len":   2048,
        "geneformer_freeze":    False,
    }
}

# Which model variants to evaluate
MODEL_VARIANTS = [
    {"model_type": "old", "use_geneformer": True,  "tag": "old_geneformer"},
    {"model_type": "new", "use_geneformer": True,  "tag": "new_geneformer"},
    {"model_type": "old", "use_geneformer": False, "tag": "old_mlp"},
    {"model_type": "new", "use_geneformer": False, "tag": "new_mlp"},
]

OUTPUT_DIR = "km_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────


def build_model(cfg, model_type, use_geneformer, gene_input_dim, device):
    """Instantiate branches + fusion layer (no weights loaded yet)."""
    cnn_branch = UNIMILBranch(uni_dim=cfg["foundation_dim"], out_dim=64)

    if use_geneformer:
        mlp_branch = GeneformerBranch(
            out_dim=64,
            freeze_backbone=cfg.get("geneformer_freeze", True),
        )
    else:
        mlp_branch = MLPBranch(input_dim=gene_input_dim, feature_dim=64)

    if model_type == "old":
        fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128)
    elif model_type == "new":
        fusion_layer = CrossAttentionFusion(d_model=64, n_heads=4, n_gene_tokens=8, fusion_dim=128)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    cnn_branch.to(device)
    mlp_branch.to(device)
    fusion_layer.to(device)
    return cnn_branch, mlp_branch, fusion_layer


@torch.no_grad()
def run_inference(cnn_branch, mlp_branch, fusion_layer, loader, device, use_geneformer):
    """Return (scores, times, events) arrays for every patient in the loader."""
    cnn_branch.eval(); mlp_branch.eval(); fusion_layer.eval()

    all_scores, all_times, all_events = [], [], []

    for batch in loader:
        if use_geneformer:
            # PatientMILGeneformerDataset batch layout:
            uni_embs, input_ids, attention_mask, surv_times, events, grades, pids = batch
            uni_embs       = uni_embs.to(device)
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            surv_times     = surv_times.to(device)
            events         = events.to(device)

            cnn_feats = cnn_branch(uni_embs)
            mlp_feats = mlp_branch(input_ids, attention_mask)
        else:
            uni_embs, gene_vectors, surv_times, events, grades, pids = batch
            uni_embs     = uni_embs.to(device)
            gene_vectors = gene_vectors.to(device)
            surv_times   = surv_times.to(device)
            events       = events.to(device)

            cnn_feats = cnn_branch(uni_embs)
            mlp_feats = mlp_branch(gene_vectors)

        scores = fusion_layer(cnn_feats, mlp_feats)   # (B, 1) or (B,)

        # .flatten() safely handles (B,1), (B,) and edge-case batch size of 1
        all_scores.extend(scores.cpu().flatten().tolist())
        all_times.extend(surv_times.cpu().flatten().tolist())
        all_events.extend(events.cpu().flatten().tolist())

    return np.array(all_scores), np.array(all_times), np.array(all_events)


def plot_km(times_hi, events_hi, times_lo, events_lo, title, save_path):
    """
    Plot high-risk vs low-risk Kaplan-Meier curves with log-rank p-value.
    """
    lr = logrank_test(times_hi, times_lo, event_observed_A=events_hi, event_observed_B=events_lo)
    p_val = lr.p_value

    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 5))

    kmf_hi.fit(times_hi, event_observed=events_hi, label="High risk")
    kmf_hi.plot_survival_function(ax=ax, ci_show=True, color="red")

    kmf_lo.fit(times_lo, event_observed=events_lo, label="Low risk")
    kmf_lo.plot_survival_function(ax=ax, ci_show=True, color="blue")

    ax.set_title(f"{title}\nLog-rank p = {p_val:.4f}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}  (p = {p_val:.4f})")
    return p_val


def evaluate_model_km(
    cancer_name, cfg, model_type, use_geneformer, tag,
    patient_ids, gene_dataset, gene_tokenizer, n_splits=5
):
    """
    For each of the 5 folds, load the saved checkpoint, run inference on the
    val split, accumulate predictions, then plot one KM curve per fold AND
    one aggregated KM curve across all folds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_patches = cfg.get("k_patches", 32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Accumulate across folds for an overall plot
    agg_scores, agg_times, agg_events = [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        val_ids = patient_ids[val_idx]

        # ── checkpoint path (must match what train_for_cancer saves) ──────────
        gene_tag = "_geneformer" if use_geneformer else "_mlp"
        ckpt_path = (
            f"Best_Model1/best_model_{cancer_name}_fold{fold_idx}_{model_type}{gene_tag}.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint not found: {ckpt_path} – skipping fold {fold_idx}")
            continue

        # ── build val dataset & loader ────────────────────────────────────────
        if use_geneformer:
            val_dataset = PatientMILGeneformerDataset(
                patch_csv=cfg["patch_csv"],
                emb_dir=cfg["emb_dir"],
                gene_df=gene_dataset.gene_df,
                gene_tokenizer=gene_tokenizer,
                k_patches=k_patches,
                subset_ids=val_ids,
                token_cache_path=f"data/TCGA_GBMLGG/geneformer/val_tokens_fold{fold_idx}.pt",
            )
            val_loader = DataLoader(
                val_dataset, batch_size=4, shuffle=False,
                collate_fn=geneformer_collate_fn
            )
        else:
            from pytorch_dataset_loader.patient_mil_dataset import PatientMILDataset
            val_dataset = PatientMILDataset(
                patch_csv=cfg["patch_csv"],
                emb_dir=cfg["emb_dir"],
                gene_df=gene_dataset.gene_df,
                k_patches=k_patches,
                subset_ids=val_ids,
            )
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # ── load weights ──────────────────────────────────────────────────────
        cnn_branch, mlp_branch, fusion_layer = build_model(
            cfg, model_type, use_geneformer,
            gene_input_dim=gene_dataset.gene_df.shape[1],
            device=device,
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        cnn_branch.load_state_dict(ckpt["cnn"])
        mlp_branch.load_state_dict(ckpt["mlp"])
        fusion_layer.load_state_dict(ckpt["fusion"])

        # ── inference ─────────────────────────────────────────────────────────
        scores, times, events = run_inference(
            cnn_branch, mlp_branch, fusion_layer,
            val_loader, device, use_geneformer
        )

        agg_scores.append(scores)
        agg_times.append(times)
        agg_events.append(events)

        # ── per-fold KM plot ──────────────────────────────────────────────────
        median_risk = np.median(scores)
        hi_mask = scores >= median_risk
        lo_mask = ~hi_mask

        plot_km(
            times[hi_mask], events[hi_mask],
            times[lo_mask], events[lo_mask],
            title=f"{cancer_name} | {tag} | Fold {fold_idx}",
            save_path=os.path.join(OUTPUT_DIR, f"{cancer_name}_{tag}_fold{fold_idx}_km.png"),
        )

    # ── aggregated KM across all folds ────────────────────────────────────────
    if agg_scores:
        all_scores = np.concatenate(agg_scores)
        all_times  = np.concatenate(agg_times)
        all_events = np.concatenate(agg_events)

        median_risk = np.median(all_scores)
        hi_mask = all_scores >= median_risk
        lo_mask = ~hi_mask

        plot_km(
            all_times[hi_mask], all_events[hi_mask],
            all_times[lo_mask], all_events[lo_mask],
            title=f"{cancer_name} | {tag} | All folds (aggregated)",
            save_path=os.path.join(OUTPUT_DIR, f"{cancer_name}_{tag}_all_folds_km.png"),
        )


def main():
    for cancer_name, cfg in cancers.items():
        print(f"\n{'='*60}")
        print(f"  Kaplan-Meier evaluation: {cancer_name}")
        print(f"{'='*60}")

        # ── shared setup ──────────────────────────────────────────────────────
        gene_dataset = GeneDataset(
            gene_expression_csv=cfg["gene_csv"],
            patient_labels_csv=cfg["label_csv"],
        )

        from pytorch_dataset_loader.patches_embedding_dataset import PatchEmbeddingDataset
        patch_dataset = PatchEmbeddingDataset(
            csv_file=cfg["patch_csv"],
            emb_dir=cfg["emb_dir"],
        )

        patch_pids = set(patch_dataset.data_frame["TCGA_ID"].unique())
        gene_pids  = set(gene_dataset.gene_df.index)
        patient_ids = np.array(sorted(list(patch_pids & gene_pids)))

        # GeneFormer tokenizer (shared across variants that need it)
        token_map = load_token_map(cfg["geneformer_token_map"])
        gene_tokenizer = BulkGeneformerTokenizer(
            gene_names=list(gene_dataset.gene_df.columns),
            token_map=token_map,
            max_len=cfg.get("geneformer_max_len", 2048),
        )

        # ── evaluate each variant ─────────────────────────────────────────────
        for variant in MODEL_VARIANTS:
            print(f"\n── Variant: {variant['tag']} ──")
            evaluate_model_km(
                cancer_name=cancer_name,
                cfg=cfg,
                model_type=variant["model_type"],
                use_geneformer=variant["use_geneformer"],
                tag=variant["tag"],
                patient_ids=patient_ids,
                gene_dataset=gene_dataset,
                gene_tokenizer=gene_tokenizer if variant["use_geneformer"] else None,
            )

    print(f"\nAll KM plots saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()