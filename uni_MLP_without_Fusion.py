# uni_Geneformer_without_Fusion.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd

from models.uni_mil_branch import UNIMILBranch
from models.geneformer_branch import GeneformerBranch
from models.MLP import MLPBranch
from pytorch_dataset_loader.patient_mil_geneformer_dataset import PatientMILGeneformerDataset
from pytorch_dataset_loader.patient_mil_dataset import PatientMILDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from pytorch_dataset_loader.geneformer_collate import geneformer_collate_fn
from geneformer_utils.gene_tokenizer import BulkGeneformerTokenizer, load_token_map
from COX.cox_loss import CustomCoxLoss


# ------------------------------------------------------------------
# Survival head
# ------------------------------------------------------------------
class SurvivalHead(nn.Module):
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# Clear stale caches
# ------------------------------------------------------------------
def _clear_fold_caches(cancer_name: str, fold_idx: int):
    base = f"data/TCGA_{cancer_name}/geneformer"
    for split in ("train", "val"):
        path = os.path.join(base, f"{split}_tokens_fold{fold_idx}.pt")
        if os.path.exists(path):
            os.remove(path)
            print(f"  [Cache] Removed stale cache: {path}")


# ------------------------------------------------------------------
# Single-branch training for one fold
# ------------------------------------------------------------------
def run_fold(
    branch_name: str,       # "uni", "mlp", or "geneformer"
    cancer_name: str,
    cancer_data: dict,
    train_ids,
    val_ids,
    fold_idx: int,
    device: torch.device,
) -> dict:

    use_geneformer = cancer_data.get("use_geneformer", False)

    # ── Gene dataset ───────────────────────────────────────────────
    gene_dataset = GeneDataset(
        gene_expression_csv=cancer_data["gene_csv"],
        patient_labels_csv=cancer_data["label_csv"]
    )
    gene_input_dim = gene_dataset.gene_df.shape[1]

    k_patches = cancer_data.get("k_patches", 32)

    # ── Datasets depending on genomics branch type ─────────────────
    if use_geneformer:
        token_map      = load_token_map(cancer_data["geneformer_token_map"])
        gene_tokenizer = BulkGeneformerTokenizer(
            gene_names=list(gene_dataset.gene_df.columns),
            token_map=token_map,
            max_len=cancer_data.get("geneformer_max_len", 2048),
        )

        # Clear and rebuild caches for this fold
        _clear_fold_caches(cancer_name, fold_idx)
        cache_dir   = f"data/TCGA_{cancer_name}/geneformer"
        train_cache = os.path.join(cache_dir, f"train_tokens_fold{fold_idx}.pt")
        val_cache   = os.path.join(cache_dir, f"val_tokens_fold{fold_idx}.pt")

        train_dataset = PatientMILGeneformerDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            gene_tokenizer=gene_tokenizer,
            k_patches=k_patches,
            subset_ids=train_ids,
            token_cache_path=train_cache,
        )
        val_dataset = PatientMILGeneformerDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            gene_tokenizer=gene_tokenizer,
            k_patches=k_patches,
            subset_ids=val_ids,
            token_cache_path=val_cache,
        )
        collate = geneformer_collate_fn

    else:
        # MLP path — uses PatientMILDataset (no tokenizer needed)
        train_dataset = PatientMILDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            k_patches=k_patches,
            subset_ids=train_ids,
        )
        val_dataset = PatientMILDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            k_patches=k_patches,
            subset_ids=val_ids,
        )
        collate = None

    # ── DataLoaders ────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # ── Build branch ───────────────────────────────────────────────
    if branch_name == "uni":
        branch = UNIMILBranch(
            uni_dim=cancer_data["foundation_dim"],
            out_dim=64,
        ).to(device)

    elif branch_name == "mlp":
        branch = MLPBranch(
            input_dim=gene_input_dim,
            feature_dim=64,
        ).to(device)

    elif branch_name == "geneformer":
        branch = GeneformerBranch(
            out_dim=64,
            freeze_backbone=False,   # always unfreeze for fine-tuning
        ).to(device)

    else:
        raise ValueError(f"Unknown branch_name: '{branch_name}'")

    head = SurvivalHead(input_dim=64).to(device)
    os.makedirs("Best_Model_Without_Fusion", exist_ok=True)

    # ── Branch-aware optimizer ─────────────────────────────────────
    if branch_name == "geneformer":
        optimizer = optim.Adam([
            {"params": branch.parameters(), "lr": 1e-5},
            {"params": head.parameters(),   "lr": 5e-5},
        ], weight_decay=1e-4)
    else:
        optimizer = optim.Adam(
            list(branch.parameters()) + list(head.parameters()),
            lr=5e-5,
            weight_decay=1e-4,
        )

    # With this:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
    )

    cox_loss = CustomCoxLoss()

    # ── Training config ────────────────────────────────────────────
    num_epochs        = 50 if branch_name == "geneformer" else 30
    patience          = 15 if branch_name == "geneformer" else 10
    best_val_cindex   = 0.0
    best_train_cindex = 0.0
    epochs_no_improve = 0

    # ==============================================================
    # Epoch loop
    # ==============================================================
    for epoch in range(num_epochs):
        branch.train()
        head.train()

        running_loss = 0.0
        all_scores, all_times, all_events = [], [], []

        for batch in tqdm(
            train_loader,
            desc=(
                f"  [{branch_name.upper()}] {cancer_name} "
                f"Fold {fold_idx} Epoch {epoch+1}/{num_epochs}"
            ),
        ):
            # ── Unpack batch depending on dataset type ─────────────
            if use_geneformer:
                uni_embs, input_ids, attention_mask, surv_times, events, grades, patient_ids = batch
                uni_embs       = uni_embs.to(device)
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            else:
                uni_embs, gene_vectors, surv_times, events, grades, patient_ids = batch
                uni_embs     = uni_embs.to(device)
                gene_vectors = gene_vectors.to(device)

            surv_times = surv_times.to(device)
            events     = events.to(device)

            # ── Forward: only active branch ────────────────────────
            if branch_name == "uni":
                feats = branch(uni_embs)
            elif branch_name == "mlp":
                feats = branch(gene_vectors)
            else:  # geneformer
                feats = branch(input_ids, attention_mask)

            risk_scores = head(feats)
            loss        = cox_loss(risk_scores, surv_times, events)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not torch.isnan(loss):
                running_loss += loss.item()
                all_scores.extend(risk_scores.detach())
                all_times.extend(surv_times.detach())
                all_events.extend(events.detach())

        # ── Train C-index ──────────────────────────────────────────
        train_cindex = (
            concordance_index(
                torch.stack(all_times).cpu().numpy(),
                (-torch.stack(all_scores)).cpu().numpy(),
                torch.stack(all_events).cpu().numpy(),
            ) if all_scores else 0.0
        )

        # ==========================================================
        # Validation
        # ==========================================================
        branch.eval()
        head.eval()
        val_scores, val_times, val_events = [], [], []

        with torch.no_grad():
            for batch in val_loader:

                if use_geneformer:
                    uni_embs, input_ids, attention_mask, surv_times, events, grades, patient_ids = batch
                    uni_embs       = uni_embs.to(device)
                    input_ids      = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                else:
                    uni_embs, gene_vectors, surv_times, events, grades, patient_ids = batch
                    uni_embs     = uni_embs.to(device)
                    gene_vectors = gene_vectors.to(device)

                surv_times = surv_times.to(device)
                events     = events.to(device)

                if branch_name == "uni":
                    feats = branch(uni_embs)
                elif branch_name == "mlp":
                    feats = branch(gene_vectors)
                else:
                    feats = branch(input_ids, attention_mask)

                risk_scores = head(feats)
                val_scores.extend(risk_scores)
                val_times.extend(surv_times)
                val_events.extend(events)

        val_cindex = (
            concordance_index(
                torch.stack(val_times).cpu().numpy(),
                (-torch.stack(val_scores)).cpu().numpy(),
                torch.stack(val_events).cpu().numpy(),
            ) if val_scores else 0.0
        )

        print(
            f"  Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {running_loss:.4f} | "
            f"Train C-idx: {train_cindex:.4f} | "
            f"Val C-idx: {val_cindex:.4f}"
        )

        scheduler.step(val_cindex)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch+1}/{num_epochs}] "
            f"Loss: {running_loss:.4f} | "
            f"Train C-idx: {train_cindex:.4f} | "
            f"Val C-idx: {val_cindex:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ── Checkpoint + early stopping ────────────────────────────
        if val_cindex > best_val_cindex + 0.001:
            best_val_cindex   = val_cindex
            best_train_cindex = train_cindex
            epochs_no_improve = 0

            ckpt_path = (
                f"Best_Model_Without_Fusion/"
                f"best_{branch_name}_{cancer_name}_fold{fold_idx}.pth"
            )
            torch.save(
                {"branch": branch.state_dict(), "head": head.state_dict()},
                ckpt_path,
            )
            print(f"  ✔ Saved → {ckpt_path}  (val C-idx: {best_val_cindex:.4f})")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  ⚑ Early stopping at epoch {epoch+1}.")
            break

    return {
        "fold":          fold_idx,
        "branch":        branch_name,
        "cancer":        cancer_name,
        "train_cindex":  best_train_cindex,
        "val_cindex":    best_val_cindex,
    }


# ------------------------------------------------------------------
# 5-Fold CV runner
# ------------------------------------------------------------------
def run_5fold_branches(cancer_name: str, cancer_data: dict):
    from pytorch_dataset_loader.patches_embedding_dataset import PatchEmbeddingDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  5-Fold Branch Evaluation — {cancer_name}  ({device})")
    print(f"{'='*60}")

    patch_dataset = PatchEmbeddingDataset(
        csv_file=cancer_data["patch_csv"],
        emb_dir=cancer_data["emb_dir"],
    )
    all_ids = patch_dataset.data_frame["TCGA_ID"].unique()
    print(f"  Total unique patients: {len(all_ids)}")

    # ── Decide which genomics branch to evaluate ───────────────────
    use_geneformer  = cancer_data.get("use_geneformer", False)
    genomics_branch = "geneformer" if use_geneformer else "mlp"
    branches        = ["uni", genomics_branch]

    print(f"  Branches to evaluate: {branches}")

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_ids), start=1):
        train_ids = all_ids[train_idx]
        val_ids   = all_ids[val_idx]

        print(f"\n{'─'*60}")
        print(f"  Fold {fold_idx}/5  |  train={len(train_ids)}  val={len(val_ids)}")
        print(f"{'─'*60}")

        for branch_name in branches:
            print(f"\n  >> Branch: {branch_name.upper()}")
            fold_result = run_fold(
                branch_name=branch_name,
                cancer_name=cancer_name,
                cancer_data=cancer_data,
                train_ids=train_ids,
                val_ids=val_ids,
                fold_idx=fold_idx,
                device=device,
            )
            results.append(fold_result)

    # ── Summary ────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  Full Results — {cancer_name}")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    print(f"\n{'─'*60}")
    print("  Mean ± Std Val C-index per branch:")
    for branch_name in branches:
        subset    = df[df["branch"] == branch_name]["val_cindex"]
        best_fold = subset.values.argmax() + 1
        print(
            f"  {branch_name.upper():>12s}:  "
            f"{subset.mean():.4f} ± {subset.std():.4f}  "
            f"(best fold: {best_fold})"
        )
    print(f"{'─'*60}\n")

    # ── Save CSV ───────────────────────────────────────────────────
    os.makedirs("Best_Model_Without_Fusion", exist_ok=True)
    csv_path = f"Best_Model_Without_Fusion/{cancer_name}_branch_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Results saved → {csv_path}")

    return df


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    from config import CANCER_CONFIGS

    for cancer_name, cancer_data in CANCER_CONFIGS.items():
        run_5fold_branches(cancer_name, cancer_data)