import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lifelines.utils import concordance_index

from models.CNN import CNNBranch
from models.MLP import MLPBranch
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from fusion.attention_fusion import AttentionFusion
from COX.cox_loss import CustomCoxLoss

from sklearn.model_selection import KFold
import numpy as np


def _build_gene_tensor_cache(gene_dataset, patient_ids, device):
    """
    Pre-compute a dict of {patient_id -> gene tensor on CPU}.
    Building this once per fold eliminates repeated pandas .loc[] calls
    inside the hot training loop — the single biggest bottleneck.
    """
    cache = {}
    gene_df = gene_dataset.gene_df
    for pid in patient_ids:
        if pid in gene_df.index:
            cache[pid] = torch.tensor(
                gene_df.loc[pid].values, dtype=torch.float32
            )
    return cache


def _run_epoch(loader, cnn_branch, mlp_branch, fusion_layer,
               gene_cache, device, optimizer, cox_loss, is_train, desc):
    """
    Single forward pass over loader.
    Shared between train and validation to eliminate code duplication
    and keep the gene-cache lookup path identical in both modes.
    """
    if is_train:
        cnn_branch.train(); mlp_branch.train(); fusion_layer.train()
    else:
        cnn_branch.eval(); mlp_branch.eval(); fusion_layer.eval()

    all_scores, all_times, all_events = [], [], []
    running_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            patches, surv_times, events, grades, patient_ids = batch

            # --- filter to patients that have gene data (vectorised) ---
            valid_mask = [i for i, pid in enumerate(patient_ids)
                          if pid in gene_cache]
            if not valid_mask:
                continue

            idx = valid_mask
            patches_v     = patches[idx].to(device, non_blocking=True)
            surv_times_v  = surv_times[idx].to(device, non_blocking=True)
            events_v      = events[idx].to(device, non_blocking=True)
            gene_vectors  = torch.stack(
                [gene_cache[patient_ids[i]] for i in idx]
            ).to(device, non_blocking=True)

            cnn_feats       = cnn_branch(patches_v)
            mlp_feats       = mlp_branch(gene_vectors)
            survival_scores = fusion_layer(cnn_feats, mlp_feats)

            if is_train:
                loss = cox_loss(survival_scores, surv_times_v, events_v)
                optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
                loss.backward()
                optimizer.step()

                if not torch.isnan(loss):
                    running_loss += loss.item()

            # Collect on CPU immediately to free GPU memory
            all_scores.extend(survival_scores.detach().cpu())
            all_times.extend(surv_times_v.detach().cpu())
            all_events.extend(events_v.detach().cpu())

    if not all_scores:
        return 0.0, running_loss

    # concordance_index expects numpy arrays (lifelines) — convert once
    scores_np = torch.stack(all_scores).numpy()
    times_np  = torch.stack(all_times).numpy()
    events_np = torch.stack(all_events).numpy()

    cindex = concordance_index(-scores_np, times_np, events_np)
    return cindex, running_loss


def train_for_cancer(cancer_name, cancer_data):
    print(f"\n==== Training for {cancer_name} (5-Fold CV) ====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4 if device.type == "cuda" else 0   # 0 on CPU avoids overhead

    patch_dataset = PatchDataset(
        csv_file=cancer_data["patch_csv"],
        image_dir=cancer_data["patch_dir"]
    )
    gene_dataset = GeneDataset(
        gene_expression_csv=cancer_data["gene_csv"],
        patient_labels_csv=cancer_data["label_csv"]
    )

    all_ids    = patch_dataset.data_frame['TCGA_ID'].unique()
    gene_dim   = gene_dataset.gene_df.shape[1]          # compute once, reuse each fold
    kf         = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        print(f"\n---- Fold {fold + 1}/5 ----")

        train_ids = all_ids[train_idx]
        val_ids   = all_ids[val_idx]

        train_dataset = PatchDataset(
            csv_file=cancer_data["patch_csv"],
            image_dir=cancer_data["patch_dir"],
            subset_ids=train_ids
        )
        val_dataset = PatchDataset(
            csv_file=cancer_data["patch_csv"],
            image_dir=cancer_data["patch_dir"],
            subset_ids=val_ids
        )

        # pin_memory speeds up CPU->GPU transfers when using CUDA
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True,
            num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False,
            num_workers=num_workers, pin_memory=pin, persistent_workers=(num_workers > 0)
        )

        # ----------------------------------------------------------------
        # Pre-build gene cache for this fold — eliminates pandas .loc[]
        # calls inside the hot loop (the main speedup).
        # ----------------------------------------------------------------
        all_fold_ids = np.concatenate([train_ids, val_ids])
        gene_cache   = _build_gene_tensor_cache(gene_dataset, all_fold_ids, device)
        print(f"  Gene cache: {len(gene_cache)} patients matched")

        # Reinitialise model weights every fold
        cnn_branch   = CNNBranch(feature_dim=64).to(device)
        mlp_branch   = MLPBranch(input_dim=gene_dim, feature_dim=64).to(device)
        fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128).to(device)

        optimizer = optim.Adam(
            list(cnn_branch.parameters()) +
            list(mlp_branch.parameters()) +
            list(fusion_layer.parameters()),
            lr=5e-5
        )
        cox_loss = CustomCoxLoss()

        num_epochs      = 30
        best_val_cindex = 0.0

        for epoch in range(num_epochs):
            train_cindex, _ = _run_epoch(
                train_loader, cnn_branch, mlp_branch, fusion_layer,
                gene_cache, device, optimizer, cox_loss,
                is_train=True,
                desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} [train]"
            )
            val_cindex, _ = _run_epoch(
                val_loader, cnn_branch, mlp_branch, fusion_layer,
                gene_cache, device, optimizer, cox_loss,
                is_train=False,
                desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs} [val]"
            )

            print(
                f"Fold {fold+1} Epoch {epoch+1}: "
                f"Train C-index={train_cindex:.4f}, "
                f"Val C-index={val_cindex:.4f}"
            )

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex

        fold_results.append(best_val_cindex)
        print(f"✅ Fold {fold+1} Best Val C-index: {best_val_cindex:.4f}")

    print("\n==== Final 5-Fold Results ====")
    print(f"Fold Scores: {fold_results}")
    print(f"Mean C-index: {sum(fold_results) / len(fold_results):.4f}")
