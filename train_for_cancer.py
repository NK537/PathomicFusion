import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.CNN import CNNBranch
from models.MLP import MLPBranch
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from fusion.attention_fusion import AttentionFusion
from COX.cox_loss import *

from sklearn.model_selection import KFold

def train_for_cancer(cancer_name, cancer_data):
    print(f"\n==== Training for {cancer_name} (5-Fold CV) ====")

    patch_dataset = PatchDataset(csv_file=cancer_data["patch_csv"], image_dir=cancer_data["patch_dir"])
    gene_dataset = GeneDataset(
        gene_expression_csv=cancer_data["gene_csv"],
        patient_labels_csv=cancer_data["label_csv"]
    )

    all_ids = patch_dataset.data_frame['TCGA_ID'].unique()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
        print(f"\n---- Fold {fold+1}/5 ----")

        train_ids = all_ids[train_idx]
        val_ids = all_ids[val_idx]

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

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # 🔁 Reinitialize model EVERY fold
        cnn_branch = CNNBranch(feature_dim=64)
        mlp_branch = MLPBranch(input_dim=gene_dataset.gene_df.shape[1], feature_dim=64)
        fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_branch.to(device)
        mlp_branch.to(device)
        fusion_layer.to(device)

        optimizer = optim.Adam(
            list(cnn_branch.parameters()) +
            list(mlp_branch.parameters()) +
            list(fusion_layer.parameters()),
            lr=5e-5
        )

        cox_loss = CustomCoxLoss()

        num_epochs = 30
        best_val_cindex = 0

        for epoch in range(num_epochs):
            cnn_branch.train()
            mlp_branch.train()
            fusion_layer.train()

            running_loss = 0.0
            all_scores, all_times, all_events = [], [], []

            for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{num_epochs}"):
                patches, surv_times, events, grades, patient_ids = batch

                valid_patches, valid_gene_vectors = [], []
                valid_surv_times, valid_events = [], []

                for i, pid in enumerate(patient_ids):
                    if pid in gene_dataset.gene_df.index:
                        gene_vector = torch.tensor(
                            gene_dataset.gene_df.loc[pid].values,
                            dtype=torch.float32
                        )
                        valid_gene_vectors.append(gene_vector)
                        valid_patches.append(patches[i])
                        valid_surv_times.append(surv_times[i])
                        valid_events.append(events[i])

                if len(valid_patches) == 0:
                    continue

                patches = torch.stack(valid_patches).to(device)
                gene_vectors = torch.stack(valid_gene_vectors).to(device)
                surv_times = torch.stack(valid_surv_times).to(device)
                events = torch.stack(valid_events).to(device)

                cnn_feats = cnn_branch(patches)
                mlp_feats = mlp_branch(gene_vectors)
                survival_scores = fusion_layer(cnn_feats, mlp_feats)

                loss = cox_loss(survival_scores, surv_times, events)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not torch.isnan(loss):
                    running_loss += loss.item()
                    all_scores.extend(survival_scores.detach())
                    all_times.extend(surv_times.detach())
                    all_events.extend(events.detach())

            # Train C-index
            train_cindex = concordance_index(
                -torch.stack(all_scores),
                torch.stack(all_times),
                torch.stack(all_events)
            ) if len(all_scores) > 0 else 0

            # Validation
            cnn_branch.eval()
            mlp_branch.eval()
            fusion_layer.eval()

            val_scores, val_times, val_events = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    patches, surv_times, events, grades, patient_ids = batch

                    valid_patches, valid_gene_vectors = [], []
                    valid_surv_times, valid_events = [], []

                    for i, pid in enumerate(patient_ids):
                        if pid in gene_dataset.gene_df.index:
                            gene_vector = torch.tensor(
                                gene_dataset.gene_df.loc[pid].values,
                                dtype=torch.float32
                            )
                            valid_gene_vectors.append(gene_vector)
                            valid_patches.append(patches[i])
                            valid_surv_times.append(surv_times[i])
                            valid_events.append(events[i])

                    if len(valid_patches) == 0:
                        continue

                    patches = torch.stack(valid_patches).to(device)
                    gene_vectors = torch.stack(valid_gene_vectors).to(device)
                    surv_times = torch.stack(valid_surv_times).to(device)
                    events = torch.stack(valid_events).to(device)

                    cnn_feats = cnn_branch(patches)
                    mlp_feats = mlp_branch(gene_vectors)
                    survival_scores = fusion_layer(cnn_feats, mlp_feats)

                    val_scores.extend(survival_scores)
                    val_times.extend(surv_times)
                    val_events.extend(events)

            val_cindex = concordance_index(
                -torch.stack(val_scores),
                torch.stack(val_times),
                torch.stack(val_events)
            )

            print(f"Fold {fold+1} Epoch {epoch+1}: Train C-index={train_cindex:.4f}, Val C-index={val_cindex:.4f}")

            if val_cindex > best_val_cindex:
                best_val_cindex = val_cindex

        fold_results.append(best_val_cindex)
        print(f"✅ Fold {fold+1} Best Val C-index: {best_val_cindex:.4f}")

    print("\n==== Final 5-Fold Results ====")
    print(f"Fold Scores: {fold_results}")
    print(f"Mean C-index: {sum(fold_results)/len(fold_results):.4f}")
