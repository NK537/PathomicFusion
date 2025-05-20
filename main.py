from matplotlib import transforms
from models.CNN import CNNBranch
from models.MLP import MLPBranch
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from fusion.attention_fusion import AttentionFusion
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from COX.cox_loss import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


     

# Initialize datasets
patch_dataset = PatchDataset(csv_file='data/TCGA_GBMLGG/patches_with_labels.csv', image_dir='data/TCGA_GBMLGG/patches/')
gene_dataset = GeneDataset(gene_expression_csv='data/TCGA_GBMLGG/gene data/clean_gene_expression.csv', patient_labels_csv='data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv')

# Train/val split
all_ids = patch_dataset.data_frame['TCGA_ID'].unique()
train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

train_dataset = PatchDataset(csv_file='data/TCGA_GBMLGG/patches_with_labels.csv', image_dir='data/TCGA_GBMLGG/patches/', subset_ids=train_ids)
val_dataset   = PatchDataset(csv_file='data/TCGA_GBMLGG/patches_with_labels.csv', image_dir='data/TCGA_GBMLGG/patches/', subset_ids=val_ids)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Models
cnn_branch = CNNBranch(feature_dim=32)
mlp_branch = MLPBranch(input_dim=gene_dataset.gene_df.shape[1], feature_dim=32)
fusion_layer = AttentionFusion(input_dim=32, fusion_dim=64)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_branch.to(device)
mlp_branch.to(device)
fusion_layer.to(device)

# Loss and optimizer
cox_loss = CustomCoxLoss()
optimizer = optim.Adam(
    list(cnn_branch.parameters()) +
    list(mlp_branch.parameters()) +
    list(fusion_layer.parameters()),
    lr=1e-4
)

# Early stopping setup
num_epochs = 50
patience = 5
best_val_cindex = 0
epochs_no_improve = 0

for epoch in range(num_epochs):
    cnn_branch.train()
    mlp_branch.train()
    fusion_layer.train()

    running_loss = 0.0
    all_scores, all_times, all_events = [], [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        patches, surv_times, events, grades, patient_ids = batch

        valid_patches, valid_gene_vectors, valid_surv_times, valid_events = [], [], [], []

        for i, pid in enumerate(patient_ids):
            if pid in gene_dataset.gene_df.index:
                gene_vector = torch.tensor(gene_dataset.gene_df.loc[pid].values, dtype=torch.float32)
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

    train_c_index = concordance_index(torch.stack(all_scores), torch.stack(all_times), torch.stack(all_events))

    # Validation
    cnn_branch.eval()
    mlp_branch.eval()
    fusion_layer.eval()

    val_scores, val_times, val_events = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            patches, surv_times, events, grades, patient_ids = batch

            valid_patches, valid_gene_vectors, valid_surv_times, valid_events = [], [], [], []

            for i, pid in enumerate(patient_ids):
                if pid in gene_dataset.gene_df.index:
                    gene_vector = torch.tensor(gene_dataset.gene_df.loc[pid].values, dtype=torch.float32)
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

    val_cindex = concordance_index(torch.stack(val_scores), torch.stack(val_times), torch.stack(val_events))
    print(f"[ Train Loss: {running_loss:.4f} | Train C-index: {train_c_index:.4f} | Val C-index: {val_cindex:.4f} ]")

    # Early stopping
    if val_cindex > best_val_cindex:
        best_val_cindex = val_cindex
        torch.save({
            'cnn': cnn_branch.state_dict(),
            'mlp': mlp_branch.state_dict(),
            'fusion': fusion_layer.state_dict()
        }, 'best_model.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("🚑 Early stopping triggered.")
        break