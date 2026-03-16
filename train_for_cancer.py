import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.CNN import CNNBranch
from models.MLP import MLPBranch
from models.foundation_branch import FoundationBranch
from pytorch_dataset_loader.patches_embedding_dataset import PatchEmbeddingDataset
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from fusion.attention_fusion import AttentionFusion
from COX.cox_loss import *
from lifelines.utils import concordance_index

from pytorch_dataset_loader.patient_mil_dataset import PatientMILDataset
from models.uni_mil_branch import UNIMILBranch


def train_for_cancer(cancer_name, cancer_data, train_ids=None, val_ids=None, fold_idx=None):
    print(f"\n==== Training for {cancer_name} ====")

    # Load datasets
    # Load datasets
    gene_dataset = GeneDataset(
        gene_expression_csv=cancer_data["gene_csv"],
        patient_labels_csv=cancer_data["label_csv"]
    )

    use_foundation = cancer_data.get("use_foundation", False)
    model_type = cancer_data.get("model_type", "old")  # "old" or "new"


    if use_foundation:
        patch_dataset = PatchEmbeddingDataset(
            csv_file=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"]
        )
    else:
        patch_dataset = PatchDataset(
            csv_file=cancer_data["patch_csv"],
            image_dir=cancer_data["patch_dir"]
        )

    # Split patients
    # Split patients (external fold split preferred)
    if train_ids is None or val_ids is None:
        all_ids = patch_dataset.data_frame['TCGA_ID'].unique()
        train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    k_patches = cancer_data.get("k_patches", 32)
    
    if use_foundation:
        train_dataset = PatientMILDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            k_patches=k_patches,
            subset_ids=train_ids
        )
        val_dataset = PatientMILDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            k_patches=k_patches,
            subset_ids=val_ids
        )
    else:
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

    bs = 4 if use_foundation else 16
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        

    # Model initialization
    if use_foundation:
        cnn_branch = UNIMILBranch(uni_dim=cancer_data["foundation_dim"], out_dim=64)
    else:
        cnn_branch = CNNBranch(feature_dim=64)

    # These MUST be created for BOTH modes
    mlp_branch = MLPBranch(input_dim=gene_dataset.gene_df.shape[1], feature_dim=64)

    if model_type == "old":
        fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128)
    elif model_type == "new":
        from fusion.cross_attention_fusion import CrossAttentionFusion
        fusion_layer = CrossAttentionFusion(d_model=64, n_heads=4, n_gene_tokens=8, fusion_dim=128)
    else:
        raise ValueError("model_type must be 'old' or 'new'")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_branch.to(device)
    mlp_branch.to(device)
    fusion_layer.to(device)

    # Optimizer + Cox loss
    optimizer = optim.Adam(list(cnn_branch.parameters()) + list(mlp_branch.parameters()) + list(fusion_layer.parameters()), lr=5e-5)  # Reduced LR
    cox_loss = CustomCoxLoss()

    # Early stopping params
    num_epochs = 30
    patience = 10
    best_val_cindex = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        cnn_branch.train()
        mlp_branch.train()
        fusion_layer.train()

        running_loss = 0.0
        all_scores, all_times, all_events = [], [], []

        for batch in tqdm(train_loader, desc=f"{cancer_name} Epoch {epoch+1}/{num_epochs}"):

            if use_foundation:
                # MIL patient-level batch
                uni_embs, gene_vectors, surv_times, events, grades, patient_ids = batch

                uni_embs = uni_embs.to(device)           # (B, K, 1024)
                gene_vectors = gene_vectors.to(device)   # (B, P)
                surv_times = surv_times.to(device)
                events = events.to(device)

                cnn_feats = cnn_branch(uni_embs)         # (B, 64)
                mlp_feats = mlp_branch(gene_vectors)     # (B, 64)
                survival_scores = fusion_layer(cnn_feats, mlp_feats)  # (B,)

            else:
                # Your original patch-level batch
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

        # Train C-Index calculation
        if len(all_scores) > 0:
           train_cindex = concordance_index(
                torch.stack(all_times).cpu().numpy(),
                (-torch.stack(all_scores)).cpu().numpy(),
                torch.stack(all_events).cpu().numpy()
            )

        else:
            train_cindex = 0.0

        # Validation step
        cnn_branch.eval()
        mlp_branch.eval()
        fusion_layer.eval()

        val_scores, val_times, val_events = [], [], []
        with torch.no_grad():
            for batch in val_loader:

                if use_foundation:
                    uni_embs, gene_vectors, surv_times, events, grades, patient_ids = batch

                    uni_embs = uni_embs.to(device)
                    gene_vectors = gene_vectors.to(device)
                    surv_times = surv_times.to(device)
                    events = events.to(device)

                    cnn_feats = cnn_branch(uni_embs)
                    mlp_feats = mlp_branch(gene_vectors)
                    survival_scores = fusion_layer(cnn_feats, mlp_feats)

                else:
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

        if len(val_scores) >0:
            val_cindex = concordance_index(
                torch.stack(val_times).cpu().numpy(),
                (-torch.stack(val_scores)).cpu().numpy(),
                torch.stack(val_events).cpu().numpy()
            )
        else:
            val_cindex = 0.0

        # print("Sample scores vs times:")
        # for i in range(min(5, len(all_scores))):
        #     print(f"Score: {all_scores[i].item():.4f} | Time: {all_times[i].item():.2f} | Event: {all_events[i].item():.0f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {running_loss:.4f} | Train C-index: {train_cindex:.4f} | Val C-index: {val_cindex:.4f}")

        # Early stopping check
        if val_cindex > best_val_cindex + 0.001:  # Slight margin to avoid micro fluctuations
            best_val_cindex = val_cindex
            suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
            torch.save({
                'cnn': cnn_branch.state_dict(),
                'mlp': mlp_branch.state_dict(),
                'fusion': fusion_layer.state_dict()
            }, f"best_model_{cancer_name}{suffix}_{model_type}.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # if epochs_no_improve >= patience:
        #     print("=� Early stopping triggered.")
        #     break
    return best_val_cindex

