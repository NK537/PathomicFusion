
import torch
torch.set_float32_matmul_precision("high")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from lifelines.utils import concordance_index

from models.CNN import CNNBranch
from models.MLP import MLPBranch
from pytorch_dataset_loader.patches_embedding_dataset import PatchEmbeddingDataset
from pytorch_dataset_loader.patches_pytorch_dataset import PatchDataset
from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
from fusion.attention_fusion import AttentionFusion
from COX.cox_loss import CustomCoxLoss

from pytorch_dataset_loader.patient_mil_dataset import PatientMILDataset
from models.uni_mil_branch import UNIMILBranch

# GeneFormer
from pytorch_dataset_loader.patient_mil_geneformer_dataset import PatientMILGeneformerDataset
from pytorch_dataset_loader.geneformer_collate import geneformer_collate_fn
from models.geneformer_branch import GeneformerBranch
from geneformer_utils.gene_tokenizer import BulkGeneformerTokenizer, load_token_map


def train_for_cancer(cancer_name, cancer_data, train_ids=None, val_ids=None, fold_idx=None):
    print(f"\n==== Training for {cancer_name} ====")

    # ------------------------------
    # Load gene dataset
    # ------------------------------
    gene_dataset = GeneDataset(
        gene_expression_csv=cancer_data["gene_csv"],
        patient_labels_csv=cancer_data["label_csv"]
    )

    use_foundation = cancer_data.get("use_foundation", False)
    model_type = cancer_data.get("model_type", "old")   # "old" or "new"
    use_geneformer = cancer_data.get("use_geneformer", False)

    # ------------------------------
    # Patch dataset (used mainly for split metadata when foundation=True)
    # ------------------------------
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

    # ------------------------------
    # Split patients
    # ------------------------------
    if train_ids is None or val_ids is None:
        all_ids = patch_dataset.data_frame["TCGA_ID"].unique()
        train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

    # ------------------------------
    # GeneFormer tokenizer (if enabled)
    # ------------------------------
    if use_geneformer:
        token_map = load_token_map(cancer_data["geneformer_token_map"])
        gene_tokenizer = BulkGeneformerTokenizer(
            gene_names=list(gene_dataset.gene_df.columns),
            token_map=token_map,
            max_len=cancer_data.get("geneformer_max_len", 2048),
        )
    else:
        gene_tokenizer = None

    # ------------------------------
    # Datasets
    # ------------------------------
    k_patches = cancer_data.get("k_patches", 32)

    if use_foundation and use_geneformer:
        train_dataset = PatientMILGeneformerDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            gene_tokenizer=gene_tokenizer,
            k_patches=k_patches,
            subset_ids=train_ids,
            token_cache_path=(
                f"data/TCGA_GBMLGG/geneformer/train_tokens_fold{fold_idx}.pt"
                if fold_idx is not None
                else "data/TCGA_GBMLGG/geneformer/train_tokens.pt"
            ),
        )
        val_dataset = PatientMILGeneformerDataset(
            patch_csv=cancer_data["patch_csv"],
            emb_dir=cancer_data["emb_dir"],
            gene_df=gene_dataset.gene_df,
            gene_tokenizer=gene_tokenizer,
            k_patches=k_patches,
            subset_ids=val_ids,
            token_cache_path=(
                f"data/TCGA_GBMLGG/geneformer/val_tokens_fold{fold_idx}.pt"
                if fold_idx is not None
                else "data/TCGA_GBMLGG/geneformer/val_tokens.pt"
            ),
        )
        train_cache_path = (
            f"data/TCGA_GBMLGG/geneformer/train_tokens_fold{fold_idx}.pt"
            if fold_idx is not None
            else "data/TCGA_GBMLGG/geneformer/train_tokens.pt"
        )

        val_cache_path = (
            f"data/TCGA_GBMLGG/geneformer/val_tokens_fold{fold_idx}.pt"
            if fold_idx is not None
            else "data/TCGA_GBMLGG/geneformer/val_tokens.pt"
        )

        print("Train cache path:", train_cache_path)
        print("Val cache path:", val_cache_path)   
        
    elif use_foundation:
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

    # ------------------------------
    # DataLoaders
    # ------------------------------
    bs = 4 if use_foundation else 16

    if use_foundation and use_geneformer:
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            collate_fn=geneformer_collate_fn,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            collate_fn=geneformer_collate_fn
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # ------------------------------
    # Model initialization
    # ------------------------------
    if use_foundation:
        cnn_branch = UNIMILBranch(
            uni_dim=cancer_data["foundation_dim"],
            out_dim=64
        )
    else:
        cnn_branch = CNNBranch(feature_dim=64)

    if use_geneformer:
        mlp_branch = GeneformerBranch(
            out_dim=64,
            freeze_backbone=cancer_data.get("geneformer_freeze", True),
        )
    else:
        mlp_branch = MLPBranch(
            input_dim=gene_dataset.gene_df.shape[1],
            feature_dim=64
        )

    if model_type == "old":
        fusion_layer = AttentionFusion(input_dim=64, fusion_dim=128)
    elif model_type == "new":
        from fusion.cross_attention_fusion import CrossAttentionFusion
        fusion_layer = CrossAttentionFusion(
            d_model=64,
            n_heads=4,
            n_gene_tokens=8,
            fusion_dim=128
        )
    else:
        raise ValueError("model_type must be 'old' or 'new'")

    # ------------------------------
    # Device
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_branch.to(device)
    mlp_branch.to(device)
    fusion_layer.to(device)

    # ------------------------------
    # Optimizer + loss
    # ------------------------------
    optimizer = optim.Adam(
        list(cnn_branch.parameters()) +
        list(mlp_branch.parameters()) +
        list(fusion_layer.parameters()),
        lr=5e-5
    )
    cox_loss = CustomCoxLoss()

    # ------------------------------
    # Training params
    # ------------------------------
    num_epochs = 30
    patience = 10
    best_val_cindex = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        cnn_branch.train()
        mlp_branch.train()
        fusion_layer.train()

        running_loss = 0.0
        all_scores, all_times, all_events = [], [], []

        # ==============================
        # TRAIN
        # ==============================
        for batch in tqdm(train_loader, desc=f"{cancer_name} Epoch {epoch+1}/{num_epochs}"):

            if use_foundation and use_geneformer:
                uni_embs, input_ids, attention_mask, surv_times, events, grades, patient_ids = batch

                uni_embs = uni_embs.to(device)              # (B, K, 1024)
                input_ids = input_ids.to(device)            # (B, L)
                attention_mask = attention_mask.to(device)  # (B, L)
                surv_times = surv_times.to(device)
                events = events.to(device)

                cnn_feats = cnn_branch(uni_embs)            # (B, 64)
                mlp_feats = mlp_branch(input_ids, attention_mask)  # (B, 64)
                survival_scores = fusion_layer(cnn_feats, mlp_feats)

            elif use_foundation:
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

                if use_geneformer:
                    # Fallback path if someone uses GeneFormer without foundation
                    tokenized = [gene_tokenizer.encode_vector(g.cpu().numpy()) for g in gene_vectors]
                    from torch.nn.utils.rnn import pad_sequence
                    input_ids = pad_sequence(
                        [x[0] for x in tokenized],
                        batch_first=True,
                        padding_value=0
                    ).to(device)
                    attention_mask = pad_sequence(
                        [x[1] for x in tokenized],
                        batch_first=True,
                        padding_value=0
                    ).to(device)
                    mlp_feats = mlp_branch(input_ids, attention_mask)
                else:
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

        # ------------------------------
        # Train C-index
        # ------------------------------
        if len(all_scores) > 0:
            train_cindex = concordance_index(
                torch.stack(all_times).cpu().numpy(),
                (-torch.stack(all_scores)).cpu().numpy(),
                torch.stack(all_events).cpu().numpy()
            )
        else:
            train_cindex = 0.0

        # ==============================
        # VALIDATION
        # ==============================
        cnn_branch.eval()
        mlp_branch.eval()
        fusion_layer.eval()

        val_scores, val_times, val_events = [], [], []

        with torch.no_grad():
            for batch in val_loader:

                if use_foundation and use_geneformer:
                    uni_embs, input_ids, attention_mask, surv_times, events, grades, patient_ids = batch

                    uni_embs = uni_embs.to(device)
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    surv_times = surv_times.to(device)
                    events = events.to(device)

                    cnn_feats = cnn_branch(uni_embs)
                    mlp_feats = mlp_branch(input_ids, attention_mask)
                    survival_scores = fusion_layer(cnn_feats, mlp_feats)

                elif use_foundation:
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

                    if use_geneformer:
                        from torch.nn.utils.rnn import pad_sequence
                        tokenized = [gene_tokenizer.encode_vector(g.cpu().numpy()) for g in gene_vectors]
                        input_ids = pad_sequence(
                            [x[0] for x in tokenized],
                            batch_first=True,
                            padding_value=0
                        ).to(device)
                        attention_mask = pad_sequence(
                            [x[1] for x in tokenized],
                            batch_first=True,
                            padding_value=0
                        ).to(device)
                        mlp_feats = mlp_branch(input_ids, attention_mask)
                    else:
                        mlp_feats = mlp_branch(gene_vectors)

                    survival_scores = fusion_layer(cnn_feats, mlp_feats)

                val_scores.extend(survival_scores)
                val_times.extend(surv_times)
                val_events.extend(events)

        # ------------------------------
        # Validation C-index
        # ------------------------------
        if len(val_scores) > 0:
            val_cindex = concordance_index(
                torch.stack(val_times).cpu().numpy(),
                (-torch.stack(val_scores)).cpu().numpy(),
                torch.stack(val_events).cpu().numpy()
            )
        else:
            val_cindex = 0.0

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {running_loss:.4f} | "
            f"Train C-index: {train_cindex:.4f} | "
            f"Val C-index: {val_cindex:.4f}"
        )

        # ------------------------------
        # Early stopping
        # ------------------------------
        if val_cindex > best_val_cindex + 0.001:
            best_val_cindex = val_cindex
            suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
            gene_tag = "_geneformer" if use_geneformer else "_mlp"

            torch.save(
                {
                    "cnn": cnn_branch.state_dict(),
                    "mlp": mlp_branch.state_dict(),
                    "fusion": fusion_layer.state_dict()
                },
                f"Best_Model/best_model_{cancer_name}{suffix}_{model_type}{gene_tag}.pth"
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Uncomment if you want strict early stopping
        # if epochs_no_improve >= patience:
        #     print("🚑 Early stopping triggered.")
        #     break

    return best_val_cindex