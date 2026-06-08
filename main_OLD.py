#use this main if i want to test GeneFormer + OLD fusion
#AND
#use main_new if i want to test GeneFormer + NEW fusion

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from train_for_cancer import train_for_cancer

cancers = {
    "GBMLGG": {
        "patch_csv": "data/TCGA_GBMLGG/patches_with_labels.csv",
        "patch_dir": "data/TCGA_GBMLGG/patches/",
        "gene_csv": "data/TCGA_GBMLGG/gene data/clean_gene_expression.csv",
        "label_csv": "data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv",

        # UNI-1
        "use_foundation": True,
        "emb_dir": "data/TCGA_GBMLGG/uni_embeddings/",
        "foundation_dim": 1024,
        "k_patches": 32,
        
        # GeneFormer options
        "use_geneformer": True,
        "geneformer_token_map": "data/TCGA_GBMLGG/geneformer/gene_token_map.json",
        "geneformer_max_len": 2048,
    }
}

def run_5fold(cancer_name, cfg, model_type: str):
    df = pd.read_csv(cfg["patch_csv"])

    # Patch patients
    patch_pids = set(df["TCGA_ID"].unique())

    # Gene patients (use the same GeneDataset as training)
    from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
    gene_dataset = GeneDataset(
        gene_expression_csv=cfg["gene_csv"],
        patient_labels_csv=cfg["label_csv"]
    )
    gene_pids = set(gene_dataset.gene_df.index)

    # Intersection: only patients that have BOTH patch + gene
    patient_ids = np.array(sorted(list(patch_pids.intersection(gene_pids))))

    print(f"[{cancer_name}] Patients in patches: {len(patch_pids)}")
    print(f"[{cancer_name}] Patients in genes:   {len(gene_pids)}")
    print(f"[{cancer_name}] Patients matched:    {len(patient_ids)}")

    if len(patient_ids) < 5:
        raise ValueError("Not enough matched patients for 5-fold CV.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    print(f"\n========== {cancer_name} | {model_type.upper()} | 5-FOLD ==========")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_ids = patient_ids[train_idx]
        val_ids = patient_ids[val_idx]

        cfg_fold = dict(cfg)
        cfg_fold["model_type"] = model_type

        best_val = train_for_cancer(
            cancer_name,
            cfg_fold,
            train_ids=train_ids,
            val_ids=val_ids,
            fold_idx=fold_idx
        )

        fold_scores.append(best_val)
        print(f"[{cancer_name}] {model_type} fold {fold_idx}: best val c-index = {best_val:.4f}")

    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    print(f"\n[{cancer_name}] {model_type} 5-fold mean±std c-index: {mean:.4f} ± {std:.4f}\n")

    return fold_scores

if __name__ == "__main__":
    for cancer, cfg in cancers.items():
        old_scores = run_5fold(cancer, cfg, "old")
        new_scores = run_5fold(cancer, cfg, "new")

        print("Fold-wise comparison:")
        for i, (o, n) in enumerate(zip(old_scores, new_scores)):
            print(f"  Fold {i}: old={o:.4f} | new={n:.4f}")