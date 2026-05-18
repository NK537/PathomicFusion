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

        # GeneFormer
        "use_geneformer": True,
        "geneformer_token_map": "data/TCGA_GBMLGG/geneformer/gene_token_map.json",
        "geneformer_max_len": 2048,
        "geneformer_freeze": True,
    }
}


def run_5fold(cancer_name, cfg, model_type: str, use_geneformer: bool):
    df = pd.read_csv(cfg["patch_csv"])

    patch_pids = set(df["TCGA_ID"].unique())

    from pytorch_dataset_loader.pytorch_GeneDataset import GeneDataset
    gene_dataset = GeneDataset(
        gene_expression_csv=cfg["gene_csv"],
        patient_labels_csv=cfg["label_csv"]
    )
    gene_pids = set(gene_dataset.gene_df.index)

    patient_ids = np.array(sorted(list(patch_pids.intersection(gene_pids))))

    print(f"[{cancer_name}] Patients in patches: {len(patch_pids)}")
    print(f"[{cancer_name}] Patients in genes:   {len(gene_pids)}")
    print(f"[{cancer_name}] Patients matched:    {len(patient_ids)}")

    if len(patient_ids) < 5:
        raise ValueError("Not enough matched patients for 5-fold CV.")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    gene_name = "GENEFORMER" if use_geneformer else "MLP"
    print(f"\n========== {cancer_name} | {model_type.upper()} | {gene_name} | 5-FOLD ==========")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(patient_ids)):
        train_ids = patient_ids[train_idx]
        val_ids = patient_ids[val_idx]

        cfg_fold = dict(cfg)
        cfg_fold["model_type"] = model_type
        cfg_fold["use_geneformer"] = use_geneformer
        
        best_val = train_for_cancer(
            cancer_name,
            cfg_fold,
            train_ids=train_ids,
            val_ids=val_ids,
            fold_idx=fold_idx
        )

        fold_scores.append(best_val)
        print(f"[{cancer_name}] {model_type} | {gene_name} | fold {fold_idx}: best val c-index = {best_val:.4f}")

    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    print(f"\n[{cancer_name}] {model_type} | {gene_name} | 5-fold mean±std c-index: {mean:.4f} ± {std:.4f}\n")

    return fold_scores


if __name__ == "__main__":
    for cancer, cfg in cancers.items():
        results = {}

        # # 1. MLP + old fusion
        results["old_mlp"] = run_5fold(cancer, cfg, model_type="old", use_geneformer=False)

        # 2. MLP + new fusion
        results["new_mlp"] = run_5fold(cancer, cfg, model_type="new", use_geneformer=False)
        
        # 3. GeneFormer + old fusion
        results["old_geneformer"] = run_5fold(cancer, cfg, model_type="old", use_geneformer=True)

        # 4. GeneFormer + new fusion
        results["new_geneformer"] = run_5fold(cancer, cfg, model_type="new", use_geneformer=True)

        print("\n===== FINAL FOLD-WISE COMPARISON =====")
        for i in range(5):
            print(
                f"Fold {i}: "
                f"old_mlp={results['old_mlp'][i]:.4f} | "
                f"new_mlp={results['new_mlp'][i]:.4f} | "
                f"old_geneformer={results['old_geneformer'][i]:.4f} | "
                f"new_geneformer={results['new_geneformer'][i]:.4f}"
            )