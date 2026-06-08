# config.py

CANCER_CONFIGS = {
    "GBMLGG": {
        "patch_csv":  "data/TCGA_GBMLGG/patches_with_labels.csv",
        "patch_dir":  "data/TCGA_GBMLGG/patches/",
        "gene_csv":   "data/TCGA_GBMLGG/gene data/clean_gene_expression.csv",
        "label_csv":  "data/TCGA_GBMLGG/merged_all_dataset_and_grade_data.csv",

        # UNI-1
        "use_foundation":  True,
        "emb_dir":         "data/TCGA_GBMLGG/uni_embeddings/",
        "foundation_dim":  1024,
        "k_patches":       32,

        # GeneFormer
        "use_geneformer":        False,
        "geneformer_token_map":  "data/TCGA_GBMLGG/geneformer/gene_token_map.json",
        "geneformer_max_len":    2048,
        "geneformer_freeze":     False,

        # Required by run_fold
        "model_type": "new",
    },

    # ── Add more cancers below if needed ───────────────────────────
    # "BRCA": {
    #     "patch_csv":   "data/TCGA_BRCA/patches.csv",
    #     ...
    # },
}