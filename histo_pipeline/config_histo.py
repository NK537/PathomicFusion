# histo_pipeline/config_histo.py
#
# Configuration for the Picasso histology survival pipeline.
#
# Label file:
#   PICASSO_dataframe.xlsx  ->  columns: ID (01-01), WSI (02_001 Rectum 2F),
#                               PHRI, RHI, NHI, outcome (0/1)
#   WSI name = exact prefix of patch filenames:
#       "02_001 Rectum 2F_0.png", "02_001 Rectum 2F_1.png", ...
#
# TTE file (optional):
#   PicassoOnly_Outcome_train.xlsx  -> columns: code (0101), days_to_outcome, ANY OUTCOME
#   If missing, training falls back to BCE loss.

HISTO_CONFIG = {
    # Primary label file — new unified dataframe
    "histo_label_xlsx": "data/Picasso/histo/PICASSO_dataframe.xlsx",

    # Time-to-event file (same one used by endoscopy pipeline).
    # code column: "0101" matches ID "01-01" with dash stripped.
    # If None or file missing -> falls back to BCE loss.
    "tte_label_xlsx":   "data/Picasso/PicassoOnly_Outcome_train.xlsx",

    # Raw patch images (only needed for precompute step).
    # Copy from: D:\Data\PICASSO_Histology\...\patch_neutrophils\images\
    # to:        data/Picasso/histo/patch_images/
    "histo_patch_dir":  "data/Picasso/histo/patch_images/",
    "histo_mask_dir":   "data/Picasso/histo/patch_masks/",   # optional

    # Pre-computed patch embeddings (populated by precompute_histo_embeddings.py)
    # One .pt per patch, named identically to the patch file stem.
    "histo_emb_dir":    "data/Picasso/histo/histo_embeddings/",

    # Foundation model output dimension: UNI (ViT-L/16) -> 1024
    "histo_dim": 1024,

    # MIL: patches randomly sampled per patient per epoch
    "k_patches": 64,

    # Model dimensions
    "out_dim":    64,
    "fusion_dim": 128,

    # Training
    "batch_size": 4,
    "lr":         5e-5,
    "num_epochs": 30,
    "patience":   10,

    # Output
    "checkpoint_dir": "Best_Model_Histo/",
}
