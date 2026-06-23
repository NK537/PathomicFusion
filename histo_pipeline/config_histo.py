# histo_pipeline/config_histo.py
#
# Configuration for the Picasso histology survival pipeline.
#
# Label files:
#   PicassoAI_DataList_Fusion.xlsx  -> patient list, binary Outcome, train/test split
#   PICASSO_outcome_tte.xlsx        -> time-to-event  (put in same folder when available)
#
# Patch images:
#   Filename format:  {site}_{patient:03d} {Section} {slide}_{row}_{col}.png
#   Example:          02_003 Sigmoid 1E_16_146.png  -> patient 02-03
#   Folder:           set histo_patch_dir to the WSL path on your machine
#
# Embeddings:
#   Run precompute_histo_embeddings.py once to populate histo_emb_dir.
#   One .pt file per patch, shape: (histo_dim,)

HISTO_CONFIG = {
    # Label files
    # Primary: patient IDs + binary outcome + train/test split
    "fusion_label_xlsx": "data/Picasso/histo/PicassoAI_DataList_Fusion.xlsx",

    # Time-to-event file. If None or file missing -> falls back to BCE loss.
    # Place PICASSO_outcome_tte.xlsx in the same histo folder when available.
    "tte_label_xlsx":    "data/Picasso/histo/PICASSO_outcome_tte.xlsx",

    # Raw patch images (only needed for precompute step).
    # WSL path: /mnt/d/Data/PICASSO_Histology/PicassoHistologyOLD/Histo_Neutriphils/patch_neutrophils/images/
    # Or copy to: data/Picasso/histo/patch_images/
    "histo_patch_dir":  "data/Picasso/histo/patch_neutrophils/images/",
    "histo_mask_dir":   "data/Picasso/histo/patch_neutrophils/masks/",   # optional, not used in training

    # Pre-computed patch embeddings (populated by precompute_histo_embeddings.py)
    # One .pt per patch, named identically to patch filename (stem only)
    "histo_emb_dir":    "data/Picasso/histo/histo_embeddings/",

    # Foundation model output dimension: UNI (ViT-L/16) -> 1024
    "histo_dim": 1024,

    # Train/test split column in PicassoAI_DataList_Fusion.xlsx
    # Values: 0=train/censored, 1=train/event, 2=test, -1=excluded
    "split_col":   "Train_Outcome_rev2",
    "train_vals":  [0, 1],
    "test_vals":   [2],

    # MIL: patches randomly sampled per patient per epoch
    # With 1310+ patches per section, 64 gives good MIL coverage
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
