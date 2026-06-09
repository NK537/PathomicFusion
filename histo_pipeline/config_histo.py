# histo_pipeline/config_histo.py
#
# Configuration for the Picasso histology survival pipeline.
#
# Mirrors config_picasso.py (endoscopy) so both pipelines stay consistent.
#
# Data layout expected under data/Picasso/histo/:
#   histo_patches.csv        columns: patient_id, patch_filename
#   histo_patches/           raw patch images  (only needed for precomputing)
#   histo_embeddings/        one .pt per patch  shape: (histo_dim,)
#
# Label file is SHARED with the endoscopy pipeline:
#   data/Picasso/PicassoOnly_Outcome_train.xlsx
#   key columns: code (patient ID), days_to_outcome, ANY OUTCOME

HISTO_CONFIG = {
    # ── Shared label file (same as endoscopy) ────────────────────────────────
    "label_xlsx": "data/Picasso/PicassoOnly_Outcome_train.xlsx",

    # ── Histology patch index ─────────────────────────────────────────────────
    # CSV with columns: patient_id (matches `code` in label file), patch_filename
    "histo_patches_csv": "data/Picasso/histo/histo_patches.csv",

    # ── Raw patch images (only needed if re-running precompute_histo_embeddings.py)
    "histo_patch_dir":   "data/Picasso/histo/histo_patches/",

    # ── Pre-computed patch embeddings ────────────────────────────────────────
    # Run precompute_histo_embeddings.py once to populate this folder.
    # One .pt file per patch,  shape: (histo_dim,)
    "histo_emb_dir":     "data/Picasso/histo/histo_embeddings/",

    # ── Foundation model output dimension ────────────────────────────────────
    # UNI      → 1024
    # Spatiopath → check paper / repo (update here once confirmed)
    "histo_dim": 1024,

    # ── MIL sampling ─────────────────────────────────────────────────────────
    # Number of patches randomly sampled per patient each epoch.
    # Increase if GPU memory allows (more patches = better MIL estimates).
    "k_patches": 32,

    # ── Branch output dim ────────────────────────────────────────────────────
    "out_dim": 64,

    # ── Survival head ────────────────────────────────────────────────────────
    "fusion_dim": 128,

    # ── Training ─────────────────────────────────────────────────────────────
    "batch_size": 4,
    "lr":         5e-5,
    "num_epochs": 30,
    "patience":   10,

    # ── Output ───────────────────────────────────────────────────────────────
    "checkpoint_dir": "Best_Model_Histo/",
}
