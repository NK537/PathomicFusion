# config_picasso.py
#
# Central config for the Picasso bimodal survival pipeline.
# Update the paths once you know the exact dataset layout.
#
# CSV format expected:
#   histo_patches_csv  ->  columns: patient_id, patch_filename
#   endo_frames_csv    ->  columns: patient_id, frame_filename
#   label_csv          ->  columns: patient_id, survival_months, censored
#                                   (censored=1 means event NOT observed)

PICASSO_CONFIG = {
    # ── Raw data paths ───────────────────────────────────────────────────────
    "histo_patches_csv": "data/Picasso/histo_patches.csv",
    "histo_patch_dir":   "data/Picasso/histo_patches/",

    "endo_frames_csv":   "data/Picasso/endo_frames.csv",
    "endo_frame_dir":    "data/Picasso/endo_frames/",

    "label_csv":         "data/Picasso/survival_labels.csv",

    # ── Pre-computed embedding directories ───────────────────────────────────
    # Run precompute_histo_embeddings.py and precompute_endo_embeddings.py
    # once before training.
    "histo_emb_dir": "data/Picasso/histo_embeddings/",
    "endo_emb_dir":  "data/Picasso/endo_embeddings/",

    # ── Foundation model dims (must match what was precomputed) ──────────────
    "histo_dim": 1024,   # UNI / Spatiopath  (update if Spatiopath differs)
    "endo_dim":  384,    # GastroNet-5M ViT-small/16

    # ── MIL sampling ─────────────────────────────────────────────────────────
    "k_histo": 32,       # patches sampled per patient  (histo)
    "k_endo":  16,       # frames  sampled per patient  (endo)

    # ── Shared branch output dim (both branches project to this) ─────────────
    "out_dim": 64,

    # ── Histopathology branch freeze ─────────────────────────────────────────
    # True  — HistoMILBranch weights are FROZEN during training.
    #         Histo embeddings are still loaded and used as fixed features.
    #         Only EndoMILBranch + fusion layer weights are updated.
    #         Set to False when you are ready to fine-tune the histo branch.
    "freeze_histo": True,

    # ── Fusion ───────────────────────────────────────────────────────────────
    # Options: "cross_attention" | "concat"
    "fusion_type": "cross_attention",
    "n_heads":     4,
    "n_tokens":    8,
    "fusion_dim":  128,

    # ── Training ─────────────────────────────────────────────────────────────
    "batch_size": 4,
    "lr":         5e-5,
    "num_epochs": 30,
    "patience":   10,    # early-stopping patience (epochs without improvement)

    # ── Output ───────────────────────────────────────────────────────────────
    "checkpoint_dir": "Best_Model_Picasso/",
}
