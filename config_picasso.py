# config_picasso.py
#
# Central config for the Picasso endoscopy survival pipeline.
#
# Label file columns (PicassoOnly_Outcome_train.xlsx):
#   code              -> patient ID, e.g. "0101"  (maps to pat0101 in filenames)
#   days_to_outcome   -> days from procedure to event  (-1 = censored / unknown)
#   ANY OUTCOME       -> 1 = event observed, 0 = censored
#   date_of_procedure -> used to compute censoring time when days_to_outcome = -1
#   date_of_visit     -> used to compute censoring time when days_to_outcome = -1

PICASSO_CONFIG = {
    # ── Label file ────────────────────────────────────────────────────────────
    "label_xlsx": "data/Picasso/PicassoOnly_Outcome_train.xlsx",

    # ── Pre-computed endoscopy embedding directory ────────────────────────────
    # Source: Picasso_WL_Train_fullframe_RN50_GastroNet5M
    # File pattern: RN50_GastroNet5M_DINOv1_feat_WLE_PicassoTrain_pat{code}_section{1|2}
    "endo_emb_dir": "data/Picasso/Picasso_WL_Train_fullframe_RN50_GastroNet5M/",
    "endo_emb_prefix": "RN50_GastroNet5M_DINOv1_feat_WLE_PicassoTrain",

    # ── Embedding dim ─────────────────────────────────────────────────────────
    # ResNet50 output = 2048.  Update here if you switch to ViT-b/16 EndoFM.
    "endo_dim": 2048,

    # ── Shared branch output dim ──────────────────────────────────────────────
    "out_dim": 64,

    # ── Histopathology (not available yet — branch is frozen / skipped) ───────
    # Set freeze_histo=True to use histo as fixed features once embeddings exist.
    # Set freeze_histo=False to train histo branch end-to-end.
    "freeze_histo": True,
    "histo_emb_dir": None,       # fill in when histo embeddings are ready
    "histo_dim":     1024,       # UNI / Spatiopath

    # ── Fusion (used when both modalities are active) ─────────────────────────
    "fusion_type": "cross_attention",
    "n_heads":     4,
    "n_tokens":    8,
    "fusion_dim":  128,

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size": 4,
    "lr":         5e-5,
    "num_epochs": 30,
    "patience":   10,

    # ── Output ────────────────────────────────────────────────────────────────
    "checkpoint_dir": "Best_Model_Picasso/",
}
