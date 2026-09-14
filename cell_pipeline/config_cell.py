# cell_pipeline/config_cell.py
#
# Configuration for the Spatiopath + cell-graph survival pipeline.
#
# Pipeline flow:
#   Step 1: segment_cells.py    — HoVer-Net detects cells, saves JSON per section
#   Step 2: crop_cells.py       — crops 64x64 px patch around each cell centroid
#   Step 3: embed_cells.py      — Spatiopath (or UNI) embeds each crop -> .pt
#   Step 4: build_graphs.py     — kNN spatial graph per patient -> .pt
#   train_cell.py / main_cell.py — GAT GNN trained with Cox loss

CELL_CONFIG = {
    # ── Label files (shared with histo pipeline) ─────────────────────────────
    "histo_label_xlsx": "data/Picasso/histo_new/PICASSO_dataframe.xlsx",
    "tte_label_xlsx":   "data/Picasso/histo_new/PICASSO_outcome_tte.xlsx",

    # ── Input: raw section images ─────────────────────────────────────────────
    "section_dir": "data/Picasso/histo_new/sections/",

    # ── Pipeline intermediate outputs ─────────────────────────────────────────
    # Step 1: one JSON per section image (cell centroids + types from HoVer-Net)
    "cell_mask_dir": "data/Picasso/cell/cell_masks/",

    # Step 2: one PNG per detected cell (64x64 crop around centroid)
    "cell_patch_dir": "data/Picasso/cell/cell_patches/",

    # Step 3: one .pt per cell crop (Spatiopath/UNI embedding)
    "cell_emb_dir": "data/Picasso/cell/cell_embeddings/",

    # Step 4: one .pt per patient (PyG Data object: graph with node features)
    "cell_graph_dir": "data/Picasso/cell/cell_graphs/",

    # ── Cell cropping parameters ──────────────────────────────────────────────
    "cell_crop_size": 64,    # pixels; crop is (crop_size x crop_size) around centroid
    "min_cells":      10,    # patients with fewer cells are skipped

    # ── Embedding dimension ───────────────────────────────────────────────────
    # UNI  (ViT-L/16)  -> 1024
    # Spatiopath       -> update after running: python3 -c "... print(out.shape)"
    "cell_emb_dim": 1024,

    # ── Graph construction ────────────────────────────────────────────────────
    "k_neighbors": 6,        # kNN edges per cell in spatial graph

    # ── GNN model ────────────────────────────────────────────────────────────
    "gnn_hidden": 256,       # GAT hidden dim per head
    "gnn_heads":  4,         # attention heads in each GAT layer
    "gnn_layers": 2,         # number of GAT layers
    "out_dim":    64,        # patient-level representation dim
    "fusion_dim": 128,       # survival head hidden dim

    # ── Training ─────────────────────────────────────────────────────────────
    "batch_size": 2,         # graphs per batch (cells count is high, keep small)
    "lr":         1e-4,
    "num_epochs": 50,
    "patience":   15,

    # ── Output ───────────────────────────────────────────────────────────────
    "checkpoint_dir": "Best_Model_Cell/",
}
