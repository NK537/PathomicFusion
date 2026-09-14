# cell_pipeline/step4_build_graphs.py
#
# STEP 4 — Build spatial cell graphs per patient.
#
# For each patient, collects all cell embeddings (from all their WSIs/sections),
# pairs them with XY centroids from Step 1 JSON files, and constructs a
# kNN spatial graph (cells = nodes, edges = proximity).
#
# Output:
#   cell_graphs/{patient_id}.pt  — PyTorch Geometric Data object with:
#       data.x          (N_cells, emb_dim)  node features
#       data.edge_index (2, E)              kNN edges
#       data.pos        (N_cells, 2)        XY centroids (normalized)
#       data.y          scalar              event label (0/1)
#       data.surv_time  scalar              days (or 1.0 if missing)
#       data.patient_id str
#
# Usage:
#   python3 cell_pipeline/step4_build_graphs.py

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cell_pipeline.config_cell import CELL_CONFIG
from histo_pipeline.patient_histo_dataset import normalize_pat_id, load_labels


def load_patient_cells(
    patient_id:    str,
    wsi_list:      list,
    cell_mask_dir: str,
    cell_emb_dir:  str,
) -> tuple:
    """
    Collect all cell embeddings + centroids for one patient across all WSIs.

    Returns:
        embeddings : (N, D) numpy array
        centroids  : (N, 2) numpy array  [x, y]
    """
    all_embs      = []
    all_centroids = []

    for wsi in wsi_list:
        # Section stems matching this WSI: start with "{wsi}_"
        prefix = wsi + "_"

        # Find matching JSON files
        if not os.path.isdir(cell_mask_dir):
            continue
        mask_jsons = [
            f for f in os.listdir(cell_mask_dir)
            if f.endswith(".json") and os.path.splitext(f)[0].startswith(prefix)
        ]

        for jf in mask_jsons:
            stem         = os.path.splitext(jf)[0]
            section_emb  = os.path.join(cell_emb_dir, stem)

            if not os.path.isdir(section_emb):
                continue

            with open(os.path.join(cell_mask_dir, jf)) as f:
                mask_json = json.load(f)
            nuc = mask_json.get("nuc", {})

            for cell_id, cell_info in nuc.items():
                emb_path = os.path.join(section_emb, f"{stem}_{cell_id}.pt")
                if not os.path.exists(emb_path):
                    continue
                try:
                    emb = torch.load(emb_path, map_location="cpu")
                    emb = emb.float().squeeze()
                    cx, cy = cell_info["centroid"]
                    all_embs.append(emb.numpy())
                    all_centroids.append([float(cx), float(cy)])
                except Exception:
                    continue

    if not all_embs:
        return None, None

    return np.array(all_embs, dtype=np.float32), np.array(all_centroids, dtype=np.float32)


def build_graph(embeddings: np.ndarray, centroids: np.ndarray, k: int = 6):
    """
    Build kNN spatial graph from cell centroids.
    Returns edge_index (2, E) as LongTensor.
    """
    n = len(embeddings)
    k_actual = min(k, n - 1)
    if k_actual < 1:
        # Single cell — self-loop
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        return edge_index

    A = kneighbors_graph(centroids, n_neighbors=k_actual, mode="connectivity",
                         include_self=False)
    rows, cols = A.nonzero()
    # Make bidirectional
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index


def build_all_graphs(
    label_xlsx:    str,
    tte_xlsx:      str,
    cell_mask_dir: str,
    cell_emb_dir:  str,
    cell_graph_dir: str,
    k_neighbors:   int  = 6,
    min_cells:     int  = 10,
    skip_existing: bool = True,
):
    os.makedirs(cell_graph_dir, exist_ok=True)

    # Load patient labels
    label_df = load_labels(label_xlsx, tte_xlsx)
    print(f"Building graphs for {len(label_df)} patients...")

    saved = skipped = failed = 0

    for patient_id, row in tqdm(label_df.iterrows(), total=len(label_df)):
        out_path = os.path.join(cell_graph_dir, f"{patient_id}.pt")

        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        embeddings, centroids = load_patient_cells(
            patient_id    = patient_id,
            wsi_list      = row["wsi_list"],
            cell_mask_dir = cell_mask_dir,
            cell_emb_dir  = cell_emb_dir,
        )

        if embeddings is None or len(embeddings) < min_cells:
            n = 0 if embeddings is None else len(embeddings)
            print(f"  [SKIP] {patient_id}: only {n} cells (min={min_cells})")
            failed += 1
            continue

        # Normalize centroids to [0, 1] for stability
        pos = centroids.copy()
        pos -= pos.min(axis=0)
        pos_range = pos.max(axis=0)
        pos_range[pos_range == 0] = 1.0
        pos /= pos_range

        edge_index = build_graph(embeddings, pos, k=k_neighbors)
        surv_time  = row["surv_time"]
        has_tte    = (not pd.isna(surv_time)) and float(surv_time) > 0

        # Build a simple dict-based graph object (PyG-compatible)
        graph = {
            "x":          torch.tensor(embeddings, dtype=torch.float32),
            "edge_index": edge_index,
            "pos":        torch.tensor(pos, dtype=torch.float32),
            "y":          torch.tensor(float(row["event"]), dtype=torch.float32),
            "surv_time":  torch.tensor(float(surv_time) if has_tte else 1.0,
                                       dtype=torch.float32),
            "patient_id": patient_id,
        }
        torch.save(graph, out_path)
        saved += 1

    print(f"\n=== DONE ===  saved={saved}  skipped={skipped}  failed={failed}")
    print(f"Graphs saved to: {cell_graph_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    build_all_graphs(
        label_xlsx     = CELL_CONFIG["histo_label_xlsx"],
        tte_xlsx       = CELL_CONFIG["tte_label_xlsx"],
        cell_mask_dir  = CELL_CONFIG["cell_mask_dir"],
        cell_emb_dir   = CELL_CONFIG["cell_emb_dir"],
        cell_graph_dir = CELL_CONFIG["cell_graph_dir"],
        k_neighbors    = CELL_CONFIG["k_neighbors"],
        min_cells      = CELL_CONFIG["min_cells"],
        skip_existing  = not args.no_skip,
    )
