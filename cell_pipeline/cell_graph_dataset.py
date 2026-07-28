# cell_pipeline/cell_graph_dataset.py
#
# PyTorch dataset that loads pre-built patient cell graphs from Step 4.
# Works with the standard DataLoader (no PyG required for loading,
# only for the GNN layers in training).

import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class CellGraphDataset(Dataset):
    """
    Loads patient-level cell graphs built by step4_build_graphs.py.

    Each sample is a dict:
        x          (N_cells, emb_dim)  node feature matrix
        edge_index (2, E)              graph edges
        pos        (N_cells, 2)        normalized XY centroids
        y          scalar              event label
        surv_time  scalar              days (or 1.0 if TTE missing)
        patient_id str
    """

    def __init__(
        self,
        cell_graph_dir: str,
        subset_ids:     list = None,
    ):
        self.cell_graph_dir = cell_graph_dir
        self.use_cox = False

        if not os.path.isdir(cell_graph_dir):
            raise FileNotFoundError(
                f"cell_graph_dir not found: {cell_graph_dir}\n"
                "Run step4_build_graphs.py first."
            )

        all_graphs = sorted(
            f for f in os.listdir(cell_graph_dir)
            if f.endswith(".pt")
        )

        if subset_ids is not None:
            subset_set = set(subset_ids)
            all_graphs = [f for f in all_graphs
                          if os.path.splitext(f)[0] in subset_set]

        # Validate and collect
        self.samples = []
        n_with_tte   = 0

        for fname in all_graphs:
            path  = os.path.join(cell_graph_dir, fname)
            graph = torch.load(path, map_location="cpu", weights_only=False)

            surv_time = graph["surv_time"].item()
            has_tte   = surv_time > 0 and not pd.isna(surv_time)
            if has_tte:
                n_with_tte += 1
                self.use_cox = True

            self.samples.append(path)

        if not self.use_cox:
            print("[CellGraphDataset] TTE unavailable — using BCE loss.")

        n_events = 0
        for p in self.samples:
            g = torch.load(p, map_location="cpu", weights_only=False)
            if g["y"].item() == 1:
                n_events += 1

        print(
            f"[CellGraphDataset]  n={len(self.samples)}  "
            f"events={n_events}  tte_available={n_with_tte}  "
            f"loss={'Cox' if self.use_cox else 'BCE'}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.load(self.samples[idx], map_location="cpu", weights_only=False)


def cell_graph_collate(batch):
    """
    Custom collate that batches graphs for PyTorch Geometric.
    Shifts node indices by cumulative node count for each graph in the batch.
    """
    xs, edge_indices, pos_list, ys, surv_times, pids, batch_vec = \
        [], [], [], [], [], [], []

    node_offset = 0
    for i, graph in enumerate(batch):
        n = graph["x"].shape[0]
        xs.append(graph["x"])
        edge_indices.append(graph["edge_index"] + node_offset)
        pos_list.append(graph["pos"])
        ys.append(graph["y"])
        surv_times.append(graph["surv_time"])
        pids.append(graph["patient_id"])
        batch_vec.append(torch.full((n,), i, dtype=torch.long))
        node_offset += n

    return {
        "x":          torch.cat(xs,          dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "pos":        torch.cat(pos_list,     dim=0),
        "y":          torch.stack(ys),
        "surv_time":  torch.stack(surv_times),
        "patient_id": pids,
        "batch":      torch.cat(batch_vec),
    }
