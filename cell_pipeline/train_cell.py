# cell_pipeline/train_cell.py
#
# Single-fold training loop for cell-graph survival prediction.
# Uses GAT GNN + Cox loss (or BCE if TTE unavailable).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from models.cell_gnn_branch import CellGNNBranch
from cell_pipeline.cell_graph_dataset import CellGraphDataset, cell_graph_collate
from COX.cox_loss import CustomCoxLoss


class _SurvivalHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,)


def train_cell(cfg: dict, train_ids: list, val_ids: list, fold_idx: int):
    """
    Train one fold of cell-graph pipeline.
    Returns (best_c_index, best_val_preds_df).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = CellGraphDataset(cfg["cell_graph_dir"], subset_ids=train_ids)
    val_ds   = CellGraphDataset(cfg["cell_graph_dir"], subset_ids=val_ids)

    if len(train_ds) == 0:
        print("[train_cell] WARN: empty training set — returning 0.5")
        return 0.5, None

    train_dl = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=cell_graph_collate, drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=cell_graph_collate, drop_last=False,
    )

    use_cox = train_ds.use_cox

    # ── Model ─────────────────────────────────────────────────────────────────
    gnn = CellGNNBranch(
        in_dim   = cfg["cell_emb_dim"],
        hidden   = cfg["gnn_hidden"],
        heads    = cfg["gnn_heads"],
        n_layers = cfg["gnn_layers"],
        out_dim  = cfg["out_dim"],
    ).to(device)

    head = _SurvivalHead(
        in_dim = cfg["out_dim"],
        hidden = cfg["fusion_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(gnn.parameters()) + list(head.parameters()),
        lr=cfg["lr"], weight_decay=1e-4,
    )

    cox_loss_fn = CustomCoxLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path      = os.path.join(cfg["checkpoint_dir"], f"fold{fold_idx}_best.pt")
    best_score     = -1.0
    best_val_preds = None
    patience       = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(cfg["num_epochs"]):
        gnn.train(); head.train()
        total_loss = 0.0

        for batch in train_dl:
            x          = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            b_vec      = batch["batch"].to(device)
            times      = batch["surv_time"].to(device)
            events     = batch["y"].to(device)

            feats = gnn(x, edge_index, b_vec)   # (B, out_dim)
            risk  = head(feats)                  # (B,)

            loss = cox_loss_fn(risk, times, events) if use_cox \
                   else bce_loss_fn(risk, events)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gnn.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        gnn.eval(); head.eval()
        all_risk, all_time, all_event, all_pids = [], [], [], []

        with torch.no_grad():
            for batch in val_dl:
                x          = batch["x"].to(device)
                edge_index = batch["edge_index"].to(device)
                b_vec      = batch["batch"].to(device)

                feats = gnn(x, edge_index, b_vec)
                risk  = head(feats)

                all_risk.extend(risk.cpu().tolist())
                all_time.extend(batch["surv_time"].tolist())
                all_event.extend(batch["y"].tolist())
                all_pids.extend(batch["patient_id"])

        try:
            if use_cox:
                score = concordance_index(all_time, [-r for r in all_risk], all_event)
            else:
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(all_event, all_risk)
        except Exception:
            score = 0.5

        metric = "C-idx" if use_cox else "AUC"
        print(f"  epoch {epoch+1:3d}/{cfg['num_epochs']}  "
              f"loss={total_loss/max(len(train_dl),1):.4f}  "
              f"val_{metric}={score:.4f}")

        if score > best_score:
            best_score = score
            patience   = 0
            best_val_preds = pd.DataFrame({
                "patient_id": all_pids,
                "risk_score": all_risk,
                "surv_time":  all_time,
                "event":      all_event,
            })
            torch.save({
                "gnn":   gnn.state_dict(),
                "head":  head.state_dict(),
                "epoch": epoch,
                "score": best_score,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best val {metric} = {best_score:.4f}  (saved: {ckpt_path})")

    if best_val_preds is None and all_pids:
        best_val_preds = pd.DataFrame({
            "patient_id": all_pids,
            "risk_score": all_risk,
            "surv_time":  all_time,
            "event":      all_event,
        })

    return best_score, best_val_preds
