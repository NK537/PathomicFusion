# histo_pipeline/train_histo.py
#
# Single-fold training loop for Picasso histology survival prediction.
# Mirrors train_picasso.py (endoscopy).
#
# Loss:
#   If TTE available  -> CustomCoxLoss  (C-index is meaningful)
#   If TTE missing    -> BCE loss        (AUC reported instead of C-index)

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index

from models.histo_mil_branch import HistoMILBranch
from histo_pipeline.patient_histo_dataset import PatientHistoDataset
from COX.cox_loss import CustomCoxLoss


# ---------------------------------------------------------------------------
# Survival head
# ---------------------------------------------------------------------------

class _SurvivalHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Collate: pad variable-length patch bags to same K within a batch
# ---------------------------------------------------------------------------

def _pad_collate(batch):
    embs, times, events, pids = zip(*batch)
    max_k = max(e.shape[0] for e in embs)
    padded = []
    for e in embs:
        if e.shape[0] < max_k:
            pad = torch.zeros(max_k - e.shape[0], e.shape[1])
            e   = torch.cat([e, pad], dim=0)
        padded.append(e)
    return (
        torch.stack(padded),          # (B, max_k, D)
        torch.stack(times),           # (B,)
        torch.stack(events),          # (B,)
        list(pids),
    )


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train_histo(cfg: dict, train_ids: list, val_ids: list, fold_idx: int) -> float:
    """
    Train one fold.  Returns validation C-index (or AUC if no TTE).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Datasets ────────────────────────────────────────────────────────────
    ds_kwargs = dict(
        fusion_label_xlsx = cfg["fusion_label_xlsx"],
        tte_label_xlsx    = cfg.get("tte_label_xlsx"),
        histo_emb_dir     = cfg["histo_emb_dir"],
        k_patches         = cfg["k_patches"],
    )

    train_ds = PatientHistoDataset(**ds_kwargs, subset_ids=train_ids)
    val_ds   = PatientHistoDataset(**ds_kwargs, subset_ids=val_ids)

    if len(train_ds) == 0:
        print("[train_histo] WARN: empty training set — returning 0.5")
        return 0.5

    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"],
                          shuffle=True,  collate_fn=_pad_collate, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                          shuffle=False, collate_fn=_pad_collate, drop_last=False)

    use_cox = train_ds.use_cox

    # ── Model ────────────────────────────────────────────────────────────────
    branch = HistoMILBranch(
        histo_dim = cfg["histo_dim"],
        out_dim   = cfg["out_dim"],
    ).to(device)

    head = _SurvivalHead(
        in_dim = cfg["out_dim"],
        hidden = cfg["fusion_dim"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(branch.parameters()) + list(head.parameters()),
        lr=cfg["lr"], weight_decay=1e-4,
    )

    cox_loss_fn = CustomCoxLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path  = os.path.join(cfg["checkpoint_dir"], f"fold{fold_idx}_best.pt")
    best_score = -1.0
    patience   = 0

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(cfg["num_epochs"]):
        branch.train(); head.train()
        total_loss = 0.0

        for embs, times, events, _ in train_dl:
            embs   = embs.to(device)
            times  = times.to(device)
            events = events.to(device)

            feats = branch(embs)       # (B, out_dim)
            risk  = head(feats)        # (B,)

            if use_cox:
                loss = cox_loss_fn(risk, times, events)
            else:
                loss = bce_loss_fn(risk, events)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(branch.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        # ── Validation ───────────────────────────────────────────────────────
        branch.eval(); head.eval()
        all_risk, all_time, all_event = [], [], []

        with torch.no_grad():
            for embs, times, events, _ in val_dl:
                embs = embs.to(device)
                feats = branch(embs)
                risk  = head(feats)
                all_risk.extend(risk.cpu().tolist())
                all_time.extend(times.tolist())
                all_event.extend(events.tolist())

        # Metric
        try:
            if use_cox:
                score = concordance_index(all_time, [-r for r in all_risk], all_event)
            else:
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(all_event, all_risk)
        except Exception:
            score = 0.5

        metric_name = "C-idx" if use_cox else "AUC"
        avg_loss    = total_loss / max(len(train_dl), 1)
        print(f"  epoch {epoch+1:3d}/{cfg['num_epochs']}  "
              f"loss={avg_loss:.4f}  val_{metric_name}={score:.4f}")

        if score > best_score:
            best_score = score
            patience   = 0
            torch.save({
                "branch": branch.state_dict(),
                "head":   head.state_dict(),
                "epoch":  epoch,
                "score":  best_score,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Best val {metric_name} = {best_score:.4f}  (saved: {ckpt_path})")
    return best_score
