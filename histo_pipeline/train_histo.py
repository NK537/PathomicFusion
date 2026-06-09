# histo_pipeline/train_histo.py
#
# Training loop for the Picasso histology survival pipeline.
# Mirrors train_picasso.py (endoscopy) — same Cox loss, same C-index metric.

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_float32_matmul_precision("high")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lifelines.utils import concordance_index

from models.histo_mil_branch import HistoMILBranch
from COX.cox_loss import CustomCoxLoss
from histo_pipeline.patient_histo_dataset import PatientHistoDataset


# ── Survival head ─────────────────────────────────────────────────────────────
class _SurvivalHead(nn.Module):
    """FC risk head on top of the pooled histology embedding."""
    def __init__(self, in_dim=64, hidden=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(1)   # (B,)


def train_histo(cfg, train_ids, val_ids, fold_idx=None):
    """
    Train one fold of the Picasso histology survival model.

    Architecture:
        HistoMILBranch  (k_patches, histo_dim) → (out_dim,)
        SurvivalHead    (out_dim,) → risk score (scalar)
        Loss            Cox partial likelihood

    Args:
        cfg:       HISTO_CONFIG dict from histo_pipeline/config_histo.py
        train_ids: list of patient ``code`` strings for training
        val_ids:   list of patient ``code`` strings for validation
        fold_idx:  fold number used in checkpoint filename

    Returns:
        best_val_cindex (float)
    """
    tag = f"_fold{fold_idx}" if fold_idx is not None else ""
    print(f"\n==== Histo | histo_only{tag} ====")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PatientHistoDataset(
        label_xlsx        = cfg["label_xlsx"],
        histo_patches_csv = cfg["histo_patches_csv"],
        histo_emb_dir     = cfg["histo_emb_dir"],
        k_patches         = cfg["k_patches"],
        subset_ids        = train_ids,
    )
    val_ds = PatientHistoDataset(
        label_xlsx        = cfg["label_xlsx"],
        histo_patches_csv = cfg["histo_patches_csv"],
        histo_emb_dir     = cfg["histo_emb_dir"],
        k_patches         = cfg["k_patches"],
        subset_ids        = val_ids,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    histo_branch = HistoMILBranch(
        histo_dim = cfg["histo_dim"],
        out_dim   = cfg["out_dim"],
    )
    head = _SurvivalHead(in_dim=cfg["out_dim"], hidden=cfg["fusion_dim"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    histo_branch.to(device)
    head.to(device)

    trainable = list(histo_branch.parameters()) + list(head.parameters())
    optimizer = optim.Adam(trainable, lr=cfg["lr"])
    cox_loss  = CustomCoxLoss()

    print(f"  Device          : {device}")
    print(f"  Train patients  : {len(train_ds)}  |  Val: {len(val_ds)}")
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_cindex   = 0.0
    epochs_no_improve = 0

    for epoch in range(cfg["num_epochs"]):

        # ---- Train ----
        histo_branch.train(); head.train()
        running_loss = 0.0
        all_scores, all_times, all_events = [], [], []

        for histo_embs, surv_times, events, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['num_epochs']} [train]",
            leave=False,
        ):
            histo_embs = histo_embs.to(device)    # (B, k_patches, histo_dim)
            surv_times = surv_times.to(device)
            events     = events.to(device)

            feat   = histo_branch(histo_embs)     # (B, out_dim)
            scores = head(feat)                   # (B,)

            loss = cox_loss(scores, surv_times, events)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not torch.isnan(loss):
                running_loss += loss.item()
                all_scores.extend(scores.detach())
                all_times.extend(surv_times.detach())
                all_events.extend(events.detach())

        train_cindex = 0.0
        if len(all_scores) > 1:
            train_cindex = concordance_index(
                torch.stack(all_times).cpu().numpy(),
                (-torch.stack(all_scores)).cpu().numpy(),
                torch.stack(all_events).cpu().numpy(),
            )

        # ---- Validate ----
        histo_branch.eval(); head.eval()
        val_scores, val_times, val_events = [], [], []

        with torch.no_grad():
            for histo_embs, surv_times, events, _ in val_loader:
                histo_embs = histo_embs.to(device)
                surv_times = surv_times.to(device)
                events     = events.to(device)

                feat   = histo_branch(histo_embs)
                scores = head(feat)

                val_scores.extend(scores)
                val_times.extend(surv_times)
                val_events.extend(events)

        val_cindex = 0.0
        if len(val_scores) > 1:
            val_cindex = concordance_index(
                torch.stack(val_times).cpu().numpy(),
                (-torch.stack(val_scores)).cpu().numpy(),
                torch.stack(val_events).cpu().numpy(),
            )

        print(
            f"  Epoch [{epoch+1:02d}/{cfg['num_epochs']}]  "
            f"Loss: {running_loss:.4f}  |  "
            f"Train C-idx: {train_cindex:.4f}  |  "
            f"Val C-idx: {val_cindex:.4f}"
        )

        # ---- Checkpoint ----
        if val_cindex > best_val_cindex + 0.001:
            best_val_cindex   = val_cindex
            epochs_no_improve = 0
            torch.save(
                {
                    "histo_branch": histo_branch.state_dict(),
                    "head":         head.state_dict(),
                    "val_cindex":   val_cindex,
                    "epoch":        epoch,
                    "cfg":          cfg,
                },
                os.path.join(cfg["checkpoint_dir"], f"histo_only{tag}.pth"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    print(f"  Best val C-index: {best_val_cindex:.4f}")
    return best_val_cindex
