import os
import torch
import pandas as pd
torch.set_float32_matmul_precision("high")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lifelines.utils import concordance_index

from models.endo_mil_branch import EndoMILBranch
from COX.cox_loss import CustomCoxLoss
from pytorch_dataset_loader.patient_endo_dataset import PatientEndoDataset


# ── Survival head (endo-only for now) ────────────────────────────────────────
class _SurvivalHead(nn.Module):
    """FC risk head on top of the pooled endo embedding."""
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


def train_picasso(cfg, train_ids, val_ids, fold_idx=None):
    """
    Train one fold of the Picasso endoscopy survival model.

    Returns:
        (best_val_cindex, oof_df)
        oof_df: DataFrame with columns [patient_id, risk_score, surv_time, event]
                containing validation predictions at the best epoch
    """
    tag = f"_fold{fold_idx}" if fold_idx is not None else ""
    print(f"\n==== Picasso | endo_only{tag} ====")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = PatientEndoDataset(
        label_xlsx  = cfg["label_xlsx"],
        emb_dir     = cfg["endo_emb_dir"],
        emb_prefix  = cfg["endo_emb_prefix"],
        subset_ids  = train_ids,
    )
    val_ds = PatientEndoDataset(
        label_xlsx  = cfg["label_xlsx"],
        emb_dir     = cfg["endo_emb_dir"],
        emb_prefix  = cfg["endo_emb_prefix"],
        subset_ids  = val_ids,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True,
        collate_fn=_pad_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=_pad_collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    endo_branch = EndoMILBranch(
        endo_dim = cfg["endo_dim"],
        out_dim  = cfg["out_dim"],
    )
    head = _SurvivalHead(in_dim=cfg["out_dim"], hidden=cfg["fusion_dim"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endo_branch.to(device)
    head.to(device)

    trainable = list(endo_branch.parameters()) + list(head.parameters())
    optimizer = optim.Adam(trainable, lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    cox_loss  = CustomCoxLoss()

    print(f"  Device: {device}")
    print(f"  Train patients: {len(train_ds)}  |  Val patients: {len(val_ds)}")
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_cindex   = 0.0
    best_val_preds    = None   # saved at best epoch for KM analysis
    epochs_no_improve = 0

    for epoch in range(cfg["num_epochs"]):

        # ---- Train ----
        endo_branch.train(); head.train()
        running_loss = 0.0
        all_scores, all_times, all_events = [], [], []

        for endo_embs, surv_times, events, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg['num_epochs']} [train]",
            leave=False,
        ):
            endo_embs  = endo_embs.to(device)   # (B, K, endo_dim)
            surv_times = surv_times.to(device)
            events     = events.to(device)

            feat   = endo_branch(endo_embs)     # (B, out_dim)
            scores = head(feat)                 # (B,)

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
        endo_branch.eval(); head.eval()
        val_scores, val_times, val_events = [], [], []

        val_pids = []
        with torch.no_grad():
            for endo_embs, surv_times, events, pids in val_loader:
                endo_embs  = endo_embs.to(device)
                surv_times = surv_times.to(device)
                events     = events.to(device)

                feat   = endo_branch(endo_embs)
                scores = head(feat)

                val_scores.extend(scores)
                val_times.extend(surv_times)
                val_events.extend(events)
                val_pids.extend(pids)

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

                # ---- LR Scheduler ----
        scheduler.step(val_cindex)

        # ---- Checkpoint ----
        if val_cindex > best_val_cindex + 0.001:
            best_val_cindex = val_cindex
            epochs_no_improve = 0
            # Save OOF predictions at this best epoch
            best_val_preds = pd.DataFrame({
                "patient_id": val_pids,
                "risk_score": torch.stack(val_scores).cpu().numpy(),
                "surv_time":  torch.stack(val_times).cpu().numpy(),
                "event":      torch.stack(val_events).cpu().numpy(),
            })
            torch.save(
                {
                    "endo_branch": endo_branch.state_dict(),
                    "head":        head.state_dict(),
                    "val_cindex":  val_cindex,
                    "epoch":       epoch,
                    "cfg":         cfg,
                },
                os.path.join(cfg["checkpoint_dir"], f"picasso_endo_only{tag}.pth"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

    print(f"  Best val C-index: {best_val_cindex:.4f}")
    if best_val_preds is None and len(val_scores) > 0:
        # Fallback: use last epoch predictions if no improvement was ever recorded
        best_val_preds = pd.DataFrame({
            "patient_id": val_pids,
            "risk_score": torch.stack(val_scores).cpu().numpy(),
            "surv_time":  torch.stack(val_times).cpu().numpy(),
            "event":      torch.stack(val_events).cpu().numpy(),
        })
    return best_val_cindex, best_val_preds


# ── Collate — pads variable-length section sequences to same K ───────────────
def _pad_collate(batch):
    """
    Pads (num_sections, D) tensors in a batch to the same K (max sections).
    Patients with fewer sections are zero-padded — AttnMILPool handles this
    correctly since padded rows get near-zero attention weights.
    """
    embs_list, times, events, pids = zip(*batch)

    max_k = max(e.shape[0] for e in embs_list)
    D     = embs_list[0].shape[1]

    padded = torch.zeros(len(embs_list), max_k, D)
    for i, e in enumerate(embs_list):
        padded[i, :e.shape[0], :] = e

    return (
        padded,
        torch.stack(list(times)),
        torch.stack(list(events)),
        list(pids),
    )
