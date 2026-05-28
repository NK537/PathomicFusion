import os
import torch
torch.set_float32_matmul_precision("high")

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lifelines.utils import concordance_index

from models.histo_mil_branch import HistoMILBranch
from models.endo_mil_branch import EndoMILBranch
from fusion.cross_attention_fusion import CrossAttentionFusion
from fusion.attention_fusion import AttentionFusion
from COX.cox_loss import CustomCoxLoss
from pytorch_dataset_loader.patient_bimodal_dataset import PatientBimodalDataset


def build_fusion(cfg):
    """Instantiate fusion layer from config."""
    if cfg["fusion_type"] == "cross_attention":
        return CrossAttentionFusion(
            d_model       = cfg["out_dim"],
            n_heads       = cfg["n_heads"],
            n_gene_tokens = cfg["n_tokens"],
            fusion_dim    = cfg["fusion_dim"],
        )
    elif cfg["fusion_type"] == "concat":
        return _ConcatFusion(in_dim=cfg["out_dim"], fusion_dim=cfg["fusion_dim"])
    else:
        raise ValueError(f"Unknown fusion_type: {cfg['fusion_type']}")


class _ConcatFusion(nn.Module):
    """Simple concat + MLP baseline fusion."""
    def __init__(self, in_dim=64, fusion_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(2 * in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, histo_feat, endo_feat):
        fused = torch.cat([histo_feat, endo_feat], dim=1)
        return self.head(fused).squeeze(1)


class _SingleBranchHead(nn.Module):
    """Survival head for single-modality ablation experiments."""
    def __init__(self, in_dim=64, fusion_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, feat, _ignored=None):
        return self.head(feat).squeeze(1)


def train_picasso(
    cfg,
    train_ids,
    val_ids,
    fold_idx = None,
    ablation = None,   # None | "histo_only" | "endo_only"
):
    """
    Train one fold of the Picasso bimodal survival model.

    Histopathology branch freezing is controlled by cfg["freeze_histo"]:
        True  (default) — HistoMILBranch weights are FROZEN.
                          Only EndoMILBranch + fusion are optimised.
                          Histo embeddings are still used as fixed features.
        False           — All three components are trained end-to-end.

    Args:
        cfg:       PICASSO_CONFIG dict from config_picasso.py
        train_ids: list of patient_id strings for training
        val_ids:   list of patient_id strings for validation
        fold_idx:  fold number used in checkpoint filename
        ablation:  None (both modalities) | "histo_only" | "endo_only"

    Returns:
        best_val_cindex (float)
    """
    freeze_histo = cfg.get("freeze_histo", False)

    tag  = f"_fold{fold_idx}" if fold_idx is not None else ""
    mode = ablation if ablation else cfg["fusion_type"]
    frozen_tag = "_histoFROZEN" if freeze_histo else ""
    print(f"\n==== Picasso | {mode}{frozen_tag}{tag} ====")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = PatientBimodalDataset(
        histo_patches_csv = cfg["histo_patches_csv"],
        endo_frames_csv   = cfg["endo_frames_csv"],
        label_csv         = cfg["label_csv"],
        histo_emb_dir     = cfg["histo_emb_dir"],
        endo_emb_dir      = cfg["endo_emb_dir"],
        k_histo           = cfg["k_histo"],
        k_endo            = cfg["k_endo"],
        subset_ids        = train_ids,
    )
    val_ds = PatientBimodalDataset(
        histo_patches_csv = cfg["histo_patches_csv"],
        endo_frames_csv   = cfg["endo_frames_csv"],
        label_csv         = cfg["label_csv"],
        histo_emb_dir     = cfg["histo_emb_dir"],
        endo_emb_dir      = cfg["endo_emb_dir"],
        k_histo           = cfg["k_histo"],
        k_endo            = cfg["k_endo"],
        subset_ids        = val_ids,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)

    # ── Models ───────────────────────────────────────────────────────────────
    histo_branch = HistoMILBranch(histo_dim=cfg["histo_dim"], out_dim=cfg["out_dim"])
    endo_branch  = EndoMILBranch(endo_dim=cfg["endo_dim"],    out_dim=cfg["out_dim"])

    if ablation in ("histo_only", "endo_only"):
        fusion = _SingleBranchHead(in_dim=cfg["out_dim"], fusion_dim=cfg["fusion_dim"])
    else:
        fusion = build_fusion(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    histo_branch.to(device)
    endo_branch.to(device)
    fusion.to(device)

    # ── Freeze histo branch (optional) ───────────────────────────────────────
    # Weights are locked: no gradients, no parameter updates, eval mode always.
    # The branch still runs forward — its output feeds into the fusion layer
    # as fixed features derived from the pre-computed embeddings.
    if freeze_histo:
        histo_branch.requires_grad_(False)
        histo_branch.eval()
        print("  [HistoMILBranch] FROZEN — weights will not be updated.")

    # ── Optimizer — only train unfrozen parameters ────────────────────────────
    trainable_params = (
        list(endo_branch.parameters()) +
        list(fusion.parameters())
    )
    if not freeze_histo:
        trainable_params = list(histo_branch.parameters()) + trainable_params

    optimizer = optim.Adam(trainable_params, lr=cfg["lr"])
    cox_loss  = CustomCoxLoss()

    print(
        f"  Trainable params: {sum(p.numel() for p in trainable_params):,}  |  "
        f"Frozen histo params: "
        f"{sum(p.numel() for p in histo_branch.parameters()):,}"
        if freeze_histo else
        f"  Trainable params: {sum(p.numel() for p in trainable_params):,}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_cindex   = 0.0
    epochs_no_improve = 0

    for epoch in range(cfg["num_epochs"]):

        # ---- Train mode (histo_branch stays eval if frozen) ----
        if not freeze_histo:
            histo_branch.train()
        endo_branch.train()
        fusion.train()

        running_loss = 0.0
        all_scores, all_times, all_events = [], [], []

        for histo_embs, endo_embs, surv_times, events, pids in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']} [train]", leave=False
        ):
            histo_embs = histo_embs.to(device)
            endo_embs  = endo_embs.to(device)
            surv_times = surv_times.to(device)
            events     = events.to(device)

            # Run histo branch — no grad if frozen
            if freeze_histo:
                with torch.no_grad():
                    histo_feat = histo_branch(histo_embs)   # (B, out_dim) — fixed
            else:
                histo_feat = histo_branch(histo_embs)

            endo_feat = endo_branch(endo_embs)              # (B, out_dim) — trained

            if ablation == "histo_only":
                scores = fusion(histo_feat)
            elif ablation == "endo_only":
                scores = fusion(endo_feat)
            else:
                scores = fusion(histo_feat, endo_feat)      # (B,)

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
        if all_scores:
            train_cindex = concordance_index(
                torch.stack(all_times).cpu().numpy(),
                (-torch.stack(all_scores)).cpu().numpy(),
                torch.stack(all_events).cpu().numpy(),
            )

        # ---- Validate ----
        histo_branch.eval(); endo_branch.eval(); fusion.eval()
        val_scores, val_times, val_events = [], [], []

        with torch.no_grad():
            for histo_embs, endo_embs, surv_times, events, pids in val_loader:
                histo_embs = histo_embs.to(device)
                endo_embs  = endo_embs.to(device)
                surv_times = surv_times.to(device)
                events     = events.to(device)

                histo_feat = histo_branch(histo_embs)
                endo_feat  = endo_branch(endo_embs)

                if ablation == "histo_only":
                    scores = fusion(histo_feat)
                elif ablation == "endo_only":
                    scores = fusion(endo_feat)
                else:
                    scores = fusion(histo_feat, endo_feat)

                val_scores.extend(scores)
                val_times.extend(surv_times)
                val_events.extend(events)

        val_cindex = 0.0
        if val_scores:
            val_cindex = concordance_index(
                torch.stack(val_times).cpu().numpy(),
                (-torch.stack(val_scores)).cpu().numpy(),
                torch.stack(val_events).cpu().numpy(),
            )

        print(
            f"  Epoch [{epoch+1:02d}/{cfg['num_epochs']}] "
            f"Loss: {running_loss:.4f} | "
            f"Train C-idx: {train_cindex:.4f} | "
            f"Val C-idx: {val_cindex:.4f}"
        )

        # ---- Checkpoint ----
        if val_cindex > best_val_cindex + 0.001:
            best_val_cindex = val_cindex
            epochs_no_improve = 0
            ckpt_name = f"picasso_{mode}{frozen_tag}{tag}.pth"
            torch.save(
                {
                    "histo_branch": histo_branch.state_dict(),
                    "endo_branch":  endo_branch.state_dict(),
                    "fusion":       fusion.state_dict(),
                    "freeze_histo": freeze_histo,
                    "val_cindex":   val_cindex,
                    "epoch":        epoch,
                },
                os.path.join(cfg["checkpoint_dir"], ckpt_name),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                print(f"  Early stopping triggered at epoch {epoch+1}.")
                break

    print(f"  Best val C-index: {best_val_cindex:.4f}")
    return best_val_cindex
