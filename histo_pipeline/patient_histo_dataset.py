# histo_pipeline/patient_histo_dataset.py
#
# Mirrors pytorch_dataset_loader/patient_endo_dataset.py but for histology patches.
# Key differences:
#   - Reads patch list from a CSV  (patient_id, patch_filename)
#   - Randomly samples k_patches patches per patient per epoch  (MIL augmentation)
#   - Embedding shape is always (histo_dim,) — one vector per patch
#   - Stacks sampled patches → (k_patches, histo_dim) fed into HistoMILBranch

import os
import random
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset


# ── Embedding loader ──────────────────────────────────────────────────────────
def _load_patch_embedding(path_no_ext: str) -> torch.Tensor:
    """
    Load one patch embedding.  Tries .pt → .npy → no extension.
    Returns a 1-D float32 tensor of shape (histo_dim,).
    """
    for ext in [".pt", ".npy", ""]:
        p = path_no_ext + ext
        if not os.path.exists(p):
            continue
        if ext == ".npy":
            t = torch.from_numpy(np.load(p)).float()
        else:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            t   = torch.from_numpy(obj).float() if isinstance(obj, np.ndarray) \
                  else obj.float()
        return t.squeeze()   # ensure (D,)
    raise FileNotFoundError(
        f"Patch embedding not found: {path_no_ext}  (.pt / .npy / no ext)"
    )


# ── Date utilities ────────────────────────────────────────────────────────────
_DATE_FMTS = ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")

def _parse_date(s):
    if pd.isna(s): return None
    for fmt in _DATE_FMTS:
        try:    return datetime.strptime(str(s).strip(), fmt)
        except: continue
    return None

def _days_between(d1, d2):
    a, b = _parse_date(d1), _parse_date(d2)
    return float(abs((b - a).days)) if a and b else None


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatientHistoDataset(Dataset):
    """
    Histology MIL dataset for the Picasso study.

    One sample = one patient:
        histo_embs   (k_patches, histo_dim)   randomly sampled patches
        surv_time    scalar                    days to outcome / follow-up
        event        scalar                    1 = event, 0 = censored

    Required files
    ──────────────
    histo_patches_csv
        CSV with columns:
            patient_id    — matches ``code`` in PicassoOnly_Outcome_train.xlsx
            patch_filename — filename of the patch image / embedding stem

    histo_emb_dir
        Folder containing one .pt (or .npy) file per patch.
        File name = ``patch_filename`` stem  (extension stripped + .pt appended).

    label_xlsx
        PicassoOnly_Outcome_train.xlsx  (shared with endoscopy pipeline).

    Args
    ────
    label_xlsx        : path to label Excel file
    histo_patches_csv : path to patch index CSV
    histo_emb_dir     : path to pre-computed embedding folder
    k_patches         : number of patches to sample per patient per epoch
    subset_ids        : restrict to these patient codes (for CV splits)
    min_patches       : minimum patches required to keep a patient (default 1)
    """

    def __init__(
        self,
        label_xlsx:        str,
        histo_patches_csv: str,
        histo_emb_dir:     str,
        k_patches:         int  = 32,
        subset_ids               = None,
        min_patches:       int  = 1,
    ):
        self.histo_emb_dir = histo_emb_dir
        self.k_patches     = k_patches

        # ── Read patch index ─────────────────────────────────────────────────
        patch_df = pd.read_csv(histo_patches_csv)
        patch_df["patient_id"] = patch_df["patient_id"].astype(str).str.zfill(4)

        # ── Read survival labels ─────────────────────────────────────────────
        raw  = pd.read_excel(label_xlsx)
        cols = list(raw.columns)
        # Rename unnamed days column (sits after date_of_outcome)
        try:
            doi = cols.index("date_of_outcome")
            cols[doi + 1] = "days_to_outcome"
        except (ValueError, IndexError):
            for i, c in enumerate(cols):
                if str(c).startswith("Unnamed:"):
                    cols[i] = "days_to_outcome"; break
        raw.columns = cols
        raw["code"]            = raw["code"].astype(str).str.zfill(4)
        raw["ANY OUTCOME"]     = pd.to_numeric(raw["ANY OUTCOME"],  errors="coerce")
        raw["days_to_outcome"] = pd.to_numeric(raw["days_to_outcome"], errors="coerce")
        label_df = raw.set_index("code")

        if subset_ids is not None:
            subset_ids = [str(s).zfill(4) for s in subset_ids]
            patch_df   = patch_df[patch_df["patient_id"].isin(subset_ids)]

        # Group patch filenames per patient
        patch_groups = (
            patch_df.groupby("patient_id")["patch_filename"]
            .apply(list).to_dict()
        )

        # ── Build patient sample list ────────────────────────────────────────
        self.samples = []
        n_skip_label = n_skip_emb = 0

        for pid, patch_files in patch_groups.items():
            # ---- Label ----
            if pid not in label_df.index:
                n_skip_label += 1
                continue
            row     = label_df.loc[pid]
            outcome = row["ANY OUTCOME"]
            if pd.isna(outcome):
                n_skip_label += 1
                continue

            event = float(int(outcome))
            days  = row["days_to_outcome"]
            if not pd.isna(days) and float(days) > 0:
                surv_time = float(days)
            else:
                t = _days_between(row.get("date_of_procedure"),
                                  row.get("date_of_visit"))
                if t is None or t <= 0:
                    n_skip_label += 1
                    continue
                surv_time = t

            # ---- Embeddings on disk ----
            valid_patches = []
            for fn in patch_files:
                base = os.path.splitext(fn)[0]
                stem = os.path.join(histo_emb_dir, base)
                if any(os.path.exists(stem + ext) for ext in [".pt", ".npy", ""]):
                    valid_patches.append(stem)

            if len(valid_patches) < min_patches:
                n_skip_emb += 1
                continue

            self.samples.append({
                "patient_id":    pid,
                "patch_stems":   valid_patches,
                "surv_time":     surv_time,
                "event":         event,
            })

        print(
            f"[PatientHistoDataset]  kept={len(self.samples)}  "
            f"skipped_no_embedding={n_skip_emb}  "
            f"skipped_bad_label={n_skip_label}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        stems = s["patch_stems"]

        # Randomly sample k_patches (with replacement if not enough patches)
        if len(stems) >= self.k_patches:
            chosen = random.sample(stems, self.k_patches)
        else:
            chosen = [random.choice(stems) for _ in range(self.k_patches)]

        # Load and stack → (k_patches, histo_dim)
        histo_embs = torch.stack(
            [_load_patch_embedding(p) for p in chosen], dim=0
        )

        return (
            histo_embs,
            torch.tensor(s["surv_time"], dtype=torch.float32),
            torch.tensor(s["event"],     dtype=torch.float32),
            s["patient_id"],
        )
