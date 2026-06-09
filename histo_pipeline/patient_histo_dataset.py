# histo_pipeline/patient_histo_dataset.py
#
# Histology MIL dataset for the Picasso study.
#
# Patch filename format:  {site}_{patient:03d} {Section} {slide}_{row}_{col}.png
#   e.g.  02_003 Sigmoid 1E_16_146.png  ->  patient 02-03
#
# Label source: PicassoAI_DataList_Fusion.xlsx
#   Pat_ID (string like "'02-03'"), Outcome (0/1), Train_Outcome_rev2 (split)
#
# TTE source: PICASSO_outcome_tte.xlsx
#   Columns: ID (e.g. "01-01"), outcome (0/1), tte (days, "-" = missing)
#
# One Dataset sample = one patient:
#   histo_embs  (k_patches, histo_dim)
#   surv_time   float   days (or 1.0 if TTE missing)
#   event       float   1=event, 0=censored
#   patient_id  str

import os
import re
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Patient ID helpers
# ---------------------------------------------------------------------------

def filename_to_pat_id(filename: str):
    """
    Parse patient ID from patch filename.
    "02_003 Sigmoid 1E_16_146.png"  ->  "02-03"
    "01_014 Rectum 2A_10_055.png"   ->  "01-14"
    Returns None if pattern does not match.
    """
    name = os.path.basename(filename)
    m = re.match(r"^(\d{2})_(\d{2,3})\s", name)
    if not m:
        return None
    site    = m.group(1)
    patient = int(m.group(2))
    return f"{site}-{patient:02d}"


def normalize_pat_id(raw) -> str:
    """
    Strip stray single-quotes from Excel cells.
    "'02-03'" -> "02-03"
    """
    return str(raw).strip().strip("'").strip()


# ---------------------------------------------------------------------------
# Embedding loader
# ---------------------------------------------------------------------------

def _load_patch_embedding(path_no_ext: str) -> torch.Tensor:
    """Load one patch embedding (.pt / .npy / no-ext). Returns (D,) tensor."""
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
        return t.squeeze()   # (D,)
    raise FileNotFoundError(
        f"Patch embedding not found: {path_no_ext}  (.pt / .npy / no ext)"
    )


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def _load_tte_file(tte_xlsx: str) -> dict:
    """
    Read PICASSO_outcome_tte.xlsx.

    Expected columns (from the actual file):
        ID        patient identifier, e.g. "01-01"
        outcome   0 or 1
        tte       days to event/censoring  ("-" = missing -> NaN)

    Returns dict: pat_id -> {"event": int, "surv_time": float or NaN}
    """
    df  = pd.read_excel(tte_xlsx)

    # Find ID column
    id_col = None
    for c in df.columns:
        if str(c).strip().upper() == "ID":
            id_col = c; break
    if id_col is None:
        # Fall back: first column that looks like patient IDs
        for c in df.columns:
            sample = df[c].dropna().astype(str).head(3).tolist()
            if all(re.match(r"\d{2}-\d{2}", s) for s in sample):
                id_col = c; break
    if id_col is None:
        print(f"[TTE] WARNING: could not identify ID column in {tte_xlsx}. "
              f"Columns: {list(df.columns)}")
        return {}

    # Find outcome column
    out_col = None
    for c in df.columns:
        if str(c).strip().lower() == "outcome":
            out_col = c; break

    # Find TTE column
    tte_col = None
    for c in df.columns:
        if str(c).strip().lower() == "tte":
            tte_col = c; break

    if tte_col is None:
        # Fall back: numeric column that looks like days (large integers)
        candidates = [c for c in df.columns if c not in [id_col, out_col]]
        for c in candidates:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(vals) > 5 and vals.mean() > 30:
                tte_col = c; break

    print(f"[TTE] Using columns: id='{id_col}'  outcome='{out_col}'  tte='{tte_col}'  "
          f"({len(df)} rows)")

    result = {}
    for _, row in df.iterrows():
        pid = normalize_pat_id(row[id_col])
        if not pid or pid == "nan":
            continue

        # TTE: convert "-" / NaN to NaN
        surv_time = float("nan")
        if tte_col is not None:
            raw_tte = str(row[tte_col]).strip()
            if raw_tte not in ("-", "", "nan", "NaN", "None"):
                try:
                    surv_time = float(raw_tte)
                except ValueError:
                    pass

        # Event
        event = float("nan")
        if out_col is not None:
            raw_out = pd.to_numeric(row[out_col], errors="coerce")
            if not pd.isna(raw_out):
                event = float(raw_out)

        result[pid] = {"event": event, "surv_time": surv_time}

    n_valid = sum(1 for v in result.values()
                  if not pd.isna(v["surv_time"]) and v["surv_time"] > 0)
    print(f"[TTE] Loaded {len(result)} patients, {n_valid} with valid TTE")
    return result


def load_labels(fusion_xlsx: str, tte_xlsx: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by clean Pat_ID (e.g. "02-03") with columns:
        event      int     0 or 1
        surv_time  float   days (NaN if TTE not available)
        split      int     Train_Outcome_rev2 value
    """
    # Primary: Fusion file for patient list + split flags
    df = pd.read_excel(fusion_xlsx)
    df["pat_id"] = df["Pat_ID"].apply(normalize_pat_id)
    df["event"]  = pd.to_numeric(df["Outcome"], errors="coerce")
    df = df[df["event"].isin([0.0, 1.0])].copy()
    df["event"]  = df["event"].astype(int)
    df["split"]  = pd.to_numeric(df["Train_Outcome_rev2"], errors="coerce").fillna(-1).astype(int)
    df["surv_time"] = float("nan")

    # TTE: overlay event + surv_time from PICASSO_outcome_tte.xlsx
    if tte_xlsx and os.path.exists(tte_xlsx):
        tte_map = _load_tte_file(tte_xlsx)
        if tte_map:
            def _get_tte(pid):
                v = tte_map.get(pid)
                return v["surv_time"] if v else float("nan")
            def _get_event(pid):
                v = tte_map.get(pid)
                # Prefer TTE file's outcome when available
                if v and not pd.isna(v["event"]):
                    return int(v["event"])
                return None

            df["surv_time"] = df["pat_id"].apply(_get_tte)
            # Update event where TTE file has it (more authoritative)
            df["event_tte"] = df["pat_id"].apply(_get_event)
            mask = df["event_tte"].notna()
            df.loc[mask, "event"] = df.loc[mask, "event_tte"].astype(int)
            df.drop(columns=["event_tte"], inplace=True)

            n_with_tte = df["surv_time"].notna().sum()
            print(f"[Labels] TTE matched for {n_with_tte}/{len(df)} patients")
    else:
        print(f"[Labels] WARNING: PICASSO_outcome_tte.xlsx not found at '{tte_xlsx}'.")
        print(f"         Cox loss disabled. Copy the file to enable full survival analysis.")

    df = df.set_index("pat_id")
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatientHistoDataset(Dataset):
    """
    Histology MIL dataset for Picasso survival prediction.

    Args:
        fusion_label_xlsx : PicassoAI_DataList_Fusion.xlsx
        tte_label_xlsx    : PICASSO_outcome_tte.xlsx (optional, enables Cox loss)
        histo_emb_dir     : folder with pre-computed .pt embeddings
        k_patches         : patches sampled per patient per epoch
        subset_ids        : restrict to these Pat_IDs  (e.g. ["02-03", "01-14"])
        min_patches       : minimum embeddings required to keep a patient
    """

    def __init__(
        self,
        fusion_label_xlsx: str,
        tte_label_xlsx,
        histo_emb_dir:     str,
        k_patches:         int = 64,
        subset_ids              = None,
        min_patches:       int = 1,
    ):
        self.histo_emb_dir = histo_emb_dir
        self.k_patches     = k_patches

        # ---- Labels --------------------------------------------------------
        label_df = load_labels(fusion_label_xlsx, tte_label_xlsx)

        if subset_ids is not None:
            subset_ids = [normalize_pat_id(s) for s in subset_ids]
            label_df   = label_df[label_df.index.isin(subset_ids)]

        # ---- Scan embedding dir for available patches ----------------------
        # Maps pat_id  ->  list of full stems (no extension)
        pat_patches: dict = {}
        if os.path.isdir(histo_emb_dir):
            for fname in sorted(os.listdir(histo_emb_dir)):
                stem = os.path.splitext(fname)[0]   # strip .pt / .npy
                pat  = filename_to_pat_id(stem)
                if pat is None:
                    pat = filename_to_pat_id(fname)
                if pat is None:
                    continue
                full_stem = os.path.join(histo_emb_dir, stem)
                pat_patches.setdefault(pat, []).append(full_stem)
        else:
            print(f"[PatientHistoDataset] WARNING: histo_emb_dir not found: {histo_emb_dir}")

        # ---- Build sample list ---------------------------------------------
        self.samples  = []
        self.use_cox  = False
        n_skip_label  = 0
        n_skip_emb    = 0

        for pat_id, row in label_df.iterrows():
            event = row["event"]
            if pd.isna(event):
                n_skip_label += 1; continue

            surv_time = row["surv_time"]
            has_tte   = (not pd.isna(surv_time)) and float(surv_time) > 0
            if has_tte:
                self.use_cox = True

            patches = pat_patches.get(pat_id, [])
            if len(patches) < min_patches:
                n_skip_emb += 1; continue

            self.samples.append({
                "patient_id":  pat_id,
                "patch_stems": patches,
                "surv_time":   float(surv_time) if has_tte else 1.0,
                "event":       float(event),
            })

        if not self.use_cox:
            print("[PatientHistoDataset] Cox loss DISABLED — TTE unavailable. "
                  "Using BCE loss on binary outcome.")

        print(
            f"[PatientHistoDataset]  kept={len(self.samples)}  "
            f"events={sum(s['event']==1 for s in self.samples)}  "
            f"censored={sum(s['event']==0 for s in self.samples)}  "
            f"skipped_no_emb={n_skip_emb}  skipped_bad_label={n_skip_label}  "
            f"loss={'Cox' if self.use_cox else 'BCE'}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        stems  = s["patch_stems"]

        if len(stems) >= self.k_patches:
            chosen = random.sample(stems, self.k_patches)
        else:
            chosen = [random.choice(stems) for _ in range(self.k_patches)]

        histo_embs = torch.stack(
            [_load_patch_embedding(p) for p in chosen], dim=0
        )   # (k_patches, histo_dim)

        return (
            histo_embs,
            torch.tensor(s["surv_time"], dtype=torch.float32),
            torch.tensor(s["event"],     dtype=torch.float32),
            s["patient_id"],
        )
