# histo_pipeline/patient_histo_dataset.py
#
# Histology MIL dataset for the Picasso study.
#
# Label source: PICASSO_dataframe.xlsx
#   ID  (e.g. "01-01")   -- patient identifier
#   WSI (e.g. "02_001 Rectum 2F") -- WSI name = prefix of patch filenames
#   outcome  0 / 1
#
# Patch / embedding naming:
#   {WSI}_{tile_idx}.png  /  {WSI}_{tile_idx}.pt
#   e.g.  "02_001 Rectum 2F_0.pt",  "02_001 Rectum 2F_1.pt", ...
#
# TTE source (optional): PicassoOnly_Outcome_train.xlsx
#   code  (e.g. "0101")  = ID with dash stripped ("01-01" -> "0101")
#   days_to_outcome, ANY OUTCOME
#
# One Dataset sample = one patient:
#   histo_embs  (k_patches, histo_dim)
#   surv_time   float   days (or 1.0 if TTE missing)
#   event       float   1=event, 0=censored
#   patient_id  str

import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Patient ID helpers
# ---------------------------------------------------------------------------

def normalize_pat_id(raw) -> str:
    """Strip stray quotes: \"'02-03'\" -> \"02-03\" """
    return str(raw).strip().strip("'").strip()


def pat_id_to_code(pat_id: str) -> str:
    """'01-01' -> '0101'  (matches `code` column in Outcome_train file)"""
    return pat_id.replace("-", "")


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
# TTE loader
# ---------------------------------------------------------------------------

def _load_tte(tte_xlsx: str) -> dict:
    """
    Read PICASSO_outcome_tte.xlsx.

    Columns used:
        ID       patient identifier, e.g. "01-01"
        outcome  event flag 0 / 1
        tte      days to event/censoring  ("-" or blank = missing -> NaN)

    Returns dict: patient_id -> {"event": float, "surv_time": float}
    """
    df = pd.read_excel(tte_xlsx)

    # Identify columns (case-insensitive)
    col_map = {str(c).strip().lower(): c for c in df.columns}
    id_col      = col_map.get("id")
    outcome_col = col_map.get("outcome")
    tte_col     = col_map.get("tte")

    if id_col is None:
        print(f"[TTE] WARNING: 'ID' column not found in {tte_xlsx}. Columns: {list(df.columns)}")
        return {}

    print(f"[TTE] Using columns: id='{id_col}'  outcome='{outcome_col}'  tte='{tte_col}'")

    result = {}
    for _, row in df.iterrows():
        pid = normalize_pat_id(row[id_col])
        if not pid or pid == "nan":
            continue

        # TTE: treat "-", blank, nan as missing
        surv_time = float("nan")
        if tte_col is not None:
            raw = str(row[tte_col]).strip()
            if raw not in ("-", "", "nan", "NaN", "None"):
                try:
                    surv_time = float(raw)
                except ValueError:
                    pass

        # Event
        event = float("nan")
        if outcome_col is not None:
            v = pd.to_numeric(row[outcome_col], errors="coerce")
            if not pd.isna(v):
                event = float(v)

        # Reject impossible values: negative, zero, or Excel date serial numbers (>3000 days ~8 yrs)
        if not pd.isna(surv_time) and (surv_time <= 0 or surv_time > 3000):
            surv_time = float("nan")

        result[pid] = {"event": event, "surv_time": surv_time}

    n_valid = sum(1 for v in result.values()
                  if not pd.isna(v["surv_time"]) and v["surv_time"] > 0)
    print(f"[TTE] Loaded {len(result)} patients, {n_valid} with valid TTE")
    return result


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_labels(histo_label_xlsx: str, tte_xlsx: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by clean patient ID (e.g. "01-01") with cols:
        event      int     0 or 1
        surv_time  float   days (NaN if TTE unavailable)
        wsi_list   list    WSI names belonging to this patient
    """
    df = pd.read_excel(histo_label_xlsx)
    df["ID"]      = df["ID"].apply(normalize_pat_id)
    df["outcome"] = pd.to_numeric(df["outcome"], errors="coerce")
    df["WSI"]     = df["WSI"].astype(str).str.strip()

    df = df.dropna(subset=["outcome", "WSI"]).copy()
    df = df[df["outcome"].isin([0.0, 1.0])].copy()

    # Aggregate to patient level: max outcome, collect WSI list
    pat_df = (
        df.groupby("ID")
        .agg(
            event    = ("outcome", "max"),
            wsi_list = ("WSI",     list),
        )
        .reset_index()
    )
    pat_df["event"]     = pat_df["event"].astype(int)
    pat_df["surv_time"] = float("nan")

    # Overlay TTE data
    if tte_xlsx and os.path.exists(tte_xlsx):
        tte_map = _load_tte(tte_xlsx)  # keyed by "01-01" directly

        def _get_surv(pid):
            v = tte_map.get(pid)
            return v["surv_time"] if v else float("nan")

        def _get_event_tte(pid):
            v = tte_map.get(pid)
            return v["event"] if (v and not pd.isna(v["event"])) else float("nan")

        pat_df["surv_time"] = pat_df["ID"].apply(_get_surv)
        event_tte           = pat_df["ID"].apply(_get_event_tte)
        mask = event_tte.notna()
        pat_df.loc[mask, "event"] = event_tte[mask].astype(int)

        n_tte = pat_df["surv_time"].notna().sum()
        print(f"[Labels] TTE matched for {n_tte}/{len(pat_df)} patients")
    else:
        print(f"[Labels] TTE file not found at '{tte_xlsx}' — Cox loss disabled.")

    pat_df = pat_df.set_index("ID")
    print(f"[Labels] Total patients: {len(pat_df)}  events={pat_df['event'].sum()}")
    return pat_df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatientHistoDataset(Dataset):
    """
    Histology MIL dataset for Picasso survival prediction.

    Matches embeddings to patients via the WSI column in PICASSO_dataframe.xlsx.
    For WSI "02_001 Rectum 2F", all embedding stems starting with "02_001 Rectum 2F_"
    are collected as patches for that WSI.

    Args:
        histo_label_xlsx : PICASSO_dataframe.xlsx
        tte_label_xlsx   : PicassoOnly_Outcome_train.xlsx (optional, enables Cox loss)
        histo_emb_dir    : folder with pre-computed .pt embeddings
        k_patches        : patches sampled per patient per epoch
        subset_ids       : restrict to these patient IDs
        min_patches      : minimum patch embeddings required to keep a patient
    """

    def __init__(
        self,
        histo_label_xlsx: str,
        tte_label_xlsx,
        histo_emb_dir:    str,
        k_patches:        int = 64,
        subset_ids             = None,
        min_patches:      int = 1,
    ):
        self.histo_emb_dir = histo_emb_dir
        self.k_patches     = k_patches

        # ---- Labels --------------------------------------------------------
        label_df = load_labels(histo_label_xlsx, tte_label_xlsx)

        if subset_ids is not None:
            subset_ids = [normalize_pat_id(s) for s in subset_ids]
            label_df   = label_df[label_df.index.isin(subset_ids)]

        # ---- Index all embedding stems in histo_emb_dir -------------------
        all_stems = []
        if os.path.isdir(histo_emb_dir):
            all_stems = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(histo_emb_dir)
                if os.path.splitext(f)[1] in (".pt", ".npy") or "." not in f
            )
        else:
            print(f"[PatientHistoDataset] WARNING: histo_emb_dir not found: {histo_emb_dir}")

        # ---- Map each WSI to its matching embedding stems ------------------
        # WSI "02_001 Rectum 2F" matches stems starting with "02_001 Rectum 2F_"
        all_wsis = set(
            wsi
            for row in label_df.itertuples()
            for wsi in row.wsi_list
        )
        wsi_stems_map: dict = {}
        for wsi in all_wsis:
            prefix  = wsi + "_"
            matched = [
                os.path.join(histo_emb_dir, s)
                for s in all_stems
                if s.startswith(prefix)
            ]
            if matched:
                wsi_stems_map[wsi] = matched

        # ---- Build sample list ---------------------------------------------
        self.samples  = []
        self.use_cox  = False
        n_skip_emb    = 0

        for pat_id, row in label_df.iterrows():
            event     = row["event"]
            surv_time = row["surv_time"]
            has_tte   = (not pd.isna(surv_time)) and float(surv_time) > 0
            if has_tte:
                self.use_cox = True

            # All patches across all WSIs for this patient
            all_pat_stems = []
            for wsi in row["wsi_list"]:
                all_pat_stems.extend(wsi_stems_map.get(wsi, []))

            if len(all_pat_stems) < min_patches:
                n_skip_emb += 1
                continue

            self.samples.append({
                "patient_id":  pat_id,
                "patch_stems": all_pat_stems,
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
            f"skipped_no_emb={n_skip_emb}  "
            f"loss={'Cox' if self.use_cox else 'BCE'}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        stems = s["patch_stems"]

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
