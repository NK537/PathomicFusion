import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset


# ── Embedding loader ──────────────────────────────────────────────────────────
def _load_embedding(path_no_ext: str) -> torch.Tensor:
    """
    Load a pre-computed embedding file.  Tries .pt → .npy → no extension.

    Handles two shapes that may exist in the Picasso folders:
        (D,)    — single aggregated embedding per section
        (K, D)  — K individual frame embeddings per section (e.g. K=15)

    Always returns a 2-D tensor  (K, D)  so callers are uniform.
    """
    for ext in [".pt", ".npy", ".mat", ""]:
        p = path_no_ext + ext
        if not os.path.exists(p):
            continue

        if ext == ".npy":
            t = torch.from_numpy(np.load(p)).float()
        elif ext == ".mat":
            import scipy.io as sio
            try:
                mat = sio.loadmat(p)
                keys = [k for k in mat.keys() if not k.startswith("_")]
                if not keys:
                    continue
                t = torch.from_numpy(mat[keys[0]]).float()
            except Exception:
                continue
        else:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(obj, np.ndarray):
                t = torch.from_numpy(obj).float()
            elif isinstance(obj, torch.Tensor):
                t = obj.float()
            else:
                raise TypeError(f"Unexpected type in {p}: {type(obj)}")

        # Ensure 2-D: (1, D) for single-vector files, (K, D) for frame stacks
        if t.dim() == 1:
            t = t.unsqueeze(0)   # (D,) → (1, D)
        elif t.dim() != 2:
            raise ValueError(f"Unexpected tensor shape {t.shape} in {p}")
        return t

    raise FileNotFoundError(
        f"Embedding not found: {path_no_ext}  (.pt / .npy / no ext)"
    )


# ── Date utilities ────────────────────────────────────────────────────────────
_DATE_FMTS = ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")

def _parse_date(s):
    if pd.isna(s):
        return None
    # Already a Timestamp (from Excel)
    if isinstance(s, (pd.Timestamp, datetime)):
        return s
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            continue
    try:
        return pd.Timestamp(s)
    except Exception:
        return None

def _days_between(d1, d2) -> float | None:
    try:
        a = _parse_date(d1)
        b = _parse_date(d2)
        if a is not None and b is not None:
            return float(abs((pd.Timestamp(b) - pd.Timestamp(a)).days))
    except Exception:
        pass
    return None


# ── VCE frame-quality mask ────────────────────────────────────────────────────
_RECTUM_FRAME_COLS  = [f"VCE rectum Frame {i}"  for i in range(1, 16)]
_SIGMOID_FRAME_COLS = [f"VCE sigmoid Frame {i}" for i in range(1, 16)]

def _vce_mask(row, section: str) -> list[int] | None:
    """
    Return a list of 0-based frame indices that are quality-1 for this section.
    Returns None if the column is absent or all entries are -1/NaN (fullframe case).
    """
    cols = _RECTUM_FRAME_COLS if section == "section1" else _SIGMOID_FRAME_COLS
    if not all(c in row.index for c in cols):
        return None
    vals = [int(pd.to_numeric(row[c], errors="coerce") or 0) for c in cols]
    good = [i for i, v in enumerate(vals) if v == 1]
    # If nothing is marked 1, return None (keep all frames)
    return good if good else None


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatientEndoDataset(Dataset):
    """
    Endoscopy-only MIL dataset for the Picasso study.

    Labels source: PicassoOnly_Outcome_train.xlsx
    ─────────────────────────────────────────────
    • patient key  : ``code``  column  (e.g. "0101")
    • survival time: unnamed column immediately after ``date_of_outcome``
                     (positive = days to event; -1 = censored / unknown)
    • event flag   : ``ANY OUTCOME``  (1 = event, 0 = censored)
    • fallback time: ``date_of_visit`` - ``date_of_procedure``  when days = -1

    Embedding file convention (train split)
    ───────────────────────────────────────
    ``{emb_prefix}_pat{code}_section1``   ← rectum
    ``{emb_prefix}_pat{code}_section2``   ← sigmoid
    (extensions tried: .pt → .npy → no extension)

    Each file may contain
    • (D,)    → single aggregated embedding  → treated as 1 frame
    • (K, D)  → K frame embeddings           → MIL pools over K

    VCE frame quality mask (optional)
    ──────────────────────────────────
    When ``use_vce_mask=True`` and the embedding file contains K=15 frames,
    only frames labelled 1 in the VCE frame columns are kept.
    Frames labelled 0 or -1 are discarded before MIL pooling.

    Args
    ────
    label_xlsx    : path to PicassoOnly_Outcome_train.xlsx
    emb_dir       : folder that holds the pre-computed embedding files
    emb_prefix    : filename stem before ``_pat{code}_section{N}``
    subset_ids    : restrict to these ``code`` values (for CV splits)
    min_sections  : minimum sections required per patient (default 1)
    use_vce_mask  : apply per-frame quality mask when available (default True)
    """

    EMB_PREFIX_TRAIN = "RN50_GastroNet5M_DINOv1_feat_WLE_PicassoTrain"
    EMB_PREFIX_TEST  = "RN50_GastroNet5M_DINOv1_feat_WLE_PicassoTest"

    def __init__(
        self,
        label_xlsx:   str,
        emb_dir:      str,
        emb_prefix:   str  = None,
        subset_ids          = None,
        min_sections: int  = 1,
        use_vce_mask: bool = True,
    ):
        self.emb_dir      = emb_dir
        self.emb_prefix   = emb_prefix or self.EMB_PREFIX_TRAIN
        self.use_vce_mask = use_vce_mask

        # ── Read Excel ───────────────────────────────────────────────────────
        raw = pd.read_excel(label_xlsx)

        # Rename the unnamed column that sits right after date_of_outcome
        cols = list(raw.columns)
        try:
            doi = cols.index("date_of_outcome")
            cols[doi + 1] = "days_to_outcome"
        except (ValueError, IndexError):
            # Fallback: look for the first unnamed column
            for i, c in enumerate(cols):
                if str(c).startswith("Unnamed:"):
                    cols[i] = "days_to_outcome"
                    break
        raw.columns = cols

        # Clean up types
        raw["code"]            = raw["code"].astype(str).str.zfill(4)
        raw["ANY OUTCOME"]     = pd.to_numeric(raw["ANY OUTCOME"],  errors="coerce")
        raw["days_to_outcome"] = pd.to_numeric(raw["days_to_outcome"], errors="coerce")

        # Replace Excel formula errors (#NAME?, #REF!, etc.) with NaN
        for col in ("File rectum", "File sigmoid"):
            if col in raw.columns:
                raw[col] = raw[col].apply(
                    lambda x: None if str(x).startswith("#") else x
                )

        if subset_ids is not None:
            subset_ids = [str(s).zfill(4) for s in subset_ids]
            raw = raw[raw["code"].isin(subset_ids)].reset_index(drop=True)

        # ── Build patient list ───────────────────────────────────────────────
        self.samples = []
        n_skipped_label = n_skipped_emb = 0

        for _, row in raw.iterrows():
            code    = str(row["code"]).zfill(4)
            outcome = row["ANY OUTCOME"]
            days    = row["days_to_outcome"]

            # ---- Survival label ----
            if pd.isna(outcome):
                n_skipped_label += 1
                continue

            event = float(int(outcome))

            if not pd.isna(days) and float(days) > 0:
                surv_time = float(days)
            else:
                # Censored / unknown time — derive from visit gap
                t = _days_between(row.get("date_of_procedure"),
                                  row.get("date_of_visit"))
                if t is None or t <= 0:
                    n_skipped_label += 1
                    continue
                surv_time = t

            # ---- Embedding discovery ----
            sections = []
            for sec_name, sec_tag in [("section1", "section1"),
                                      ("section2", "section2")]:
                stem = f"{self.emb_prefix}_pat{code}_{sec_tag}"
                full = os.path.join(self.emb_dir, stem)
                def _is_valid_mat(path):
                    try:
                        import scipy.io as sio
                        sio.loadmat(path)
                        return True
                    except Exception:
                        return False

                if any(
                    (os.path.exists(full + ext) and (ext != ".mat" or _is_valid_mat(full + ext)))
                    for ext in [".pt", ".npy", ".mat", ""]
                ):
                    # Collect VCE mask for this section (may be None)
                    mask = (
                        _vce_mask(row, sec_name)
                        if self.use_vce_mask
                        else None
                    )
                    sections.append((full, mask))

            if len(sections) < min_sections:
                n_skipped_emb += 1
                continue

            self.samples.append({
                "patient_id": code,
                "sections":   sections,   # list of (path_no_ext, mask_or_None)
                "surv_time":  surv_time,
                "event":      event,
            })

        print(
            f"[PatientEndoDataset]  kept={len(self.samples)}  "
            f"skipped_no_embedding={n_skipped_emb}  "
            f"skipped_bad_label={n_skipped_label}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        frames_list = []
        for path, mask in s["sections"]:
            emb = _load_embedding(path)   # always (K, D)

            # Apply VCE quality mask when the embedding really has 15 frames
            if mask is not None and emb.shape[0] == 15 and len(mask) > 0:
                emb = emb[mask]           # keep only quality-1 frames

            frames_list.append(emb)

        # Concatenate all sections → (total_frames, D)
        endo_embs = torch.cat(frames_list, dim=0)

        return (
            endo_embs,
            torch.tensor(s["surv_time"], dtype=torch.float32),
            torch.tensor(s["event"],     dtype=torch.float32),
            s["patient_id"],
        )
