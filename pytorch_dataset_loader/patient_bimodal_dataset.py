import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset


class PatientBimodalDataset(Dataset):
    """
    One sample = one patient with TWO imaging modalities:
        histo_embs  (K_h, histo_dim)  - histopathology patch embeddings
        endo_embs   (K_e, endo_dim)   - endoscopy frame embeddings
        surv_time   scalar            - survival months
        event       scalar            - 1 = event observed, 0 = censored

    Required CSVs
    -------------
    histo_patches_csv  columns: patient_id, patch_filename
    endo_frames_csv    columns: patient_id, frame_filename
    label_csv          columns: patient_id, survival_months, censored
                                (censored=1 means event NOT observed)

    Embedding directories
    ---------------------
    histo_emb_dir  one .pt file (shape: histo_dim,) per patch_filename stem
    endo_emb_dir   one .pt file (shape: endo_dim,)  per frame_filename stem
    """

    def __init__(
        self,
        histo_patches_csv: str,
        endo_frames_csv:   str,
        label_csv:         str,
        histo_emb_dir:     str,
        endo_emb_dir:      str,
        k_histo:           int  = 32,
        k_endo:            int  = 16,
        subset_ids               = None,
    ):
        histo_df = pd.read_csv(histo_patches_csv)
        endo_df  = pd.read_csv(endo_frames_csv)
        label_df = pd.read_csv(label_csv).set_index("patient_id")

        if subset_ids is not None:
            histo_df = histo_df[histo_df["patient_id"].isin(subset_ids)]
            endo_df  = endo_df[endo_df["patient_id"].isin(subset_ids)]

        self.histo_emb_dir = histo_emb_dir
        self.endo_emb_dir  = endo_emb_dir
        self.k_histo       = k_histo
        self.k_endo        = k_endo
        self.label_df      = label_df

        # Group filenames per patient
        histo_groups = histo_df.groupby("patient_id")["patch_filename"].apply(list).to_dict()
        endo_groups  = endo_df.groupby("patient_id")["frame_filename"].apply(list).to_dict()

        # Keep only patients present in ALL three sources
        valid_ids = (
            set(histo_groups.keys())
            & set(endo_groups.keys())
            & set(label_df.index.tolist())
        )

        dropped_histo = set(histo_groups.keys()) - valid_ids
        dropped_endo  = set(endo_groups.keys())  - valid_ids
        if dropped_histo or dropped_endo:
            print(
                f"[PatientBimodalDataset] Dropped {len(dropped_histo | dropped_endo)} "
                f"patient(s) missing from one or more sources."
            )

        self.patient_ids   = sorted(valid_ids)
        self.histo_groups  = histo_groups
        self.endo_groups   = endo_groups

        print(
            f"[PatientBimodalDataset] {len(self.patient_ids)} patients  "
            f"| k_histo={k_histo}  k_endo={k_endo}"
        )

    def __len__(self):
        return len(self.patient_ids)

    def _sample_embs(self, file_list, emb_dir, k):
        """Sample k files (with replacement if needed), load and stack embeddings."""
        if len(file_list) >= k:
            chosen = random.sample(file_list, k)
        else:
            chosen = [random.choice(file_list) for _ in range(k)]

        embs = []
        for fn in chosen:
            base = os.path.splitext(fn)[0]
            path = os.path.join(emb_dir, base + ".pt")
            embs.append(torch.load(path, map_location="cpu").float())
        return torch.stack(embs, dim=0)   # (k, dim)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        histo_embs = self._sample_embs(
            self.histo_groups[pid], self.histo_emb_dir, self.k_histo
        )   # (K_h, histo_dim)

        endo_embs = self._sample_embs(
            self.endo_groups[pid], self.endo_emb_dir, self.k_endo
        )   # (K_e, endo_dim)

        row      = self.label_df.loc[pid]
        surv     = torch.tensor(float(row["survival_months"]), dtype=torch.float32)
        censored = float(row["censored"])
        event    = torch.tensor(1.0 - censored, dtype=torch.float32)

        return histo_embs, endo_embs, surv, event, pid
