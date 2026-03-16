import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset

class PatientMILDataset(Dataset):
    """
    One sample = one patient:
      - uni_embs: (K, D_uni)
      - gene_vec: (P,)
      - surv_time: scalar
      - event: scalar (1=event observed, 0=censored)
    """
    def __init__(self, patch_csv: str, emb_dir: str, gene_df: pd.DataFrame, k_patches: int = 32, subset_ids=None):
        df = pd.read_csv(patch_csv)

        if subset_ids is not None:
            df = df[df["TCGA_ID"].isin(subset_ids)].copy()

        self.emb_dir = emb_dir
        self.gene_df = gene_df
        self.k = k_patches

        # group patch filenames per patient
        self.patch_groups = df.groupby("TCGA_ID")["patch_filename"].apply(list).to_dict()

        # patient-level info (first row per patient)
        info = df.groupby("TCGA_ID").first()
        self.info = info

        self.patient_ids = list(self.patch_groups.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patch_list = self.patch_groups[pid]

        # sample K patches (with replacement if not enough)
        if len(patch_list) >= self.k:
            chosen = random.sample(patch_list, self.k)
        else:
            chosen = [random.choice(patch_list) for _ in range(self.k)]

        # load UNI embeddings: (K, D_uni)
        embs = []
        for fn in chosen:
            base = os.path.splitext(fn)[0]
            ep = os.path.join(self.emb_dir, base + ".pt")
            embs.append(torch.load(ep, map_location="cpu").float())
        uni_embs = torch.stack(embs, dim=0)  # (K, D_uni)

        # labels
        surv = torch.tensor(float(self.info.loc[pid]["Survival months"]), dtype=torch.float32)

        # Your CSV column is "censored"
        # Common meaning: censored=1 means event NOT observed -> event=0
        censored = float(self.info.loc[pid]["censored"])
        event = torch.tensor(1.0 - censored, dtype=torch.float32)

        grade = torch.tensor(float(self.info.loc[pid]["grade"]), dtype=torch.float32)

        # gene vector
        gene_vec = torch.tensor(self.gene_df.loc[pid].values, dtype=torch.float32)

        return uni_embs, gene_vec, surv, event, grade, pid