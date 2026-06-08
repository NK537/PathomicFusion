import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset


class PatientMILGeneformerDataset(Dataset):
    """
    One sample = one patient
    Returns:
      - uni_embs: (K, D_uni)
      - input_ids: (L,)
      - attention_mask: (L,)
      - surv, event, grade, pid
    """
    def __init__(
        self,
        patch_csv,
        emb_dir,
        gene_df,
        gene_tokenizer,
        k_patches=32,
        subset_ids=None,
        token_cache_path=None,
    ):
        df = pd.read_csv(patch_csv)

        if subset_ids is not None:
            df = df[df["TCGA_ID"].isin(subset_ids)].copy()

        self.emb_dir = emb_dir
        self.gene_df = gene_df
        self.k = k_patches
        self.gene_tokenizer = gene_tokenizer

        self.patch_groups = df.groupby("TCGA_ID")["patch_filename"].apply(list).to_dict()
        self.info = df.groupby("TCGA_ID").first()
        self.patient_ids = [pid for pid in self.patch_groups.keys() if pid in self.gene_df.index]

        # Precompute/load token cache
        self.token_cache = {}
        if token_cache_path is not None and os.path.exists(token_cache_path):
            print(f"[GeneFormer cache] Loading existing cache: {token_cache_path}")
            self.token_cache = torch.load(token_cache_path)
        else:
            print(f"[GeneFormer cache] Building cache: {token_cache_path}")
            for pid in self.patient_ids:
                gene_vec = self.gene_df.loc[pid].values
                input_ids, attention_mask = self.gene_tokenizer.encode_vector(gene_vec)
                self.token_cache[pid] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }

            if token_cache_path is not None:
                os.makedirs(os.path.dirname(token_cache_path), exist_ok=True)
                torch.save(self.token_cache, token_cache_path)
                print(f"[GeneFormer cache] Saved cache: {token_cache_path}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        patch_list = self.patch_groups[pid]

        if len(patch_list) >= self.k:
            chosen = random.sample(patch_list, self.k)
        else:
            chosen = [random.choice(patch_list) for _ in range(self.k)]

        embs = []
        for fn in chosen:
            base = os.path.splitext(fn)[0]
            ep = os.path.join(self.emb_dir, base + ".pt")
            embs.append(torch.load(ep, map_location="cpu").float())
        uni_embs = torch.stack(embs, dim=0)

        tok = self.token_cache[pid]
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        surv = torch.tensor(float(self.info.loc[pid]["Survival months"]), dtype=torch.float32)
        censored = float(self.info.loc[pid]["censored"])
        event = torch.tensor(1.0 - censored, dtype=torch.float32)
        grade = torch.tensor(float(self.info.loc[pid]["grade"]), dtype=torch.float32)

        return uni_embs, input_ids, attention_mask, surv, event, grade, pid