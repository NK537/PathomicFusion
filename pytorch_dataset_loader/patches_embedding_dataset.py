import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class PatchEmbeddingDataset(Dataset):
    """
    Minimal dataset used ONLY to get patient IDs from the patch CSV
    when use_foundation=True. In your current pipeline you train using
    PatientMILDataset, but train_for_cancer still initializes this dataset
    for ID splitting.
    """
    def __init__(self, csv_file, emb_dir, subset_ids=None):
        self.data_frame = pd.read_csv(csv_file)
        if subset_ids is not None:
            self.data_frame = self.data_frame[self.data_frame["TCGA_ID"].isin(subset_ids)].copy()
        self.emb_dir = emb_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        pid = row["TCGA_ID"]
        # Return dummy embedding to satisfy Dataset interface (not used in MIL path)
        return torch.zeros(1), 0.0, 0.0, 0.0, pid