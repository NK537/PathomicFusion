import torch
import torch.nn as nn
from models.mil_pooling import AttnMILPool

class UNIMILBranch(nn.Module):
    """
    Input:  UNI embeddings per patient: (B, K, D_uni=1024)
    Output: patient-level image embedding: (B, 64)
    """
    def __init__(self, uni_dim: int = 1024, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(uni_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )
        self.pool = AttnMILPool(d=out_dim, hidden=128, dropout=0.1)

    def forward(self, uni_embs: torch.Tensor) -> torch.Tensor:
        # uni_embs: (B, K, D_uni)
        x = self.proj(uni_embs)     # (B, K, 64)
        pooled = self.pool(x)       # (B, 64)
        return pooled