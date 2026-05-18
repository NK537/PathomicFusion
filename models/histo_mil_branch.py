import torch
import torch.nn as nn
from models.mil_pooling import AttnMILPool


class HistoMILBranch(nn.Module):
    """
    Histopathology MIL branch.

    Input  : pre-computed patch embeddings  (B, K, histo_dim)
             histo_dim = 1024 for UNI / check Spatiopath paper for its value
    Output : patient-level representation   (B, out_dim)

    Architecture:
        Linear projection  histo_dim -> 256 -> out_dim
        Attention MIL pool over K patches
    """
    def __init__(self, histo_dim: int = 1024, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(histo_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )
        self.pool = AttnMILPool(d=out_dim, hidden=128, dropout=0.1)

    def forward(self, histo_embs: torch.Tensor) -> torch.Tensor:
        # histo_embs: (B, K, histo_dim)
        x      = self.proj(histo_embs)   # (B, K, out_dim)
        pooled = self.pool(x)            # (B, out_dim)
        return pooled
