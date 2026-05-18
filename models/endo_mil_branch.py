import torch
import torch.nn as nn
from models.mil_pooling import AttnMILPool


class EndoMILBranch(nn.Module):
    """
    Endoscopy MIL branch.

    Input  : pre-computed GastroNet-5M CLS embeddings  (B, K, endo_dim)
             endo_dim = 384  (ViT-small/16 CLS token)
    Output : patient-level representation               (B, out_dim)

    Architecture:
        Linear projection  endo_dim -> 256 -> out_dim
        Attention MIL pool over K frames
    """
    def __init__(self, endo_dim: int = 384, out_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(endo_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim),
        )
        self.pool = AttnMILPool(d=out_dim, hidden=128, dropout=0.1)

    def forward(self, endo_embs: torch.Tensor) -> torch.Tensor:
        # endo_embs: (B, K, endo_dim)
        x      = self.proj(endo_embs)   # (B, K, out_dim)
        pooled = self.pool(x)           # (B, out_dim)
        return pooled
