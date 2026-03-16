import torch
import torch.nn as nn

class AttnMILPool(nn.Module):
    """
    Attention pooling over K instances.
    Input:  x (B, K, d)
    Output: pooled (B, d)
    """
    def __init__(self, d: int = 64, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attn(x)                 # (B, K, 1)
        w = torch.softmax(scores, dim=1)      # (B, K, 1)
        pooled = (w * x).sum(dim=1)           # (B, d)
        return pooled