import torch.nn as nn

class FoundationBranch(nn.Module):
    """
    Input: (B, D_foundation) patch embedding
    Output: (B, 64)
    """
    def __init__(self, in_dim: int, feature_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
        )

    def forward(self, x):
        return self.proj(x)
