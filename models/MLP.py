import torch
import torch.nn as nn

class MLPBranch(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(MLPBranch, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),          # ✅ replaces BatchNorm1d(256)
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.LayerNorm(128),          # ✅ replaces BatchNorm1d(128)
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),  # ✅ replaces BatchNorm1d(feature_dim)
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.mlp(x)