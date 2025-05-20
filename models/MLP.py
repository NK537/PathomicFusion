import torch
import torch.nn as nn

class MLPBranch(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(MLPBranch, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),          # Corrected BatchNorm
            nn.LeakyReLU(0.1),            # Improved activation
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),          # Added BatchNorm here
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),  # Final feature normalization for fusion
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.mlp(x)