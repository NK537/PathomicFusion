import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, input_dim=64, fusion_dim=128):
        """
        Attention-based fusion module for CNN and MLP feature vectors.

        Args:
            input_dim (int): Dimension of each input feature vector (default: 32).
            fusion_dim (int): Size of the fusion output layer (default: 64).
        """
        super(AttentionFusion, self).__init__()

        # Attention score generators
        self.attention_cnn = nn.Linear(input_dim, 1)
        self.attention_mlp = nn.Linear(input_dim, 1)

        # Fusion layer (final classifier head)
        self.fusion_fc = nn.Sequential(
            nn.Linear(input_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),          # 🔥 Added normalization
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 1)  # Survival risk score
        )

    def forward(self, cnn_feat, mlp_feat):
        """
        Args:
            cnn_feat (Tensor): CNN output (batch_size x input_dim)
            mlp_feat (Tensor): MLP output (batch_size x input_dim)

        Returns:
            Tensor: Predicted survival risk scores (batch_size,)
        """

        # Compute attention weights
        attn_cnn = self.attention_cnn(cnn_feat)  # (batch_size, 1)
        attn_mlp = self.attention_mlp(mlp_feat)  # (batch_size, 1)

        # Combine attentions
        attn = torch.cat([attn_cnn, attn_mlp], dim=1)  # (batch_size, 2)
        attn = F.softmax(attn, dim=1)  # Normalize across modalities

        # Weighted sum
        weighted = attn[:, 0:1] * cnn_feat + attn[:, 1:2] * mlp_feat  # (batch_size, input_dim)

        # Concatenate both original vectors before fusion
        combined_feat = torch.cat([cnn_feat, mlp_feat], dim=1)  # (batch_size, input_dim*2)

        final = torch.cat([combined_feat, weighted], dim=1)  
        # Pass through fusion FC
        output = self.fusion_fc(final)  # (batch_size, 1)
        return output.squeeze(1)  # (batch_size,)
