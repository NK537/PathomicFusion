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
            nn.Linear(input_dim * 2, fusion_dim),
            nn.BatchNorm1d(fusion_dim),          # 🔥 Added normalization
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
        # print("CNN feat shape:", cnn_feat.shape, "MLP feat shape:", mlp_feat.shape)

        # # Ensure correct shapes
        # if cnn_feat.dim() == 1:
        #     cnn_feat = cnn_feat.unsqueeze(0)  # Add batch dimension
        # if mlp_feat.dim() == 1:
        #     mlp_feat = mlp_feat.unsqueeze(0)

        # Compute attention weights
        attn_cnn = self.attention_cnn(cnn_feat)  # (batch_size, 1)
        attn_mlp = self.attention_mlp(mlp_feat)  # (batch_size, 1)

        # # Sanity reshape if necessary
        # if attn_cnn.dim() == 1:
        #     attn_cnn = attn_cnn.unsqueeze(1)
        # if attn_mlp.dim() == 1:
        #     attn_mlp = attn_mlp.unsqueeze(1)

        # Combine attentions
        attn = torch.cat([attn_cnn, attn_mlp], dim=1)  # (batch_size, 2)
        attn = F.softmax(attn, dim=1)  # Normalize across modalities

        # Weighted sum
        weighted = attn[:, 0:1] * cnn_feat + attn[:, 1:2] * mlp_feat  # (batch_size, input_dim)

        # Concatenate both original vectors before fusion
        combined_feat = torch.cat([cnn_feat, mlp_feat], dim=1)  # (batch_size, input_dim*2)

        # Pass through fusion FC
        output = self.fusion_fc(combined_feat)  # (batch_size, 1)
        return output.squeeze(1)  # (batch_size,)
