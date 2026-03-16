import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorTokenizer(nn.Module):
    """
    Turns a single vector (B, d) into tokens (B, T, d) so cross-attention is meaningful.
    """
    def __init__(self, d: int = 64, n_tokens: int = 8):
        super().__init__()
        self.d = d
        self.n_tokens = n_tokens
        self.proj = nn.Linear(d, n_tokens * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d) -> (B, T, d)
        B, d = x.shape
        out = self.proj(x).view(B, self.n_tokens, d)
        return out


class CrossAttentionFusion(nn.Module):
    """
    Drop-in replacement for AttentionFusion:
      forward(cnn_feat, mlp_feat) -> risk score (B,)

    cnn_feat: (B, 64)
    mlp_feat: (B, 64)
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_gene_tokens: int = 8, fusion_dim: int = 128):
        super().__init__()

        # Tokenize both modalities into sequences of tokens
        self.cnn_tok = VectorTokenizer(d=d_model, n_tokens=n_gene_tokens)
        self.mlp_tok = VectorTokenizer(d=d_model, n_tokens=n_gene_tokens)

        # Cross-attention blocks (bidirectional)
        self.cnn_to_mlp = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp_to_cnn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.norm_cnn_1 = nn.LayerNorm(d_model)
        self.norm_mlp_1 = nn.LayerNorm(d_model)

        # Lightweight FFNs (Transformer style)
        self.ff_cnn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )
        self.ff_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm_cnn_2 = nn.LayerNorm(d_model)
        self.norm_mlp_2 = nn.LayerNorm(d_model)

        # Final fusion head (keeps same style as your AttentionFusion)
        # We pool token sequences -> (B, d) + (B, d) -> concat -> (B, 2d) then head -> (B,1)
        self.fusion_fc = nn.Sequential(
            nn.Linear(2 * d_model, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, cnn_feat: torch.Tensor, mlp_feat: torch.Tensor) -> torch.Tensor:
        """
        cnn_feat: (B, d)
        mlp_feat: (B, d)
        returns: (B,)
        """
        # Tokenize
        cnn_tokens = self.cnn_tok(cnn_feat)  # (B, T, d)
        mlp_tokens = self.mlp_tok(mlp_feat)  # (B, T, d)

        # CNN attends to MLP (image queries gene)
        cnn_att, _ = self.cnn_to_mlp(query=cnn_tokens, key=mlp_tokens, value=mlp_tokens)
        cnn_tokens = self.norm_cnn_1(cnn_tokens + cnn_att)
        cnn_tokens = self.norm_cnn_2(cnn_tokens + self.ff_cnn(cnn_tokens))

        # MLP attends to CNN (gene queries image)
        mlp_att, _ = self.mlp_to_cnn(query=mlp_tokens, key=cnn_tokens, value=cnn_tokens)
        mlp_tokens = self.norm_mlp_1(mlp_tokens + mlp_att)
        mlp_tokens = self.norm_mlp_2(mlp_tokens + self.ff_mlp(mlp_tokens))

        # Pool tokens -> vectors
        cnn_vec = cnn_tokens.mean(dim=1)  # (B, d)
        mlp_vec = mlp_tokens.mean(dim=1)  # (B, d)

        fused = torch.cat([cnn_vec, mlp_vec], dim=1)  # (B, 2d)
        out = self.fusion_fc(fused).squeeze(1)  # (B,)
        return out
