# models/cell_gnn_branch.py
#
# Graph Attention Network (GAT) branch for cell-level survival prediction.
#
# Input : batched cell graph  (node features + edge_index + batch vector)
# Output: patient-level representation  (B, out_dim)
#
# Requires: pip install torch-geometric

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


class CellGNNBranch(nn.Module):
    """
    2-layer Graph Attention Network with global pooling.

    Args:
        in_dim    : node feature dim  (cell embedding dim, e.g. 1024)
        hidden    : hidden dim per attention head
        heads     : number of GAT attention heads
        n_layers  : number of GAT layers (2 or 3)
        out_dim   : patient-level representation dim
        dropout   : dropout rate
    """

    def __init__(
        self,
        in_dim:   int = 1024,
        hidden:   int = 256,
        heads:    int = 4,
        n_layers: int = 2,
        out_dim:  int = 64,
        dropout:  float = 0.2,
    ):
        super().__init__()
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "torch-geometric not installed.\n"
                "Run: pip install torch-geometric"
            )

        self.dropout  = dropout
        self.n_layers = n_layers

        # Input projection: reduce high-dim embeddings before GNN
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden * heads),
            nn.LayerNorm(hidden * heads),
            nn.ReLU(),
        )

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_ch  = hidden * heads   # all layers receive (hidden*heads)-dim input
            out_ch = hidden
            # Last layer: single head, no concat → (hidden,)
            n_heads = heads if i < n_layers - 1 else 1
            concat  = (i < n_layers - 1)
            self.convs.append(
                GATConv(in_ch, out_ch, heads=n_heads, concat=concat, dropout=dropout)
            )
            out_after = out_ch * n_heads if concat else out_ch
            self.norms.append(nn.LayerNorm(out_after))

        # Patient-level projection after pooling
        pool_in = out_ch * 2   # mean + max pooling concatenated
        self.output_proj = nn.Sequential(
            nn.Linear(pool_in, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index, batch):
        """
        Args:
            x          : (N_total_cells, in_dim)
            edge_index : (2, E)
            batch      : (N_total_cells,)  graph assignment per node

        Returns:
            (B, out_dim)  patient-level representations
        """
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Pool: concat mean + max for richer aggregation
        x_mean = global_mean_pool(x, batch)   # (B, hidden)
        x_max  = global_max_pool(x, batch)    # (B, hidden)
        x_pool = torch.cat([x_mean, x_max], dim=-1)   # (B, hidden*2)

        return self.output_proj(x_pool)   # (B, out_dim)
