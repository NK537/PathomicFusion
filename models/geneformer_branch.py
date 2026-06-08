import torch
import torch.nn as nn
from transformers import BertModel


class GeneformerBranch(nn.Module):

    def __init__(self, out_dim=64, freeze_backbone=True):
        super().__init__()

        self.backbone = BertModel.from_pretrained("ctheodoris/Geneformer")

        hidden = self.backbone.config.hidden_size

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim)
        )

    def forward(self, input_ids, attention_mask):

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = outputs.last_hidden_state[:, 0]

        return self.proj(pooled)