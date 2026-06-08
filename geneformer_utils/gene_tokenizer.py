import json
import numpy as np
import torch


def load_token_map(path):
    with open(path, "r") as f:
        return json.load(f)


class BulkGeneformerTokenizer:

    def __init__(self, gene_names, token_map, max_len=2048):

        self.gene_names = list(gene_names)
        self.token_map = token_map
        self.max_len = max_len

    def encode_vector(self, gene_values):

        gene_values = np.asarray(gene_values)

        ranked = np.argsort(-gene_values)

        tokens = []

        for idx in ranked:

            gene = self.gene_names[idx]

            if gene in self.token_map:
                tokens.append(self.token_map[gene])

            if len(tokens) >= self.max_len:
                break

        if len(tokens) == 0:
            tokens = [0]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        attention_mask = torch.ones(len(tokens), dtype=torch.long)

        return input_ids, attention_mask