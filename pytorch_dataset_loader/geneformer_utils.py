import torch

def pad_geneformer_batch(token_dicts, pad_token_id=0):
    max_len = max(x["input_ids"].shape[0] for x in token_dicts)

    input_ids = []
    attention_masks = []

    for x in token_dicts:
        ids = x["input_ids"]
        mask = x["attention_mask"]

        pad_len = max_len - ids.shape[0]

        if pad_len > 0:
            ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)])

        input_ids.append(ids)
        attention_masks.append(mask)

    return torch.stack(input_ids), torch.stack(attention_masks)