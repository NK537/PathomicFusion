import torch
from torch.nn.utils.rnn import pad_sequence


def geneformer_collate_fn(batch):
    uni_embs, input_ids, attention_masks, surv, event, grade, pid = zip(*batch)

    uni_embs = torch.stack(uni_embs, dim=0)  # (B, K, D)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    surv = torch.stack(surv)
    event = torch.stack(event)
    grade = torch.stack(grade)

    return uni_embs, input_ids, attention_masks, surv, event, grade, list(pid)