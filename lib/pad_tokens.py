import torch
import numpy as np


def pad_tokens(batch, dtype):
    max_len = max(len(x) for x in batch)
    items = np.zeros((len(batch), max_len), dtype=dtype)
    for idx, item_ids in enumerate(batch):
        items[idx, :len(item_ids)] = item_ids
    return torch.from_numpy(items)
