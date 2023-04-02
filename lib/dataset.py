import numpy as np
import random

from torch.utils.data import Dataset, DataLoader

from .pad_tokens import pad_tokens


def collate_examples(batch):
    train_items, train_weights, test_items, test_weights = list(zip(*batch))
    return (
        pad_tokens(train_items, np.int64),
        pad_tokens(train_weights, np.float32),
        pad_tokens(test_items, np.int64),
        pad_tokens(test_weights, np.float32)
    )


class PairsDataset(Dataset):
    def __init__(self, items, weights, random_split, shuffle):
        self.items = [np.int64(item) for item in items]
        self.weights = [np.float32(weight) for weight in weights]
        self.random_split = random_split
        self.shuffle = shuffle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        items = self.items[idx]
        weights = self.weights[idx]

        if self.random_split:
            pivot = random.randint(1, len(items) - 1)
            indices = np.arange(len(items), dtype=np.int32)
            np.random.shuffle(indices)
            items = items[indices]
            weights = weights[indices]
        else:
            pivot = len(items)

        return items[:pivot], weights[:pivot], items[pivot:], weights[pivot:]

    def loader(self, batch_size):
        data_loader = DataLoader(
            self,
            collate_fn=collate_examples,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=self.shuffle
        )

        return data_loader
