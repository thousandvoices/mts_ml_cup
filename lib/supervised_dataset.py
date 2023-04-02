import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from .pad_tokens import pad_tokens


def collate_examples(batch):
    items, weights, unique_weights, numeric_features, categorical_features, labels, age = list(zip(*batch))
    return (
        pad_tokens(items, np.int64),
        pad_tokens(weights, np.float32),
        pad_tokens(unique_weights, np.float32),
        torch.from_numpy(np.float32(numeric_features)),
        torch.from_numpy(np.int64(categorical_features)),
        torch.from_numpy(np.float32(labels)),
        torch.from_numpy(np.int64(age))
    )


class SupervisedDataset(Dataset):
    def __init__(
            self,
            items,
            weights,
            unique_weights,
            numeric_features,
            categorical_features,
            labels,
            age,
            shuffle):
        self.items = [np.int64(item) for item in items]
        self.weights = [np.float32(weight) for weight in weights]
        self.unique_weights = [np.float32(weight) for weight in unique_weights]
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.labels = labels
        self.age = age
        self.shuffle = shuffle

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        order = np.argsort(-self.weights[idx])
        return (
            np.concatenate([np.int64([0]), self.items[idx][order]], axis=0),
            np.concatenate([np.float32([1]), self.weights[idx][order]], axis=0),
            np.concatenate([np.float32([1]), self.unique_weights[idx][order]], axis=0),
            self.numeric_features[idx],
            self.categorical_features[idx],
            self.labels[idx],
            self.age[idx]
        )

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
