import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pickle
import argparse
from pathlib import Path

import pyarrow.parquet as pq

from lib.dataset import PairsDataset
from lib.embedder import Embedder


def metric_criterion(item, pair):
    margin = 0.4
    scale = 25
    scores = torch.mm(item, pair.transpose(0, 1))
    mask = torch.eye(scores.size()[0]).cuda()
    scores = scale * (scores - margin * mask)
    probs = (-scores.log_softmax(1) * mask).sum(dim=1)
    reverse_probs = (-scores.log_softmax(0) * mask).sum(dim=0)

    nll = (probs + reverse_probs) / 2
    return nll.mean()


class LightningModel(pl.LightningModule):
    def __init__(self, num_items, embedding_dim, train_loader, lr, num_updates):
        super().__init__()

        self.embeddings = nn.Embedding(num_items, embedding_dim)
        self.train_loader = train_loader
        self.num_updates = num_updates

        self.lr = lr

    def forward(self, input_ids, weights):
        weights = weights.unsqueeze(2)
        embedded = self.embeddings(input_ids) * weights
        embedded = embedded.sum(1) / weights.sum(1)

        return torch.nn.functional.normalize(embedded)

    def training_step(self, batch, batch_idx):
        left, weight, right, target_weight = batch

        left = self.forward(left, weight)
        right = self.forward(right, target_weight)
        loss = metric_criterion(left, right)
        self.log('loss', loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.train_loader

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 0, self.num_updates)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'reduce_on_plateau': False,
            'frequency': 1
        }
        return [self.optimizer], [scheduler_config]


class MetricTrainer:
    def __init__(self, num_items, embedding_dim, num_epochs, batch_size):
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def fit(self, train_items, train_weights):
        grad_steps = 1
        lr = 1e-2

        train_loader = PairsDataset(train_items, train_weights, random_split=True, shuffle=True).loader(self.batch_size)

        effective_batch_size = self.batch_size * grad_steps
        num_updates = self.num_epochs * (len(train_items) + effective_batch_size - 1) // effective_batch_size

        self.model = LightningModel(
            self.num_items,
            self.embedding_dim,
            train_loader,
            lr,
            num_updates
        ).to('cuda')

        trainer = pl.Trainer(
            max_steps=num_updates,
            num_sanity_val_steps=0,
            accumulate_grad_batches=grad_steps,
            enable_checkpointing=False,
            logger=False,
            gpus=[0],
        )
        trainer.fit(self.model)

    def predict(self, items, weights):
        loader = PairsDataset(items, weights, random_split=False, shuffle=False).loader(self.batch_size)
        result = []
        with torch.inference_mode():
            for batch in loader:
                result.append(self.model(batch[0], batch[1]).detach().cpu().numpy())

        return np.concatenate(result, axis=0)


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--column', dest='column', required=True)
    parser.add_argument('--dim', dest='dim', required=True, type=int)
    args = parser.parse_args()

    item_column = args.column
    embedding_dim = args.dim

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', item_column, 'request_cnt'],
    )

    data_agg = data.select(['user_id', item_column, 'request_cnt']).\
        group_by(['user_id', item_column]).aggregate([('request_cnt', 'sum')])
    item_set = set(data_agg.select([item_column]).to_pandas()[item_column])
    item_dict = {url: idx for idx, url in enumerate(item_set)}
    user_set = set(data_agg.select(['user_id']).to_pandas()['user_id'])
    user_dict = {user: idx for idx, user in enumerate(user_set)}
    users = np.array(data_agg.select(['user_id']).to_pandas()['user_id'].map(user_dict))
    items = np.array(data_agg.select([item_column]).to_pandas()[item_column].map(item_dict))
    counts = np.array(data_agg.select(['request_cnt_sum']).to_pandas()['request_cnt_sum'])
    df = pd.DataFrame.from_dict({'user_id': users, 'items': items, 'counts': counts, 'ones': np.ones_like(counts)})
    orig_df = df.groupby('user_id').agg({'items': list, 'counts': list, 'ones': list}).reset_index()

    df = orig_df[orig_df['items'].apply(lambda x: len(x) > 1)]

    trainer = MetricTrainer(len(item_set), embedding_dim, 30, 512)
    trainer.fit(df['items'].values, df['ones'].values)

    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    item_embedder = Embedder(item_dict, trainer.model.embeddings.weight.detach().cpu().numpy())
    with open(ARTIFACTS_PATH / f'{item_column}_{embedding_dim}.pickle', 'wb') as f:
        pickle.dump(item_embedder, f)

    user_embeddings = trainer.predict(orig_df['items'], orig_df['ones'])
    user_embedder = Embedder(user_dict, user_embeddings)

    with open(ARTIFACTS_PATH / f'users_{item_column}_{embedding_dim}.pickle', 'wb') as f:
        pickle.dump(user_embedder, f)

    weighted_embeddings = trainer.predict(orig_df['items'], orig_df['counts'])
    weighted_embedder = Embedder(user_dict, weighted_embeddings)

    with open(ARTIFACTS_PATH / f'weighted_users_{item_column}_{embedding_dim}.pickle', 'wb') as f:
        pickle.dump(weighted_embedder, f)
