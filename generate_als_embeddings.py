import numpy as np
import scipy
import implicit
import pickle
from pathlib import Path

import pyarrow.parquet as pq

from lib.embedder import Embedder


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'

    data = pq.read_table(
        f'{LOCAL_DATA_PATH}/{DATA_FILE}',
        columns=['user_id', 'url_host', 'request_cnt', 'price'],
    )

    data_agg = data.select(['user_id', 'url_host', 'request_cnt']).\
        group_by(['user_id', 'url_host']).aggregate([('request_cnt', 'sum')])
    item_set = set(data_agg.select(['url_host']).to_pandas()['url_host'])
    item_dict = {url: idx for idx, url in enumerate(item_set)}
    user_set = set(data_agg.select(['user_id']).to_pandas()['user_id'])
    user_dict = {user: idx for idx, user in enumerate(user_set)}
    users = np.array(data_agg.select(['user_id']).to_pandas()['user_id'].map(user_dict))
    items = np.array(data_agg.select(['url_host']).to_pandas()['url_host'].map(item_dict))
    counts = np.array(data_agg.select(['request_cnt_sum']).to_pandas()['request_cnt_sum'])
    mat = scipy.sparse.coo_matrix((counts, (users, items)), shape=(users.max() + 1, items.max() + 1))
    als = implicit.als.AlternatingLeastSquares(
        factors=100, iterations=15, calculate_training_loss=False, regularization=0.1, alpha=5, random_state=321)

    als.fit(mat)
    als = als.to_cpu()
    embedder = Embedder(item_dict, als.item_factors)

    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_PATH / 'als_items_100.pickle', 'wb') as f:
        pickle.dump(embedder, f)
