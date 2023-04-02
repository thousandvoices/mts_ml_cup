import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
import numpy as np
import pickle
import argparse
from pathlib import Path


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--num-clusters', dest='num_clusters', required=True, type=int)
    parser.add_argument('--embedder', dest='embedder', required=True)
    args = parser.parse_args()
    num_clusters = args.num_clusters
    normalize = args.normalize
    embedder_path = args.embedder

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', 'request_cnt', 'url_host']
    )

    with open(embedder_path, 'rb') as f:
        embedder = pickle.load(f)

    data_agg = data.group_by(['user_id', 'url_host']).aggregate([('request_cnt', 'sum')])
    data_agg = data_agg.to_pandas()
    data_agg['url_id'] = data_agg.url_host.map(embedder.item_to_id)

    items = embedder.embeddings
    if normalize:
        items /= np.linalg.norm(items, axis=-1, keepdims=True) + 1e-10

    clusters = KMeans(n_clusters=num_clusters).fit_predict(items)
    items_df = pd.DataFrame.from_dict({'url_id': range(items.shape[0]), 'cluster_id': clusters})
    data_agg = data_agg.merge(items_df, on='url_id', how='left')
    data_agg = data_agg.drop('url_host', axis='columns')
    data_agg = data_agg.groupby(['user_id', 'cluster_id']).agg({'request_cnt_sum': 'sum'}).reset_index()
    data_agg['cluster_id_sum'] = data_agg.apply(lambda row: (row.cluster_id, row.request_cnt_sum), axis=1)
    data_agg = data_agg.groupby('user_id').agg({'cluster_id_sum': lambda x: {k: v for k, v in x}}).reset_index()

    name_prefix = 'normal_' if normalize else ''
    for cluster_id in range(num_clusters):
        column_name = f'fraction_{name_prefix}cluster_{num_clusters}_{cluster_id}'
        data_agg[column_name] = data_agg.cluster_id_sum.apply(lambda x: x.get(cluster_id, 0) / sum(x.values()))

    data_agg = data_agg.drop('cluster_id_sum', axis='columns')
    data_agg.to_csv(ARTIFACTS_PATH / f'{name_prefix}clusters_{num_clusters}.csv', index=False)
