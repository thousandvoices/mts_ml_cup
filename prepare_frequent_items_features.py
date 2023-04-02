import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'
    CATEGORICAL_FEATURES = ['city_name', 'url_host']

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', 'request_cnt'] + CATEGORICAL_FEATURES
    )

    frequency_features = pd.DataFrame.from_dict({
        'user_id': data.select(['user_id']).to_pandas()['user_id'].unique()
    })
    frequency_features = frequency_features.sort_values('user_id')

    for feature in CATEGORICAL_FEATURES:
        grouped = data.group_by(['user_id', feature]).aggregate([('request_cnt', 'sum')]).to_pandas()
        grouped['frequency'] = grouped.apply(lambda row: (row[feature], row['request_cnt_sum']), axis=1)
        grouped = grouped.groupby('user_id').agg({
            'frequency': lambda x: sorted(list(x), key=lambda item: -item[1])
        }).reset_index()

        for i in range(5):
            grouped[f'{feature}_{i}'] = grouped.frequency.apply(lambda x: x[i][0] if i < len(x) else '')

        grouped = grouped.drop(['frequency'], axis='columns')
        frequency_features = frequency_features.merge(grouped, on='user_id', how='left')

    frequency_features.to_csv(ARTIFACTS_PATH / 'frequent_items.csv', index=False)
