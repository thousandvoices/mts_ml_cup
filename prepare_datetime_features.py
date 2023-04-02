import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


def create_fraction_features(df, column):
    grouped = df.groupby(['user_id', column]).agg({'request_cnt': 'sum'}).reset_index()
    grouped['value_sum'] = grouped.apply(lambda row: (row[column], row.request_cnt), axis=1)
    user_grouped = grouped.groupby('user_id').agg({'value_sum': lambda x: {k: v for k, v in x}}).reset_index()
    for value in grouped[column].unique():
        user_grouped[f'fraction_{value}'] = user_grouped.value_sum.apply(lambda x: x.get(value, 0) / sum(x.values()))

    return user_grouped.drop('value_sum', axis='columns')


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'
    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', 'request_cnt', 'part_of_day', 'date']
    ).to_pandas()

    data['date'] = pd.to_datetime(data.date).dt.dayofweek

    daily_features = create_fraction_features(data, 'date')
    part_of_day_features = create_fraction_features(data, 'part_of_day')

    datetime_features = daily_features.merge(part_of_day_features, on='user_id', how='left')
    datetime_features.to_csv(ARTIFACTS_PATH / 'datetime.csv', index=False)
