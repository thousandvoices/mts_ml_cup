import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', 'request_cnt', 'date']
    )

    data_agg = data.group_by(['user_id']).aggregate([
        ('request_cnt', 'sum'),
        ('request_cnt', 'count'),
        ('date', 'count_distinct'),
        ('date', 'min'),
        ('date', 'max')
    ])
    data_agg = data_agg.to_pandas()
    data_agg['date_min'] = pd.to_datetime(data_agg['date_min'])
    data_agg['date_max'] = pd.to_datetime(data_agg['date_max'])
    data_agg['all_days'] = (data_agg.date_max - data_agg.date_min).apply(lambda x: x.days + 1)
    data_agg['requests_per_day'] = data_agg['request_cnt_sum'] / data_agg['date_count_distinct']
    data_agg['requests_per_day_missing'] = data_agg['request_cnt_sum'] / data_agg['all_days']
    data_agg['unique_requests_per_day'] = data_agg['request_cnt_count'] / data_agg['date_count_distinct']
    data_agg['unique_requests_per_day_missing'] = data_agg['request_cnt_count'] / data_agg['all_days']
    data_agg['active_days_fraction'] = data_agg['date_count_distinct'] / data_agg['all_days']
    data_agg = data_agg[[
        'user_id',
        'requests_per_day',
        'requests_per_day_missing',
        'unique_requests_per_day',
        'unique_requests_per_day_missing',
        'active_days_fraction',
        'all_days'
    ]]
    data_agg.to_csv(ARTIFACTS_PATH / 'request_frequency.csv', index=False)
