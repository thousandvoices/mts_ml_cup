import pyarrow.parquet as pq
from statistics import mode
from pathlib import Path


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'
    CATEGORICAL_FEATURES = ['region_name', 'cpe_model_name', 'cpe_manufacturer_name', 'part_of_day', 'cpe_type_cd']

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', 'request_cnt'] + CATEGORICAL_FEATURES,
        read_dictionary=CATEGORICAL_FEATURES
    )

    grouped = data.select(['user_id'] + CATEGORICAL_FEATURES).to_pandas()
    grouped = grouped.groupby('user_id').aggregate({f: lambda x: mode(x) for f in CATEGORICAL_FEATURES})
    grouped = grouped.reset_index()
    grouped.to_csv(ARTIFACTS_PATH / 'categorical.csv', index=False)
