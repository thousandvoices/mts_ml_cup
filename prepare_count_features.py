import pyarrow.parquet as pq
from pathlib import Path


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'
    CATEGORICAL_FEATURES = ['region_name', 'city_name']

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id'] + CATEGORICAL_FEATURES,
        read_dictionary=CATEGORICAL_FEATURES
    ).to_pandas()

    grouped = data.groupby('user_id').agg({f: 'nunique' for f in CATEGORICAL_FEATURES}).reset_index()
    grouped = grouped.rename({f: f'unique_{f}' for f in CATEGORICAL_FEATURES}, axis='columns')
    grouped.to_csv(ARTIFACTS_PATH / 'categorical_counts.csv', index=False)
