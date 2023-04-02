import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from scipy.special import logit, softmax
import bisect
import sklearn.metrics as m
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pathlib import Path


def age_bucket(x):
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], x)


def add_prefix(df, prefix):
    return df.rename({c: f'{prefix}_{c}' for c in df.columns if c != 'user_id'}, axis='columns')


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'
    TARGET_FILE = 'public_train.pqt'
    SPLIT_SEED = 42
    SUBMISSION_FILE = 'submit_2.pqt'

    id_to_submit = pq.read_table(LOCAL_DATA_PATH / SUBMISSION_FILE).to_pandas()
    nn_df = add_prefix(pd.read_csv(ARTIFACTS_PATH / 'trainable.csv'), 'nn')
    frozen_df = add_prefix(pd.read_csv(ARTIFACTS_PATH / 'frozen_100.csv'), 'frozen')
    frozen_150_df = add_prefix(pd.read_csv(ARTIFACTS_PATH / 'frozen_150.csv'), 'frozen_150')

    frozen_df = frozen_df.merge(frozen_150_df, on='user_id', how='left')
    frozen_df['frozen_is_male'] = 0.5 * (frozen_df['frozen_is_male'] + frozen_df['frozen_150_is_male'])

    for i in range(7):
        frozen_df[f'frozen_age_{i}'] = 0.5 * (frozen_df[f'frozen_age_{i}'] + frozen_df[f'frozen_150_age_{i}'])
    lgbm_df = add_prefix(pd.read_csv(ARTIFACTS_PATH / 'lgbm.csv'), 'lgbm')
    print(lgbm_df.head())

    targets = pq.read_table(LOCAL_DATA_PATH / TARGET_FILE).to_pandas()
    targets = targets[targets['is_male'] != 'NA']
    targets = targets.dropna()
    targets['is_male'] = targets['is_male'].map(int)
    targets['age'] = targets['age'].map(age_bucket)

    val_users = train_test_split(targets[['user_id']], test_size=0.1, random_state=SPLIT_SEED)[1]
    y_nn = val_users.merge(nn_df, on='user_id', how='left')['nn_is_male']
    y_frozen = val_users.merge(frozen_df, on='user_id', how='left')['frozen_is_male']
    y_lgbm = val_users.merge(lgbm_df, on='user_id', how='left')['lgbm_is_male']

    y_test = train_test_split(targets[['is_male']], test_size=0.1, random_state=SPLIT_SEED)[1]

    print(f'GINI по полу {2 * m.roc_auc_score(y_test, y_nn) - 1:2.4f}')
    print(f'GINI по полу {2 * m.roc_auc_score(y_test, y_frozen) - 1:2.4f}')
    print(f'GINI по полу {2 * m.roc_auc_score(y_test, y_lgbm) - 1:2.4f}')

    all_data = logit(np.stack([y_nn, y_frozen, y_lgbm], axis=1))

    train_data, test_data, train_target, test_target = train_test_split(
        all_data, y_test, test_size=0.5, random_state=42)

    stacker = LogisticRegression()
    stacker.fit(train_data, train_target)
    test_predicted = stacker.predict_proba(test_data)[:, 1]
    print(stacker.coef_)

    print(f'GINI stacked {2 * m.roc_auc_score(test_target, test_predicted) - 1:2.5f}')

    test_df = id_to_submit.merge(nn_df, on='user_id', how='left')
    test_df = test_df.merge(frozen_df, on='user_id', how='left')
    test_df = test_df.merge(lgbm_df, on='user_id', how='left')

    all_predictions = logit(np.stack(
        [test_df.nn_is_male.values, test_df.frozen_is_male.values, test_df.lgbm_is_male.values], axis=1))

    submission = id_to_submit.copy()
    submission['is_male'] = stacker.predict_proba(all_predictions)[:, 1]

    y_nn = val_users.merge(nn_df, on='user_id', how='left')[[f'nn_age_{i}' for i in range(7)]].values
    y_frozen = val_users.merge(frozen_df, on='user_id', how='left')[[f'frozen_age_{i}' for i in range(7)]].values
    y_lgbm = val_users.merge(lgbm_df, on='user_id', how='left')[[f'lgbm_age_{i}' for i in range(7)]].values

    y_test = train_test_split(targets[['age']], test_size=0.1, random_state=SPLIT_SEED)[1]

    weights = np.float32([[1, 3, 0.8, 1, 3, 4, 10]])

    blended = np.argmax(softmax(0.35 * logit(y_lgbm) + 0.6 * y_nn + 2.2 * logit(y_frozen)) * weights, axis=-1)
    print(m.classification_report(
        y_test, blended, target_names=['<18', '18-25', '25-34', '35-44', '45-54', '55-65', '65+'], digits=5))

    test_df = id_to_submit[['user_id']]
    columns = [f'nn_age_{i}' for i in range(7)]
    test_df = test_df.merge(nn_df, on='user_id', how='left')
    frozen_columns = [f'frozen_age_{i}' for i in range(7)]
    test_df = test_df.merge(frozen_df, on='user_id', how='left')
    lgbm_columns = [f'lgbm_age_{i}' for i in range(7)]
    test_df = test_df.merge(lgbm_df, on='user_id', how='left')

    submission['age'] = np.argmax(
        softmax(
            0.35 * logit(test_df[lgbm_columns].values) +
            0.6 * logit(test_df[columns].values) +
            2.2 * logit(test_df[frozen_columns].values)
        ) * weights,
        axis=-1
    )

    submission.to_csv(ARTIFACTS_PATH / 'submission.csv', index=False)
