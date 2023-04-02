import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import scipy
import implicit
import bisect
import sklearn.metrics as m
import pickle
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.special import logit, softmax
import gc
from pathlib import Path

from lib.target_encoder import TargetEncoder


if __name__ == '__main__':
    LOCAL_DATA_PATH = Path('data')
    ARTIFACTS_PATH = Path('artifacts')
    DATA_FILE = 'competition_data_final_pqt'
    TARGET_FILE = 'public_train.pqt'
    SPLIT_SEED = 42
    CATEGORICAL_FEATURES = ['region_name', 'cpe_model_name', 'cpe_manufacturer_name', 'part_of_day', 'cpe_type_cd']

    data = pq.read_table(
        LOCAL_DATA_PATH / DATA_FILE,
        columns=['user_id', 'url_host', 'request_cnt', 'price']
    )

    agg_users = data.select(['user_id', 'price']).group_by('user_id').aggregate([('price', 'mean')]).to_pandas()
    agg_users['price_mean'] = agg_users['price_mean'].fillna(0.0)

    targets = pq.read_table(LOCAL_DATA_PATH / TARGET_FILE)
    common_hosts = pd.read_csv(ARTIFACTS_PATH / 'frequent_items.csv').fillna('')
    CUSTOM_CATEGORICAL_FEATURES = [c for c in common_hosts.columns if c != 'user_id']

    features = common_hosts.merge(pd.read_csv(ARTIFACTS_PATH / 'categorical.csv'), on='user_id', how='left')
    features = features.merge(agg_users, on='user_id', how='left')
    features = features.merge(pd.read_csv(ARTIFACTS_PATH / 'categorical_counts.csv'), on='user_id', how='left')
    features = features.merge(pd.read_csv(ARTIFACTS_PATH / 'datetime.csv'), on='user_id', how='left')
    features = features.merge(pd.read_csv(ARTIFACTS_PATH / 'normal_clusters_192.csv'), on='user_id', how='left')
    features = features.merge(pd.read_csv(ARTIFACTS_PATH / 'clusters_256.csv'), on='user_id', how='left')
    features = features.merge(pd.read_csv(ARTIFACTS_PATH / 'request_frequency.csv'), on='user_id', how='left')

    data_agg = data.select(['user_id', 'url_host', 'request_cnt']).\
        group_by(['user_id', 'url_host']).aggregate([('request_cnt', 'sum')])

    url_set = set(data_agg.select(['url_host']).to_pandas()['url_host'])
    url_dict = {url: idx for idx, url in enumerate(url_set)}
    user_set = set(data_agg.select(['user_id']).to_pandas()['user_id'])
    user_dict = {user: idx for idx, user in enumerate(user_set)}

    values = np.array(data_agg.select(['request_cnt_sum']).to_pandas()['request_cnt_sum'])
    rows = np.array(data_agg.select(['user_id']).to_pandas()['user_id'].map(user_dict))
    cols = np.array(data_agg.select(['url_host']).to_pandas()['url_host'].map(url_dict))
    mat = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(rows.max() + 1, cols.max() + 1))
    als = implicit.als.AlternatingLeastSquares(
        factors=150, iterations=15, calculate_training_loss=False, regularization=0.1, alpha=5, random_state=321)

    del data_agg
    del data
    gc.collect()

    als.fit(mat)
    als = als.to_cpu()
    u_factors = als.user_factors
    d_factors = als.item_factors

    with open(ARTIFACTS_PATH / 'users_url_host_96.pickle', 'rb') as f:
        embedder = pickle.load(f)

    with open(ARTIFACTS_PATH / 'users_url_host_150.pickle', 'rb') as f:
        embedder_150 = pickle.load(f)

    with open(ARTIFACTS_PATH / 'weighted_users_url_host_100.pickle', 'rb') as f:
        weighted_embedder = pickle.load(f)

    with open(ARTIFACTS_PATH / 'users_city_name_16.pickle', 'rb') as f:
        city_embedder = pickle.load(f)

    u_factors = np.concatenate([
        weighted_embedder.embeddings,
        embedder_150.embeddings,
        embedder.embeddings,
        als.user_factors,
        city_embedder.embeddings], axis=1)

    inv_usr_map = {v: k for k, v in user_dict.items()}
    usr_emb = pd.DataFrame(u_factors)
    usr_emb['user_id'] = usr_emb.index.map(inv_usr_map)
    usr_targets = targets.to_pandas()
    df = usr_targets.merge(usr_emb, how='inner', on=['user_id'])
    df = df[df['is_male'] != 'NA']
    df = df.dropna()
    df['is_male'] = df['is_male'].map(int)

    df = df.merge(features, how='left', on='user_id')

    ALL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + CUSTOM_CATEGORICAL_FEATURES

    x_train, x_test, y_train, y_test = train_test_split(
         df.drop(['user_id', 'age', 'is_male'], axis=1), df['is_male'], test_size=0.1, random_state=SPLIT_SEED)

    target_encoders = {}
    for f in ALL_CATEGORICAL_FEATURES:
        encoder = TargetEncoder()
        x_train[f] = encoder.fit_transform(x_train[f], y_train)
        x_test[f] = encoder.transform(x_test[f])
        target_encoders[f] = encoder

    clf = LGBMClassifier(
        objective='binary',
        learning_rate=0.02,
        num_leaves=63,
        colsample_bytree=0.6,
        subsample=0.8,
        reg_alpha=1,
        reg_lambda=8,
        importance_type='gain',
        n_estimators=8000
    )
    clf.fit(x_train, y_train, eval_set=(x_test, y_test), eval_metric='auc', verbose=100)

    best_indices = np.argsort(clf.feature_importances_)[-20:]
    for idx in best_indices:
        print(clf.feature_importances_[idx], list(x_train.columns)[idx])

    test_df = usr_emb
    test_df = test_df.merge(features, how='left', on='user_id')

    for f in ALL_CATEGORICAL_FEATURES:
        test_df[f] = target_encoders[f].transform(test_df[f])

    lgbm_prediction = clf.predict_proba(test_df.drop(['user_id'], axis=1))[:, 1]
    submission = test_df[['user_id']].copy()
    submission['is_male'] = lgbm_prediction

    df = usr_targets.merge(usr_emb, how='inner', on=['user_id'])
    df = df[df['is_male'] != 'NA']
    df = df.dropna()
    df['age'] = df['age'].apply(lambda age: bisect.bisect_left([18, 25, 35, 45, 55, 65], age))
    df = df.merge(features, how='left', on=['user_id'])

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(['user_id', 'age', 'is_male'], axis=1), df['age'], test_size=0.1, random_state=SPLIT_SEED)

    n_values = np.max(y_train) + 1
    one_hot_age = np.eye(n_values)[y_train]

    target_encoders = {}
    for f in ALL_CATEGORICAL_FEATURES:
        for i in range(n_values):
            encoder = TargetEncoder()
            feature_name = f'{f}_{i}'
            x_train[feature_name] = encoder.fit_transform(x_train[f], one_hot_age[:, i])
            x_test[feature_name] = encoder.transform(x_test[f])
            target_encoders[feature_name] = encoder

    x_train = x_train.drop(ALL_CATEGORICAL_FEATURES, axis=1)
    x_test = x_test.drop(ALL_CATEGORICAL_FEATURES, axis=1)

    age_classifier = LGBMClassifier(
        objective='multiclass',
        learning_rate=0.015,
        colsample_bytree=0.6,
        subsample=0.8,
        reg_alpha=1,
        reg_lambda=8,
        importance_type='gain',
        n_estimators=4500
    )
    age_classifier.fit(x_train, y_train, eval_set=(x_test, y_test), eval_metric='multi_error', verbose=100)

    best_indices = np.argsort(age_classifier.feature_importances_)[-20:]
    for idx in best_indices:
        print(age_classifier.feature_importances_[idx], list(x_train.columns)[idx])

    embedding_features = [i for i in range(u_factors.shape[-1])]
    logistic_model = LogisticRegression(solver='sag')
    logistic_model.fit(x_train[embedding_features], y_train)

    y_pred = logit(age_classifier.predict_proba(x_test))
    y_pred += 0.4 * logit(logistic_model.predict_proba(x_test[embedding_features]))
    weights = np.float32([[1, 3, 0.8, 1, 3, 4, 10]])

    blended = np.argmax(softmax(y_pred) * weights, axis=-1)
    report = m.classification_report(
        y_test,
        blended,
        target_names=['<18', '18-25', '25-34', '35-44', '45-54', '55-65', '65+'], digits=5)
    print(report)

    test_df = usr_emb
    test_df = test_df.merge(features, how='left', on='user_id')

    for f in ALL_CATEGORICAL_FEATURES:
        for i in range(n_values):
            feature_name = f'{f}_{i}'
            encoder = target_encoders[feature_name]
            test_df[feature_name] = encoder.transform(test_df[f])

    test_df = test_df.drop(ALL_CATEGORICAL_FEATURES, axis=1)
    lgbm_logits = logit(age_classifier.predict_proba(test_df.drop(['user_id'], axis=1)))
    logistic_logits = logit(logistic_model.predict_proba(test_df[embedding_features]))
    prediction = softmax(lgbm_logits + 0.4 * logistic_logits, axis=1)
    for i in range(7):
        submission[f'age_{i}'] = prediction[:, i]

    submission.to_csv(ARTIFACTS_PATH / 'lgbm.csv', index=False)
