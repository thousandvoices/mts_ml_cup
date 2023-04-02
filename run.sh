set -e

python3 generate_als_embeddings.py
python3 generate_embeddings.py --column city_name --dim 16
python3 generate_embeddings.py --column url_host --dim 96
python3 generate_embeddings.py --column url_host --dim 100
python3 generate_embeddings.py --column url_host --dim 150

python3 prepare_categorical_features.py && echo "Categorical features generated"
python3 prepare_datetime_features.py && echo "Datetime features generated"

python3 train_nn.py --embedder artifacts/url_host_100.pickle --output-path artifacts/frozen_100.csv --freeze-embeddings
python3 train_nn.py --embedder artifacts/url_host_150.pickle --output-path artifacts/frozen_150.csv --freeze-embeddings
python3 train_nn.py --embedder artifacts/url_host_96.pickle --output-path artifacts/trainable.csv

python3 prepare_count_features.py && echo "Count features generated"
python3 prepare_frequent_items_features.py && echo "Frequent items features generated"
python3 prepare_request_frequency_features.py && echo "Frequency features generated"

python3 prepare_cluster_features.py --embedder artifacts/als_items_100.pickle --num-clusters 256
python3 prepare_cluster_features.py --embedder artifacts/url_host_96.pickle --num-clusters 192 --normalize

python3 train_lgbm.py
python3 combine_predictions.py
