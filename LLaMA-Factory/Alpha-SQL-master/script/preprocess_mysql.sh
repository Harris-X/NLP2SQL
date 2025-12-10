#! /bin/bash

python -m alphasql.runner.preprocessor \
    --data_file_path "mini_dev_mysql_alpaca_filtered.json" \
    --database_root_dir "dummy_dir" \
    --save_root_dir "data/preprocessed/mysql_test" \
    --lsh_threshold 0.5 \
    --lsh_signature_size 128 \
    --lsh_n_gram 3 \
    --lsh_top_k 20 \
    --edit_similarity_threshold 0.5 \
    --embedding_similarity_threshold 0.5 \
    --n_parallel_processes 1 \
    --max_dataset_samples -1
