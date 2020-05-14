#!/bin/bash

NETID=sk7685
export TRAIN_FILE=/data/${NETID}/treebank_3/concat/ptb_train_v2.txt
export TEST_FILE=/data/${NETID}/treebank_3/concat/ptb_test_v2.txt

TSEED=0

python run_language_modeling_hsfmx.py \
    --output_dir=ptb_hsfmx_seed${TSEED} \
    --model_type=bert --model_name_or_path=bert-base-uncased \
    --do_train --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --seed $TSEED \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --overwrite_output_dir \
    --hsfmx \
    --trees_path="../../utils/build_trees/hsfmx_trees_most_freq/*" \
    --epochs_train_hsfmx 8 \
    --idx_mapping_path "../../utils/most_frequent_indices/most_frequent_indices_dict.pkl"
