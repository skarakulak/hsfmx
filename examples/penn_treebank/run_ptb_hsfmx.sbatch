#!/bin/bash
#
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --time=22:00:00
#SBATCH --job-name="hsfmx"
#SBATCH --output=logs/train_MPUR_%j.out
#SBATCH --error=logs/train_MPUR_%j.err
#SBATCH --exclude="weaver1, weaver2, weaver3, weaver4, weaver5, vine5, vine11"
#SBATCH --constraint="gpu_12gb&pascal"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --qos=batch

# environment variables
NETID=sk7685
export TRAIN_FILE=/data/${NETID}/treebank_3/concat/ptb_train_v2.txt
export TEST_FILE=/data/${NETID}/treebank_3/concat/ptb_test_v2.txt

eval "$(conda shell.bash hook)"
conda activate nlu_env
module load cuda-10.1
#export PYTHONPATH=/data/${NETID}/transformers/src:$PYTHONPATH
#cd /data/${NETID}/transformers

TSEED=0
python run_language_modeling_hsfmx.py \
    --output_dir=ptb_hsfmx_seed${TSEED} \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --seed $TSEED \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --overwrite_output_dir \
    --hsfmx \
    --trees_path=/data/sk7685/nlu_project/utils/build_trees/temp_trees/*
