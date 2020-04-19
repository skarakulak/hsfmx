#!/bin/bash
#
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name="trees"
#SBATCH --output=logs/build_trees_%j.out
#SBATCH --error=logs/build_trees_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --qos=batch

# environment variables
NETID=sk7685

eval "$(conda shell.bash hook)"
conda activate nlu_env

srun python build_trees_parallel.py \
    -save_dir trees_partitioned \
    -n_proc 10
