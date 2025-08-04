#!/usr/bin/env bash
# filepath: merge_sharded_slurm.sh
#SBATCH --job-name=merge_shards
#SBATCH --output=merge_shards_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --partition=general
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=youremail@domain.com

# load modules or conda env if needed
# module load python/3.10
# source ~/.bashrc && conda activate your_env

echo "[$(date)] Starting merge_sharded_dataset job on $HOSTNAME"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env_dev_copy

# change to project dir if needed
cd /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/util_scripts


# run the Python merge script
srun python -u merge_sharded_dataset.py

echo "[$(date)] Finished merging shards"