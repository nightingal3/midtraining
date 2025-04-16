#!/bin/sh
#SBATCH --job-name=pretrain_small
#SBATCH --output=pretrain_small_%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=array
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --array=1-4

# Read personal user vars
set -a
source configs/.env
set +a


source ${MINICONDA_PATH}
conda activate llm_env_dev_copy


model_config_files=(
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pythia_70m_from_scratch_32B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pythia_160m_from_scratch_32B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pythia_70m_from_scratch_64B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pythia_160m_from_scratch_64B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pythia_70m_from_scratch_128B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pythia_160m_from_scratch_64B.yaml"
)
model_config_file=${model_config_files[$((SLURM_ARRAY_TASK_ID - 1))]}

cd /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/litgpt

if [[ "$(hostname)" =~ ^(shire-2-(9|5)|babel-8-5|babel-4-(1|5|9|13|17|21|25|29)|babel-6-(5|9|13|29)|babel-7-(1|5|9)|babel-12-(5|9|13)|babel-13-(1|5|9|13|17|21|25|29)|babel-14-(1|5|9|13|17|21|25|29|37)|babel-5-15|babel-10-17|babel-0-19|babel-11-25|babel-9-3)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi

BUCKET_NAME="gs://cmu-gpucloud-mengyan3"
dset_base_name="c4_pythia"

rm -rf /scratch/mengyan3/${dset_base_name}
rm -rf /scratch/mengyan3/${dset_base_name}_val
mkdir -p /scratch/mengyan3/${dset_base_name}
mkdir -p /scratch/mengyan3/${dset_base_name}_val

gcloud storage cp -r ${BUCKET_NAME}/${dset_base_name} /scratch/mengyan3/${dset_base_name}

./fix_dir_structure.sh /scratch/mengyan3/${dset_base_name} /scratch/mengyan3/${dset_base_name}_val

srun python -m litgpt pretrain $model_name --config ${model_config_file}