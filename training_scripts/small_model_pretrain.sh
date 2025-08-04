#!/bin/sh
#SBATCH --job-name=07_25_cts_math
#SBATCH --output=%x_%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=preempt
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --array=3-4
#SBATCH --exclude=babel-10-5,babel-10-9,babel-10-13,babel-10-17,babel-13-1,babel-13-5,babel-13-9,babel-13-13,babel-13-17,babel-13-21,babel-13-25,babel-13-29,babel-13-33,babel-13-37,babel-4-1,babel-8-9,shire-2-9,shire-2-5,babel-2-13,babel-2-17,babel-8-13,babel-11-25,babel-12-9,babel-4-25

# Read personal user vars
set -a
source configs/.env
set +a

source ${MINICONDA_PATH}
conda activate llm_env_dev_copy

echo "Running on host $(hostname)"
echo nvidia-smi

model_config_files=(
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_from_scratch_128B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_from_scratch_128B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_from_scratch_128B.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_6k.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_20k_math.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_40k_flan.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_70m_midtrain_from_40k_knowledgeqa.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_40k_knowledgeqa.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_70m_midtrain_from_40k_web.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_40k_web.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_70m_midtrain_from_40k_flan.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_6k.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_20k_math.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_40k_flan.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_40k_knowledgeqa.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_40k_web.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_6k.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_20k_math.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_40k_flan.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_40k_knowledgeqa.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_40k_web.yaml"
)

model_config_files_ablations=(
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_20k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_30k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_40k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_50k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_20k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_30k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_40k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_50k_sc.yaml"
)

model_config_files_ablations_2=(
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_10.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_30.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_50.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_80.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_10.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_30.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_50.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_80.yaml"
)

cts_pretrain_ablations=(
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_70m_midtrain_from_40k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_160m_midtrain_from_40k_sc.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_70m_midtrain_from_56k_math.yaml"
    "/data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_160m_midtrain_from_56k_math.yaml"
)
# NOTE: doing ablations now!!! switch this back if not
#model_config_file=${model_config_files[$((SLURM_ARRAY_TASK_ID - 1))]}
model_config_file=${cts_pretrain_ablations[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "Using model config file: $model_config_file"

# Auto-calculate remaining steps for preempt restarts
checkpoint_dir=$(grep "out_dir:" "$model_config_file" | awk '{print $2}')
target_steps=61035

# Get current step from checkpoints
current_step=0
if [ -d "$checkpoint_dir" ]; then
    # Check if training is complete (final checkpoint exists)
    if [ -d "$checkpoint_dir/final" ]; then
        current_step=$target_steps
    else
        # Find latest step checkpoint
        latest_checkpoint=$(ls -1 "$checkpoint_dir" 2>/dev/null | grep -E "^step-[0-9]+$" | sort -V | tail -1)
        if [ -n "$latest_checkpoint" ]; then
            # Extract step number and remove leading zeros to prevent octal interpretation
            step_num=$(echo "$latest_checkpoint" | sed 's/step-//' | sed 's/^0*//')
            current_step=${step_num:-0}  # Use 0 if step_num is empty
        fi
    fi
fi

remaining_steps=$((target_steps - current_step))

echo "Training progress: $current_step/$target_steps steps (${remaining_steps} remaining)"

# Exit if training is complete
if [ $remaining_steps -le 0 ]; then
    echo "Training already completed! Exiting."
    exit 0
fi

# Update config with remaining steps
sed -i "s/max_additional_steps: [0-9]*/max_additional_steps: $remaining_steps/" "$model_config_file"
echo "Updated config: max_additional_steps = $remaining_steps"

cd /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/litgpt

if [[ "$(hostname)" =~ ^(shire-2-(9|5)|babel-8-5|babel-4-(1|5|9|13|17|21|25|29)|babel-6-(5|9|13|29)|babel-7-(1|5|9)|babel-12-(5|9|13)|babel-13-(1|5|9|13|17|21|25|29)|babel-14-(1|5|9|13|17|21|25|29|37)|babel-5-15|babel-10-17|babel-0-19|babel-11-25|babel-9-3)$ ]]; then
  export NCCL_P2P_DISABLE=1
fi

srun python -m litgpt pretrain --config ${model_config_file}