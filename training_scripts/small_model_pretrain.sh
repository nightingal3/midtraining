#!/bin/sh
#SBATCH --job-name=09_30_small_model_pretrain
#SBATCH --output=%x_%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpuA100x8
#SBATCH --account=bfcu-delta-gpu
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu
#SBATCH --array=21

# Read personal user vars
set -a
source configs/.env
set +a

source ${MINICONDA_PATH}
conda activate llm_env_dev_copy

echo "Running on host $(hostname)"
echo nvidia-smi

model_config_files=(
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_from_scratch_128B.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_from_scratch_128B.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_from_scratch_128B.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_6k.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_20k_math.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_40k_flan.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_70m_midtrain_from_40k_knowledgeqa.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_40k_knowledgeqa.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_70m_midtrain_from_40k_web.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_160m_midtrain_from_40k_web.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_70m_midtrain_from_40k_flan.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_6k.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_20k_math.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_40k_flan.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_40k_knowledgeqa.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_410m_midtrain_from_40k_web.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_6k.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_20k_math.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_40k_flan.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_40k_knowledgeqa.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/pretrain/pythia_1b_midtrain_from_40k_web.yaml"
)

model_config_files_ablations=(
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_20k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_30k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_40k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_70m_midtrain_from_50k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_20k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_30k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_40k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/timing/pythia_160m_midtrain_from_50k_sc.yaml"
)

model_config_files_ablations_2=(
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_10.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_30.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_50.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_70m_midtrain_sc_80.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_10.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_30.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_50.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/percentage/pythia_140m_midtrain_sc_80.yaml"
)

cts_pretrain_ablations=(
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_70m_midtrain_from_40k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_160m_midtrain_from_40k_sc.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_70m_midtrain_from_56k_math.yaml"
    "/projects/bfcu/mliu7/all_in_one_pretraining/midtrain_configs/ablations/cts_pretrain/pythia_160m_midtrain_from_56k_math.yaml"
)
# NOTE: doing ablations now!!! switch this back if not
#model_config_file=${model_config_files[$((SLURM_ARRAY_TASK_ID - 1))]}
model_config_file=${model_config_files[$((SLURM_ARRAY_TASK_ID - 1))]}
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

cd /projects/bfcu/mliu7/all_in_one_pretraining/litgpt

srun python -m litgpt pretrain --config ${model_config_file}