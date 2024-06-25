#!/bin/bash

model_name=${1:-pythia-410m}
checkpoint_dir=${2:-/data/users/nightingal3/all_in_one_pretraining/models/EleutherAI/pythia-410m/}
step=${3:-140000}
total_steps=${4:-180000}
max_seq_len=${5:-2048}

conda activate towerllm-env
export CUDA_VISIBLE_DEVICES="7"
# Read personal user vars
set -a
source configs/.env
set +a

pretraining_data_dir="${FINEWEB_DIR}"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/sft/openai/gsm8k/train.json"

# 1) Download a tokenizer
echo -e "\033[32m> Downloading tokenizer \033[0m"
litgpt download EleutherAI/$model_name \
  --tokenizer_only True

# 2) Pretrain the model
steps_remaining=$((total_steps - step))
if [ $steps_remaining -lt 0 ]; then
    echo -e "\033[1;31m> total_steps should > steps. Total steps passed: ${total_steps}, step to start from passed: ${step}\033[0m" >&2
    exit 1
fi

if [ ! -d "${checkpoint_dir}/step${step}/lit_model.pth" ]; then
    echo -e "\033[32m> Converting model... \033[0m"
    litgpt convert_to_litgpt "${checkpoint_dir}/step${step}" --model_name $model_name
fi

pretrained_checkpoint_dir="${checkpoint_dir}/step${step}"
if [ $steps_remaining -gt 0 ]; then
  # TODO - train details such as batch size should be passed from a config instead of manually passed in.
  # legit runs should have 20B. Test runs with 1B-5B for sanity checking
  echo -e "\033[32m> > Pretraining for ${steps_remaining} more steps\033[0m"
  litgpt pretrain $model_name \
    --initial_checkpoint_dir "${checkpoint_dir}/step${step}" \
    --tokenizer_dir "${checkpoint_dir}/step${step}" \
    --data FineWebDataset \
    --data.data_path "${FINEWEB_DIR}" \
    --train.max_tokens "1_000_000_000" \
    --train.global_batch_size 1024 \
    --train.max_seq_len $max_seq_len \
    --out_dir "out/${model_name}_pretrained_from_${step}"

  pretrained_checkpoint_dir="out/${model_name}_pretrained_from_${step}"
fi

# 3) Instruction-tune the model
echo "\033[32m> > SFT \033[0m"
litgpt finetune full --checkpoint_dir $pretrained_checkpoint_dir \
  --data "JSON" \
  --data.json_path $instruction_data_json \
  --data.val_split_fraction 0.1 \
  --train.max_seq_len $max_seq_len \
  --logger_name wandb \
  --out_dir "out/${model_name}_instruction_posttune"

# TODO - need to add some tasks to eval harness.
echo "> Evaluating"
litgpt evaluate "out/${model_name}_instruction_posttune/final" \
    --batch_size 8 \
    --out_dir "eval_results/${model_name}_instruction_posttune" \
    --tasks "gsm8k"
