#!/bin/bash

model_name=${1:-pythia-410m}
checkpoint_dir=${2:-/home/nightingal3/fbsource/fbcode/ads_content_understanding/fmcu/projects/continue_pretrain_llm/models/EleutherAI/pythia-410m}
step=${3:-130000}
mixed_data_dir=${4:-/home/nightingal3/fbsource/fbcode/ads_content_understanding/fmcu/projects/continue_pretrain_llm/datasets/mixed_training}
total_steps=${5:-143000}
max_seq_len=${6:-2048}


conda activate towerllm-env
export CUDA_VISIBLE_DEVICES=7

# example pretraining text if pretraining_data_dir doesn't exist

if [ ! -d $pretraining_data_dir ]; then
    mkdir -p $pretraining_data_dir
    curl https://www.gutenberg.org/cache/epub/24440/pg24440.txt --output custom_texts/book1.txt
    curl https://www.gutenberg.org/cache/epub/26393/pg26393.txt --output custom_texts/book2.txt
fi

# 1) Download a tokenizer
litgpt download EleutherAI/$model_name \
  --tokenizer_only True

# 2) Pretrain the model
steps_remaining=$((total_steps - step))
if [ $steps_remaining -lt 0 ]; then
    echo "total_steps should > steps. Total steps passed: ${total_steps}, step to start from passed: ${step}" >&2
    exit 1
fi

if [ ! -d "${checkpoint_dir}/step${step}/lit_model.pth" ]; then
    litgpt convert_to_litgpt "${checkpoint_dir}/step${step}" --model_name $model_name
fi

pretrained_checkpoint_dir="${checkpoint_dir}/step${step}"
if [ $steps_remaining -gt 0 ]; then
  # TODO - train details such as batch size should be passed from a config instead of manually passed in.
  echo "> Pretraining for ${steps_remaining} more steps"
  litgpt pretrain_mixed $model_name \
    --initial_checkpoint_dir "${checkpoint_dir}/step${step}" \
    --tokenizer_dir "${checkpoint_dir}/step${step}" \
    --data MixedDataset \
    --data.data_path $mixed_data_dir \
    --train.max_tokens 20_000_000 \
    --train.global_batch_size 128 \
    --train.max_seq_len $max_seq_len \
    --train.min_lr 1e-7 \
    --out_dir "out/${model_name}_mixtrained_from_${step}"

  pretrained_checkpoint_dir="out/${model_name}_pretrained_from_${step}"
fi
# 3) Instruction-tune the model
# echo "> Instruction finetuning (additional)"
# litgpt finetune full --checkpoint_dir $pretrained_checkpoint_dir \
#   --data "JSON" \
#   --data.json_path $instruction_data_json \
#   --data.val_split_fraction 0.1 \
#   --train.max_seq_len $max_seq_len \
#   --logger_name wandb \
#   --train.num_epochs 1 \
#   --out_dir "out/${model_name}_instruction_posttune"

# # TODO - need to add some tasks to eval harness. Only see this instruction tuning one in eval harness
# echo "> Evaluating"
# litgpt evaluate --checkpoint_dir "out/${model_name}_instruction_posttune" \
#     --batch_size 128 \
#     --out_dir "eval_results/${model_name}_instruction_posttune" \
#     --tasks "ifeval"
