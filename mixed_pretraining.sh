#!/bin/bash

model_name=${1:-pythia-1b}
checkpoint_dir=${2:-/data/users/nightingal3/all_in_one_pretraining/models/EleutherAI/pythia-1b}
step=${3:-140000}
total_steps=${4:-180000}
max_seq_len=${5:-2048}


conda activate towerllm-env
#export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="6"
eval_initial=false
do_pretrain=true
do_sft=false

RUN_ID=$(date +%s)
#RUN_ID=1719949208
echo -e "\033[32m> The run id is ${RUN_ID}. Please keep this for your records \033[0m"
# Read personal user vars
set -a
source configs/.env
set +a

pretraining_data_dir="${FINEWEB_DIR}"
instruction_data_json="/data/users/nightingal3/manifold/all_in_one_pretraining/datasets/sft/concat/"


# $1 = checkpoint_dir, $2 - out_dir
# Note: evaluate always expects a relative path...
# strange error with litgpt eval - something about .git??
evaluate() {
  echo -e "\033[32m> Evaluating \033[0m"
  litgpt evaluate $1 \
    --batch_size 64 \
    --out_dir $2 \
    --tasks "gsm8k,arc_easy" \
    --force_conversion true
}

# # 1) Download a tokenizer
litgpt download EleutherAI/$model_name \
  --tokenizer_only True \
  --checkpoint_dir "${checkpoint_dir}/step${step}"

# # 2) Pretrain the model
steps_remaining=$((total_steps - step))
if [ $steps_remaining -lt 0 ]; then
    echo -e "\033[1;31m> total_steps should > steps. Total steps passed: ${total_steps}, step to start from passed: ${step}\033[0m" >&2
    exit 1
fi

if [ ! -d "${checkpoint_dir}/step${step}/lit_model.pth" ]; then
    echo -e "\033[32m> Converting model... \033[0m"
    litgpt convert_to_litgpt "${checkpoint_dir}/step${step}" --model_name $model_name
fi

if [[ $eval_initial == true ]]; then
  echo -e "\033[32m> Evaluating before pretraining...\033[0m"
  evaluate "${checkpoint_dir}/step${step}" "results/"
fi

pretrained_checkpoint_dir="${checkpoint_dir}/step${step}"
if [[ $do_pretrain == true ]]; then
  if [ $steps_remaining -gt 0 ]; then
    # TODO - train details such as batch size should be passed from a config instead of manually passed in.
    echo -e "\033[32m> > Pretraining for ${steps_remaining} more steps\033[0m"

    # temp hack: need to change max_iters in the dataloader
    # train.lr_warmup_fraction also doesn't seem to be passed through?

    litgpt pretrain_mixed $model_name \
      --initial_checkpoint_dir "${checkpoint_dir}/step${step}" \
      --tokenizer_dir "${checkpoint_dir}/step${step}" \
      --data MixedDataset \
      --data.pretraining_data_path $pretraining_data_dir \
      --data.sft_data_path $instruction_data_json \
      --train.micro_batch_size 16 \
      --train.max_seq_len $max_seq_len \
      --train.min_lr 1e-6 \
      --train.max_steps 100000 \
      --train.save_interval 1000 \
      --train.lr_warmup_fraction 0.01 \
      --eval.interval 1000 \
      --out_dir "out/${model_name}_mixtrained_from_${step}_id${RUN_ID}" \
      --data.num_repeats 1
      #--logger_name wandb

    exit 0

    pretrained_checkpoint_dir="out/${model_name}_mixtrained_from_${step}_id${RUN_ID}"
  fi

  echo -e "\033[32m> Evaluating after pretraining...\033[0m"
  litgpt evaluate "out/${model_name}_mixtrained_from_${step}_id${RUN_ID}/final" \
      --batch_size 16 \
      --out_dir "post_results/${model_name}_mixed_posttune_id${RUN_ID}" \
      --tasks "gsm8k,arc_easy"
fi


if [[ $do_sft == true ]]; then
  pretrained_checkpoint_dir="out/${model_name}_mixtrained_from_${step}_id${RUN_ID}/final"
  echo -e "\033[32m> > SFT \033[0m"
  litgpt finetune_full $pretrained_checkpoint_dir \
    --data "JSON" \
    --data.json_path $instruction_data_json \
    --train.max_seq_len $max_seq_len \
    --train.epochs 3 \
    --train.lr_warmup_steps 100 \
    --logger_name wandb \
    --out_dir "out/${model_name}_mixtrained_sft_from_${step}_id${RUN_ID}"


  echo -e "\033[32m> Evaluating after pretraining and SFT...\033[0m"
  litgpt evaluate "out/${model_name}_mixtrained_sft_from_${step}_id${RUN_ID}/final" \
      --batch_size 8 \
      --out_dir "post_results/${model_name}_mixed_posttune_id${RUN_ID}.json" \
      --tasks "gsm8k,arc_easy"
fi
