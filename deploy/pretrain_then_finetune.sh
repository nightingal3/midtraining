#!/bin/bash

model_name=${1:-pythia-410m}
checkpoint_dir=${2:-./models/EleutherAI/pythia-410m/}
step=${3:-140000}
max_seq_len=${5:-2048}

#conda activate towerllm-env
export CUDA_VISIBLE_DEVICES="5"
eval_initial=false
# Read personal user vars
set -a
source configs/.env
set +a

pretraining_data_dir="${FINEWEB_DIR}"
instruction_data_json="./datasets/sft/allenai/ai2_arc/"

RUN_ID=$(date +%s)
echo -e "\033[32m> The run id is ${RUN_ID}. Please keep this for your records \033[0m"

# $1 = model_name
evaluate() {
  echo -e "\033[32m> Evaluating \033[0m"
  with-proxy lm_eval --model hf \
    --model_args pretrained=$1 \
    --batch_size 32 \
    --tasks "gsm8k" \
    --output_path base_results
}

# 1) Download a tokenizer
echo -e "\033[32m> Downloading tokenizer \033[0m"
litgpt download EleutherAI/$model_name \
  --tokenizer_only True

# 2) Pretrain the model
if [ ! -d "${checkpoint_dir}/step${step}/lit_model.pth" ]; then
    echo -e "\033[32m> Converting model... \033[0m"
    litgpt convert_to_litgpt "${checkpoint_dir}/step${step}" --model_name $model_name
fi

if [[ $eval_initial == true ]]; then
  echo -e "\033[32m> Evaluating before pretraining...\033[0m"
  evaluate "${checkpoint_dir}/step${step}"
fi

pretrained_checkpoint_dir="${checkpoint_dir}/step${step}"
# TODO - train details such as batch size should be passed from a config instead of manually passed in.
# legit runs should have 20B. Test runs with 1B-5B for sanity checking
# echo -e "\033[32m> > Pretraining ... \033[0m"
litgpt pretrain $model_name \
  --initial_checkpoint_dir "${checkpoint_dir}/step${step}" \
  --tokenizer_dir "${checkpoint_dir}/step${step}" \
  --data FineWebDataset \
  --data.data_path "${FINEWEB_DIR}" \
  --train.micro_batch_size 16 \
  --train.max_seq_len $max_seq_len \
  --train.min_lr 1e-6 \
  --train.max_steps 10000 \
  --train.save_interval 1000 \
  --train.lr_warmup_fraction 0.01 \
  --eval.interval 1000 \
  --out_dir "out/${model_name}_pretrained_from_${step}" \
  --logger_name wandb

pretrained_checkpoint_dir="out/${model_name}_pretrained_from_${step}"

# echo -e "\033[32m> Evaluating after pretraining and sft...\033[0m"
# litgpt evaluate "out/${model_name}_pretrained_from_${step}/final" \
#     --batch_size 8 \
#     --out_dir "post_results/${model_name}_pretrained_id${RUN_ID}" \
#     --tasks "gsm8k,arc_easy"

# 3) Instruction-tune the model
# warmup steps should be rougly 1%? 500 steps, 5 warmup steps? Seems kind of few
pretrained_checkpoint_dir="out/${model_name}_pretrained_from_${step}/final"
echo -e "\033[32m> > SFT \033[0m"
litgpt finetune_full $pretrained_checkpoint_dir \
  --data "JSON" \
  --data.json_path $instruction_data_json \
  --train.max_seq_len $max_seq_len \
  --train.epochs 5 \
  --train.lr_warmup_steps 100 \
  --logger_name wandb \
  --out_dir "out/${model_name}_pretrained_sft_from_${step}_id${RUN_ID}"

# TODO - need to add some tasks to eval harness.
echo -e "\033[32m> Evaluating after pretraining and sft...\033[0m"
litgpt evaluate "out/${model_name}_pretrained_sft_from_${step}_id${RUN_ID}/final" \
    --batch_size 8 \
    --out_dir "post_results/${model_name}_instruction_posttune_id${RUN_ID}" \
    --tasks "gsm8k,arc_easy"
