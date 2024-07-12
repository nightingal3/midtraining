#!/bin/bash

set -a
source configs/.env
set +a

pretrained_checkpoint_dir="out/pythia-1b_pretrained_from_140000_id1720487487/step-00010000"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/sft/concat/"
max_seq_len=2048
model_name="pythia-1b"
step=140000
RUN_ID=1720487487

echo -e "\033[32m> > SFT \033[0m"
litgpt finetune_full ${pretrained_checkpoint_dir} \
    --data "JSON" \
    --data.json_path $instruction_data_json \
    --train.max_seq_len $max_seq_len \
    --train.epochs 5 \
    --train.lr_warmup_steps 100 \
    --logger_name wandb \
    --out_dir "out/${model_name}_pretrained_sft_from_${step}_id${RUN_ID}"


echo -e "\033[32m> Evaluating after pretraining and SFT...\033[0m"
litgpt evaluate "out/${model_name}_pretrained_sft_from_${step}_id${RUN_ID}/final" \
    --batch_size 8 \
    --out_dir "${MANIFOLD_DIR}/all_in_one_pretraining/post_results/${model_name}_mixed_posttune_id${RUN_ID}.json" \
    --tasks "gsm8k,arc_easy"
