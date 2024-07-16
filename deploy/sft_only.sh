#!/bin/bash

set -a
source configs/.env
set +a


### Default args
pretrained_checkpoint_dir="${MANIFOLD_DIR}/out/pythia-1b_seq_train_d_10000-ljcp0f9r/final/"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/sft_final/1000/"
max_seq_len=2048
run_id=${EPOCHSECONDS}
is_on_tc=false
### Default args

while [[ $# -gt 0 ]]; do
    case $1 in
        --pretrained_checkpoint_dir)
            checkpoint_dir="${2:-$checkpoint_dir}"
            shift 2
            ;;
        --instruction_data_json)
            instruction_data_json="${2:-$instruction_data_json}"
            shift 2
            ;;
        --run_id)
            run_id="${2:-$run_id}"
            shift 2
            ;;
        --is_on_tc)
            is_on_tc=true
            shift 1
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

### Passed args
echo "pretrained_checkpoint_dir: $pretrained_checkpoint_dir"
echo "instruction_data_json: $instruction_data_json"
echo "max_seq_len: $max_seq_len"
echo "run id: $run_id"
echo "is_on_tc: $is_on_tc"
### Passed args

if [[ $is_on_tc == true ]]; then
  # mount manifold
  if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    source /packages/conda_mast_core/mount/mount.sh
  fi
fi


echo -e "\033[32m> > SFT \033[0m"
litgpt finetune_full ${pretrained_checkpoint_dir} \
    --data "JSON" \
    --data.json_path $instruction_data_json \
    --train.max_seq_len $max_seq_len \
    --train.epochs 1 \
    --train.micro_batch_size 8 \
    --train.lr_warmup_steps 100 \
    --train.log_interval 100 \
    --eval.interval 500 \
    --train.min_lr 1e-6 \
    --logger_name tensorboard \
    --out_dir "${MANIFOLD_DIR}/all_in_one_pretraining/out/post_sft_from_id${run_id}"

echo "Final model saved to ${MANIFOLD_DIR}/all_in_one_pretraining/out/post_sft_from_id${run_id}"
# mkdir -p  "${MANIFOLD_DIR}/all_in_one_pretraining/post_results/"

# echo -e "\033[32m> Evaluating after pretraining and SFT...\033[0m"
# litgpt evaluate "${MANIFOLD_DIR}/all_in_one_pretraining/out/post_sft_from_id${run_id}" \
#     --batch_size 8 \
#     --out_dir "${MANIFOLD_DIR}/all_in_one_pretraining/post_results/post_sft_from_id${run_id}.json" \
#     --tasks "gsm8k,arc_easy,mathqa,logiqa2,hellaswag,piqa,commonsense_qa"
