#!/bin/bash

eval_initial=false
do_pretrain=true
do_sft=false

# Read personal user vars
set -a
source configs/.env
set +a

### Default args
model_name="pythia-1b"
step="-00045200"
steps_to_train=100000
max_seq_len=2048
checkpoint_dir="${MANIFOLD_DIR}/all_in_one_pretraining/models/EleutherAI/${model_name}/"
pretraining_data_dir="${FINEWEB_DIR}"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/sft_reasoning/concat"
instruction_data_json_2="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/tulu"
#TODO: testing cycling. switch batck to the non-toy example later
instruction_data_json_3="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/instruction/hkust-nlp/deita-10k-v0/train.json"
instruction_data_paths="concat_sft ${instruction_data_json} 0.33"
run_id=${EPOCHSECONDS}
is_on_tc=false
decay_lr=false
out_dir="${MANIFOLD_DIR}/all_in_one_pretraining/out/${model_name}_mixtrained_from_${step}_id${run_id}"
logs_dir="${MANIFOLD_DIR}/all_in_one_pretraining/out/${model_name}_mixtrained_from_${step}_id${run_id}"
### Default args

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            model_name="${2:-$model_name}"
            shift 2
            ;;
        --step)
            step="${2:-$step}"
            shift 2
            ;;
        --steps_to_train)
            steps_to_train="${2:-$steps_to_train}"
            shift 2
            ;;
        --max_seq_len)
            max_seq_len="${2:-$max_seq_len}"
            shift 2
            ;;
        --checkpoint_dir)
            checkpoint_dir="${2:-$checkpoint_dir}"
            shift 2
            ;;
        --pretraining_data_dir)
            pretraining_data_dir="${2:-$pretraining_data_dir}"
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
        --logs_dir)
            logs_dir="${2:-$logs_dir}"
            shift 2
            ;;
        --out_dir)
            out_dir="${2:-$out_dir}"
            shift 2
            ;;
        --decay_lr)
            decay_lr=true
            shift 1
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

echo -e "\033[32m> The run id is ${run_id}. Please keep this for your records \033[0m"


### Print all settings
echo "model_name: ${model_name}"
echo "is on tc: ${is_on_tc}"
echo "out_dir: ${out_dir}"
echo "logs_dir: ${logs_dir}"
###
if [[ $is_on_tc == true ]]; then
  # mount manifold
  if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    source /packages/conda_mast_core/mount/mount.sh
  fi
fi

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

#if [ ! -f "${checkpoint_dir}/step${step}/lit_model.pth" ]; then
    #echo -e "\033[32m> Converting model... \033[0m"
    #litgpt convert_to_litgpt "${checkpoint_dir}/step${step}" --model_name $model_name
#fi

if [[ $eval_initial == true ]]; then
  echo -e "\033[32m> Evaluating before pretraining...\033[0m"
  evaluate "${checkpoint_dir}/step${step}" "results/"
fi

pretrained_checkpoint_dir="${checkpoint_dir}/step${step}"
if [[ $do_pretrain == true ]]; then
  if [ $steps_to_train -gt 0 ]; then
    # TODO - train details such as batch size should be passed from a config instead of manually passed in.
    echo -e "\033[32m> > Pretraining for ${steps_to_train} more steps\033[0m"

    # temp hack: need to change max_iters in the dataloader
    # train.lr_warmup_fraction also doesn't seem to be passed through?

    litgpt pretrain_mixed $model_name \
      --resume "${checkpoint_dir}/step${step}/lit_model.pth" \
      --precision "bf16-true" \
      --tokenizer_dir "${checkpoint_dir}/step${step}" \
      --data MixedDataset \
      --data.pretraining_data_path ${pretraining_data_dir} \
      --data.sft_data_paths "${instruction_data_paths}" \
      --data.use_adaptive_sampling true \
      --train.freeze_sampling_rate true \
      --data.initial_sampling_rates "[0.75, 0.25]" \
      --train.micro_batch_size 16 \
      --train.max_seq_len $max_seq_len \
      --train.min_lr 1e-6 \
      --train.max_steps $steps_to_train \
      --train.save_interval 500 \
      --train.lr_warmup_fraction 0.01 \
      --train.episode_length 2000 \
      --train.log_interval 50 \
      --train.decay_lr $decay_lr \
      --eval.interval 1000 \
      --eval.max_iters 100 \
      --out_dir $out_dir \
      --logs_dir $logs_dir \
      --data.num_repeats 1 \
      --logger_name tensorboard
    fi
fi


if [[ $do_sft == true ]]; then
  echo -e "\033[32m> > SFT \033[0m"
  litgpt finetune_full ${out_dir} \
      --data "JSON" \
      --data.json_path $instruction_data_json \
      --train.max_seq_len $max_seq_len \
      --train.epochs 1 \
      --train.micro_batch_size 4 \
      --train.lr_warmup_steps 100 \
      --eval.interval 500 \
      --train.min_lr 1e-6 \
      --train.log_interval 100 \
      --logger_name wandb \
      --out_dir "${MANIFOLD_DIR}/all_in_one_pretraining/out/post_sft_from_id${run_id}"

    echo "Final model saved to ${MANIFOLD_DIR}/all_in_one_pretraining/out/post_sft_from_id${run_id}"
fi
