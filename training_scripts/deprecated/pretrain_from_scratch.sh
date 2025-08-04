#!/bin/bash
#SBATCH --job-name=pretrain_small
#SBATCH --output=pretrain_small_%A_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --partition=general
#SBATCH --mail-type=END
#SBATCH --mail-user=emmy@cmu.edu

eval_initial=false

# Read personal user vars
set -a
source configs/.env
set +a

### Default args
model_name="pythia-14m"
max_additional_steps=100000
max_seq_len=512
max_iters=100000000
pretraining_data_dir="${FINEWEB_DIR}/train"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/sft/concat/"
run_id=${EPOCHSECONDS}
is_on_tc=false
lr_scheduler="wsd"
out_dir="${MANIFOLD_DIR}/all_in_one_pretraining/out/${model_name}_raw_pretrained_from_${step}_id${run_id}"
micro_batch_size=16
log_interval=10
### Default args
export CUDA_VISIBLE_DEVICES="0"

echo_all_params() {
    echo "All parameters (including defaults):"
    echo "------------------------------------"
    echo "model_name: $model_name"
    echo "step: $step"
    echo "max_iters: $max_iters"
    echo "max_additional_steps: $max_additional_steps"
    echo "max_seq_len: $max_seq_len"
    echo "checkpoint_dir: $checkpoint_dir"
    echo "pretraining_data_dir: $pretraining_data_dir"
    echo "instruction_data_paths: $instruction_data_paths"
    echo "run_id: $run_id"
    echo "is_on_tc: $is_on_tc"
    echo "out_dir: $out_dir"
    echo "logs_dir: $logs_dir"
    echo "data_ratios: $data_ratios"
    echo "sft_template: $sft_template"
    echo "lr_scheduler: $lr_scheduler"
    echo "micro_batch_size: $micro_batch_size"
    echo "log_interval: $log_interval"
    echo "------------------------------------"
}

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
        --max_additional_steps)
            max_additional_steps="${2:-$max_additional_steps}"
            shift 2
            ;;
        --max_iters)
            max_iters="${2:-$max_iters}"
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
        --decay_lr)
            lr_scheduler="decay"
            shift 1
            ;;
        --const_lr)
            lr_scheduler="constant"
            shift 1
            ;;
        --out_dir)
            out_dir="${2:-$out_dir}"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

echo -e "\033[32m> The run id is ${run_id}. Please keep this for your records \033[0m"


if [[ $is_on_tc == true ]]; then
    # mount manifold
    if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
        source /packages/conda_mast_core/mount/mount.sh
    fi
fi

### Print all settings
echo "model_name: ${model_name}"
echo "is on tc: ${is_on_tc}"
echo "checkpoint dir: ${checkpoint_dir}"
###

evaluate() {
  echo -e "\033[32m> Evaluating \033[0m"
  with-proxy lm_eval --model hf \
    --model_args pretrained=$1 \
    --batch_size 32 \
    --tasks "gsm8k" \
    --output_path base_results
}

if [[ $eval_initial == true ]]; then
  echo -e "\033[32m> Evaluating before pretraining...\033[0m"
  evaluate "${checkpoint_dir}/step${step}"
fi

pretrained_checkpoint_dir="${checkpoint_dir}/step${step}"

export NCCL_P2P_DISABLE=1

srun python -m litgpt pretrain $model_name \
  --tokenizer_dir $model_name \
  --precision "bf16-true" \
  --data FineWebDataset \
  --data.data_path $pretraining_data_dir \
  --data.val_data_path "${FINEWEB_DIR}/val" \
  --train.micro_batch_size $micro_batch_size \
  --train.max_seq_len $max_seq_len \
  --train.min_lr 1e-6 \
  --train.max_iters ${max_iters} \
  --train.max_additional_steps $max_additional_steps \
  --train.save_interval 500 \
  --train.log_interval $log_interval \
  --train.lr_warmup_fraction 0.01 \
  --train.lr_scheduler $lr_scheduler \
  --train.stable_steps 99000 \
  --eval.interval 1000 \
  --optimizer.class_path torch.optim.AdamW \
  --optimizer.init_args.lr 0.005 \
  --out_dir $out_dir \
  --logs_dir $out_dir \
  --logger_name wandb
