#!/bin/bash
set -a
source /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/configs/.env
set +a

get_next_available_gpu() {
    # Get the GPU with the lowest memory usage
    GPU_ID=$(nvidia-smi --query-gpu=memory.used,index --format=csv,nounits,noheader | sort -n | awk '{print $2}' | head -n 1)
    echo $GPU_ID
}

# Default args
MODEL_CONFIG_FILE="./configs/model_paths.yaml"  # Hardcoded path to the YAML file
pretrained_checkpoint_dir="${MANIFOLD_DIR}/all_in_one_pretraining/out/detailed_exps/minicpm_data/decay_from_45000_with_sft_and_input_0-5_pre_mates_default_for_4500steps_cpm/final/"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/knowledgeqa_formatted/concat_new.jsonl"
max_seq_len=2048
run_id=${EPOCHSECONDS}
is_on_tc=false
out_dir=""
max_lr=1e-3
n_devices=1
micro_batch_size=16
const_lr=false
wandb_group="training"
no_save_chkpt=false
wandb_tags=""
mask_prompt=false
prompt_style="default"
seed=1337
num_epochs=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --pretrained_checkpoint_dir)
            pretrained_checkpoint_dir="$2"
            shift 2
            ;;
        --instruction_data_json)
            instruction_data_json="$2"
            shift 2
            ;;
        --run_id)
            run_id="$2"
            shift 2
            ;;
        --is_on_tc)
            is_on_tc=true
            shift 1
            ;;
        --const_lr)
            const_lr=true
            shift 1
            ;;
        --out_dir)
            out_dir="$2"
            shift 2
            ;;
        --max_lr)
            max_lr="$2"
            shift 2
            ;;
        --wandb_group)
            wandb_group="$2"
            shift 2
            ;;
        --no_save_chkpt)
            no_save_chkpt=true
            shift 1
            ;;
        --wandb_tags)
            wandb_tags="$2"
            shift 2
            ;;
        --mask_prompt)
            mask_prompt=true
            shift 1
            ;;
        --prompt_style)
            prompt_style="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --n_devices)
            n_devices="$2"
            shift 2
            ;;
        --micro_batch_size)
            micro_batch_size="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

# If a model_config is provided, use the YAML configuration
if [ ! -z "$MODEL_CONFIG" ]; then
    if [ ! -f "$MODEL_CONFIG_FILE" ]; then
        echo "Error: Configuration file $MODEL_CONFIG_FILE not found."
        exit 1
    fi

    # Extract model name and path from YAML file
    MODEL_NAME=$(grep -A2 "$MODEL_CONFIG:" "$MODEL_CONFIG_FILE" | grep "name:" | sed 's/.*name: *"\(.*\)".*/\1/')
    MODEL_PATH=$(grep -A2 "$MODEL_CONFIG:" "$MODEL_CONFIG_FILE" | grep "path:" | sed 's/.*path: *"\(.*\)".*/\1/')

    if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ]; then
        echo "Error: Model config '$MODEL_CONFIG' not found in configuration or is incomplete."
        exit 1
    fi

    # Expand environment variables in the MODEL_PATH
    MODEL_PATH=$(eval echo $MODEL_PATH)

    # Override the pretrained_checkpoint_dir with the path from YAML
    pretrained_checkpoint_dir=$MODEL_PATH

    echo "Using model config: $MODEL_CONFIG"
    echo "Model name: $MODEL_NAME"
    echo "Model path: $MODEL_PATH"
fi

if [[ $is_on_tc == false ]]; then
    if [[ $n_devices -gt 1 ]]; then
        # For multi-GPU, set CUDA_VISIBLE_DEVICES to use multiple GPUs
        export CUDA_VISIBLE_DEVICES="0,1"  # Adjust based on available GPUs
        echo "Using multiple GPUs: $CUDA_VISIBLE_DEVICES (n_devices: $n_devices)"
    else
        NEXT_GPU=$(get_next_available_gpu)
        export CUDA_VISIBLE_DEVICES="$NEXT_GPU"
        echo "Automatically selected single GPU: $NEXT_GPU"
    fi
fi


if [[ $is_on_tc == true ]]; then
    # mount manifold
    if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
        source /packages/conda_mast_core/mount/mount.sh
    fi
    # set # devices to not break the script.
    # NOTE: assuming running SFT on one node always
    n_devices=8
fi

echo -e "\033[32m> > SFT \033[0m"
echo "pretrained_checkpoint_dir: $pretrained_checkpoint_dir"
echo "instruction_data_json: $instruction_data_json"
echo "max_seq_len: $max_seq_len"
echo "run id: $run_id"
echo "is_on_tc: $is_on_tc"
echo "n_devices: $n_devices"
echo "micro_batch_size: $micro_batch_size"
echo "max_lr: $max_lr"
echo "prompt_style: $prompt_style"
echo "seed: $seed"
echo "num_epochs: $num_epochs"

cd /data/tir/projects/tir3/users/mengyan3/all_in_one_pretraining/
set -a
source configs/.env
set +a


source ${MINICONDA_PATH}
conda activate llm_env_dev_copy

litgpt finetune_full ${pretrained_checkpoint_dir} \
    --data "MultiJSON" \
    --precision "bf16-true" \
    --data.json_path $instruction_data_json \
    --train.max_seq_len $max_seq_len \
    --train.epochs $num_epochs \
    --data.mask_prompt $mask_prompt \
    --data.prompt_style $prompt_style \
    --train.micro_batch_size $micro_batch_size \
    --train.lr_warmup_fraction 0.1 \
    --eval.interval 500 \
    --train.max_lr $max_lr \
    --devices $n_devices \
    --train.log_interval 10 \
    --logger_name wandb \
    --wandb_group ${wandb_group} \
    --wandb_tags ${wandb_tags} \
    --out_dir $out_dir \
    --no_save_chkpt $no_save_chkpt \
    --seed $seed \
    #--data.val_split_fraction 0.05 \