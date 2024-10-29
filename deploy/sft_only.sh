#!/bin/bash
set -a
source configs/.env
set +a

get_next_available_gpu() {
    # Get the GPU with the lowest memory usage
    GPU_ID=$(nvidia-smi --query-gpu=memory.used,index --format=csv,nounits,noheader | sort -n | awk '{print $2}' | head -n 1)
    echo $GPU_ID
}

# Default args
MODEL_CONFIG_FILE="./configs/model_paths.yaml"
pretrained_checkpoint_dir="${MANIFOLD_DIR}/all_in_one_pretraining/out/detailed_exps/minicpm_data/decay_from_45000_with_sft_and_input_0-5_pre_mates_default_for_4500steps_cpm/final/"
instruction_data_json="${MANIFOLD_DIR}/all_in_one_pretraining/datasets/knowledgeqa_formatted/concat_new.jsonl"
max_seq_len=2048
run_id=${EPOCHSECONDS}
is_on_tc=true
out_dir=""
max_lr=2e-5
n_devices=1
const_lr=false

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
    NEXT_GPU=$(get_next_available_gpu)
    #export CUDA_VISIBLE_DEVICES="$NEXT_GPU"
    export CUDA_VISIBLE_DEVICES="2,7"
    echo "Automatically selected GPU: $NEXT_GPU"
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
echo "max_lr: $max_lr"

litgpt finetune_full ${pretrained_checkpoint_dir} \
    --data "JSON" \
    --precision "bf16-true" \
    --data.json_path $instruction_data_json \
    --data.val_split_fraction 0.05 \
    --train.max_seq_len $max_seq_len \
    --train.epochs 1 \
    --data.mask_prompt false \
    --data.prompt_style "default" \
    --train.micro_batch_size 8 \
    --train.lr_warmup_steps 100 \
    --eval.interval 500 \
    --train.max_lr $max_lr \
    --devices $n_devices \
    --train.log_interval 5 \
    --logger_name wandb \
    --out_dir $out_dir

echo "Final model saved to $out_dir"
