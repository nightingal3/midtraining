#!/bin/bash

set -a
source configs/.env
set +a

get_next_available_gpu() {
    # Get the GPU with the lowest memory usage
    GPU_ID=$(nvidia-smi --query-gpu=memory.used,index --format=csv,nounits,noheader | sort -n | awk '{print $2}' | head -n 1)
    echo $GPU_ID
}

### Default args
NEXT_GPU=$(get_next_available_gpu)
export CUDA_VISIBLE_DEVICES="$NEXT_GPU"
echo "Automatically selected GPU: $NEXT_GPU"

root_checkpoint_dir=""
include_dir=""
run_id=${EPOCHSECONDS}
is_on_tc=false
out_dir="evals"
all_tasks="arc_easy,asdiv,sciq,commonsense_qa,hellaswag,logiqa2,piqa,mmlu,mathqa"
#all_tasks="mmlu"
#remaining_tasks="mathqa,svamp,wikiqa,bbh"
### Default args

while [[ $# -gt 0 ]]; do
    case $1 in
        --root_checkpoint_dir)
            root_checkpoint_dir="${2:-$root_checkpoint_dir}"
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
        --is_local)
            is_on_tc=false
            shift 1
            ;;
        --logs_dir)
            logs_dir="${2:-$logs_dir}"
            shift 2
            ;;
        --tasks)
            tasks="${2:-tasks}"
            shift 2
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

if [[ $is_on_tc == true ]]; then
  # mount manifold
  if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
    source /packages/conda_mast_core/mount/mount.sh
  fi
fi

# NOTE change this back
#find "$root_checkpoint_dir" -type d \( -name "final" -o -name "step-*" \) | while read -r dir; do
find "$root_checkpoint_dir" -type d \( -name "final" \) | while read -r dir; do
    subdir=$(basename "$dir")
    full_path="${root_checkpoint_dir}/${out_dir}/${subdir}"
    #full_path="../manifold/all_in_one_pretraining/training_analysis/pythia-1b-fw/evals/${subdir}"
    mkdir -p "$full_path"

  # Do something with the subdirectory here
    echo -e "\033[32m> Evaluating ${dir} ...\033[0m"
    litgpt evaluate $dir \
        --batch_size 16 \
        --out_dir $full_path \
        --tasks ${all_tasks} \
        --force_conversion true \
        --num_fewshot 5 \
        --use_cli false 


    # set use cli to true and parallelize to true for larger models
done
