#!/bin/bash
set -a
source configs/.env
set +a

### Default args
export CUDA_VISIBLE_DEVICES="5"
root_path=""
root_checkpoint_dirs=()
include_dir=""
run_id=${EPOCHSECONDS}
is_on_tc=false
out_dir="evals"
#all_tasks="arc_easy,asdiv,sciq,gsm8k,commonsense_qa,hellaswag,logiqa2,piqa,mmlu,mathqa,svamp"
all_tasks="mmlu"
### Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --root_path)
            root_path="$2"
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
            tasks="${2:-$tasks}"
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

if [[ -z "$root_path" ]]; then
    echo "Error: --root_path must be specified."
    exit 1
fi

# Get all immediate subdirectories of root_path
while IFS= read -r -d '' dir; do
    root_checkpoint_dirs+=("$dir")
done < <(find "$root_path" -maxdepth 1 -mindepth 1 -type d -print0)

if [ ${#root_checkpoint_dirs[@]} -eq 0 ]; then
    echo "No subdirectories found in $root_path"
    exit 1
fi

if [[ $is_on_tc == true ]]; then
    # mount manifold
    if [ "$LOCAL_RANK" = "0" ] && [ -z "$DISABLE_MOUNT" ]; then
        source /packages/conda_mast_core/mount/mount.sh
    fi
fi

for root_checkpoint_dir in "${root_checkpoint_dirs[@]}"; do
    echo -e "\033[34m> Processing checkpoints in ${root_checkpoint_dir} ...\033[0m"
    # NOTE: change this back to -* later just wanted to have quick comparison
    #find "$root_checkpoint_dir" -type d \( -name "step-00*" \) | while read -r dir; do
    find "$root_checkpoint_dir" -type d \( -name "step-*" -o -name "final" \) | while read -r dir; do
    #find "$root_checkpoint_dir" -type d -name "final" | while read -r dir; do

        subdir=$(basename "$dir")
        full_path="${root_checkpoint_dir}/${out_dir}/${subdir}"
        if [[ -d $full_path ]]; then # somehow this sometimes causes a problem if model files copied but eval crashed
            if [[ -f "${full_path}/results.json" ]]; then
                echo "Done evaluating this path, continuing..."
                continue
            fi
            rm -rf $full_path
        fi
        mkdir -p "$full_path"

        echo -e "\033[32m> Evaluating ${dir} ...\033[0m"
        litgpt evaluate "$dir" \
            --batch_size 16 \
            --out_dir "$full_path" \
            --tasks ${all_tasks} \
            --force_conversion true \
            --use_cli false
    done
done
