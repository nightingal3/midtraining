#!/bin/bash

set -a
source configs/.env
set +a

### Default args

checkpoint_dir=""
include_dir=""
tasks="gsm8k,arc_easy,mathqa,logiqa2,hellaswag,piqa,commonsense_qa"
run_id=${EPOCHSECONDS}
is_on_tc=false
out_dir=""
### Default args

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            checkpoint_dir="${2:-$checkpoint_dir}"
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

echo "checkpoint: ${checkpoint_dir}"

echo -e "\033[32m> Evaluating after pretraining...\033[0m"
litgpt evaluate $checkpoint_dir \
    --batch_size 16 \
    --out_dir $out_dir \
    --tasks $tasks \
    --force_conversion true
