#!/bin/bash
set -ex
/packages/torchx_torchrun/torchrun \
  \
  --rdzv_backend c10d \
  --rdzv_endpoint localhost:0 \
  --rdzv_id $MAST_HPC_JOB_NAME \
  --nnodes 1 \
  --nproc-per-node 8 \
  --no-python ./run.sh pretrain_then_finetune.sh \
  --out_dir /mnt/mffuse/out/$MAST_HPC_JOB_NAME \
  --is_on_tc
