#!/bin/bash
# Distributed training script for Safe-MM-DPO
# Usage: bash scripts/train_dpo_distributed.sh <num_gpus>
# Example: bash scripts/train_dpo_distributed.sh 8

set -e

NUM_GPUS=${1:-8}

echo "Starting distributed Safe-MM-DPO training with ${NUM_GPUS} GPUs"

# Use torchrun for distributed training
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo

echo "Training completed!"
