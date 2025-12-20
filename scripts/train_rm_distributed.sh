#!/bin/bash
# Distributed training script for Reward Models (Helpful-RM / Harmless-RM)
# Usage: bash scripts/train_rm_distributed.sh <task> <num_gpus>
# Example: bash scripts/train_rm_distributed.sh helpful 8

set -e

TASK=${1:-helpful}
NUM_GPUS=${2:-8}

if [ "$TASK" != "helpful" ] && [ "$TASK" != "harmless" ]; then
    echo "Error: Task must be 'helpful' or 'harmless'"
    exit 1
fi

echo "Starting distributed training for ${TASK}_rm with ${NUM_GPUS} GPUs"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use torchrun for distributed training
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task ${TASK} \
    --output_dir models/${TASK}_rm

echo "Training completed!"
