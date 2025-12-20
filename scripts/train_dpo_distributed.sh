#!/bin/bash
# Distributed training script for Safe-MM-DPO
# Usage: bash scripts/train_dpo_distributed.sh <num_gpus> [--background]
# Example: bash scripts/train_dpo_distributed.sh 8
# Example (background): bash scripts/train_dpo_distributed.sh 8 --background

set -e

NUM_GPUS=${1:-8}
BACKGROUND_MODE=false

# Check if background mode is requested
if [ "$2" == "--background" ] || [ "$2" == "-b" ]; then
    BACKGROUND_MODE=true
fi

# Create logs directory if it doesn't exist
LOG_DIR="logs/training/safe_mm_dpo"
mkdir -p "$LOG_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

echo "Starting distributed Safe-MM-DPO training with ${NUM_GPUS} GPUs"
if [ "$BACKGROUND_MODE" = true ]; then
    echo "Running in background mode. Logs: ${LOG_FILE}"
    echo "PID file: ${PID_FILE}"
fi

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Function to run training
run_training() {
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29501 \
        -m src.training.train_safe_mm_dpo \
        --config configs/dpo_config.yaml \
        --output_dir models/aligned \
        --logging_dir logs/training/safe_mm_dpo
}

if [ "$BACKGROUND_MODE" = true ]; then
    # Run in background with nohup
    nohup bash -c "run_training 2>&1 | tee ${LOG_FILE}" > /dev/null 2>&1 &
    TRAIN_PID=$!
    
    # Save PID to file
    echo $TRAIN_PID > "${PID_FILE}"
    
    echo "Training started in background!"
    echo "Process ID (PID): ${TRAIN_PID}"
    echo "Log file: ${LOG_FILE}"
    echo "PID file: ${PID_FILE}"
    echo ""
    echo "To monitor training:"
    echo "  tail -f ${LOG_FILE}"
    echo ""
    echo "To check if training is running:"
    echo "  ps -p ${TRAIN_PID}"
    echo ""
    echo "To stop training:"
    echo "  kill ${TRAIN_PID}"
else
    # Run in foreground, but still log to file
    echo "Logs will be saved to: ${LOG_FILE}"
    run_training 2>&1 | tee "${LOG_FILE}"
    echo "Training completed!"
fi
