#!/bin/bash
# Script to check and manage training processes
# Usage: bash scripts/check_training.sh [status|stop|logs]

ACTION=${1:-status}
LOG_DIR="logs/training"

case $ACTION in
    status)
        echo "=== Checking Training Processes ==="
        echo ""
        
        # Check RM training
        if [ -d "${LOG_DIR}/rm" ]; then
            echo "Reward Model Training:"
            for pid_file in ${LOG_DIR}/rm/*.pid; do
                if [ -f "$pid_file" ]; then
                    pid=$(cat "$pid_file")
                    task=$(basename "$pid_file" .pid | sed 's/_[0-9]*$//')
                    timestamp=$(basename "$pid_file" .pid | sed 's/^[^_]*_//')
                    if ps -p $pid > /dev/null 2>&1; then
                        echo "  [RUNNING] ${task} (PID: $pid, Started: $timestamp)"
                        echo "    Log: ${LOG_DIR}/rm/${task}_${timestamp}.log"
                    else
                        echo "  [STOPPED] ${task} (PID: $pid, Started: $timestamp)"
                    fi
                fi
            done
            echo ""
        fi
        
        # Check DPO training
        if [ -d "${LOG_DIR}/safe_mm_dpo" ]; then
            echo "Safe-MM-DPO Training:"
            for pid_file in ${LOG_DIR}/safe_mm_dpo/*.pid; do
                if [ -f "$pid_file" ]; then
                    pid=$(cat "$pid_file")
                    timestamp=$(basename "$pid_file" .pid | sed 's/train_//')
                    if ps -p $pid > /dev/null 2>&1; then
                        echo "  [RUNNING] DPO Training (PID: $pid, Started: $timestamp)"
                        echo "    Log: ${LOG_DIR}/safe_mm_dpo/train_${timestamp}.log"
                    else
                        echo "  [STOPPED] DPO Training (PID: $pid, Started: $timestamp)"
                    fi
                fi
            done
            echo ""
        fi
        
        # Check GPU usage
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Usage:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
                awk -F', ' '{printf "  GPU %s (%s): %s%% util, %s/%s MB\n", $1, $2, $3, $4, $5}'
        fi
        ;;
    
    stop)
        echo "=== Stopping All Training Processes ==="
        
        # Stop RM training
        if [ -d "${LOG_DIR}/rm" ]; then
            for pid_file in ${LOG_DIR}/rm/*.pid; do
                if [ -f "$pid_file" ]; then
                    pid=$(cat "$pid_file")
                    task=$(basename "$pid_file" .pid | sed 's/_[0-9]*$//')
                    if ps -p $pid > /dev/null 2>&1; then
                        echo "Stopping ${task} training (PID: $pid)..."
                        kill $pid
                        rm "$pid_file"
                    else
                        echo "${task} training already stopped. Cleaning up PID file..."
                        rm "$pid_file"
                    fi
                fi
            done
        fi
        
        # Stop DPO training
        if [ -d "${LOG_DIR}/safe_mm_dpo" ]; then
            for pid_file in ${LOG_DIR}/safe_mm_dpo/*.pid; do
                if [ -f "$pid_file" ]; then
                    pid=$(cat "$pid_file")
                    if ps -p $pid > /dev/null 2>&1; then
                        echo "Stopping DPO training (PID: $pid)..."
                        kill $pid
                        rm "$pid_file"
                    else
                        echo "DPO training already stopped. Cleaning up PID file..."
                        rm "$pid_file"
                    fi
                fi
            done
        fi
        
        echo "Done!"
        ;;
    
    logs)
        # Show latest log file
        LATEST_LOG=$(find ${LOG_DIR} -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$LATEST_LOG" ]; then
            echo "Showing latest log: $LATEST_LOG"
            echo "Press Ctrl+C to exit"
            tail -f "$LATEST_LOG"
        else
            echo "No log files found in ${LOG_DIR}"
        fi
        ;;
    
    *)
        echo "Usage: bash scripts/check_training.sh [status|stop|logs]"
        echo "  status: Show status of all training processes"
        echo "  stop:   Stop all training processes"
        echo "  logs:   Show latest training log (tail -f)"
        exit 1
        ;;
esac
