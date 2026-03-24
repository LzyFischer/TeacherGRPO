#!/bin/bash

# Define all experiments
experiments=(
    # Baseline experiments
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher"
    # "python train_redi.py --train_file ./data/commonsense_qa/train.jsonl --max_steps 100 --num_epochs 1 --run_name teacher"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma"
    
    # # Ablation: no_curriculum
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_no_curriculum"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_no_curriculum"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_no_curriculum"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_no_curriculum --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_no_curriculum"
    # "python train_redi.py --max_steps 100 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_no_curriculum"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_curriculum --max_train_samples 500 --max_length 2048"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_no_curriculum"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_curriculum"
    
    # # Ablation: no_length
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_no_length --w_length 0.0 --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_no_length --w_length 0.0"
    # "python train_redi.py --max_steps 100 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_no_length --w_length 0.0"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_length --w_length 0.0 --max_train_samples 500 --max_length 2048"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_no_length --w_length 0.0"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_length --w_length 0.0"
    
    # SFT baseline
    # "python train_sft.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_sft"
    # "python train_sft.py --train_file ./data/anli/train.jsonl --num_epochs 2 --run_name teacher_sft"
    # "python train_sft.py --train_file ./data/date/train.jsonl --num_epochs 2 --run_name teacher_sft"
    # "python train_sft.py --train_file ./data/math/train.jsonl --num_epochs 1 --max_train_samples 500 --max_length 2048 --run_name teacher_sft"
    # "python train_sft.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_sft"
    "python train_sft.py --max_steps 800 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_sft"

    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 2 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 2 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --max_train_samples 500 --max_length 2048 --run_name teacher_gemma_sft"
    # "python train_sft.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_sft"
    "python train_sft.py --max_steps 800 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_sft"
    

    # Ablation: no_answer
    # "python train_redi.py --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_no_answer --w_answer_pred 0.0 --max_train_samples 500 --max_length 2048" 
    # "python train_redi.py --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --max_steps 100 --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_no_answer --w_answer_pred 0.0"

    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/strategy_qa/train.jsonl --max_steps 100 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/anli/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/date/train.jsonl --num_epochs 5 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/math/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_answer --w_answer_pred 0.0 --max_train_samples 500 --max_length 2048"
    # "python train_redi.py --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/arc_challenge/train.jsonl --num_epochs 2 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
    # "python train_redi.py --max_steps 100 --teacher_model unsloth/gemma-3-1b-it --student_model unsloth/gemma-3-270m-it --train_file ./data/commonsense_qa/train.jsonl --num_epochs 1 --run_name teacher_gemma_no_answer --w_answer_pred 0.0"
)

# Number of GPUs
CUDA_VISIBLE_DEVICES=1,2,3
NUM_GPUS=3

# Create log directory
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# Task queue index
task_idx=0
total_tasks=${#experiments[@]}

# Store process IDs for each GPU
declare -a gpu_pids

# Initialize GPU state
for i in $(seq 0 $((NUM_GPUS-1))); do
    gpu_pids[$i]=""
done

echo "Scheduling $total_tasks tasks across $NUM_GPUS GPUs"
echo "Logs saved in $LOG_DIR"
echo "======================================"

# Function: run a task on a specific GPU
run_task() {
    local gpu_id=$1
    local task_id=$2
    local cmd=$3
    
    local log_file="$LOG_DIR/task_${task_id}_gpu_${gpu_id}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu_id starting task $task_id: $cmd"
    
    # Run task in background with CUDA_VISIBLE_DEVICES set
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd > $log_file 2>&1 &
    
    # Save process ID
    gpu_pids[$gpu_id]=$!
}

# Function: check if a GPU is free
is_gpu_free() {
    local gpu_id=$1
    local pid=${gpu_pids[$gpu_id]}
    
    if [ -z "$pid" ]; then
        return 0  # Free
    fi
    
    # Check if process is still running
    if ps -p $pid > /dev/null 2>&1; then
        return 1  # Busy
    else
        return 0  # Free
    fi
}

# Main loop: schedule tasks
while [ $task_idx -lt $total_tasks ]; do
    # Check each GPU
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        # If there are remaining tasks and the GPU is free
        if [ $task_idx -lt $total_tasks ] && is_gpu_free $gpu_id; then
            run_task $gpu_id $task_idx "${experiments[$task_idx]}"
            task_idx=$((task_idx+1))
        fi
    done
    
    # Wait before checking again
    sleep 5
done

echo "======================================"
echo "All tasks submitted, waiting for completion..."

# Wait for all tasks to finish
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    pid=${gpu_pids[$gpu_id]}
    if [ -n "$pid" ] && ps -p $pid > /dev/null 2>&1; then
        echo "Waiting for task on GPU $gpu_id to complete (PID: $pid)..."
        wait $pid
    fi
done

echo "======================================"
echo "All tasks completed!"
echo "View logs: ls -lh $LOG_DIR"