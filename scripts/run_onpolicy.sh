#!/usr/bin/env bash
# Iterative on-policy distillation: repeat {student samples responses -> train on them against the teacher}.
# Run from the repository root. Prints the final student checkpoint path as the last line.
#
#   bash scripts/run_onpolicy.sh -i 2 -b unsloth/Qwen2.5-0.5B-Instruct -t ckpts/date/teacher/final_lora \
#       -f data/date/train.jsonl -v data/date/test.jsonl -n qwen_onpolicy
set -eo pipefail

ITERATIONS=3
BASE_MODEL="unsloth/Qwen2.5-0.5B-Instruct"
TEACHER_MODEL="unsloth/Qwen2.5-3B-Instruct"
TRAIN_DATA="./data/date/train.jsonl"
VAL_DATA="./data/date/test.jsonl"
START_STUDENT=""
RUN_SUFFIX="default"

usage() {
    cat <<EOF
Usage: $0 [options]
  -i <int>  Number of iterations (default: $ITERATIONS)
  -b <str>  Base student model (default: $BASE_MODEL)
  -t <str>  Teacher model or LoRA checkpoint (default: $TEACHER_MODEL)
  -f <str>  Training file providing the prompts (default: $TRAIN_DATA)
  -v <str>  Validation file; its parent directory names the dataset (default: $VAL_DATA)
  -s <str>  Student checkpoint to start from (default: the base student)
  -n <str>  Run name suffix (default: $RUN_SUFFIX)
EOF
    exit 1
}

while getopts "i:b:t:f:v:s:n:h" opt; do
    case ${opt} in
        i) ITERATIONS=$OPTARG ;;
        b) BASE_MODEL=$OPTARG ;;
        t) TEACHER_MODEL=$OPTARG ;;
        f) TRAIN_DATA=$OPTARG ;;
        v) VAL_DATA=$OPTARG ;;
        s) START_STUDENT=$OPTARG ;;
        n) RUN_SUFFIX=$OPTARG ;;
        *) usage ;;
    esac
done

DATASET=$(basename "$(dirname "$VAL_DATA")")
WORK_DIR="./generated/onpolicy/${DATASET}_${RUN_SUFFIX}"
mkdir -p "$WORK_DIR"
STUDENT="${START_STUDENT:-$BASE_MODEL}"

echo ">>> On-policy distillation: $RUN_SUFFIX on $DATASET ($ITERATIONS iterations)"

for ((i = 1; i <= ITERATIONS; i++)); do
    GEN_FILE="${WORK_DIR}/iter_${i}_data.jsonl"
    RUN_NAME="iter_${i}_${RUN_SUFFIX}"

    echo ">>> [iter $i] sampling from $STUDENT"
    python generate.py \
        --model_path "$STUDENT" \
        --base_model "$BASE_MODEL" \
        --input_file "$TRAIN_DATA" \
        --output_file "$GEN_FILE" \
        --max_new_tokens 1024 \
        --temperature 1.0 \
        --source student_generated > "${WORK_DIR}/iter_${i}_gen.log" 2>&1

    echo ">>> [iter $i] training $RUN_NAME"
    python train_student.py \
        --teacher_model "$TEACHER_MODEL" \
        --student_model "$STUDENT" \
        --train_file "$GEN_FILE" \
        --val_file "$VAL_DATA" \
        --run_name "$RUN_NAME" \
        --loss_type generalized \
        --beta 0.5 \
        --lr 2e-5 \
        --batch_size 4 \
        --num_epochs 1 \
        --save_steps 0 \
        --max_length 1024 \
        --use_wandb False > "${WORK_DIR}/iter_${i}_train.log" 2>&1

    STUDENT="ckpts/${DATASET}/student_model_${RUN_NAME}"
    if [ ! -d "$STUDENT" ]; then
        echo "ERROR: checkpoint $STUDENT not found (see ${WORK_DIR}/iter_${i}_train.log)" >&2
        exit 1
    fi
done

echo "$STUDENT"
