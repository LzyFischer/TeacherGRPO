#!/usr/bin/env bash
# KL-ranked data curriculum: sample N teacher responses per prompt, rank them by
# KL(teacher || student), train one student per rank bucket, then evaluate each.
# Run from the repository root; edit the knobs below.
set -euo pipefail

# ===== knobs =====
DATASETS=("date")
# family|teacher (HF id or LoRA checkpoint)|teacher base|student base
MODEL_SETS=(
  "qwen25|unsloth/Qwen2.5-3B-Instruct|unsloth/Qwen2.5-3B-Instruct|unsloth/Qwen2.5-0.5B-Instruct"
  # "gemma3|unsloth/gemma-3-1b-it|unsloth/gemma-3-1b-it|unsloth/gemma-3-270m-it"
)
GPU=0

NUM_GENERATIONS=5
RANKS=(0 1 2 3 4)          # which rank buckets to train on
GEN_TEMPERATURE=0.5
GEN_MAX_NEW_TOKENS=1024
MAX_DATA_SAMPLES=""        # e.g. 100 for a quick run; empty = full train set

NUM_EPOCHS=2
BATCH_SIZE=4
LOSS_TYPE="forward"        # sft | forward | reverse | generalized
LR="2e-5"
MAX_LENGTH=1024
SAVE_STEPS=0               # 0 = only save the final student

EVAL_MAX_TOKENS=1024
EVAL_TEMPERATURE=0.1
EVAL_LIMIT=""              # e.g. 500; empty = full test set

OUT_DIR="generated/kl_rank"
RESULTS_DIR="results/kl_rank"
# ==================

export CUDA_VISIBLE_DEVICES=${GPU}

for dataset in "${DATASETS[@]}"; do
  train_file="data/${dataset}/train.jsonl"

  for model_set in "${MODEL_SETS[@]}"; do
    IFS='|' read -r family teacher teacher_base student_base <<< "${model_set}"
    work_dir="${OUT_DIR}/${dataset}/${family}"
    mkdir -p "${work_dir}"
    echo "==== ${dataset} / ${family}: teacher=${teacher} student=${student_base} ===="

    echo "== 1) Sample ${NUM_GENERATIONS} teacher responses per prompt"
    python generate.py \
      --model_path "${teacher}" \
      --base_model "${teacher_base}" \
      --input_file "${train_file}" \
      --output_file "${work_dir}/candidates.jsonl" \
      --num_generations "${NUM_GENERATIONS}" \
      --temperature "${GEN_TEMPERATURE}" \
      --max_new_tokens "${GEN_MAX_NEW_TOKENS}" \
      --source teacher_generated \
      ${MAX_DATA_SAMPLES:+--max_samples "${MAX_DATA_SAMPLES}"}

    echo "== 2) Rank candidates by KL -> rank_0..rank_$((NUM_GENERATIONS - 1))"
    python rank_candidates.py \
      --teacher_model "${teacher}" \
      --teacher_base "${teacher_base}" \
      --student_model "${student_base}" \
      --candidate_file "${work_dir}/candidates.jsonl" \
      --output_dir "${work_dir}"

    for i in "${RANKS[@]}"; do
      run_name="${dataset}_${family}_rank${i}"

      echo "== 3) Train student on rank_${i}"
      python train_student.py \
        --teacher_model "${teacher}" \
        --student_model "${student_base}" \
        --dataset_name "${dataset}" \
        --train_file "${work_dir}/rank_${i}.jsonl" \
        --loss_type "${LOSS_TYPE}" \
        --lr "${LR}" \
        --num_epochs "${NUM_EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --max_length "${MAX_LENGTH}" \
        --save_steps "${SAVE_STEPS}" \
        --run_name "${run_name}"

      echo "== 4) Evaluate rank_${i} student"
      python evaluate.py \
        --model_path "ckpts/${dataset}/student_model_${run_name}" \
        --base_model "${student_base}" \
        --datasets "${dataset}" \
        --output_dir "${RESULTS_DIR}" \
        --max_tokens "${EVAL_MAX_TOKENS}" \
        --temperature "${EVAL_TEMPERATURE}" \
        ${EVAL_LIMIT:+--limit "${EVAL_LIMIT}"}
    done
  done
done

echo "All finished."
