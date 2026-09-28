# TeacherGRPO

**Teach the teacher to be learnable.** TeacherGRPO is a knowledge-distillation framework for transferring reasoning from a large LLM to a small one. Before distilling, the *teacher* is fine-tuned with GRPO using a reward that asks for answers that are correct *and* easy for a specific student to imitate. The improved teacher is then distilled into the student with standard (forward/reverse KL, JSD, SFT), sequence-level, or on-policy distillation.

```
            ┌──────────────── Stage 1: train_teacher.py ────────────────┐      ┌──── Stage 2: train_student.py ────┐
 prompts ──▶│ teacher (LoRA) samples G rollouts per prompt              │      │ student (LoRA) learns from the    │
            │ reward = w_verified · correct?                            │ ───▶ │ trained teacher's logits / samples│──▶ evaluate.py
            │        + w_alignment · −KL(teacher ‖ student)  curriculum │      │ loss: sft | forward | reverse |   │
            │        + w_length · SWLP length penalty                   │      │       generalized (JSD)           │
            │        + w_answer_pred · answer-prediction similarity     │      └───────────────────────────────────┘
            └───────────────────────────────────────────────────────────┘
```

## Method

The teacher's GRPO reward (`teachergrpo/rewards.py`) has four terms, all computed with a **frozen copy of the student**:

| Term | What it rewards | Flag |
|---|---|---|
| **Verified** | The extracted final answer matches the reference (1/0). | `--w_verified` (0.4) |
| **Alignment** | Low student–teacher divergence on the rollout. A curriculum moves from top-1 CE (first 30% of training) to top-k KL (to 90%) to full-vocabulary KL, computed over a growing share (30% → 100%) of the highest-divergence tokens. | `--w_alignment` (0.6), `--curriculum_warmup_steps` |
| **SWLP** (Surprisal-Weighted Length Penalty) | Short reasoning. The rollout is split into steps at `.` / `\n`, and every token of a step is penalised by how *unsurprising* the step's first token is to **both** student and teacher, so long redundant steps cost the most. | `--w_length` (0.6), `--swlp_beta`, `--swlp_temperature` |
| **Answer prediction** | Cosine similarity between the student's hidden state over the last `n` tokens and its embedding of the reference answer. | `--w_answer_pred` (0.4), `--n_answer_tokens` |

## Repository layout

```
.
├── train_teacher.py        # Stage 1: TeacherGRPO (TRL GRPOTrainer + Unsloth LoRA)
├── train_student.py        # Stage 2: distil teacher -> student (sft / forward / reverse / generalized)
├── evaluate.py             # vLLM evaluation of base models, checkpoints or LoRA adapters
├── generate.py             # vLLM sampling (SeqKD data, KL-rank candidates, on-policy data)
├── rank_candidates.py      # Rank teacher samples by KL(teacher‖student) into rank_0..rank_{N-1}
│
├── teachergrpo/            # Library code shared by the scripts
│   ├── rewards.py          #   curriculum scheduler, SWLP, alignment & answer-prediction rewards
│   ├── losses.py           #   SFT, forward/reverse KL, generalised JSD
│   ├── data.py             #   dataset registry, prompt formatters, loading, batching
│   ├── answers.py          #   answer extraction + correctness checking (shared by reward and eval)
│   └── utils.py            #   YAML-config arg parsing, LoRA merging
│
├── scripts/
│   ├── run_experiments.py  # GPU-pool launcher for the full grid (teachers, students, evals)
│   ├── run_onpolicy.sh     # iterative on-policy distillation
│   └── run_kl_rank.sh      # generate -> KL-rank -> train one student per rank -> evaluate
│
├── configs/                # example YAML configs for train_teacher.py / train_student.py
└── data/                   # bundled datasets: arc_challenge, commonsense_qa, date, strategy_qa
```

Run every command from the repository root.

## Installation

A CUDA GPU is required (Unsloth, vLLM).

```bash
pip install -r requirements.txt
```

The tested model pairs share a tokenizer within each family, which the alignment reward requires:

| Family | Teacher | Student |
|---|---|---|
| Qwen 2.5 | `unsloth/Qwen2.5-3B-Instruct` | `unsloth/Qwen2.5-0.5B-Instruct` |
| Gemma 3 | `unsloth/gemma-3-1b-it` | `unsloth/gemma-3-270m-it` |

## Quick start (Date Understanding, Qwen 2.5)

```bash
# 1. Train the teacher with TeacherGRPO -> ckpts/date/teacher/final_lora
python train_teacher.py \
    --teacher_model unsloth/Qwen2.5-3B-Instruct \
    --student_model unsloth/Qwen2.5-0.5B-Instruct \
    --train_file data/date/train.jsonl \
    --num_epochs 5 --run_name teacher

# 2. Distil it into the student -> ckpts/date/student_model_qwen_kl
python train_student.py \
    --teacher_model ckpts/date/teacher/final_lora \
    --student_model unsloth/Qwen2.5-0.5B-Instruct \
    --train_file data/date/train.jsonl --val_file data/date/test.jsonl \
    --loss_type forward --num_epochs 2 --run_name qwen_kl

# 3. Evaluate (LoRA adapters are merged into --base_model automatically)
python evaluate.py \
    --model_path ckpts/date/student_model_qwen_kl \
    --base_model unsloth/Qwen2.5-0.5B-Instruct \
    --datasets date
# -> results/date/student_model_qwen_kl/eval_results.json
```

Both training scripts accept `--config <yaml>` (see `configs/`). Values from the file replace the argparse defaults, and flags given on the command line override the file:

```bash
python train_teacher.py --config configs/teacher_date_qwen.yaml --use_wandb False
```

Run any script with `--help` to see all options.

## Distillation variants

`train_student.py` always trains on `(prompt, response)` pairs from `--train_file`. The source of those responses decides the kind of distillation:

| Variant | Training responses | How |
|---|---|---|
| **SFT** | dataset responses | `--loss_type sft` |
| **Word-level KD** | dataset responses, matching teacher logits | `--loss_type forward` / `reverse` / `generalized` (`--beta`) |
| **SeqKD** | samples from the teacher | `generate.py` on the teacher, then `train_student.py --loss_type sft` |
| **On-policy** | samples from the current student, iterated | `scripts/run_onpolicy.sh` (JSD against the teacher) |
| **KL-ranked curriculum** | the teacher sample the student finds easiest | `scripts/run_kl_rank.sh` |

For example, SeqKD:

```bash
python generate.py --model_path ckpts/date/teacher/final_lora --base_model unsloth/Qwen2.5-3B-Instruct \
    --input_file data/date/train.jsonl --output_file generated/seqkd/date.jsonl --temperature 1.0
python train_student.py --teacher_model ckpts/date/teacher/final_lora \
    --train_file generated/seqkd/date.jsonl --val_file data/date/test.jsonl --loss_type sft --run_name qwen_seqkd
```

## Running the full experiment grid

`scripts/run_experiments.py` puts every job in a queue and runs one job per GPU at a time. Logs go to `logs/<command>/`, and jobs whose inputs are missing are skipped.

```bash
python scripts/run_experiments.py teacher      --gpus 0 1 2 3  # teachers × {main, no_length, no_answer}
python scripts/run_experiments.py student      --gpus 0 1 2 3  # {sft, kl, seqkd, onpolicy} students for every teacher, then eval
python scripts/run_experiments.py eval-teacher --gpus 0 1 2 3  # trained teachers  -> results/teachers
python scripts/run_experiments.py eval-base    --gpus 0 1 2 3  # untrained models  -> results/base
```

Narrow the grid with `--datasets`, `--families {qwen,gemma}`, `--ablations` and `--methods`. The per-dataset teacher schedule (epochs / max steps) is defined in `TEACHER_SCHEDULE` at the top of the script. A teacher for ablation `X` is saved to `ckpts/<dataset>/teacher_X/final_lora` (`teacher_gemma_X` for Gemma; `main` has no suffix).

## Data

Bundled: `arc_challenge`, `commonsense_qa`, `date`, `strategy_qa` (`data/<name>/{train,test}.jsonl`). Each training row includes a reference chain-of-thought `response`.

The code also supports `anli`, `math`, `gsm8k` and `table_mwp`. To use one, add `data/<name>/{train,test}.jsonl`. The dataset name, taken from the directory, selects both the prompt formatter and the answer checker:

| Answer type | Datasets | Reference field | Prediction extracted from |
|---|---|---|---|
| `mcq` | arc_challenge, commonsense_qa, date | `answerKey` | "The best answer is X" and similar |
| `yesno` | strategy_qa | `answer` (bool) | last True/False |
| `anli` | anli | `label` (entailment/neutral/contradiction) | True / Neither / False |
| `math` | math, gsm8k, table_mwp | `answer` | last `\boxed{}` (or last number) |

To add a new dataset, register a formatter and answer type in `DATASET_REGISTRY` (`teachergrpo/data.py`). Rows that already contain an `instruction` field, such as files written by `generate.py` or `rank_candidates.py`, are used as-is without reformatting.

## Notes and known limitations

- **Teacher epochs.** `train_teacher.py` does not forward `--num_epochs` to `GRPOConfig`, so without `--max_steps` TRL trains for its default number of epochs (3). `--num_epochs` only sizes the curriculum schedule. This matches the code the reported runs were made with. Use `--max_steps` for precise control.
- **`no_curriculum` ablation.** There is no flag to disable the curriculum, so `run_experiments.py teacher` does not produce it. `student` / `eval-teacher` will still use `teacher_no_curriculum` checkpoints if you train them separately. The same applies to an SFT-trained teacher (`teacher_sft`).
- The teacher's reward is computed with the **student's tokenizer**, so teacher and student must come from the same model family.

## License

This project is released under the MIT License.
