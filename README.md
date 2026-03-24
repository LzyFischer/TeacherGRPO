# TeacherGRPO

This repository contains the official implementation for **TeacherGRPO**, a knowledge distillation framework for transferring reasoning capabilities from a large teacher LLM to a smaller student LLM.

TeacherGRPO trains the teacher with curriculum-aware GRPO (using difficulty and answer-prediction rewards) and then distills its improved reasoning traces into the student via a generalized JSD objective.

---

## Method Overview

```
Teacher Fine-tuning (train_redi.py)
  ├── GRPO with verifiable correctness reward
  ├── Surprisal-Weighted Length Penalty (SWLP) — penalizes redundant reasoning steps
  ├── Answer-prediction alignment reward
  └── Curriculum scheduler: top1_kl → topk_kl → full_kl

Student Distillation (train_student.py)
  ├── Supervised mode: distill from dataset responses
  ├── On-policy mode: student generates, teacher scores
  ├── Teacher-generated mode: teacher generates, student imitates
  └── Loss options: forward KL | reverse KL | generalized JSD | SFT
```

---

## Repository Structure

```
.
├── train_redi.py              # Main teacher fine-tuning (GRPO + curriculum + SWLP)
├── train_student.py           # Student distillation training
├── train_teacher.py           # Teacher-only GRPO training (without student distillation)
├── eval_vllm.py               # Evaluation script using vLLM (all datasets)
├── generate_candidates.py     # Generate candidate responses from teacher (vLLM)
├── generate_dataset_vllm.py   # Generate student on-policy data (vLLM)
├── rank_candidates_by_kl.py   # Rank candidates by KL divergence (curriculum data)
├── merge_dataset.py           # Merge original + generated datasets
├── batch_scheduler.py         # Multi-GPU batch job scheduler
├── run_eval_teacher.py        # Automated parallel teacher evaluation runner
├── load_data.py               # Dataset loaders and prompt formatters
├── utils.py                   # Loss functions, generation, reward utilities
├── config.yaml                # Example configuration file
│
├── pre_1.sh                   # End-to-end pipeline: generate → rank → train → eval
├── run_iterative_pipeline.sh  # Adaptive off-line method
├── run_teacher.sh             # Example teacher training commands
├── run_eval_teacher_base.sh   # Parallel baseline evaluation (4 GPUs)
│
└── data/
    ├── arc_challenge/         # ARC-Challenge (train/test)
    ├── commonsense_qa/        # CommonsenseQA (train/test)
    ├── date/                  # BIG-Bench Date Understanding (train/test)
    ├── strategy_qa/           # StrategyQA (train/test)
    ├── filter_correct.py      # Filter correctly-answered examples
    ├── math_utils.py          # Math answer normalization utilities
    └── utils.py               # Data processing helpers
```

---

## Installation

```bash
pip install unsloth transformers accelerate trl vllm peft datasets wandb
```

> **Note**: [Unsloth](https://github.com/unslothai/unsloth) is used for memory-efficient LoRA training. vLLM is required for fast generation and evaluation. Both require CUDA.

---

## Quick Start

### 1. Teacher Fine-tuning with TeacherGRPO

Train the teacher model with GRPO and curriculum-aware distillation alignment:

```bash
python train_redi.py \
    --teacher_model unsloth/Qwen2.5-3B-Instruct \
    --student_model unsloth/Qwen2.5-0.5B-Instruct \
    --train_file ./data/date/train.jsonl \
    --num_epochs 5 \
    --run_name teacher_date
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--teacher_model` | `unsloth/Qwen2.5-3B-Instruct` | Teacher model path or HF name |
| `--student_model` | `unsloth/Qwen2.5-0.5B-Instruct` | Student model (used to compute alignment reward) |
| `--train_file` | — | Path to training JSONL |
| `--w_verified` | `0.4` | Weight for verifiable correctness reward |
| `--w_alignment` | `0.6` | Weight for alignment (KL-based) reward |
| `--w_length` | `0.6` | Weight for SWLP length penalty |
| `--w_answer_pred` | `0.4` | Weight for answer-prediction reward |
| `--curriculum_warmup_steps` | `0` | Steps before curriculum starts |
| `--num_teacher_samples` | `4` | GRPO rollouts per prompt |
| `--lora_r` | `8` | LoRA rank |
| `--save_steps` | `20` | Checkpoint every N steps |
| `--use_wandb` | `True` | Log to W&B |

---

### 2. Student Distillation

Distill the fine-tuned teacher into a smaller student:

```bash
python train_student.py \
    --teacher_model ./ckpts/date/teacher_date \
    --student_model unsloth/Qwen2.5-0.5B-Instruct \
    --train_file ./data/date/train.jsonl \
    --val_file ./data/date/test.jsonl \
    --loss_type forward \
    --student_mode supervised \
    --num_epochs 2 \
    --run_name student_date
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--loss_type` | — | `sft` \| `forward` \| `reverse` \| `generalized` |
| `--student_mode` | — | `supervised` \| `on_policy` \| `teacher_generated` |
| `--beta` | `0.5` | Interpolation weight in generalized JSD |
| `--temperature` | `1.0` | Distillation temperature |
| `--lr` | — | Student learning rate |

---

### 3. Evaluation

Evaluate any model (base or fine-tuned) across supported datasets:

```bash
python eval_vllm.py \
    --model_path ./ckpts/date/student_model_student_date \
    --base_model unsloth/Qwen2.5-0.5B-Instruct \
    --datasets date arc_challenge commonsense_qa \
    --output_dir results
```

Supported datasets: `date`, `arc_challenge`, `commonsense_qa`, `strategy_qa`, `anli`, `math`, `gsm8k`.

---

## Supported Models

The codebase has been tested with the following model families:

| Family | Teacher | Student |
|---|---|---|
| Qwen 2.5 | `unsloth/Qwen2.5-3B-Instruct` | `unsloth/Qwen2.5-0.5B-Instruct` |
| Gemma 3 | `unsloth/gemma-3-1b-it` | `unsloth/gemma-3-270m-it` |

Any instruction-tuned causal LM with a chat template should be compatible.

---

## Configuration File

Most hyperparameters can be specified in a YAML config and loaded with `--config`:

```bash
python train_redi.py --config config.yaml
```

See `config.yaml` for a complete example. Command-line arguments override config values.

---

## License

This project is released under the MIT License.
