"""Stage 1: train the teacher with TeacherGRPO.

GRPO (TRL + Unsloth, LoRA) on the teacher, rewarded for being correct *and*
easy for a frozen student to imitate (see ``teachergrpo/rewards.py``).

    python train_teacher.py --train_file data/date/train.jsonl --num_epochs 5 --run_name teacher

Output: ``ckpts/<dataset>/<run_name>/final_lora`` (LoRA adapter + tokenizer).
"""

from unsloth import FastLanguageModel  # must be imported before trl / transformers

import argparse
import os
import warnings

import wandb
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from teachergrpo.answers import extract_reference
from teachergrpo.data import get_dataset_config, infer_dataset_name, read_records
from teachergrpo.rewards import CurriculumRewardFunction, CurriculumScheduler
from teachergrpo.utils import is_lora_adapter, parse_args_with_config, str2bool

warnings.filterwarnings("ignore")

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class CurriculumCallback(TrainerCallback):
    """Advances the curriculum each step and logs the reward breakdown."""

    def __init__(self, scheduler: CurriculumScheduler, reward_fn: CurriculumRewardFunction):
        self.scheduler = scheduler
        self.reward_fn = reward_fn

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and wandb.run is not None:
            wandb.define_metric("reward/*", step_metric="train/global_step")
            wandb.define_metric("curriculum/*", step_metric="train/global_step")
            wandb.define_metric("train/global_step", summary="max")

    def on_step_end(self, args, state, control, **kwargs):
        self.scheduler.step()
        metrics = dict(self.reward_fn.last_metrics or {})
        if metrics:
            print("[Rewards] " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            if state.is_world_process_zero and wandb.run is not None:
                metrics["train/global_step"] = state.global_step
                wandb.log(metrics)
        if state.global_step % 10 == 0:
            print(f"\n[Curriculum] Step {state.global_step} | Progress: {self.scheduler.get_progress():.2%}")


def build_dataset(records, dataset_config) -> Dataset:
    formatter, eval_type = dataset_config["formatter"], dataset_config["type"]
    prompts, references = [], []
    for row in records:
        try:
            prompt_text, _ = formatter(row)
            references.append(extract_reference(row, eval_type))
            prompts.append([{"role": "user", "content": prompt_text}])
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return Dataset.from_dict({"prompt": prompts, "reference": references})


def train_teacher(args):
    dataset_name = args.dataset_name or infer_dataset_name(args.train_file)
    dataset_config = get_dataset_config(dataset_name)
    eval_type = dataset_config["type"]
    print(f"Dataset: {dataset_name} | answer type: {eval_type}")

    records = read_records(args.train_file, limit=args.max_train_samples)

    steps_per_epoch = len(records) // (args.batch_size * args.gradient_accumulation_steps)
    if args.max_steps > 0:
        total_steps = args.max_steps
        print(f"Training for {total_steps} steps.")
    else:
        total_steps = steps_per_epoch * args.num_epochs
        print(f"Curriculum sized for {args.num_epochs} epochs (~{total_steps} steps).")
    curriculum_scheduler = CurriculumScheduler(total_steps=total_steps, warmup_steps=args.curriculum_warmup_steps)

    # --- Teacher (trainable, LoRA) ---
    print(f"Loading teacher: {args.teacher_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.teacher_model,
        max_seq_length=args.max_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=True,
        max_lora_rank=args.lora_r,
        gpu_memory_utilization=0.25,
    )
    if is_lora_adapter(args.teacher_model):
        # Unsloth loads adapters in inference mode; re-enable gradients.
        print(f"Resuming from LoRA checkpoint: {args.teacher_model}")
        FastLanguageModel.for_training(model)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=args.lora_alpha,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    # --- Student (frozen, only used to compute rewards) ---
    print(f"Loading student for rewards: {args.student_model}")
    student_model, student_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.student_model,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        dtype=None,
        gpu_memory_utilization=0.05,
    )
    FastLanguageModel.for_inference(student_model)

    train_dataset = build_dataset(records, dataset_config)
    print(f"Processed {len(train_dataset)} training examples.")

    # Teacher and student must share a tokenizer (same model family).
    reward_function = CurriculumRewardFunction(
        student_model=student_model,
        teacher_model=model,
        tokenizer=student_tokenizer,
        curriculum_scheduler=curriculum_scheduler,
        eval_type=eval_type,
        device="cuda",
        w_verified=args.w_verified,
        w_alignment=args.w_alignment,
        w_length=args.w_length,
        w_answer_pred=args.w_answer_pred,
        n_answer_tokens=args.n_answer_tokens,
        swlp_beta=args.swlp_beta,
        swlp_temperature=args.swlp_temperature,
        max_length=args.max_new_tokens,
    )

    output_dir = os.path.join("ckpts", dataset_name, args.run_name)
    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=args.run_name,
        learning_rate=args.teacher_lr,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_teacher_samples,
        max_prompt_length=args.max_length // 2,
        max_completion_length=args.max_new_tokens,
        max_steps=args.max_steps,
        save_steps=args.save_steps if args.save_steps > 0 else 100,
        max_grad_norm=1.0,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.add_callback(CurriculumCallback(curriculum_scheduler, reward_function))

    print("Starting GRPO training with curriculum...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final_lora")
    model.save_lora(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Adapter saved to {final_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Models / data
    p.add_argument("--teacher_model", type=str, default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--student_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train_file", type=str, default="./data/date/train.jsonl")
    p.add_argument("--dataset_name", type=str, default=None, help="Defaults to the train_file's parent directory")
    p.add_argument("--max_train_samples", type=int, default=None)
    # Optimisation
    p.add_argument("--teacher_lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=1,
                   help="Sizes the curriculum schedule (not forwarded to GRPOConfig; see README)")
    p.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides --num_epochs")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--num_teacher_samples", type=int, default=4, help="GRPO rollouts per prompt")
    p.add_argument("--curriculum_warmup_steps", type=int, default=0)
    # Reward weights
    p.add_argument("--w_verified", type=float, default=0.4, help="Correctness reward weight")
    p.add_argument("--w_alignment", type=float, default=0.6, help="Student-teacher alignment reward weight")
    p.add_argument("--w_length", type=float, default=0.6, help="SWLP length reward weight")
    p.add_argument("--w_answer_pred", type=float, default=0.4, help="Answer-prediction reward weight")
    p.add_argument("--swlp_beta", type=float, default=0.01, help="SWLP penalty coefficient")
    p.add_argument("--swlp_temperature", type=float, default=4, help="SWLP surprisal temperature")
    p.add_argument("--n_answer_tokens", type=int, default=10, help="Tokens used for the answer-prediction reward")
    # LoRA / misc
    p.add_argument("--load_in_4bit", type=str2bool, default=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--use_wandb", type=str2bool, default=True)
    p.add_argument("--run_name", type=str, default="teacher")
    p.add_argument("--save_steps", type=int, default=20)
    return parse_args_with_config(p)


if __name__ == "__main__":
    train_teacher(parse_args())
