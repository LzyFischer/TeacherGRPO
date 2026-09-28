"""Stage 2: distil a (trained) teacher into a small student.

The student (LoRA) is trained on (prompt, response) pairs from ``--train_file``
with one of four losses against the frozen teacher's logits:

    sft          cross-entropy on the response tokens (teacher not used)
    forward      KL(teacher || student)
    reverse      KL(student || teacher)
    generalized  generalised JSD with mixture weight --beta

The training file decides the flavour of distillation: the original dataset
responses (standard KD), teacher samples (SeqKD, see ``generate.py``), or
student samples (on-policy, see ``scripts/run_onpolicy.sh``).

    python train_student.py --teacher_model ckpts/date/teacher/final_lora \\
        --train_file data/date/train.jsonl --val_file data/date/test.jsonl --loss_type forward

Output: ``ckpts/<dataset>/student_model_<run_name>``.
"""

from unsloth import FastModel  # must be imported before transformers

import argparse
import os
import random
import warnings

import torch
import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from teachergrpo.data import collate_fn_builder, infer_dataset_name, load_examples
from teachergrpo.losses import compute_sft_loss, forward_kl_div_loss, generalized_jsd_loss, reverse_kl_div_loss
from teachergrpo.utils import is_lora_adapter, parse_args_with_config, str2bool

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_math_sdp(True)

warnings.filterwarnings("ignore", category=UserWarning, module="unsloth.kernels.utils")
warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}


def distillation_loss(student, teacher, batch, args) -> torch.Tensor:
    input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
    student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
    if args.loss_type == "sft":
        return compute_sft_loss(student_logits, labels)

    with torch.no_grad():
        teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

    # logits[t] predicts token t+1; only score response tokens.
    mask = (labels != -100).float()[:, 1:]
    s_logits, t_logits = student_logits[:, :-1, :], teacher_logits[:, :-1, :]
    if args.loss_type == "forward":
        return forward_kl_div_loss(s_logits, t_logits, mask=mask, temperature=args.temperature)
    if args.loss_type == "reverse":
        return reverse_kl_div_loss(s_logits, t_logits, mask=mask, temperature=args.temperature)
    return generalized_jsd_loss(s_logits, t_logits, mask=mask, beta=args.beta, temperature=args.temperature)


@torch.no_grad()
def evaluate_ce_loss(model, dataloader: DataLoader, accelerator: Accelerator) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in dataloader:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        total += accelerator.gather(out.loss).mean().item()
        n += 1
    model.train()
    return total / n if n else 0.0


def load_models(args, accelerator):
    if args.use_unsloth:
        student, tokenizer = FastModel.from_pretrained(
            model_name=args.student_model, max_seq_length=args.max_length, load_in_4bit=args.load_in_4bit, dtype=args.dt
        )
        if is_lora_adapter(args.student_model):
            # Continue training an existing adapter (e.g. on-policy iterations).
            if accelerator.is_main_process:
                print(f"Resuming from LoRA checkpoint: {args.student_model}")
            FastModel.for_training(student)
        else:
            student = FastModel.get_peft_model(
                student,
                r=args.lora_r,
                target_modules=LORA_TARGET_MODULES,
                lora_alpha=args.lora_alpha,
                lora_dropout=0,
                bias="none",
                finetune_vision_layers=False,
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
        teacher, _ = FastModel.from_pretrained(
            model_name=args.teacher_model, max_seq_length=args.max_length, load_in_4bit=args.load_in_4bit, dtype=args.dt
        )
        FastModel.for_inference(teacher)
        student.config.use_cache = True
        teacher.config.use_cache = True
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True, padding_side="left")
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, torch_dtype=dtype)
        teacher.eval()
        teacher.requires_grad_(False)
        student = AutoModelForCausalLM.from_pretrained(args.student_model, torch_dtype=dtype)
        student.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return student, teacher, tokenizer


def train_student(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_name = args.dataset_name or infer_dataset_name(args.val_file or args.train_file)
    run_name = args.run_name or f"student_{args.loss_type}"
    output_dir = os.path.join("ckpts", dataset_name, f"student_model_{run_name}")

    accelerator = Accelerator(log_with="wandb" if args.use_wandb else None, mixed_precision=args.mixed_precision)
    if args.use_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": {"name": run_name}}
        )
    if accelerator.is_main_process:
        print(f"Teacher: {args.teacher_model} | Student: {args.student_model} | Loss: {args.loss_type}")

    student, teacher, tokenizer = load_models(args, accelerator)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    collate = collate_fn_builder(tokenizer, args.max_length)
    load = lambda path, limit: load_examples(  # noqa: E731
        path, dataset_name, prompt_col=args.prompt_column, resp_col=args.response_column, limit=limit
    )
    train_loader = DataLoader(
        load(args.train_file, args.max_train_samples), batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = None
    if args.val_file:
        val_loader = DataLoader(
            load(args.val_file, args.max_val_samples), batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )

    student, teacher, optimizer, train_loader = accelerator.prepare(student, teacher, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    global_step = 0
    progress_bar = tqdm.tqdm(range(args.num_epochs * len(train_loader)), disable=not accelerator.is_main_process)
    for _ in range(args.num_epochs):
        for batch in train_loader:
            student.train()
            teacher.eval()
            loss = distillation_loss(student, teacher, batch, args)
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            global_step += 1
            progress_bar.update(1)
            if accelerator.is_main_process:
                accelerator.log({f"student/loss_{args.loss_type}": loss.item()}, step=global_step)

            if args.save_steps > 0 and global_step % args.save_steps == 0 and accelerator.is_main_process:
                ckpt_dir = f"{output_dir}_step_{global_step}"
                print(f"Saving checkpoint to {ckpt_dir}")
                accelerator.unwrap_model(student).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

            if val_loader is not None and global_step % args.eval_steps == 0:
                val_loss = evaluate_ce_loss(student, val_loader, accelerator)
                if accelerator.is_main_process:
                    print(f" Step {global_step} | Val CE Loss: {val_loss:.4f}")
                    accelerator.log({"val/ce_loss": val_loss}, step=global_step)

    if accelerator.is_main_process:
        accelerator.unwrap_model(student).save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved student to {output_dir}")
        if args.use_wandb:
            accelerator.end_training()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Models / data
    p.add_argument("--teacher_model", type=str, default="unsloth/Qwen2.5-3B-Instruct",
                   help="HF model or path (a LoRA adapter from train_teacher.py works with Unsloth)")
    p.add_argument("--student_model", type=str, default="unsloth/Qwen2.5-0.5B-Instruct")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--val_file", type=str, default=None, help="Optional; used for validation CE loss")
    p.add_argument("--dataset_name", type=str, default=None,
                   help="Selects the prompt formatter and ckpts/ subdirectory; defaults to the data file's parent dir")
    p.add_argument("--prompt_column", type=str, default="instruction",
                   help="Rows with this field are treated as pre-formatted prompts")
    p.add_argument("--response_column", type=str, default="response")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=50)
    # Optimisation
    p.add_argument("--loss_type", type=str, default="forward", choices=["sft", "forward", "reverse", "generalized"])
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--beta", type=float, default=0.5, help="Mixture weight for --loss_type generalized")
    p.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature")
    p.add_argument("--seed", type=int, default=42)
    # LoRA / precision
    p.add_argument("--use_unsloth", type=str2bool, default=True)
    p.add_argument("--load_in_4bit", type=str2bool, default=True)
    p.add_argument("--lora_r", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=list(DTYPES))
    # Logging / checkpointing
    p.add_argument("--use_wandb", type=str2bool, default=False)
    p.add_argument("--wandb_project", type=str, default="student-phase-distillation")
    p.add_argument("--run_name", type=str, default=None, help="Defaults to student_<loss_type>")
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200, help="Intermediate checkpoints every N steps; 0 = final only")
    args = parse_args_with_config(p)
    args.dt = DTYPES[args.mixed_precision]
    return args


if __name__ == "__main__":
    train_student(parse_args())
