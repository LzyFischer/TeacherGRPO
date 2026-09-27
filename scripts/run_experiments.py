"""Run the full experiment grid on a pool of GPUs (one job per GPU at a time).

Run from the repository root:

    python scripts/run_experiments.py teacher      --gpus 0 1 2 3   # train teachers (+ ablations)
    python scripts/run_experiments.py student      --gpus 0 1 2 3   # distil every teacher into students, then evaluate
    python scripts/run_experiments.py eval-teacher --gpus 0 1 2 3   # evaluate trained teachers
    python scripts/run_experiments.py eval-base    --gpus 0 1 2 3   # evaluate untrained teachers and students

Every sub-command accepts ``--datasets``, ``--families`` and (where relevant)
``--ablations`` / ``--methods`` to narrow the grid. Logs go to ``logs/<command>/``;
jobs whose inputs are missing are skipped with a warning.

Checkpoint naming: a teacher trained for ablation ``X`` on dataset ``D`` lives in
``ckpts/D/<teacher_prefix>_X/final_lora`` (``main`` has no suffix).
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

DATASETS = ["anli", "date", "math", "arc_challenge", "commonsense_qa", "strategy_qa"]

FAMILIES = {
    "qwen": {
        "teacher": "unsloth/Qwen2.5-3B-Instruct",
        "student": "unsloth/Qwen2.5-0.5B-Instruct",
        "teacher_prefix": "teacher",
    },
    "gemma": {
        "teacher": "unsloth/gemma-3-1b-it",
        "student": "unsloth/gemma-3-270m-it",
        "teacher_prefix": "teacher_gemma",
    },
}

# Teacher training schedule per dataset.
TEACHER_SCHEDULE = {
    "strategy_qa":    ["--max_steps", "100"],
    "anli":           ["--num_epochs", "5"],
    "date":           ["--num_epochs", "5"],
    "math":           ["--num_epochs", "1", "--max_train_samples", "500", "--max_length", "2048"],
    "arc_challenge":  ["--num_epochs", "2"],
    "commonsense_qa": ["--max_steps", "100", "--num_epochs", "1"],
}

# Teacher ablations that can be expressed with train_teacher.py flags.
TEACHER_ABLATIONS = {
    "main":      [],
    "no_length": ["--w_length", "0.0"],
    "no_answer": ["--w_answer_pred", "0.0"],
}

STUDENT_METHODS = ["sft", "kl", "seqkd", "onpolicy"]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, gpu: int, log_file: str, append: bool = False) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(log_file, "a" if append else "w") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def run_pool(jobs, gpus):
    """Run ``job(gpu)`` callables, one worker thread per GPU."""
    q = queue.Queue()
    for job in jobs:
        q.put(job)
    log(f"{q.qsize()} jobs on GPUs {gpus}")

    def worker(gpu):
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            try:
                job(gpu)
            except Exception as e:  # keep the pool alive
                log(f"EXCEPTION on GPU {gpu}: {e}")

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log("All jobs finished.")


def suffix(ablation: str) -> str:
    return "" if ablation == "main" else f"_{ablation}"


def teacher_ckpt(dataset: str, family: dict, ablation: str) -> str:
    return f"ckpts/{dataset}/{family['teacher_prefix']}{suffix(ablation)}/final_lora"


def data_file(dataset: str, split: str) -> str:
    return f"data/{dataset}/{split}.jsonl"


def exists(path: str) -> bool:
    return os.path.exists(os.path.join(ROOT, path))


# -----------------------------------------------------------------------------
# Sub-commands
# -----------------------------------------------------------------------------

def teacher_jobs(args, log_dir):
    for ablation in args.ablations:
        for dataset in args.datasets:
            for name in args.families:
                family = FAMILIES[name]
                run_name = f"{family['teacher_prefix']}{suffix(ablation)}"

                def job(gpu, dataset=dataset, family=family, ablation=ablation, run_name=run_name):
                    if not exists(data_file(dataset, "train")):
                        return log(f"SKIP {dataset}/{run_name}: no training data")
                    cmd = [PY, "train_teacher.py",
                           "--teacher_model", family["teacher"], "--student_model", family["student"],
                           "--train_file", data_file(dataset, "train"), "--run_name", run_name,
                           *TEACHER_SCHEDULE.get(dataset, []), *TEACHER_ABLATIONS[ablation]]
                    log(f"GPU {gpu} START {dataset}/{run_name}")
                    rc = run(cmd, gpu, os.path.join(log_dir, f"{dataset}_{run_name}.log"))
                    log(f"GPU {gpu} {'DONE' if rc == 0 else 'FAIL'} {dataset}/{run_name}")

                yield job


def student_jobs(args, log_dir):
    for ablation in args.ablations:
        for dataset in args.datasets:
            for name in args.families:
                for method in args.methods:
                    family = FAMILIES[name]
                    run_id = f"{name}{suffix(ablation)}_{method}"

                    def job(gpu, dataset=dataset, family=family, ablation=ablation, method=method, run_id=run_id):
                        teacher = teacher_ckpt(dataset, family, ablation)
                        if not exists(teacher):
                            return log(f"SKIP {dataset}/{run_id}: teacher missing ({teacher})")
                        log_file = os.path.join(log_dir, f"{dataset}_{run_id}.log")
                        train_file, val_file = data_file(dataset, "train"), data_file(dataset, "test")
                        epochs = "2" if dataset in ("anli", "date") else "1"
                        log(f"GPU {gpu} START {dataset}/{run_id}")

                        def train(train_path, loss_type, append=False):
                            return run([PY, "train_student.py",
                                        "--teacher_model", teacher, "--student_model", family["student"],
                                        "--train_file", train_path, "--val_file", val_file,
                                        "--run_name", run_id, "--loss_type", loss_type,
                                        "--num_epochs", epochs, "--use_wandb", "False"], gpu, log_file, append)

                        model_path = f"ckpts/{dataset}/student_model_{run_id}"
                        if method == "sft":
                            ok = train(train_file, "sft") == 0
                        elif method == "kl":
                            ok = train(train_file, "forward") == 0
                        elif method == "seqkd":
                            gen_file = f"generated/seqkd/{dataset}_{run_id}.jsonl"
                            ok = run([PY, "generate.py", "--model_path", teacher, "--base_model", family["teacher"],
                                      "--input_file", train_file, "--output_file", gen_file,
                                      "--temperature", "1.0", "--source", "teacher_generated"], gpu, log_file) == 0
                            ok = ok and train(gen_file, "sft", append=True) == 0
                        else:  # onpolicy
                            ok = run(["bash", "scripts/run_onpolicy.sh", "-i", "2", "-b", family["student"],
                                      "-t", teacher, "-f", train_file, "-v", val_file, "-n", run_id],
                                     gpu, log_file) == 0
                            model_path = f"ckpts/{dataset}/student_model_iter_2_{run_id}"

                        if not (ok and exists(model_path)):
                            return log(f"GPU {gpu} FAIL {dataset}/{run_id} (see {log_file})")
                        log(f"GPU {gpu} EVAL {dataset}/{run_id}")
                        run([PY, "evaluate.py", "--model_path", model_path, "--base_model", family["student"],
                             "--datasets", dataset, "--output_dir", "results/students"], gpu, log_file, append=True)
                        log(f"GPU {gpu} DONE {dataset}/{run_id}")

                    yield job


def eval_teacher_jobs(args, log_dir):
    for ablation in args.ablations:
        for dataset in args.datasets:
            for name in args.families:
                family = FAMILIES[name]
                run_id = f"{name}{suffix(ablation)}_teacher"

                def job(gpu, dataset=dataset, family=family, ablation=ablation, run_id=run_id):
                    ckpt = teacher_ckpt(dataset, family, ablation)
                    if not exists(ckpt):
                        return log(f"SKIP {dataset}/{run_id}: checkpoint missing ({ckpt})")
                    log(f"GPU {gpu} START {dataset}/{run_id}")
                    rc = run([PY, "evaluate.py", "--model_path", ckpt, "--base_model", family["teacher"],
                              "--datasets", dataset, "--output_dir", "results/teachers"],
                             gpu, os.path.join(log_dir, f"{dataset}_{run_id}.log"))
                    log(f"GPU {gpu} {'DONE' if rc == 0 else 'FAIL'} {dataset}/{run_id}")

                yield job


def eval_base_jobs(args, log_dir):
    for dataset in args.datasets:
        for name in args.families:
            for role in ("teacher", "student"):
                model = FAMILIES[name][role]

                def job(gpu, dataset=dataset, model=model):
                    tag = model.replace("/", "_")
                    log(f"GPU {gpu} START {dataset}/{tag}")
                    rc = run([PY, "evaluate.py", "--model_path", model, "--datasets", dataset,
                              "--output_dir", "results/base"], gpu, os.path.join(log_dir, f"{dataset}_{tag}.log"))
                    log(f"GPU {gpu} {'DONE' if rc == 0 else 'FAIL'} {dataset}/{tag}")

                yield job


COMMANDS = {
    "teacher": teacher_jobs,
    "student": student_jobs,
    "eval-teacher": eval_teacher_jobs,
    "eval-base": eval_base_jobs,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=list(COMMANDS))
    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    p.add_argument("--ablations", nargs="+", default=None,
                   help="Teacher variants. teacher: main/no_length/no_answer; others: any trained variant name")
    p.add_argument("--methods", nargs="+", default=STUDENT_METHODS, choices=STUDENT_METHODS,
                   help="Student distillation methods (student command only)")
    args = p.parse_args()

    if args.ablations is None:
        args.ablations = list(TEACHER_ABLATIONS)
    if args.command == "teacher":
        bad = set(args.ablations) - set(TEACHER_ABLATIONS)
        if bad:
            p.error(f"Unknown teacher ablations {sorted(bad)}; choose from {list(TEACHER_ABLATIONS)}")

    log_dir = os.path.join(ROOT, "logs", args.command)
    os.makedirs(log_dir, exist_ok=True)
    run_pool(list(COMMANDS[args.command](args, log_dir)), args.gpus)


if __name__ == "__main__":
    main()
