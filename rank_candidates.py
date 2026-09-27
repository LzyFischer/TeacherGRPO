"""Rank sampled teacher responses by teacher->student KL and split them into buckets.

For every prompt in ``--candidate_file`` (written by ``generate.py`` with
``--num_generations N``), each candidate is scored by the mean forward
KL(teacher || student) over the full chat sequence. Candidates are sorted from
lowest to highest KL and the i-th ranked response of every prompt goes to
``<output_dir>/rank_i.jsonl``. ``rank_0`` is therefore the set of responses the
student finds easiest to imitate. Teacher and student must share a tokenizer.

    python rank_candidates.py --teacher_model unsloth/Qwen2.5-3B-Instruct \\
        --student_model unsloth/Qwen2.5-0.5B-Instruct \\
        --candidate_file candidates/date.jsonl --output_dir kl_partition_results/date
"""

import argparse
import json
import os
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from teachergrpo.losses import forward_kl_div_loss
from teachergrpo.utils import cleanup_merged, merge_lora_if_needed


@torch.no_grad()
def compute_kl(teacher_model, student_model, tokenizer, prompt: str, response: str) -> float:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        return_tensors="pt",
        return_dict=True,
    ).to(teacher_model.device)
    t_logits = teacher_model(**encoded).logits[:, :-1, :]
    s_logits = student_model(**encoded).logits[:, :-1, :].to(t_logits.device)
    return forward_kl_div_loss(s_logits, t_logits, mask=None, temperature=1.0, reduction="mean").item()


def load_model(path: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher_model", type=str, required=True, help="HF model id, full checkpoint or LoRA adapter")
    p.add_argument("--student_model", type=str, required=True, help="HF model id, full checkpoint or LoRA adapter")
    p.add_argument("--teacher_base", type=str, default=None, help="Base model if --teacher_model is a LoRA adapter")
    p.add_argument("--student_base", type=str, default=None, help="Base model if --student_model is a LoRA adapter")
    p.add_argument("--candidate_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--prompt_column", type=str, default="instruction")
    p.add_argument("--responses_field", type=str, default="responses")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    teacher_path = merge_lora_if_needed(args.teacher_model, args.teacher_base)
    student_path = merge_lora_if_needed(args.student_model, args.student_base)

    try:
        tokenizer = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        teacher_model = load_model(teacher_path)
        student_model = load_model(student_path)

        buckets: List[List[dict]] = []
        with open(args.candidate_file, "r", encoding="utf-8") as fin:
            for line_idx, line in enumerate(fin):
                record = json.loads(line)
                prompt, responses = record[args.prompt_column], record[args.responses_field]
                if not buckets:
                    buckets = [[] for _ in responses]
                    print(f"{len(responses)} candidates per prompt.")
                if len(responses) != len(buckets):
                    raise ValueError(f"Record {line_idx} has {len(responses)} responses, expected {len(buckets)}.")

                scored = []
                for response in responses:
                    try:
                        kl = compute_kl(teacher_model, student_model, tokenizer, prompt, response)
                    except Exception as e:
                        print(f"Warning: KL failed for prompt {line_idx} ({e}); assigning inf.")
                        kl = float("inf")
                    scored.append((kl, response))
                scored.sort(key=lambda x: x[0])

                for rank, (kl, response) in enumerate(scored):
                    buckets[rank].append({
                        args.prompt_column: prompt,
                        "response": response,
                        "kl_value": kl,
                        "source": record.get("source", "teacher_generated"),
                    })
                if (line_idx + 1) % 50 == 0:
                    print(f"Processed {line_idx + 1} prompts...")
    finally:
        cleanup_merged(teacher_path, args.teacher_model)
        cleanup_merged(student_path, args.student_model)

    for i, bucket in enumerate(buckets):
        out_path = os.path.join(args.output_dir, f"rank_{i}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for item in bucket:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(bucket)} samples to {out_path}")


if __name__ == "__main__":
    main()
