"""Sample responses from a model with vLLM.

Used to build training data for the student:

* SeqKD / KL-ranked curricula - sample the (trained) teacher
* on-policy distillation      - sample the current student

Each output line is ``{"instruction": <formatted prompt>, "response": <first sample>,
"responses": [<all samples>], "source": ...}``, so the file can be fed directly
to ``train_student.py`` (uses ``response``) or ``rank_candidates.py`` (uses
``responses``).

    python generate.py --model_path ckpts/date/teacher/final_lora --base_model unsloth/Qwen2.5-3B-Instruct \\
        --input_file data/date/train.jsonl --output_file candidates/date.jsonl --num_generations 5
"""

import argparse
import json
import os
from typing import List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from teachergrpo.data import get_dataset_config, infer_dataset_name, read_records
from teachergrpo.utils import cleanup_merged, merge_lora_if_needed


def load_prompts(path: str, dataset_name: Optional[str], prompt_col: str, limit: Optional[int]) -> List[str]:
    formatter = get_dataset_config(dataset_name or infer_dataset_name(path))["formatter"]
    prompts = []
    for row in read_records(path, limit):
        prompt = row[prompt_col] if prompt_col in row else formatter(row)[0]
        if prompt:
            prompts.append(prompt)
    return prompts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, required=True, help="HF model id, full checkpoint or LoRA adapter")
    p.add_argument("--base_model", type=str, default=None, help="Base model for LoRA adapters; also used for the tokenizer")
    p.add_argument("--input_file", type=str, required=True, help="Dataset file providing the prompts")
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default=None, help="Defaults to the input file's parent directory")
    p.add_argument("--prompt_column", type=str, default="instruction")
    p.add_argument("--num_generations", type=int, default=1, help="Samples per prompt")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None, help="Only use the first N prompts")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--source", type=str, default="generated", help="Value written to the 'source' field")
    args = p.parse_args()

    prompts = load_prompts(args.input_file, args.dataset_name, args.prompt_column, args.max_samples)
    print(f"Loaded {len(prompts)} prompts from {args.input_file}")

    model_path = merge_lora_if_needed(args.model_path, args.base_model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model or model_path, trust_remote_code=True)
        if getattr(tokenizer, "chat_template", None):
            inputs = [
                tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                for q in prompts
            ]
        else:
            inputs = prompts

        llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization)
        sampling_params = SamplingParams(
            n=args.num_generations, temperature=args.temperature, max_tokens=args.max_new_tokens
        )
        outputs = llm.generate(inputs, sampling_params)
    finally:
        cleanup_merged(model_path, args.model_path)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for prompt, output in zip(prompts, outputs):
            responses = [o.text.strip() for o in output.outputs]
            record = {args.prompt_column: prompt, "response": responses[0], "responses": responses, "source": args.source}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(prompts)} x {args.num_generations} samples to {args.output_file}")


if __name__ == "__main__":
    main()
