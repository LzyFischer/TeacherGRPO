"""Evaluate a model (base, full fine-tune or LoRA adapter) with vLLM.

Reads ``<data_root>/<dataset>/test.jsonl`` for each dataset and writes
``<output_dir>/<dataset>/<model_slug>/eval_results.json`` with accuracy and
per-example predictions.

    python evaluate.py --model_path ckpts/date/student_model_run --base_model unsloth/Qwen2.5-0.5B-Instruct \\
        --datasets date strategy_qa
"""

import argparse
import glob
import json
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

os.environ.setdefault("VLLM_USE_V1", "1")

from transformers import AutoTokenizer, PreTrainedTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from teachergrpo.answers import extract_prediction, extract_reference, is_correct  # noqa: E402
from teachergrpo.data import get_dataset_config, read_records  # noqa: E402
from teachergrpo.utils import cleanup_merged, merge_lora_if_needed  # noqa: E402


@dataclass
class EvalExample:
    prompt: str
    reference: str
    metadata: Dict[str, Any]


def find_eval_file(data_dir: str) -> Optional[str]:
    for fname in ["test.jsonl", "test.json", "validation.jsonl", "validation.json", "val.jsonl"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return path
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    return jsonl_files[0] if jsonl_files else None


def load_benchmark(dataset_name: str, data_root: str = "data", limit: Optional[int] = None):
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        matches = [d for d in os.listdir(data_root) if dataset_name in d]
        if not matches:
            raise FileNotFoundError(f"Directory not found: {dataset_dir}")
        dataset_dir = os.path.join(data_root, matches[0])
        print(f"Redirecting {dataset_name} -> {matches[0]}")

    data_file = find_eval_file(dataset_dir)
    if not data_file:
        raise FileNotFoundError(f"No evaluation file found in {dataset_dir}")

    conf = get_dataset_config(dataset_name)
    formatter, eval_type = conf["formatter"], conf["type"]
    examples = []
    for row in read_records(data_file, limit):
        try:
            prompt, _ = formatter(row)
            examples.append(EvalExample(prompt=prompt, reference=extract_reference(row, eval_type), metadata=row))
        except Exception:
            continue
    return examples, eval_type


def evaluate_model(
    llm: LLM,
    examples: List[EvalExample],
    eval_type: str,
    sampling_params: SamplingParams,
    tokenizer: PreTrainedTokenizer,
):
    if getattr(tokenizer, "chat_template", None):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ex.prompt}], tokenize=False, add_generation_prompt=True
            )
            for ex in examples
        ]
    else:
        prompts = [ex.prompt for ex in examples]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated = output.outputs[0].text
        prediction = extract_prediction(generated, eval_type)
        results.append({
            "prompt": ex.prompt,
            "generated": generated,
            "prediction": prediction,
            "reference": ex.reference,
            "correct": is_correct(prediction, ex.reference, eval_type),
        })
    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    return accuracy, results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", type=str, required=True, help="HF model id, full checkpoint or LoRA adapter")
    p.add_argument("--base_model", type=str, default=None,
                   help="Base model for LoRA adapters (defaults to the one in adapter_config.json); also used for the tokenizer")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples")
    p.add_argument("--tp_size", type=int, default=1, help="Tensor-parallel size")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()

    model_path = merge_lora_if_needed(args.model_path, args.base_model)
    model_slug = os.path.basename(args.model_path.rstrip("/"))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model or model_path, trust_remote_code=True)

    try:
        print(f"Loading vLLM from {model_path}...")
        llm = LLM(model=model_path, tensor_parallel_size=args.tp_size, trust_remote_code=True, gpu_memory_utilization=0.85)
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        for dataset_name in args.datasets:
            print(f"\n{'=' * 40}\nProcessing: {dataset_name}\n{'=' * 40}")
            try:
                examples, eval_type = load_benchmark(dataset_name, args.data_root, args.limit)
                if not examples:
                    print(f"Skipping {dataset_name} (empty).")
                    continue
                accuracy, results = evaluate_model(llm, examples, eval_type, sampling_params, tokenizer)
                print(f"Accuracy for {dataset_name}: {accuracy:.2%}")

                out_dir = os.path.join(args.output_dir, dataset_name, model_slug)
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, "eval_results.json")
                with open(out_file, "w") as f:
                    json.dump(
                        {"dataset": dataset_name, "model": args.model_path, "accuracy": accuracy, "details": results},
                        f,
                        indent=2,
                    )
                print(f"Saved results to: {out_file}")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                traceback.print_exc()
    finally:
        cleanup_merged(model_path, args.model_path)


if __name__ == "__main__":
    main()
