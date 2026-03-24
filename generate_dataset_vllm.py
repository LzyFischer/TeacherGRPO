import argparse
import json
import os
import shutil
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import gc
import tempfile

# Reuse logic from load_data
from load_data import load_data_source

def apply_chat_template_if_available(tokenizer, prompts, add_generation_prompt=True):
    """If tokenizer has chat_template, wrap prompts into chat format for inference."""
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        wrapped = []
        for p in prompts:
            wrapped.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            )
        return wrapped
    return prompts

def merge_if_needed(model_path: str, base_model: str) -> str:
    """Merge LoRA adapter into base model if needed, so vLLM can load a full model."""
    # Simple check: if adapter_config.json exists, it's a LoRA adapter
    if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
        return model_path

    print(f"\n[Merge] Merging adapter '{model_path}' into base '{base_model}'...")
    temp_dir = tempfile.mkdtemp(prefix="merged_student_")
    
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        
        print(f"Saving merged model to {temp_dir}...")
        model.save_pretrained(temp_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(temp_dir)
        
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
        return temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Original base model (e.g. Qwen2.5-0.5B)")
    parser.add_argument("--model_path", type=str, required=True, help="Current student checkpoint (LoRA or full)")
    parser.add_argument("--input_file", type=str, required=True, help="Original train.jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the generated jsonl")
    parser.add_argument("--prompt_column", type=str, default="instruction")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)  # On-policy generation typically uses non-zero temperature
    args = parser.parse_args()

    # 1. Prepare data
    print(f"Loading prompts from {args.input_file}...")
    # Load prompts only; responses will be generated
    examples = load_data_source(args.input_file, split="train", prompt_col=args.prompt_column, resp_col="response")
    prompts = [ex.prompt for ex in examples]
    print(f"Loaded {len(prompts)} prompts.")

    # 2. Prepare model (merge LoRA if needed)
    model_path_for_vllm = merge_if_needed(args.model_path, args.base_model)

    # 3. vLLM inference
    print(f"Initializing vLLM with {model_path_for_vllm}...")

    tokenizer_path = args.base_model or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Apply chat template if available
    prompts_for_vllm = apply_chat_template_if_available(tokenizer, prompts)

    llm = LLM(
        model=model_path_for_vllm,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.80
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )

    print("Generating responses...")
    outputs = llm.generate(prompts_for_vllm, sampling_params)

    # 4. Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for ex, output in zip(examples, outputs):
            generated_text = output.outputs[0].text
            
            # Build record in the format expected by train_student.py
            json_line = {
                args.prompt_column: ex.prompt,
                "response": generated_text, 
                "source": "student_generated"
            }
            f.write(json.dumps(json_line) + "\n")

    print(f"Saved {len(outputs)} generated examples to {args.output_file}")

    # Clean up temporary directory
    if model_path_for_vllm != args.model_path and "merged_student_" in model_path_for_vllm:
        print(f"Cleaning up temp dir {model_path_for_vllm}...")
        shutil.rmtree(model_path_for_vllm)

if __name__ == "__main__":
    main()