"""Small helpers shared by the entry-point scripts."""

import argparse
import gc
import json
import os
import shutil
import tempfile
from typing import Optional

import yaml


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args, taking defaults from ``--config <yaml>`` when given.

    Precedence: command line > YAML config > argparse defaults. Unknown YAML
    keys raise an error so typos do not go unnoticed.
    """
    parser.add_argument("--config", type=str, default=None, help="YAML file with argument defaults")
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(config) - known)
        if unknown:
            parser.error(f"Unknown keys in {pre_args.config}: {unknown}")
        # argparse applies `type` to string defaults, so YAML strings like "2e-5" still work.
        parser.set_defaults(**config)
        for action in parser._actions:
            if action.dest in config:
                action.required = False
    return parser.parse_args()


def is_lora_adapter(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def merge_lora_if_needed(model_path: str, base_model: Optional[str] = None) -> str:
    """Return a path vLLM / HF can load as a full model.

    If ``model_path`` is a LoRA adapter, merge it into ``base_model`` (or the
    base recorded in ``adapter_config.json``) and return a temporary directory,
    which the caller should delete with ``cleanup_merged``. Otherwise return
    ``model_path`` unchanged.
    """
    if not is_lora_adapter(model_path):
        return model_path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if base_model is None:
        with open(os.path.join(model_path, "adapter_config.json")) as f:
            base_model = json.load(f)["base_model_name_or_path"]

    print(f"[Merge] Merging adapter '{model_path}' into base '{base_model}'...")
    temp_dir = tempfile.mkdtemp(prefix="merged_model_")
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload()
        model.save_pretrained(temp_dir)
        AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(temp_dir)
        del model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def cleanup_merged(merged_path: str, original_path: str) -> None:
    if merged_path != original_path and os.path.basename(merged_path).startswith("merged_model_"):
        shutil.rmtree(merged_path, ignore_errors=True)
