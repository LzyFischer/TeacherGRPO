"""Dataset loading, prompt formatting and batching.

Every dataset lives in ``data/<name>/{train,test}.jsonl``. Raw rows are turned
into prompts by the formatter registered for that dataset. Rows that already
carry a pre-formatted ``instruction`` field (e.g. files written by
``generate.py`` or ``rank_candidates.py``) are used as-is.
"""

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer


@dataclass
class Example:
    prompt: str
    response: str


# -----------------------------------------------------------------------------
# Prompt formatters: row -> (prompt, response)
# -----------------------------------------------------------------------------

def _choice_str(labels: List[str]) -> str:
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _mcq_prompt(question: str, labels: List[str], texts: List[str]) -> str:
    choice_str = _choice_str(labels)
    formatted_choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    return (
        f"Given the following question and {len(labels)} candidate answers ({choice_str}), choose the best answer.\n"
        f"Question: {question}\n"
        f"{formatted_choices}\n"
        f"Please reason step by step, and conclude with your choice. Your response should end with "
        f'"The best answer is []" where the [] is one of {choice_str}.'
    )


def format_anli(row) -> Tuple[str, str]:
    prompt = (
        f'Given that "{row.get("premise", "")}"\n'
        f'Question: {row.get("hypothesis", "")} True, False, or Neither?\n\n'
        f"Please reason step by step, and conclude with your final answer."
    )
    return prompt, row.get("response", "")


def format_arc_challenge(row) -> Tuple[str, str]:
    question = row.get("question", "")
    choices = row.get("choices", {})
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        prompt = _mcq_prompt(question, choices["label"], choices["text"])
    else:
        prompt = f"Question: {question}\n{str(choices)}"
    return prompt, row.get("response", "")


def format_commonsense_qa(row) -> Tuple[str, str]:
    question = row.get("question", "")
    choices = row.get("choices", {})
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        formatted_choices = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        prompt = (
            f"Given the following question and five candidate answers (A, B, C, D, and E), choose the best answer.\n"
            f"Question: {question}\n"
            f"{formatted_choices}\n"
            f"Please reason step by step, and conclude with your choice. Your response should end with "
            f'"The best answer is []" where the [] is one of A, B, C, D, or E.'
        )
    else:
        prompt = f"Question: {question}"
    return prompt, row.get("response", "")


def format_date(row) -> Tuple[str, str]:
    question = row.get("question", "")
    choices = row.get("choices", {})
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        prompt = _mcq_prompt(question, choices["label"], choices["text"])
    else:
        prompt = f"Question: {question}"
    return prompt, row.get("response", "")


def format_math(row) -> Tuple[str, str]:
    """Used for both MATH and GSM8K."""
    prompt = (
        f"Question: {row.get('question', '')}\n\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    return prompt, row.get("response", "")


def format_strategy_qa(row) -> Tuple[str, str]:
    prompt = (
        f"Question: True or False: {row.get('question', '')}\n\n"
        f'Please reason step by step, and conclude with either "True" or "False".'
    )
    return prompt, row.get("response", "")


def format_table_mwp(row) -> Tuple[str, str]:
    prompt = (
        f'Read the following table and regarding "{row.get("table_title", "")}" and then answer a question:\n\n'
        f'{row.get("table", "")}\n\n'
        f'Question: {row.get("question", "")}\n\n'
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )
    return prompt, row.get("response", "")


def format_generic(row, prompt_col: str = "instruction", resp_col: str = "response") -> Tuple[str, str]:
    return str(row.get(prompt_col, "")), str(row.get(resp_col, ""))


# name -> formatter and answer type (see answers.py for how each type is scored)
DATASET_REGISTRY: Dict[str, Dict] = {
    "gsm8k":          {"formatter": format_math,           "type": "math"},
    "math":           {"formatter": format_math,           "type": "math"},
    "table_mwp":      {"formatter": format_table_mwp,      "type": "math"},
    "arc_challenge":  {"formatter": format_arc_challenge,  "type": "mcq"},
    "commonsense_qa": {"formatter": format_commonsense_qa, "type": "mcq"},
    "date":           {"formatter": format_date,           "type": "mcq"},
    "strategy_qa":    {"formatter": format_strategy_qa,    "type": "yesno"},
    "anli":           {"formatter": format_anli,           "type": "anli"},
}
DEFAULT_DATASET_CONFIG = {"formatter": format_generic, "type": "text"}


def get_dataset_config(name: Optional[str]) -> Dict:
    """Look up a dataset by exact name, falling back to a substring match."""
    if not name:
        return DEFAULT_DATASET_CONFIG
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]
    for key, conf in DATASET_REGISTRY.items():
        if name in key or key in name:
            return conf
    return DEFAULT_DATASET_CONFIG


def infer_dataset_name(path: str) -> str:
    """``data/date/train.jsonl`` -> ``date``."""
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def read_records(path: str, limit: Optional[int] = None) -> List[Dict]:
    """Read a .jsonl file, or a .json file holding a list / ``{"instances": [...]}``."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            records = []
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        else:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "instances" in data:
                records = data["instances"]
            else:
                records = [data]
    return records[:limit] if limit else records


def load_examples(
    path: str,
    dataset_name: Optional[str] = None,
    prompt_col: str = "instruction",
    resp_col: str = "response",
    limit: Optional[int] = None,
) -> List[Example]:
    """Load (prompt, response) pairs for SFT / distillation.

    Rows containing ``prompt_col`` are treated as already formatted; all other
    rows go through the formatter of ``dataset_name`` (inferred from the parent
    directory when omitted).
    """
    formatter = get_dataset_config(dataset_name or infer_dataset_name(path))["formatter"]
    examples = []
    for row in read_records(path, limit):
        if prompt_col in row:
            p, r = format_generic(row, prompt_col, resp_col)
        else:
            p, r = formatter(row)
        if p and r:
            examples.append(Example(prompt=p, response=r))
    print(f"Loaded {len(examples)} examples from {path}")
    return examples


# -----------------------------------------------------------------------------
# Tokenisation / batching
# -----------------------------------------------------------------------------

def chat_template_pair(
    tokenizer: PreTrainedTokenizer, prompt: str, response: str, max_length: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenise a (user, assistant) turn; labels mask out the prompt with -100."""
    user_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", return_dict=True
    )
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = full["input_ids"][0][:max_length]
    attention_mask = full["attention_mask"][0][:max_length]

    labels = input_ids.clone()
    labels[: min(user_only["input_ids"].size(-1), labels.size(0))] = -100
    return input_ids, attention_mask, labels


def collate_fn_builder(
    tokenizer: PreTrainedTokenizer, max_length: int
) -> Callable[[List[Example]], Dict[str, Union[torch.Tensor, List[str]]]]:
    def collate(batch: List[Example]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        ids, masks, labels = zip(*(chat_template_pair(tokenizer, ex.prompt, ex.response, max_length) for ex in batch))
        pad = torch.nn.utils.rnn.pad_sequence
        return {
            "input_ids": pad(list(ids), batch_first=True, padding_value=tokenizer.pad_token_id),
            "attention_mask": pad(list(masks), batch_first=True, padding_value=0),
            "labels": pad(list(labels), batch_first=True, padding_value=-100),
            "prompts": [ex.prompt for ex in batch],
            "responses": [ex.response for ex in batch],
        }

    return collate
