"""Answer extraction and correctness checking.

Shared by evaluation (``evaluate.py``) and the verifiable reward used during
teacher training, so both score answers identically. Answer types:

* ``math``  - last ``\\boxed{...}`` (or last number), compared after normalisation
* ``mcq``   - option letter A-F
* ``yesno`` - True / False
* ``anli``  - True / False / Neither
"""

import re
from fractions import Fraction
from typing import Dict, Optional

def normalize_math_answer(text: str) -> str:
    """
    Normalize mathematical answers to handle various formatting differences.
    
    Handles:
    - LaTeX formatting (\\frac, \\text, \\sqrt, etc.)
    - Text units (students, cm, km, degrees, etc.)
    - Spacing differences
    - Fraction representations (1/2 vs \\frac{1}{2})
    - Interval notation
    - Degree symbols
    - List formatting
    """
    if not text or text == "N/A":
        return text
    
    text = str(text).strip()
    
    # Remove common LaTeX commands and their braces
    latex_patterns = [
        (r'\\text\{[^}]*\}', ''),  # Remove text units like \text{ cm}
        (r'\\,', ''),               # Thin space
        (r'\\;', ''),               # Medium space
        (r'\\:', ''),               # Medium space
        (r'\\!', ''),               # Negative thin space
        (r'\\ ', ' '),              # Escaped space
        (r'\\mathrm\{([^}]*)\}', r'\1'),  # \mathrm{text} -> text
        (r'\\operatorname\{([^}]*)\}', r'\1'),  # \operatorname{text} -> text
    ]
    
    for pattern, replacement in latex_patterns:
        text = re.sub(pattern, replacement, text)
    
    # Remove text units (students, cm, km, degrees, etc.)
    text = re.sub(r'\s*\\text\s*\{[^}]*\}', '', text)
    text = re.sub(r'\s+(students?|cm|km|meters?|m|ft|inches?|in|degrees?|°|hours?|minutes?|seconds?|years?|days?)\b', '', text, flags=re.IGNORECASE)
    
    # Normalize degree symbols
    text = re.sub(r'\\circ|°|degrees?', '', text, flags=re.IGNORECASE)
    
    # Convert \frac{a}{b} to a/b
    def replace_frac(match):
        num = match.group(1)
        denom = match.group(2)
        return f"{num}/{denom}"
    
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', replace_frac, text)
    text = re.sub(r'\\dfrac\{([^}]*)\}\{([^}]*)\}', replace_frac, text)
    text = re.sub(r'\\tfrac\{([^}]*)\}\{([^}]*)\}', replace_frac, text)
    
    # Remove remaining backslashes (for incomplete LaTeX)
    text = text.replace('\\', '')
    
    # Normalize spacing around operators and brackets
    text = re.sub(r'\s*([,\[\]\(\)])\s*', r'\1', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Try to evaluate fractions to decimals for comparison
    # This handles cases like "1/4" vs "0.25"
    try:
        # Check if it's a simple fraction
        if '/' in text and not any(c in text for c in ['[', ']', '(', ')', ',', 'sqrt', 'x', 'in']):
            parts = text.split('/')
            if len(parts) == 2:
                frac = Fraction(text)
                # Return both fraction and decimal forms for matching
                decimal = float(frac)
                # Store as tuple internally for flexible matching
                return f"{text}|{decimal}"
    except Exception:
        pass
    
    # For interval notation, normalize spacing: x in [-2, 7] -> [-2,7]
    if 'in' in text.lower():
        # Extract just the interval part
        interval_match = re.search(r'[\[\(][-\d.,\s]+[\]\)]', text)
        if interval_match:
            interval = interval_match.group(0)
            interval = re.sub(r'\s*,\s*', ',', interval)
            return interval
    
    return text

def math_answers_equal(pred: str, ref: str) -> bool:
    """
    Compare two mathematical answers with robust normalization.
    """
    if not pred or not ref or pred == "N/A":
        return False
    
    # Normalize both
    norm_pred = normalize_math_answer(pred)
    norm_ref = normalize_math_answer(ref)
    
    # Direct string match after normalization
    if norm_pred == norm_ref:
        return True
    
    # Handle fraction/decimal equivalence
    if '|' in norm_pred or '|' in norm_ref:
        pred_forms = norm_pred.split('|')
        ref_forms = norm_ref.split('|')
        
        for pf in pred_forms:
            for rf in ref_forms:
                try:
                    if abs(float(pf) - float(rf)) < 1e-6:
                        return True
                except Exception:
                    if pf == rf:
                        return True
    
    # Try numeric comparison
    try:
        # Extract all numbers and compare
        pred_nums = re.findall(r'-?\d+\.?\d*', norm_pred)
        ref_nums = re.findall(r'-?\d+\.?\d*', norm_ref)
        
        if pred_nums and ref_nums and len(pred_nums) == len(ref_nums):
            all_match = True
            for p, r in zip(pred_nums, ref_nums):
                try:
                    if abs(float(p) - float(r)) > 1e-6:
                        all_match = False
                        break
                except Exception:
                    if p != r:
                        all_match = False
                        break
            if all_match:
                return True
    except Exception:
        pass
    
    # Fallback: strip all non-alphanumeric and compare
    pred_clean = re.sub(r'[^a-zA-Z0-9.]', '', norm_pred)
    ref_clean = re.sub(r'[^a-zA-Z0-9.]', '', norm_ref)
    
    return pred_clean == ref_clean


def extract_last_boxed(text: str) -> Optional[str]:
    """Extracts the content of the LAST \\boxed{...}."""
    box_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    for start_idx in reversed(box_starts):
        open_braces = 1
        content_start = start_idx + 7
        for i in range(content_start, len(text)):
            char = text[i]
            if char == '{': open_braces += 1
            elif char == '}': open_braces -= 1
            if open_braces == 0: return text[content_start:i]
    return None

def extract_mcq_prediction(text: str) -> str:
    """
    Extract MCQ answer (A-F).
    Handles:
    - "The best/correct/final answer is (A)"
    - "The answer is **A**"
    - "The option is C."
    - "Answer: A"
    - Fallback to last bracketed/bolded letter
    """
    # 1. Broad "Answer is" pattern (covers "best answer", "correct answer", "option is")
    # We use findall and take the LAST match to handle chain-of-thought reasoning 
    # where the model might discuss why "A is wrong" before concluding "B is correct".
    # Regex breakdown:
    #   (?:answer|option|choice)  -> keyword
    #   (?: is)?                  -> optional verb (e.g., "The answer: A")
    #   (?:[:\s\*\(\[\{]*)        -> junk separators (spaces, colons, *, [, (, {)
    #   ([A-F])                   -> The target letter
    #   \b                        -> Word boundary (prevents matching "A" in "Apple")
    matches = re.findall(r"(?:answer|option|choice)(?: is)?\s*(?:[:\s\*\(\[\{]*)\s*([A-F])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # 2. Explicit "Answer:" header (common in some finetunes)
    match = re.search(r"Answer:\s*(?:[:\s\*\(\[\{]*)\s*([A-F])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
        
    # 3. Fallback: Look for the last occurrence of specific patterns
    # Handles: (A), [A], {A}, **A**, **A.**
    matches = re.findall(r"(?:[\(\[\{]|\*\*)\s*([A-F])\s*(?:[\)\]\}]|\*\*|\.)", text)
    if matches:
        return matches[-1].upper()
        
    return "N/A"

def extract_yesno_prediction(text: str) -> str:
    """Extract Yes/No answer."""
    snippet = text[-50:].lower() if len(text) > 50 else text.lower()
    
    if "true" in snippet and "false" not in snippet: return "True"
    if "false" in snippet and "true" not in snippet: return "False"
    
    matches = re.findall(r"\b(true|false)\b", snippet)
    if matches:
        return matches[-1].capitalize()
        
    return "N/A"

def extract_anli_prediction(text: str) -> str:
    """Extract ANLI prediction (True/False/Neither)."""
    snippet = text[-100:].lower()
    
    if "true" in snippet or "entailment" in snippet or "yes" in snippet: return "True"
    if "false" in snippet or "contradiction" in snippet or "no" in snippet: return "False"
    if "neither" in snippet or "neutral" in snippet: return "Neither"
    
    return "N/A"

def extract_prediction(text: str, eval_type: str) -> str:
    """Extract prediction based on evaluation type."""
    if eval_type == "math":
        res = extract_last_boxed(text)
        if res: return res.strip()
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        return nums[-1] if nums else "N/A"
        
    elif eval_type == "mcq":
        return extract_mcq_prediction(text)
        
    elif eval_type == "yesno":
        return extract_yesno_prediction(text)
        
    elif eval_type == "anli":
        return extract_anli_prediction(text)
        
    return text.strip()

def normalize_anli_ref(ref: str) -> str:
    """Normalize ANLI ground truth."""
    r = str(ref).lower().strip()
    if r in ["entailment", "0"]: return "True"
    if r in ["neutral", "1"]: return "Neither"
    if r in ["contradiction", "2"]: return "False"
    return ref.capitalize()

def extract_reference(row: Dict, eval_type: str) -> str:
    """
    Extract ground truth from dataset row.
    Handles distinct key names for different dataset types (e.g., 'label' for ANLI).
    """
    if eval_type == "mcq":
        ref = row.get("answerKey", "")
    elif eval_type == "anli":
        # ANLI uses 'label' and needs normalization (entailment -> True)
        raw_ref = row.get("label", row.get("answer", ""))
        ref = normalize_anli_ref(raw_ref)
    else:
        # Default fallback for Math/others usually found in 'answer'
        ref = str(row.get("answer", ""))
        
    return ref.strip()


def is_correct(prediction: str, reference: str, eval_type: str) -> bool:
    if eval_type == "math":
        return math_answers_equal(prediction, reference)
    return str(prediction).lower().strip() == str(reference).lower().strip()
