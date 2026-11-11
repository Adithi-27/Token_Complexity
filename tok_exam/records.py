"""Output rows and JSONL I/O (planning.md, step 6)."""
import json
from pathlib import Path

from .grader import grade

FIELDS = (
    "id", "dataset", "tier", "split", "sample_idx", "prompt", "full_output",
    "thinking_text", "thinking_tokens", "answer_tokens", "pred", "gold", "correct",
)
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_thinking(full_output):
    """Return (thinking_text, answer_text).

    The chat template already ends the prompt with <think>, so the output
    normally starts inside the thinking block. If </think> never appears the
    model ran out of budget: everything is thinking and there is no answer.
    """
    text = full_output
    if text.lstrip().startswith(THINK_OPEN):
        text = text.lstrip()[len(THINK_OPEN):]
    if THINK_CLOSE not in text:
        return text, None
    thinking, answer = text.split(THINK_CLOSE, 1)
    return thinking, answer


def make_row(item, sample_idx, prompt, full_output, tokenizer):
    """Build one output row. `item` is a row of data/processed/pool.jsonl."""
    thinking, answer = split_thinking(full_output)
    # Only a box after </think> counts as the model's answer.
    pred, correct = grade(answer, item["gold"]) if answer is not None else (None, False)
    n_tokens = lambda t: len(tokenizer.encode(t, add_special_tokens=False)) if t else 0
    row = {
        "id": item["id"],
        "dataset": item["dataset"],
        "tier": item["tier"],
        "split": item["split"],
        "sample_idx": sample_idx,
        "prompt": prompt,
        "full_output": full_output,
        "thinking_text": thinking,
        "thinking_tokens": n_tokens(thinking),
        "answer_tokens": n_tokens(answer),
        "pred": pred,
        "gold": item["gold"],
        "correct": correct,
    }
    validate_row(row)
    return row


def validate_row(row):
    if tuple(row) != FIELDS:
        raise ValueError(f"row fields {list(row)} != {list(FIELDS)}")
    if not isinstance(row["gold"], str) or not isinstance(row["correct"], bool):
        raise ValueError("gold must be str and correct must be bool")
    if row["pred"] is not None and not isinstance(row["pred"], str):
        raise ValueError("pred must be str or None")


def hit_cap(row):
    """True if the trace ran out of budget before closing its thinking."""
    return THINK_CLOSE not in row["full_output"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
