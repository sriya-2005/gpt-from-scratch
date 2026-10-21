"""Shared question-answering helpers: the prompt format, tokenization, and answer scoring.

Training (finetune_qa.py), evaluation (eval_qa.py) and the GUI all import these, so the
model always sees questions in exactly the same format it was trained on.
"""
import json
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_DIR = ROOT / "data" / "qa"
UNANSWERABLE = "The story does not say."
IGNORE = -100  # label value that F.cross_entropy skips: "don't grade the model on this token"


def prompt_text(question: str) -> str:
    """What the model is given. It must continue with the answer."""
    return f"Question: {question.strip()}\nAnswer:"


def encode_example(tok, question: str, answer: str) -> tuple[list[int], list[int]]:
    """Return (input ids, labels) for one training example.

    The labels copy the answer tokens and mark every question token as IGNORE, so the loss
    only measures how well the model produces the answer. The answer ends with
    <|endoftext|> so the model learns where to stop.
    """
    prompt_ids = tok.encode(prompt_text(question))
    answer_ids = tok.encode(" " + answer.strip()) + [tok.eos_token_id]
    ids = prompt_ids + answer_ids
    labels = [IGNORE] * len(prompt_ids) + answer_ids
    # next-token prediction: the input at position t is used to predict the label at t+1
    return ids[:-1], labels[1:]


def load_split(split: str) -> list[dict]:
    path = QA_DIR / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_answer(generated: str) -> str:
    """Keep only the first line the model wrote (answers are one line)."""
    return generated.strip().split("\n")[0].strip()


def _normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation and a/an/the: the usual normalization for answer scoring."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation) | {"“", "”", "’"})
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def f1_score(prediction: str, reference: str) -> float:
    """Word-overlap F1 between two answers (0 = nothing in common, 1 = same words).

    A rough automatic measure: it rewards shared words, not correct meaning, so a fluent
    wrong answer that reuses the right names can still score above 0.
    """
    pred, ref = _normalize(prediction), _normalize(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def says_unanswerable(answer: str) -> bool:
    return "does not say" in answer.lower()
