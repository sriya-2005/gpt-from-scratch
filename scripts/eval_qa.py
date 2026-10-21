"""Ask a checkpoint every question in a split and score its answers.

Answers are generated greedily (temperature 0), so the same checkpoint always gives the
same answers. Scores are automatic word-overlap F1 plus a check of how the model handles
questions the story cannot answer. Every answer is also written to a CSV file with an
empty "human_grade" column, because word overlap is only a rough stand-in for correctness.

Usage:
    python scripts/eval_qa.py --ckpt models/checkpoints/verdict/latest.pt --split val
    python scripts/eval_qa.py --ckpt models/checkpoints/verdict_qa/best.pt --split test
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch

from inference import ROOT, SamplingParams, load_model, load_tokenizer, stream_generate
from qa_format import clean_answer, f1_score, load_split, prompt_text, says_unanswerable

MAX_ANSWER_TOKENS = 60


def answer_question(lm, tok, question: str) -> str:
    params = SamplingParams(temperature=0.0, max_new_tokens=MAX_ANSWER_TOKENS, stop_at_eos=True)
    text = "".join(piece for _, piece in stream_generate(lm, tok, prompt_text(question), params))
    return clean_answer(text)


def evaluate(lm, tok, rows: list[dict], show: bool = True) -> dict:
    """Generate an answer for every row; return summary scores and the per-question results."""
    results = []
    for i, row in enumerate(rows, start=1):
        answer = answer_question(lm, tok, row["question"])
        unanswerable = row["type"] == "unanswerable"
        f1 = f1_score(answer, row["answer"])
        results.append({**row, "model_answer": answer, "auto_f1": round(f1, 3),
                        "said_does_not_say": says_unanswerable(answer)})
        if show:
            print(f"[{i:>2}/{len(rows)}] F1 {f1:.2f}  Q: {row['question']}\n"
                  f"          model: {answer!r}\n          ref:   {row['answer']!r}")

    answerable = [r for r in results if r["type"] != "unanswerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]
    by_type = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r["auto_f1"])
    return {
        "results": results,
        "mean_f1": sum(r["auto_f1"] for r in answerable) / max(1, len(answerable)),
        "f1_at_least_half": sum(r["auto_f1"] >= 0.5 for r in answerable),
        "n_answerable": len(answerable),
        "wrongly_said_does_not_say": sum(r["said_does_not_say"] for r in answerable),
        "unanswerable_correct": sum(r["said_does_not_say"] for r in unanswerable),
        "n_unanswerable": len(unanswerable),
        "by_type": {t: sum(v) / len(v) for t, v in sorted(by_type.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "probe"], default="val",
                        help="probe = reworded questions about TRAINING facts (data/qa/probe.jsonl)")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
    lm = load_model(path, torch.device("cpu"))
    tok = load_tokenizer()
    rows = load_split(args.split)
    print(f"Asking {path.relative_to(ROOT)} the {len(rows)} {args.split} questions...\n")
    s = evaluate(lm, tok, rows, show=not args.quiet)

    print(f"\n=== {path.relative_to(ROOT)} on {args.split} ===")
    print(f"Answerable questions ({s['n_answerable']}):")
    print(f"  mean word-overlap F1            : {s['mean_f1']:.2f}")
    print(f"  answers with F1 >= 0.5          : {s['f1_at_least_half']} of {s['n_answerable']}")
    print(f"  wrongly said 'does not say'     : {s['wrongly_said_does_not_say']}")
    print(f"  F1 by type: " + ", ".join(f"{t} {v:.2f}" for t, v in s["by_type"].items()))
    print(f"Unanswerable questions: said 'does not say' for {s['unanswerable_correct']} "
          f"of {s['n_unanswerable']}")

    out = ROOT / "logs" / f"qa_eval_{path.parent.name}_{path.stem}_{args.split}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel shows quotes correctly
        w = csv.writer(f)
        w.writerow(["fact", "type", "question", "reference_answer", "model_answer", "auto_f1", "human_grade",
                    "notes"])
        for r in s["results"]:
            w.writerow([r["fact"], r["type"], r["question"], r["answer"], r["model_answer"], r["auto_f1"], "", ""])
    print(f"\nAll answers saved to {out.relative_to(ROOT)} (fill in human_grade: correct / partly / wrong)")


if __name__ == "__main__":
    main()
