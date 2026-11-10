"""How often does the search find the passage that holds the answer?

Two tests:
  1. Single questions: every answerable question in the QA splits, asked on its own.
     A hit means one of the top-k passages contains one of the question's evidence quotes.
  2. Conversations (data/qa/followups.jsonl): follow-ups like "How did he die?" after
     "Who was Stroud?". Each turn is searched twice, with the conversation memory and
     without it, to show what the memory adds.

Usage:
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --show-misses
"""
import argparse
import json

from inference import load_tokenizer
from qa_format import QA_DIR, load_split
from retrieval import Conversation, build_index, gold_passages, is_followup

KS = (1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-misses", action="store_true", help="print the questions whose passage was not found")
    args = parser.parse_args()

    index = build_index(load_tokenizer())
    passages = index.passages
    print(f"{len(passages)} passages\n")

    rows = [r for s in ("train", "val", "test", "probe") for r in load_split(s)]
    by_question = {r["question"]: r for r in rows}

    # ---- 1. single questions ----
    print("Single questions (top-1 / top-3 contains the evidence):")
    no_gold = 0
    for split in ("train", "val", "test", "probe"):
        hits, n = zero_counts(), 0
        for r in load_split(split):
            if r["type"] == "unanswerable":
                continue
            gold = gold_passages(passages, r.get("evidence", []))
            if not gold:
                no_gold += 1
                continue
            ranked = [i for i, _ in Conversation(index).search(r["question"], k=max(KS))]
            n += 1
            for k in KS:
                hits[k] += bool(gold & set(ranked[:k]))
            if args.show_misses and not gold & set(ranked[:1]):
                print(f"    miss: {r['question']}  (gold {sorted(gold)}, got {ranked})")
        print(f"  {split:<6} {n:>3} questions   top-1 {hits[1] / n:.0%}   top-3 {hits[3] / n:.0%}")
    if no_gold:
        print(f"  ({no_gold} questions skipped: their evidence quotes were not found in any passage)")

    # ---- 2. conversations ----
    convs = [json.loads(l) for l in (QA_DIR / "followups.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\nConversations ({len(convs)}), turns after the first:")
    with_mem, without_mem, n = zero_counts(), zero_counts(), 0
    for c in convs:
        conv = Conversation(index)
        for t, turn in enumerate(c["turns"]):
            gold = gold_passages(passages, by_question[turn["evidence_from"]]["evidence"])
            ranked = [i for i, _ in conv.search(turn["question"], k=max(KS))]
            alone = [i for i, _ in Conversation(index).search(turn["question"], k=max(KS))]
            if t > 0:
                n += 1
                for k in KS:
                    with_mem[k] += bool(gold & set(ranked[:k]))
                    without_mem[k] += bool(gold & set(alone[:k]))
                if args.show_misses and not gold & set(ranked[:1]):
                    print(f"    miss [{c['id']}] {turn['question']}  (gold {sorted(gold)}, got {ranked}, "
                          f"{'follow-up' if is_followup(turn['question']) else 'new topic'})")
            # answers are stored but not searched (see Conversation), so the reference one is fine here
            conv.add(turn["question"], by_question[turn["evidence_from"]]["answer"], ranked)
    print(f"  with memory     top-1 {with_mem[1] / n:.0%}   top-3 {with_mem[3] / n:.0%}")
    print(f"  without memory  top-1 {without_mem[1] / n:.0%}   top-3 {without_mem[3] / n:.0%}   ({n} turns)")


def zero_counts() -> dict[int, int]:
    return {k: 0 for k in KS}


if __name__ == "__main__":
    main()
