"""Ask questions about "The Verdict" in a conversation; follow-ups are searched in context.

Each question goes through two steps:
  1. Search (retrieval.py): find the story passages that best match the question. Earlier
     questions are part of the search, so "How did he die?" after "Who was Stroud?" still
     finds the passage about Stroud.
  2. Answer: the model reads the best passage and answers from it. If it says the passage
     does not contain the answer, it tries the next one (up to 3).

The model must be fine-tuned with context first (finetune_qa.py --with-context). With
--search-only no model is loaded; the answer quotes the best-matching sentences of the passage.

Usage:
    python scripts/chat.py --ckpt models/checkpoints/verdict_rag/best.pt
    python scripts/chat.py --search-only
Commands while chatting:  new = start a new conversation,  empty line = quit
"""
import argparse
import textwrap
from pathlib import Path

from eval_qa import PASSAGES_TO_TRY, answer_from_passages
from inference import ROOT, load_model, load_tokenizer
from retrieval import Conversation, build_index, is_followup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", help="a checkpoint fine-tuned with --with-context")
    parser.add_argument("--search-only", action="store_true", help="no model: show the best passage")
    parser.add_argument("--show-passage", action="store_true", help="also print the passage each answer came from")
    args = parser.parse_args()
    if not args.ckpt and not args.search_only:
        parser.error("give --ckpt, or --search-only to try the search without a model")

    tok = load_tokenizer()
    index = build_index(tok)
    lm = None
    if not args.search_only:
        path = Path(args.ckpt) if Path(args.ckpt).is_absolute() else ROOT / args.ckpt
        lm = load_model(path)
        print(f"Loaded {path.relative_to(ROOT)}: {lm.describe()}")
        if not lm.train_args.get("with_context"):
            print("warning: this checkpoint was not fine-tuned with --with-context; its answers will ignore the passage")
    conv = Conversation(index)
    print(f"{len(index.passages)} story passages indexed. Ask about \"The Verdict\" "
          f"('new' = new conversation, empty line = quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        if question.lower() == "new":
            conv.reset()
            print("(started a new conversation)\n")
            continue

        followup = bool(conv.turns) and is_followup(question)
        hits = conv.search(question, k=PASSAGES_TO_TRY)
        ids = [i for i, _ in hits]
        if lm is None:
            answer, used = conv.best_sentences(question, ids[0]), ids[0]
        else:
            answer, n = answer_from_passages(lm, tok, question, [index.passages[i] for i in ids])
            used = None if n is None else ids[n]
        conv.add(question, answer, ids)

        note = " (read as a follow-up)" if followup else ""
        source = f"passage {used}" if used is not None else f"none of passages {ids}"
        print(f"Model: {answer}" if lm else textwrap.fill(f"Story: {answer}", 100, subsequent_indent="       "))
        print(f"       [{source}{note}]")
        if lm and args.show_passage and used is not None:
            print(textwrap.fill(index.passages[used], 100, initial_indent="       > ", subsequent_indent="       > "))
        print()


if __name__ == "__main__":
    main()
