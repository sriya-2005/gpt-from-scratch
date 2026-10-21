"""Sentence-completion test: does the model know facts beyond the exact wording it trained on?

For each fact the model completes two sentence starts:
  (a) the beginning of one of the training fact statements, word for word
  (b) the same fact asked in new wording the model never saw
A model that only memorized word sequences gets (a) right and (b) wrong. Getting (b) right
needs general English (knowing that "the illness that killed" asks the same as "died of").

Usage:
    python scripts/eval_cloze.py --ckpt models/checkpoints/verdict_facts/best.pt
"""
import argparse
from pathlib import Path

from inference import ROOT, SamplingParams, load_model, load_tokenizer, stream_generate

# (fact, expected word, (a) training wording, (b) new wording)
TESTS = [
    ("F33", "Stroud", "The donkey sketch was made by", "The man who drew the tired donkey was"),
    ("F38", "heart", "Stroud died suddenly of", "The illness that killed Stroud was"),
    ("F42", "Devonshire", "Stroud made the donkey drawing in", "The county where Stroud drew the donkey was"),
    ("F02", "widow", "Jack Gisburn married a", "Before she married Jack, Mrs. Gisburn had been a"),
    ("F48", "Grindle", "It was Jack Gisburn who started", "The young painter whose career Jack launched was"),
    ("F13", "Rome", "Rickham had expected Jack to settle in", "The narrator thought Jack would go to live in"),
    ("F17", "beauty", "Jack said that money's only excuse is to", "According to Jack, the only reason to have money is to"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    args = parser.parse_args()
    tok = load_tokenizer()
    params = SamplingParams(temperature=0, max_new_tokens=8, stop_at_eos=True)
    for ckpt in args.ckpt:
        path = Path(ckpt) if Path(ckpt).is_absolute() else ROOT / ckpt
        lm = load_model(path)
        print(f"\n=== {path.relative_to(ROOT)}")
        score = {"a": 0, "b": 0}
        for fact, expected, exact, new in TESTS:
            for key, prompt in (("a", exact), ("b", new)):
                out = "".join(t for _, t in stream_generate(lm, tok, prompt, params)).split("\n")[0]
                ok = expected.lower() in out.lower()
                score[key] += ok
                print(f"{fact} ({key}) {'OK ' if ok else '-- '} {prompt} ->{out!r}")
        print(f"(a) training wording: {score['a']}/{len(TESTS)}   (b) new wording: {score['b']}/{len(TESTS)}")


if __name__ == "__main__":
    main()
