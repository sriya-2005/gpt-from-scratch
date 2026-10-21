"""Measure how much of "The Verdict" a checkpoint actually knows.

Two tests, both read-only (the checkpoint is never changed):
  1. Prediction accuracy: at every position in the story the model sees the real text
     before it and guesses the next token. We count how often its top guess is right,
     separately for each tenth of the story.
  2. Recitation: the model gets 64 real tokens and continues on its own (greedy, no
     randomness). We count how many tokens it gets right in a row before the first mistake.

Usage:
    python scripts/eval_story.py                                   # both original checkpoints
    python scripts/eval_story.py --ckpt models/checkpoints/verdict/latest.pt
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from inference import REAL_VOCAB, ROOT, load_model
from prepare_data import clean, load_tokenizer

STORY = ROOT / "data" / "raw" / "verdict" / "the-verdict.txt"
TRAIN_FRAC = 0.8          # the original training run used the first 80% of the story
PROMPT_LEN = 64           # real tokens given to the model before each recitation test
RECITE_LEN = 30           # tokens the model generates in each recitation test
DEFAULT_CKPTS = ["models/checkpoints/verdict/best.pt", "models/checkpoints/verdict/latest.pt"]


@torch.no_grad()
def score_tokens(model, ids: list[int], block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (correct, loss) for every token from position 1 onward.

    The story is longer than the model's 256-token context, so we slide a window
    along it in steps of half a window. Each token is scored once, in the window
    where it has the most real text before it.
    """
    n = len(ids)
    tokens = torch.tensor(ids)
    correct = torch.zeros(n, dtype=torch.bool)
    losses = torch.zeros(n)
    window = block_size + 1                    # 256 inputs + 1 extra token for the last target
    stride = block_size // 2
    start, scored_up_to = 0, 1                 # targets before scored_up_to are already done
    while True:
        end = min(start + window, n)
        x = tokens[start:end - 1][None]        # model input
        y = tokens[start + 1:end][None]        # the real next token at every input position
        logits, _ = model(x, y)
        logits = logits[0, :, :REAL_VOCAB]
        first = scored_up_to - (start + 1)     # skip targets an earlier window already scored
        correct[scored_up_to:end] = (logits.argmax(-1) == y[0])[first:]
        losses[scored_up_to:end] = F.cross_entropy(logits, y[0], reduction="none")[first:]
        scored_up_to = end
        if end == n:
            return correct[1:], losses[1:]     # position 0 has nothing before it to predict from
        start += stride


@torch.no_grad()
def recite(model, ids: list[int], pos: int, block_size: int) -> tuple[int, list[int]]:
    """Give the model the PROMPT_LEN real tokens before `pos`, let it continue greedily.

    Returns (how many tokens matched the story in a row, the tokens it generated).
    """
    idx = torch.tensor([ids[pos - PROMPT_LEN:pos]])
    generated = []
    for _ in range(RECITE_LEN):
        logits, _ = model(idx[:, -block_size:])
        next_id = int(logits[0, -1, :REAL_VOCAB].argmax())
        generated.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]])], dim=1)
    run = 0
    for got, want in zip(generated, ids[pos:pos + RECITE_LEN]):
        if got != want:
            break
        run += 1
    return run, generated


def evaluate(ckpt: Path, ids: list[int], tok, trained_frac: float = TRAIN_FRAC) -> dict:
    lm = load_model(ckpt, torch.device("cpu"))
    lm.model.eval()                            # turn off dropout so results are repeatable
    block = lm.config.block_size
    n = len(ids)
    split = int(n * trained_frac)

    correct, losses = score_tokens(lm.model, ids, block)
    # correct[i] / losses[i] belong to story token i + 1
    sections = []
    for k in range(10):
        lo, hi = max(1, n * k // 10), n * (k + 1) // 10
        c, l = correct[lo - 1:hi - 1], losses[lo - 1:hi - 1]
        sections.append({"part": f"{k * 10}-{(k + 1) * 10}%", "trained_on": hi <= split,
                         "accuracy": c.float().mean().item(), "loss": l.mean().item()})
    seen = slice(0, split - 1)
    unseen = slice(split - 1, None)

    probes = []
    for k in range(1, 10):                     # at 10%, 20%, ... 90% of the story
        pos = n * k // 10
        run, generated = recite(lm.model, ids, pos, block)
        probes.append({"at": f"{k * 10}%", "continuation_seen_in_training": pos + RECITE_LEN <= split,
                       "tokens_right_in_a_row": run, "model": tok.decode(generated),
                       "story": tok.decode(ids[pos:pos + RECITE_LEN])})

    return {
        "checkpoint": str(ckpt), "step": lm.step, "story_tokens": n, "trained_on_tokens": split,
        "accuracy_trained_part": correct[seen].float().mean().item(),
        "accuracy_unseen_part": correct[unseen].float().mean().item(),
        "loss_trained_part": losses[seen].mean().item(),
        "loss_unseen_part": losses[unseen].mean().item(),
        "sections": sections, "recitation": probes,
    }


def report(r: dict) -> None:
    print(f"\n=== {r['checkpoint']}  (training step {r['step']}) ===")
    print(f"Prediction accuracy, part it trained on : {r['accuracy_trained_part']:6.1%}   "
          f"(loss {r['loss_trained_part']:.2f})")
    if not math.isnan(r["accuracy_unseen_part"]):   # nan when it trained on the whole story
        print(f"Prediction accuracy, part it never saw  : {r['accuracy_unseen_part']:6.1%}   "
              f"(loss {r['loss_unseen_part']:.2f})")
    print("\nBy tenth of the story:")
    for s in r["sections"]:
        bar = "#" * round(s["accuracy"] * 40)
        print(f"  {s['part']:>8}  {'trained' if s['trained_on'] else 'UNSEEN '}  "
              f"{s['accuracy']:6.1%}  {bar}")
    print(f"\nRecitation (out of {RECITE_LEN} tokens, right in a row):")
    print("  " + "  ".join(f"{p['at']}:{p['tokens_right_in_a_row']}" for p in r["recitation"]))
    example = r["recitation"][2]  # the 30% probe, well inside the trained part
    print(f"\nExample at {example['at']} of the story:")
    print(f"  story: {example['story']!r}")
    print(f"  model: {example['model']!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", default=DEFAULT_CKPTS, help="one or more checkpoint files")
    parser.add_argument("--trained-frac", type=float, default=TRAIN_FRAC,
                        help="share of the story the checkpoint trained on (0.8 originally, 1.0 for verdict_story)")
    args = parser.parse_args()

    tok = load_tokenizer()
    ids = tok.encode(clean(STORY.read_text(encoding="utf-8")))   # same cleaning as prepare_data.py
    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    for ckpt in args.ckpt:
        path = Path(ckpt) if Path(ckpt).is_absolute() else ROOT / ckpt
        result = evaluate(path, ids, tok, args.trained_frac)
        report(result)
        out = out_dir / f"eval_story_{path.parent.name}_{path.stem}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"\nSaved full results to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
