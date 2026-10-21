"""Fine-tune the story-trained GPT so it answers questions about "The Verdict".

What happens in each training step:
  1. Take 8 training questions, formatted as "Question: ...\nAnswer: ...<|endoftext|>".
     Loss is measured on the answer tokens only (question tokens are labelled IGNORE).
  2. Take one random 256-token passage of the story, with loss on every token.
     This "rehearsal" keeps the model from forgetting the story while it learns the format.
  3. total loss = QA loss + story_weight x story loss; one AdamW update on all weights.

Every --eval-every epochs the model answers the validation questions, and the checkpoint with
the best answer score (word-overlap F1) is kept as best.pt. latest.pt holds everything needed
to resume.

Usage:
    python scripts/finetune_qa.py                 # starts from models/checkpoints/verdict_story/best.pt
    python scripts/finetune_qa.py --resume        # continue an interrupted run
"""
import argparse
import csv
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataloader import PROCESSED_DIR, TokenDataset
from eval_qa import evaluate
from inference import load_tokenizer
from model import GPT, GPTConfig, get_device
from qa_format import IGNORE, encode_example, load_split

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", default="models/checkpoints/verdict_story/best.pt",
                   help="checkpoint whose weights we start from (the story-trained model)")
    p.add_argument("--run-name", default="verdict_qa", help="output folder under models/checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="questions per step")
    p.add_argument("--lr", type=float, default=5e-5, help="peak learning rate (lower than pretraining)")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--story-weight", type=float, default=0.5, help="weight of the story rehearsal loss")
    p.add_argument("--rehearsal-data", default="verdict_full",
                   help="dataset folder under data/processed used for rehearsal (verdict_facts adds the statements)")
    p.add_argument("--eval-every", type=int, default=2, help="evaluate every N epochs")
    p.add_argument("--patience", type=int, default=4, help="stop after N evals without a better val F1 (0 = never)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def make_batch(examples: list[tuple[list[int], list[int]]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad examples to the same length. Padding goes at the END and is labelled IGNORE:
    the model only looks backwards, so tokens added after an example cannot change it."""
    longest = max(len(x) for x, _ in examples)
    xs = [x + [pad_id] * (longest - len(x)) for x, _ in examples]
    ys = [y + [IGNORE] * (longest - len(y)) for _, y in examples]
    return torch.tensor(xs, device=device), torch.tensor(ys, device=device)


@torch.no_grad()
def answer_loss(model, examples, pad_id, device, batch_size: int) -> float:
    """Average loss per answer token over a set of examples."""
    total, count = 0.0, 0
    for i in range(0, len(examples), batch_size):
        x, y = make_batch(examples[i:i + batch_size], pad_id, device)
        _, loss = model(x, y)
        n = int((y != IGNORE).sum())
        total, count = total + loss.item() * n, count + n
    return total / count


@torch.no_grad()
def story_loss(model, story: TokenDataset, device) -> float:
    """Average loss over the whole story in fixed 256-token blocks: did it forget the story?"""
    losses = []
    for i in range(len(story)):
        x, y = story[i]
        _, loss = model(x[None].to(device), y[None].to(device))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def lr_at(step: int, args, total_steps: int) -> float:
    """Linear warm-up, then cosine decay to lr * min_lr_ratio (same shape as train.py)."""
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
    min_lr = args.lr * args.min_lr_ratio
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - min_lr)


def save_plot(history, evals, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    if history:
        steps = [h["step"] for h in history]
        axes[0].plot(steps, [h["qa_loss"] for h in history], label="QA answer loss", color="tab:blue", alpha=0.8)
        axes[0].plot(steps, [h["story_loss"] for h in history], label="story rehearsal loss", color="tab:green",
                     alpha=0.8)
        axes[0].legend()
    axes[0].set_title("Training loss per step")
    axes[0].set_xlabel("step")
    if evals:
        es = [e["step"] for e in evals]
        axes[1].plot(es, [e["val_answer_loss"] for e in evals], "o-", label="val answer loss", color="tab:orange")
        axes[1].plot(es, [e["story_loss"] for e in evals], "o-", label="whole-story loss", color="tab:green")
        axes[2].plot(es, [e["val_f1"] for e in evals], "o-", color="tab:purple")
        axes[1].legend()
    axes[1].set_title("Validation losses")
    axes[1].set_xlabel("step")
    axes[2].set_title("Validation answer F1 (higher is better)")
    axes[2].set_xlabel("step")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)
    tok = load_tokenizer()
    pad_id = tok.eos_token_id

    run_dir = ROOT / "models" / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, latest_path = run_dir / "best.pt", run_dir / "latest.pt"

    # ---- data ----
    train_rows, val_rows = load_split("train"), load_split("val")
    train_ex = [encode_example(tok, r["question"], r["answer"]) for r in train_rows]
    val_ex = [encode_example(tok, r["question"], r["answer"]) for r in val_rows]
    story_bin = PROCESSED_DIR / args.rehearsal_data / "train.bin"
    if not story_bin.exists():
        sys.exit(f"Missing {story_bin.parent}. Run: python scripts/prepare_data.py --dataset {args.rehearsal_data}")

    # ---- model: start from the story-trained weights ----
    init_path = Path(args.init_from) if Path(args.init_from).is_absolute() else ROOT / args.init_from
    init = torch.load(init_path, map_location=device, weights_only=False)
    config = GPTConfig(**{**init["config"], "dropout": args.dropout, "gradient_checkpointing": False})
    model = GPT(config).to(device)
    model.load_state_dict(init["model"])
    optimizer = model.configure_optimizer(lr=args.lr, weight_decay=args.weight_decay, device_type=device.type)
    story_train = TokenDataset(story_bin, config.block_size, random_windows=True)
    story_eval = TokenDataset(story_bin, config.block_size, random_windows=False)

    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    step, start_epoch, best_f1, evals_since_best = 0, 0, -1.0, 0
    history, evals = [], []

    if args.resume:
        if not latest_path.exists():
            sys.exit(f"--resume given but {latest_path} does not exist")
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step, start_epoch = ck["step"], ck["epoch"]
        best_f1, evals_since_best = ck["best_f1"], ck["evals_since_best"]
        history, evals = ck["history"], ck["evals"]
        torch.set_rng_state(ck["rng_state"])
        print(f"Resumed {latest_path.relative_to(ROOT)} after epoch {start_epoch} (best val F1 {best_f1:.3f})")
    else:
        print(f"Starting from {init_path.relative_to(ROOT)} (its training step {init.get('step')})")

    print(f"{len(train_ex)} training questions, {len(val_ex)} validation questions, "
          f"{steps_per_epoch} steps per epoch, up to {args.epochs} epochs ({total_steps} steps)")
    print(f"lr {args.lr}, dropout {args.dropout}, story weight {args.story_weight}, "
          f"evaluating every {args.eval_every} epochs\n")

    lm = SimpleNamespace(model=model, config=config, device=device)  # what evaluate() needs

    def save(path: Path, full: bool, metrics: dict) -> None:
        state = {"model": model.state_dict(), "config": asdict(config), "step": step,
                 "val_loss": metrics["val_answer_loss"], "val_f1": metrics["val_f1"],
                 "init_from": str(init_path), "args": vars(args)}
        if full:
            state |= {"optimizer": optimizer.state_dict(), "epoch": metrics["epoch"], "best_f1": best_f1,
                      "evals_since_best": evals_since_best, "history": history, "evals": evals,
                      "rng_state": torch.get_rng_state()}
        tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def run_eval(epoch: int) -> bool:
        """Score the model on the validation questions; save checkpoints; return True to stop."""
        nonlocal best_f1, evals_since_best
        model.eval()
        t0 = time.time()
        v_loss = answer_loss(model, val_ex, pad_id, device, args.batch_size)
        s_loss = story_loss(model, story_eval, device)
        scores = evaluate(lm, tok, val_rows, show=False)
        model.train()
        recent = history[-steps_per_epoch * args.eval_every:]
        record = {"epoch": epoch, "step": step, "val_answer_loss": v_loss, "story_loss": s_loss,
                  "val_f1": scores["mean_f1"], "val_f1_half": scores["f1_at_least_half"],
                  "unanswerable_correct": scores["unanswerable_correct"],
                  "train_qa_loss": sum(h["qa_loss"] for h in recent) / len(recent) if recent else float("nan")}
        evals.append(record)

        improved = record["val_f1"] > best_f1 + 0.005
        if improved:
            best_f1, evals_since_best = record["val_f1"], 0
            save(best_path, full=False, metrics=record)
        else:
            evals_since_best += 1
        save(latest_path, full=True, metrics=record)

        with open(run_dir / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(record))
            w.writeheader()
            w.writerows(evals)
        save_plot(history, evals, run_dir / "curves.png")

        tqdm.write(f"[eval] epoch {epoch:>2} step {step:>4} | train QA loss {record['train_qa_loss']:.3f} | "
                   f"val answer loss {v_loss:.3f} | story loss {s_loss:.3f} | val F1 {record['val_f1']:.3f} "
                   f"({record['val_f1_half']}/{scores['n_answerable']} >= 0.5) | unanswerable "
                   f"{record['unanswerable_correct']}/{scores['n_unanswerable']} | "
                   f"{'NEW BEST' if improved else f'best {best_f1:.3f}'} | {time.time() - t0:.0f}s")
        for r in scores["results"][:3]:
            tqdm.write(f"         Q: {r['question']}\n         A: {r['model_answer']!r}")
        if args.patience and evals_since_best >= args.patience:
            tqdm.write(f"Validation F1 has not improved in {args.patience} evaluations; stopping.")
            return True
        return False

    if not args.resume:
        run_eval(epoch=0)  # baseline: how the starting model answers before any QA training

    model.train()
    pbar = tqdm(total=total_steps, initial=step, unit="step", dynamic_ncols=True)
    stopped = False
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(train_ex)))
        random.Random(args.seed + epoch).shuffle(order)          # new question order each epoch
        for b in range(steps_per_epoch):
            lr = lr_at(step, args, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = make_batch([train_ex[i] for i in order[b * args.batch_size:(b + 1) * args.batch_size]],
                              pad_id, device)
            sx, sy = story_train[0]                               # a random story passage
            _, qa_loss = model(x, y)
            _, st_loss = model(sx[None].to(device), sy[None].to(device))
            loss = qa_loss + args.story_weight * st_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            history.append({"step": step, "qa_loss": qa_loss.item(), "story_loss": st_loss.item(), "lr": lr})
            pbar.update(1)
            pbar.set_postfix(epoch=epoch + 1, qa=f"{qa_loss.item():.3f}", story=f"{st_loss.item():.3f}",
                             lr=f"{lr:.1e}")
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            if run_eval(epoch + 1):
                stopped = True
                break
    pbar.close()

    print(f"\n{'Stopped early' if stopped else 'Finished'} at step {step}. Best validation F1: {best_f1:.3f}")
    print(f"best checkpoint: {best_path.relative_to(ROOT)}\nresume state:    {latest_path.relative_to(ROOT)}")
    print(f"curves: {(run_dir / 'curves.png').relative_to(ROOT)}   log: {(run_dir / 'log.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
