"""Train the GPT model on a tokenized dataset from data/processed/<name>/.

Features: mixed precision, gradient accumulation, gradient clipping, warmup + cosine
LR schedule, periodic validation, checkpointing + exact resume, early stopping when
validation loss plateaus, tqdm progress bars, and loss curves saved as PNG.

Usage:
    python scripts/train.py --preset verdict               # 124M model on "The Verdict" (minutes on CPU)
    python scripts/train.py --preset gpt2                  # 124M model on Wikipedia, for a CUDA GPU
    python scripts/train.py --preset small                 # ~30M model on Wikipedia, feasible on CPU
    python scripts/train.py --preset gpt2 --resume         # continue from the run's latest checkpoint
    python scripts/train.py --preset small --max-steps 50  # quick test
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataloader import PROCESSED_DIR, TokenDataset, load_meta
from model import GPT, GPTConfig, autocast_settings, get_device

ROOT = Path(__file__).resolve().parent.parent

# used when neither the command line nor the preset sets a value
DEFAULTS = dict(data="wikipedia", epochs=3, eval_interval=250, ckpt_interval=500, dropout=0.0,
                random_windows=False, init_from=None, patience=5)

PRESETS = {
    # GPT-2 small: 124M params. 16 x 1024 x 4 accum = 65,536 tokens per optimizer step.
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                 batch_size=16, grad_accum=4, lr=6e-4, warmup_steps=200, grad_ckpt=True),
    # ~30M params with short context, sized for CPU experiments. 16 x 256 = 4,096 tokens per step.
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                  batch_size=16, grad_accum=1, lr=1e-3, warmup_steps=200, grad_ckpt=False),
    # The same 124M model on "The Verdict" (4,116 train tokens = 16 blocks of 256 -> 8 steps/epoch).
    # The data is tiny, so the model overfits within a few epochs: evaluate every epoch, use dropout,
    # and let early stopping keep the best-validation weights. Checkpoints only at the end (~1.4 GB).
    "verdict": dict(n_layer=12, n_head=12, n_embd=768, block_size=256,
                    batch_size=2, grad_accum=1, lr=4e-4, warmup_steps=8, grad_ckpt=False,
                    data="verdict", epochs=20, eval_interval=8, ckpt_interval=10**9, dropout=0.1),
    # QA step 2: continue from the original Verdict weights and learn the WHOLE story.
    # Random window starts stop the model tying text to fixed positions; dropout is off
    # because here we want it to store the story. 20 blocks / batch 2 = 10 steps per epoch.
    # No early stopping (patience=0): the step-0 score is flattered by the old fixed block
    # positions, so the loss rises for a few epochs before it falls.
    "verdict_story": dict(n_layer=12, n_head=12, n_embd=768, block_size=256,
                          batch_size=2, grad_accum=1, lr=3e-4, warmup_steps=20, grad_ckpt=False,
                          data="verdict_full", epochs=40, eval_interval=10, ckpt_interval=10**9,
                          dropout=0.0, random_windows=True, patience=0,
                          init_from="models/checkpoints/verdict/latest.pt"),
    # QA step 5: keep reading the story, now together with 4 reworded statements of every
    # fact, so facts are stored as facts rather than as one exact sentence. Continues from
    # the story-trained model. ~35 blocks / batch 2 = 17 steps per epoch.
    "verdict_facts": dict(n_layer=12, n_head=12, n_embd=768, block_size=256,
                          batch_size=2, grad_accum=1, lr=2e-4, warmup_steps=20, grad_ckpt=False,
                          data="verdict_facts", epochs=25, eval_interval=17, ckpt_interval=10**9,
                          dropout=0.0, random_windows=True, patience=0,
                          init_from="models/checkpoints/verdict_story/best.pt"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=PRESETS, default="gpt2")
    p.add_argument("--run-name", help="defaults to the preset name")
    p.add_argument("--data", help="dataset folder under data/processed (e.g. wikipedia, verdict)")
    # anything below left as None falls back to the preset's value
    p.add_argument("--n-layer", type=int)
    p.add_argument("--n-head", type=int)
    p.add_argument("--n-embd", type=int)
    p.add_argument("--block-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float)
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--grad-ckpt", type=int, choices=[0, 1], help="gradient checkpointing on/off")
    p.add_argument("--random-windows", type=int, choices=[0, 1],
                   help="start each training block at a random token instead of fixed positions")
    p.add_argument("--init-from", help="start from this checkpoint's weights instead of random ones")
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine decays to lr * this")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--max-steps", type=int, help="cap total optimizer steps (overrides --epochs)")
    p.add_argument("--time-limit", type=float, help="stop (with a checkpoint) after this many hours")
    p.add_argument("--eval-interval", type=int)
    p.add_argument("--eval-batches", type=int, default=100, help="val batches per evaluation")
    p.add_argument("--ckpt-interval", type=int)
    p.add_argument("--patience", type=int,
                   help="stop after this many evals without val improvement (0 = never)")
    p.add_argument("--min-delta", type=float, default=0.005, help="val improvement that counts as progress")
    p.add_argument("--num-workers", type=int,
                   help="DataLoader workers; default 2 on GPU, 0 on CPU (workers compete with compute threads)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (needs Triton)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    for k, v in {**DEFAULTS, **PRESETS[args.preset]}.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.grad_ckpt = bool(args.grad_ckpt)
    args.random_windows = bool(args.random_windows)
    args.run_name = args.run_name or args.preset
    return args


def lr_at(step: int, args, total_steps: int) -> float:
    """Linear warmup, then cosine decay to lr * min_lr_ratio."""
    min_lr = args.lr * args.min_lr_ratio
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - min_lr)


def epoch_loader(dataset, epoch: int, start_batch: int, args, pin: bool) -> DataLoader:
    """Shuffled loader for one epoch. The order depends only on (seed, epoch), so a resumed
    run skips straight to `start_batch` without re-reading the batches it already saw."""
    g = torch.Generator().manual_seed(args.seed + epoch)
    order = torch.randperm(len(dataset), generator=g)
    n_batches = len(dataset) // args.batch_size
    order = order[start_batch * args.batch_size: n_batches * args.batch_size].tolist()
    return DataLoader(Subset(dataset, order), batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pin,
                      persistent_workers=False, drop_last=True)


@torch.no_grad()
def evaluate(model, loader, device, amp_ctx) -> float:
    model.eval()
    total, n = torch.zeros((), device=device), 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_ctx():
            _, loss = model(x, y)
        total += loss
        n += 1
    model.train()
    return (total / n).item()


def save_plot(history: list[dict], evals: list[dict], path: Path, steps_per_epoch: int) -> None:
    if not history:
        return
    steps = np.array([h["step"] for h in history])
    losses = np.array([h["loss"] for h in history])
    window = max(1, min(50, len(losses) // 10))
    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True)
    ax1.plot(steps, losses, color="tab:blue", alpha=0.25, linewidth=0.8, label="train (per step)")
    ax1.plot(steps[window - 1:], smooth, color="tab:blue", linewidth=1.8, label=f"train ({window}-step avg)")
    if evals:
        ax1.plot([e["step"] for e in evals], [e["val_loss"] for e in evals], "o-", color="tab:orange",
                 linewidth=1.8, markersize=4, label="validation")
    for ep in range(1, int(steps[-1] // steps_per_epoch) + 1):
        ax1.axvline(ep * steps_per_epoch, color="gray", linestyle=":", linewidth=1)
        ax1.text(ep * steps_per_epoch, ax1.get_ylim()[1], f" epoch {ep}", va="top", fontsize=8, color="gray")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Training progress")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(steps, [h["lr"] for h in history], color="tab:green")
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("optimizer step")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    amp_dtype, use_scaler = autocast_settings(device)
    if args.num_workers is None:
        args.num_workers = 2 if device.type == "cuda" else 0
    amp_ctx = lambda: torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 tensor cores for any remaining fp32 matmuls
        torch.backends.cudnn.allow_tf32 = True

    run_dir = ROOT / "models" / "checkpoints" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, best_path = run_dir / "latest.pt", run_dir / "best.pt"
    log_path, plot_path = run_dir / "log.csv", run_dir / "loss_curve.png"

    # ---- data ----
    data_dir = PROCESSED_DIR / args.data
    if not (data_dir / "meta.json").exists():
        sys.exit(f"No dataset at {data_dir}. Run: python scripts/prepare_data.py --dataset {args.data}")
    meta = load_meta(data_dir)
    train_ds = TokenDataset(data_dir / "train.bin", args.block_size, random_windows=args.random_windows)
    val_ds = TokenDataset(data_dir / "val.bin", args.block_size)
    # fixed random subset of validation blocks, so every evaluation is comparable
    g = torch.Generator().manual_seed(0)
    val_idx = torch.randperm(len(val_ds), generator=g)[:args.eval_batches * args.batch_size].tolist()
    pin = device.type == "cuda"
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.batch_size, num_workers=0, pin_memory=pin)

    batches_per_epoch = len(train_ds) // args.batch_size
    steps_per_epoch = batches_per_epoch // args.grad_accum
    if steps_per_epoch == 0 or len(val_ds) == 0:
        sys.exit(f"Dataset '{args.data}' is too small for batch {args.batch_size} x block {args.block_size} "
                 f"x accum {args.grad_accum}: {len(train_ds)} train / {len(val_ds)} val blocks. "
                 "Lower --batch-size, --block-size or --grad-accum.")
    total_steps = args.max_steps or math.ceil(args.epochs * steps_per_epoch)
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum

    # ---- model / optimizer ----
    config = GPTConfig(vocab_size=50304, block_size=args.block_size, n_layer=args.n_layer,
                       n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
                       gradient_checkpointing=args.grad_ckpt)
    model = GPT(config).to(device)
    if args.init_from and not args.resume:
        # Fine-tuning: copy in previously trained weights. Only the weights are taken; the
        # optimizer starts fresh, because this is a new training phase with its own schedule.
        init_path = Path(args.init_from) if Path(args.init_from).is_absolute() else ROOT / args.init_from
        init = torch.load(init_path, map_location=device, weights_only=False)
        shape_keys = ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "bias")
        mismatched = [k for k in shape_keys if init["config"][k] != getattr(config, k)]
        if mismatched:
            sys.exit(f"--init-from {init_path} has a different architecture ({', '.join(mismatched)}); "
                     "its weights cannot be loaded into this model.")
        model.load_state_dict(init["model"])
        print(f"Initialized weights from {init_path} (its training step {init.get('step')})")
    optimizer = model.configure_optimizer(lr=args.lr, weight_decay=args.weight_decay, device_type=device.type)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    step, best_val, evals_since_best = 0, float("inf"), 0
    history, evals = [], []
    if args.resume:
        if not ckpt_path.exists():
            sys.exit(f"--resume given but no checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt["config"] != asdict(config):
            sys.exit("checkpoint model config differs from the current arguments")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, best_val, evals_since_best = ckpt["step"], ckpt["best_val"], ckpt["evals_since_best"]
        history, evals = ckpt["history"], ckpt["evals"]
        torch.set_rng_state(ckpt["rng_state"])
        print(f"Resumed from {ckpt_path} at step {step} (best val {best_val:.4f})")
    else:
        history_file_note = " (overwriting previous run)" if ckpt_path.exists() else ""
        print(f"Starting new run in {run_dir}{history_file_note}")

    raw_model = model
    if args.compile:
        model = torch.compile(model)

    print(f"device={device}  mixed precision={'off' if amp_dtype == torch.float32 else amp_dtype}  "
          f"grad checkpointing={args.grad_ckpt}  compile={args.compile}")
    print(f"model: {raw_model.num_params() / 1e6:.1f}M params  "
          f"({args.n_layer} layers, {args.n_embd} dim, {args.n_head} heads, context {args.block_size})")
    print(f"data: {args.data}, {meta['train']['tokens']:,} train tokens, {len(train_ds):,} blocks  |  "
          f"val: {args.eval_batches} batches ({len(val_idx) * args.block_size:,} tokens) per eval")
    print(f"batch: {args.batch_size} x {args.block_size} x {args.grad_accum} accum = {tokens_per_step:,} tokens/step  |  "
          f"{steps_per_epoch:,} steps/epoch  |  training for {total_steps:,} steps "
          f"({total_steps / steps_per_epoch:.2f} epochs)")

    def save_checkpoint(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({
            "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "config": asdict(config), "args": vars(args), "step": step, "best_val": best_val,
            "evals_since_best": evals_since_best, "history": history, "evals": evals,
            "rng_state": torch.get_rng_state(),
        }, tmp)
        os.replace(tmp, path)  # atomic: a crash mid-save never corrupts the previous checkpoint

    def write_log() -> None:
        eval_by_step = {e["step"]: e["val_loss"] for e in evals}
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "tokens_seen"])
            for h in history:
                w.writerow([h["step"], f"{h['step'] / steps_per_epoch:.4f}", f"{h['loss']:.5f}",
                            f"{eval_by_step[h['step']]:.5f}" if h["step"] in eval_by_step else "",
                            f"{h['lr']:.3e}", h["step"] * tokens_per_step])

    def run_eval() -> bool:
        """Evaluate, save best weights, update plot. Returns True if training should stop early."""
        nonlocal best_val, evals_since_best
        val_loss = evaluate(model, val_loader, device, amp_ctx)
        recent = [h["loss"] for h in history[-args.eval_interval:]]
        train_loss = sum(recent) / len(recent) if recent else float("nan")
        evals.append({"step": step, "val_loss": val_loss, "train_loss": train_loss})
        improved = val_loss < best_val - args.min_delta
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": raw_model.state_dict(), "config": asdict(config), "step": step,
                        "val_loss": val_loss}, best_path)
        evals_since_best = 0 if improved else evals_since_best + 1
        save_plot(history, evals, plot_path, steps_per_epoch)
        write_log()
        tqdm.write(f"[eval] step {step:>6}  epoch {step / steps_per_epoch:.2f}  train {train_loss:.4f}  "
                   f"val {val_loss:.4f}  ppl {math.exp(val_loss):.1f}  best {best_val:.4f}"
                   f"{'  *' if val_loss == best_val else ''}")
        if args.patience and evals_since_best >= args.patience:
            tqdm.write(f"Validation loss hasn't improved by {args.min_delta} in {args.patience} evals; stopping.")
            return True
        return False

    # ---- training loop ----
    model.train()
    start_time = time.time()
    stop_reason = "reached step target"
    if step == 0:
        run_eval()  # baseline before any training

    pbar = tqdm(total=total_steps, initial=step, unit="step", dynamic_ncols=True)
    t_last = time.time()
    while step < total_steps:
        epoch = step // steps_per_epoch
        batch_in_epoch = (step % steps_per_epoch) * args.grad_accum
        batches = iter(epoch_loader(train_ds, epoch, batch_in_epoch, args, pin))

        stop = False
        while step < total_steps and step // steps_per_epoch == epoch:
            lr = lr_at(step, args, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss_sum = torch.zeros((), device=device)
            for _ in range(args.grad_accum):
                x, y = next(batches)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with amp_ctx():
                    _, loss = model(x, y)
                loss = loss / args.grad_accum  # average gradients over the accumulated micro-batches
                scaler.scale(loss).backward()
                loss_sum += loss.detach()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            loss_val = loss_sum.item()
            history.append({"step": step, "loss": loss_val, "lr": lr})
            now = time.time()
            tok_s = tokens_per_step / (now - t_last)
            t_last = now
            pbar.update(1)
            pbar.set_postfix(epoch=f"{step / steps_per_epoch:.2f}", loss=f"{loss_val:.3f}",
                             lr=f"{lr:.1e}", gnorm=f"{grad_norm:.2f}", tok_s=f"{tok_s:,.0f}")

            if not math.isfinite(loss_val):
                stop, stop_reason = True, "loss became NaN/inf"
            if step % args.eval_interval == 0 and run_eval():
                stop, stop_reason = True, "validation loss plateaued"
            if step % args.ckpt_interval == 0:
                save_checkpoint(ckpt_path)
                tqdm.write(f"[ckpt] saved {ckpt_path.relative_to(ROOT)} at step {step}")
            if args.time_limit and now - start_time > args.time_limit * 3600:
                stop, stop_reason = True, f"time limit of {args.time_limit}h reached"
            if stop:
                break
        if stop:
            break
    pbar.close()

    if not evals or evals[-1]["step"] != step:
        run_eval()
    save_checkpoint(ckpt_path)
    save_plot(history, evals, plot_path, steps_per_epoch)
    write_log()
    (run_dir / "summary.json").write_text(json.dumps({
        "stop_reason": stop_reason, "steps": step, "epochs": step / steps_per_epoch,
        "tokens_seen": step * tokens_per_step, "best_val_loss": best_val,
        "final_val_loss": evals[-1]["val_loss"], "wall_time_s": time.time() - start_time,
        "params": raw_model.num_params(), "args": vars(args),
    }, indent=2))
    print(f"\nStopped: {stop_reason} at step {step} ({step / steps_per_epoch:.2f} epochs)")
    print(f"best val loss {best_val:.4f} (perplexity {math.exp(best_val):.1f})")
    print(f"checkpoints: {ckpt_path}, {best_path}\nloss curve: {plot_path}\nlog: {log_path}")


if __name__ == "__main__":
    main()
