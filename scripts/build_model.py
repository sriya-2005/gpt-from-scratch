"""Build the GPT model, print a summary and parameter count, and run one
mixed-precision, gradient-checkpointed training step on real data to verify it works."""
import time

import torch

from dataloader import create_dataloaders
from model import GPT, GPTConfig, autocast_settings, get_device

BATCH_SIZE = 2
BLOCK_SIZE = 256  # short context for the smoke test; the model supports up to config.block_size


def fmt(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n:,}"


def print_summary(model: GPT) -> None:
    cfg = model.config
    print("=== GPT config ===")
    for k, v in vars(cfg).items():
        print(f"  {k:24s} {v}")

    print("\n=== Model summary ===")
    print(f"{'Layer':42s} {'Shape':>22s} {'Params':>12s}")
    print("-" * 78)
    t = model.transformer
    rows = [
        ("transformer.wte (token embedding)", tuple(t.wte.weight.shape), t.wte.weight.numel()),
        ("transformer.wpe (position embedding)", tuple(t.wpe.weight.shape), t.wpe.weight.numel()),
    ]
    blk = t.h[0]
    for name, mod in [("ln_1 (LayerNorm)", blk.ln_1), ("attn.c_attn (Q,K,V proj)", blk.attn.c_attn),
                      ("attn.c_proj (output proj)", blk.attn.c_proj), ("ln_2 (LayerNorm)", blk.ln_2),
                      ("mlp.c_fc (FFN up, 4x)", blk.mlp.c_fc), ("mlp.c_proj (FFN down)", blk.mlp.c_proj)]:
        rows.append((f"  h[i].{name}", tuple(mod.weight.shape), sum(p.numel() for p in mod.parameters())))
    per_block = sum(p.numel() for p in blk.parameters())
    rows.append((f"transformer.h  (Block x {cfg.n_layer})", f"{fmt(per_block)} each", per_block * cfg.n_layer))
    rows.append(("transformer.ln_f (final LayerNorm)", tuple(t.ln_f.weight.shape),
                 sum(p.numel() for p in t.ln_f.parameters())))
    rows.append(("lm_head (tied to wte)", tuple(model.lm_head.weight.shape), 0))
    for name, shape, n in rows:
        print(f"{name:42s} {str(shape):>22s} {n:>12,}")
    print("-" * 78)

    total = model.num_params()
    print(f"Total parameters:          {total:>14,}  ({total / 1e6:.1f}M)")
    print(f"Non-embedding parameters:  {model.num_params(non_embedding=True):>14,}")
    print(f"Trainable parameters:      {sum(p.numel() for p in model.parameters() if p.requires_grad):>14,}")
    print(f"fp32 weights size:         {total * 4 / 1024**2:>11.0f} MB "
          f"(+ ~{total * 12 / 1024**2:.0f} MB for grads and AdamW state during training)")


def main() -> None:
    torch.manual_seed(1337)
    device = get_device()
    amp_dtype, use_scaler = autocast_settings(device)

    model = GPT(GPTConfig(gradient_checkpointing=True)).to(device)
    print_summary(model)

    optimizer = model.configure_optimizer(lr=6e-4, weight_decay=0.1, device_type=device.type)
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    n_decay, n_no_decay = (sum(p.numel() for p in g["params"]) for g in optimizer.param_groups)
    print("\n=== Training setup ===")
    amp_desc = f"{amp_dtype} autocast" if amp_dtype != torch.float32 else "off (fp32; bf16 is slower on this CPU)"
    print(f"device: {device}   mixed precision: {amp_desc}   GradScaler: {use_scaler}")
    print(f"gradient checkpointing: {model.config.gradient_checkpointing}")
    print(f"optimizer: AdamW lr=6e-4 betas=(0.9, 0.95) weight_decay=0.1 "
          f"on {n_decay:,} params, 0.0 on {n_no_decay:,}")
    print("objective: next-token prediction (cross-entropy of logits[t] vs token[t+1])")

    train_loader, _ = create_dataloaders(BATCH_SIZE, BLOCK_SIZE)
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    print(f"\n=== Smoke test: {BATCH_SIZE}x{BLOCK_SIZE} batch, 3 training steps ===")
    print(f"expected initial loss ~ ln(50257) = {torch.log(torch.tensor(50257.0)):.2f}")
    model.train()
    for step in range(3):
        t0 = time.time()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            logits, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        print(f"step {step}: loss {loss.item():.4f}  grad_norm {grad_norm:.3f}  "
              f"logits {tuple(logits.shape)} {logits.dtype}  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
