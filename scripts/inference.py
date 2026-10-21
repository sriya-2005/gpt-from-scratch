"""Load a trained GPT checkpoint and generate text with temperature / top-k / top-p sampling.

Usage:
    python scripts/inference.py --prompt "The history of Rome"
    python scripts/inference.py --ckpt models/checkpoints/gpt2/best.pt --prompt "Paris is" \
        --temperature 0.8 --top-k 50 --top-p 0.95 --max-new-tokens 200
"""
import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from model import GPT, GPTConfig, get_device

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "models" / "checkpoints"
REAL_VOCAB = 50257  # GPT-2 vocab; ids above this exist only as padding in the model's output layer


@dataclass
class SamplingParams:
    temperature: float = 0.8   # 0 = greedy (always pick the most likely token)
    top_k: int = 50            # 0 = disabled
    top_p: float = 0.95        # 1.0 = disabled
    max_new_tokens: int = 200
    stop_at_eos: bool = True
    seed: int | None = None


@dataclass
class LoadedModel:
    model: GPT
    config: GPTConfig
    device: torch.device
    path: Path
    step: int | None
    val_loss: float | None

    def describe(self) -> str:
        c = self.config
        parts = [f"{self.model.num_params() / 1e6:.1f}M params",
                 f"{c.n_layer}L/{c.n_embd}d/{c.n_head}h", f"context {c.block_size}"]
        if self.step is not None:
            parts.append(f"step {self.step:,}")
        if self.val_loss is not None:
            parts.append(f"val loss {self.val_loss:.3f}")
        parts.append(f"on {self.device.type}")
        return "  |  ".join(parts)


def find_checkpoints() -> list[Path]:
    """All best.pt / latest.pt files under models/checkpoints, best first, newest runs first."""
    found = sorted(CKPT_DIR.glob("*/*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sorted(found, key=lambda p: p.name != "best.pt")


def load_model(path: Path, device: torch.device | None = None) -> LoadedModel:
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**{**ckpt["config"], "gradient_checkpointing": False, "dropout": 0.0})
    model = GPT(config)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}  # torch.compile prefix
    model.load_state_dict(state)
    model.to(device).eval()
    val_loss = ckpt.get("val_loss")
    if val_loss is None and ckpt.get("evals"):
        val_loss = ckpt["evals"][-1]["val_loss"]
    return LoadedModel(model, config, device, Path(path), ckpt.get("step"), val_loss)


def load_tokenizer() -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained("gpt2")


def sample_next_token(logits: torch.Tensor, params: SamplingParams,
                      generator: torch.Generator | None = None) -> int:
    """Pick the next token id from a 1-D logits vector."""
    logits = logits.float().clone()
    logits[REAL_VOCAB:] = -float("inf")  # never sample padding ids

    if params.temperature <= 0:
        return int(torch.argmax(logits))
    logits /= params.temperature

    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.numel())).values[-1]
        logits[logits < kth] = -float("inf")

    if params.top_p < 1.0:
        # nucleus sampling: keep the smallest set of tokens whose probabilities sum to >= top_p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = probs.cumsum(dim=-1) - probs > params.top_p  # tokens fully past the threshold
        remove[0] = False                                      # always keep the top token
        logits[sorted_idx[remove]] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


def stream_generate(lm: LoadedModel, tokenizer: GPT2TokenizerFast, prompt: str, params: SamplingParams,
                    should_stop=lambda: False) -> Iterator[tuple[int, str]]:
    """Yield (token_id, new_text) as each token is generated.

    new_text can be "" when a token is only part of a multi-byte UTF-8 character;
    its text is emitted together with the token that completes the character.
    """
    generator = None
    if params.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(params.seed)

    ids = tokenizer.encode(prompt) if prompt else []
    if not ids:
        ids = [tokenizer.eos_token_id]  # empty prompt: start as if at a document boundary
    prompt_len = len(ids)
    idx = torch.tensor([ids], dtype=torch.long, device=lm.device)
    emitted = ""

    for _ in range(params.max_new_tokens):
        if should_stop():
            return
        with torch.no_grad():
            logits, _ = lm.model(idx[:, -lm.config.block_size:])
        next_id = sample_next_token(logits[0, -1].cpu(), params, generator)
        if params.stop_at_eos and next_id == tokenizer.eos_token_id:
            return
        idx = torch.cat([idx, torch.tensor([[next_id]], device=lm.device)], dim=1)

        text = tokenizer.decode(idx[0, prompt_len:].tolist())
        if text.endswith("�"):   # incomplete UTF-8 sequence; wait for the next token
            yield next_id, ""
            continue
        new_text, emitted = text[len(emitted):], text
        yield next_id, new_text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, help="checkpoint path (default: newest best.pt)")
    p.add_argument("--prompt", default="")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int)
    p.add_argument("--no-stop-at-eos", action="store_true")
    args = p.parse_args()

    path = args.ckpt
    if path is None:
        ckpts = find_checkpoints()
        if not ckpts:
            sys.exit(f"No checkpoints found in {CKPT_DIR}. Train a model first (scripts/train.py).")
        path = ckpts[0]
    lm = load_model(path)
    tokenizer = load_tokenizer()
    print(f"Loaded {path}\n{lm.describe()}\n", file=sys.stderr)

    params = SamplingParams(args.temperature, args.top_k, args.top_p, args.max_new_tokens,
                            not args.no_stop_at_eos, args.seed)
    print(args.prompt, end="", flush=True)
    n, t0 = 0, time.time()
    for _, text in stream_generate(lm, tokenizer, args.prompt, params):
        print(text, end="", flush=True)
        n += 1
    print(f"\n\n[{n} tokens, {n / (time.time() - t0):.1f} tokens/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
