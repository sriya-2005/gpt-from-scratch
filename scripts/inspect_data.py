"""Show sample tokenizer output and a batch from the data loaders to verify the pipeline.

Usage:
    python scripts/inspect_data.py [dataset]    # dataset folder under data/processed, default wikipedia
"""
import sys
import textwrap

from transformers import GPT2TokenizerFast

from dataloader import PROCESSED_DIR, create_dataloaders, load_meta

BATCH_SIZE = 8
BLOCK_SIZE = 256


def show(label: str, text: str) -> None:
    print(f"{label}:")
    print(textwrap.indent(textwrap.fill(repr(text), 100), "    "))


def main() -> None:
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    data_dir = PROCESSED_DIR / (sys.argv[1] if len(sys.argv) > 1 else "wikipedia")
    meta = load_meta(data_dir)

    print("=== Dataset stats ===")
    print(f"source: {meta['source']}  tokenizer: {meta['tokenizer']} (vocab {meta['vocab_size']})")
    for s in ("train", "val"):
        st = meta[s]
        print(f"  {s:5s}: {st['articles']:>9,} articles  {st['bytes'] / 1e6:>7.1f} MB  {st['tokens']:>12,} tokens"
              f"  ({st['bytes'] / st['tokens']:.2f} bytes/token)")
    total = meta["train"]["tokens"] + meta["val"]["tokens"]
    print(f"  split by tokens: {meta['train']['tokens'] / total:.1%} train / {meta['val']['tokens'] / total:.1%} val")

    print("\n=== Tokenizer round trip ===")
    sample = "The quick brown fox jumps over the lazy dog. Tokenization isn't magic: it's byte-level BPE!"
    ids = tok.encode(sample)
    show("text", sample)
    print(f"token ids ({len(ids)}): {ids}")
    print(f"tokens: {tok.convert_ids_to_tokens(ids)}")
    show("decoded", tok.decode(ids))
    print(f"round trip exact: {tok.decode(ids) == sample}")

    train_loader, val_loader = create_dataloaders(BATCH_SIZE, BLOCK_SIZE, data_dir)
    print("\n=== Data loaders ===")
    print(f"block_size={BLOCK_SIZE}, batch_size={BATCH_SIZE}")
    print(f"train: {len(train_loader.dataset):,} blocks -> {len(train_loader):,} batches/epoch")
    print(f"val:   {len(val_loader.dataset):,} blocks -> {len(val_loader):,} batches")

    x, y = next(iter(train_loader))
    print(f"\nbatch x: shape={tuple(x.shape)} dtype={x.dtype}   y: shape={tuple(y.shape)}")
    print(f"x[0][:12] = {x[0, :12].tolist()}")
    print(f"y[0][:12] = {y[0, :12].tolist()}   (x shifted by one)")
    print(f"targets correctly shifted: {bool((x[:, 1:] == y[:, :-1]).all())}")
    print(f"max token id in batch: {x.max().item()} (< {meta['vocab_size']})")

    print("\n=== Decoded samples ===")
    show("train sample x[0] (first 80 tokens)", tok.decode(x[0, :80]))
    vx, _ = next(iter(val_loader))
    show("val sample x[0] (first 80 tokens)", tok.decode(vx[0, :80]))


if __name__ == "__main__":
    main()
