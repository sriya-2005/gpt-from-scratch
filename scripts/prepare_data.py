"""Download a text dataset, tokenize it with the GPT-2 tokenizer, and write
train/val token files for language-model training.

Datasets:
    wikipedia - one shard of English Wikipedia (~0.5-1 GB of text), split 80/20 by article
    verdict   - "The Verdict" by Edith Wharton (~20 KB, 5,145 tokens), split 80/20 by position:
                the first 80% of the story is training data, the last 20% validation

Output (in data/processed/<dataset>/):
    train.bin, val.bin  - flat arrays of uint16 GPT-2 token ids; each document ends with <|endoftext|>
    meta.json           - tokenizer info and dataset statistics

Usage:
    python scripts/prepare_data.py --dataset verdict
    python scripts/prepare_data.py --dataset wikipedia
    python scripts/prepare_data.py --dataset wikipedia --max-mb 200   # smaller subset
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import GPT2TokenizerFast

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

WIKI_REPO = "wikimedia/wikipedia"
WIKI_SHARD = "20231101.en/train-{:05d}-of-00041.parquet"
VERDICT_REPO = "jyothibindu/the-verdict"
VERDICT_FILE = "the-verdict.txt"

MIN_CHARS = 200  # skip stubs / near-empty Wikipedia articles


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def load_tokenizer() -> GPT2TokenizerFast:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # byte-level BPE, 50257-token vocab
    tokenizer.model_max_length = int(1e12)  # we chunk later; silence the >1024 length warning
    assert tokenizer.vocab_size < 2**16  # ids fit in uint16
    return tokenizer


def write_meta(out_dir: Path, source: str, tokenizer, args, stats: dict) -> None:
    meta = {
        "source": source,
        "tokenizer": "gpt2",
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "dtype": "uint16",
        "seed": args.seed,
        "val_frac": args.val_frac,
        **stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    for s in ("train", "val"):
        st = stats[s]
        print(f"  {s:5s}: {st['articles']:>9,} documents  {st['bytes'] / 1e6:>9.3f} MB text  "
              f"{st['tokens']:>12,} tokens  -> {out_dir / (s + '.bin')}")
    total_tokens = stats["train"]["tokens"] + stats["val"]["tokens"]
    print(f"  val share of tokens: {stats['val']['tokens'] / total_tokens:.1%}")


def prepare_wikipedia(args, out_dir: Path) -> None:
    print(f"Downloading {WIKI_REPO} shard {args.shard} ...")
    path = hf_hub_download(WIKI_REPO, WIKI_SHARD.format(args.shard), repo_type="dataset", local_dir=RAW_DIR)
    pf = pq.ParquetFile(path)
    n_articles = pf.metadata.num_rows
    print(f"  {path} ({Path(path).stat().st_size / 1e6:.0f} MB, {n_articles:,} articles)")

    tokenizer = load_tokenizer()
    eos = tokenizer.eos_token_id

    # Split by article (not by token) so no article appears in both sets
    rng = np.random.default_rng(args.seed)
    is_val = rng.random(n_articles) < args.val_frac

    max_bytes = args.max_mb * 1e6
    stats = {s: {"articles": 0, "tokens": 0, "bytes": 0} for s in ("train", "val")}
    files = {s: open(out_dir / f"{s}.bin", "wb") for s in ("train", "val")}
    total_bytes = 0
    row = 0
    start = time.time()

    pbar = tqdm(total=round(args.max_mb), unit="MB", desc="Tokenizing")
    try:
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            texts = batch.column("text").to_pylist()
            splits = is_val[row:row + len(texts)]
            row += len(texts)

            keep = [(clean(t), "val" if v else "train") for t, v in zip(texts, splits)]
            keep = [(t, s) for t, s in keep if len(t) >= MIN_CHARS]
            if not keep:
                continue

            ids_list = tokenizer([t for t, _ in keep])["input_ids"]
            for (text, split), ids in zip(keep, ids_list):
                ids.append(eos)
                np.asarray(ids, dtype=np.uint16).tofile(files[split])
                n_bytes = len(text.encode("utf-8"))
                stats[split]["articles"] += 1
                stats[split]["tokens"] += len(ids)
                stats[split]["bytes"] += n_bytes
                total_bytes += n_bytes

            pbar.n = min(round(total_bytes / 1e6), pbar.total)
            pbar.refresh()
            if total_bytes >= max_bytes:
                break
    finally:
        pbar.close()
        for f in files.values():
            f.close()

    print(f"\nDone in {time.time() - start:.0f}s")
    write_meta(out_dir, f"{WIKI_REPO} {WIKI_SHARD.format(args.shard)}", tokenizer, args, stats)


def prepare_verdict(args, out_dir: Path, full: bool = False) -> None:
    print(f"Downloading {VERDICT_REPO}/{VERDICT_FILE} ...")
    path = hf_hub_download(VERDICT_REPO, VERDICT_FILE, repo_type="dataset", local_dir=RAW_DIR / "verdict")
    text = clean(Path(path).read_text(encoding="utf-8"))
    print(f"  {path} ({len(text):,} characters)")

    tokenizer = load_tokenizer()
    ids = tokenizer.encode(text) + [tokenizer.eos_token_id]

    # A single story can't be split by document; split by position instead, so the
    # validation set is a contiguous passage the model never sees during training.
    split_at = int(len(ids) * (1 - args.val_frac))
    parts = {"train": ids[:split_at], "val": ids[split_at:]}
    if full:
        # verdict_full: train on the WHOLE story, so every fact a question might ask about is
        # in the training data. There is no unseen story text left, so val is the same text:
        # it measures how well the story has been learned, not generalization (held-out
        # questions take over that job later).
        parts = {"train": ids, "val": ids}
    stats = {}
    for name, part in parts.items():
        np.asarray(part, dtype=np.uint16).tofile(out_dir / f"{name}.bin")
        stats[name] = {"articles": 1, "tokens": len(part),
                       "bytes": len(tokenizer.decode(part).encode("utf-8"))}
    write_meta(out_dir, f"{VERDICT_REPO} {VERDICT_FILE}", tokenizer, args, stats)


def prepare_verdict_facts(args, out_dir: Path) -> None:
    """The whole story plus every rewritten fact statement from data/qa/fact_statements.jsonl.

    The statements are shuffled so the four versions of a fact are not next to each other:
    otherwise the model could copy a nearby sentence instead of having to remember the fact.
    Each statement is its own document, ending with <|endoftext|>. val.bin is the story alone,
    so its loss stays comparable with the verdict_full run.
    """
    import json
    import random

    story_path = hf_hub_download(VERDICT_REPO, VERDICT_FILE, repo_type="dataset", local_dir=RAW_DIR / "verdict")
    story = clean(Path(story_path).read_text(encoding="utf-8"))
    statements_file = ROOT / "data" / "qa" / "fact_statements.jsonl"
    statements = [json.loads(line)["statement"] for line in statements_file.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    random.Random(args.seed).shuffle(statements)

    tokenizer = load_tokenizer()
    eos = tokenizer.eos_token_id
    story_ids = tokenizer.encode(story) + [eos]
    fact_ids = [t for s in statements for t in tokenizer.encode(s) + [eos]]
    parts = {"train": story_ids + fact_ids, "val": story_ids}
    stats = {}
    for name, part in parts.items():
        np.asarray(part, dtype=np.uint16).tofile(out_dir / f"{name}.bin")
        stats[name] = {"articles": 1 + (len(statements) if name == "train" else 0), "tokens": len(part),
                       "bytes": len(tokenizer.decode(part).encode("utf-8"))}
    print(f"  story: {len(story_ids):,} tokens + {len(statements)} fact statements: {len(fact_ids):,} tokens")
    write_meta(out_dir, f"{VERDICT_REPO} {VERDICT_FILE} + {statements_file.name}", tokenizer, args, stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wikipedia", "verdict", "verdict_full", "verdict_facts"],
                        default="wikipedia")
    parser.add_argument("--shard", type=int, default=0, help="wikipedia: which of the 41 English shards to use")
    parser.add_argument("--max-mb", type=float, default=1000, help="wikipedia: stop after this many MB of text")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = PROCESSED_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "wikipedia":
        prepare_wikipedia(args, out_dir)
    elif args.dataset == "verdict_facts":
        prepare_verdict_facts(args, out_dir)
    else:
        prepare_verdict(args, out_dir, full=args.dataset == "verdict_full")


if __name__ == "__main__":
    main()
