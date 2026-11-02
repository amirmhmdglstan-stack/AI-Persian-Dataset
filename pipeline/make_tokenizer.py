#!/usr/bin/env python3
"""
make_tokenizer.py — train a Persian BPE tokenizer on the training split and
count tokens exactly with it.

The tokenizer is a by-product: the dataset itself is plain UTF-8 text. Having a
real tokenizer turns "estimated tokens" into a measured number, and gives the
user a ready-to-use vocabulary for a ~100M parameter GPT.

Usage:
  python3 pipeline/make_tokenizer.py --train train.txt --validation validation.txt \
      --out-dir tokenizer --vocab 32000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SEP = "<|endoftext|>"


def iter_corpus(path: str, max_bytes: int | None = None):
    """Yield documents (not raw lines) for tokenizer training."""
    buf = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n") == SEP:
                doc = "".join(buf).strip()
                if doc:
                    yield doc
                seen += len(doc)
                buf = []
                if max_bytes and seen >= max_bytes:
                    return
            else:
                buf.append(line)
    if buf:
        yield "".join(buf)


def train(train_path: str, vocab_size: int, sample_chars: int) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[SEP],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    t0 = time.time()
    tok.train_from_iterator(iter_corpus(train_path, sample_chars), trainer=trainer)
    print(f"[tokenizer] trained in {time.time() - t0:.0f}s, vocab={tok.get_vocab_size()}")
    return tok


def count_tokens(tok: Tokenizer, path: str, batch_docs: int = 500) -> dict:
    """Exact token/document/char counts for one file."""
    ids = tok.encode_batch(list(iter_corpus_n(path, batch_docs)), add_special_tokens=False)
    tokens = sum(len(e.ids) for e in ids)
    # characters counted independently of the tokenizer
    chars = 0
    docs = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.rstrip("\n") == SEP:
                docs += 1
            chars += len(line) - 1
    return {
        "path": os.path.basename(path),
        "documents": docs,
        "characters": chars,
        "tokens": tokens,
        "characters_per_token": round(chars / max(tokens, 1), 3),
        "bytes_per_token": round(os.path.getsize(path) / max(tokens, 1), 3),
    }


def iter_corpus_n(path: str, batch_docs: int):
    batch = []
    for doc in iter_corpus(path):
        batch.append(doc)
        if len(batch) >= batch_docs:
            yield from batch
            batch = []
    if batch:
        yield from batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--validation", default="validation.txt")
    ap.add_argument("--out-dir", default="tokenizer")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--sample-chars", type=int, default=40_000_000,
                    help="characters of training text used to fit the BPE model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = train(args.train, args.vocab, args.sample_chars)
    tok_path = os.path.join(args.out_dir, f"persian-bpe-{args.vocab // 1000}k.json")
    tok.save(tok_path)
    vocab_path = os.path.join(args.out_dir, f"persian-bpe-{args.vocab // 1000}k-vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for i, (piece, idx) in enumerate(sorted(tok.get_vocab().items(), key=lambda kv: kv[1])):
            f.write(f"{idx}\t{piece}\n")

    print("[tokenizer] counting tokens (this encodes the whole corpus) ...")
    tr = count_tokens(tok, args.train)
    va = count_tokens(tok, args.validation)
    stats = {
        "tokenizer": {
            "type": "byte-level BPE",
            "vocab_size": tok.get_vocab_size(),
            "special_tokens": [SEP],
            "file": tok_path,
            "trained_on": f"{args.sample_chars:,} characters sampled from train.txt",
            "note": "shipped as a convenience; the dataset is plain text and can be "
                    "tokenized with any Persian vocabulary",
        },
        "train": tr,
        "validation": va,
        "total": {
            "tokens": tr["tokens"] + va["tokens"],
            "characters": tr["characters"] + va["characters"],
            "characters_per_token": round(
                (tr["characters"] + va["characters"]) / max(tr["tokens"] + va["tokens"], 1), 3),
        },
        "method": "exact: encode_batch() over every document, no sampling",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = os.path.join(args.out_dir, "token_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== tokens (measured with the shipped tokenizer) ===")
    print(f"train      {tr['tokens']:>12,} tokens  ({tr['characters_per_token']} chars/token)")
    print(f"validation {va['tokens']:>12,} tokens")
    print(f"total      {stats['total']['tokens']:>12,} tokens")
    print(f"wrote {tok_path}, {vocab_path}, {out}")


if __name__ == "__main__":
    sys.exit(main())
