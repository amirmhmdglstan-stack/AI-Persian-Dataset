#!/usr/bin/env python3
"""Compare 8K/16K/32K Persian BPE vocabularies trained FROM SCRATCH.

The unchanged V1 32K byte-level BPE remains the baseline.  Alternative
vocabularies are fitted on a capped Persian sample using the `tokenizers`
library only; no pretrained tokenizer, embedding, or model weights are used.
This script is intentionally suitable for small-sample profiling; a final
vocabulary decision must be rerun on the user's full V2 corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOC_SEPARATOR, config_tokenizer, load_config, now_utc, sha256_file,
)
from compute_statistics import iter_documents  # noqa: E402

RE_PERSIAN_WORD = re.compile(r"[آ-یءأإئؤۀٔ]")
RE_WORD = re.compile(r"\S+")


def load_sample(path: Path, max_chars: int) -> tuple[list[str], int, int]:
    texts: list[str] = []
    chars = 0
    with open(path, "r", encoding="utf-8") as f:
        has_sep = any(line.rstrip("\n") == DOC_SEPARATOR for line in f)
    docs = iter_documents(path) if has_sep else (
        line.strip() for line in open(path, "r", encoding="utf-8"))
    for text in docs:
        text = (text or "").strip()
        if not text:
            continue
        if chars + len(text) > max_chars:
            break
        texts.append(text)
        chars += len(text)
    return texts, chars, sum(len(t.encode("utf-8")) for t in texts)


def train_bpe(texts: list[str], vocab_size: int) -> Any:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ImportError as e:
        raise RuntimeError(
            "Tokenizer comparison needs the 'tokenizers' package. Install with: "
            "D:\\GPT\\venv\\Scripts\\python.exe -m pip install tokenizers") from e
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[DOC_SEPARATOR],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train_from_iterator(iter(texts), trainer=trainer)
    return tok


def evaluate(tok, texts: list[str], label: str, vocab_requested: int,
             dims: list[int], trained_on_chars: int) -> dict:
    ids_list = [tok.encode(t, add_special_tokens=False).ids for t in texts]
    total_tokens = sum(len(ids) for ids in ids_list)
    chars = sum(len(t) for t in texts)
    byte_count = sum(len(t.encode("utf-8")) for t in texts)
    roundtrip_ok = 0
    failures = []
    for text, ids in zip(texts, ids_list):
        decoded = tok.decode(ids)
        if decoded == text:
            roundtrip_ok += 1
        elif len(failures) < 3:
            failures.append({"original": text[:120], "decoded": decoded[:120]})
    persian_words = [w for t in texts for w in RE_WORD.findall(t) if RE_PERSIAN_WORD.search(w)]
    word_token_counts = [len(tok.encode(w, add_special_tokens=False).ids) for w in persian_words]
    split_words = sum(1 for n in word_token_counts if n > 1)
    freq = Counter(i for ids in ids_list for i in ids)
    rare_types = sum(1 for c in freq.values() if c <= 2)
    rare_occurrences = sum(c for c in freq.values() if c <= 2)
    special_id = tok.token_to_id(DOC_SEPARATOR)
    special_ids = tok.encode(DOC_SEPARATOR, add_special_tokens=False).ids
    json_bytes = len(tok.to_str().encode("utf-8"))
    return {
        "label": label,
        "requested_vocab_size": vocab_requested,
        "actual_vocab_size": tok.get_vocab_size(),
        "trained_from_scratch_on_chars": trained_on_chars,
        "special_token": DOC_SEPARATOR,
        "special_token_id": special_id,
        "special_token_is_single_token": special_ids == [0],
        "evaluation_documents": len(texts),
        "evaluation_characters": chars,
        "evaluation_utf8_bytes": byte_count,
        "token_count": total_tokens,
        "exact_roundtrip_accuracy": round(roundtrip_ok / max(1, len(texts)), 6),
        "roundtrip_failures": len(texts) - roundtrip_ok,
        "roundtrip_failure_examples": failures,
        "characters_per_token": round(chars / max(1, total_tokens), 4),
        "bytes_per_token": round(byte_count / max(1, total_tokens), 4),
        "tokens_per_1000_characters": round(1000 * total_tokens / max(1, chars), 2),
        "token_fragmentation": {
            "persian_words_measured": len(persian_words),
            "mean_tokens_per_persian_word": round(
                sum(word_token_counts) / max(1, len(word_token_counts)), 4),
            "persian_words_split_into_multiple_tokens": split_words,
            "persian_word_split_ratio": round(split_words / max(1, len(persian_words)), 4),
            "distinct_token_types_used": len(freq),
            "rare_token_types_frequency_le_2": rare_types,
            "rare_token_occurrences_frequency_le_2": rare_occurrences,
            "rare_occurrence_share": round(rare_occurrences / max(1, total_tokens), 4),
            "rare_note": "Rare is relative to this capped evaluation sample, not the full corpus.",
        },
        "vocabulary_memory_cost": {
            "tokenizer_json_bytes": json_bytes,
            "tokenizer_json_mb": round(json_bytes / 1024 ** 2, 3),
            "embedding_parameters_by_model_dimension": {
                str(d): {
                    "parameters": tok.get_vocab_size() * d,
                    "float32_megabytes": round(tok.get_vocab_size() * d * 4 / 1024 ** 2, 2),
                    "float16_megabytes": round(tok.get_vocab_size() * d * 2 / 1024 ** 2, 2),
                    "note": "One shared/tied embedding matrix; untied input+output embeddings double this cost.",
                } for d in dims
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare 8K/16K/32K Persian BPE vocabularies on a capped sample")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", required=True, help="Persian text sample; V2 train.txt or a small fixture")
    ap.add_argument("--tokenizer-baseline", default=None)
    ap.add_argument("--vocab-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=2_000_000)
    ap.add_argument("--reports-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--scope", default="small_sample", choices=["small_sample", "full_corpus_sample"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok_path = Path(args.tokenizer_baseline) if args.tokenizer_baseline else config_tokenizer(cfg)
    sample = Path(args.sample)
    reports_dir = Path(args.reports_dir) if args.reports_dir else (
        Path(cfg["outdir"]) / "reports")
    out = Path(args.out) if args.out else reports_dir / "tokenizer_comparison.json"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not tok_path.exists():
        raise FileNotFoundError(f"Baseline tokenizer not found: {tok_path}")
    if not sample.exists():
        raise FileNotFoundError(f"Sample text not found: {sample}")
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError("The 'tokenizers' package is required; install it with pip.") from e

    texts, chars, byte_count = load_sample(sample, args.max_chars)
    if not texts:
        raise RuntimeError(f"No usable Persian sample text found in {sample}")
    vocab_sizes = args.vocab_sizes or cfg.get("tokenizer", {}).get("comparison_vocab_sizes", [8000, 16000, 32000])
    dims = cfg.get("tokenizer", {}).get("model_embedding_dimensions", [512, 768, 1024])
    baseline = Tokenizer.from_file(str(tok_path))
    results = [evaluate(baseline, texts, "V1_baseline_32k", 32000, dims, 0)]
    for v in vocab_sizes:
        if v == 32000:
            # A from-scratch 32K is still trained for comparison; the baseline stays separate.
            label = "from_scratch_32k"
        else:
            label = f"from_scratch_{v // 1000}k"
        tok = train_bpe(texts, v)
        results.append(evaluate(tok, texts, label, v, dims, chars))

    report = {
        "measurement_scope": args.scope,
        "generated_at_utc": now_utc(),
        "sample": {
            "path": str(sample), "sha256": sha256_file(sample),
            "documents_used": len(texts), "characters_used": chars,
            "utf8_bytes_used": byte_count, "max_chars_cap": args.max_chars,
            "note": "On the agent machine this is a SMALL SAMPLE (V1 validation split or fixture), not the full V2 corpus.",
        },
        "baseline_tokenizer": {
            "path": str(tok_path), "sha256": sha256_file(tok_path),
            "vocab_size": baseline.get_vocab_size(),
            "special_token_id": baseline.token_to_id(DOC_SEPARATOR),
            "policy": "The existing V1 32K tokenizer remains the baseline and is NOT automatically replaced.",
        },
        "alternatives_note": "8K/16K/32K alternatives were trained from scratch on the sample with tokenizers.BpeTrainer. No pretrained tokenizer or model weights were imported.",
        "results": results,
        "recommendation": (
            "Keep the verified 32K baseline for dataset statistics. This capped sample is too small to choose a final vocabulary; "
            "rerun compare_tokenizers.py on the user's final V2 train.txt. If a small model's embedding matrix dominates parameters, "
            "8K or 16K may be worth testing, at the cost of more tokens per Persian word."
        ),
        "limitations": [
            "Round-trip accuracy is exact string equality on this sample, not semantic quality.",
            "Rare-token frequency is sample-relative.",
            "Full-corpus token counts are measured separately by compute_statistics.py.",
            "Python 3.12 / Windows execution was not performed on the agent machine.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({
        "scope": args.scope,
        "sample_documents": len(texts),
        "sample_characters": chars,
        "vocabularies": [{"label": r["label"], "actual_vocab_size": r["actual_vocab_size"],
                          "chars_per_token": r["characters_per_token"],
                          "roundtrip": r["exact_roundtrip_accuracy"]} for r in results],
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
