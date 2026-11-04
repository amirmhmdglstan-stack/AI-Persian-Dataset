#!/usr/bin/env python3
"""Compare 8K/16K/32K Persian BPE vocabularies trained FROM SCRATCH.

The unchanged V1 32K byte-level BPE remains the baseline. Alternative
vocabularies are fitted on a hard-capped Persian sample using the ``tokenizers``
library only; no pretrained tokenizer, embedding or model weights are used.

Memory policy (audit fix):
* evaluation streams the capped sample once per tokenizer and aggregates scalar
  statistics; it does NOT retain lists of texts, token IDs, words or an
  unbounded frequency map. Token frequency uses a fixed array bounded by the
  vocabulary size;
* BPE training input is hard-capped at 10M characters (default 8M). A 40M-char
  request is refused rather than described as bounded-memory;
* process peak RSS is measured and recorded. ``tokenizers.BpeTrainer`` may
  internally materialise counts, so this is a bounded INPUT and a measured
  process peak, NOT a strict O(1) memory guarantee or a full-corpus measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from array import array
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DOC_SEPARATOR, config_tokenizer, load_config, now_utc, peak_rss_mb,
    sha256_file,
)
from compute_statistics import iter_documents  # noqa: E402

RE_PERSIAN_WORD = re.compile(r"[آ-یءأإئؤۀٔ]")
RE_WORD = re.compile(r"\S+")
FALLBACK_MAX_TRAIN_CHARS = 10_000_000
FALLBACK_DEFAULT_MAX_CHARS = 8_000_000


def has_document_separator(path: Path) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        return any(line.rstrip("\n") == DOC_SEPARATOR for line in f)


def iter_sample_texts(path: Path, max_chars: int) -> Iterator[str]:
    """Yield a bounded sample; reopen the file for each consumer, never a list."""
    chars = 0
    if has_document_separator(path):
        for text in iter_documents(path):
            text = (text or "").strip()
            if not text:
                continue
            if chars + len(text) > max_chars:
                break
            chars += len(text)
            yield text
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                if chars + len(text) > max_chars:
                    break
                chars += len(text)
                yield text


def sample_stats(path: Path, max_chars: int) -> tuple[int, int, int]:
    docs = chars = byte_count = 0
    for text in iter_sample_texts(path, max_chars):
        docs += 1
        chars += len(text)
        byte_count += len(text.encode("utf-8"))
    return docs, chars, byte_count


def train_bpe(path: Path, max_chars: int, vocab_size: int, doc_count: int) -> Any:
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ImportError as e:
        raise RuntimeError(
            "Tokenizer comparison needs the 'tokenizers' package. Install with: "
            "D:\\GPT\\venv\\Scripts\\python.exe -m pip install -r dataset_v2\\requirements.txt") from e
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
    # The iterator is fresh and bounded. We do not retain the sample texts.
    tok.train_from_iterator(iter_sample_texts(path, max_chars), trainer=trainer,
                            length=doc_count)
    return tok


def evaluate(tok, path: Path, max_chars: int, label: str, vocab_requested: int,
             dims: list[int], trained_on_chars: int) -> dict:
    """Stream documents and aggregate scalars; no retained ID/word lists."""
    vocab_size = tok.get_vocab_size()
    # Bounded by the vocabulary, not by the corpus: at most ~32K counters.
    freq = array("Q", [0]) * (vocab_size + 1)
    total_tokens = chars = byte_count = docs = 0
    roundtrip_ok = 0
    out_of_vocab_ids = 0
    failures: list[dict] = []
    persian_words = word_token_sum = split_words = 0
    for text in iter_sample_texts(path, max_chars):
        docs += 1
        chars += len(text)
        byte_count += len(text.encode("utf-8"))
        ids = tok.encode(text, add_special_tokens=False).ids
        total_tokens += len(ids)
        for token_id in ids:
            if 0 <= token_id < len(freq):
                freq[token_id] += 1
            else:
                out_of_vocab_ids += 1
        decoded = tok.decode(ids)
        if decoded == text:
            roundtrip_ok += 1
        elif len(failures) < 3:
            failures.append({"original": text[:120], "decoded": decoded[:120]})
        for match in RE_WORD.finditer(text):
            word = match.group()
            if RE_PERSIAN_WORD.search(word):
                n = len(tok.encode(word, add_special_tokens=False).ids)
                persian_words += 1
                word_token_sum += n
                if n > 1:
                    split_words += 1
    distinct_types = sum(1 for c in freq if c)
    rare_types = sum(1 for c in freq if 0 < c <= 2)
    rare_occurrences = sum(c for c in freq if 0 < c <= 2)
    special_id = tok.token_to_id(DOC_SEPARATOR)
    special_ids = tok.encode(DOC_SEPARATOR, add_special_tokens=False).ids
    json_bytes = len(tok.to_str().encode("utf-8"))
    return {
        "label": label,
        "requested_vocab_size": vocab_requested,
        "actual_vocab_size": vocab_size,
        "trained_from_scratch_on_chars": trained_on_chars,
        "special_token": DOC_SEPARATOR,
        "special_token_id": special_id,
        "special_token_is_single_token": special_ids == [0],
        "evaluation_documents": docs,
        "evaluation_characters": chars,
        "evaluation_utf8_bytes": byte_count,
        "token_count": total_tokens,
        "exact_roundtrip_accuracy": round(roundtrip_ok / max(1, docs), 6),
        "roundtrip_failures": docs - roundtrip_ok,
        "roundtrip_failure_examples": failures,
        "characters_per_token": round(chars / max(1, total_tokens), 4),
        "bytes_per_token": round(byte_count / max(1, total_tokens), 4),
        "tokens_per_1000_characters": round(1000 * total_tokens / max(1, chars), 2),
        "evaluation_memory_strategy": (
            "streaming documents; scalar aggregation; fixed frequency array bounded by "
            "vocabulary size; no retained token-ID, word or unbounded frequency lists"),
        "token_fragmentation": {
            "persian_words_measured": persian_words,
            "mean_tokens_per_persian_word": round(
                word_token_sum / max(1, persian_words), 4),
            "persian_words_split_into_multiple_tokens": split_words,
            "persian_word_split_ratio": round(split_words / max(1, persian_words), 4),
            "distinct_token_types_used": distinct_types,
            "rare_token_types_frequency_le_2": rare_types,
            "rare_token_occurrences_frequency_le_2": rare_occurrences,
            "rare_occurrence_share": round(rare_occurrences / max(1, total_tokens), 4),
            "out_of_vocabulary_id_occurrences": out_of_vocab_ids,
            "rare_note": ("Rare is relative to this capped evaluation sample. Frequency "
                          "counts use a fixed array of vocabulary size, not a retained corpus list."),
        },
        "vocabulary_memory_cost": {
            "tokenizer_json_bytes": json_bytes,
            "tokenizer_json_mb": round(json_bytes / 1024 ** 2, 3),
            "embedding_parameters_by_model_dimension": {
                str(d): {
                    "parameters": vocab_size * d,
                    "float32_megabytes": round(vocab_size * d * 4 / 1024 ** 2, 2),
                    "float16_megabytes": round(vocab_size * d * 2 / 1024 ** 2, 2),
                    "note": "One shared/tied embedding matrix; untied input+output embeddings double this cost.",
                } for d in dims
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare Persian BPE vocabularies on a memory-bounded capped sample")
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", required=True, help="Persian text sample; V2 train.txt or a small fixture")
    ap.add_argument("--tokenizer-baseline", default=None)
    ap.add_argument("--vocab-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=None,
                    help="bounded sample cap; default 8M, hard maximum 10M characters")
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
    tokenizer_cfg = cfg.get("tokenizer", {})
    default_max_chars = int(tokenizer_cfg.get("bounded_memory_default_sample_chars",
                                              FALLBACK_DEFAULT_MAX_CHARS))
    hard_max_chars = int(tokenizer_cfg.get("bounded_memory_max_sample_chars",
                                           FALLBACK_MAX_TRAIN_CHARS))
    max_chars = args.max_chars if args.max_chars is not None else default_max_chars
    if max_chars <= 0 or max_chars > hard_max_chars:
        raise ValueError(
            f"--max-chars {max_chars:,} exceeds the bounded-memory BPE training cap "
            f"{hard_max_chars:,}. A 40M-character request is NOT bounded-memory on this path; "
            "use <=10M characters and measure peak RSS instead.")
    if not tok_path.exists():
        raise FileNotFoundError(f"Baseline tokenizer not found: {tok_path}")
    if not sample.exists():
        raise FileNotFoundError(f"Sample text not found: {sample}")
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError("The 'tokenizers' package is required; install the core requirements.") from e

    doc_count, chars, byte_count = sample_stats(sample, max_chars)
    if not doc_count:
        raise RuntimeError(f"No usable Persian sample text found in {sample}")
    vocab_sizes = args.vocab_sizes or tokenizer_cfg.get("comparison_vocab_sizes",
                                                        [8000, 16000, 32000])
    dims = tokenizer_cfg.get("model_embedding_dimensions", [512, 768, 1024])
    baseline = Tokenizer.from_file(str(tok_path))
    results = [evaluate(baseline, sample, max_chars, "V1_baseline_32k", 32000, dims, 0)]
    for v in vocab_sizes:
        label = ("from_scratch_32k" if v == 32000
                 else f"from_scratch_{v // 1000}k")
        tok = train_bpe(sample, max_chars, v, doc_count)
        results.append(evaluate(tok, sample, max_chars, label, v, dims, chars))

    measured_peak = peak_rss_mb()
    report = {
        "measurement_scope": args.scope,
        "generated_at_utc": now_utc(),
        "sample": {
            "path": str(sample), "sha256": sha256_file(sample),
            "documents_used": doc_count, "characters_used": chars,
            "utf8_bytes_used": byte_count, "max_chars_cap": max_chars,
            "hard_max_chars_cap": hard_max_chars,
            "note": ("Capped streaming sample; on the agent machine this is a SMALL SAMPLE, "
                     "not the full V2 corpus. A full-run launcher uses at most 8M characters."),
        },
        "baseline_tokenizer": {
            "path": str(tok_path), "sha256": sha256_file(tok_path),
            "vocab_size": baseline.get_vocab_size(),
            "special_token_id": baseline.token_to_id(DOC_SEPARATOR),
            "policy": "The existing V1 32K tokenizer remains the baseline and is NOT automatically replaced.",
        },
        "alternatives_note": ("8K/16K/32K alternatives were trained from scratch on the capped sample "
                              "with tokenizers.BpeTrainer. No pretrained tokenizer or model weights were imported."),
        "results": results,
        "memory": {
            "evaluation_strategy": ("Streaming aggregation; no retained lists of sample texts, token IDs, "
                                    "Persian words or unbounded token frequencies. Frequencies use a "
                                    "fixed array bounded by vocabulary size."),
            "bpe_training_input_cap_chars": hard_max_chars,
            "default_sample_chars": default_max_chars,
            "sample_characters_used": chars,
            "sample_utf8_bytes_used": byte_count,
            "process_peak_rss_mb": measured_peak,
            "process_peak_rss_note": ("Measured with resource.getrusage on Linux / GetProcessMemoryInfo on "
                                      "Windows when available. This is process-wide peak RSS for this capped "
                                      "run, NOT a full-corpus memory measurement."),
            "bpe_training_note": ("tokenizers.BpeTrainer may internally materialise counts. Input is "
                                  "hard-capped at 10M chars; no strict O(1)-memory claim is made and a "
                                  "40M-char sample is refused as unbounded for this path."),
        },
        "recommendation": (
            "Keep the verified 32K baseline for dataset statistics. This capped sample is too small to choose "
            "a final vocabulary; rerun compare_tokenizers.py on the user's final V2 train.txt with the same "
            "bounded cap and review measured peak RSS. If a small model's embedding matrix dominates "
            "parameters, 8K or 16K may be worth testing, at the cost of more tokens per Persian word."),
        "limitations": [
            "Round-trip accuracy is exact string equality on this capped sample, not semantic quality.",
            "Rare-token frequency is sample-relative and vocabulary-bounded.",
            "Full-corpus token counts are measured separately by compute_statistics.py.",
            "Full-corpus memory, full-corpus extraction and Windows 11 / Python 3.12 execution were not measured here.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({
        "scope": args.scope,
        "sample_documents": doc_count,
        "sample_characters": chars,
        "sample_cap_characters": max_chars,
        "process_peak_rss_mb": measured_peak,
        "vocabularies": [{"label": r["label"], "actual_vocab_size": r["actual_vocab_size"],
                          "chars_per_token": r["characters_per_token"],
                          "roundtrip": r["exact_roundtrip_accuracy"]} for r in results],
        "output": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
