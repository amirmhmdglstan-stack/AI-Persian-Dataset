# Persian Dataset V2 — Model / Data Balance

**Status:** planning report. The full V2 corpus has **not** been built or tokenized on the agent machine. The only measured V2 numbers here come from the small synthetic integration test; full-corpus expectations are explicitly labelled **estimates**.

## 1. Measured small sample (not full corpus)

From `reports/small_sample_statistics.json` / `dataset_statistics.json` after the actual small-sample integration run:

| Metric | Measured value |
|---|---:|
| Synthetic fixture documents generated | 54 |
| Documents in final files after cleaning/dedup | 30 |
| Unique documents | 30 |
| Characters | 6,681 |
| Words | 1,208 |
| Persian letters | 5,104 |
| Latin letters | 27 |
| ZWNJ | 142 |
| **Content tokens, unchanged V1 32K tokenizer** | **1,886** |
| Tokens including one `<|endoftext|>` per document | 1,916 |
| Characters per token | 3.5424 |
| Bytes per token | 6.6925 |
| Train / validation / test documents | 28 / 1 / 1 |
| Exact duplicates removed | 1 |
| Near duplicates removed | 6 |
| Verification checks | 27 run, 26 passed, 1 warning, 0 failed |

The warning is `split_ratios_reasonable`: a 30-document sample cannot reproduce a 97/2/1 ratio once cluster-consistent minimum representation is enforced. This is expected and is **not** a full-run failure.

A separate tokenizer comparison used the 1.7 MB V1 validation split (728 documents / 935,924 characters), still a small sample: baseline 32K achieved 3.9643 characters/token and 100% exact round-trip; from-scratch 8K achieved 3.6354; from-scratch 16K and the requested 32K both reached only 15,958 actual vocab entries because the sample is too small for a true 32K vocabulary. See `reports/tokenizer_comparison.json`.

## 2. Full-corpus expectations (ESTIMATES)

| Scenario | Estimated V2 tokens | Basis |
|---|---:|---|
| Balanced strict PROFILE A | **≈36–50 M** | non-Wikipedia verified sources are small; Wikipedia capped near 50% |
| PROFILE A upper bound | **≈100–150 M** | would reintroduce ~70–85% Wikipedia dominance |
| PROFILE B potential | 100–150 M+ | speculative; requires NC/privacy-cleared sources |
| V1 baseline | 11.35 M measured | immutable baseline |

These are **not measured V2 tokens**. `compute_statistics.py` on the user's PC will replace them with exact numbers. The user's stated 100–150 M-token goal is feasible only if we accept either a Wikipedia-heavy corpus or carefully licensed new modern Persian prose; corpus size alone does not guarantee good Persian generation.

## 3. The 20-tokens-per-parameter reference

The ~20 training tokens per parameter ratio is a scaling reference, not a law. Small models often benefit from more tokens than the ratio suggests, while a model trained for many epochs on a small corpus overfits. We use `T_required ≈ 20 × P`. Dataset coverage is `T_dataset / T_required`; estimated epochs to reach the reference budget are `T_required / T_dataset`.

### A. Assuming a 100–150 M-token V2 corpus (upper-bound A or A+B)

| Model | Parameters P | Reference tokens 20×P | Dataset coverage 100–150 M | Estimated epochs | Overfitting risk | GTX 1060 6 GB feasibility |
|---|---:|---:|---:|---:|---|---|
| Small GPT (e.g. d=512, 6 layers, 8 heads, seq 1024) | ~30 M | 600 M | 0.17–0.25 | 4–6 | moderate–high; early stopping and dedup essential | plausible with fp16, small batch, gradient checkpointing |
| Balanced GPT (e.g. d=768, 12 layers, 12 heads) | ~100 M | 2 B | 0.05–0.075 | 13–20 | high; repeated epochs likely | marginal / generally not practical at seq 1024 without offload |
| Larger experimental GPT (e.g. d=1024, 24 layers) | ~250 M | 5 B | 0.02–0.03 | 33–50 | very high; memorization likely | not feasible on 6 GB |

### B. Assuming a realistic balanced strict-A corpus of 36–50 M tokens

| Model | Reference tokens | Dataset coverage | Estimated epochs | Risk |
|---|---:|---:|---:|---|
| ~30 M | 600 M | 0.06–0.083 | 12–17 | high overfitting; still the most sensible small experiment |
| ~100 M | 2 B | 0.018–0.025 | 40–56 | very high; not recommended |
| ~250 M | 5 B | 0.007–0.010 | 100–139 | effectively unusable without much more data |

**Conclusion:** with the currently verified corpus, a **small ~30 M-parameter model** is the only reasonably matched experiment, and even it needs careful regularization, early stopping and a held-out test split. A 100 M-parameter model should wait for a larger, genuinely diverse Persian corpus. Do not choose the architecture merely because the GTX 1060 has 6 GB; choose it from measured tokens, then check memory.

## 4. Vocabulary parameter cost

`compare_tokenizers.py` measured tokenizer JSON sizes and embedding costs on the V1 validation sample. Values below are for **one tied/shared embedding matrix**; untied input+output embeddings double the parameter and memory cost.

| Vocab | Tokenizer JSON (compact) | d=512 params / fp32 | d=768 params / fp32 | d=1024 params / fp32 |
|---:|---:|---:|---:|---:|
| 8,000 | 0.382 MB | 4.10 M / 15.6 MB | 6.14 M / 23.4 MB | 8.19 M / 31.3 MB |
| 15,958 (from-scratch "16K/32K" sample result) | 0.834 MB | 8.17 M / 31.2 MB | 12.26 M / 46.8 MB | 16.34 M / 62.3 MB |
| 32,000 V1 baseline | 1.748 MB compact (3.1 MB shipped file) | 16.38 M / 62.5 MB | 24.58 M / 93.8 MB | 32.77 M / 125.0 MB |

For a ~30 M-parameter small model with d=512, a 32K vocabulary embedding is **16.4 M parameters**, over half the model if embeddings are tied and even more dominant if untied. An 8K or 16K vocabulary could free capacity, but measured small-sample fragmentation rises: mean tokens per Persian word was 1.51 for the 32K baseline, 1.73 for the 16K sample model and 1.92 for the 8K sample model. Fewer embedding parameters therefore buy a longer token sequence and more compute per word.

**Tokenizer policy:** keep the independently verified V1 32K tokenizer as the baseline for all V2 statistics. Do **not** replace it based on a 0.94 M-character sample. Rerun `compare_tokenizers.py` on the user's final `profile_A/train.txt` (the Windows CMD uses a 40 M-character cap) and only then consider 8K/16K alternatives. `<|endoftext|>` must remain ID 0 and a single token.

## 5. Training-token accounting

`compute_statistics.py` reports both:

* `tokens_content_excluding_document_end`: exact tokens from `Tokenizer.encode(document, add_special_tokens=False)`;
* `tokens_including_document_end`: content tokens + one `<|endoftext|>` per document.

The separator line is not tokenized as ordinary text. If the training loader appends the special token per document, use the second count. Mixing the two conventions would change a 100 M-token corpus by roughly one token per document; V1's 36,592 documents added 36,592 separator tokens to 11,350,801 content tokens.

## 6. Recommendation

1. Finalize and **measure** the dataset before fixing the architecture.
2. Start with a small ~30 M-parameter GPT, 32K baseline tokenizer, 97/2/1 splits and strict early stopping on `validation.txt`; reserve `test.txt` for the final report only.
3. If measured strict-A tokens are below ~50 M, prioritize **rights-cleared modern Persian prose, educational text and dialogue** over a larger model.
4. Reconsider 8K/16K vocabularies only after a full-corpus tokenizer comparison shows the embedding savings outweigh extra tokens.
5. Do not begin model training until `verify_dataset.py` exits zero and the manual audit samples are reviewed.
