# SigLIP2 full-catalog decision log

## 2026-09-18 — Isolated project boundary

All new full-catalog work lives under `SigLIP2_training/`. Existing exploratory notebooks remain unchanged.

## 2026-09-18 — Local-first architecture

The machine has 16 GiB RAM, an AMD Radeon Pro 5300M with 4 GiB VRAM, and a working PyTorch Metal device outside the workspace sandbox. Training will run natively on macOS rather than inside Docker. Docker packages CPU preprocessing only.

What this optimizes: reliable access to `mps:0` while preserving a portable production preprocessing artifact.

## 2026-09-18 — Cache budget

The training cache is capped at 8 GiB and evicts to 6 GiB. Validation images have a separate 2 GiB allowance, active downloads have a 1 GiB allowance, and downloading pauses below 30 GiB free disk. The machine had approximately 101 GiB free when this decision was recorded.

What this optimizes: enough working reuse and prefetch capacity without threatening macOS swap or model artifacts.

## 2026-09-18 — Image URL choice

For each product, select the `MAIN` image and prefer the `large` URL over `hi_res`, with `thumb` as the final fallback. SigLIP2 currently trains at 224px, so high-resolution originals add bandwidth and decode pressure without proportional training value.

What this optimizes: download throughput and predictable cache occupancy.

## 2026-09-18 — Frozen-feature-first training

The default full-data pass streams each image through the frozen vision backbone once and stores float16 pre-head features. Description strategy and projection-head experiments reuse those features. Backbone unfreezing is reserved for a finalist experiment.

What this optimizes: larger true contrastive batches, reusable hard-negative experiments, and tractable training on 4 GiB GPU memory.

## 2026-09-18 — Versioned deterministic descriptions

Description behavior is defined by `configs/description_v1.yaml`, reusable package code, a JSON schema, and golden unit tests. Each record carries a semantic pipeline version, configuration hash, and source hash.

What this optimizes: identical product ingestion across all four phases and future production inference.

## 2026-09-18 — Exact tokenizer chunking instead of blind truncation

The canonical description remains the single-view baseline, but long descriptions can also be represented by up to four exact-tokenizer chunks of 64 tokens with 12-token overlap. When more than four chunks are possible, selection is spread across the document rather than keeping only the beginning.

What this optimizes: retention of useful details near the end of a listing while respecting SigLIP2's native text length and avoiding positional-embedding surgery.

## 2026-09-18 — Conservative duplicate removal

Exact normalized sentence duplicates are removed. Near duplicates are considered only after inexpensive token-overlap and length gates, then confirmed with a character sequence comparison. Borderline text is preserved.

What this optimizes: removal of repeated marketing copy without silently erasing different dimensions, materials, or product claims. The gates also reduced a 1,000-record metadata pass from roughly 29.5 seconds to 1.25 seconds.

## 2026-09-18 — Validation before catalog-scale work

The initial gate used 1,000 real metadata rows, 100 exact-tokenized rows, and five real image downloads. All nine deterministic unit tests passed. Every observed chunk stayed within 64 tokens; the five-image cache test produced four downloads, one cache hit, and no failures.

What this optimizes: catching schema, tokenizer, URL-selection, image-validation, and eviction defects before spending hours on full-catalog extraction.
