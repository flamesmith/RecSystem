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

The default full-data pass streams each image through the complete frozen pretrained towers once and stores normalized 768-dimensional float16 embeddings. Description-strategy and small post-embedding adapter experiments reuse those features. The native SigLIP vision pooling head consumes all 196 patch tokens; caching its inputs would cost roughly 300 KB per product, or hundreds of gigabytes at catalog scale. Caching a 768-dimensional float16 vector costs about 1.5 KB per view. Backbone or native-head unfreezing is therefore reserved for a finalist experiment that streams images again.

What this optimizes: larger true contrastive batches, reusable hard-negative experiments, and tractable storage and training on 4 GiB GPU memory.

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

## 2026-09-18 — Explicit slow image processor

The feature configuration pins `use_fast_image_processor: false` and `use_fast_tokenizer: true` independently. The locally cached checkpoint was created with the slow image processor, and Transformers warns that a future default will switch to the fast implementation with slightly different outputs. Keeping the cached fast tokenizer avoids adding SentencePiece solely because a shared `use_fast` flag was passed to both components.

What this optimizes: stable preprocessing across runs and library-default changes. A fast processor can be benchmarked later as an explicit versioned experiment.
