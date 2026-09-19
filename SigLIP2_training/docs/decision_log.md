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

## 2026-09-18 — Representative deterministic 5K pilot

The pilot takes the 5,000 lowest seeded product-ID hashes from all eligible records rather than the first rows in the source file. The complete 3,735,584-row metadata file is scanned once, while only 5,000 raw candidates remain in memory. Taxonomy-balanced sampling is reserved for a controlled Phase 2 comparison rather than silently changing the production distribution in the first benchmark.

What this optimizes: repeatability and representative catalog coverage without source-order bias.

## 2026-09-18 — Leakage-safe split groups

Products connected by an exact normalized description or shared image URL receive one duplicate group ID and one split assignment. The resulting pilot contains 4,000 train, 512 validation, and 488 test records. Validation selects the strategy; test remains unopened until selection is complete.

What this optimizes: honest retrieval metrics that are not inflated by duplicate listings crossing split boundaries.

## 2026-09-18 — Identity-initialized post-embedding adapters

Each modality receives a small residual bottleneck adapter initialized to produce the unchanged pretrained embedding. Training uses SigLIP's pairwise sigmoid loss, true batches of 256 cached pairs, group-safe batching, validation early stopping, and the checkpoint's learned logit scale and bias.

What this optimizes: measurable learning with large negative sets while retaining a no-regression path back to the pretrained representation.

## 2026-09-18 — Explicit scaling guardrails

The 5K gate compares every candidate against its matched pretrained baseline. Recall regression limits, finite-feature checks, positive-pair margin, download success, supported-category slices, and taxonomy-neighbor metrics are recorded. Notebook metrics are retained as historical context, not direct gates, because their domains and candidate-set sizes differ.

What this optimizes: stopping architectural drift early instead of discovering retrieval deterioration after a 50K or full-catalog run.

## 2026-09-18 — 5K gate passed with multi-chunk adapter

The selected multi-chunk adapter improved sealed-test bidirectional Recall@10 from 58.50% to 70.90% and Recall@1 from 27.46% to 41.19%. The paired 95% bootstrap intervals for both gains remained entirely above zero. Same-leaf precision@10 also improved from 7.69% to 9.88%. See `docs/pilot_5k_results.md` for the complete comparison.

What this optimizes: scale to 50K only after demonstrating measurable retrieval improvement, numerical stability, category-slice health, and a no-regression fallback.

## 2026-09-18 — Fixed-evaluation 50K expansion

The 50K hash sample reuses the exact 512 validation and 488 test product IDs from the 5K pilot. All additional eligible products are training-only. Duplicate grouping across the expanded sample excluded 266 records whose descriptions or image URLs would connect training to fixed evaluation, including five former 5K training products.

What this optimizes: a true data-scaling comparison where metric changes reflect additional training data rather than an easier or harder evaluation draw.

## 2026-09-18 — Eight concurrent downloads for 50K

The 5K run used four concurrent downloads. After measuring the new-image portion of the 50K prefetch, concurrency was increased to eight while retaining the same atomic writes, image validation, 8 GiB cache ceiling, and 30 GiB free-disk floor.

What this optimizes: wall-clock download time without changing image content, model inputs, or storage safety.

## 2026-09-18 — Memory-safe pooled 50K feature extraction

The final 50K extractor uses 32-image MPS micro-batches inside 128-product record batches, 128-text batches, full-precision model computation, and float16 stored vectors. Repeated taxonomy strings are embedded once per run. The unused canonical-only view is omitted. A 64-image configuration exceeded the MPS memory ceiling, while half-precision model computation was slower on this Intel-era Mac.

What this optimizes: safe local memory use and fewer redundant text-model calls without changing the selected comparison strategies.

## 2026-09-18 — 50K scaling gate passed with taxonomy-blended multi-chunk

The selected 50K multi-chunk adapter with a 20% taxonomy blend improved fixed-test bidirectional Recall@10 from 70.90% for the selected 5K checkpoint to 81.56%. The paired 95% interval for the difference was +7.58 to +13.93 percentage points. Recall@1 improved from 41.19% to 47.95%, and same-leaf precision@10 improved from 9.88% to 11.11%. The pre-agreed deterioration thresholds were not triggered. See `docs/pilot_50k_results.md` for the complete audit.

What this optimizes: advance to negative-sampling and taxonomy-weight experiments only after additional training data demonstrated a statistically supported gain on fixed held-out products.
