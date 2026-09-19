# 50K Home & Kitchen scaling results

Run date: 2026-09-18

## Decision

**Pass the 50K scaling gate and proceed to the optimization phase.** The selected configuration is the trained multi-chunk adapter with a 20% taxonomy blend. It improved the fixed held-out test's bidirectional Recall@10 from 70.90% for the selected 5K model to 81.56%, with a paired 95% bootstrap interval of **+7.58 to +13.93 percentage points** for the difference.

The result does not justify unfreezing the SigLIP2 towers yet. The evidence supports continuing with frozen 768-dimensional SigLIP2 features and small post-embedding adapters while testing negative sampling, taxonomy weight, and sampling balance at the next scale.

## Data and execution

| Item | Result |
|---|---:|
| Full metadata rows scanned | 3,735,584 |
| Eligible image/text rows | 3,735,423 |
| Nominal deterministic sample | 50,000 |
| Rows after fixed-evaluation leakage exclusions | 49,734 |
| Train / validation / test before image failures | 48,734 / 512 / 488 |
| Successfully embedded rows | 49,649 (99.83%) |
| Train / validation / test after image failures | 48,663 / 498 / 488 |
| Reused embeddings from the 5K run | 4,975 |
| Newly excluded rows with unusable images | 85 |
| Feature extraction | 4,888.86 seconds on MPS |
| Guarded three-strategy training and evaluation | 547.99 seconds on MPS |

The fixed validation and test membership came from the 5K pilot. New rows were training-only, and 266 sampled rows were excluded because an exact normalized description or image URL connected them to the fixed evaluation set. After image failures, no duplicate group crossed a split.

The image prefetch resolved 49,623 unique URLs: 5,709 were already cached, 43,843 were newly downloaded, and 71 unique URLs failed. A failed URL can belong to more than one product, which is why feature extraction excluded 85 rows. The bounded cache occupied approximately 1.23 GiB, well below its 8 GiB ceiling.

## Local-compute choices

The 16 GiB Mac could not safely run 64-image or 256-text full-precision batches; both exceeded the MPS memory ceiling. Half-precision model inference was stable but slower on this hardware: 63.0 seconds versus 36.0 seconds for the same 128-row benchmark.

The final extractor therefore used:

- 32-image MPS micro-batches;
- 128-product pooled record batches;
- 128-text batches;
- full-precision model computation and float16 stored embeddings;
- cached embeddings for matching 5K products;
- one cached embedding per repeated taxonomy string; and
- no canonical-only embedding, because none of the three 50K candidates consumes that view.

Pooling changed stored text vectors by at most `3.1e-05` in the benchmark and left image and multi-chunk vectors bit-identical. The completed set contained 49,649 unique product IDs, finite embeddings, consistent shard schemas, and vector norms between 0.9997 and 1.0003.

What this optimized: safe memory use and lower wall-clock time without changing the comparison's text strategies or the fixed evaluation products.

## Fixed validation comparison

All values use the same 498 available validation products. The 5K reference is the previously selected trained multi-chunk checkpoint evaluated on this identical set.

| Candidate | Best epoch | Bidirectional R@1 | Bidirectional R@10 | Same-leaf P@10 | Scaling gate |
|---|---:|---:|---:|---:|---:|
| 5K multi-chunk reference | — | 42.27% | 71.29% | 9.93% | Reference |
| 50K raw first 64 | 13 | 27.51% | 58.84% | 5.21% | Fail |
| 50K multi-chunk | 8 | **49.20%** | 79.12% | **10.40%** | Pass |
| 50K multi-chunk + 20% taxonomy | 22 | 47.79% | **81.53%** | 9.89% | **Pass / selected** |

All three trained adapters passed their matched-pretrained regression, finite-feature, feature-success, positive-margin, and supported-slice checks. The raw-first candidate nevertheless failed the separate 5K scaling gate: its Recall@10 was 12.45 points lower and same-leaf precision@10 was 4.72 points lower than the 5K reference.

Both multi-chunk candidates passed the scaling gate. The taxonomy blend won selection because its validation Recall@10 was 2.41 points above plain multi-chunk. Plain multi-chunk retained higher Recall@1 and same-leaf precision@10, so the 20% taxonomy weight is a useful next-phase tuning parameter rather than a settled universal optimum.

## Sealed test result

The 488-product test split was opened only after the taxonomy-blended strategy won on validation.

| Metric | Matched pretrained view | Selected 50K adapter | Change |
|---|---:|---:|---:|
| Bidirectional Recall@1 | 26.23% | **47.95%** | **+21.72 pp** |
| Bidirectional Recall@5 | 47.75% | **73.87%** | **+26.13 pp** |
| Bidirectional Recall@10 | 57.89% | **81.56%** | **+23.67 pp** |
| Positive-pair cosine margin | 0.0368 | **0.1041** | +0.0672 |
| Same-leaf precision@1 | 17.32% | **26.56%** | +9.24 pp |
| Same-leaf precision@10 | 7.67% | **11.11%** | +3.44 pp |
| Same-leaf hit@10 | 40.42% | **47.11%** | +6.70 pp |

Paired bootstrap uncertainty versus the matched pretrained representation:

- Recall@1 gain: +21.72 percentage points; 95% interval **+18.13 to +25.41**.
- Recall@10 gain: +23.67 percentage points; 95% interval **+19.88 to +27.46**.

## Direct 5K-versus-50K scaling check

Because both checkpoints were evaluated on the same surviving 488 test products, this is the primary scaling comparison.

| Metric | Selected 5K adapter | Selected 50K adapter | Change |
|---|---:|---:|---:|
| Bidirectional Recall@1 | 41.19% | **47.95%** | **+6.76 pp** |
| Bidirectional Recall@5 | 64.24% | **73.87%** | +9.63 pp |
| Bidirectional Recall@10 | 70.90% | **81.56%** | **+10.66 pp** |
| Same-leaf precision@10 | 9.88% | **11.11%** | +1.22 pp |
| Positive-pair cosine margin | 0.0619 | **0.1041** | +0.0421 |

Paired bootstrap uncertainty for 50K minus 5K:

- Recall@1 difference: +6.76 percentage points; 95% interval **+3.59 to +10.04**.
- Recall@10 difference: +10.66 percentage points; 95% interval **+7.58 to +13.93**.

Both intervals are entirely above zero. The pre-agreed stop thresholds were a Recall@10 loss greater than 2 points or a same-leaf precision@10 loss greater than 3 points. Neither occurred; both point estimates improved. There is no evidence of architectural deterioration at 50K.

## Next-phase recommendation

Use the 50K taxonomy-blended adapter as the incumbent and retain plain multi-chunk as the principal challenger. At the optimization pilot, test a small predeclared taxonomy-weight grid, group-safe hard-negative sampling, and catalog-balance controls while preserving the same validation/test products and leakage exclusions.

Do not use the sealed test split for those choices. Select all optimization settings on validation, then run repeated-seed confirmation before any full-catalog pass. Human substitute/complement relevance evaluation remains necessary because exact-pair Recall@K and same-leaf precision do not by themselves establish recommendation quality.
