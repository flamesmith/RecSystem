# 5K Home & Kitchen pilot results

Run date: 2026-09-18

## Decision

**Pass the 5K gate and proceed to a controlled 50K experiment.** Use the trained multi-chunk 64-token adapter as the primary configuration. Keep multi-chunk plus taxonomy as a challenger for category-oriented retrieval, retain raw-first-64 as a baseline, and drop canonical-only text from the next primary comparison.

This decision does not authorize changing the frozen SigLIP2 towers. The 50K experiment should continue using cached pretrained embeddings and post-embedding adapters.

## Data and execution

| Item | Result |
|---|---:|
| Full metadata rows scanned | 3,735,584 |
| Eligible image/text rows | 3,735,423 |
| Deterministic pilot rows | 5,000 |
| Train / validation / test before image failures | 4,000 / 512 / 488 |
| Successfully embedded rows | 4,980 (99.6%) |
| Train / validation / test after image failures | 3,994 / 498 / 488 |
| Unique leaf categories in the selected sample | 854 |
| Products with real taxonomy among embedded rows | 4,473 |
| Feature extraction | 695.65 seconds on MPS |
| Stored feature size | 46.17 MiB |
| Tuned adapter comparison | 85.44 seconds on MPS |

Twenty rows failed because of seven unusable image URLs: five images were below the configured minimum dimensions and two URLs returned HTTP 404. One undersized image URL was shared by multiple products. These failures were retained in the failure log and did not cross the 97% success guardrail.

## Validation strategy comparison

All values below use the same 498 validation products. “Before” is the untouched pretrained representation for that text strategy; “after” is its best identity-initialized adapter checkpoint.

| Text strategy | Best epoch | R@1 before | R@1 after | R@10 before | R@10 after | Same-leaf P@10 before | Same-leaf P@10 after |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw first 64 | 20 | 15.16% | 20.88% | 31.63% | 44.28% | 5.55% | 5.57% |
| Canonical 64 | 49 | 3.21% | 4.92% | 8.23% | 26.81% | 1.41% | 6.58% |
| Multi-chunk 64 | 20 | 30.12% | **42.27%** | **59.24%** | **71.29%** | 9.04% | 9.93% |
| Multi-chunk + taxonomy | 21 | 29.82% | 41.67% | 57.13% | **71.29%** | 8.84% | **10.38%** |

Multi-chunk wins the exact-pair tie at Recall@10 through its higher Recall@1. Adding taxonomy gives the best taxonomy-neighbor precision but slightly reduces exact-pair retrieval. Canonical-only text loses too much product detail and should not remain a primary 50K candidate in its current form.

All four adapters passed the configured regression, numerical-health, feature-success, positive-margin, and supported-slice checks. Every supported broad-category slice improved for the selected multi-chunk adapter; its smallest validation Recall@10 slice gain was approximately 9.88 percentage points.

## Sealed test result

The 488-product test split was evaluated after multi-chunk was selected on validation.

| Metric | Pretrained multi-chunk | Trained adapter | Change |
|---|---:|---:|---:|
| Bidirectional Recall@1 | 27.46% | **41.19%** | **+13.73 pp** |
| Bidirectional Recall@5 | 48.87% | **64.24%** | **+15.37 pp** |
| Bidirectional Recall@10 | 58.50% | **70.90%** | **+12.40 pp** |
| Positive-pair cosine margin | 0.0437 | **0.0619** | +0.0183 |
| Same-leaf precision@1 | 18.01% | **22.63%** | +4.62 pp |
| Same-leaf precision@10 | 7.69% | **9.88%** | +2.19 pp |
| Same-leaf hit@10 | 40.18% | **42.73%** | +2.54 pp |

Paired bootstrap uncertainty over test queries:

- Recall@1 change: +13.73 percentage points; 95% interval **+10.96 to +16.60**.
- Recall@10 change: +12.40 percentage points; 95% interval **+9.63 to +15.37**.

Both intervals are entirely above zero. The adapter improvement is larger than the guardrail tolerance and is not consistent with sampling noise in this test set.

## Context from earlier notebooks

These figures are contextual only because the older runs use smaller candidate pools and, in two cases, furniture-only data.

| Experiment | Evaluation candidates | Bidirectional R@1 | Bidirectional R@10 | R@10 lift over random |
|---|---:|---:|---:|---:|
| Current 5K all-Home-and-Kitchen pilot | 488 | 41.19% | 70.90% | **34.60×** |
| General Home & Kitchen notebook | 64 | 61.72% | 82.03% | 5.25× |
| Furniture playground | 64 | 51.56% | 77.34% | 4.95× |
| Furniture multi-chunk benchmark | 128 | 38.67% | 80.08% | 10.25× |

The current raw Recall@K is lower than experiments with 64 or 128 candidates, which is expected with a 488-product and broader catalog. Relative to random exact-pair retrieval at the same candidate count, the current pilot has the strongest lift. This does not replace human relevance evaluation, but it provides no evidence of architectural deterioration at 5K.

## Fixed 50K comparison protocol

To make the proposed deterioration rule meaningful, 50K must be evaluated on the exact same 498 validation and 488 test products used here. New products may enlarge training, but any new row that duplicates a fixed evaluation description or image must be excluded from training. The 5K test set must remain sealed until the 50K configuration is selected.

Stop and review the architecture before further scaling if the 50K winner loses more than:

- 2 percentage points of bidirectional Recall@10 versus the selected 5K adapter, or
- 3 percentage points of same-leaf precision@10.

The review should examine false negatives, sampling balance, chunk aggregation, adapter capacity, and whether frozen towers have reached their limit before considering backbone unfreezing.
