# 200K Home & Kitchen scaling results

Run date: 2026-09-22

## Decision

Pass the 200K scaling gate and use `taxonomy_w20_random_200k.pt` as the initialization checkpoint for the final catalog phase. The selected representation remains multi-chunk text with a 20% taxonomy blend and random, duplicate-safe batches.

The 200K model improved Recall@1 and taxonomy-neighbor quality over the 50K incumbent. Recall@10 was 0.92 percentage points lower on the fixed test set, but the paired 95% interval crossed zero and the change remained within the predefined two-point scaling tolerance.

## Data and execution

| Item | Result |
|---|---:|
| Full metadata rows scanned | 3,735,584 |
| Eligible image/text rows | 3,735,423 |
| Nominal deterministic sample | 200,000 |
| Rows after fixed-evaluation leakage exclusions | 198,459 |
| Successfully embedded rows | 198,130 (99.83%) |
| Train / validation / test after image failures | 197,144 / 498 / 488 |
| Unique leaf categories before image failures | 1,607 |
| Feature extraction | 7,295.24 seconds on MPS |
| Adapter training and evaluation | 528.58 seconds on MPS |

The fixed validation and test products are the same products used for the 5K and 50K phases. A total of 1,541 sampled rows were excluded because they connected to fixed evaluation products through duplicate text or image URLs.

## Optimization sweep preceding 200K

Before scaling, three controlled 50K challengers tested a lower taxonomy weight and taxonomy-aware hard batches. None passed every incumbent guardrail:

| Candidate | Validation R@1 | Validation R@10 | Same-leaf P@10 | Result |
|---|---:|---:|---:|---|
| 10% taxonomy, random batches | 48.80% | 80.02% | 10.20% | Failed R@10 guardrail |
| 20% taxonomy, random batches | 49.40% | 81.53% | 10.81% | Failed one supported-category slice |
| 10% taxonomy, taxonomy-hard batches | 48.39% | 79.92% | 10.56% | Failed R@10 and slice guardrails |

The correct outcome of that sweep was to retain the existing 50K incumbent. This is why the 200K run changed scale but did not introduce a new negative-sampling method.

## Fixed held-out test

| Metric | Pretrained representation | Selected 200K adapter | 50K incumbent |
|---|---:|---:|---:|
| Bidirectional Recall@1 | 26.23% | **49.49%** | 47.95% |
| Bidirectional Recall@5 | 47.75% | **73.98%** | 73.87% |
| Bidirectional Recall@10 | 57.89% | **80.64%** | 81.56% |
| Positive-pair cosine margin | 0.0368 | **0.0801** | 0.1041 |
| Same-leaf precision@1 | 17.32% | **28.87%** | 26.56% |
| Same-leaf precision@10 | 7.67% | **12.19%** | 11.11% |
| Same-leaf hit@10 | 40.42% | **48.04%** | 47.11% |

Relative to the matched pretrained representation, the 200K adapter improved bidirectional Recall@1 by 23.26 points and Recall@10 by 22.75 points. Both paired 95% bootstrap intervals were entirely above zero.

Relative to the 50K incumbent:

- Recall@1 changed by **+1.54 points**, with a 95% interval of -1.33 to +4.30.
- Recall@10 changed by **-0.92 points**, with a 95% interval of -3.18 to +1.23.
- Same-leaf precision@10 changed from 11.11% to **12.19%**.

## Checkpoint contract

The promoted file is `checkpoints/pilot_200k/taxonomy_w20_random_200k.pt`. It is approximately 1.5 MB because it stores the learned image adapter, text adapter, logit parameters, training configuration, and feature-index metadata. It is not a duplicate of the frozen SigLIP2 base model.

Inference requires both:

1. `google/siglip2-base-patch16-224`, using the versioned preprocessing pipeline; and
2. the promoted adapter checkpoint.

The checkpoint is sufficient to reproduce the learned mapping and generate embeddings for unseen products when combined with those frozen base-model and preprocessing dependencies.
