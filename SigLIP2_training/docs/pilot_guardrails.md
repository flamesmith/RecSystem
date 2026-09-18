# 5K pilot performance guardrails

## Comparison hierarchy

The primary comparison is always apples-to-apples: each trained adapter versus the untouched pretrained SigLIP2 embeddings on the same 5K sample, the same leakage-safe validation rows, and the same text representation.

Earlier notebook experiments are historical context only. Their candidate pools, product domains, training sizes, and evaluation sizes differ. A raw Recall@K value from a 64-row furniture evaluation cannot be used as a promotion threshold for a roughly 500-row all-Home-and-Kitchen evaluation.

## Metrics

| Metric | Purpose |
|---|---|
| Bidirectional Recall@1 | Exact paired retrieval at the hardest cutoff, averaged across image-to-text and text-to-image. |
| Bidirectional Recall@10 | More tolerant paired retrieval, averaged across both directions. This is the primary early-stopping metric. |
| Positive-pair margin | Mean paired cosine minus mean off-diagonal cosine. It detects embedding collapse. |
| Same-leaf precision@10 | Taxonomy proxy after excluding the paired item. It asks whether neighboring products remain in the same product type. |
| Broad-category slice Recall@10 | Detects gains that hide a material regression in a supported Home & Kitchen category. |
| Feature success fraction | Share of selected products that survive download, validation, and embedding extraction. |

## Promotion rules after 5K

- No more than a 2 percentage-point regression in bidirectional Recall@10 versus the matched pretrained baseline.
- No more than a 3 percentage-point regression in bidirectional Recall@1.
- All stored floating-point embeddings must be finite.
- At least 97% of selected records must produce features.
- Positive-pair margin must remain above zero.
- No broad category with at least 25 validation queries may regress by more than 5 percentage points at Recall@10.
- Validation chooses the strategy and whether to deploy its adapter. The test split is opened only once after that decision.

Passing these rules permits the next experiment; it does not prove production quality.

## Reconsideration rule at 50K

Stop automatic scaling if the 50K run loses more than 2 percentage points of Recall@10 or 3 percentage points of same-leaf precision@10 relative to the selected 5K configuration. The comparison must reuse the exact fixed 5K validation and test products; new training rows that duplicate either fixed split are excluded. Review sampling, text aggregation, false negatives, adapter capacity, and the frozen-tower assumption before continuing.

## Historical reference points

- General Home & Kitchen notebook: 64 train / 64 evaluation rows; trained bidirectional Recall@10 was approximately 0.820.
- Furniture playground: 64 train / 64 evaluation rows; trained bidirectional Recall@10 was approximately 0.773.
- Furniture text-strategy benchmark: 256 train / 128 evaluation rows; multi-chunk native bidirectional Recall@10 was approximately 0.801 and ranked first overall.

These values are intentionally not encoded as hard gates because the smaller search spaces make them easier and the furniture-only domain is narrower.
