# Text strategy screening recommendation

Selected working strategy: `multichunk64`.

It had the best mean rank across eight retrieval views. It led category-query image precision@10 (0.1611), category-query description precision@10 (0.1500), and cross-modal same-subcategory@1 (0.6800). Its mean natural exact-pair Recall@10 was 0.7812, tied with `compact64` for best.

This is a directional screening based on seed 42, 256 training pairs, 128 held-out exact-pair examples, and a 500-product retrieval catalog. Confirm on the full 2,000-product test catalog and multiple seeds before treating it as a final production choice.
