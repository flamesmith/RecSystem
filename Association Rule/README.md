# Association Rule — "Bought together", steered by TTN's categories

Not a trained model. Item-level association rules (which items the same
reviewers bought together), but **which categories are allowed, and how many
of the 20 slots each gets, comes from TTN's category-pair table** rather than
from raw support alone.

## Run

```
python "Association Rule/build_association_rules.py" [--window-days 90]
```

No `--snapshot`: reads the raw review log directly (like Popularity). Reads
`data/Home_and_Kitchen_filtered.csv`, `data/df_features.pkl`,
`data/complementary_categories.pkl`, `data/category_taxonomy.json`,
`TTN/constants.json`. Writes
`Association Rule/generated/w{window_days}_{date_threshold}/recommendations/association_rules.parquet`.

## What is reused from TTN

- **Category filters** — `complementary_categories.pkl` is the category-pair
  table kept by `filter_pairs()`: `edges >= 5` (support floor, in whole
  co-purchases) **and** `lift >= 2.0`. A `(source category → target category)`
  row licenses that target for a query in that source category; direction
  matters. (TTN uses support + lift; there is no confidence filter anywhere.)
  Same-category rows (391 of them) are dropped, as in TTN's recommendation step.
- **Co-purchase weight** — that table's `edges`, split across a query's
  licensed target categories with TTN's `allocate_slots` (largest remainder).
- Same train cutoff (`date_threshold`, strictly before), same
  reviewer-within-`window_days` pairing rule, same cat_4 whitelist fold.

## What is new

- `pair_count(A, B)` = distinct reviewers who bought both A and B, the two
  purchases ≤ `window_days` apart, before the cutoff. `pairs.co_purchase_pairs`
  isn't reused because it keeps each pair once however many reviewers bought it
  (the distinct-pair set of the new counter was checked identical to it:
  10,595,885 pairs at 90d). `support = pair_count / 764,164` reviewers.
- A candidate must be licensed: (query's category → candidate's category) ∈ the
  filtered table; both items need a row in `df_features.pkl`.

## How the 20 slots are filled, per query item

1. Split 20 slots across **all** licensed target categories of the query's
   category by `edges` share. A category can't give more items than the query
   has licensed candidates in it.
2. Unfilled slots are redistributed over the categories that still have unused
   candidates, again by `edges` share, repeating until 20 are filled or nothing
   is left. No back-fill: a query with fewer than 20 licensed candidates gets
   fewer than 20 rows.
3. Inside a category, rank by `pair_count` descending. **Ties** (96% of pairs
   have count 1) are broken by candidate asin ascending — deterministic but
   arbitrary. Output order: categories by `edges` descending, then within.
   Counts are never compared across categories.

## Output columns

`query_asin, rank, candidate_asin, pair_count, support, target_cat_2,
target_cat_3, target_cat_4, category_edges`

## Result (w90_2017-12-09)

- 128,838 of 1,134,566 categorised items get ≥1 recommendation; the rest have no
  licensed co-purchased candidate in the train period.
- 23,708 (18.4%) get a full 20; mean 8.3 rows per query (item-pair evidence is sparse).
- 83,790 queries needed redistribution.
