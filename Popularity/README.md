# Popularity — Trending in Category

Not a trained model, and not a fallback for TTN's complement recommendations
— it's an independent carousel: "most popular desks," when viewing a desk,
not "most popular things bought alongside a desk." Raw purchase-proxy count
(one review = one purchase), grouped by the item's **own** category
(`cat_2`/`cat_3`/`cat_4` together — the most specific path this project's
taxonomy has). No query read beyond "what category is this item in": every
item in a category gets the same list.

## Run

```
python Popularity/build_popularity.py
```

No `--snapshot` — unlike TTN and SigLIP2, this doesn't depend on any
`--window-days` snapshot at all. An earlier version did (it restricted
counts to `data/tower/<snapshot_id>/item_asins.npy`, TTN's
co-purchase-pair-derived item list), but that was an unjustified
restriction inherited from reusing TTN's infrastructure, not something
Popularity's own logic needs — a raw purchase count has no concept of
"co-purchase." Categories now come straight from `df_features.pkl`'s own
`cat_2`/`cat_3`/`cat_4` columns, covering every item with valid features
(1,134,566), not just the smaller subset that also happened to clear
TTN's co-purchase and category-licensing filters (134,749 for the `w60`
snapshot) — concretely, this restores roughly 25,000 items that were
previously excluded despite having real review activity.

Reads `data/df_features.pkl` and `Home_and_Kitchen_filtered.csv`; writes
`Popularity/generated/recommendations/popularity.parquet` — one universal
file, not one per snapshot, since nothing left is snapshot-specific. One
row per `cat_2`/`cat_3`/`cat_4`/`variant`/`rank` (not per-asin JSON), with
two variants, both anchored at `TTN/constants.json`'s `date_threshold`:

- `all_time` — every review before the cutoff, no decay.
- `recency` — only the `RECENCY_WINDOW_DAYS` (default 60, set in the script)
  immediately before it.

`RECENCY_WINDOW_DAYS` here is unrelated to `data_creation/complementary_cats_pairs`'
`window_days` (the co-purchase *pairing* window used to build TTN's
training pairs) — same word, two different concepts. See the script's
docstring.

## Consuming this output

`prepare_serving.py` expands these category-level rows into per-item rows
by joining `(cat_2, cat_3, cat_4)` against `df_features.pkl` for every item
in whichever snapshot it's serving — not against `node_of_item.npy`, which
Popularity itself no longer reads.
