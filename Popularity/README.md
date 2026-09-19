# Popularity — Trending in Category

Not a trained model, and not a fallback for TTN's complement recommendations
— it's an independent carousel: "most popular desks," when viewing a desk,
not "most popular things bought alongside a desk." Raw purchase-proxy count
(one review = one purchase), grouped by the item's **own** category (`cat_4`
— the most specific level this project's taxonomy has). No query read
beyond "what category is this item in": every item in a category gets the
same list.

## Run

```
# after TTN/build_data.py --window-days 90 has run at least once
python Popularity/build_popularity.py --snapshot w90_2017-12-09
```

Reads that snapshot's `item_asins.npy`, `node_of_item.npy`, and
`Home_and_Kitchen_filtered.csv`; writes
`data/tower/<snapshot_id>/popularity_top100.json`, keyed by asin, with two
variants, both anchored at `TTN/constants.json`'s `date_threshold`:

- `all_time` — every review before the cutoff, no decay.
- `recency` — only the `RECENCY_WINDOW_DAYS` (default 60, set in the script)
  immediately before it.

`RECENCY_WINDOW_DAYS` here is unrelated to `complementary_cats_pairs`'
`window_days` (the co-purchase *pairing* window used to build TTN's
training pairs) — same word, two different concepts. See the script's
docstring.
