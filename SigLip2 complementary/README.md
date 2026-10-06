# SigLip2 complementary

Complementary recommendations ranked by **image similarity only**. Same question
as TTN ("given this item, what goes with it?"), same complementary categories,
but inside each category the candidates are ordered by how visually similar their
picture is to the query's picture -- there is no trained two-tower model here.

For a coffee table: the licensed target categories (TV stands, bookcases, desks,
dining tables, ...) come from the same mapping TTN uses, and within each of them
the items closest in SigLIP2 image space to the table are recommended.

## Run

```
python "SigLip2 complementary/generate_complementary_recommendations.py"
```

From the repo root. The folder name contains a space, so quote the path. It is
run as a script, not imported as a package. About 90 seconds on this machine.

Writes `recommendations/complements_top25.parquet` (gitignored like every other
parquet in the repo).

Prerequisites, all produced elsewhere:
- `SigLIP2/image_cache/` -- raw SigLIP2 image embeddings, filled by
  `SigLIP2/encode_siglip2_images.py` and/or `SigLIP2/encode_images_by_category.py`
- `SigLIP2/champion_selection_siglip.json` -- which adapter to apply
  (currently `SigLIP2/checkpoints/pilot_200k/taxonomy_w20_random_200k.pt`)
- `data/complementary_categories.pkl` -- the complementary-category mapping
- `data/df_features.pkl` -- each item's `cat_2` / `cat_3` / `cat_4`

## How it works

1. **Embeddings.** Load every item with a real image embedding from the cache
   (~558K), apply the champion adapter to the raw vectors, L2-normalise. Scores
   are cosine similarities.
2. **Licensed categories.** `complementary_categories.pkl` lists directed
   `cat_2 > cat_3 > cat_4` pairs with an `edges` count (how much co-purchase
   evidence backs the pair). The query's **own category is dropped** as a target
   -- the mapping pairs a category with itself (391 of 5,845 pairs) and TTN drops
   those too. "Own category" means the identical `cat_2 > cat_3 > cat_4` path.
3. **Slot split.** The `TOP_K = 25` slots are split across the query's licensed
   categories in proportion to `edges`, by largest-remainder apportionment (the
   rule in `TTN/generate_recommendations_proportional.py`).
4. **Unfilled slots are redistributed.** A category with fewer image-embedded
   items than its share cannot use all its slots; the unused slots are re-split
   among the categories that still have room, by edge share, repeatedly, until 25
   are placed or every candidate is used. **TTN does not do this** -- it loses
   them, so its lists can be shorter than K. With no shortfall the two allocations
   are identical (tested on 2,000 random cases).
5. **Ranking.** Inside a category, candidates are ranked by cosine similarity to
   the query. Slots depend only on the query's category, so they are computed once
   per category.

## Output schema

`query_asin | rank | candidate_asin | score | target_category`

- `rank` runs 1..25 per query. Categories appear in **descending order of
  co-purchase evidence**, and inside a category by similarity, best first -- the
  same layout as `TTN/complements_proportional.parquet`. **`rank` is not one
  similarity ordering across categories.** Scores are comparable only within the
  same `target_category`.
- `target_category` is the candidate's `cat_2 > cat_3 > cat_4`, as text (TTN's
  file has a node id instead; this folder has no node vocabulary).

## What it covers

- **Queries:** an embedded item is a query when its own category is a source in
  the mapping and at least one licensed target has an embedded item. 552,352 of
  558,324 embedded items qualify.
- **Candidates:** only items with an image embedding. Items with no image (about
  half the catalogue) are never recommended and are never queries.
- Every list has the full 25 rows in the current run.
- Not snapshot-scoped: it uses the full cache, so it covers far more items than
  TTN (137K for w90).

## Checked

Recomputed independently for 150 random queries (1,383 query-by-category blocks)
from the raw cache vectors, with a fresh run of the adapter and the mapping file,
without using the script's own data structures:
- every candidate lies in a licensed, non-own target category of the query, is
  embedded, and is never the query itself
- each category's returned items are exactly the most similar in that category;
  stored scores match the recomputation to within 1.4e-6
- categories appear in descending-evidence order, and scores descend inside each

## Measured on the current output (top 10 per query)

| | this folder | TTN weighted |
| --- | --- | --- |
| distinct items ever recommended | 274,634 | 3,844 |
| mean Jaccard between same-category queries (random-list baseline) | 0.058 (0.023) | 0.64 (0.037) |
| share of slots held by the 100 most-recommended items | 4.8% | 31.7% |

These measure how much different queries' lists overlap and how concentrated the
recommendations are. They say nothing about whether the recommendations are
*right*; no accuracy evaluation (e.g. Recall@K on held-out co-purchases) has been
run for this folder.

## Things to be aware of

- With 25 slots and up to ~50 licensed categories, most categories get 1-2 slots,
  and only the strongest few get several.
- The adapter was trained for image-to-text matching; here it is used for
  image-to-image similarity.
- `Missing` as a `cat_4` pools many unrelated categories under one name; paths are
  still distinguished by `cat_2` and `cat_3`.
