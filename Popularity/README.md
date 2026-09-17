# Popularity — baseline / fallback for "Complete the Look"

Not a trained model — a candidate item's membership in its category's
top-10 / top-100 most frequent training targets, all-time (no time
window). No query read at all: same list for every query in a category.

## Run

```
# after TTN/build_data.py has run at least once
python Popularity/build_popularity.py
```

Reads `data/tower/pairs_train.parquet`, writes
`data/tower/popularity_top100.json` (per-category top-10/top-100 item lists).

## Its role

TTN's advantage over popularity isn't uniform — below roughly 25 training
observations of the target item, TTN performs at or below random chance
(see `results.ipynb`), so this is the safer default for those items rather
than trusting TTN's output on something it hasn't really learned.

## A gap, not yet resolved

This is **all-time** popularity, not the 60-day window mentioned for a
separate "popular in category" carousel — no time-windowed variant exists
in this repo yet.
