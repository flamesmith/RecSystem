# Complementary Cats & Pairs

Two views of "these get bought together", kept in one package because they
answer the same question from different evidence.

| Half | Evidence | Grain | Output |
| --- | --- | --- | --- |
| **categories** | Amazon's `also_buy` lists — what Amazon *says* is bought together | category pair | `data/complementary_pairs.pkl` |
| **pairs** | the review log — what users *actually* bought together | item pair | `data/co_purchase_pairs.pkl` |

They need not agree, and the disagreement is itself informative.

## Layout

| File | Role |
| --- | --- |
| `categories.py` | every function for the `also_buy` half; defines that pipeline |
| `category_analysis.ipynb` | builds and scores the category pairs, plots what each threshold costs, saves `data/pair_stats.pkl` |
| `categories.ipynb` | applies the chosen thresholds to that table and writes `data/complementary_pairs.pkl` |
| `pairs.py` | every function for the co-purchase half |
| `pairs.ipynb` | builds the item-item pairs and writes `data/co_purchase_pairs.pkl` |

All three notebooks are thin drivers: they define no logic, so edits to the
modules take effect with no notebook change. The one exception is the charts,
which live in the analysis notebook's own cells because they exist to be looked
at and adjusted rather than reused.

The two halves are independent — neither reads the other's output, and they can
be run in either order.

---

# The `also_buy` half — categories

Turns Amazon's `also_buy` lists — the asins shown as "frequently bought
together" — into a table of category pairs that get bought together, scored by
support and lift and cut down to the pairs strong enough to act on.

Output: `data/complementary_pairs.pkl`, one row per directed category pair.

**Run `category_analysis.ipynb` first.** It does the expensive work — the 2.7 GB
pickle and the 2.1 GB catalogue CSV, several minutes — once, and saves the full
scored table. `categories.ipynb` reads only that, so trying a different
threshold costs seconds.

## Inputs

- `data/df_features.pkl` — the **source** side of every edge, one row per
  `asin`. `run_feature_extraction` filters it to the categories that have a
  schema, so it holds only those items, but it carries the extracted features.
  **Must contain `also_buy`**, which only reaches the pickle if the extraction
  ran against a CSV rebuilt by `data/variable_selection.ipynb` after `also_buy`
  was added to `fields_to_keep`. `build_base_table` raises with that message if
  it is missing.
- `data/meta_Home_and_Kitchen_filtered.csv` — the **target** side lookup.
  `also_buy` points anywhere, including outside Home & Kitchen, and this CSV was
  never narrowed to the schema categories. Only `asin`, `category`, `imageURL`
  and `imageURLHighRes` are read.
- `data/category_taxonomy.json` — the reviewed whitelist of `(cat_3, cat_4)`
  pairs that are real categories rather than product bullets that leaked into
  the category path (`'Imported'`, `'10" high'`, and 550-odd others).

## Pipeline stages

```python
valid_pairs = load_taxonomy(TAXONOMY_PATH)
df_base     = build_base_table(df_features, valid_pairs)          # source side
cat_lookup  = load_catalogue_lookup(CATALOGUE_PATH, valid_pairs)  # target side
pairs       = build_pairs(df_base, cat_lookup)                    # one row per edge
pair_stats  = score_pairs(pairs)                                  # + support, lift
result      = filter_pairs(pair_stats, min_edges=5, min_lift=2.0) # the split
save_pairs(result, OUT_PATH)
```

`run_complementary_pairs` chains all seven in one call, for rebuilding without
either notebook.

A cat_4 value is kept only if it is valid **under its own parent**, since the
same label can be real in one branch and junk in another; everything else
becomes `<cat_3>_Other`. The same folding runs on both ends of an edge, so a
category is cleaned identically whether it turns up as a source or a target.

Edges whose target is in neither table are kept and marked `Not in catalogue`:
how much of `also_buy` leaves the catalogue is a finding, not something to drop
silently. They are excluded from the scoring, since a pair needs a category on
both ends.

## The two metrics

Both are computed over the resolved edges only, where `N` is their total:

- **support** = `edges(s -> d) / N` — the share of co-purchase traffic this pair
  accounts for. *Is there enough evidence here?*
- **lift** = `P(s, d) / (P(s) * P(d))`, where `P(s)` is the share of edges
  leaving that source path and `P(d)` the share arriving at that target path.
  `1.0` is exactly what independence would produce. *Is this an association, or
  two busy categories bumping into each other?*

Direction is kept: `A -> B` and `B -> A` are separate rows with their own
support and lift, since `also_buy` is listed per source item and the two
directions can carry very different traffic.

Neither threshold works alone. Support on its own promotes whatever is popular,
since a high-traffic category pairs with everything. Lift on its own does the
opposite: a single-edge pair between two otherwise-unseen paths scores `N`, the
maximum possible, on one co-purchase — which is what the far tail of the lift
distribution is made of. The support floor is what keeps a lift threshold
honest.

## Choosing the thresholds

The analysis notebook plots the exact survival function of each metric: for
every distinct value, how many pairs clear it. Read the two numbers off those
curves, then type them into `categories.ipynb`:

```python
MIN_EDGES = 5     # support floor, as a whole number of co-purchases
MIN_LIFT = 2.0
```

`MIN_EDGES` is a count rather than a fraction because it is the readable unit;
the equivalent support floor is `MIN_EDGES / n_edges`.

`threshold_grid` adds what the curves cannot show — whether the survivors are
spread across source categories or bunched on a few busy ones:

```python
from complementary_cats_pairs import threshold_grid

threshold_grid(pair_stats, edge_grid=(5, 10, 15, 25), lift_grid=(1.5, 2.0, 3.0))
```

It returns one row per combination with `pairs`, `edge_coverage`, and
`sources_ok` — the source paths keeping at least `min_degree` pairs. A source
that loses all of its pairs drops out and counts as not ok.

## Output schema

`data/complementary_pairs.pkl`, one row per surviving directed pair:

| Column | Meaning |
| --- | --- |
| `src_cat_2`, `src_cat_3`, `src_cat_4` | source category path (`cat_4` folded) |
| `dst_cat_2`, `dst_cat_3`, `dst_cat_4` | target category path (`cat_4` folded) |
| `edges` | co-purchase edges behind this pair |
| `src_edges`, `dst_edges` | the two marginals the lift is against |
| `support` | `edges / N` |
| `lift` | `P(s,d) / (P(s) P(d))` |

`N` is recoverable from an unfiltered table as `edges.sum()` — every resolved
edge lands in exactly one pair.

`save_pairs` casts the six category columns to `category` dtype before pickling.
A few hundred distinct strings repeat across thousands of rows, so the file
lands at roughly a quarter the size of the equivalent CSV while keeping the
dtypes across the round trip.

Read it back with `pd.read_pickle`. `data/*.pkl` is gitignored, so both outputs
stay local like every other generated table in this project.

## Notes

- Pairs whose source and target paths are identical are **kept** — a chair
  listed with another chair is a real `also_buy` edge. Filter them out
  downstream if the use needs strict complements.
- The category path is parsed here rather than with
  `feature_extraction_workflow.ensure_cat_columns`, which does the same thing:
  importing that package pulls in `nltk` at module load, and nothing here needs
  it.

---

# The co-purchase half — pairs

Every unordered pair of items the same user bought within a short window of
each other, read straight off the interaction log.

Output: `data/co_purchase_pairs.pkl`, one row per distinct item pair.

## Inputs

- `data/Home_and_Kitchen_filtered.csv` — the interaction log, one row per
  review. Only `reviewerID`, `asin` and `unixReviewTime` are read; the file is
  851 MB across 11 columns and the other eight are dead weight here.
- `ttn/constants.json` — `date_threshold`, the train/test split point, shared
  with `ttn/ttn.ipynb` §6. Tracked in git, unlike the tables in `data/`.

## Pipeline stages

```python
df_reviews  = load_interactions(INTERACTIONS_PATH)
cutoff_time = pd.Timestamp(DATE_THRESHOLD).timestamp()   # ttn/constants.json
pairs       = co_purchase_pairs(df_reviews, cutoff_time=cutoff_time,
                                window_days=90)
save_co_purchase_pairs(pairs, OUT_PATH)
```

`run_co_purchase_pairs` chains the three stages in one call, for rebuilding without the
notebook.

## The two filters

Applied in this order, and both matter:

- **`cutoff_time`** drops every interaction at or after the threshold *before*
  any pairing happens, so a table built for training cannot encode what happened
  during the evaluation window. Strictly-before, the same convention as
  `compute_cooccurrence_before_time` in `functions/rs_baseline_models.py`,
  `InteractionMatrixBuilder`, and the split in `ttn/ttn.ipynb` §6. One global
  cutoff, not a per-user one, so every training row precedes every test row.
  The date is not written in this package at all: `pairs.ipynb` reads
  `date_threshold` from **`ttn/constants.json`**, and `ttn/ttn.ipynb` §6 reads
  the same key for the model's train/test split. One value, in one tracked file,
  read by both notebooks — so the two splits are the same by construction rather
  than by convention, and changing the date is a one-line diff that both follow.
  `run_co_purchase_pairs` **requires** `cutoff_time`: there is no derived
  fallback, so no call site can end up splitting on a date nobody stated.
- **`window_days`** requires the two purchases to be within that many days *of
  each other*. A user who bought a blender in 2019 and a duvet in 2023 was not
  shopping for a set; without a window every item a long-lived account ever
  bought pairs with every other one. The test is on the gap, not on recency, so
  a tightly spaced 2014 pair still counts.

What each costs on the full log, at the 2017-12-09 cutoff:

| Filters | Pairs |
| --- | --- |
| none | 40,151,437 |
| cutoff alone | 34,559,421 |
| 365-day window alone | 22,320,760 |
| cutoff + 365-day window (`pairs.py` defaults) | 20,265,651 |
| 90-day window alone | 11,572,783 |
| **cutoff + 90-day window (what `pairs.ipynb` ships with)** | **10,595,885** |

`DEFAULT_WINDOW_DAYS` in `pairs.py` is still 365; `pairs.ipynb` overrides it to
90 in its settings cell, which is the value the saved pickle reflects.

## Output schema

`data/co_purchase_pairs.pkl`, one row per distinct pair:

| Column | Meaning |
| --- | --- |
| `asinA` | the alphabetically smaller asin of the pair |
| `asinB` | the other one |

Order is not meaningful, so the pair is stored with the smaller asin first:
`(A, B)` is present and `(B, A)` never is. A pair shared by fifty users still
appears once.

Both columns are `category` dtype, which is what keeps a 10M-row table near
116 MB — each side is an integer code against one shared asin vocabulary rather
than a repeated 10-character string. Pickle keeps that dtype across the round
trip; CSV would not.

## Notes

- Repeat purchases of the same item are **not** collapsed. A pair counts if any
  event of A and any event of B fall inside the window, so an item bought twice
  years apart bridges to whatever sat near either date. Only exact duplicate
  `(user, item, timestamp)` triples are dropped — they can only repeat pairs
  another row already makes.
- An item pairing with itself is not a pair and is dropped.
- Events are sorted by `(user, time)` so each user's history is one ascending
  block, which turns "everything within the window of event i" into a contiguous
  slice. A single `searchsorted` finds every bound at once and the ragged ranges
  are expanded arithmetically — no Python-level loop over the 777k users. The
  full log takes about nine seconds.
