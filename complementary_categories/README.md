# Complementary Categories

Turns Amazon's `also_buy` lists — the asins shown as "frequently bought
together" — into a table of category pairs that get bought together, scored by
support and lift and cut down to the pairs strong enough to act on.

Output: `data/complementary_pairs.pkl`, one row per directed category pair.

## Layout

| File | Role |
| --- | --- |
| `complementary_categories.py` | every function; defines the pipeline |
| `complementary_category_analysis.ipynb` | builds and scores the pairs, plots what each threshold costs, saves `data/pair_stats.pkl` |
| `complementary_categories.ipynb` | applies the chosen thresholds to that table and writes `data/complementary_pairs.pkl` |

The two notebooks are thin drivers: they define no logic, so edits to the module
take effect in both with no notebook change. The one exception is the charts,
which live in the analysis notebook's own cells because they exist to be looked
at and adjusted rather than reused.

**Run the analysis notebook first.** It does the expensive work — the 2.7 GB
pickle and the 2.1 GB catalogue CSV, several minutes — once, and saves the full
scored table. `complementary_categories.ipynb` reads only that, so trying a
different threshold costs seconds.

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
curves, then type them into `complementary_categories.ipynb`:

```python
MIN_EDGES = 5     # support floor, as a whole number of co-purchases
MIN_LIFT = 2.0
```

`MIN_EDGES` is a count rather than a fraction because it is the readable unit;
the equivalent support floor is `MIN_EDGES / n_edges`.

`threshold_grid` adds what the curves cannot show — whether the survivors are
spread across source categories or bunched on a few busy ones:

```python
from complementary_categories import threshold_grid
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
