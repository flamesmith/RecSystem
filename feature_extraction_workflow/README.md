# Feature Extraction Workflow

This package holds **two independent pipelines**:

| Pipeline | Module | Output | Grain |
| --- | --- | --- | --- |
| **Item features** (below) | `extract_features.py` | `data/df_features.pkl` | one row per `asin` |
| **User features** ([jump](#user-feature-extraction)) | `extract_features_user.py` | `data/df_user_features.pkl` | one row per purchase event |

---

Reusable functions that turn raw item text (title / description / feature) into a
structured `extracted_features` dict per item, using a category-specific schema.

This is the same pipeline that lives in `notebooks/create_features.ipynb`,
re-packaged so it can be called from any notebook or script.
`extract_features.ipynb` in this folder is the runnable driver — it defines no
logic of its own, it only calls the functions below, so edits to this module take
effect there with no notebook change.

## Inputs

- A pandas DataFrame `df` containing one row per item.
- One or more text columns to extract from (e.g. `['title', 'description', 'feature']`).
- `master_metadata` — `{cat_3: {field_name: {type: 'dictionary'|'regex', values|patterns: [...]}}}`.
  Loaded from `data/master_metadata.json` by default. 69 `cat_3` schemas.
- `global_filters` — list of marketing phrases to scrub before extraction
  (e.g. `"100% satisfaction guaranteed"`). Loaded from `data/global_filters.json`.
- A `category` column **or** an existing `cat_3` column. Categories are stringified
  hierarchical lists like `"['Home & Kitchen', 'Kitchen & Dining', 'Dining & Entertaining', ...]"`.

## Pipeline stages

Five calls end to end:

```python
df_result   = run_feature_extraction(df_items, text_columns, master_metadata, global_filters, ...)
df_expanded = expand_features(df_result)
df_clean    = clean_numeric_ranges(df_expanded)
save_features(df_clean, 'data/df_features.pkl')
```

`run_feature_extraction` covers stages 1–5:

1. **Ensure `cat_3`** — if `cat_1`…`cat_6` are missing, parse them from `category`
   (`ast.literal_eval` + index lookup). `cat_3` is the schema key.
2. **Filter to valid `cat_3`** — drop rows whose `cat_3` isn't in `master_metadata`
   (no extraction schema means no features can be extracted). *See Filter 1.*
3. **Clean text** — for each text column:
   - if it's a stringified list (description, feature), flatten with `parse_list_string`,
   - lowercase, convert number words to digits, normalize unit hyphens,
   - strip non-alphanumeric (keep digits, spaces, decimals, hyphens),
   - tokenize → drop NLTK stopwords (minus `{no, not, non, off, self}`) → lemmatize,
   - remove `global_filters` phrases (*see Filter 2*),
   - collapse whitespace.

   Output: `<col>_cleaned` for each text column.
4. **Extract features per source** — run `extract_features(text, cat_3)` on each
   cleaned column. Uses `master_metadata[cat_3]` to know which fields to look for:
   - `dictionary` fields: scan for any value, longest first; first match wins.
   - `regex` fields: try each pattern in order; first match wins, and the **whole
     match** is stored (`match.group(0)`), not just the captured number.
     `Dimensions` greedily extends the match to capture a trailing unit token.

   Fields are independent — nothing stops two fields in the same schema from
   matching the same text, so overlapping patterns fill both columns.

   Output: `extracted_features_<col>` (a dict per row) for each text column.
5. **Merge** — combine the per-source dicts into one `extracted_features` dict.
   Priority follows the order you pass in (`priority[0]` overwrites `priority[-1]`).
   Default priority is the order of `text_columns`.

Then, as separate calls:

6. **Expand to per-field columns** (`expand_features`) — turn the merged dict into
   typed columns. See *What gets extracted* below for the full inventory. All
   `_unit` columns are standardized through `UNIT_MAP` (`feet`→`ft`, `gram`→`g`,
   `liter`→`l`). Note this normalizes the unit **label** only — it never converts
   the value.
7. **Clean numeric ranges** (`clean_numeric_ranges`) — add a `<col>_cleaned`
   column for each entry in `VALID_RANGES`, with out-of-range values set to NaN.
   The original column is preserved untouched. *See Filter 3.*

## What gets extracted

**25 fields**, in five groups. "cat_3" is how many of the 69 schemas declare the
field — a field only exists where its schema asks for it.

### Categorical — closed vocabulary (11)

`dictionary` fields: a value can only be one of the strings listed in that
schema, so junk can't enter.

| Field | cat_3 | Field | cat_3 |
| --- | --- | --- | --- |
| `Brand` | 69 | `Sub_Type` | 9 |
| `Color` | 69 | `Scent` | 2 |
| `Features` | 69 | `Shape_Style` | 2 |
| `Material` | 69 | | |
| `Product_Type` | 69 | | |
| `Theme` | 19 | | |
| `Size` | 17 | | |
| `Shape` | 14 | | |

### Categorical — unparsed strings (2)

`regex` fields that are never parsed into numbers, so they stay as raw text and
have no closed vocabulary.

| Field | cat_3 | Note |
| --- | --- | --- |
| `Filter_Rating` | 1 | holds `'5 micron'`, `'40 gallon'` — a number and a unit in one string |
| `Part_Number` | 1 | |

### Numeric — single unit (7)

Parsed to one number; the unit is implicit and identical for every row, so a
range means something about the quantity itself.

| Field | Column | Range |
| --- | --- | --- |
| `Bar_Pressure` | `bar_pressure_numeric` | `[1, 25]` |
| `Capacity_Cups` | `capacity_cups_numeric` | `[1, 30]` |
| `Density_Weight` | `density_weight_lb` | `[0.5, 30]` |
| `Pocket_Depth` | `pocket_depth_in` | `[5, 25]` |
| `Power_Rating` | `power_rating_w` | `[1, 5000]` |
| `Stage_Count` | `stage_count_numeric` | `[1, 10]` |
| `Voltage` | `voltage_numeric` | `[110, 240]` |

### Numeric — value + unit pair (4)

Parsed into `<field>_numeric` **and** `<field>_unit`. The number alone is
meaningless — `1.5` is litres or ounces depending on the unit column on that row.

| Field | cat_3 | Columns | Units seen |
| --- | --- | --- | --- |
| `Piece_Count` | 59 | `piece_count_numeric`, `piece_count_unit` | `pc`, `pack`, `count`, `dz`, and the noun being counted (`chair`, `door`, `hook`) |
| `Capacity_Volume` | 16 | `capacity_volume_numeric`, `capacity_volume_unit` | `oz`, `ml`, `l`, `gal`, `qt`, `cup`, plus non-volume `lb`, `g`, `cubic foot`, `bottle`, `slice` |
| `Thread_Count` | 6 | `thread_count_numeric`, `thread_count_unit` | `thread count`, `tc`, `count`, `series` — dimensionless |
| `Weight` | 1 | `weight_numeric`, `weight_unit` | `lb`, `g` |

### Dimensions (1)

| Field | cat_3 | Columns |
| --- | --- | --- |
| `Dimensions` | 55 | `dimension_1`, `dimension_2`, `dimension_3` (sorted descending), `dimension_unit` |

`dimension_unit` mixes `in`, `cm`, `mm`, `ft` and also picks up non-length values
(`count`, `count foot`) from schemas whose `Dimensions` patterns claim `count`,
`qt` or `oz`.

## Filters applied

Three different things get filtered, at three different stages.

### Filter 1 — rows, by `cat_3` (`filter_by_cat_3`)

Drops every item whose `cat_3` has no schema in `master_metadata.json`. Only the
69 covered categories survive; nothing else can produce features.

### Filter 2 — text, by `global_filters` (`remove_global_filters`)

Strips marketing boilerplate from the cleaned text *before* extraction, so
phrases like `"100% satisfaction guaranteed"` can't be mistaken for attributes.
Matching is whole-phrase with word boundaries, longest phrase first.

### Filter 3 — values, by `VALID_RANGES` (`clean_numeric_ranges`)

Adds `<col>_cleaned` with out-of-range values replaced by NaN. **The original
column is never modified**, so you can always compare or fall back.

Entries are `(low, high)` — inclusive on both ends — or `(low, high, inclusive)`
where the third element goes straight to pandas' `Series.between`:
`"both"` | `"neither"` | `"left"` | `"right"`.

| Column | Bound | Meaning |
| --- | --- | --- |
| `bar_pressure_numeric` | `(1, 25)` | `1 <= v <= 25` |
| `capacity_cups_numeric` | `(1, 30)` | `1 <= v <= 30` |
| `density_weight_lb` | `(0.5, 30)` | `0.5 <= v <= 30` |
| `pocket_depth_in` | `(5, 25)` | `5 <= v <= 25` |
| `power_rating_w` | `(1, 5000)` | `1 <= v <= 5000` |
| `stage_count_numeric` | `(1, 10)` | `1 <= v <= 10` |
| `voltage_numeric` | `(110, 240)` | `110 <= v <= 240` |
| `capacity_volume_numeric` | `(0, 1000, "neither")` | `0 < v < 1000` |
| `weight_numeric` | `(0, 501, "neither")` | `0 < v < 501` |
| `piece_count_numeric` | `(0, 501, "neither")` | `0 < v < 501` |
| `thread_count_numeric` | `(0, 2001, "neither")` | `0 < v < 2001` |

The two groups mean different things:

- On the **seven single-unit columns** the bound describes the quantity — mains
  voltage really is 110–240 V, so a value outside it is wrong.
- On the **four two-part columns** the bound is only a plausibility guard against
  parse junk — zeros, model numbers, digits grabbed from the wrong place. The
  number is read in whatever unit that row happens to use, so the bound is
  deliberately loose. Converting each to a single unit first would let it mean
  something sharper.

`Dimensions` has no range filter at all.

## A note on `Capacity`

There used to be a `Capacity` field producing `capacity_numeric` /
`capacity_unit`. It was declared in exactly one schema — `Coffee, Tea &
Espresso` — where its patterns duplicated `Capacity_Cups` (an identical cup
regex) and `Capacity_Volume` (oz/litres) **in that same schema**, so every match
landed in two columns at once. It has been removed. Cups now come from
`Capacity_Cups`, oz and litres from `Capacity_Volume`.

Pickles written before that change still contain `capacity_numeric` and
`capacity_unit`; re-running extraction removes them.

## Quick example

```python
from feature_extraction_workflow import (
    run_feature_extraction, expand_features, clean_numeric_ranges, save_features,
)

df_with_features = run_feature_extraction(
    df_items,
    text_columns=['title', 'description', 'feature'],
    master_metadata='data/master_metadata.json',
    global_filters='data/global_filters.json',
    list_columns=['description', 'feature'],   # these are stringified lists
    priority=['title', 'description', 'feature'],
)

df_table = expand_features(df_with_features)   # dict -> typed columns
df_clean = clean_numeric_ranges(df_table)      # adds the *_cleaned columns
save_features(df_clean, 'data/df_features.pkl')

present = (df_with_features['extracted_features'].apply(len) > 0).sum()
print(f"{present:,} / {len(df_with_features):,} items have extracted features")
```

## Function reference

| Function | Purpose |
| --- | --- |
| `ensure_cat_columns(df, category_col='category')` | Parse `category` into `cat_1..cat_6` if missing. |
| `filter_by_cat_3(df, master_metadata)` | **Filter 1** — keep only rows with `cat_3` in the schema. |
| `parse_list_string(text)` | `"['a', 'b']"` → `'a b'`. |
| `build_stopwords()` | NLTK English stopwords minus a kept whitelist. |
| `clean_text(text, stop_words, lemmatizer)` | Lowercase, normalize, lemmatize, drop stopwords. |
| `build_global_filter_regex(filters)` | Compile the global-filter pattern. |
| `remove_global_filters(text, regex)` | **Filter 2** — strip the marketing phrases. |
| `extract_features(text, cat_3, master_metadata)` | Single-row schema-driven extractor. |
| `run_feature_extraction(df, text_columns, master_metadata, global_filters, ...)` | End-to-end driver for stages 1–5. |
| `parse_numeric_and_unit(value)` | `"16 oz"` → `(16.0, "oz")`. |
| `parse_dimensions(value)` | `"12x18 in"` → `(18.0, 12.0, NaN, "in")`. |
| `expand_features(df, ...)` | Expand merged dict to per-field + numeric/unit columns. |
| `clean_numeric_ranges(df, valid_ranges=VALID_RANGES)` | **Filter 3** — add `<col>_cleaned` with out-of-range values set to NaN. |
| `save_features(df, path)` | Pickle the final dataframe to disk (handles dict columns). |

## Notes

- The cleaning step needs NLTK's `stopwords` and `wordnet`. The module downloads them
  on first use if not already cached.
- Coverage scales with how many text sources you pass in: titles alone cover ~92% of
  items in this dataset, description alone ~81%, all three combined a bit higher.
- The merge step is order-sensitive — pick a `priority` that matches how clean each
  source is (titles are usually cleanest, bullet `feature` text is usually noisiest).
- A regex field stores the whole match, so a mis-anchored pattern loses information
  permanently — `(\d+)[\s-]?liter` on `"1.5 liter"` stores `"5 liter"`, and the `1`
  is gone from every column. Patterns capture `(\d+\.?\d*)` so decimals survive.
- Known open issues in the schemas: `Bakeware` has `Dimensions` and `Piece_Count`
  both claiming `count`; `Bedding Accessories` has `Piece_Count` claiming `x(\d+)`,
  so `24x36` parses as a piece count of 36; `Color` and `Material` share wood words
  (`mahogany`, `pine`, `oak`) in ~50 schemas, so one mention fills both fields.

---

# User Feature Extraction

Lives in `extract_features_user.py`, with `extract_features_user.ipynb` as the
runnable companion. Where the pipeline above describes **items**, this one
describes **users** — and it does so in a way that is safe to train on.

The output has **one row per purchase event**, and every feature on that row is
computed only from that user's purchases on **strictly earlier days**. A model
trained on these rows never sees the current purchase, anything from the same
day, or anything later.

## The leakage guarantee

If a user bought on Mar 1, Apr 2, May 3:

| Row | History it sees |
| --- | --- |
| Mar 1 | *(empty — all features NaN)* |
| Apr 2 | Mar 1 |
| May 3 | Mar 1, Apr 2 |

Time resolution is **days**, not seconds (`unixReviewTime // 86400`). Purchases
made on the same day therefore do **not** see each other — they all read the
history as it stood at the end of the previous day, and become history only for
strictly later days.

Note that "purchase" here means a review record: the log is a review log, and
each review is treated as evidence of a purchase.

## Inputs

- **Purchase log** — `data/Home_and_Kitchen_filtered.csv`. Only
  `reviewerID`, `asin`, `unixReviewTime`, `reviewTime` are read. `asin` and
  `reviewerID` are read as **strings** so IDs with leading zeros
  (e.g. `0560467893`) survive and still join.
- **Item features** — `data/df_features.pkl`, for `cat_2`, `cat_3`, `cat_4` and
  `brand`. These aren't in the purchase log, so they're joined in on `asin`.
  Items with no extracted features contribute no category/brand signal, and the
  category/brand features below stay NaN for those purchases.

Rows with no usable `unixReviewTime` are dropped — they can't be placed on the
timeline. (In the current dataset there are none, so the output is row-for-row
aligned with the CSV.)

## Pipeline stages

1. **`load_purchases`** — read the four log columns, drop timestamp-less rows,
   left-join `cat_2/3/4` + `brand` from `df_features` on `asin`.
2. **`compute_user_features`** — a single ordered pass. Rows are sorted by
   (user, day) with `np.lexsort`, then walked in **(user, day) batches**. For
   each batch:
   1. **emit** features for every row in the batch from the accumulator as it
      currently stands, then
   2. **fold** the batch into the accumulator, so those purchases count as
      history only from the next day onward.

   That ordering is the whole leakage guarantee. The accumulator holds counters
   (brand, `cat_2/3/4`, day-of-week, month, season), the set of seen asins,
   running gap sums, and four regression sums for the activity trend — so the
   pass is O(n) with no per-row lookback. A few minutes on the full ~6.9M rows.
3. **`run_user_feature_extraction`** — chains the two above and returns the key
   columns (`reviewerID`, `asin`, `reviewTime`, `unixReviewTime`) plus the
   features, one row per purchase event.
4. **`save_user_features`** — pickle it (object columns round-trip cleanly).

## Features produced

22 columns, all describing the user's **prior** behaviour.

### Counts & breadth

| Column | Definition |
| --- | --- |
| `prior_purchase_count` | purchases on strictly earlier days |
| `purchase_frequency` | prior count / (days since first / 30) — purchases per 30 days |
| `distinct_items` | distinct `asin` bought before |
| `distinct_brands` | distinct non-null brands bought before |
| `distinct_categories` | distinct non-null `cat_3` bought before |

### Recency & tenure (days)

| Column | Definition |
| --- | --- |
| `account_tenure` | last prior day − first prior day |
| `recency` | current day − last prior day |
| `days_since_first_purchase` | current day − first prior day |

### Temporal / cadence

| Column | Definition |
| --- | --- |
| `avg_inter_purchase_gap` | mean days between consecutive prior purchases |
| `inter_purchase_gap_std` | std of those gaps (burstiness) |
| `preferred_dow` | modal day-of-week of prior purchases (0 = Mon) |
| `preferred_month` | modal month (1–12) |
| `preferred_season` | modal meteorological season (`winter`/`spring`/`summer`/`fall`) |
| `activity_trend` | least-squares slope of cumulative purchases vs. day |

### Category & brand

| Column | Definition |
| --- | --- |
| `favorite_cat_2` / `favorite_cat_3` / `favorite_cat_4` | modal category at each level |
| `category_entropy` | Shannon entropy (bits) of the `cat_2` mix |
| `category_diversity` | distinct `cat_2` / prior count |
| `favorite_brand` | modal brand |
| `brand_loyalty` | top brand's share of brand-known purchases |
| `brand_diversity` | distinct brands / prior count |

### When features are NaN

Missing is meaningful here — it means the history needed to define the feature
doesn't exist:

- **Empty history** — the user's first purchase, and any purchase on that same
  first day: all 22 features NaN. On the current dataset that's **1,624,415 of
  6,898,955 rows (23.5%)**. This is intended; there is no history to summarize.
- `inter_purchase_gap_std` — fewer than two prior gaps.
- `activity_trend` — fewer than two prior purchases.
- `favorite_brand` / `brand_loyalty` — no prior purchase had a known brand.
- `favorite_cat_*` / `category_entropy` — no prior purchase had a known category.

## Quick example

```python
from feature_extraction_workflow import (
    run_user_feature_extraction,
    save_user_features,
)

df_user = run_user_feature_extraction(
    'data/Home_and_Kitchen_filtered.csv',
    'data/df_features.pkl',      # supplies cat_2/3/4 + brand
)
save_user_features(df_user, 'data/df_user_features.pkl')

n_cold = df_user['prior_purchase_count'].isna().sum()
print(f'{n_cold:,} / {len(df_user):,} rows are cold-start (empty history)')
```

## Function reference

| Function | Purpose |
| --- | --- |
| `load_purchases(reviews_path, features_path=None)` | Read the purchase log; join `cat_2/3/4` + `brand` from `df_features`. |
| `compute_user_features(df)` | The leakage-safe pass. Returns a feature frame index-aligned to `df`. |
| `run_user_feature_extraction(reviews_path, features_path, key_columns=...)` | End-to-end driver: keys + features, one row per purchase event. |
| `save_user_features(df, path)` | Pickle the per-event feature table. |

Module constants: `PURCHASE_COLUMNS`, `ITEM_COLUMNS`, `FEATURE_COLUMNS`
(the 22 output columns, in order).

## Notes

- **Row order is preserved.** The output is row-for-row aligned with the input
  CSV, which is what lets a consumer attach these features by position instead
  of by key. That matters: the natural key `(reviewerID, asin, unixReviewTime)`
  is **not unique** in this log — 501,704 rows share a triple with another row,
  so a key merge fans out by ~8%. See `ttn/TTN_Implementation.ipynb`, which
  concats positionally and asserts the keys match row-for-row.
- **`preferred_dow` and `preferred_month` are categorical values in float
  columns.** Month 12 is not "more" than month 1. Embed or cyclically encode
  them; don't scale them as ordinals.
- **`brand_loyalty` and `brand_diversity` use different denominators** —
  loyalty divides by brand-*known* prior purchases, diversity by *all* prior
  purchases. For a user whose history is mostly brand-less items, diversity
  reads low while loyalty doesn't. Don't read them as a matched pair.
- **Same-day purchases contribute a gap of 0**, so `avg_inter_purchase_gap`
  measures days between *events*, not between shopping *sessions*. A user who
  buys three items in one session folds two zero-gaps into the mean.
- **Rerun this after regenerating `df_features.pkl`.** The category/brand
  features depend on that join, so a new extraction run changes them.
