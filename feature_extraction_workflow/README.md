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

## Inputs

- A pandas DataFrame `df` containing one row per item.
- One or more text columns to extract from (e.g. `['title', 'description', 'feature']`).
- `master_metadata` — `{cat_3: {field_name: {type: 'dictionary'|'regex', values|patterns: [...]}}}`.
  Loaded from `data/master_metadata.json` by default.
- `global_filters` — list of marketing phrases to scrub before extraction
  (e.g. `"100% satisfaction guaranteed"`). Loaded from `data/global_filters.json`.
- A `category` column **or** an existing `cat_3` column. Categories are stringified
  hierarchical lists like `"['Home & Kitchen', 'Kitchen & Dining', 'Dining & Entertaining', ...]"`.

## Pipeline stages

For each item:

1. **Ensure `cat_3`** — if `cat_1`…`cat_6` are missing, parse them from `category`
   (`ast.literal_eval` + index lookup). `cat_3` is the schema key.
2. **Filter to valid `cat_3`** — drop rows whose `cat_3` isn't in `master_metadata`
   (no extraction schema means no features can be extracted).
3. **Clean text** — for each text column:
   - if it's a stringified list (description, feature), flatten with `parse_list_string`,
   - lowercase, convert number words to digits, normalize unit hyphens,
   - strip non-alphanumeric (keep digits, spaces, decimals, hyphens),
   - tokenize → drop NLTK stopwords (minus `{no, not, non, off, self}`) → lemmatize,
   - remove `global_filters` phrases,
   - collapse whitespace.

   Output: `<col>_cleaned` for each text column.
4. **Extract features per source** — run `extract_features(text, cat_3)` on each
   cleaned column. Uses `master_metadata[cat_3]` to know which fields to look for:
   - `dictionary` fields: scan for any value, longest first; first match wins.
   - `regex` fields: try each pattern; first match wins.
     `Dimensions` greedily extends the match to capture a trailing unit token.

   Output: `extracted_features_<col>` (a dict per row) for each text column.
5. **Merge** — combine the per-source dicts into one `extracted_features` dict.
   Priority follows the order you pass in (`priority[0]` overwrites `priority[-1]`).
   Default priority is the order of `text_columns`.
6. **Expand to per-field columns** (`expand_features`, optional follow-up step) —
   turn the merged dict into typed columns:
   - One column per field key (`Color`, `Material`, `Dimensions`, …).
   - `<field>_numeric` + `<field>_unit` for `Capacity`, `Capacity_Volume`,
     `Piece_Count`, `Thread_Count`, `Weight` (e.g. `"16 oz"` → `16.0` + `"oz"`).
   - Single-unit fields parsed numeric-only into named columns: `power_rating_w`,
     `voltage_numeric`, `pocket_depth_in`, etc.
   - `Dimensions` parsed into sorted-descending `dimension_1`, `dimension_2`,
     `dimension_3`, plus `dimension_unit`.
   - All `_unit` columns standardized through `UNIT_MAP` (`feet`→`ft`, `gram`→`g`).
7. **Clean numeric ranges** (`clean_numeric_ranges`, optional) — for each numeric
   column in `VALID_RANGES`, add a `<col>_cleaned` column where out-of-range
   values are replaced with NaN. The original column is preserved untouched, so
   you can compare or fall back. Same range table as `notebooks/analyze_features.ipynb`.

The returned DataFrame keeps the original columns and adds:

- `cat_1` … `cat_6` (if not already present)
- `<col>_cleaned` for each text column
- `extracted_features_<col>` for each text column
- `extracted_features` — the merged dict per row
- After `expand_features`: per-field columns plus the `_numeric` / `_unit` /
  `dimension_*` columns described above.

## Quick example

```python
from feature_extraction_workflow import run_feature_extraction, expand_features

df_with_features = run_feature_extraction(
    df_items,
    text_columns=['title', 'description', 'feature'],
    master_metadata='data/master_metadata.json',
    global_filters='data/global_filters.json',
    list_columns=['description', 'feature'],   # these are stringified lists
    priority=['title', 'description', 'feature'],
)

# Optional: turn the dict into typed columns (table form like create_features.ipynb)
df_table = expand_features(df_with_features)

present = (df_with_features['extracted_features'].apply(len) > 0).sum()
print(f"{present:,} / {len(df_with_features):,} items have extracted features")
```

## Function reference

| Function | Purpose |
| --- | --- |
| `ensure_cat_columns(df, category_col='category')` | Parse `category` into `cat_1..cat_6` if missing. |
| `filter_by_cat_3(df, master_metadata)` | Keep only rows with `cat_3` in the schema. |
| `parse_list_string(text)` | `"['a', 'b']"` → `'a b'`. |
| `build_stopwords()` | NLTK English stopwords minus a kept whitelist. |
| `clean_text(text, stop_words, lemmatizer)` | Lowercase, normalize, lemmatize, drop stopwords. |
| `build_global_filter_regex(filters)` | Compile the global-filter pattern. |
| `remove_global_filters(text, regex)` | Strip the marketing phrases. |
| `extract_features(text, cat_3, master_metadata)` | Single-row schema-driven extractor. |
| `run_feature_extraction(df, text_columns, master_metadata, global_filters, ...)` | End-to-end driver. |
| `parse_numeric_and_unit(value)` | `"16 oz"` → `(16.0, "oz")`. |
| `parse_dimensions(value)` | `"12x18 in"` → `(18.0, 12.0, NaN, "in")`. |
| `expand_features(df, ...)` | Expand merged dict to per-field + numeric/unit columns. |
| `clean_numeric_ranges(df, valid_ranges=VALID_RANGES)` | Add `<col>_cleaned` columns clipping out-of-range numerics to NaN. |
| `save_features(df, path)` | Pickle the final dataframe to disk (handles dict columns). |

## Notes

- The cleaning step needs NLTK's `stopwords` and `wordnet`. The module downloads them
  on first use if not already cached.
- Coverage scales with how many text sources you pass in: titles alone cover ~92% of
  items in this dataset, description alone ~81%, all three combined a bit higher.
- The merge step is order-sensitive — pick a `priority` that matches how clean each
  source is (titles are usually cleanest, bullet `feature` text is usually noisiest).

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
