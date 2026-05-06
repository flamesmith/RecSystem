# Feature Extraction Workflow

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

The returned DataFrame keeps the original columns and adds:

- `cat_1` … `cat_6` (if not already present)
- `<col>_cleaned` for each text column
- `extracted_features_<col>` for each text column
- `extracted_features` — the merged dict per row

## Quick example

```python
from feature_extraction_workflow.extract_features import run_feature_extraction

df_with_features = run_feature_extraction(
    df_items,
    text_columns=['title', 'description', 'feature'],
    master_metadata='data/master_metadata.json',
    global_filters='data/global_filters.json',
    list_columns=['description', 'feature'],   # these are stringified lists
    priority=['title', 'description', 'feature'],
)

# Inspect coverage
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

## Notes

- The cleaning step needs NLTK's `stopwords` and `wordnet`. The module downloads them
  on first use if not already cached.
- Coverage scales with how many text sources you pass in: titles alone cover ~92% of
  items in this dataset, description alone ~81%, all three combined a bit higher.
- The merge step is order-sensitive — pick a `priority` that matches how clean each
  source is (titles are usually cleanest, bullet `feature` text is usually noisiest).
