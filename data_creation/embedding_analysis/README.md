# Embedding Analysis

Convert item titles to dense SBERT vectors and use them, alongside the
extracted features, to find similar products.

This module is the analysis layer that sits on top of
`feature_extraction_workflow/`: it consumes the dataframe produced there
(`data/df_features.pkl`), embeds the cleaned title text, and exposes
similarity helpers and PCA visualizations ported from
`notebooks/analyze_features_eda.ipynb`.

## Pipeline

1. **Load** the feature dataframe written by the extraction notebook
   (`data/df_features.pkl`).
2. **Embed** the `title_cleaned` column with `all-MiniLM-L6-v2` (the
   small SBERT model — 384-dim vectors, ~22M parameters). Output is a
   new `title_embedding` column on the dataframe.
3. **Save** the result to `data/df_features_with_embeddings.pkl`. Large
   pickle (vector × rows) — ignored via `.gitignore`.
4. **Build** the embedding matrix: `matrix, asins = build_matrix(df)`.
   Cosine similarity reduces to a plain dot product because
   `all-MiniLM-L6-v2` already returns unit-norm vectors.
5. **Query** with the similarity helpers (one ASIN at a time):
   - `get_top_n_similar` — plain top-N, no filter.
   - `get_similar_items_by_category` — adds cat_3 / cat_4 score boosts.
   - `get_similar_items_same_cat3` — restrict to same cat_3.
   - `get_similar_items_same_cat3_diff_cat4` — same cat_3, different cat_4.
   - `get_similar_items_same_cat3_diff_cat4_product_type` — same cat_3 but
     different cat_4 *and* different `Product_Type`.
   - `get_similar_items_diff_cat4_product_type` — different cat_4 *and*
     different `Product_Type`.
   - `get_similar_items_diff_cat3` — different cat_3.
6. **Visualize** the embedding space:
   - `visualize_pca_overall` — PCA to 2D, colored by `cat_2`.
   - `visualize_pca_per_cat2` — one scatter per `cat_2` value, colored by
     `cat_3`.

## Inputs

- `data/df_features.pkl` — produced by `feature_extraction_workflow`.
  Must include at minimum: `asin`, `title`, `title_cleaned`,
  `cat_2`, `cat_3`, `cat_4`, `brand`, `Product_Type`, `extracted_features`.

## Outputs

- `data/df_features_with_embeddings.pkl` — same dataframe with a
  `title_embedding` column (list of float32 arrays). Gitignored.

## Quick example

```python
import pandas as pd
from embedding_analysis import (
    create_embeddings,
    build_matrix,
    get_similar_items_by_category,
    visualize_pca_overall,
)

df = pd.read_pickle("data/df_features.pkl")
df = create_embeddings(df, text_col="title_cleaned")
df.to_pickle("data/df_features_with_embeddings.pkl")

matrix, asins = build_matrix(df)
get_similar_items_by_category("B00029TCRG", df, matrix, asins, n=10)
visualize_pca_overall(matrix, df, color_by="cat_2")
```

## Notes

- **Single-query similarity, not pairwise**: each similarity helper takes
  one query ASIN and scores it against the whole matrix. There's no
  precomputed top-N table over all 1.13M items — that would be costly and
  isn't needed for ad-hoc lookup.
- **Title-only**: only `title_cleaned` is embedded, matching what
  `notebooks/analyze_features_eda.ipynb` and `analyze_features.ipynb`
  cell 87 already do. Description and feature columns are not embedded
  here.
- **Heads-up on preprocessing**: `title_cleaned` is the heavily-processed
  string (lowercased, stopwords stripped, lemmatized). Modern SBERT
  models are trained on natural sentences and generally perform better on
  raw text. Using `title_cleaned` here keeps the pipeline consistent with
  the existing project; switch the `text_col` argument of
  `create_embeddings` to `"title"` if you ever want to compare.

## Function reference

| Function | Purpose |
| --- | --- |
| `create_embeddings(df, text_col, ...)` | Generate SBERT vectors for a text column. |
| `build_matrix(df, embedding_col)` | Stack embeddings into a matrix; return `(matrix, asins)`. |
| `get_similar_items_by_category(asin, df, matrix, asins, ...)` | Top-N with cat_3/cat_4 boosts. |
| `get_similar_items_same_cat3(...)` | Top-N within the same cat_3. |
| `get_similar_items_same_cat3_diff_cat4(...)` | Same cat_3, different cat_4. |
| `get_similar_items_same_cat3_diff_cat4_product_type(...)` | Same cat_3, different cat_4 + Product_Type. |
| `get_similar_items_diff_cat4_product_type(...)` | Different cat_4 + Product_Type. |
| `get_similar_items_diff_cat3(...)` | Top-N outside the query's cat_3. |
| `get_top_n_similar(asin, df, matrix, asins, n=10)` | Plain cosine top-N, no filter. |
| `visualize_pca_overall(matrix, df, color_by='cat_2')` | 2D PCA scatter, all items. |
| `visualize_pca_per_cat2(matrix, df)` | One 2D PCA scatter per cat_2 value. |
