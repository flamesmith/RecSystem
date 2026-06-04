"""SBERT embedding generation and similarity analysis.

Converts item titles to dense vectors using a small sentence-transformer model,
then exposes:

- a pairwise cosine-similarity matrix builder,
- six category-aware similarity helpers ported from
  `notebooks/analyze_features_eda.ipynb`,
- a generic single-query top-N similarity function,
- PCA-based 2D visualizations colored by category.

Each similarity helper takes the dataframe and the normalized embedding matrix
explicitly so the functions are testable and free of module globals.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Bright, distinct colors for category overlays in the PCA plots.
BRIGHT_COLORS: Sequence[str] = (
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
    "#e6beff", "#aa6e28", "#fffac8", "#00ff00", "#ff4500",
)


# ---------------------------------------------------------------------------
# Embedding generation + matrix construction
# ---------------------------------------------------------------------------

def create_embeddings(
    df: pd.DataFrame,
    text_col: str = "title_cleaned",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    embedding_col: str = "title_embedding",
    show_progress_bar: bool = True,
) -> pd.DataFrame:
    """Generate SBERT embeddings for `text_col` and store them as a new column.

    Returns a copy of `df` with `embedding_col` added (a list of float32 arrays,
    one per row). Empty/NaN text is encoded as the empty string, which yields a
    valid (but uninformative) vector.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = df[text_col].fillna("").tolist()
    print(f"Encoding {len(texts):,} rows from '{text_col}' with {model_name}...")
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress_bar
    )
    df = df.copy()
    df[embedding_col] = list(embeddings)
    print(f"Done. Embedding shape: {embeddings.shape}")
    return df


def build_matrix(
    df: pd.DataFrame,
    embedding_col: str = "title_embedding",
    asin_col: str = "asin",
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack the embedding column into a matrix.

    Returns `(matrix, asins)` so cosine similarity can be computed via a plain
    dot product. `asins` is the parallel array of identifiers — its order
    matches the rows in `matrix` and the rows in `df`.

    Assumes the embeddings are already unit-norm, which holds for
    `all-MiniLM-L6-v2` (its final layer L2-normalizes). If you swap to an
    encoder that doesn't return unit vectors, normalize the matrix here.
    """
    matrix = np.stack(df[embedding_col].values)
    asins = df[asin_col].values
    return matrix, asins


# ---------------------------------------------------------------------------
# Similarity helpers (ported verbatim from analyze_features_eda.ipynb)
# Each takes (df, matrix, asins) explicitly instead of relying on globals.
# ---------------------------------------------------------------------------

def _result_columns(df: pd.DataFrame, top_indices: np.ndarray, similarities: np.ndarray,
                    extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Build the small results frame returned by every helper."""
    cols = ["title", "title_cleaned", "cat_3", "cat_4"]
    for c in extra_cols:
        if c not in cols:
            cols.append(c)
    for c in ["brand", "extracted_features", "imageURL", "imageURLHighRes"]:
        if c not in cols:
            cols.append(c)
    out = {"asin": df["asin"].values[top_indices], "similarity": similarities.round(4)}
    for c in cols:
        if c in df.columns:
            out[c] = df.iloc[top_indices][c].values
    return pd.DataFrame(out)


def get_similar_items_by_category(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
    cat3_boost: float = 0.1,
    cat4_boost: float = 0.05,
) -> Optional[pd.DataFrame]:
    """Top N similar items, with cat_3 and cat_4 score boosts."""
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    query_cat3 = df.iloc[idx]["cat_3"]
    query_cat4 = df.iloc[idx]["cat_4"]

    scores = matrix[idx] @ matrix.T
    scores[idx] = -1

    boosted = scores.copy()
    cat3_vals = df["cat_3"].values
    cat4_vals = df["cat_4"].values
    same_cat3 = cat3_vals == query_cat3
    same_cat4 = same_cat3 & (cat4_vals == query_cat4)
    boosted[same_cat3] += cat3_boost
    boosted[same_cat4] += cat4_boost

    top_indices = np.argsort(boosted)[-n:][::-1]
    results = _result_columns(df, top_indices, scores[top_indices], extra_cols=["cat_5", "cat_6"])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin}")
    print(f"cat_3: {query_cat3} | cat_4: {query_cat4}\n")
    return results


def get_similar_items_same_cat3(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Top N similar items restricted to the same cat_3 as the query."""
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    query_cat3 = df.iloc[idx]["cat_3"]
    same_mask = df["cat_3"].values == query_cat3
    same_idx = np.where(same_mask)[0]

    scores = matrix[idx] @ matrix[same_idx].T
    self_pos = np.where(same_idx == idx)[0]
    if len(self_pos) > 0:
        scores[self_pos[0]] = -1

    top_n = min(n, len(scores))
    top_local = np.argsort(scores)[-top_n:][::-1]
    top_indices = same_idx[top_local]
    results = _result_columns(df, top_indices, scores[top_local])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin} | cat_3: {query_cat3}")
    print(f"Same-category items: {same_mask.sum():,}\n")
    return results


def get_similar_items_same_cat3_diff_cat4(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Same cat_3 as the query, different cat_4."""
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    q_cat3 = df.iloc[idx]["cat_3"]
    q_cat4 = df.iloc[idx]["cat_4"]
    mask = (df["cat_3"].values == q_cat3) & (df["cat_4"].values != q_cat4)
    filt_idx = np.where(mask)[0]
    if len(filt_idx) == 0:
        print(f"No items found in cat_3='{q_cat3}' with a different cat_4 than '{q_cat4}'.")
        return None

    scores = matrix[idx] @ matrix[filt_idx].T
    top_n = min(n, len(scores))
    top_local = np.argsort(scores)[-top_n:][::-1]
    top_indices = filt_idx[top_local]
    results = _result_columns(df, top_indices, scores[top_local])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin} | cat_3: {q_cat3} | cat_4: {q_cat4}")
    print(f"Same cat_3, different cat_4 items: {mask.sum():,}\n")
    return results


def get_similar_items_same_cat3_diff_cat4_product_type(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Same cat_3, but different cat_4 and Product_Type."""
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    q_cat3 = df.iloc[idx]["cat_3"]
    q_cat4 = df.iloc[idx]["cat_4"]
    q_pt = df.iloc[idx]["Product_Type"]
    mask = (
        (df["cat_3"].values == q_cat3)
        & (df["cat_4"].values != q_cat4)
        & (df["Product_Type"].values != q_pt)
    )
    filt_idx = np.where(mask)[0]
    if len(filt_idx) == 0:
        print(
            f"No items found in cat_3='{q_cat3}' with cat_4!='{q_cat4}' "
            f"and Product_Type!='{q_pt}'."
        )
        return None

    scores = matrix[idx] @ matrix[filt_idx].T
    top_n = min(n, len(scores))
    top_local = np.argsort(scores)[-top_n:][::-1]
    top_indices = filt_idx[top_local]
    results = _result_columns(df, top_indices, scores[top_local], extra_cols=["Product_Type"])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin} | cat_3: {q_cat3} | cat_4: {q_cat4} | Product_Type: {q_pt}")
    print(f"Same cat_3, different cat_4 & Product_Type items: {mask.sum():,}\n")
    return results


def get_similar_items_diff_cat4_product_type(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Different cat_4 AND different Product_Type from the query."""
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    q_cat4 = df.iloc[idx]["cat_4"]
    q_pt = df.iloc[idx]["Product_Type"]
    mask = (df["cat_4"].values != q_cat4) & (df["Product_Type"].values != q_pt)
    filt_idx = np.where(mask)[0]
    if len(filt_idx) == 0:
        print(f"No items with cat_4!='{q_cat4}' AND Product_Type!='{q_pt}'.")
        return None

    scores = matrix[idx] @ matrix[filt_idx].T
    top_n = min(n, len(scores))
    top_local = np.argsort(scores)[-top_n:][::-1]
    top_indices = filt_idx[top_local]
    results = _result_columns(df, top_indices, scores[top_local], extra_cols=["Product_Type"])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin} | cat_4: {q_cat4} | Product_Type: {q_pt}")
    print(f"Different cat_4 & Product_Type items: {mask.sum():,}\n")
    return results


def get_similar_items_diff_cat3(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Top N similar items NOT in the same cat_3 as the query."""
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    q_cat3 = df.iloc[idx]["cat_3"]
    mask = df["cat_3"].values != q_cat3
    filt_idx = np.where(mask)[0]
    scores = matrix[idx] @ matrix[filt_idx].T

    top_n = min(n, len(scores))
    top_local = np.argsort(scores)[-top_n:][::-1]
    top_indices = filt_idx[top_local]
    results = _result_columns(df, top_indices, scores[top_local])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin} | cat_3: {q_cat3}")
    print(f"Different-category items: {mask.sum():,}\n")
    return results


def get_top_n_similar(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Plain top-N cosine similarity for one query item, no category filter or boost.

    Use this when you just want the most similar items overall — equivalent to
    `get_similar_items_by_category` with both boosts set to 0.
    """
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    scores = matrix[idx] @ matrix.T
    scores[idx] = -1

    top_indices = np.argsort(scores)[-n:][::-1]
    results = _result_columns(df, top_indices, scores[top_indices])

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin}\n")
    return results


def get_similar_items_by_features(
    asin: str,
    df: pd.DataFrame,
    matrix: np.ndarray,
    asins: np.ndarray,
    n: int = 10,
) -> Optional[pd.DataFrame]:
    """Top N similar items filtered by extracted feature values.

    Keeps candidates whose:
      - `Product_Type` matches the query OR is missing on the candidate,
      - `Material`     matches the query OR is missing on the candidate,
      - `Color`        matches the query exactly.

    A filter is skipped when the query itself is missing that field.
    """
    idx = np.where(asins == asin)[0]
    if len(idx) == 0:
        print(f"ASIN '{asin}' not found.")
        return None
    idx = idx[0]

    q_pt = df.iloc[idx].get("Product_Type")
    q_mat = df.iloc[idx].get("Material")
    q_color = df.iloc[idx].get("Color")

    mask = np.ones(len(df), dtype=bool)
    if "Product_Type" in df.columns and pd.notna(q_pt):
        pt = df["Product_Type"]
        mask &= ((pt == q_pt) | pt.isna()).values
    if "Material" in df.columns and pd.notna(q_mat):
        mat = df["Material"]
        mask &= ((mat == q_mat) | mat.isna()).values
    if "Color" in df.columns and pd.notna(q_color):
        mask &= (df["Color"].values == q_color)

    sub_idx = np.where(mask)[0]
    scores = matrix[idx] @ matrix[sub_idx].T
    self_pos = np.where(sub_idx == idx)[0]
    if len(self_pos) > 0:
        scores[self_pos[0]] = -1

    top_n = min(n, len(scores))
    top_local = np.argsort(scores)[-top_n:][::-1]
    top_indices = sub_idx[top_local]
    results = _result_columns(
        df, top_indices, scores[top_local],
        extra_cols=["Product_Type", "Material", "Color"],
    )

    print(f"Query: {df.iloc[idx]['title_cleaned'][:80]}")
    print(f"ASIN:  {asin}")
    print(f"Product_Type: {q_pt} | Material: {q_mat} | Color: {q_color}")
    print(f"Matching items: {mask.sum():,}\n")
    return results


# ---------------------------------------------------------------------------
# PCA visualization
# ---------------------------------------------------------------------------

def _white_background() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"


def visualize_pca_overall(
    matrix: np.ndarray,
    df: pd.DataFrame,
    color_by: str = "cat_2",
    figsize: Tuple[int, int] = (14, 10),
    random_state: int = 42,
) -> None:
    """PCA-reduce all embeddings to 2D and scatter colored by `color_by`."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    _white_background()
    pca = PCA(n_components=2, random_state=random_state)
    embeddings_2d = pca.fit_transform(matrix)

    labels = df[color_by].values
    unique_labels = sorted(set(labels))

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            s=1, alpha=0.15, label=label,
            color=BRIGHT_COLORS[i % len(BRIGHT_COLORS)],
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"SBERT Embeddings (PCA 2D) colored by {color_by}")
    ax.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=8, fontsize=12,
        frameon=True, facecolor="white", edgecolor="black",
    )
    plt.tight_layout()
    plt.show()


def visualize_pca_per_cat2(
    matrix: np.ndarray,
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    random_state: int = 42,
) -> None:
    """One PCA scatter per cat_2 value, colored by cat_3."""
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    _white_background()
    unique_cat2 = sorted(df["cat_2"].dropna().unique())
    for cat2 in unique_cat2:
        cat2_mask = df["cat_2"].values == cat2
        cat2_idx = np.where(cat2_mask)[0]
        if len(cat2_idx) < 2:
            continue

        pca_sub = PCA(n_components=2, random_state=random_state)
        emb_2d = pca_sub.fit_transform(matrix[cat2_idx])

        cat3_labels = df.iloc[cat2_idx]["cat_3"].values
        unique_cat3 = sorted(set(cat3_labels))

        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        for i, cat3 in enumerate(unique_cat3):
            mask = cat3_labels == cat3
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                s=1, alpha=0.15, label=cat3,
                color=BRIGHT_COLORS[i % len(BRIGHT_COLORS)],
            )
        ax.set_xlabel(f"PC1 ({pca_sub.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca_sub.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title(f"SBERT Embeddings (PCA 2D) — cat_2: {cat2} — colored by cat_3")
        ax.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=8, fontsize=12,
            frameon=True, facecolor="white", edgecolor="black",
        )
        plt.tight_layout()
        plt.show()
