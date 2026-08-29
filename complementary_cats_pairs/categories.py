"""Complementary category pairs from Amazon's `also_buy` co-purchase lists.

For each item Amazon lists the asins shown as "frequently bought together".
Mapping both ends of every one of those edges to its category path turns a
product-level co-purchase list into a category-by-category table: which
categories get bought with which, scored by support and lift.

Two sources, and the split between them matters. `df_features.pkl` is the
source side of every edge — `run_feature_extraction` filters it to the
categories that have a schema, so it holds only those items, but it carries
the extracted features. `also_buy` points anywhere, including outside Home &
Kitchen, so target categories are looked up in the unfiltered catalogue CSV.
Edges whose target is in neither table keep their edge and are marked
`Not in catalogue`: how much of `also_buy` leaves the catalogue is a finding,
not something to drop silently.

Entry point: `run_complementary_pairs`.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

NOT_IN_CATALOGUE = "Not in catalogue"
MISSING = "Missing"
OTHER_SUFFIX = "_Other"

# imageURL/imageURLHighRes are stringified lists of several shots of the same
# product, frequently empty. Only the first url is wanted, and a regex beats a
# per-row ast.literal_eval because the same helper runs over the whole 2 GB
# catalogue.
IMAGE_URL_RE = r"[\'\"](https?://[^\'\"]+)[\'\"]"

SRC_COLS = ["src_cat_2", "src_cat_3", "src_cat_4"]
DST_COLS = ["dst_cat_2", "dst_cat_3", "dst_cat_4"]
PAIR_COLS = SRC_COLS + DST_COLS

# Read off the support and lift curves: 5 co-purchases puts the support floor
# clear of the once-or-twice tail, and lift 2.0 asks for a pair that turns up
# at least twice as often as independence predicts.
DEFAULT_MIN_EDGES = 5
DEFAULT_MIN_LIFT = 2.0


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
def parse_asin_list(value) -> list:
    """Stringified list -> list of asins. Anything unparseable -> []."""
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def first_image_url(high_res: pd.Series, fallback: pd.Series) -> pd.Series:
    """First url of imageURLHighRes, else of imageURL. No image at all -> NaN."""
    urls = high_res.astype(str).str.extract(IMAGE_URL_RE, expand=False)
    return urls.fillna(fallback.astype(str).str.extract(IMAGE_URL_RE, expand=False))


def parse_category_levels(series: pd.Series, n_levels: int = 4) -> pd.DataFrame:
    """Stringified category path -> cat_1..cat_n columns (same rule as the pipeline).

    Parsed here rather than with `feature_extraction_workflow.ensure_cat_columns`,
    which does the same thing: importing that package pulls in `nltk` at module
    load, and nothing else in this workflow needs it.
    """
    def as_list(value):
        if isinstance(value, list):
            return value
        if not isinstance(value, str) or not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return parsed if isinstance(parsed, list) else []

    paths = series.apply(as_list)
    return pd.DataFrame(
        {f"cat_{i + 1}": paths.apply(lambda p, i=i: p[i] if len(p) > i else None)
         for i in range(n_levels)},
        index=series.index,
    )


# --------------------------------------------------------------------------
# cat_4 cleaning
# --------------------------------------------------------------------------
def load_taxonomy(path: PathLike) -> set:
    """`category_taxonomy.json` -> the set of (cat_3, cat_4) pairs that survived review.

    The file is a whitelist of the cat_4 values that are real categories rather
    than product bullets that leaked into the category path — 'Imported',
    '10" high', 'measures 25x25x14cm' and 550-odd others like them. The pair is
    the key, not the value alone, since the same label can be real under one
    parent and junk under another.
    """
    with open(path) as f:
        taxonomy = json.load(f)
    return {
        (cat_3, value)
        for cat_2 in taxonomy
        for cat_3, values in taxonomy[cat_2].items()
        for value in values
    }


def fold_cat_4(cat_3: pd.Series, cat_4: pd.Series, valid_pairs: set) -> pd.Series:
    """Keep cat_4 where the (cat_3, cat_4) pair survived review, else `<cat_3>_Other`.

    Applied to both ends of every edge, so a category is cleaned identically
    whether it turns up as a source or as a target.
    """
    c3 = cat_3.astype(str)
    c4 = cat_4.fillna(MISSING).astype(str)
    keep = pd.Series(list(zip(c3, c4)), index=c3.index).isin(valid_pairs)
    return pd.Series(np.where(keep, c4, c3 + OTHER_SUFFIX), index=c3.index)


# --------------------------------------------------------------------------
# The two tables an edge is built from
# --------------------------------------------------------------------------
def build_base_table(df_features: pd.DataFrame, valid_pairs: set) -> pd.DataFrame:
    """The source side: one row per product, with its cleaned path and `also_buy` list.

    Parameters
    ----------
    df_features : DataFrame
        `df_features.pkl` — the raw metadata with the extraction results joined
        on. Must carry `also_buy`, which only reaches the pickle if the
        extraction ran against a CSV that already had the column.
    valid_pairs : set
        From `load_taxonomy`.

    Returns
    -------
    DataFrame with asin, cat_2, cat_3, cat_4_clean, image_url, also_buy (a real
    list), and also_buy_n.
    """
    if "also_buy" not in df_features.columns:
        raise KeyError(
            "df_features has no 'also_buy' column. The extraction ran against a "
            "CSV built before 'also_buy' was added to fields_to_keep in "
            "data/variable_selection.ipynb. Rebuild the CSV there, then re-run "
            "feature_extraction_workflow/extract_features.ipynb."
        )

    cat_4_clean = fold_cat_4(df_features["cat_3"], df_features["cat_4"], valid_pairs)

    df_base = df_features[["asin", "cat_2", "cat_3"]].copy()
    df_base["cat_4_clean"] = cat_4_clean
    df_base["image_url"] = first_image_url(
        df_features["imageURLHighRes"], df_features["imageURL"]
    )
    df_base["also_buy"] = df_features["also_buy"].apply(parse_asin_list)
    df_base["also_buy_n"] = df_base["also_buy"].str.len()
    return df_base


def load_catalogue_lookup(path: PathLike, valid_pairs: set) -> pd.DataFrame:
    """The target side: asin -> category path and image url over the whole catalogue.

    `also_buy` targets are mostly *not* in `df_features` — that table stops at
    the categories with a schema, while a co-purchase edge can point at any
    item. The filtered CSV was never narrowed that way, so it is the lookup.
    Only the four columns needed are read; the rest of the 2 GB stays on disk.

    About a quarter of the CSV is exact full-row duplicates, the same repeats
    `extract_features.ipynb` drops on load. They are byte-identical, so
    `keep="first"` discards nothing.
    """
    cat_lookup = pd.read_csv(
        path,
        usecols=["asin", "category", "imageURL", "imageURLHighRes"],
        low_memory=False,
    )
    cat_lookup = pd.concat(
        [
            cat_lookup[["asin"]],
            parse_category_levels(cat_lookup["category"]),
            first_image_url(
                cat_lookup["imageURLHighRes"], cat_lookup["imageURL"]
            ).rename("image_url"),
        ],
        axis=1,
    )
    cat_lookup["cat_4_clean"] = fold_cat_4(
        cat_lookup["cat_3"], cat_lookup["cat_4"], valid_pairs
    )
    return (
        cat_lookup[["asin", "cat_2", "cat_3", "cat_4_clean", "image_url"]]
        .drop_duplicates(subset="asin", keep="first")
    )


# --------------------------------------------------------------------------
# Edges -> scored pairs -> the split
# --------------------------------------------------------------------------
def build_pairs(df_base: pd.DataFrame, cat_lookup: pd.DataFrame) -> pd.DataFrame:
    """One row per co-purchase edge, with the category path on both ends.

    ~5M rows at this grain, so the six category columns and the two image urls
    are cast to `category` dtype: the categories hold a few hundred distinct
    values between them, and while the urls run to about a million, one asin is
    repeated across many edges, so the codes-plus-dictionary layout still beats
    storing the string on every row.
    """
    pairs = (
        df_base.loc[df_base["also_buy_n"] > 0,
                    ["asin", "cat_2", "cat_3", "cat_4_clean", "image_url", "also_buy"]]
        .explode("also_buy", ignore_index=True)
        .rename(columns={
            "asin": "src_asin",
            "cat_2": "src_cat_2",
            "cat_3": "src_cat_3",
            "cat_4_clean": "src_cat_4",
            "image_url": "src_image_url",
            "also_buy": "dst_asin",
        })
    )

    pairs = pairs.merge(
        cat_lookup.rename(columns={
            "asin": "dst_asin",
            "cat_2": "dst_cat_2",
            "cat_3": "dst_cat_3",
            "cat_4_clean": "dst_cat_4",
            "image_url": "dst_image_url",
        }),
        on="dst_asin",
        how="left",
    )

    for col in DST_COLS:
        pairs[col] = pairs[col].fillna(NOT_IN_CATALOGUE)
    img_cols = ["src_image_url", "dst_image_url"]
    pairs[PAIR_COLS + img_cols] = pairs[PAIR_COLS + img_cols].astype("category")

    return pairs[["src_asin"] + SRC_COLS + ["src_image_url",
                  "dst_asin"] + DST_COLS + ["dst_image_url"]]


def score_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the edges into distinct category pairs and score each one.

    Both metrics are computed over the resolved edges only: an edge whose
    target is `Not in catalogue` has no category on one end, so it can neither
    support a pair nor count toward the marginals.

      support = edges(s -> d) / N
      lift    = P(s, d) / (P(s) * P(d))

    where `P(s)` is the share of edges leaving that source path and `P(d)` the
    share arriving at that target path. Lift of 1.0 is exactly what
    independence would produce.

    Direction is kept: `A -> B` and `B -> A` are separate rows with their own
    support and lift, since `also_buy` is listed per source item and the two
    directions can carry very different traffic.

    Returns
    -------
    DataFrame, one row per distinct pair, sorted by `edges` descending, with
    the six category columns plus edges, src_edges, dst_edges, support, lift.
    `N` is recoverable as `pair_stats["edges"].sum()` — every resolved edge
    lands in exactly one pair.
    """
    # The category columns go through str first: a categorical groupby key fans
    # out over every unused combination of the six.
    resolved = pairs.loc[pairs["dst_cat_2"] != NOT_IN_CATALOGUE, PAIR_COLS].astype(str)
    n_edges = len(resolved)

    pair_stats = (
        resolved.groupby(PAIR_COLS, sort=False).size()
        .rename("edges").reset_index()
    )
    pair_stats["src_edges"] = pair_stats.groupby(SRC_COLS)["edges"].transform("sum")
    pair_stats["dst_edges"] = pair_stats.groupby(DST_COLS)["edges"].transform("sum")
    pair_stats["support"] = pair_stats["edges"] / n_edges
    pair_stats["lift"] = (pair_stats["edges"] * n_edges) / (
        pair_stats["src_edges"] * pair_stats["dst_edges"]
    )
    return pair_stats.sort_values("edges", ascending=False, ignore_index=True)


def filter_pairs(
    pair_stats: pd.DataFrame,
    min_edges: int = DEFAULT_MIN_EDGES,
    min_lift: float = DEFAULT_MIN_LIFT,
) -> pd.DataFrame:
    """The split: keep pairs clearing both thresholds.

    Neither threshold works alone. Support on its own promotes whatever is
    popular, since a high-traffic category pairs with everything. Lift on its
    own does the opposite: a single-edge pair between two otherwise-unseen
    paths scores `N`, the maximum possible, on one co-purchase.

    `min_edges` is a whole number of co-purchases rather than a support
    fraction because it is the readable unit; the equivalent support floor is
    `min_edges / pair_stats["edges"].sum()`.
    """
    keep = (pair_stats["edges"] >= min_edges) & (pair_stats["lift"] >= min_lift)
    return pair_stats[keep].reset_index(drop=True)


def threshold_grid(
    pair_stats: pd.DataFrame,
    edge_grid: Sequence[int] = (5, 10, 15, 25),
    lift_grid: Sequence[float] = (1.5, 2.0, 3.0),
    min_degree: int = 3,
) -> pd.DataFrame:
    """Cross the two thresholds and report what each combination costs.

    A cut is worth little if the survivors all hang off a few busy source
    categories, so `sources_ok` counts the source paths that keep at least
    `min_degree` pairs. A source that loses all of its pairs drops out of the
    grouping and is counted as not ok. Use this to re-derive the thresholds
    `filter_pairs` is given.
    """
    n_edges = int(pair_stats["edges"].sum())
    n_sources = pair_stats[SRC_COLS].drop_duplicates().shape[0]

    rows = []
    for min_edges in edge_grid:
        for min_lift in lift_grid:
            kept = filter_pairs(pair_stats, min_edges, min_lift)
            degree = kept.groupby(SRC_COLS).size()
            rows.append({
                "min_edges": min_edges,
                "min_support": min_edges / n_edges,
                "min_lift": min_lift,
                "pairs": len(kept),
                "sources_ok": int((degree >= min_degree).sum()),
                "sources_total": n_sources,
                "edge_coverage": kept["edges"].sum() / n_edges,
            })
    return pd.DataFrame(rows)


def save_pairs(df: pd.DataFrame, path: PathLike) -> None:
    """Persist the pair table as a pickle, category columns cast to `category`.

    Pickle over CSV because the six category columns repeat a few hundred
    distinct strings across thousands of rows: as `category` dtype the file is
    roughly a quarter the size of the equivalent CSV, and the dtypes survive
    the round trip.
    """
    df = df.copy()
    for col in PAIR_COLS:
        df[col] = df[col].astype("category")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def run_complementary_pairs(
    features: Union[pd.DataFrame, PathLike],
    catalogue_path: PathLike,
    taxonomy_path: PathLike,
    min_edges: int = DEFAULT_MIN_EDGES,
    min_lift: float = DEFAULT_MIN_LIFT,
    out_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Whole pipeline: features + catalogue -> scored pairs -> the split.

    Parameters
    ----------
    features : DataFrame or path
        `df_features.pkl`, or the frame already loaded. Must carry `also_buy`.
    catalogue_path : path
        `meta_Home_and_Kitchen_filtered.csv`, the unfiltered target-side lookup.
    taxonomy_path : path
        `category_taxonomy.json`, the reviewed (cat_3, cat_4) whitelist.
    min_edges, min_lift : thresholds for `filter_pairs`.
    out_path : path, optional
        If given, the result is written there with `save_pairs`.

    Returns
    -------
    The filtered pair table. To see what was dropped, run the stages
    separately and keep the `score_pairs` output.
    """
    if isinstance(features, (str, Path)):
        features = pd.read_pickle(features)

    valid_pairs = load_taxonomy(taxonomy_path)
    df_base = build_base_table(features, valid_pairs)
    cat_lookup = load_catalogue_lookup(catalogue_path, valid_pairs)
    pairs = build_pairs(df_base, cat_lookup)
    pair_stats = score_pairs(pairs)
    complementary_pairs = filter_pairs(pair_stats, min_edges, min_lift)

    if out_path is not None:
        save_pairs(complementary_pairs, out_path)
    return complementary_pairs
