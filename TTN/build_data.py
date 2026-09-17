"""Build the model-ready data for TTN, from the raw feature/pairs tables.

Extracted verbatim from ttn_complementary.ipynb's sections 1-9 (data
loading through array export). Produces everything build_model.py needs:
data/tower/items.npz, vocabs.json, pairs_{train,test}.parquet,
node_of_item.npy, item_asins.npy.

Prerequisites (built by feature_extraction_workflow/, embedding_analysis/,
and complementary_cats_pairs/ -- see their own READMEs):
  data/Home_and_Kitchen_filtered.csv
  data/df_features.pkl
  data/df_features_with_embeddings.pkl
  data/co_purchase_pairs_{train,test}.pkl
  data/complementary_categories.pkl
  data/meta_Home_and_Kitchen_filtered.csv
  data/category_taxonomy.json, data/master_metadata.json
  TTN/constants.json (date_threshold)

Usage:
  python TTN/build_data.py
  python TTN/encode_descriptions.py    # optional but recommended -- see its
                                        # own docstring; adds description
                                        # signal to the model trained next
  python TTN/build_model.py
"""

# ============================================================================
# Setup
# ============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# data/ lives at the repo root, one level up from this ttn/ folder
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Show every column/variable when displaying a dataframe
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)
# Turn off scientific notation (e.g. 2.447268e+06 -> 2447268.00)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# ============================================================================
# 1. Load the datasets
# ============================================================================
import json
from pathlib import Path

# --- 1. Interactions: one row per review ----------------------------------
df_reviews = pd.read_csv(
    DATA_DIR / "Home_and_Kitchen_filtered.csv",
    dtype={"asin": str, "reviewerID": str},
    low_memory=False,
)

# --- 2. Item features: one row per asin, the extracted attributes ---------
df_features = pd.read_pickle(DATA_DIR / "df_features.pkl")

# --- 3. Co-purchase pairs, one table per side of the cutoff ---------------
co_pairs = {
    "train": pd.read_pickle(DATA_DIR / "co_purchase_pairs_train.pkl"),
    "test": pd.read_pickle(DATA_DIR / "co_purchase_pairs_test.pkl"),
}

# The split point, read from the same file pairs.ipynb reads. Only needed to
# scope the fitted statistics in §5 and §7 to the training period.
DATE_THRESHOLD = json.loads((Path(__file__).parent / "constants.json").read_text())["date_threshold"]
cutoff_time = pd.Timestamp(DATE_THRESHOLD).timestamp()

# --- 4. The complementary category mapping --------------------------------
comp_cat = pd.read_pickle(DATA_DIR / "complementary_categories.pkl")

# --- 5. The unfiltered catalogue: coverage for items df_features lacks ----
meta_catalogue = pd.read_csv(
    DATA_DIR / "meta_Home_and_Kitchen_filtered.csv",
    usecols=["asin", "category", "title", "brand", "price"],
    dtype={"asin": str},
    low_memory=False,
)

for name, frame in [
    ("df_reviews", df_reviews), ("df_features", df_features),
    ("co_pairs[train]", co_pairs["train"]), ("co_pairs[test]", co_pairs["test"]),
    ("comp_cat", comp_cat),
    ("meta_catalogue", meta_catalogue),
]:
    print(f"{name:<18} {str(frame.shape):>18}")

print(f"\nunique users: {df_reviews['reviewerID'].nunique():,} | "
      f"unique items: {df_reviews['asin'].nunique():,}")
print(f"items described by df_features : {df_features['asin'].nunique():,}")
print(f"items described by the catalogue: {meta_catalogue['asin'].nunique():,}")
print(f"split point (ttn/constants.json): {DATE_THRESHOLD}")
df_reviews.head(5)

# ============================================================================
# 2. Validate the feature table
# ============================================================================
from feature_extraction_workflow.validations import run_all

report = run_all(df_features, DATA_DIR / "master_metadata.json")
print(report if len(report) else "no findings")

# ============================================================================
# 3. Clean the item category path
# ============================================================================
import json

TAXONOMY_PATH = DATA_DIR / "category_taxonomy.json"
with open(TAXONOMY_PATH) as f:
    taxonomy = json.load(f)

# cat_3 -> the set of cat_4 values that survived the review
valid_cat_4 = {c3: set(vals) for c2 in taxonomy for c3, vals in taxonomy[c2].items()}
print(f"taxonomy: {len(taxonomy)} cat_2 | {len(valid_cat_4)} cat_3 | "
      f"{sum(len(v) for v in valid_cat_4.values())} valid cat_4 slots")

MISSING = "Missing"
OTHER_SUFFIX = "_Other"

cat_3 = df_features["cat_3"].astype(str)
cat_4 = df_features["cat_4"].fillna(MISSING).astype(str)

# A value is kept only if it is valid *under its own parent* — the same label
# can be real in one branch and junk in another.
valid_pairs = {(c3, v) for c3, vals in valid_cat_4.items() for v in vals}
keep = pd.Series(list(zip(cat_3, cat_4)), index=df_features.index).isin(valid_pairs)

df_features["cat_4_clean"] = np.where(keep, cat_4, cat_3 + OTHER_SUFFIX)

n_before = df_features["cat_4"].nunique(dropna=False)
n_after = df_features["cat_4_clean"].nunique()
n_folded = int((~keep).sum())
print(f"\ndistinct cat_4 : {n_before:,} -> {n_after:,}")
print(f"items folded into '<cat_3>{OTHER_SUFFIX}': {n_folded:,} "
      f"({n_folded / len(df_features) * 100:.2f}% of the catalog)")
df_features[["asin", "cat_2", "cat_3", "cat_4", "cat_4_clean"]].head(5)

# ============================================================================
# 4. Build the pair tables
# ============================================================================
from complementary_cats_pairs import DST_COLS, SRC_COLS

CAT_LEVELS = ["cat_2", "cat_3", "cat_4_clean"]     # matches the mapping's levels
QUERY_CATS = ["query_cat_2", "query_cat_3", "query_cat_4"]
TARGET_CATS = ["target_cat_2", "target_cat_3", "target_cat_4"]
PAIR_COLS = ["asin_query", "asin_target"] + QUERY_CATS + TARGET_CATS

item_cats = df_features[["asin"] + CAT_LEVELS]
mapping = comp_cat[SRC_COLS + DST_COLS].astype(str)


def build_tower_pairs(pairs):
    """Co-purchase pairs -> directed (query, target) rows the mapping licenses."""
    with_cats = (
        pairs
        .assign(asinA=pairs["asinA"].astype(str), asinB=pairs["asinB"].astype(str))
        .merge(item_cats.add_prefix("a_"), left_on="asinA", right_on="a_asin", how="inner")
        .merge(item_cats.add_prefix("b_"), left_on="asinB", right_on="b_asin", how="inner")
        .drop(columns=["a_asin", "b_asin"])
    )
    a_cats = [f"a_{c}" for c in CAT_LEVELS]
    b_cats = [f"b_{c}" for c in CAT_LEVELS]

    def directed(query_asin, query_cats, target_asin, target_cats):
        out = with_cats.merge(mapping, left_on=query_cats + target_cats,
                              right_on=SRC_COLS + DST_COLS, how="inner")
        return out.rename(columns=dict(
            [(query_asin, "asin_query"), (target_asin, "asin_target")]
            + list(zip(query_cats, QUERY_CATS))
            + list(zip(target_cats, TARGET_CATS))))[PAIR_COLS]

    both = pd.concat([directed("asinA", a_cats, "asinB", b_cats),
                      directed("asinB", b_cats, "asinA", a_cats)], ignore_index=True)
    return both, len(with_cats)


slices = {}
for name, pairs in co_pairs.items():
    slices[name], joined = build_tower_pairs(pairs)
    print(f"{name:<6} {len(pairs):>11,} pairs -> {joined:>11,} with features both ends "
          f"({joined / len(pairs):>5.1%}) -> {len(slices[name]):>10,} directed rows")

held_out = len(slices["test"]) / sum(len(v) for v in slices.values())
print(f"\nheld out: {held_out:.1%}")
slices["train"].head(5)

# ============================================================================
# 5. Fold rare brands -- fitted on train
# ============================================================================
# Fold rare brands: keep those carried by MORE THAN 10 distinct items. A brand
# on three items would get an embedding trained by a handful of gradient
# updates; bucketing those into one `other_brands` symbol is more honest.
MIN_ITEMS = 10
OTHER = "other_brands"

# FITTED ON TRAIN. The count is taken over items appearing in the training
# pairs, so the test period has no say in which brands survive.
train_items = pd.unique(pd.concat(
    [slices["train"]["asin_query"], slices["train"]["asin_target"]], ignore_index=True))

# Count on brand_norm where the pipeline produced it, so "3d rose" and "3drose"
# are not counted separately and pushed under the threshold by a split spelling.
source = "brand_norm" if "brand_norm" in df_features.columns else "brand"
train_features = df_features[df_features["asin"].isin(train_items)]
brand_counts = train_features.groupby(source)["asin"].nunique()
kept_brands = brand_counts[brand_counts > MIN_ITEMS].index

# APPLIED TO ALL ITEMS, so test rows are folded by the training rule.
df_features["brand_clean"] = df_features[source].where(
    df_features[source].isin(kept_brands) | df_features[source].isna(), OTHER)

print(f"counted on   : {source}, over {len(train_items):,} training items")
print(f"brands kept  : {len(kept_brands):,} of "
      f"{train_features[source].nunique():,} seen in training")
print(f"all items    : {(df_features['brand_clean'] == OTHER).mean():.1%} in {OTHER}, "
      f"{df_features[source].isna().mean():.1%} missing (left as NaN for §7)")
print()
print(df_features["brand_clean"].value_counts().head(10))

# ============================================================================
# 6. Attach the item attributes, and fit the vocabularies
# ============================================================================
# --- Item attributes, one row per asin ------------------------------------
# Source column in df_features -> the name it takes in the pair tables.
ITEM_ATTRS = {
    "title_cleaned": "title_cleaned",
    "price": "price",
    "brand_clean": "brand_clean",
    "Color": "color",
    "Features": "features",
    "Material": "material",
    "Product_Type": "product_type",
}
CATEGORICAL_ATTRS = ["cat_2", "cat_3", "cat_4", "brand_clean",
                     "color", "features", "material", "product_type"]
SIDE_COLS = ["asin_{s}", "{s}_cat_2", "{s}_cat_3", "{s}_cat_4",
             "{s}_title_cleaned", "{s}_price", "{s}_brand_clean",
             "{s}_color", "{s}_features", "{s}_material", "{s}_product_type"]

item_attrs = (df_features[["asin"] + list(ITEM_ATTRS)].rename(columns=ITEM_ATTRS))
item_attrs["price"] = pd.to_numeric(
    item_attrs["price"].astype(str).str.replace(r"[$,]", "", regex=True),
    errors="coerce")
attrs_by_asin = item_attrs.set_index("asin")
print(f"price parsed : {item_attrs['price'].notna().sum():,} of {len(item_attrs):,} "
      f"items ({item_attrs['price'].notna().mean():.1%})")


def attach_attributes(frame):
    """Item content for both sides. `.map`, not `merge`, so re-running is safe."""
    for side in ("query", "target"):
        asin = frame[f"asin_{side}"]
        for attr in ITEM_ATTRS.values():
            frame[f"{side}_{attr}"] = asin.map(attrs_by_asin[attr])
    return frame[[c.format(s=s) for s in ("query", "target") for c in SIDE_COLS]]


for name in slices:
    slices[name] = attach_attributes(slices[name])

# --- Fit the vocabularies on TRAIN, apply to both -------------------------
# `title_cleaned` is excluded: it is free text for the encoder, not a symbol to
# look up, and 145k levels would only cost memory.
vocabularies = {}
for attr in CATEGORICAL_ATTRS:
    values = set()
    for side in ("query", "target"):
        values |= set(slices["train"][f"{side}_{attr}"].dropna().astype(str))
    vocabularies[attr] = pd.CategoricalDtype(sorted(values))

print(f"\nvocabularies fitted on {len(slices['train']):,} training rows:")
for attr, dtype in vocabularies.items():
    print(f"  {attr:<14} {len(dtype.categories):>8,} levels")
slices["train"].head(5)

# ============================================================================
# 7. Fill the missing values -- fitted on train
# ============================================================================
from complementary_cats_pairs import fold_cat_4, parse_category_levels

# --- Category price medians: FITTED ON TRAIN ------------------------------
# Population: items reviewed strictly before the cutoff. The category path is
# parsed and folded exactly as §3 folds df_features, so the medians key onto
# query_cat_2/3/4 and target_cat_2/3/4 directly.
train_reviewed = set(df_reviews.loc[df_reviews["unixReviewTime"] < cutoff_time, "asin"])

cat_levels = parse_category_levels(meta_catalogue["category"], n_levels=4)
catalogue_prices = pd.DataFrame({
    "asin": meta_catalogue["asin"],
    "cat_2": cat_levels["cat_2"].astype(str),
    "cat_3": cat_levels["cat_3"].astype(str),
    "cat_4": fold_cat_4(cat_levels["cat_3"], cat_levels["cat_4"], valid_pairs),
    "price": pd.to_numeric(
        meta_catalogue["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"),
})
fit_prices = catalogue_prices[catalogue_prices["asin"].isin(train_reviewed)]

median_4 = fit_prices.groupby(["cat_2", "cat_3", "cat_4"], observed=True)["price"].median()
median_3 = fit_prices.groupby(["cat_2", "cat_3"], observed=True)["price"].median()
median_2 = fit_prices.groupby(["cat_2"], observed=True)["price"].median()
median_all = fit_prices["price"].median()

print(f"median population : {len(fit_prices):,} items reviewed before "
      f"{DATE_THRESHOLD} ({fit_prices['price'].notna().sum():,} priced)")
print(f"median tables     : {median_4.notna().sum():,} cat_4 paths | "
      f"{median_3.notna().sum()} cat_3 | {median_2.notna().sum()} cat_2 | "
      f"global ${median_all:,.2f}")


def fill_prices(frame):
    """Fill each missing price from its category median, falling back up."""
    for side in ("query", "target"):
        raw = frame[f"asin_{side}"].map(attrs_by_asin["price"])
        c2 = frame[f"{side}_cat_2"].astype(str)
        c3 = frame[f"{side}_cat_3"].astype(str)
        c4 = frame[f"{side}_cat_4"].astype(str)
        at_4 = pd.Series(median_4.reindex(pd.MultiIndex.from_arrays([c2, c3, c4])).to_numpy(),
                         index=frame.index)
        at_3 = pd.Series(median_3.reindex(pd.MultiIndex.from_arrays([c2, c3])).to_numpy(),
                         index=frame.index)
        at_2 = pd.Series(median_2.reindex(pd.Index(c2)).to_numpy(), index=frame.index)
        missing = raw.isna()
        frame[f"{side}_price"] = raw.fillna(
            at_4.fillna(at_3).fillna(at_2).fillna(median_all))
        frame[f"{side}_price_imputed"] = missing
        yield side, missing, at_4, at_3, at_2


def apply_vocabularies(frame):
    """Cast to the train-fitted dtypes; unseen values fall out as NaN."""
    unseen = {}
    for attr, dtype in vocabularies.items():
        for side in ("query", "target"):
            col = f"{side}_{attr}"
            values = frame[col].astype(object)
            cast = pd.Categorical(values, dtype=dtype)
            unseen[attr] = unseen.get(attr, 0) + int(
                (values.notna() & pd.isna(cast)).sum())
            frame[col] = cast
    return unseen


for name, frame in slices.items():
    print(f"\n--- {name} ---")
    for side, missing, at_4, at_3, at_2 in fill_prices(frame):
        rungs = (int((missing & at_4.notna()).sum()),
                 int((missing & at_4.isna() & at_3.notna()).sum()),
                 int((missing & at_4.isna() & at_3.isna() & at_2.notna()).sum()),
                 int((missing & at_4.isna() & at_3.isna() & at_2.isna()).sum()))
        print(f"  {side:<7} {missing.sum():>9,} missing ({missing.mean():>5.1%}) -> "
              f"cat_4 {rungs[0]:,} | cat_3 {rungs[1]:,} | cat_2 {rungs[2]:,} | "
              f"global {rungs[3]:,} | still NaN "
              f"{frame[f'{side}_price'].isna().sum():,}")
    unseen = apply_vocabularies(frame)
    total_unseen = sum(unseen.values())
    print(f"  values outside the training vocabulary: {total_unseen:,} "
          f"({total_unseen / (len(frame) * 2 * len(vocabularies)):.4%} of slots)"
          + (f"  {ute}" if (ute := {k: v for k, v in unseen.items() if v}) else ""))

# ============================================================================
# 7b. Fill missing categoricals
# ============================================================================
MISSING_LABEL = "Missing"


def fill_missing_categories(frame, attrs, label=MISSING_LABEL):
    """Give every missing categorical value an explicit `label` level.

    Covers both the genuinely absent and anything that fell outside the
    train-fitted vocabulary in the cell above -- after the cast both are NaN,
    and both mean the same thing to the encoder: no usable value here.

    A pandas Categorical refuses a value that is not one of its categories, so
    the level is added before it is assigned. Idempotent.
    """
    for attr in attrs:
        for side in ("query", "target"):
            col = f"{side}_{attr}"
            series = frame[col]
            if label not in series.cat.categories:
                series = series.cat.add_categories([label])
            frame[col] = series.fillna(label)
    return frame


for name, frame in slices.items():
    before = {c: frame[c].isna().sum()
              for c in (f"{s}_{a}" for s in ("query", "target") for a in CATEGORICAL_ATTRS)}
    slices[name] = fill_missing_categories(frame, CATEGORICAL_ATTRS)
    filled = sum(before.values())
    print(f"{name:<6} filled {filled:>9,} categorical values with {MISSING_LABEL!r} "
          f"({filled / (len(frame) * 2 * len(CATEGORICAL_ATTRS)):.2%} of slots)")

print("\nvocabularies identical across towers and slices:")
for attr in CATEGORICAL_ATTRS:
    levels = [list(slices[n][f"{s}_{attr}"].cat.categories)
              for n in slices for s in ("query", "target")]
    print(f"  {attr:<14} {all(l == levels[0] for l in levels)}  "
          f"({len(levels[0]):,} levels, {MISSING_LABEL!r} present: "
          f"{MISSING_LABEL in levels[0]})")

# ============================================================================
# 8. target_node -- the target category as one symbol
# ============================================================================
# One symbol per target category path, for the query tower's target-category
# embedding. `" > "` is cosmetic -- it only has to be readable and not appear
# inside a category name.
NODE_SEP = " > "
TARGET_LEVELS = ["target_cat_2", "target_cat_3", "target_cat_4"]


def target_node(frame):
    return (frame["target_cat_2"].astype(str) + NODE_SEP
            + frame["target_cat_3"].astype(str) + NODE_SEP
            + frame["target_cat_4"].astype(str))


# FITTED ON TRAIN, applied to both.
node_vocabulary = pd.CategoricalDtype(sorted(set(target_node(slices["train"]))))
print(f"target_node vocabulary: {len(node_vocabulary.categories):,} paths, "
      f"fitted on {len(slices['train']):,} training rows")

for name, frame in slices.items():
    nodes = target_node(frame)
    cast = pd.Categorical(nodes, dtype=node_vocabulary)
    unseen = int(pd.isna(cast).sum())
    if unseen:                     # never expected: §4 filters on the mapping
        cast = cast.add_categories([MISSING_LABEL]).fillna(MISSING_LABEL)
    frame["target_node"] = cast
    print(f"  {name:<6} {nodes.nunique():>4,} distinct | outside the training "
          f"vocabulary: {unseen:,}")

for name, frame in slices.items():
    print(f"\n{name:<6} {str(frame.shape):>16} | "
          f"{frame.memory_usage(deep=True).sum() / 1e6:>6,.0f} MB | "
          f"price complete: "
          f"{frame['query_price'].notna().all() and frame['target_price'].notna().all()}")
print(f"\ncolumns ({slices['train'].shape[1]}): {list(slices['train'].columns)}")
slices["train"].head(5)

# ============================================================================
# 8b. Drop same-category pairs
# ============================================================================
# --- Drop same-category pairs: substitutes, not complements ---------------
# Applied to both slices with the same rule -- this is a per-row predicate,
# not a fitted statistic, so it carries no train/test asymmetry.
def drop_same_category(frame):
    same = ((frame["query_cat_4"] == frame["target_cat_4"])
            & (frame["query_cat_4"] != MISSING_LABEL))
    return frame.loc[~same].reset_index(drop=True), int(same.sum())


for name in slices:
    before = len(slices[name])
    slices[name], removed = drop_same_category(slices[name])
    print(f"{name:<6} {before:>10,} -> {len(slices[name]):>10,} rows "
          f"(dropped {removed:,}, {removed / before:.1%})")

tower_pairs_train = slices["train"]
tower_pairs_test = slices["test"]
print(f"\nheld out: "
      f"{len(tower_pairs_test) / (len(tower_pairs_train) + len(tower_pairs_test)):.1%}")

# The node vocabulary was fitted before this filter, so some of its 448 paths
# may no longer occur. Harmless -- an unused embedding row is never updated --
# but worth seeing rather than assuming.
still_used = tower_pairs_train["target_node"].nunique()
print(f"target_node paths still present in train: {still_used:,} of "
      f"{len(node_vocabulary.categories):,}")

# ============================================================================
# 9. Export the model-ready arrays
# ============================================================================
from pathlib import Path

OUT_DIR = DATA_DIR / "tower"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ENCODE_TITLES = False          # True -> run SBERT instead of reusing the pickle
USE_DESCRIPTION = True         # encode `description_cleaned` alongside the title
DESC_CACHE = OUT_DIR / "desc_emb.npy"   # SBERT over 137k rows is slow; cache it
ALLOW_INLINE_ENCODE = False    # True -> encode here if the cache is missing.
                               # Leave False and use ttn/encode_descriptions.py:
                               # encoding inside this cell competes for memory
                               # with df_features and stalls badly.

CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material",
             "product_type", "features"]
# column in the pair tables (without the query_/target_ prefix) -> canonical name
COLMAP = {"cat_2": "cat_2", "cat_3": "cat_3", "cat_4": "cat_4",
          "brand_clean": "brand", "color": "color", "material": "material",
          "product_type": "product_type", "features": "features"}


def _side(df, prefix):
    cols = {f"{prefix}_{src}": dst for src, dst in COLMAP.items()}
    cols[f"asin_{prefix}"] = "asin"
    cols[f"{prefix}_title_cleaned"] = "title"
    cols[f"{prefix}_price"] = "price"
    cols[f"{prefix}_price_imputed"] = "price_imputed"
    return df[list(cols)].rename(columns=cols)


def build_items(train, test):
    """One row per asin, from every side of both slices."""
    frames = [_side(d, p) for d in (train, test) for p in ("query", "target")]
    items = (pd.concat(frames, ignore_index=True)
               .drop_duplicates("asin").reset_index(drop=True))
    items["idx"] = np.arange(len(items))
    return items


def fit_vocabs(train):
    """FIT ON TRAIN ONLY. Id 0 is reserved for padding / unseen."""
    vocabs = {}
    for src, dst in COLMAP.items():
        values = pd.unique(pd.concat(
            [train[f"query_{src}"].astype(str), train[f"target_{src}"].astype(str)],
            ignore_index=True))
        vocabs[dst] = {v: i + 1 for i, v in enumerate(sorted(values))}
    nodes = sorted(train["target_node"].astype(str).unique())
    vocabs["target_node"] = {v: i + 1 for i, v in enumerate(nodes)}
    return vocabs


def encode_items(items, vocabs, title_emb):
    cat_ids = np.zeros((len(items), len(CAT_ORDER)), dtype=np.int64)
    for j, name in enumerate(CAT_ORDER):
        cat_ids[:, j] = items[name].astype(str).map(vocabs[name]).fillna(0).to_numpy()

    # FIX: the price, not the imputation flag. `price_imputed` is the indicator.
    price = items["price"].astype(float)
    imputed = items["price_imputed"].astype("float32")
    price = price.fillna(price.median())          # defensive; §7 leaves none

    # FIX: normalise by the bins qcut actually produced, not a hardcoded 10.
    decile = pd.qcut(price, 10, labels=False, duplicates="drop")
    decile = decile.fillna(decile.median())
    numeric = np.stack([
        np.log1p(price.to_numpy()).astype("float32"),
        (decile.to_numpy() / max(decile.max(), 1)).astype("float32"),
        imputed.to_numpy(),
    ], axis=1).astype("float32")
    return {"title_emb": title_emb.astype("float32"),
            "cat_ids": cat_ids, "numeric": numeric}


def slim_pairs(df, items, vocabs):
    idx = dict(zip(items["asin"], items["idx"]))
    out = pd.DataFrame({
        "query_idx": df["asin_query"].map(idx),
        "target_idx": df["asin_target"].map(idx),
        "target_node_id": df["target_node"].astype(str)
                            .map(vocabs["target_node"]).fillna(0).astype(int),
        "weight": df.get("weight", pd.Series(1.0, index=df.index)).astype("float32"),
    }).dropna()
    return out.astype({"query_idx": int, "target_idx": int})


def item_title_embeddings(items):
    if ENCODE_TITLES:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(model.encode(items["title"].astype(str).fillna("").tolist(),
                                       batch_size=256, show_progress_bar=True))
    # Reuse the vectors embedding_analysis/ already produced (384-d, unit-norm).
    # Narrowed to our asins before anything is stacked: the pickle holds 1.13M
    # vectors and we need ~160k, so a lookup over the whole frame is the
    # difference between seconds and minutes.
    wanted = set(items["asin"])
    frame = pd.read_pickle(DATA_DIR / "df_features_with_embeddings.pkl")[
        ["asin", "title_embedding"]]
    frame = frame[frame["asin"].isin(wanted)]
    vectors = frame.set_index("asin")["title_embedding"].reindex(items["asin"])
    del frame

    found = vectors.notna().to_numpy()
    dim = len(vectors[found].iloc[0])
    out = np.zeros((len(items), dim), dtype="float32")
    out[found] = np.stack(vectors[found].to_numpy()).astype("float32")
    print(f"title vectors: {int(found.sum()):,} matched, "
          f"{int((~found).sum()):,} missing (left as zeros)")
    return out


def item_description_embeddings(items):
    """SBERT over `description_cleaned`, cached to disk.

    The description is not carried through the pair tables, so it is joined back
    on from `df_features` by asin. Several attributes (`color`, `material`,
    `product_type`) were already extracted FROM this text, but extraction keeps
    only what its patterns matched -- the free text still carries size, style,
    intended use and compatibility that no categorical slot holds.

    This is the one signal a cold item has on day one: a description exists
    before any purchase does. §18 measures R@10 at 0.003 for targets never seen
    in training against 0.48 for targets seen more than 100 times, and the
    per-feature ablation there shows the model falls back on `cat_4` for rare
    items -- a field thousands of items share, so it cannot separate them.

    all-MiniLM-L6-v2 truncates at 256 word-pieces, so only the opening of a long
    description is encoded. That is usually where the product summary sits.
    """
    desc = (df_features[["asin", "description_cleaned"]]
            .drop_duplicates("asin").set_index("asin")["description_cleaned"]
            .reindex(items["asin"]).fillna("").astype(str))
    blank = (desc.str.strip() == "").to_numpy()
    dup = float(desc[~blank].duplicated(keep=False).mean()) if (~blank).any() else 0.0
    print(f"descriptions: {int((~blank).sum()):,} non-empty, "
          f"{int(blank.sum()):,} blank (left as zeros)")
    print(f"  share sharing their text with another item: {dup:.1%} -- boilerplate "
          f"cannot separate items, so read that as a ceiling on what this adds")

    if DESC_CACHE.exists():
        cached = np.load(DESC_CACHE)
        if len(cached) == len(items):
            print(f"  reusing {DESC_CACHE.name} {cached.shape}")
            return cached.astype("float32")
        print(f"  {DESC_CACHE.name} has {len(cached):,} rows, need {len(items):,} "
              f"-- re-encoding")

    if not ALLOW_INLINE_ENCODE:
        # Encoding here holds df_features and the 4.3 GB embeddings pickle in
        # memory alongside SBERT. Measured on 16 GB, the batches stalled for
        # minutes at a time and the run reached 14% in an hour and a half.
        # The standalone script loads two columns and nothing else.
        print("  no cache, and ALLOW_INLINE_ENCODE is False — skipping.\n"
              "  Build it once in a lean process, then re-run this cell:\n"
              "      python ttn/encode_descriptions.py\n"
              "  (§10 detects the absence and trains without the block.)")
        return None

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text = desc.str.slice(0, 1200).tolist()
    out = np.asarray(encoder.encode(text, batch_size=64,
                                    show_progress_bar=True)).astype("float32")
    out[blank] = 0.0                       # a blank description says nothing
    np.save(DESC_CACHE, out)
    print(f"  encoded and cached -> {DESC_CACHE.name} {out.shape}")
    return out


# --------------------------------------------------------------------------
items = build_items(tower_pairs_train, tower_pairs_test)
vocabs = fit_vocabs(tower_pairs_train)               # TRAIN ONLY
print(f"items: {len(items):,} unique asins across both slices")
# The item order, so `ttn/encode_descriptions.py` can build desc_emb.npy in a
# lean process instead of inside this cell.
np.save(OUT_DIR / "item_asins.npy", items["asin"].to_numpy().astype("U16"))

title_emb = item_title_embeddings(items)
arrays = encode_items(items, vocabs, title_emb)
if USE_DESCRIPTION:
    _desc = item_description_embeddings(items)
    if _desc is not None:
        arrays["desc_emb"] = _desc

pairs_train = slim_pairs(tower_pairs_train, items, vocabs)
pairs_test = slim_pairs(tower_pairs_test, items, vocabs)

# FIX: a node is a function of the item's own path, so derive it for every
# item. Scattering from the training pairs left query-only items at 0.
item_node = (items["cat_2"].astype(str) + NODE_SEP + items["cat_3"].astype(str)
             + NODE_SEP + items["cat_4"].astype(str))
node_of_item = item_node.map(vocabs["target_node"]).fillna(0).astype(np.int64).to_numpy()

np.savez_compressed(OUT_DIR / "items.npz", **arrays)
json.dump(vocabs, open(OUT_DIR / "vocabs.json", "w"))
pairs_train.to_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test.to_parquet(OUT_DIR / "pairs_test.parquet")
np.save(OUT_DIR / "node_of_item.npy", node_of_item)

# Id 0 is the reserved unseen/padding index, and it exists for exactly one
# case: an item that appears only in test carrying a value the train-fitted
# vocabulary never saw. A TRAINING item landing there would be a real defect,
# since the vocabulary was fitted on those very rows.
train_items = np.unique(pairs_train[["query_idx", "target_idx"]].to_numpy())
test_items = np.unique(pairs_test[["query_idx", "target_idx"]].to_numpy())
assert (arrays["cat_ids"][train_items] > 0).all(), \
    "a training item fell on the reserved id 0 — the vocabulary is inconsistent"
assert (pairs_train["target_node_id"] > 0).all(), "a train node fell on id 0"

test_only = np.setdiff1d(test_items, train_items)
zero_slots = int((arrays["cat_ids"][test_only] == 0).sum())
unseen_nodes = int((pairs_test["target_node_id"] == 0).sum())

print(f"\nitems {len(items):,} | train {len(pairs_train):,} | test {len(pairs_test):,}")
print(f"title_emb {arrays['title_emb'].shape} | cat_ids {arrays['cat_ids'].shape} "
      f"| numeric {arrays['numeric'].shape}"
      + (f" | desc_emb {arrays['desc_emb'].shape}" if "desc_emb" in arrays else ""))
print(f"vocab sizes: {({k: len(v) for k, v in vocabs.items()})}")
print(f"test-only items                              : {len(test_only):,}")
print(f"  their categorical slots on the reserved id 0: {zero_slots:,} of "
      f"{len(test_only) * len(CAT_ORDER):,}")
print(f"items with no node in the training vocabulary : {(node_of_item == 0).sum():,}")
print(f"test pairs whose target node is unseen       : {unseen_nodes:,}")
print(f"\nwritten to {OUT_DIR.relative_to(ROOT)}/")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name:<24} {f.stat().st_size / 1e6:>8,.1f} MB")
