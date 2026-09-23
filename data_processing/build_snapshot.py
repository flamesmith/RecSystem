"""Build the SHARED data snapshot every model reads from -- lives in
data_processing/, not inside any one model's folder, since TTN, SigLIP2,
and Popularity all consume its output.

Parameterized by --window-days, the co-purchase pairing window (how many
days apart two purchases can be and still count as "bought together").
Different values produce distinct, coexisting snapshots
(data/tower/w{window_days}_{date_threshold}/) instead of overwriting each
other.

Produces, under data/tower/<snapshot_id>/:
  item_asins.npy       -- the catalogue this snapshot covers, in a fixed
                           order. Read by ALL THREE models (TTN, SigLIP2,
                           Popularity).
  node_of_item.npy     -- each item's OWN category, encoded against the
                           target_node vocabulary fitted here (train only).
                           Read by ALL THREE models -- this is what "same
                           category" means throughout this project.
  tower_pairs_{train,test}.parquet
                        -- the fully cleaned, joined, category-mapping-
                           licensed co-purchase pairs (wide: every query_*/
                           target_* attribute column, target_node assigned).
                           TTN-only handoff -- SigLIP2 and Popularity never
                           read this, only data_processing/build_ttn_arrays.py
                           does, to avoid redoing these joins for a concern
                           that's actually shared infrastructure (the
                           items/nodes above) vs. a concern that's genuinely
                           TTN-only (the pairs themselves, and everything
                           encoded from them).
  snapshot_manifest.json -- window_days, date_threshold, git commit, row
                           counts at each stage of this funnel.

What's NOT here, because it's TTN-only, not shared: items.npz (categorical
IDs, numeric features, title embeddings), vocabs.json (the model's own
integer-coded embedding-table vocabularies), pairs_{train,test}.parquet
(the SLIM query_idx/target_idx encoding used for training) -- all built by
data_processing/build_ttn_arrays.py from this script's tower_pairs_*.parquet
output.

Prerequisites (built by data_creation/feature_extraction_workflow/,
data_creation/embedding_analysis/, and
data_creation/complementary_cats_pairs/categories.* -- see their own
READMEs; none of these are window_days-specific, so they're shared across
every snapshot):
  data/Home_and_Kitchen_filtered.csv
  data/df_features.pkl
  data/complementary_categories.pkl
  data/meta_Home_and_Kitchen_filtered.csv
  data/category_taxonomy.json, data/master_metadata.json
  TTN/constants.json (date_threshold)

Usage:
  python data_processing/build_snapshot.py --window-days 90   # -> data/tower/w90_2017-12-09/
  python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
"""

# ============================================================================
# Setup
# ============================================================================
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_creation.complementary_cats_pairs import DST_COLS, SRC_COLS, co_purchase_pairs, fold_cat_4, parse_category_levels

parser = argparse.ArgumentParser()
parser.add_argument("--window-days", type=int, default=90,
                     help="max gap, in days, between two purchases of a "
                          "co-purchase pair (default 90, matching the "
                          "value data_creation/complementary_cats_pairs/pairs.ipynb used "
                          "before this became a CLI parameter)")
WINDOW_DAYS = parser.parse_args().window_days

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

MISSING = "Missing"
OTHER_SUFFIX = "_Other"
NODE_SEP = " > "

# ============================================================================
# 1. Load the datasets
# ============================================================================
df_reviews = pd.read_csv(
    DATA_DIR / "Home_and_Kitchen_filtered.csv",
    dtype={"asin": str, "reviewerID": str},
    low_memory=False,
)
df_features = pd.read_pickle(DATA_DIR / "df_features.pkl")

DATE_THRESHOLD = json.loads((ROOT / "TTN" / "constants.json").read_text())["date_threshold"]
cutoff_time = pd.Timestamp(DATE_THRESHOLD).timestamp()
SNAPSHOT_ID = f"w{WINDOW_DAYS}_{DATE_THRESHOLD}"
print(f"snapshot: {SNAPSHOT_ID}  (window_days={WINDOW_DAYS}, date_threshold={DATE_THRESHOLD})")

after_cutoff = df_reviews[df_reviews["unixReviewTime"] >= cutoff_time]
co_pairs = {
    "train": co_purchase_pairs(df_reviews, cutoff_time=cutoff_time, window_days=WINDOW_DAYS),
    "test": co_purchase_pairs(after_cutoff, cutoff_time=None, window_days=WINDOW_DAYS),
}
print(f"co-purchase pairs: train {len(co_pairs['train']):,} | test {len(co_pairs['test']):,}")

comp_cat = pd.read_pickle(DATA_DIR / "complementary_categories.pkl")
meta_catalogue = pd.read_csv(
    DATA_DIR / "meta_Home_and_Kitchen_filtered.csv",
    usecols=["asin", "category", "title", "brand", "price"],
    dtype={"asin": str},
    low_memory=False,
)
print(f"df_features {df_features.shape} | comp_cat {comp_cat.shape} | "
      f"meta_catalogue {meta_catalogue.shape}")

# ============================================================================
# 2. Clean the item category path
# ============================================================================
taxonomy = json.loads((DATA_DIR / "category_taxonomy.json").read_text())
valid_cat_4 = {c3: set(vals) for c2 in taxonomy for c3, vals in taxonomy[c2].items()}
print(f"taxonomy: {len(taxonomy)} cat_2 | {len(valid_cat_4)} cat_3 | "
      f"{sum(len(v) for v in valid_cat_4.values())} valid cat_4 slots")

cat_3 = df_features["cat_3"].astype(str)
cat_4 = df_features["cat_4"].fillna(MISSING).astype(str)
valid_pairs = {(c3, v) for c3, vals in valid_cat_4.items() for v in vals}
keep = pd.Series(list(zip(cat_3, cat_4)), index=df_features.index).isin(valid_pairs)
df_features["cat_4_clean"] = np.where(keep, cat_4, cat_3 + OTHER_SUFFIX)
n_folded = int((~keep).sum())
print(f"items folded into '<cat_3>{OTHER_SUFFIX}': {n_folded:,} "
      f"({n_folded / len(df_features) * 100:.2f}% of the catalog)")

# ============================================================================
# 3. Build the pair tables -- licensed by the complementary-category mapping
# ============================================================================
CAT_LEVELS = ["cat_2", "cat_3", "cat_4_clean"]
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

# ============================================================================
# 4. Fold rare brands -- fitted on train
# ============================================================================
MIN_ITEMS = 10
OTHER = "other_brands"

train_items = pd.unique(pd.concat(
    [slices["train"]["asin_query"], slices["train"]["asin_target"]], ignore_index=True))
source = "brand_norm" if "brand_norm" in df_features.columns else "brand"
train_features = df_features[df_features["asin"].isin(train_items)]
brand_counts = train_features.groupby(source)["asin"].nunique()
kept_brands = brand_counts[brand_counts > MIN_ITEMS].index
df_features["brand_clean"] = df_features[source].where(
    df_features[source].isin(kept_brands) | df_features[source].isna(), OTHER)
print(f"brands kept: {len(kept_brands):,} of {train_features[source].nunique():,} seen in training")

# ============================================================================
# 5. Attach the item attributes, and fit the categorical vocabularies
# ============================================================================
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
    item_attrs["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
attrs_by_asin = item_attrs.set_index("asin")


def attach_attributes(frame):
    for side in ("query", "target"):
        asin = frame[f"asin_{side}"]
        for attr in ITEM_ATTRS.values():
            frame[f"{side}_{attr}"] = asin.map(attrs_by_asin[attr])
    return frame[[c.format(s=s) for s in ("query", "target") for c in SIDE_COLS]]


for name in slices:
    slices[name] = attach_attributes(slices[name])

vocabularies = {}
for attr in CATEGORICAL_ATTRS:
    values = set()
    for side in ("query", "target"):
        values |= set(slices["train"][f"{side}_{attr}"].dropna().astype(str))
    vocabularies[attr] = pd.CategoricalDtype(sorted(values))
print(f"categorical vocabularies fitted on {len(slices['train']):,} training rows")

# ============================================================================
# 6. Fill the missing values -- fitted on train
# ============================================================================
train_reviewed = set(df_reviews.loc[df_reviews["unixReviewTime"] < cutoff_time, "asin"])
cat_levels = parse_category_levels(meta_catalogue["category"], n_levels=4)
catalogue_prices = pd.DataFrame({
    "asin": meta_catalogue["asin"],
    "cat_2": cat_levels["cat_2"].astype(str),
    "cat_3": cat_levels["cat_3"].astype(str),
    "cat_4": fold_cat_4(cat_levels["cat_3"], cat_levels["cat_4"], valid_pairs),
    "price": pd.to_numeric(
        meta_catalogue["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"),
})
fit_prices = catalogue_prices[catalogue_prices["asin"].isin(train_reviewed)]
median_4 = fit_prices.groupby(["cat_2", "cat_3", "cat_4"], observed=True)["price"].median()
median_3 = fit_prices.groupby(["cat_2", "cat_3"], observed=True)["price"].median()
median_2 = fit_prices.groupby(["cat_2"], observed=True)["price"].median()
median_all = fit_prices["price"].median()


def fill_prices(frame):
    for side in ("query", "target"):
        raw = frame[f"asin_{side}"].map(attrs_by_asin["price"])
        c2 = frame[f"{side}_cat_2"].astype(str)
        c3 = frame[f"{side}_cat_3"].astype(str)
        c4 = frame[f"{side}_cat_4"].astype(str)
        at_4 = pd.Series(median_4.reindex(pd.MultiIndex.from_arrays([c2, c3, c4])).to_numpy(), index=frame.index)
        at_3 = pd.Series(median_3.reindex(pd.MultiIndex.from_arrays([c2, c3])).to_numpy(), index=frame.index)
        at_2 = pd.Series(median_2.reindex(pd.Index(c2)).to_numpy(), index=frame.index)
        missing = raw.isna()
        frame[f"{side}_price"] = raw.fillna(at_4.fillna(at_3).fillna(at_2).fillna(median_all))
        frame[f"{side}_price_imputed"] = missing


def apply_vocabularies(frame):
    unseen = 0
    for attr, dtype in vocabularies.items():
        for side in ("query", "target"):
            col = f"{side}_{attr}"
            values = frame[col].astype(object)
            cast = pd.Categorical(values, dtype=dtype)
            unseen += int((values.notna() & pd.isna(cast)).sum())
            frame[col] = cast
    return unseen


for name, frame in slices.items():
    fill_prices(frame)
    unseen = apply_vocabularies(frame)
    print(f"{name:<6} values outside the training vocabulary: {unseen:,}")

# ============================================================================
# 6b. Fill missing categoricals
# ============================================================================
def fill_missing_categories(frame, attrs, label=MISSING):
    for attr in attrs:
        for side in ("query", "target"):
            col = f"{side}_{attr}"
            series = frame[col]
            if label not in series.cat.categories:
                series = series.cat.add_categories([label])
            frame[col] = series.fillna(label)
    return frame


for name, frame in slices.items():
    slices[name] = fill_missing_categories(frame, CATEGORICAL_ATTRS)

# ============================================================================
# 7. target_node -- the target category as one symbol, fitted on train
# ============================================================================
def target_node(frame):
    return (frame["target_cat_2"].astype(str) + NODE_SEP
            + frame["target_cat_3"].astype(str) + NODE_SEP
            + frame["target_cat_4"].astype(str))


node_vocabulary = pd.CategoricalDtype(sorted(set(target_node(slices["train"]))))
print(f"target_node vocabulary: {len(node_vocabulary.categories):,} paths, "
      f"fitted on {len(slices['train']):,} training rows")

for name, frame in slices.items():
    nodes = target_node(frame)
    cast = pd.Categorical(nodes, dtype=node_vocabulary)
    unseen = int(pd.isna(cast).sum())
    if unseen:
        cast = cast.add_categories([MISSING]).fillna(MISSING)
    frame["target_node"] = cast
    print(f"  {name:<6} {nodes.nunique():>4,} distinct | outside training vocabulary: {unseen:,}")

# ============================================================================
# 7b. Drop same-category pairs -- substitutes, not complements
# ============================================================================
def drop_same_category(frame):
    same = ((frame["query_cat_4"] == frame["target_cat_4"]) & (frame["query_cat_4"] != MISSING))
    return frame.loc[~same].reset_index(drop=True), int(same.sum())


for name in slices:
    before = len(slices[name])
    slices[name], removed = drop_same_category(slices[name])
    print(f"{name:<6} {before:>10,} -> {len(slices[name]):>10,} rows (dropped {removed:,}, {removed / before:.1%})")

tower_pairs_train = slices["train"]
tower_pairs_test = slices["test"]

# ============================================================================
# 8. The shared exports -- item list, each item's OWN category, and the
#    wide pair tables build_ttn_arrays.py picks up from
# ============================================================================
OUT_DIR = DATA_DIR / "tower" / SNAPSHOT_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_items(train, test):
    """One row per asin, cat_2/cat_3/cat_4 only -- enough to place every
    item in the shared node/category scheme. Full attribute extraction
    (title, price, brand, color, ...) is TTN-specific and lives in
    build_ttn_arrays.py instead."""
    frames = []
    for df in (train, test):
        for side in ("query", "target"):
            frames.append(pd.DataFrame({
                "asin": df[f"asin_{side}"],
                "cat_2": df[f"{side}_cat_2"].astype(str),
                "cat_3": df[f"{side}_cat_3"].astype(str),
                "cat_4": df[f"{side}_cat_4"].astype(str),
            }))
    return pd.concat(frames, ignore_index=True).drop_duplicates("asin").reset_index(drop=True)


items = build_items(tower_pairs_train, tower_pairs_test)
print(f"\nitems: {len(items):,} unique asins across both slices")

node_vocab_ids = {v: i + 1 for i, v in enumerate(sorted(target_node(tower_pairs_train).astype(str).unique()))}
item_node = items["cat_2"] + NODE_SEP + items["cat_3"] + NODE_SEP + items["cat_4"]
node_of_item = item_node.map(node_vocab_ids).fillna(0).astype(np.int64).to_numpy()

np.save(OUT_DIR / "item_asins.npy", items["asin"].to_numpy().astype("U16"))
np.save(OUT_DIR / "node_of_item.npy", node_of_item)
tower_pairs_train.to_parquet(OUT_DIR / "tower_pairs_train.parquet")
tower_pairs_test.to_parquet(OUT_DIR / "tower_pairs_test.parquet")

try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_commit = None
manifest = {
    "snapshot_id": SNAPSHOT_ID,
    "window_days": WINDOW_DAYS,
    "date_threshold": DATE_THRESHOLD,
    "git_commit": git_commit,
    "n_items": int(len(items)),
    "n_tower_pairs_train": int(len(tower_pairs_train)),
    "n_tower_pairs_test": int(len(tower_pairs_test)),
    "co_purchase_pairs_train": int(len(co_pairs["train"])),
    "co_purchase_pairs_test": int(len(co_pairs["test"])),
    "items_with_no_node": int((node_of_item == 0).sum()),
}
json.dump(manifest, open(OUT_DIR / "snapshot_manifest.json", "w"), indent=1)

print(f"\nwritten to {OUT_DIR.relative_to(ROOT)}/")
for f in sorted(OUT_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name:<28} {f.stat().st_size / 1e6:>8,.1f} MB")
