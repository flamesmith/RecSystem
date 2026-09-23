"""Build TTN-specific model-ready arrays, from the shared snapshot's cleaned
pair tables.

Lives in data_processing/, not TTN/ -- even though only TTN consumes this
script's output, *building* it (fitting vocabularies, encoding arrays) is a
data-processing concern, not a training concern, so it sits alongside
build_snapshot.py rather than inside the model's own folder. TTN/ itself now
only has genuinely model-specific files: encode_descriptions.py,
build_model.py, generate_recommendations.py.

This script picks up from data_processing/build_snapshot.py's shared
snapshot and does only what's actually TTN-specific: fitting the model's
own integer-coded embedding-table vocabularies, encoding item attributes
into arrays, and slimming the pair tables down to the (query_idx,
target_idx, target_node_id) form BPR training reads.

Produces, under data/tower/<snapshot_id>/ (alongside build_snapshot.py's
shared item_asins.npy / node_of_item.npy, untouched by this script):
  items.npz              -- title_emb, cat_ids, numeric[, desc_emb once
                             encode_descriptions.py has run]
  vocabs.json             -- integer-coded embedding-table vocabularies
                             (different from build_snapshot.py's categorical
                             dtype vocabularies -- these are what
                             nn.Embedding sizes itself against)
  pairs_{train,test}.parquet
                          -- slim (query_idx, target_idx, target_node_id,
                             weight) training/eval pairs

Prerequisite: data_processing/build_snapshot.py must have already run for
the given --snapshot, to produce tower_pairs_{train,test}.parquet and
item_asins.npy.

Usage:
  python data_processing/build_snapshot.py --window-days 90
  python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
  python TTN/encode_descriptions.py --snapshot w90_2017-12-09   # optional
                                        # but recommended -- see its own
                                        # docstring; adds description signal
                                        # to the model trained next
  python TTN/build_model.py --snapshot w90_2017-12-09
"""

# ============================================================================
# Setup
# ============================================================================
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from build_snapshot.py, e.g. w90_2017-12-09")
SNAPSHOT_ID = parser.parse_args().snapshot
OUT_DIR = DATA_DIR / "tower" / SNAPSHOT_ID

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

NODE_SEP = " > "
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]
# column in the wide pair tables (without the query_/target_ prefix) -> canonical name
COLMAP = {"cat_2": "cat_2", "cat_3": "cat_3", "cat_4": "cat_4",
          "brand_clean": "brand", "color": "color", "material": "material",
          "product_type": "product_type", "features": "features"}

tower_pairs_train = pd.read_parquet(OUT_DIR / "tower_pairs_train.parquet")
tower_pairs_test = pd.read_parquet(OUT_DIR / "tower_pairs_test.parquet")
print(f"loaded {OUT_DIR.relative_to(ROOT)}/tower_pairs_{{train,test}}.parquet -- "
      f"train {len(tower_pairs_train):,} | test {len(tower_pairs_test):,}")

# ============================================================================
# Export the model-ready arrays
# ============================================================================
ENCODE_TITLES = False          # True -> run SBERT instead of reusing the pickle
USE_DESCRIPTION = True         # encode `description_cleaned` alongside the title
DESC_CACHE = OUT_DIR / "desc_emb.npy"   # SBERT over 137k rows is slow; cache it
ALLOW_INLINE_ENCODE = False    # True -> encode here if the cache is missing.
                               # Leave False and use TTN/encode_descriptions.py:
                               # encoding inside this cell competes for memory
                               # with df_features and stalls badly.


def _side(df, prefix):
    cols = {f"{prefix}_{src}": dst for src, dst in COLMAP.items()}
    cols[f"asin_{prefix}"] = "asin"
    cols[f"{prefix}_title_cleaned"] = "title"
    cols[f"{prefix}_price"] = "price"
    cols[f"{prefix}_price_imputed"] = "price_imputed"
    return df[list(cols)].rename(columns=cols)


def build_items(train, test):
    """One row per asin, with every TTN-specific attribute -- superset of
    build_snapshot.py's cat_2/3/4-only item table, same union of asins
    (both are derived from the same tower_pairs_{train,test})."""
    frames = [_side(d, p) for d in (train, test) for p in ("query", "target")]
    items = (pd.concat(frames, ignore_index=True)
               .drop_duplicates("asin").reset_index(drop=True))
    items["idx"] = np.arange(len(items))
    return items


def fit_vocabs(train):
    """FIT ON TRAIN ONLY. Id 0 is reserved for padding / unseen. These are
    the model's own integer-coded embedding-table vocabularies -- distinct
    from build_snapshot.py's categorical dtype vocabularies, which only
    exist to keep train/test casting consistent upstream."""
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

    price = items["price"].astype(float)
    imputed = items["price_imputed"].astype("float32")
    price = price.fillna(price.median())          # defensive; build_snapshot.py leaves none

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
    """SBERT over `description_cleaned`, cached to disk. See
    TTN/encode_descriptions.py's docstring for the reasoning behind caching
    this in a lean process rather than inline here."""
    df_features = pd.read_pickle(DATA_DIR / "df_features.pkl")[["asin", "description_cleaned"]]
    desc = (df_features.drop_duplicates("asin").set_index("asin")["description_cleaned"]
            .reindex(items["asin"]).fillna("").astype(str))
    blank = (desc.str.strip() == "").to_numpy()
    print(f"descriptions: {int((~blank).sum()):,} non-empty, {int(blank.sum()):,} blank (left as zeros)")

    if DESC_CACHE.exists():
        cached = np.load(DESC_CACHE)
        if len(cached) == len(items):
            print(f"  reusing {DESC_CACHE.name} {cached.shape}")
            return cached.astype("float32")
        print(f"  {DESC_CACHE.name} has {len(cached):,} rows, need {len(items):,} -- re-encoding")

    if not ALLOW_INLINE_ENCODE:
        print("  no cache, and ALLOW_INLINE_ENCODE is False — skipping.\n"
              "  Build it once in a lean process, then re-run this script:\n"
              "      python TTN/encode_descriptions.py --snapshot " + SNAPSHOT_ID + "\n"
              "  (build_model.py detects the absence and trains without the block.)")
        return None

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text = desc.str.slice(0, 1200).tolist()
    out = np.asarray(encoder.encode(text, batch_size=64,
                                    show_progress_bar=True)).astype("float32")
    out[blank] = 0.0
    np.save(DESC_CACHE, out)
    print(f"  encoded and cached -> {DESC_CACHE.name} {out.shape}")
    return out


# --------------------------------------------------------------------------
items = build_items(tower_pairs_train, tower_pairs_test)
vocabs = fit_vocabs(tower_pairs_train)               # TRAIN ONLY
print(f"items: {len(items):,} unique asins across both slices")

title_emb = item_title_embeddings(items)
arrays = encode_items(items, vocabs, title_emb)
if USE_DESCRIPTION:
    _desc = item_description_embeddings(items)
    if _desc is not None:
        arrays["desc_emb"] = _desc

pairs_train = slim_pairs(tower_pairs_train, items, vocabs)
pairs_test = slim_pairs(tower_pairs_test, items, vocabs)

np.savez_compressed(OUT_DIR / "items.npz", **arrays)
json.dump(vocabs, open(OUT_DIR / "vocabs.json", "w"))
pairs_train.to_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test.to_parquet(OUT_DIR / "pairs_test.parquet")

# Id 0 is the reserved unseen/padding index, and it exists for exactly one
# case: an item that appears only in test carrying a value the train-fitted
# vocabulary never saw. A TRAINING item landing there would be a real defect.
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
print(f"test-only items: {len(test_only):,} | their zeroed categorical slots: {zero_slots:,} of "
      f"{len(test_only) * len(CAT_ORDER):,}")
print(f"test pairs whose target node is unseen: {unseen_nodes:,}")
print(f"\nwritten to {OUT_DIR.relative_to(ROOT)}/")
for f in sorted(OUT_DIR.glob("*.npz")) + sorted(OUT_DIR.glob("vocabs.json")) + \
         sorted(OUT_DIR.glob("pairs_*.parquet")):
    print(f"  {f.name:<24} {f.stat().st_size / 1e6:>8,.1f} MB")
