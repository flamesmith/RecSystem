"""Complementary recommendations from IMAGE similarity only.

Same question TTN answers -- "given this item, which items from its
complementary categories go with it?" -- but the ranking signal is how
visually similar a candidate's picture is to the query's picture, not a
trained two-tower score. For a coffee table: the licensed target categories
(TV stands, bookcases, dining tables, ...) come from the same
complementary-category mapping TTN uses, and inside each of them the
candidates are ordered by SigLIP2 image similarity to the table.

Decisions this script implements (each one chosen explicitly, none assumed):
  - Items: every item with a real image embedding in SigLIP2/image_cache/
    (the full cache, not any one --snapshot), as both query and candidate.
  - Licensed categories: data/complementary_categories.pkl, at its own
    cat_2 > cat_3 > cat_4 grain. The query's OWN category is excluded as a
    target (the mapping pairs a category with itself; TTN drops those too).
  - Embeddings: the champion adapter named in
    SigLIP2/champion_selection_siglip.json applied to the raw cached vectors
    (same adapter/same step as SigLIP2/generate_full_catalog_adapted_recommendations.py).
  - Slot split: TOP_K slots are split across a query's licensed categories in
    proportion to each category's co-purchase evidence (`edges`), by
    largest-remainder apportionment -- the same rule as
    TTN/generate_recommendations_proportional.py -- and each category's slots
    are filled by cosine similarity WITHIN that category only.
  - Unfilled slots: a licensed category with fewer candidates than its share
    (or none with an image) can't use all its slots. Those slots are
    re-split among the remaining categories by edge share, repeatedly, until
    TOP_K are placed or every candidate is used. TTN does NOT do this (it
    loses them, so its lists can be shorter than K) -- that is the one
    deliberate difference in how slots are allocated.

Rank order inside one query's list follows TTN's weighted file: categories in
descending order of co-purchase evidence, and inside each category by image
similarity, best first. `rank` is therefore NOT a single similarity ordering
across categories -- scores are only comparable within one `target_category`.

An item is a query only if its own cat_2 > cat_3 > cat_4 is a source category
in the mapping and at least one of that category's licensed targets has an
image-embedded item.

Output: SigLip2 complementary/recommendations/complements_top25.parquet --
query_asin | rank | candidate_asin | score | target_category

Prerequisite: SigLIP2/encode_siglip2_images.py and/or
SigLIP2/encode_images_by_category.py must already have populated
SigLIP2/image_cache/, and data/complementary_categories.pkl and
data/df_features.pkl must exist.

Usage: python "SigLip2 complementary/generate_complementary_recommendations.py"
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

TOP_K = 25
ADAPT_BATCH = 8192      # items per adapt_image() call
QUERY_BATCH = 2048      # queries per scoring batch, within one (source, target) pair
SEP = "\x1f"            # joins cat_2/cat_3/cat_4 into one key (cannot occur in a category name)
LABEL_SEP = " > "

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SigLIP2.adapter import load_adapter, adapt_image

CACHE_DIR = ROOT / "SigLIP2" / "image_cache"
CHAMPION_FILE = ROOT / "SigLIP2" / "champion_selection_siglip.json"
OUT_DIR = Path(__file__).resolve().parent / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "complements_top25.parquet"

t0 = time.time()

# ============================================================================
# 1. Load the shared image cache, keep only real (status==1) embeddings
# ============================================================================
cache_asins = np.load(CACHE_DIR / "cache_asins.npy", allow_pickle=False).astype(str)
cache_emb = np.load(CACHE_DIR / "cache_emb.npy").astype("float32")
cache_status = np.load(CACHE_DIR / "cache_status.npy")
valid = cache_status == 1
asins = cache_asins[valid]
raw_emb = cache_emb[valid]
del cache_emb
n_items = len(asins)
print(f"cache: {len(cache_asins):,} tracked, {n_items:,} with a real embedding")

# ============================================================================
# 2. Apply the champion adapter -- image side only
# ============================================================================
champion = json.loads(CHAMPION_FILE.read_text())
checkpoint_path = ROOT / champion["checkpoint_path"]
assert checkpoint_path.exists(), f"missing: {checkpoint_path}"
print(f"champion checkpoint: {checkpoint_path.relative_to(ROOT)}")

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
model = load_adapter(checkpoint_path=checkpoint_path, device=DEVICE)
print(f"adapter loaded on {DEVICE}")

emb = np.empty_like(raw_emb)
for i in range(0, n_items, ADAPT_BATCH):
    j = min(i + ADAPT_BATCH, n_items)
    emb[i:j] = adapt_image(model, raw_emb[i:j])
del model, raw_emb
# L2-normalize once; every score below is then a plain dot product (= cosine)
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
print(f"adapted {n_items:,} embeddings [{time.time() - t0:.0f}s]")

# ============================================================================
# 3. Each item's own category -- from df_features.pkl, the same convention
# Popularity/build_popularity.py and the full-catalog SigLIP2 script use
# ============================================================================
df_features = pd.read_pickle(ROOT / "data" / "df_features.pkl")
cat_by_asin = (df_features[["asin", "cat_2", "cat_3", "cat_4"]]
               .drop_duplicates("asin").set_index("asin"))
cat_by_asin["cat_4"] = cat_by_asin["cat_4"].fillna("Missing")
del df_features

cats = cat_by_asin.reindex(asins)
have_cat = cats["cat_2"].notna().to_numpy()
print(f"items with a category: {have_cat.sum():,} of {n_items:,}")
path_key = (cats["cat_2"].astype(str) + SEP + cats["cat_3"].astype(str)
            + SEP + cats["cat_4"].astype(str)).to_numpy()

# category key -> indices (into asins/emb) of the embedded items in it
idx_with_cat = np.where(have_cat)[0]
order = idx_with_cat[np.argsort(path_key[idx_with_cat], kind="stable")]
unique_keys, starts = np.unique(path_key[order], return_index=True)
ends = np.append(starts[1:], len(order))
items_of_path = {k: order[s:e] for k, s, e in zip(unique_keys, starts, ends)}
print(f"categories with >=1 embedded item: {len(items_of_path):,}")

# ============================================================================
# 4. Licensed target categories per source category, with co-purchase evidence
# ============================================================================
comp_cat = pd.read_pickle(ROOT / "data" / "complementary_categories.pkl")
for c in ("src_cat_2", "src_cat_3", "src_cat_4", "dst_cat_2", "dst_cat_3", "dst_cat_4"):
    comp_cat[c] = comp_cat[c].astype(str)
comp_cat["src_key"] = comp_cat["src_cat_2"] + SEP + comp_cat["src_cat_3"] + SEP + comp_cat["src_cat_4"]
comp_cat["dst_key"] = comp_cat["dst_cat_2"] + SEP + comp_cat["dst_cat_3"] + SEP + comp_cat["dst_cat_4"]
n_pairs_total = len(comp_cat)
comp_cat = comp_cat[comp_cat["src_key"] != comp_cat["dst_key"]]     # own category is not a target
print(f"mapping: {n_pairs_total:,} directed category pairs, {len(comp_cat):,} after dropping "
      f"{n_pairs_total - len(comp_cat):,} self pairs")

licensed = {src: list(zip(grp["dst_key"], grp["edges"].astype(float)))
            for src, grp in comp_cat.sort_values(["edges", "dst_key"], ascending=[False, True])
                                    .groupby("src_key", sort=False)}


def allocate_slots(shares, k):
    """Largest-remainder apportionment: k slots split by share, summing to k.

    Same rule as TTN/generate_recommendations_proportional.py's allocate_slots.
    """
    raw = np.asarray(shares, dtype=float) * k
    base = np.floor(raw).astype(int)
    remaining = k - base.sum()
    if remaining > 0:
        for j in np.argsort(-(raw - base))[:remaining]:
            base[j] += 1
    return base


def allocate_with_redistribution(edges, capacity, k):
    """Split k slots across targets by edge share, never giving a target more
    slots than it has candidates; slots a target cannot take are re-split among
    the targets that still have room, by edge share, until k slots are placed or
    every candidate is used. Reduces to allocate_slots when nothing is capped."""
    edges = np.asarray(edges, dtype=float)
    capacity = np.asarray(capacity, dtype=int)
    slots = np.zeros(len(edges), dtype=int)
    while slots.sum() < k:
        active = slots < capacity
        if not active.any():
            break
        want = allocate_slots(edges[active] / edges[active].sum(), k - slots.sum())
        take = np.minimum(want, capacity[active] - slots[active])
        if take.sum() == 0:
            break
        slots[active] += take
    return slots


# ============================================================================
# 5. Score each source category's items against each licensed target
# category's items, in query batches -- never a full item x item matrix.
# Slots per target depend only on the source category, so they are computed
# once per source category.
# ============================================================================
sources = sorted(set(licensed) & set(items_of_path))
print(f"source categories in the mapping: {len(licensed):,} | of those with >=1 embedded item: {len(sources):,}")

target_code = {}            # dst_key -> small int, becomes the target_category label
q_parts, c_parts, s_parts, blk_parts, win_parts, tgt_parts = [], [], [], [], [], []
n_sources_no_target = 0
t1 = time.time()
for gi, src in enumerate(sources):
    queries = items_of_path[src]
    targets = [(dst, e) for dst, e in licensed[src] if dst in items_of_path]
    if not targets:
        n_sources_no_target += 1
        continue
    edges = np.array([e for _, e in targets])
    capacity = np.array([len(items_of_path[dst]) for dst, _ in targets])
    slots = allocate_with_redistribution(edges, capacity, TOP_K)

    for block, ((dst, _), n_slots) in enumerate(zip(targets, slots)):
        if n_slots == 0:
            continue
        k = int(n_slots)
        cand = items_of_path[dst]
        C = emb[cand]
        code = target_code.setdefault(dst, len(target_code))

        for b in range(0, len(queries), QUERY_BATCH):
            batch = queries[b:b + QUERY_BATCH]
            scores = emb[batch] @ C.T                          # (batch, n_cand), cosine

            if k < scores.shape[1]:
                top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            else:
                top_idx = np.tile(np.arange(scores.shape[1]), (len(batch), 1))
            top_scores = np.take_along_axis(scores, top_idx, axis=1)
            order_k = np.argsort(-top_scores, axis=1)
            top_idx = np.take_along_axis(top_idx, order_k, axis=1)
            top_scores = np.take_along_axis(top_scores, order_k, axis=1)

            n_rows = len(batch) * k
            q_parts.append(np.repeat(batch, k))
            c_parts.append(cand[top_idx].ravel())
            s_parts.append(top_scores.ravel().astype("float32"))
            blk_parts.append(np.full(n_rows, block, dtype=np.int32))
            win_parts.append(np.tile(np.arange(k, dtype=np.int32), len(batch)))
            tgt_parts.append(np.full(n_rows, code, dtype=np.int32))

    if (gi + 1) % 25 == 0 or gi + 1 == len(sources):
        print(f"  source category {gi + 1}/{len(sources)}  [{time.time() - t1:.0f}s]", flush=True)

# ============================================================================
# 6. Order each query's rows (categories by evidence, then by similarity),
# assign rank, write.
# ============================================================================
q_all = np.concatenate(q_parts)
c_all = np.concatenate(c_parts)
s_all = np.concatenate(s_parts)
blk_all = np.concatenate(blk_parts)
win_all = np.concatenate(win_parts)
tgt_all = np.concatenate(tgt_parts)
del q_parts, c_parts, s_parts, blk_parts, win_parts, tgt_parts

# primary key query, then category block (descending evidence), then position within the block
o = np.lexsort((win_all, blk_all, q_all))
q_all, c_all, s_all, tgt_all = q_all[o], c_all[o], s_all[o], tgt_all[o]
del o, blk_all, win_all

group_start = np.r_[0, np.flatnonzero(np.diff(q_all)) + 1]
group_len = np.diff(np.r_[group_start, len(q_all)])
rank = np.arange(len(q_all)) - np.repeat(group_start, group_len) + 1

target_labels = [None] * len(target_code)
for dst, code in target_code.items():
    target_labels[code] = dst.replace(SEP, LABEL_SEP)

asin_arr = pa.array(asins, type=pa.large_string())
table = pa.table({
    "query_asin": asin_arr.take(pa.array(q_all)),
    "rank": pa.array(rank, type=pa.int64()),
    "candidate_asin": asin_arr.take(pa.array(c_all)),
    "score": pa.array(s_all, type=pa.float32()),
    "target_category": pa.array(target_labels, type=pa.large_string()).take(pa.array(tgt_all)),
})
pq.write_table(table, OUT_PATH)

n_queries = len(group_start)
print(f"\nsource categories with no licensed target that has an embedded item: {n_sources_no_target:,}")
print(f"queries with >=1 recommendation: {n_queries:,} of {n_items:,} embedded items")
print(f"lists shorter than {TOP_K}: {int((group_len < TOP_K).sum()):,} "
      f"(mean length {group_len.mean():.2f})")
print(f"written -> {OUT_PATH.relative_to(ROOT)} ({len(q_all):,} rows, "
      f"{OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(f"total runtime: {time.time() - t0:.0f}s")
