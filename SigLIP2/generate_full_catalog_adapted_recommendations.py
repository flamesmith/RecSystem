"""Apply the champion SigLIP2 adapter (checkpoints/pilot_200k) to EVERY real
image embedding in the shared cache (SigLIP2/image_cache/, 558,324 items --
the full df_features.pkl population with a usable image, not any one
--snapshot) and store each item's top-40 most visually similar items,
restricted to its own category -- same in-category discipline every other
recommendation in this project already uses (TTN, the existing
substitutes.parquet, the notebook's compare_siglip()).

Image-to-image only, by deliberate choice made earlier this session -- no
text side, no cross-modal search.

Scale note: this is ~2.5x the previous largest run (the w90 snapshot's
137,362 items). Categories vary enormously in size (Kitchen & Dining alone
has ~214K encoded items), so this batches queries against each category's
full candidate block rather than ever materializing a full item x item
similarity matrix (558,324^2 would be far too large to hold in memory).

Output: SigLIP2/checkpoints/pilot_200k/recommendations/full_catalog_top40.parquet
-- query_asin | rank | candidate_asin | score. Placed inside the imported
checkpoint's own folder per explicit instruction, not under SigLIP2/generated/
like every other model's output in this project -- a deliberate deviation
from that convention, not an oversight (flagged when this script was written).

Prerequisite: SigLIP2/encode_siglip2_images.py and/or
SigLIP2/encode_images_by_category.py must have already populated
SigLIP2/image_cache/.

Usage: python SigLIP2/generate_full_catalog_adapted_recommendations.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TOP_K = 40
ADAPT_BATCH = 8192      # items per adapt_image() call
QUERY_BATCH = 2048      # queries per scoring batch, within one category

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SigLIP2.adapter import load_adapter, adapt_image

CACHE_DIR = ROOT / "SigLIP2" / "image_cache"
CHAMPION_PATH = ROOT / "SigLIP2" / "checkpoints" / "pilot_200k" / "taxonomy_w20_random_200k.pt"
OUT_DIR = ROOT / "SigLIP2" / "checkpoints" / "pilot_200k" / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "full_catalog_top40.parquet"

t0 = time.time()

# ============================================================================
# 1. Load the shared cache, keep only real (status==1) embeddings
# ============================================================================
cache_asins = np.load(CACHE_DIR / "cache_asins.npy", allow_pickle=False).astype(str)
cache_emb = np.load(CACHE_DIR / "cache_emb.npy").astype("float32")
cache_status = np.load(CACHE_DIR / "cache_status.npy")
valid = cache_status == 1
asins = cache_asins[valid]
raw_emb = cache_emb[valid]
n_items = len(asins)
print(f"cache: {len(cache_asins):,} tracked, {n_items:,} with a real embedding")

# ============================================================================
# 2. Apply the champion adapter -- image side only
# ============================================================================
DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
model = load_adapter(checkpoint_path=CHAMPION_PATH, device=DEVICE)
print(f"adapter loaded on {DEVICE}")

adapted = np.empty_like(raw_emb)
for i in range(0, n_items, ADAPT_BATCH):
    j = min(i + ADAPT_BATCH, n_items)
    adapted[i:j] = adapt_image(model, raw_emb[i:j])
del model, raw_emb
print(f"adapted {n_items:,} embeddings [{time.time() - t0:.0f}s]")

# L2-normalize once; every score below is then a plain dot product
adapted = adapted / (np.linalg.norm(adapted, axis=1, keepdims=True) + 1e-9)

# ============================================================================
# 3. Each item's own category -- from df_features.pkl, same convention
# Popularity/build_popularity.py uses (not any --snapshot's node_of_item.npy)
# ============================================================================
df_features = pd.read_pickle(ROOT / "data" / "df_features.pkl")
cat_by_asin = (df_features[["asin", "cat_2", "cat_3", "cat_4"]]
               .drop_duplicates("asin").set_index("asin"))
cat_by_asin["cat_4"] = cat_by_asin["cat_4"].fillna("Missing")

cats = cat_by_asin.reindex(asins)
have_cat = cats["cat_2"].notna().to_numpy()
print(f"items with a category: {have_cat.sum():,} of {n_items:,}")

category_key = (cats["cat_2"].astype(str) + "\x1f" + cats["cat_3"].astype(str)
                 + "\x1f" + cats["cat_4"].astype(str)).to_numpy()

# Group item indices by category -- same sort + searchsorted technique
# TTN/SigLIP2's existing generate_recommendations.py scripts already use.
idx_with_cat = np.where(have_cat)[0]
order = idx_with_cat[np.argsort(category_key[idx_with_cat], kind="stable")]
sorted_keys = category_key[order]
boundaries = np.searchsorted(sorted_keys, np.unique(sorted_keys))
boundaries = np.append(boundaries, len(order))
unique_keys = np.unique(sorted_keys)

# ============================================================================
# 4. Score each category's items against each other, in query batches --
# never materialize a full item x item matrix.
# ============================================================================
rows = []
t1 = time.time()
n_categories = len(unique_keys)
for gi in range(n_categories):
    cand = order[boundaries[gi]:boundaries[gi + 1]]
    if len(cand) < 2:
        continue
    C = adapted[cand]  # (n_cand, 768), already normalized

    for b in range(0, len(cand), QUERY_BATCH):
        batch = cand[b:b + QUERY_BATCH]
        Q = adapted[batch]
        scores = Q @ C.T                                 # (batch, n_cand)
        self_mask = cand[None, :] == batch[:, None]
        scores = np.where(self_mask, -np.inf, scores)

        k = min(TOP_K, scores.shape[1])
        top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        order_k = np.argsort(-np.take_along_axis(scores, top_idx, axis=1), axis=1)
        ranked = np.take_along_axis(top_idx, order_k, axis=1)
        ranked_scores = np.take_along_axis(scores, ranked, axis=1)

        for qi, q_item in enumerate(batch):
            valid_mask = ranked_scores[qi] != -np.inf
            n_valid = int(valid_mask.sum())
            if n_valid == 0:
                continue
            cand_pos = ranked[qi, :n_valid]
            rows.append(pd.DataFrame({
                "query_asin": asins[q_item],
                "rank": np.arange(1, n_valid + 1),
                "candidate_asin": asins[cand[cand_pos]],
                "score": ranked_scores[qi, :n_valid].astype("float32"),
            }))

    if (gi + 1) % 200 == 0 or gi + 1 == n_categories:
        print(f"  category {gi + 1:,}/{n_categories:,}  [{time.time() - t1:.0f}s]", flush=True)

out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["query_asin", "rank", "candidate_asin", "score"])
out.to_parquet(OUT_PATH)
print(f"\nqueries with >=1 recommendation: {out['query_asin'].nunique():,} of {n_items:,}")
print(f"written -> {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows, "
      f"{OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(f"total runtime: {time.time() - t0:.0f}s")
