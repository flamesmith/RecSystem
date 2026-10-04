"""Filtering / serving prep: unify TTN, SigLIP2, and Popularity's separate
recommendation-generation outputs into one table the serving DB loads.

Three carousels, three different natural shapes going in:
  complements  -- TTN,      recommendations/complements.parquet  (per query item)
  complements_proportional
               -- TTN,      recommendations/complements_proportional.parquet
                  (same model, different merge rule across licensed categories)
  substitutes  -- SigLIP2,  recommendations/substitutes.parquet  (per query item)
  popular      -- Popularity, recommendations/popularity.parquet (per CATEGORY,
                  two variants: all_time / recency)

This is specifically where popularity's per-category rows get expanded into
per-item rows -- joining (cat_2, cat_3, cat_4) to every snapshot item whose
OWN category matches it (from df_features.pkl, not node_of_item.npy --
Popularity stopped using TTN's co-purchase-derived item/node vocabulary,
see Popularity/build_popularity.py's own docstring for why) -- so the
output below is uniform: one row per (query_asin, carousel, rank),
regardless of which of the three produced it.

Output schema (data/tower/<snapshot_id>/serving/recommendations.parquet):
  query_asin | carousel | variant | rank | candidate_asin | score
  carousel in {"complements", "complements_proportional", "substitutes", "popular"}
  variant  is None for complements/substitutes; "all_time" or "recency" for popular

"Filtering" here is exactly: truncate every carousel down to DISPLAY_K
(the number actually shown), from whatever TOP_K recommendation generation
stored (20). No cold-start routing is applied (v1 scope, per this session's
"raw TTN passthrough for v1" decision) -- complements shows TTN's raw
top-K as-is, including its known weakness on low-training-frequency
targets.

Prerequisite: TTN/generate_recommendations.py and SigLIP2/generate_recommendations.py
must have already run for the given --snapshot; Popularity/build_popularity.py
must have run too, but it isn't --snapshot-specific (just run once).

Usage: python website/prepare_serving.py --snapshot w90_2017-12-09
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DISPLAY_K = 10
# The proportional carousel is NOT truncated to DISPLAY_K: its 20 slots are
# allocated across licensed categories and ranked in category blocks, so
# cutting to the top 10 would drop whole categories (measured on w90: ~9.6
# categories per item across all 20 slots vs ~2.9 within the first 10) and
# defeat the point of that variant.
PROPORTIONAL_K = 20

ROOT = Path(__file__).resolve().parents[1]   # repo root; this file lives in website/
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_snapshot.py, e.g. w90_2017-12-09")
SNAPSHOT_ID = parser.parse_args().snapshot
SNAPSHOT_DIR = ROOT / "data" / "tower" / SNAPSHOT_ID
# Each model keeps its own recommendations under its own folder now, not all
# under the shared snapshot dir -- one recommendations dir per carousel.
# Popularity's isn't snapshot-scoped (see its own build script), so its
# path has no SNAPSHOT_ID in it, unlike the other two.
REC_DIRS = {
    "complements": ROOT / "TTN" / "generated" / SNAPSHOT_ID / "recommendations",
    "complements_proportional": ROOT / "TTN" / "generated" / SNAPSHOT_ID / "recommendations",
    "substitutes": ROOT / "SigLIP2" / "generated" / SNAPSHOT_ID / "recommendations",
    "popular": ROOT / "Popularity" / "generated" / "recommendations",
}
OUT_DIR = SNAPSHOT_DIR / "serving"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "recommendations.parquet"

asins = np.load(SNAPSHOT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
# Each snapshot item's own (cat_2, cat_3, cat_4) -- from df_features.pkl
# directly, the same source Popularity itself now uses, not node_of_item.npy
# (TTN's co-purchase-derived vocabulary; Popularity no longer reads it).
df_features = pd.read_pickle(ROOT / "data" / "df_features.pkl")
cat_by_asin = (df_features[["asin", "cat_2", "cat_3", "cat_4"]]
               .drop_duplicates("asin").set_index("asin"))
cat_by_asin["cat_4"] = cat_by_asin["cat_4"].fillna("Missing")
item_by_cat = pd.DataFrame({"query_asin": asins}).merge(
    cat_by_asin, left_on="query_asin", right_index=True, how="inner")


def truncate(df, key_col, k=DISPLAY_K):
    """Keep rank 1..k per key_col, re-deriving rank in case the
    source already truncated to fewer than k for some keys."""
    df = df[df["rank"] <= k].copy()
    df["rank"] = df.groupby(key_col)["rank"].rank(method="first").astype(int)
    return df


parts = []

for carousel, filename, k in (("complements", "complements.parquet", DISPLAY_K),
                               ("complements_proportional", "complements_proportional.parquet", PROPORTIONAL_K),
                               ("substitutes", "substitutes.parquet", DISPLAY_K)):
    path = REC_DIRS[carousel] / filename
    if not path.exists():
        print(f"skipping {carousel}: {path.relative_to(ROOT)} not found")
        continue
    df = pd.read_parquet(path)
    df = truncate(df, "query_asin", k)
    df["carousel"] = carousel
    df["variant"] = None
    parts.append(df[["query_asin", "carousel", "variant", "rank", "candidate_asin", "score"]])
    print(f"{carousel}: {len(df):,} rows, {df['query_asin'].nunique():,} query items")

pop_path = REC_DIRS["popular"] / "popularity.parquet"
if pop_path.exists():
    pop = pd.read_parquet(pop_path)
    # expand (cat_2, cat_3, cat_4) -> every snapshot item whose OWN category matches it
    expanded = pop.merge(item_by_cat, on=["cat_2", "cat_3", "cat_4"], how="inner")
    # A popular item can BE the query (it's popular in its own category) --
    # exclude that row, same self-exclusion complements/substitutes already
    # apply at generation time, just done here since this is the first point
    # that knows both the specific query_asin and candidate_asin together.
    expanded = expanded[expanded["candidate_asin"] != expanded["query_asin"]]
    # truncate() groups by a single key; popularity needs (query_asin, variant)
    # jointly, so inline the same logic here instead of reusing the helper.
    # Re-derive rank BEFORE truncating: removing the self-match can leave a
    # gap inside the top DISPLAY_K that should backfill from ranks 11-20
    # (popularity.parquet stores top-20), not just drop a slot.
    expanded = expanded.sort_values(["query_asin", "variant", "rank"]).copy()
    expanded["rank"] = expanded.groupby(["query_asin", "variant"]).cumcount() + 1
    expanded = expanded[expanded["rank"] <= DISPLAY_K]
    expanded["carousel"] = "popular"
    parts.append(expanded[["query_asin", "carousel", "variant", "rank", "candidate_asin", "score"]])
    print(f"popular: {len(expanded):,} rows (both variants), "
          f"{expanded['query_asin'].nunique():,} query items")
else:
    print(f"skipping popular: {pop_path.relative_to(ROOT)} not found")

out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
    columns=["query_asin", "carousel", "variant", "rank", "candidate_asin", "score"])
out.to_parquet(OUT_PATH)
print(f"\nwritten -> {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows, "
      f"{OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(out.groupby("carousel").size())
