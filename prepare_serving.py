"""Filtering / serving prep: unify TTN, SigLIP2, and Popularity's separate
recommendation-generation outputs into one table the serving DB loads.

Three carousels, three different natural shapes going in:
  complements  -- TTN,      recommendations/complements.parquet  (per query item)
  substitutes  -- SigLIP2,  recommendations/substitutes.parquet  (per query item)
  popular      -- Popularity, recommendations/popularity.parquet (per CATEGORY,
                  two variants: all_time / recency)

This is specifically where popularity's per-category rows get expanded into
per-item rows -- joining category_node_id to every asin whose OWN category
(node_of_item) equals it -- so the output below is uniform: one row per
(query_asin, carousel, rank), regardless of which of the three produced it.

Output schema (data/tower/<snapshot_id>/serving/recommendations.parquet):
  query_asin | carousel | variant | rank | candidate_asin | score
  carousel in {"complements", "substitutes", "popular"}
  variant  is None for complements/substitutes; "all_time" or "recency" for popular

"Filtering" here is exactly: truncate every carousel down to DISPLAY_K
(the number actually shown), from whatever TOP_K recommendation generation
stored (20). No cold-start routing is applied (v1 scope, per this session's
"raw TTN passthrough for v1" decision) -- complements shows TTN's raw
top-K as-is, including its known weakness on low-training-frequency
targets.

Prerequisite: TTN/generate_recommendations.py, SigLIP2/generate_recommendations.py,
and Popularity/build_popularity.py must have all already run for the given
--snapshot.

Usage: python prepare_serving.py --snapshot w90_2017-12-09
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DISPLAY_K = 10

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from TTN/build_data.py, e.g. w90_2017-12-09")
SNAPSHOT_DIR = ROOT / "data" / "tower" / parser.parse_args().snapshot
REC_DIR = SNAPSHOT_DIR / "recommendations"
OUT_DIR = SNAPSHOT_DIR / "serving"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "recommendations.parquet"

asins = np.load(SNAPSHOT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
node_of_item = np.load(SNAPSHOT_DIR / "node_of_item.npy")
item_by_node = pd.DataFrame({"query_asin": asins, "category_node_id": node_of_item})
item_by_node = item_by_node[item_by_node["category_node_id"] > 0]


def truncate(df, key_col):
    """Keep rank 1..DISPLAY_K per key_col, re-deriving rank in case the
    source already truncated to fewer than DISPLAY_K for some keys."""
    df = df[df["rank"] <= DISPLAY_K].copy()
    df["rank"] = df.groupby(key_col)["rank"].rank(method="first").astype(int)
    return df


parts = []

for carousel, filename in (("complements", "complements.parquet"),
                            ("substitutes", "substitutes.parquet")):
    path = REC_DIR / filename
    if not path.exists():
        print(f"skipping {carousel}: {path.relative_to(ROOT)} not found")
        continue
    df = pd.read_parquet(path)
    df = truncate(df, "query_asin")
    df["carousel"] = carousel
    df["variant"] = None
    parts.append(df[["query_asin", "carousel", "variant", "rank", "candidate_asin", "score"]])
    print(f"{carousel}: {len(df):,} rows, {df['query_asin'].nunique():,} query items")

pop_path = REC_DIR / "popularity.parquet"
if pop_path.exists():
    pop = pd.read_parquet(pop_path)
    # expand category_node_id -> every item whose OWN category is that node
    expanded = pop.merge(item_by_node, on="category_node_id", how="inner")
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
