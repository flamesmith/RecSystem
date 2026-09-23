"""Build the Popularity "trending in this category" carousel: an all-time
version and a recency-windowed version.

Not a trained model, and not a co-purchase-based fallback for "Complete the
Look" (see README) -- raw purchase count per item, grouped by the item's
OWN category (cat_4, the most specific level this project's taxonomy has --
there is no cat_5). "Most popular desks" when viewing a desk -- not "most
popular things bought alongside a desk," which is a different question this
file does not answer.

Two variants, both anchored at TTN/constants.json's date_threshold (the same
train/test cutoff every other fitted statistic in this pipeline uses, so
this never counts a review from the held-out test period):
  all_time -- every review before date_threshold, no decay.
  recency  -- only reviews in the RECENCY_WINDOW_DAYS immediately before
              date_threshold (default 60).

RECENCY_WINDOW_DAYS is UNRELATED to complementary_cats_pairs/pairs.py's
`window_days` (the co-purchase PAIRING window -- how far apart two
purchases can be and still count as "bought together" for TTN's training
pairs). Same word, two different concepts: one gates which purchases count
as a pair for TTN, the other gates which purchases count as "recent" for
this file. Anchoring both variants at date_threshold rather than today's
real-world date keeps this reproducible against the fixed historical
dataset; a live deployment recomputing this periodically would anchor at
the actual current date instead (swap REFERENCE_DATE below).

Purchases aren't directly observable in this dataset (Amazon review data,
not a transaction log), so -- consistent with how every co-purchase pair in
this project is built -- a review is used as the purchase proxy: one row in
Home_and_Kitchen_filtered.csv per (reviewer, item), counted once each.

Prerequisite: data_processing/build_snapshot.py must have already run for the given
--snapshot, to produce that snapshot's item_asins.npy and node_of_item.npy
(each item's own category, the same node encoding used everywhere else in
this project).

Output IS this carousel's final recommendation-generation artifact --
Popularity has no separate "model" from its recommendation list, so unlike
TTN and SigLIP2 there's no distinct generate_recommendations.py for it.
Written directly in the locked recommendation-generation schema (see
analysis notes / PLAN.md): one parquet, category_node_id | variant | rank |
candidate_asin | score, top-20 per (category, variant) -- the same K used
for TTN's and SigLIP2's per-query recommendation files, so filtering/
serving prep can treat all three uniformly. Keyed by asin (not item_idx),
matching every other stage's recommendation output -- item_idx is a
snapshot-internal row position, not stable across snapshots with different
item coverage.

Usage: python Popularity/build_popularity.py --snapshot w90_2017-12-09
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = 20      # matches TTN's and SigLIP2's recommendation-generation depth

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_snapshot.py, e.g. w90_2017-12-09")
SNAPSHOT_DIR = DATA_DIR / "tower" / parser.parse_args().snapshot
OUT_DIR = SNAPSHOT_DIR / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "popularity.parquet"

RECENCY_WINDOW_DAYS = 60          # Popularity's own recency window -- see
                                   # the module docstring for why this is
                                   # NOT complementary_cats_pairs' window_days

DATE_THRESHOLD = json.loads((ROOT / "TTN" / "constants.json").read_text())["date_threshold"]
REFERENCE_DATE = pd.Timestamp(DATE_THRESHOLD)     # swap for pd.Timestamp.now() in a live deployment
cutoff_time = REFERENCE_DATE.timestamp()
recency_start_time = (REFERENCE_DATE - pd.Timedelta(days=RECENCY_WINDOW_DAYS)).timestamp()

asins = np.load(SNAPSHOT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
node_of_item = np.load(SNAPSHOT_DIR / "node_of_item.npy")
idx_of = {a: i for i, a in enumerate(asins)}
print(f"tower items: {len(asins):,}")

df_reviews = pd.read_csv(
    DATA_DIR / "Home_and_Kitchen_filtered.csv",
    usecols=["asin", "unixReviewTime"],
    dtype={"asin": str},
    low_memory=False,
)
all_time = df_reviews[df_reviews["unixReviewTime"] < cutoff_time]
recency = all_time[all_time["unixReviewTime"] >= recency_start_time]
print(f"reviews before {DATE_THRESHOLD}: {len(all_time):,} of {len(df_reviews):,}")
print(f"  of those, in the {RECENCY_WINDOW_DAYS}-day window before it: {len(recency):,}")


def top_by_node(reviews, variant):
    """Raw purchase-proxy count per tower item, ranked within its own category.

    Returns rows in the locked recommendation-generation schema:
    category_node_id | variant | rank | candidate_asin | score.
    """
    counts = reviews["asin"].value_counts()
    item_idx = counts.index.map(idx_of)
    in_tower = item_idx.notna()
    table = pd.DataFrame({
        "candidate_asin": counts.index[in_tower].astype(str),
        "item_idx": item_idx[in_tower].astype(int),
        "score": counts.to_numpy()[in_tower].astype("float32"),
    })
    table["category_node_id"] = node_of_item[table["item_idx"].to_numpy()]
    table = table[table["category_node_id"] > 0]   # 0 is the reserved unseen/padding node

    top = (table.sort_values("score", ascending=False)
                .groupby("category_node_id", group_keys=False)
                .head(TOP_K)
                .sort_values(["category_node_id", "score"], ascending=[True, False]))
    top["rank"] = top.groupby("category_node_id").cumcount() + 1
    top["variant"] = variant
    return top[["category_node_id", "variant", "rank", "candidate_asin", "score"]]


all_time_rows = top_by_node(all_time, "all_time")
recency_rows = top_by_node(recency, "recency")
print(f"all_time: {all_time_rows['candidate_asin'].nunique():,} items with >=1 review, "
      f"{all_time_rows['category_node_id'].nunique():,} categories covered")
print(f"recency ({RECENCY_WINDOW_DAYS}d): {recency_rows['candidate_asin'].nunique():,} items with >=1 review, "
      f"{recency_rows['category_node_id'].nunique():,} categories covered")

out = pd.concat([all_time_rows, recency_rows], ignore_index=True)
out.to_parquet(OUT_PATH)

print(f"written -> {OUT_PATH.relative_to(ROOT)} "
      f"({len(out):,} rows, {OUT_PATH.stat().st_size / 1e3:,.0f} KB)")
