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

Prerequisite: TTN/build_data.py must have already run for the given
--snapshot, to produce that snapshot's item_asins.npy and node_of_item.npy
(each item's own category, the same node encoding used everywhere else in
this project).

Output is keyed by asin (not item_idx), matching every other stage's
recommendation output -- item_idx is a snapshot-internal row position, not
stable across snapshots with different item coverage.

Usage: python Popularity/build_popularity.py --snapshot w90_2017-12-09
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from TTN/build_data.py, e.g. w90_2017-12-09")
OUT_DIR = DATA_DIR / "tower" / parser.parse_args().snapshot
OUT_PATH = OUT_DIR / "popularity_top100.json"

RECENCY_WINDOW_DAYS = 60          # Popularity's own recency window -- see
                                   # the module docstring for why this is
                                   # NOT complementary_cats_pairs' window_days

DATE_THRESHOLD = json.loads((ROOT / "TTN" / "constants.json").read_text())["date_threshold"]
REFERENCE_DATE = pd.Timestamp(DATE_THRESHOLD)     # swap for pd.Timestamp.now() in a live deployment
cutoff_time = REFERENCE_DATE.timestamp()
recency_start_time = (REFERENCE_DATE - pd.Timedelta(days=RECENCY_WINDOW_DAYS)).timestamp()

asins = np.load(OUT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
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


def top_by_node(reviews):
    """Raw purchase-proxy count per tower item, ranked within its own category."""
    counts = reviews["asin"].value_counts()
    item_idx = counts.index.map(idx_of)
    in_tower = item_idx.notna()
    table = pd.DataFrame({
        "asin": counts.index[in_tower].astype(str),
        "item_idx": item_idx[in_tower].astype(int),
        "n": counts.to_numpy()[in_tower],
    })
    table["node"] = node_of_item[table["item_idx"].to_numpy()]
    table = table[table["node"] > 0]      # 0 is the reserved unseen/padding node
    top10 = {int(k): g.nlargest(10, "n")["asin"].tolist()
             for k, g in table.groupby("node")}
    top100 = {int(k): g.nlargest(100, "n")["asin"].tolist()
              for k, g in table.groupby("node")}
    return table, top10, top100


all_time_table, all_time_top10, all_time_top100 = top_by_node(all_time)
recency_table, recency_top10, recency_top100 = top_by_node(recency)
print(f"all_time: {len(all_time_table):,} items with >=1 review, "
      f"{len(all_time_top100):,} categories covered")
print(f"recency ({RECENCY_WINDOW_DAYS}d): {len(recency_table):,} items with >=1 review, "
      f"{len(recency_top100):,} categories covered")

json.dump({
    "all_time": {"top10_by_node": all_time_top10, "top100_by_node": all_time_top100},
    "recency": {"window_days": RECENCY_WINDOW_DAYS,
                "top10_by_node": recency_top10, "top100_by_node": recency_top100},
}, open(OUT_PATH, "w"))

print(f"written -> {OUT_PATH.relative_to(ROOT)} "
      f"({OUT_PATH.stat().st_size / 1e3:,.0f} KB)")
