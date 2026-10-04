"""Build the Popularity "trending in this category" carousel: an all-time
version and a recency-windowed version.

Not a trained model, and not a co-purchase-based fallback for "Complete the
Look" (see README) -- raw purchase count per item, grouped by the item's
OWN category (cat_2/cat_3/cat_4 together). "Most popular desks" when
viewing a desk -- not "most popular things bought alongside a desk," which
is a different question this file does not answer.

No --snapshot argument, unlike every other model's script -- deliberately
changed from an earlier version that took one. That version restricted
counts to items.in `data/tower/<snapshot_id>/item_asins.npy`, TTN's
co-purchase-pair-derived item list -- a restriction that had nothing to do
with Popularity's own logic (a raw count has no concept of "co-purchase"
at all) and silently excluded any item that was reviewed hundreds of times
but never happened to appear in a surviving TTN training pair. Categories
here come straight from df_features.pkl's own cat_2/cat_3/cat_4 instead,
covering every item with valid features (1,134,566 of them), not just the
much smaller subset (134,749 for w60) that also cleared TTN's co-purchase
and category-licensing filters. Since nothing left is snapshot-specific,
this output is no longer snapshot-scoped either -- one universal result,
not one per --window-days.

Two variants, both anchored at TTN/constants.json's date_threshold (the
same train/test cutoff every other fitted statistic in this pipeline
uses, so this never counts a review from the held-out test period):
  all_time -- every review before date_threshold, no decay.
  recency  -- only reviews in the RECENCY_WINDOW_DAYS immediately before
              date_threshold (default 60).

RECENCY_WINDOW_DAYS is UNRELATED to data_creation/complementary_cats_pairs/pairs.py's
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

Items with no cat_4 (15.8% of df_features.pkl) are grouped at "Missing",
same convention data_processing/build_snapshot.py uses, rather than
dropped -- a missing leaf category doesn't mean the item has no category
at all, cat_2/cat_3 are still real.

Prerequisite: data_creation/build_data.py must have already produced
data/df_features.pkl.

Output IS this carousel's final recommendation-generation artifact --
Popularity has no separate "model" from its recommendation list, so unlike
TTN and SigLIP2 there's no distinct generate_recommendations.py for it.
One parquet: cat_2 | cat_3 | cat_4 | variant | rank | candidate_asin |
score, top-20 per (category, variant) -- the same K used for TTN's and
SigLIP2's per-query recommendation files. Keyed by the category path
directly (cat_2/cat_3/cat_4), not an opaque integer id -- there's no
shared vocabulary to encode against now that this doesn't read
node_of_item.npy.

Usage: python Popularity/build_popularity.py
"""
import json
from pathlib import Path

import pandas as pd

TOP_K = 20      # matches TTN's and SigLIP2's recommendation-generation depth
MISSING = "Missing"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "Popularity" / "generated" / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "popularity.parquet"

RECENCY_WINDOW_DAYS = 60          # Popularity's own recency window -- see
                                   # the module docstring for why this is
                                   # NOT data_creation/complementary_cats_pairs' window_days

DATE_THRESHOLD = json.loads((ROOT / "TTN" / "constants.json").read_text())["date_threshold"]
REFERENCE_DATE = pd.Timestamp(DATE_THRESHOLD)     # swap for pd.Timestamp.now() in a live deployment
cutoff_time = REFERENCE_DATE.timestamp()
recency_start_time = (REFERENCE_DATE - pd.Timedelta(days=RECENCY_WINDOW_DAYS)).timestamp()

df_features = pd.read_pickle(DATA_DIR / "df_features.pkl")
cat_by_asin = df_features[["asin", "cat_2", "cat_3", "cat_4"]].drop_duplicates("asin").set_index("asin")
cat_by_asin["cat_4"] = cat_by_asin["cat_4"].fillna(MISSING)
print(f"items with a category (df_features.pkl): {len(cat_by_asin):,}")

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


def top_by_category(reviews, variant):
    """Raw purchase-proxy count per item, ranked within its own category.

    Returns rows in the schema: cat_2 | cat_3 | cat_4 | variant | rank |
    candidate_asin | score.
    """
    counts = reviews["asin"].value_counts()
    cats = cat_by_asin.reindex(counts.index)
    has_cat = cats["cat_2"].notna()   # false only for asins absent from df_features.pkl entirely

    table = pd.DataFrame({
        "candidate_asin": counts.index[has_cat].astype(str),
        "cat_2": cats.loc[has_cat, "cat_2"].to_numpy(),
        "cat_3": cats.loc[has_cat, "cat_3"].to_numpy(),
        "cat_4": cats.loc[has_cat, "cat_4"].to_numpy(),
        "score": counts.to_numpy()[has_cat].astype("float32"),
    })

    group_cols = ["cat_2", "cat_3", "cat_4"]
    top = (table.sort_values("score", ascending=False)
                .groupby(group_cols, group_keys=False)
                .head(TOP_K)
                .sort_values(group_cols + ["score"], ascending=[True, True, True, False]))
    top["rank"] = top.groupby(group_cols).cumcount() + 1
    top["variant"] = variant
    return top[group_cols + ["variant", "rank", "candidate_asin", "score"]]


all_time_rows = top_by_category(all_time, "all_time")
recency_rows = top_by_category(recency, "recency")
print(f"all_time: {all_time_rows['candidate_asin'].nunique():,} items with >=1 review, "
      f"{len(all_time_rows[['cat_2', 'cat_3', 'cat_4']].drop_duplicates()):,} categories covered")
print(f"recency ({RECENCY_WINDOW_DAYS}d): {recency_rows['candidate_asin'].nunique():,} items with >=1 review, "
      f"{len(recency_rows[['cat_2', 'cat_3', 'cat_4']].drop_duplicates()):,} categories covered")

out = pd.concat([all_time_rows, recency_rows], ignore_index=True)
out.to_parquet(OUT_PATH)

print(f"written -> {OUT_PATH.relative_to(ROOT)} "
      f"({len(out):,} rows, {OUT_PATH.stat().st_size / 1e3:,.0f} KB)")
