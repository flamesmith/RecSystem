"""Builds the cache/c3_counts_{N}d.parquet files app.py reads.

Run once (or whenever you need a different N):
    python notebooks/review_category_shiny/precompute.py

For every review, every OTHER item the same reviewer reviewed in the following
N days becomes a directed (source, destination) category-pair edge -- same
construction as notebooks/review_category_pairs.ipynb §§1-4, at the cat_2 >
cat_3 level. A reviewer's own re-review of the same asin is excluded.

Reads 6.9M reviews and takes ~10-20s per window on this machine -- far too
slow to redo live on every Shiny input change, hence the cache.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_DAYS_TO_CACHE = (30, 90, 180)
MAX_FOLLOWUPS_PER_REVIEW = 15
SEED = 0

t0 = time.time()
reviews = pd.read_csv(
    ROOT / "data/Home_and_Kitchen_filtered.csv",
    usecols=["reviewerID", "asin", "unixReviewTime"],
    dtype={"reviewerID": str, "asin": str}, low_memory=False,
)
reviews["t"] = pd.to_numeric(reviews["unixReviewTime"], errors="coerce")
reviews = reviews.dropna(subset=["t", "reviewerID", "asin"])

item_cats = (pd.read_pickle(ROOT / "data/df_features.pkl")[["asin", "cat_2", "cat_3", "cat_4"]]
             .drop_duplicates("asin"))
for c in ("cat_2", "cat_3", "cat_4"):
    item_cats[c] = item_cats[c].fillna("Missing").astype(str)

reviews = reviews.merge(item_cats, on="asin", how="inner")
reviews = reviews.sort_values(["reviewerID", "t"]).reset_index(drop=True)
print(f"prep [{time.time() - t0:.0f}s]  {len(reviews):,} rows")

rid = reviews["reviewerID"].to_numpy()
asin = reviews["asin"].to_numpy()
times = reviews["t"].to_numpy()
n = len(reviews)
change = np.flatnonzero(rid[1:] != rid[:-1]) + 1
starts = np.concatenate([[0], change, [n]])
c2, c3 = reviews["cat_2"].to_numpy(), reviews["cat_3"].to_numpy()


def build_pairs(window_days, max_followups=MAX_FOLLOWUPS_PER_REVIEW, seed=SEED):
    rng = np.random.default_rng(seed)
    window_seconds = window_days * 86400
    src_chunks, tgt_chunks = [], []
    for g in range(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        m = e - s
        if m < 2:
            continue
        tt = times[s:e]
        upper = np.searchsorted(tt, tt + window_seconds, side="right")
        for i in range(m - 1):
            j_max = upper[i]
            if j_max <= i + 1:
                continue
            cand = np.arange(i + 1, j_max)
            if len(cand) > max_followups:
                cand = rng.choice(cand, max_followups, replace=False)
            src_chunks.append(np.full(len(cand), s + i))
            tgt_chunks.append(s + cand)
    src_idx = np.concatenate(src_chunks)
    tgt_idx = np.concatenate(tgt_chunks)
    keep = asin[src_idx] != asin[tgt_idx]      # a re-review of the same asin is not a "next item"
    return src_idx[keep], tgt_idx[keep]


for window_days in WINDOW_DAYS_TO_CACHE:
    t1 = time.time()
    src_idx, tgt_idx = build_pairs(window_days)
    edges = pd.DataFrame({
        "src_label": c2[src_idx] + " > " + c3[src_idx],
        "dst_label": c2[tgt_idx] + " > " + c3[tgt_idx],
    })
    counts = edges.value_counts(["src_label", "dst_label"]).rename("edges").reset_index()
    total = counts.groupby("src_label")["edges"].transform("sum")
    counts["share"] = counts["edges"] / total
    counts["rank"] = (counts.groupby("src_label")["share"]
                      .rank(ascending=False, method="first").astype(int))
    out_path = OUT / f"c3_counts_{window_days}d.parquet"
    counts.to_parquet(out_path, index=False)
    print(f"window {window_days}d: {len(src_idx):,} pairs -> {len(counts):,} category-pair rows "
          f"-> {out_path.relative_to(ROOT)}  [{time.time() - t1:.0f}s]")

print(f"\ntotal [{time.time() - t0:.0f}s]")
