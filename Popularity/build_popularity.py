"""Build the Popularity baseline/fallback for "Complete the Look".

Not a trained model: "popularity within the category" -- a candidate item's
membership in its target category's top-10 (or top-100) most frequent
training targets. This is the exact definition used as TTN's baseline
throughout ../TTN/ttn_complementary.ipynb and every comparison script that
reports a POP column; it was never factored out into its own artifact before
now, only recomputed inline wherever it was needed.

All-time training-window popularity: a target's raw count across the full
training pairs table, no time decay. See README.md for the gap this leaves
(no 60-day-window variant exists yet).

Prerequisite: ../TTN/build_ttn.py must have already run at least once, to
produce data/tower/pairs_train.parquet and data/tower/node_of_item.npy.

Usage: python Popularity/build_popularity.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "tower"
OUT_PATH = OUT_DIR / "popularity_top100.json"

pairs_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
print(f"training pairs: {len(pairs_train):,}")

counts = (pairs_train.groupby(["target_node_id", "target_idx"]).size()
          .rename("n").reset_index())

top10_by_node = {int(k): g.nlargest(10, "n")["target_idx"].astype(int).tolist()
                  for k, g in counts.groupby("target_node_id")}
top100_by_node = {int(k): g.nlargest(100, "n")["target_idx"].astype(int).tolist()
                   for k, g in counts.groupby("target_node_id")}

json.dump(
    {"top10_by_node": top10_by_node, "top100_by_node": top100_by_node},
    open(OUT_PATH, "w"),
)

print(f"categories covered: {len(top100_by_node):,}")
print(f"written -> {OUT_PATH.relative_to(ROOT)} "
      f"({OUT_PATH.stat().st_size / 1e3:,.0f} KB)")
