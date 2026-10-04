"""Load the unified recommendations table into a SQLite serving DB.

The last step before the API: prepare_serving.py's parquet is a batch
artifact, not something a request handler should query directly (loading a
15MB+ parquet per request is not what parquet is for). SQLite here is a
deliberate choice for this project's stage, not a placeholder for something
heavier: the data is a precomputed lookup that gets wholesale-replaced per
snapshot/promotion, never written to concurrently by request traffic, and
zero-infrastructure fits "local demo" scope exactly.

Schema:
  CREATE TABLE recommendations (
    query_asin TEXT, carousel TEXT, variant TEXT, rank INTEGER,
    candidate_asin TEXT, score REAL
  )
  Indexed on (query_asin, carousel, variant) -- the API's one query pattern:
  "give me this item's recommendations for this carousel."

Also loads an `items (asin, title, image_url)` table for the demo page.

Prerequisite: prepare_serving.py must have already run for the given
--snapshot, and data/asin_image_urls.json must exist
(python data_creation/build_image_url_cache.py).

Usage: python website/load_serving_db.py --snapshot w90_2017-12-09
"""
import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repo root; this file lives in website/
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_snapshot.py, e.g. w90_2017-12-09")
SNAPSHOT_DIR = ROOT / "data" / "tower" / parser.parse_args().snapshot
SRC_PATH = SNAPSHOT_DIR / "serving" / "recommendations.parquet"
DB_PATH = SNAPSHOT_DIR / "serving" / "recommendations.db"

df = pd.read_parquet(SRC_PATH)
print(f"loaded {SRC_PATH.relative_to(ROOT)}: {len(df):,} rows")

if DB_PATH.exists():
    DB_PATH.unlink()   # wholesale replace, not an incremental update
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE recommendations (
        query_asin TEXT NOT NULL,
        carousel TEXT NOT NULL,
        variant TEXT,
        rank INTEGER NOT NULL,
        candidate_asin TEXT NOT NULL,
        score REAL NOT NULL
    )
""")
df.to_sql("recommendations", conn, if_exists="append", index=False)
conn.execute("CREATE INDEX idx_query_carousel ON recommendations (query_asin, carousel, variant)")

# Display metadata (title, image) for every asin that can appear on screen,
# as query or candidate, so the demo page can show pictures without the API
# ever touching df_features.pkl. Image urls come from the small cache
# data_creation/build_image_url_cache.py writes; items without one are
# simply absent from it (about half the catalogue).
shown = pd.unique(pd.concat([df["query_asin"], df["candidate_asin"]]))
titles = (pd.read_pickle(ROOT / "data" / "df_features.pkl")[["asin", "title"]]
          .drop_duplicates("asin").set_index("asin")["title"])
image_by_asin = json.loads((ROOT / "data" / "asin_image_urls.json").read_text())
items = pd.DataFrame({"asin": shown})
items["title"] = items["asin"].map(titles)
items["image_url"] = items["asin"].map(image_by_asin)
conn.execute("CREATE TABLE items (asin TEXT PRIMARY KEY, title TEXT, image_url TEXT)")
items.to_sql("items", conn, if_exists="append", index=False)
print(f"items metadata: {len(items):,} asins, "
      f"{items['image_url'].notna().sum():,} with an image")
conn.commit()

n_rows = conn.execute("SELECT COUNT(*) FROM recommendations").fetchone()[0]
n_queries = conn.execute("SELECT COUNT(DISTINCT query_asin) FROM recommendations").fetchone()[0]
conn.close()

print(f"written -> {DB_PATH.relative_to(ROOT)} "
      f"({n_rows:,} rows, {n_queries:,} distinct query items, "
      f"{DB_PATH.stat().st_size / 1e6:,.1f} MB)")
