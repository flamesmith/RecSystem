"""Build every data_creation artifact the rest of the pipeline reads, from the
two raw sources -- lives at the top of data_creation/, not inside any one
subpackage, since its output feeds all three models (via data_processing/).

Prerequisites, already on disk and tracked in git (schema/config, not
generated -- see data/.gitignore):
  data/master_metadata.json      -- category -> field extraction schema
  data/global_filters.json       -- marketing phrases stripped before extraction
  data/category_taxonomy.json    -- reviewed (cat_3, cat_4) whitelist
  TTN/constants.json             -- shared train/test date_threshold (read by
                                     data_processing/, not by this script)

Raw inputs (gitignored, must be present locally):
  data/meta_Home_and_Kitchen_filtered.csv  -- item catalogue metadata
  data/Home_and_Kitchen_filtered.csv       -- interaction/review log (not read
                                               directly by this script, but
                                               required downstream by
                                               data_processing/build_snapshot.py
                                               and Popularity/build_popularity.py)

Produces, under data/ (all gitignored, regenerated from the above):
  df_features.pkl                -- one row per catalogued item: cleaned text,
                                     extracted structured features, category
                                     path, also_buy. Read by every downstream
                                     stage (data_processing/, SigLIP2/, TTN/).
  df_features_with_embeddings.pkl -- df_features.pkl + a 384-d SBERT title
                                     vector per item. Read by
                                     data_processing/build_ttn_arrays.py.
  pair_stats.pkl                  -- EVERY scored also_buy category pair,
                                     unfiltered. Cached because scoring it is
                                     the expensive step; re-thresholding from
                                     here costs seconds. Not read downstream --
                                     it exists so --min-edges/--min-lift can be
                                     retried cheaply.
  complementary_categories.pkl    -- pair_stats.pkl cut to the pairs clearing
                                     both thresholds. Read by
                                     data_processing/build_snapshot.py.

NOT produced here, by design (see complementary_cats_pairs/pairs.py and
feature_extraction_workflow/extract_features_user.py for the underlying,
still-available functions):
  co_purchase_pairs_{train,test}.pkl -- window_days is a data_processing-time
                                         parameter (--window-days on
                                         build_snapshot.py), so a fixed-window
                                         copy built here would be stale the
                                         moment a different window is used.
                                         build_snapshot.py calls
                                         co_purchase_pairs(...) directly.
  df_user_features.pkl               -- leakage-safe per-purchase user
                                         features; nothing downstream reads
                                         this yet (no model is user-aware).

Every step is resumable: it's skipped (and the existing file reused) unless
--force is passed, since feature extraction alone takes ~40 minutes.

Usage:
  python data_creation/build_data.py
  python data_creation/build_data.py --min-edges 10 --min-lift 3.0
  python data_creation/build_data.py --force   # rebuild everything from scratch
"""

# ============================================================================
# Setup
# ============================================================================
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_creation.feature_extraction_workflow import (
    canonicalize_values,
    clean_brand,
    clean_numeric_ranges,
    expand_features,
    normalize_dimensions,
    run_feature_extraction,
    save_features,
)
from data_creation.embedding_analysis import create_embeddings
from data_creation.complementary_cats_pairs import (
    build_base_table,
    build_pairs,
    filter_pairs,
    load_catalogue_lookup,
    load_taxonomy,
    save_pairs,
    score_pairs,
)

parser = argparse.ArgumentParser()
parser.add_argument("--min-edges", type=int, default=5,
                     help="support floor for complementary_categories.pkl, "
                          "as a whole number of co-purchases (default 5)")
parser.add_argument("--min-lift", type=float, default=2.0,
                     help="lift floor for complementary_categories.pkl (default 2.0)")
parser.add_argument("--force", action="store_true",
                     help="rebuild every artifact even if it already exists on disk")
ARGS = parser.parse_args()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)

TEXT_COLUMNS = ["title", "description", "feature"]
LIST_COLUMNS = ["description", "feature"]

FEATURES_PATH = DATA_DIR / "df_features.pkl"
EMBEDDINGS_PATH = DATA_DIR / "df_features_with_embeddings.pkl"
PAIR_STATS_PATH = DATA_DIR / "pair_stats.pkl"
COMPLEMENTARY_PATH = DATA_DIR / "complementary_categories.pkl"

# ============================================================================
# 1. Item feature extraction -- meta_Home_and_Kitchen_filtered.csv -> df_features.pkl
# ============================================================================
if FEATURES_PATH.exists() and not ARGS.force:
    df_features = pd.read_pickle(FEATURES_PATH)
    print(f"[1/3] reusing {FEATURES_PATH.name} ({len(df_features):,} rows) -- pass --force to rebuild")
else:
    print("[1/3] extracting item features from meta_Home_and_Kitchen_filtered.csv "
          "(text cleaning + per-category schema extraction; can take ~40 min)")
    df_items = pd.read_csv(
        DATA_DIR / "meta_Home_and_Kitchen_filtered.csv", low_memory=False,
    ).drop_duplicates()

    df_result = run_feature_extraction(
        df_items,
        text_columns=TEXT_COLUMNS,
        master_metadata=str(DATA_DIR / "master_metadata.json"),
        global_filters=str(DATA_DIR / "global_filters.json"),
        list_columns=LIST_COLUMNS,
        priority=TEXT_COLUMNS,
    )
    df_features = clean_numeric_ranges(normalize_dimensions(clean_brand(
        canonicalize_values(expand_features(df_result)))))
    save_features(df_features, FEATURES_PATH)
    print(f"      saved {len(df_features):,} rows x {df_features.shape[1]} cols -> {FEATURES_PATH.name}")

# ============================================================================
# 2. Title embeddings -- df_features.pkl -> df_features_with_embeddings.pkl
# ============================================================================
if EMBEDDINGS_PATH.exists() and not ARGS.force:
    print(f"[2/3] reusing {EMBEDDINGS_PATH.name} -- pass --force to rebuild")
else:
    print("[2/3] encoding item titles with SBERT (all-MiniLM-L6-v2)")
    df_emb = create_embeddings(
        df_features, text_col="title_cleaned", model_name="all-MiniLM-L6-v2",
        batch_size=256, embedding_col="title_embedding",
    )
    df_emb.to_pickle(EMBEDDINGS_PATH)
    print(f"      saved {len(df_emb):,} rows -> {EMBEDDINGS_PATH.name}")

# ============================================================================
# 3. Complementary categories -- also_buy edges -> scored, filtered category pairs
# ============================================================================
if PAIR_STATS_PATH.exists() and not ARGS.force:
    pair_stats = pd.read_pickle(PAIR_STATS_PATH)
    print(f"[3/3] reusing {PAIR_STATS_PATH.name} ({len(pair_stats):,} scored pairs) -- pass --force to rebuild")
else:
    print("[3/3] scoring also_buy category pairs (support + lift over the full catalogue)")
    valid_pairs = load_taxonomy(DATA_DIR / "category_taxonomy.json")
    df_base = build_base_table(df_features, valid_pairs)
    cat_lookup = load_catalogue_lookup(DATA_DIR / "meta_Home_and_Kitchen_filtered.csv", valid_pairs)
    pairs = build_pairs(df_base, cat_lookup)
    pair_stats = score_pairs(pairs)
    pair_stats.to_pickle(PAIR_STATS_PATH)
    print(f"      scored {len(pair_stats):,} distinct pairs -> {PAIR_STATS_PATH.name}")

complementary_categories = filter_pairs(pair_stats, min_edges=ARGS.min_edges, min_lift=ARGS.min_lift)
save_pairs(complementary_categories, COMPLEMENTARY_PATH)
print(f"      kept {len(complementary_categories):,} pairs at min_edges={ARGS.min_edges}, "
      f"min_lift={ARGS.min_lift} -> {COMPLEMENTARY_PATH.name}")

# ============================================================================
# Summary
# ============================================================================
print("\nready for data_processing/build_snapshot.py:")
for f in (FEATURES_PATH, EMBEDDINGS_PATH, PAIR_STATS_PATH, COMPLEMENTARY_PATH):
    print(f"  {f.name:<32} {f.stat().st_size / 1e6:>10,.1f} MB")
