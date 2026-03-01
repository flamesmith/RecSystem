"""Daily model training script.

Trains all recommendation models on historical review data up to a configurable
cutoff timestamp, then serializes each model to the models/ directory.

Run this script once per day (e.g. via cron or a workflow scheduler):

    python scripts/train_all_models.py
    python scripts/train_all_models.py --cutoff 2014-01-01
    python scripts/train_all_models.py --data data/Home_and_Kitchen_filtered.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

# Ensure project root is importable regardless of working directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functions.ials_implicit_package import IALSImplicitRecommender
from functions.interaction_matrix import InteractionMatrixBuilder
from functions.item_item_knn import ItemItemRecommenderKnn
from functions.model_store import save_model
from functions.rs_baseline_models import (
    CooccurrenceRecommender,
    PopularityRecommender,
    TrendingRecommender,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, 'data', 'Home_and_Kitchen_filtered.csv')

# Model hyperparameters — adjust here or pass via CLI flags in the future
IALS_CONFIG = dict(factors=64, regularization=0.01, alpha=40, epochs=15, use_gpu=False)
KNN_CONFIG = dict(k=50, min_users=5, shrinkage=10)
MIN_USERS_MATRIX = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elapsed(start):
    return f"{time.time() - start:.1f}s"


def _determine_cutoff(cutoff_arg, df):
    """Return a unix timestamp to use as the training cutoff.

    If --cutoff is provided as a date string (YYYY-MM-DD) use that.
    Otherwise default to the most recent interaction timestamp in the data
    (i.e. train on everything available).
    """
    if cutoff_arg:
        ts = pd.to_datetime(cutoff_arg)
        return int(ts.timestamp())
    return int(df['unixReviewTime'].max())


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(data_path, cutoff_arg=None, models_dir=None):
    print("=" * 60)
    print("RecSystem — daily model training")
    print("=" * 60)

    # 1. Load data
    t0 = time.time()
    print(f"\n[1/6] Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    df = df.sort_values('unixReviewTime').reset_index(drop=True)
    print(f"      {len(df):,} interactions loaded  ({_elapsed(t0)})")

    # 2. Determine cutoff
    cutoff_unix = _determine_cutoff(cutoff_arg, df)
    cutoff_ts = pd.to_datetime(cutoff_unix, unit='s')
    print(f"\n[2/6] Training cutoff: {cutoff_ts} (unix={cutoff_unix})")

    df_train = df[df['unixReviewTime'] < cutoff_unix].copy()
    print(f"      Training set: {len(df_train):,} interactions")

    # 3. Build interaction matrix (used by IALS)
    t0 = time.time()
    print(f"\n[3/6] Building interaction matrix (min_users={MIN_USERS_MATRIX}) ...")
    builder = InteractionMatrixBuilder(min_users=MIN_USERS_MATRIX)
    train_matrix = builder.build(df_train)
    print(
        f"      Matrix shape: {train_matrix.shape[0]:,} users × {train_matrix.shape[1]:,} items"
        f"  ({_elapsed(t0)})"
    )

    # 4. Train each model
    trained = {}

    # --- Popularity ---
    t0 = time.time()
    print("\n[4/6] Training models ...")
    print("      Popularity ...")
    trained['popularity'] = PopularityRecommender().fit(df_train, cutoff_unix)
    print(f"        done  ({_elapsed(t0)})")

    # --- Trending ---
    t0 = time.time()
    print("      Trending ...")
    trained['trending'] = TrendingRecommender(n_days=7).fit(df_train, cutoff_unix)
    print(f"        done  ({_elapsed(t0)})")

    # --- Co-occurrence ---
    t0 = time.time()
    print("      Cooccurrence ...")
    trained['cooccurrence'] = CooccurrenceRecommender().fit(df_train, cutoff_unix)
    print(f"        done  ({_elapsed(t0)})")

    # --- Item-Item kNN ---
    t0 = time.time()
    print(f"      Item-Item kNN  (k={KNN_CONFIG['k']}) ...")
    trained['item_knn'] = ItemItemRecommenderKnn(**KNN_CONFIG).fit(df_train, cutoff_unix)
    print(f"        done  ({_elapsed(t0)})")

    # --- IALS ---
    t0 = time.time()
    print(f"      IALS  (factors={IALS_CONFIG['factors']}, epochs={IALS_CONFIG['epochs']}) ...")
    ials = IALSImplicitRecommender(**IALS_CONFIG)
    ials.fit(train_matrix, builder.item_map, builder.items, builder.user_map, builder.users)
    trained['ials'] = ials
    print(f"        done  ({_elapsed(t0)})")

    # 5. Persist all models
    print("\n[5/6] Saving model artifacts ...")
    kwargs = dict(models_dir=models_dir) if models_dir else {}
    for name, model in trained.items():
        save_model(model, name, **kwargs)

    # 6. Summary
    print("\n[6/6] Training complete.")
    print(f"      Models saved: {list(trained.keys())}")
    print("=" * 60)

    return trained


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train all recommendation models.')
    parser.add_argument(
        '--data',
        default=DEFAULT_DATA_PATH,
        help='Path to review CSV file (default: data/Home_and_Kitchen_filtered.csv)',
    )
    parser.add_argument(
        '--cutoff',
        default=None,
        help='Training cutoff date as YYYY-MM-DD.  Defaults to latest timestamp in data.',
    )
    parser.add_argument(
        '--models-dir',
        default=None,
        dest='models_dir',
        help='Directory to write model artifacts (default: models/).',
    )
    args = parser.parse_args()

    train(data_path=args.data, cutoff_arg=args.cutoff, models_dir=args.models_dir)
