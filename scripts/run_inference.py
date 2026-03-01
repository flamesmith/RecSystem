"""Inference example — load models once, serve recommendations per user.

This script demonstrates the correct pattern for serving recommendations
at scale: models are loaded from disk exactly once (at process startup),
then recommend() is called for each user request without touching raw data
or retraining.

In a production setting the "load once" block would live in your service
initialisation (e.g. FastAPI startup event, Lambda cold-start handler, etc.)
and the per-user block would be a request handler.

Usage:
    # First train the models (if not already done)
    python scripts/train_all_models.py

    # Then run inference
    python scripts/run_inference.py
    python scripts/run_inference.py --user A1B2C3D4E5 --top-n 5
"""

import argparse
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functions.ials_implicit_package import IALSImplicitRecommender
from functions.item_item_knn import ItemItemRecommenderKnn
from functions.model_store import list_saved_models, load_model
from functions.rs_baseline_models import (
    CooccurrenceRecommender,
    PopularityRecommender,
    TrendingRecommender,
)

DATA_PATH = os.path.join(_PROJECT_ROOT, 'data', 'Home_and_Kitchen_filtered.csv')


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps a logical name to the class used to deserialise it.
MODEL_REGISTRY = {
    'popularity':   PopularityRecommender,
    'trending':     TrendingRecommender,
    'cooccurrence': CooccurrenceRecommender,
    'item_knn':     ItemItemRecommenderKnn,
    'ials':         IALSImplicitRecommender,
}


def load_all_models():
    """Load every model that has a saved artifact on disk.

    Returns:
        dict mapping model_name -> loaded model instance.
    """
    available = list_saved_models()
    if not available:
        raise RuntimeError(
            "No saved models found in models/.  "
            "Run scripts/train_all_models.py first."
        )

    models = {}
    for name in available:
        if name not in MODEL_REGISTRY:
            print(f"  Skipping unknown artifact '{name}'")
            continue
        models[name] = load_model(MODEL_REGISTRY[name], name)

    print(f"\nLoaded {len(models)} model(s): {list(models.keys())}\n")
    return models


# ---------------------------------------------------------------------------
# Per-user inference
# ---------------------------------------------------------------------------

def get_recommendations(models, user_id, user_history, top_n=10):
    """Generate recommendations from every loaded model for a single user.

    Args:
        models: dict of name -> model instance (from load_all_models()).
        user_id: User identifier string (used by IALS for known-user lookup).
        user_history: List of ASINs the user has interacted with.
        top_n: Number of recommendations per model.

    Returns:
        dict mapping model_name -> list of recommended ASINs.
    """
    results = {}
    for name, model in models.items():
        if name == 'ials':
            # IALS needs user_id for known-user embedding lookup
            recs = model.recommend(user_id, user_history, top_n=top_n)
        else:
            # All other models only need the item history
            recs = model.recommend(user_history, n=top_n)
        results[name] = recs
    return results


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _pick_demo_user(data_path, min_interactions=5):
    """Pick a real user from the dataset to demo with."""
    df = pd.read_csv(data_path, usecols=['reviewerID', 'asin'])
    counts = df.groupby('reviewerID').size()
    eligible = counts[counts >= min_interactions].index.tolist()
    if not eligible:
        raise ValueError("No users with enough interactions found in the data.")
    return eligible[0], df[df['reviewerID'] == eligible[0]]['asin'].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run per-user inference using saved models.')
    parser.add_argument('--user', default=None, help='User ID to generate recommendations for.')
    parser.add_argument('--top-n', type=int, default=10, dest='top_n')
    args = parser.parse_args()

    # --- LOAD ONCE (server startup / cold start) ---
    print("Loading models from disk ...")
    models = load_all_models()

    # --- PER-USER (request handler) ---
    if args.user:
        df = pd.read_csv(DATA_PATH, usecols=['reviewerID', 'asin'])
        user_id = args.user
        user_history = df[df['reviewerID'] == user_id]['asin'].tolist()
        if not user_history:
            print(f"User '{user_id}' not found in data or has no interactions.")
            sys.exit(1)
    else:
        print("No --user specified; picking a demo user from the data ...")
        user_id, user_history = _pick_demo_user(DATA_PATH)

    print(f"User:    {user_id}")
    print(f"History: {len(user_history)} interactions\n")

    recommendations = get_recommendations(models, user_id, user_history, top_n=args.top_n)

    print(f"Top-{args.top_n} recommendations per model:")
    print("-" * 50)
    for model_name, recs in recommendations.items():
        print(f"\n  [{model_name}]")
        for i, asin in enumerate(recs, 1):
            print(f"    {i:2d}. {asin}")
