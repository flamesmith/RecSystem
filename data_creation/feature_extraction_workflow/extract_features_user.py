"""Leakage-safe per-purchase user feature extraction.

Builds USER features from the purchase log (here: the review/comment data in
``Home_and_Kitchen_filtered.csv``). The output has **one row per purchase
event**, and every feature for that row is computed only from the user's
purchases on **strictly earlier dates** — so a model trained on these rows
never sees information from the current purchase or anything that happened on
the same day or later.

Concretely, if a user bought on Mar 1, Apr 2, May 3:
  - the Mar 1 row has an empty history (all features NaN),
  - the Apr 2 row sees only Mar 1,
  - the May 3 row sees Mar 1 and Apr 2.

Category/brand fields (``cat_2/3/4``, ``brand``) are not in the purchase log;
they are joined in from ``data/df_features.pkl`` on ``asin``.

Features produced (see the project discussion for the section numbering):

  Section 1 (first three only)
    - prior_purchase_count        number of strictly-earlier purchases
    - purchase_frequency          prior_count / (days_since_first / 30)
    - distinct_items              distinct asins bought before
    - distinct_brands             distinct (non-null) brands bought before
    - distinct_categories         distinct (non-null) cat_3 bought before

  Section 2 (recency & tenure, in days)
    - account_tenure              last_prior_day - first_prior_day
    - recency                     current_day - last_prior_day
    - days_since_first_purchase   current_day - first_prior_day

  Section 3 (temporal / cadence)
    - avg_inter_purchase_gap      mean days between consecutive prior purchases
    - inter_purchase_gap_std      std of those gaps (burstiness)
    - preferred_dow               modal day-of-week of prior purchases (0=Mon)
    - preferred_month             modal month (1-12)
    - preferred_season            modal meteorological season
    - activity_trend              slope of cumulative purchases vs. day

  Section 4 (category & brand, scalars only)
    - favorite_cat_2/3/4          modal category at each level
    - category_entropy            Shannon entropy (bits) of cat_2 mix
    - category_diversity          distinct cat_2 / prior_count
    - favorite_brand              modal brand
    - brand_loyalty               top-brand share among brand-known purchases
    - brand_diversity             distinct brands / prior_count

All features are NaN when the relevant history is empty/undefined (e.g. the
user's first purchase, or std with fewer than two gaps).
"""

from __future__ import annotations

import math
from collections import Counter
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathType = Union[str, PathLike]

# Columns read from the purchase log.
PURCHASE_COLUMNS = ["reviewerID", "asin", "unixReviewTime", "reviewTime"]

# Item attributes joined in from df_features (keyed on asin).
ITEM_COLUMNS = ["cat_2", "cat_3", "cat_4", "brand"]

# Output feature columns, in order.
FEATURE_COLUMNS = [
    # section 1 (first three)
    "prior_purchase_count",
    "purchase_frequency",
    "distinct_items",
    "distinct_brands",
    "distinct_categories",
    # section 2
    "account_tenure",
    "recency",
    "days_since_first_purchase",
    # section 3
    "avg_inter_purchase_gap",
    "inter_purchase_gap_std",
    "preferred_dow",
    "preferred_month",
    "preferred_season",
    "activity_trend",
    # section 4
    "favorite_cat_2",
    "favorite_cat_3",
    "favorite_cat_4",
    "category_entropy",
    "category_diversity",
    "favorite_brand",
    "brand_loyalty",
    "brand_diversity",
]

_SECONDS_PER_DAY = 86_400
# Meteorological seasons keyed by month (1-12).
_SEASON_BY_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}


# ---------------------------------------------------------------------------
# Loading / joining
# ---------------------------------------------------------------------------
def load_purchases(
    reviews_path: PathType,
    features_path: Optional[PathType] = None,
) -> pd.DataFrame:
    """Read the purchase log and attach item category/brand from df_features.

    ``asin`` is read as a string so leading zeros survive (otherwise the join
    to df_features, whose asins are strings, silently misses).
    Rows without a usable ``unixReviewTime`` are dropped (they can't be placed
    on the timeline).
    """
    df = pd.read_csv(
        reviews_path,
        usecols=PURCHASE_COLUMNS,
        dtype={"asin": str, "reviewerID": str},
    )
    df = df.dropna(subset=["unixReviewTime"]).copy()
    df["unixReviewTime"] = df["unixReviewTime"].astype("int64")

    if features_path is not None:
        feats = pd.read_pickle(features_path)
        keep = ["asin"] + [c for c in ITEM_COLUMNS if c in feats.columns]
        df = df.merge(
            feats[keep].drop_duplicates("asin"),
            on="asin",
            how="left",
            validate="many_to_one",
        )
    return df


# ---------------------------------------------------------------------------
# Core leakage-safe computation
# ---------------------------------------------------------------------------
def _shannon_entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return math.nan
    ent = 0.0
    for c in counter.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def compute_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-event DataFrame of leakage-safe user features.

    Index is aligned to ``df``. For every row, features reflect only the
    user's purchases on **strictly earlier days**. Same-day purchases are not
    counted as history (day resolution comes from ``unixReviewTime``).
    """
    n = len(df)

    # --- Derive per-row time fields once, vectorized. ---
    unix = df["unixReviewTime"].to_numpy(dtype="int64")
    day_num = unix // _SECONDS_PER_DAY  # integer day index on the timeline
    dt = pd.to_datetime(unix, unit="s", utc=True)
    dow_arr = dt.weekday.to_numpy()
    month_arr = dt.month.to_numpy()
    season_arr = np.array([_SEASON_BY_MONTH[m] for m in month_arr], dtype=object)

    def _obj_or_none(col: str) -> np.ndarray:
        if col in df.columns:
            s = df[col]
            return s.where(s.notna(), None).to_numpy(dtype=object)
        return np.full(n, None, dtype=object)

    user_arr = df["reviewerID"].to_numpy(dtype=object)
    asin_arr = df["asin"].to_numpy(dtype=object)
    brand_arr = _obj_or_none("brand")
    cat2_arr = _obj_or_none("cat_2")
    cat3_arr = _obj_or_none("cat_3")
    cat4_arr = _obj_or_none("cat_4")

    # --- Sort by (user, day) so each (user, day) batch is contiguous. ---
    order = np.lexsort((day_num, user_arr))

    # --- Output buffers (NaN / None defaults => empty history). ---
    out_num = {c: np.full(n, np.nan, dtype="float64") for c in FEATURE_COLUMNS
               if c not in ("preferred_season", "favorite_cat_2", "favorite_cat_3",
                            "favorite_cat_4", "favorite_brand")}
    out_obj = {c: np.full(n, None, dtype=object) for c in
               ("preferred_season", "favorite_cat_2", "favorite_cat_3",
                "favorite_cat_4", "favorite_brand")}

    # --- Per-user accumulator state. ---
    def _reset():
        return {
            "n": 0, "first_day": None, "last_day": None, "prev_day": None,
            "items": set(),
            "brand_ctr": Counter(), "cat2_ctr": Counter(),
            "cat3_ctr": Counter(), "cat4_ctr": Counter(),
            "dow_ctr": Counter(), "month_ctr": Counter(), "season_ctr": Counter(),
            "sum_gap": 0.0, "sumsq_gap": 0.0, "n_gap": 0,
            "sx": 0.0, "sy": 0.0, "sxy": 0.0, "sxx": 0.0,  # trend regression sums
        }

    st = _reset()
    cur_user = None

    i = 0
    while i < n:
        row0 = order[i]
        u = user_arr[row0]
        d = int(day_num[row0])

        if u != cur_user:
            st = _reset()
            cur_user = u

        # Span of contiguous rows for this (user, day) batch.
        j = i
        while j < n:
            rj = order[j]
            if user_arr[rj] != u or int(day_num[rj]) != d:
                break
            j += 1

        # ---- Compute features from current (strictly-prior) state. ----
        if st["n"] > 0:
            n_prior = st["n"]
            days_since_first = d - st["first_day"]
            account_tenure = st["last_day"] - st["first_day"]
            recency = d - st["last_day"]
            freq = (n_prior / (days_since_first / 30.0)
                    if days_since_first > 0 else np.nan)

            avg_gap = st["sum_gap"] / st["n_gap"] if st["n_gap"] > 0 else np.nan
            if st["n_gap"] >= 2:
                mean_g = st["sum_gap"] / st["n_gap"]
                var_g = max(st["sumsq_gap"] / st["n_gap"] - mean_g * mean_g, 0.0)
                gap_std = math.sqrt(var_g)
            else:
                gap_std = np.nan

            if st["n"] >= 2:
                denom = st["n"] * st["sxx"] - st["sx"] * st["sx"]
                trend = ((st["n"] * st["sxy"] - st["sx"] * st["sy"]) / denom
                         if denom > 0 else np.nan)
            else:
                trend = np.nan

            cat2_known = sum(st["cat2_ctr"].values())
            brand_known = sum(st["brand_ctr"].values())
            cat_entropy = _shannon_entropy_bits(st["cat2_ctr"])
            cat_div = len(st["cat2_ctr"]) / n_prior if n_prior else np.nan
            brand_div = len(st["brand_ctr"]) / n_prior if n_prior else np.nan
            if brand_known > 0:
                top_brand, top_brand_n = st["brand_ctr"].most_common(1)[0]
                brand_loyalty = top_brand_n / brand_known
                fav_brand = top_brand
            else:
                brand_loyalty = np.nan
                fav_brand = None

            fav_c2 = st["cat2_ctr"].most_common(1)[0][0] if st["cat2_ctr"] else None
            fav_c3 = st["cat3_ctr"].most_common(1)[0][0] if st["cat3_ctr"] else None
            fav_c4 = st["cat4_ctr"].most_common(1)[0][0] if st["cat4_ctr"] else None
            pref_dow = st["dow_ctr"].most_common(1)[0][0] if st["dow_ctr"] else np.nan
            pref_month = st["month_ctr"].most_common(1)[0][0] if st["month_ctr"] else np.nan
            pref_season = st["season_ctr"].most_common(1)[0][0] if st["season_ctr"] else None

            num_vals = {
                "prior_purchase_count": n_prior,
                "purchase_frequency": freq,
                "distinct_items": len(st["items"]),
                "distinct_brands": len(st["brand_ctr"]),
                "distinct_categories": len(st["cat3_ctr"]),
                "account_tenure": account_tenure,
                "recency": recency,
                "days_since_first_purchase": days_since_first,
                "avg_inter_purchase_gap": avg_gap,
                "inter_purchase_gap_std": gap_std,
                "preferred_dow": pref_dow,
                "preferred_month": pref_month,
                "activity_trend": trend,
                "category_entropy": cat_entropy,
                "category_diversity": cat_div,
                "brand_loyalty": brand_loyalty,
                "brand_diversity": brand_div,
            }
            obj_vals = {
                "preferred_season": pref_season,
                "favorite_cat_2": fav_c2,
                "favorite_cat_3": fav_c3,
                "favorite_cat_4": fav_c4,
                "favorite_brand": fav_brand,
            }
            for k in range(i, j):
                rk = order[k]
                for col, val in num_vals.items():
                    out_num[col][rk] = val
                for col, val in obj_vals.items():
                    out_obj[col][rk] = val
        # else: first batch for this user -> leave NaN/None defaults.

        # ---- Fold this batch into the accumulator (now it's "history"). ----
        for k in range(i, j):
            rk = order[k]
            st["items"].add(asin_arr[rk])
            if brand_arr[rk] is not None:
                st["brand_ctr"][brand_arr[rk]] += 1
            if cat2_arr[rk] is not None:
                st["cat2_ctr"][cat2_arr[rk]] += 1
            if cat3_arr[rk] is not None:
                st["cat3_ctr"][cat3_arr[rk]] += 1
            if cat4_arr[rk] is not None:
                st["cat4_ctr"][cat4_arr[rk]] += 1
            st["dow_ctr"][int(dow_arr[rk])] += 1
            st["month_ctr"][int(month_arr[rk])] += 1
            st["season_ctr"][season_arr[rk]] += 1

            # inter-purchase gap vs. the previous event (0 for same-day events)
            if st["prev_day"] is not None:
                gap = d - st["prev_day"]
                st["sum_gap"] += gap
                st["sumsq_gap"] += gap * gap
                st["n_gap"] += 1
            st["prev_day"] = d

            # trend regression sums: x = day, y = cumulative purchase index
            st["n"] += 1
            y = st["n"]
            st["sx"] += d
            st["sy"] += y
            st["sxy"] += d * y
            st["sxx"] += d * d

            if st["first_day"] is None:
                st["first_day"] = d
            st["last_day"] = d

        i = j

    data = {}
    for col in FEATURE_COLUMNS:
        data[col] = out_obj[col] if col in out_obj else out_num[col]
    return pd.DataFrame(data, index=df.index)


# ---------------------------------------------------------------------------
# Orchestration / persistence
# ---------------------------------------------------------------------------
def run_user_feature_extraction(
    reviews_path: PathType,
    features_path: Optional[PathType] = None,
    key_columns: tuple = ("reviewerID", "asin", "reviewTime", "unixReviewTime"),
) -> pd.DataFrame:
    """Load the purchase log, compute leakage-safe user features, return a
    DataFrame of the key columns + features (one row per purchase event)."""
    df = load_purchases(reviews_path, features_path)
    feats = compute_user_features(df)
    keys = [c for c in key_columns if c in df.columns]
    return pd.concat([df[keys].reset_index(drop=True),
                      feats.reset_index(drop=True)], axis=1)


def save_user_features(df: pd.DataFrame, path: PathType) -> None:
    """Persist the per-event user-feature table as a pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
