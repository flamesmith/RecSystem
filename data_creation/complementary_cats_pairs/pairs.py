"""Item-item co-purchase pairs read off the interaction log.

The sibling module `categories.py` works from Amazon's `also_buy` lists — what
Amazon *says* is bought together. This one works from the reviews themselves:
every unordered pair of items the same user actually bought. Two different
signals, and they need not agree.

Two filters keep the pairs honest, applied in this order:

`cutoff_time` drops every interaction at or after the threshold before any
pairing happens, so pairs built for training cannot see the test period.
Strictly-before, the same convention as `compute_cooccurrence_before_time` in
`functions/rs_baseline_models.py` and `InteractionMatrixBuilder`.

`window_days` then requires the two purchases to be close *to each other*: a
user who bought a blender in 2019 and a duvet in 2023 was not shopping for a
set. Without it, every item a long-lived account ever bought pairs with every
other one. The test is on the gap, not on recency, so a tightly spaced 2014
pair still counts.

Entry point: `run_co_purchase_pairs`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

USER_COL = "reviewerID"
ITEM_COL = "asin"
TIME_COL = "unixReviewTime"

SECONDS_PER_DAY = 86_400

# A year is the gap at which two purchases stop reading as one shopping
# intent. It halves the pair count on the full log (40.2M -> 22.3M), which is
# the size of the "bought these four years apart" tail being cut.
DEFAULT_WINDOW_DAYS = 365


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_interactions(path: PathLike) -> pd.DataFrame:
    """The interaction log, three columns: user, item, timestamp.

    `Home_and_Kitchen_filtered.csv` is 851 MB across 11 columns; only these
    three are ever needed here. `asin` and `reviewerID` are pinned to `str` so
    ids with leading zeros (e.g. `0560467893`) survive the read.
    """
    return pd.read_csv(
        path,
        usecols=[USER_COL, ITEM_COL, TIME_COL],
        dtype={ITEM_COL: str, USER_COL: str},
        low_memory=False,
    )


# --------------------------------------------------------------------------
# The pairs
# --------------------------------------------------------------------------
def co_purchase_pairs(df: pd.DataFrame, user_col: str = USER_COL,
                      item_col: str = ITEM_COL, time_col: str = TIME_COL,
                      cutoff_time: Optional[float] = None,
                      window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    """Every unordered pair of items one user bought within `window_days`.

    One row per distinct pair across the whole table: if a pair turns up for
    fifty users it still appears once. Order is not meaningful either, so the
    pair is stored with the alphabetically smaller asin in `asinA` — meaning
    (A, B) is present and (B, A) never is.

    Repeat purchases are not collapsed — a pair counts if ANY event of A and
    ANY event of B fall inside the window — so an item bought twice can bridge
    to whatever sat near either date. Exact duplicate (user, item, timestamp)
    rows are dropped, since they can only repeat pairs another row already
    makes, and an item pairing with itself is not a pair.

    Events are sorted by (user, time) so each user's history is one ascending
    block, which turns "everything within the window of event i" into a
    contiguous slice ending at `end[i]`. One `searchsorted` finds every one of
    those bounds at once, and the ragged ranges are expanded arithmetically —
    no Python-level loop over the 777k users.

    Parameters
    ----------
    df : DataFrame with a user column, an item column and a timestamp column.
    cutoff_time : threshold in the same units as `time_col` (unix seconds).
        `None` uses every interaction, which leaks the test period. For the
        training-safe version pass the shared split point, `date_threshold`
        from `ttn/constants.json`, as
        `pd.Timestamp(date_threshold).timestamp()` — do not hardcode a date
        here, or it becomes a second definition that can drift.
    window_days : maximum gap, in days, between the two purchases of a pair.

    Returns
    -------
    DataFrame with columns `asinA`, `asinB`, both `category` dtype, sorted.
    """
    d = df[[user_col, item_col, time_col]]

    if cutoff_time is not None:
        d = d[d[time_col] < cutoff_time]

    # Same (user, item, timestamp) twice can only repeat another row's pairs.
    # Repeat buys on DIFFERENT days survive — they are separate events.
    d = d.drop_duplicates()

    users = d[user_col].astype("category").cat.codes.to_numpy()
    item_cat = d[item_col].astype("category")
    items = item_cat.cat.codes.to_numpy()
    vocabulary = item_cat.cat.categories
    n_items = len(vocabulary)
    times = d[time_col].to_numpy(dtype=np.int64)

    # Sort by user, then by time, so each user's events sit in one ascending
    # block -- which is what makes the window a contiguous slice.
    order = np.lexsort((times, users))
    users, items, times = users[order], items[order], times[order]

    window = np.int64(window_days) * SECONDS_PER_DAY

    # One strictly increasing key per event: user_code * stride + timestamp.
    # stride exceeds any timestamp plus the window, so a search for
    # `key + window` stops inside the user's own block and can never run past
    # the boundary into the next user's events.
    stride = np.int64(times.max()) + window + 1 if len(times) else np.int64(1)
    key = users.astype(np.int64) * stride + times
    end = np.searchsorted(key, key + window, side="right")

    # Event i pairs with events (i, end_i) -- a ragged range per event, so
    # expand the counts into flat index arrays rather than looping.
    idx = np.arange(len(key))
    counts = end - idx - 1
    starts = np.cumsum(counts) - counts
    left = np.repeat(idx, counts)
    right = left + 1 + (np.arange(counts.sum()) - np.repeat(starts, counts))

    a, b = items[left], items[right]
    same_item = a == b                 # a repeat buy of one item is not a pair
    a, b = a[~same_item], b[~same_item]

    # One integer per pair, so the dedupe is a sort over a flat array.
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = np.unique(lo * n_items + hi)

    return pd.DataFrame({
        "asinA": pd.Categorical.from_codes(keys // n_items, vocabulary),
        "asinB": pd.Categorical.from_codes(keys % n_items, vocabulary),
    })


def save_co_purchase_pairs(df: pd.DataFrame, path: PathLike) -> None:
    """Persist the pair table as a pickle, both columns kept as `category`.

    Same reasoning as `categories.save_pairs`, and it matters more here: the
    table is tens of millions of rows over ~190k distinct asins, so `category`
    stores each side as an integer code against one shared vocabulary instead
    of repeating a 10-character string every time. Pickle keeps that dtype
    across the round trip; CSV would not.
    """
    df = df.copy()
    for col in ("asinA", "asinB"):
        df[col] = df[col].astype("category")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def run_co_purchase_pairs(
    interactions: Union[pd.DataFrame, PathLike],
    cutoff_time: float,
    window_days: int = DEFAULT_WINDOW_DAYS,
    out_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Whole pipeline: interaction log -> windowed, leak-free pair table.

    Parameters
    ----------
    interactions : DataFrame or path
        `Home_and_Kitchen_filtered.csv`, or the frame already loaded.
    cutoff_time : the split point, in the same units as the timestamp column.
        Required — there is no derived fallback, so no call site can end up
        splitting on a date nobody stated.
    window_days : maximum gap between the two purchases of a pair.
    out_path : path, optional
        If given, the result is written there with `save_co_purchase_pairs`.

    Returns
    -------
    DataFrame with columns `asinA`, `asinB`.
    """
    if isinstance(interactions, (str, Path)):
        interactions = load_interactions(interactions)

    pairs = co_purchase_pairs(
        interactions, cutoff_time=cutoff_time, window_days=window_days,
    )

    if out_path is not None:
        save_co_purchase_pairs(pairs, out_path)
    return pairs
