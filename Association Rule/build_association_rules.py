"""Association-rule "bought together" recommendations, steered by TTN's
complementary categories.

Not a trained model. For each query item, the 20 recommendation slots are split
across the target categories TTN licenses for the query's own category, in
proportion to each category pair's co-purchase evidence (`edges`); inside each
target category, candidates are ranked by raw item-pair support.

What is reused from TTN, unchanged:
  * `data/complementary_categories.pkl` -- the category-pair table already
    filtered by `edges >= 5 AND lift >= 2.0` (data_creation/complementary_cats_pairs/
    categories.py:filter_pairs). A (source category -> target category) row there
    is what licenses a target category for a query. Direction matters.
  * `edges` as the co-purchase weight, and TTN/generate_recommendations_proportional.py's
    `allocate_slots` (largest-remainder apportionment, totals always sum to k).
  * the cat_4 whitelist fold (`category_taxonomy.json`), so category paths match
    complementary_categories.pkl's.
  * the train/test cutoff `TTN/constants.json:date_threshold` (strictly-before) and
    the reviewer-within-`window_days` co-purchase definition of
    data_creation/complementary_cats_pairs/pairs.py.
  * self-pairs (query category == target category) are excluded, as in TTN's
    recommendation step; complementary_categories.pkl contains them.

What is new here -- item-level association rules:
  * pair_count(A, B) = number of DISTINCT reviewers who bought both A and B with
    the two purchases at most `window_days` apart, before the cutoff.
    `pairs.co_purchase_pairs` is not reused because it deliberately keeps each
    pair once however many reviewers bought it, which discards the count.
  * support(A, B) = pair_count / number of distinct reviewers before the cutoff.
    It is a constant multiple of pair_count, so ranking by either is identical.
  * Candidates must be licensed: (category of A -> category of B) must be a row
    of complementary_categories.pkl. Both ends need a row in df_features.pkl.
  * Ties on pair_count are broken by candidate asin, ascending. Ties are common
    (most pairs are co-purchased once), so who falls on the cut-off is arbitrary
    but deterministic.

Slot allocation, per query item:
  1. Round 1: split k=20 slots across ALL licensed target categories of the
     query's category by `edges` share. A category can take at most as many
     items as the query has licensed candidates in it.
  2. Unfilled slots are redistributed over the target categories that still
     have unused candidates, again by `edges` share, and repeated until k are
     filled or no candidate is left. A query with fewer than k licensed
     candidates in total gets fewer than k rows -- nothing is back-filled.
  Output order: target categories by `edges` descending, items inside a category
  by pair_count descending. Pair counts are never compared across categories.

No query item with no licensed co-purchased candidate appears in the output.

Usage: python "Association Rule/build_association_rules.py" [--window-days 90]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_creation.complementary_cats_pairs.categories import fold_cat_4, load_taxonomy
from data_creation.complementary_cats_pairs.pairs import (
    ITEM_COL, SECONDS_PER_DAY, TIME_COL, USER_COL, load_interactions,
)

TOP_K = 20          # matches TTN's, SigLIP2's and Popularity's recommendation depth
NODE_SEP = " > "
DEFAULT_WINDOW_DAYS = 90    # TTN production window


def allocate_slots(shares, k):
    """Largest-remainder apportionment: k slots split by share, summing to k.

    Same function as TTN/generate_recommendations_proportional.py.
    """
    raw = np.asarray(shares, dtype=float) * k
    base = np.floor(raw).astype(int)
    remaining = k - base.sum()
    if remaining > 0:
        for j in np.argsort(-(raw - base))[:remaining]:
            base[j] += 1
    return base


def co_purchase_counts(df, cutoff_time, window_days):
    """Distinct-reviewer count per unordered item pair.

    Same pairing rule as pairs.co_purchase_pairs (strictly before the cutoff;
    exact duplicate (user, item, time) rows dropped; a pair counts if ANY event
    of A and ANY event of B are within `window_days`; an item never pairs with
    itself), but each pair is counted once PER REVIEWER instead of once overall.

    Returns (item_vocabulary, lo, hi, n_reviewers, n_users): `lo`/`hi` are codes
    into item_vocabulary (lo < hi, so alphabetical by asin), `n_reviewers` the
    number of distinct reviewers who bought both, `n_users` the number of
    distinct reviewers in the pre-cutoff log (the support denominator).
    """
    d = df[[USER_COL, ITEM_COL, TIME_COL]]
    d = d[d[TIME_COL] < cutoff_time].drop_duplicates()

    users = d[USER_COL].astype("category").cat.codes.to_numpy()
    item_cat = d[ITEM_COL].astype("category")
    items = item_cat.cat.codes.to_numpy()
    vocabulary = item_cat.cat.categories
    n_items = len(vocabulary)
    n_users = int(users.max()) + 1
    times = d[TIME_COL].to_numpy(dtype=np.int64)

    order = np.lexsort((times, users))
    users, items, times = users[order], items[order], times[order]

    window = np.int64(window_days) * SECONDS_PER_DAY
    stride = np.int64(times.max()) + window + 1
    key = users.astype(np.int64) * stride + times
    end = np.searchsorted(key, key + window, side="right")

    idx = np.arange(len(key))
    counts = end - idx - 1
    starts = np.cumsum(counts) - counts
    left = np.repeat(idx, counts)
    right = left + 1 + (np.arange(counts.sum()) - np.repeat(starts, counts))

    a, b = items[left], items[right]
    u = users[left]
    keep = a != b
    a, b, u = a[keep], b[keep], u[keep]
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)

    # (user, lo, hi) must fit in int64 for the single-integer dedupe below.
    assert n_users * n_items * n_items < np.iinfo(np.int64).max, "key overflow"
    pair_key = lo * n_items + hi
    user_pair = np.unique(u.astype(np.int64) * (n_items * n_items) + pair_key)
    pair_key, n_reviewers = np.unique(user_pair % (n_items * n_items), return_counts=True)
    return vocabulary, pair_key // n_items, pair_key % n_items, n_reviewers, n_users


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                        help="max gap between the two purchases of a pair (default 90, "
                             "TTN production). Unrelated to Popularity's recency window.")
    args = parser.parse_args()
    t0 = time.time()

    data_dir = ROOT / "data"
    date_threshold = json.loads((ROOT / "TTN" / "constants.json").read_text())["date_threshold"]
    cutoff_time = pd.Timestamp(date_threshold).timestamp()
    snapshot_id = f"w{args.window_days}_{date_threshold}"
    out_dir = ROOT / "Association Rule" / "generated" / snapshot_id / "recommendations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "association_rules.parquet"
    print(f"window_days={args.window_days}, date_threshold={date_threshold} ({snapshot_id})")

    # ------------------------------------------------------------------
    # 1. Item categories (df_features), cat_4 folded as everywhere else
    # ------------------------------------------------------------------
    valid_pairs = load_taxonomy(data_dir / "category_taxonomy.json")
    feats = pd.read_pickle(data_dir / "df_features.pkl")[["asin", "cat_2", "cat_3", "cat_4"]]
    feats = feats.drop_duplicates("asin")
    feats["cat_4"] = fold_cat_4(feats["cat_3"], feats["cat_4"], valid_pairs)
    for c in ("cat_2", "cat_3"):
        feats[c] = feats[c].astype(str)
    feats["node"] = feats["cat_2"] + NODE_SEP + feats["cat_3"] + NODE_SEP + feats["cat_4"]
    print(f"items with a category (df_features.pkl): {len(feats):,}")

    # ------------------------------------------------------------------
    # 2. Licensed category pairs and their co-purchase weight (`edges`)
    # ------------------------------------------------------------------
    comp = pd.read_pickle(data_dir / "complementary_categories.pkl")
    comp["src_node"] = (comp["src_cat_2"].astype(str) + NODE_SEP + comp["src_cat_3"].astype(str)
                        + NODE_SEP + comp["src_cat_4"].astype(str))
    comp["dst_node"] = (comp["dst_cat_2"].astype(str) + NODE_SEP + comp["dst_cat_3"].astype(str)
                        + NODE_SEP + comp["dst_cat_4"].astype(str))
    n_self = int((comp["src_node"] == comp["dst_node"]).sum())
    comp = comp[comp["src_node"] != comp["dst_node"]]
    print(f"complementary_categories.pkl: {len(comp):,} directed pairs after dropping "
          f"{n_self:,} same-category pairs")

    node_vocab = pd.Index(sorted(set(feats["node"]) | set(comp["src_node"]) | set(comp["dst_node"])))
    n_nodes = len(node_vocab)
    comp_src = node_vocab.get_indexer(comp["src_node"])
    comp_dst = node_vocab.get_indexer(comp["dst_node"])
    comp_edges = comp["edges"].to_numpy(dtype=np.int64)
    licensed_edges = pd.Series(comp_edges, index=comp_src.astype(np.int64) * n_nodes + comp_dst)

    # ------------------------------------------------------------------
    # 3. Reviewer-level item-pair counts, train period only
    # ------------------------------------------------------------------
    reviews = load_interactions(data_dir / "Home_and_Kitchen_filtered.csv")
    vocabulary, lo, hi, n_rev, n_users = co_purchase_counts(reviews, cutoff_time, args.window_days)
    del reviews
    print(f"distinct item pairs: {len(lo):,} | reviewers before cutoff: {n_users:,} "
          f"[{time.time() - t0:.0f}s]")

    # item code (review vocabulary, alphabetical by asin) -> node id; -1 = no features
    feat_node = pd.Series(node_vocab.get_indexer(feats["node"]), index=feats["asin"].to_numpy())
    node_of = feat_node.reindex(vocabulary.astype(str)).fillna(-1).to_numpy(dtype=np.int64)
    n_a, n_b = node_of[lo], node_of[hi]
    has_feat = (n_a >= 0) & (n_b >= 0)
    lo, hi, n_a, n_b, n_rev = lo[has_feat], hi[has_feat], n_a[has_feat], n_b[has_feat], n_rev[has_feat]
    print(f"pairs with features at both ends: {len(lo):,}")

    # ------------------------------------------------------------------
    # 4. Directed, licensed (query -> candidate) rows
    # ------------------------------------------------------------------
    frames = []
    for q, c, nq, nc in ((lo, hi, n_a, n_b), (hi, lo, n_b, n_a)):
        e = licensed_edges.reindex(nq * n_nodes + nc).to_numpy()
        ok = ~np.isnan(e)
        frames.append(pd.DataFrame({
            "q": q[ok].astype(np.int32), "c": c[ok].astype(np.int32),
            "tnode": nc[ok].astype(np.int32), "n": n_rev[ok].astype(np.int32),
            "edges": e[ok].astype(np.int64)}))
    D = pd.concat(frames, ignore_index=True)
    del frames
    print(f"directed licensed (query, candidate) rows: {len(D):,}")
    # item codes are alphabetical by asin, so ascending `c` is the asin tie-break
    D = D.sort_values(["q", "tnode", "n", "c"], ascending=[True, True, False, True],
                      ignore_index=True)

    # licensed target categories per source node, edges descending (stable)
    lic = {}
    for src, grp in comp.assign(s=comp_src, d=comp_dst).sort_values(
            "edges", ascending=False, kind="stable").groupby("s"):
        lic[int(src)] = (grp["d"].to_numpy(), grp["edges"].to_numpy(dtype=float))

    # ------------------------------------------------------------------
    # 5. Per query: edges-proportional slots with redistribution
    # ------------------------------------------------------------------
    q_arr, t_arr = D["q"].to_numpy(), D["tnode"].to_numpy()
    q_bounds = np.flatnonzero(np.r_[True, q_arr[1:] != q_arr[:-1], True])
    sel_rows = []
    n_redistributed = 0
    for qi in range(len(q_bounds) - 1):
        s, e = q_bounds[qi], q_bounds[qi + 1]
        q = int(q_arr[s])
        dst_nodes, edges = lic[int(node_of[q])]
        # candidate block of each licensed target category inside rows s:e
        t = t_arr[s:e]
        cat_start = np.full(len(dst_nodes), -1, dtype=np.int64)
        avail = np.zeros(len(dst_nodes), dtype=np.int64)
        starts_t = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
        ends_t = np.r_[starts_t[1:], len(t)]
        pos = {int(d): j for j, d in enumerate(dst_nodes)}
        for a, b in zip(starts_t, ends_t):
            j = pos[int(t[a])]
            cat_start[j], avail[j] = s + a, b - a

        taken = np.minimum(allocate_slots(edges / edges.sum(), TOP_K), avail)   # round 1
        redistributed = False
        while taken.sum() < TOP_K:
            active = avail > taken
            if not active.any():
                break
            redistributed = True
            share = edges[active] / edges[active].sum()
            alloc = allocate_slots(share, TOP_K - int(taken.sum()))
            taken[active] += np.minimum(alloc, avail[active] - taken[active])
        n_redistributed += redistributed

        for j in np.flatnonzero(taken):
            sel_rows.append(np.arange(cat_start[j], cat_start[j] + taken[j]))

    rows = np.concatenate(sel_rows)
    out = D.iloc[rows].reset_index(drop=True)
    out["rank"] = out.groupby("q").cumcount() + 1
    n_queries = out["q"].nunique()

    node_parts = node_vocab.to_series().str.split(NODE_SEP, n=2, expand=True)
    asins = np.asarray(vocabulary.astype(str))
    final = pd.DataFrame({
        "query_asin": asins[out["q"]],
        "rank": out["rank"].astype("int64"),
        "candidate_asin": asins[out["c"]],
        "pair_count": out["n"],
        "support": (out["n"] / n_users).astype("float64"),
        "target_cat_2": node_parts[0].to_numpy()[out["tnode"]],
        "target_cat_3": node_parts[1].to_numpy()[out["tnode"]],
        "target_cat_4": node_parts[2].to_numpy()[out["tnode"]],
        "category_edges": out["edges"],
    })
    final.to_parquet(out_path)

    per_q = final.groupby("query_asin").size()
    print(f"\nqueries with >=1 recommendation: {n_queries:,} of {len(feats):,} items with a category")
    print(f"  with a full {TOP_K}: {(per_q == TOP_K).sum():,} ({(per_q == TOP_K).mean():.1%}); "
          f"mean recs {per_q.mean():.1f}")
    print(f"  queries where unfilled slots were redistributed: {n_redistributed:,}")
    print(f"written -> {out_path.relative_to(ROOT)} ({len(final):,} rows, "
          f"{out_path.stat().st_size / 1e6:,.1f} MB)")
    print(f"total runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
