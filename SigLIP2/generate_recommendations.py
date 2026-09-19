"""Batch-generate "Visually Similar Products" recommendations for every item.

For every catalogue item as query, score its own category's candidate pool
by SigLIP2 image cosine similarity (self excluded), keep the top-20, write
one parquet in the locked recommendation-generation schema: query_asin |
rank | candidate_asin | score. This is the file filtering/serving prep and,
eventually, the serving DB actually read -- not the raw embeddings.

Same in-node candidate-pool discipline used throughout this project's
evaluation: candidates restricted to node_of_item == the query's own node,
top-k by membership (no scores-vs-true-score comparison, which credits
zero-vector ties -- see this session's earlier description-similarity
postmortem). An item with no usable image (about 28% of the catalogue)
can't be ranked by this signal and is skipped as a query entirely.

Prerequisite: SigLIP2/build_siglip2.py must have already run for the given
--snapshot, to produce that snapshot's siglip_img_emb.npy / _status.npy.

Usage: python SigLIP2/generate_recommendations.py --snapshot w90_2017-12-09
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = 20

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from TTN/build_data.py, e.g. w90_2017-12-09")
SNAPSHOT_DIR = ROOT / "data" / "tower" / parser.parse_args().snapshot
OUT_DIR = SNAPSHOT_DIR / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "substitutes.parquet"

t0 = time.time()
asins = np.load(SNAPSHOT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
node_of_item = np.load(SNAPSHOT_DIR / "node_of_item.npy")
img_emb = np.load(SNAPSHOT_DIR / "siglip_img_emb.npy").astype("float32")
img_status = np.load(SNAPSHOT_DIR / "siglip_img_status.npy")
has_img = img_status == 1
n_items = len(asins)
print(f"items: {n_items:,} | with image: {has_img.sum():,} ({has_img.mean():.1%})")

img_norm = img_emb / (np.linalg.norm(img_emb, axis=1, keepdims=True) + 1e-9)

# --- in-node candidate pools, same construction used throughout this
# project's evaluation scripts --------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


rows = []
n_nodes = len(np.unique(node_of_item[node_of_item > 0]))
t1 = time.time()
for gi, node_id in enumerate(sorted(set(node_of_item[node_of_item > 0].tolist()))):
    cand = cands_for_node(node_id)
    if len(cand) < 2:
        continue
    cand_has_img = has_img[cand]
    if not cand_has_img.any():
        continue
    queries = cand[has_img[cand]]      # only items with an image can query this signal
    Q = img_norm[queries]
    C = img_norm[cand]
    scores = Q @ C.T                                    # (n_query, n_cand)
    scores = np.where(cand_has_img[None, :], scores, -np.inf)
    self_mask = cand[None, :] == queries[:, None]
    scores = np.where(self_mask, -np.inf, scores)

    k = min(TOP_K, scores.shape[1])
    top_idx = np.argpartition(-scores, kth=min(k - 1, scores.shape[1] - 1), axis=1)[:, :k]
    order_k = np.argsort(-np.take_along_axis(scores, top_idx, axis=1), axis=1)
    ranked = np.take_along_axis(top_idx, order_k, axis=1)
    ranked_scores = np.take_along_axis(scores, ranked, axis=1)

    for qi, q_item in enumerate(queries):
        valid = ranked_scores[qi] != -np.inf
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        cand_pos = ranked[qi, :n_valid]
        rows.append(pd.DataFrame({
            "query_asin": asins[q_item],
            "rank": np.arange(1, n_valid + 1),
            "candidate_asin": asins[cand[cand_pos]],
            "score": ranked_scores[qi, :n_valid].astype("float32"),
        }))

    if (gi + 1) % 50 == 0 or gi + 1 == n_nodes:
        print(f"  node {gi + 1}/{n_nodes}  [{time.time() - t1:.0f}s]", flush=True)

out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["query_asin", "rank", "candidate_asin", "score"])
out.to_parquet(OUT_PATH)
print(f"\nqueries with >=1 recommendation: {out['query_asin'].nunique():,} of {n_items:,}")
print(f"written -> {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows, "
      f"{OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(f"total runtime: {time.time() - t0:.0f}s")
