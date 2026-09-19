"""Batch-generate "Complete the Look" (complements) recommendations for
every item, from a specific trained TTN version.

For every catalogue item as query, score candidates in its licensed
target/complementary categories (self excluded), keep the top-20 overall,
write one parquet in the locked recommendation-generation schema:
query_asin | rank | candidate_asin | score.

DESIGN CALL, not yet settled elsewhere: the complementary-category mapping
is one-to-many (e.g. a desk may license both "desk lamps" and "office
chairs" as targets), but the locked schema has no target_node column. This
script merges candidates across ALL of a query's licensed target
categories into one pool and ranks by score across that merged pool --
scores are comparable as raw cosine similarities in the shared embedding
space, though the query projection itself differs per target_node (the
node embedding is concatenated in before scoring), so this is a reasonable
default, not a proven-optimal one. Revisit if per-target-category lists
turn out to matter more than one merged "best complements" list.

Licensed target categories are read off pairs_train rather than re-joining
complementary_categories.pkl: every (query, target_node) pair in
pairs_train was already licensed by that mapping when build_data.py built
it, so grouping "query's own node -> the target_node values seen for it in
training" reproduces the same licensing relationship for free.

Same in-node candidate-pool discipline as SigLIP2/generate_recommendations.py
and every evaluation script this session: hit@k by top-k membership, an
item with a known reserved id 0 node is skipped.

Prerequisite: TTN/build_model.py must have already produced the given
--version under this --snapshot.

Usage: python TTN/generate_recommendations.py --snapshot w90_2017-12-09 --version 2026-09-19_v_001
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TOP_K = 20
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from TTN/build_data.py, e.g. w90_2017-12-09")
parser.add_argument("--version", required=True,
                     help="version_id from TTN/build_model.py, e.g. 2026-09-19_v_001 "
                          "(deliberately no 'latest'/'champion' default -- be explicit "
                          "about which candidate is being scored)")
args = parser.parse_args()

t0 = time.time()
SNAPSHOT_DIR = ROOT / "data" / "tower" / args.snapshot
VERSION_DIR = SNAPSHOT_DIR / "models" / "ttn" / args.version
OUT_DIR = SNAPSHOT_DIR / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "complements.parquet"

import json
vocabs = json.load(open(SNAPSHOT_DIR / "vocabs.json"))
arrays = dict(np.load(SNAPSHOT_DIR / "items.npz"))
node_of_item = np.load(SNAPSHOT_DIR / "node_of_item.npy")
asins = np.load(SNAPSHOT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
pairs_train = pd.read_parquet(SNAPSHOT_DIR / "pairs_train.parquet")
n_items = len(asins)
if (SNAPSHOT_DIR / "desc_emb.npy").exists():
    arrays["desc_emb"] = np.load(SNAPSHOT_DIR / "desc_emb.npy")
if (SNAPSHOT_DIR / "siglip_img_emb.npy").exists():
    arrays["img_emb"] = np.load(SNAPSHOT_DIR / "siglip_img_emb.npy")
print(f"loaded {SNAPSHOT_DIR.relative_to(ROOT)}/ -- {n_items:,} items, version {args.version}")

# --- licensed target categories per source node, derived from pairs_train -
query_node = node_of_item[pairs_train["query_idx"].to_numpy()]
licensed = (pd.DataFrame({"source_node": query_node,
                          "target_node": pairs_train["target_node_id"].to_numpy()})
            .drop_duplicates().groupby("source_node")["target_node"].apply(list).to_dict())
print(f"source categories with >=1 licensed target: {len(licensed):,}")

# --- in-node candidate pools -------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


# ============================================================================
# Load the model
# ============================================================================
ck = torch.load(VERSION_DIR / "model.pt", weights_only=False)
cfg = ck["config"]
DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")

title_t = torch.tensor(arrays["title_emb"], device=DEVICE)
USE_DESCRIPTION = cfg["use_description"]
desc_t = (torch.tensor(arrays["desc_emb"], device=DEVICE) if USE_DESCRIPTION
          else torch.zeros(n_items, 1, device=DEVICE))
USE_IMAGE = cfg.get("use_image", False)
img_t = torch.tensor(arrays["img_emb"], device=DEVICE) if USE_IMAGE else torch.zeros(n_items, 1, device=DEVICE)
cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)
num_t = torch.tensor(arrays["numeric"], device=DEVICE)
mu, sd = ck["numeric_standardisation"]["mean"].to(DEVICE), ck["numeric_standardisation"]["std"].to(DEVICE)
num_t = (num_t - mu) / sd
CAT_DIM, NODE_DIM, HIDDEN, OUT_DIM = cfg["cat_dim"], cfg["node_dim"], cfg["hidden"], cfg["out_dim"]
vocab_sizes = ck["vocab_sizes"]


class ProductEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_sizes[name] + 1, CAT_DIM, padding_idx=0)
            for name in CAT_ORDER])
        self.title = nn.Linear(title_t.shape[1], 128)
        self.description = nn.Linear(desc_t.shape[1], 128) if USE_DESCRIPTION else None
        self.image = nn.Linear(img_t.shape[1], 128) if USE_IMAGE else None
        self.numeric = nn.Linear(num_t.shape[1], 16)
        self.norm_cat = nn.LayerNorm(len(CAT_ORDER) * CAT_DIM, elementwise_affine=False)
        self.norm_title = nn.LayerNorm(128, elementwise_affine=False)
        self.norm_description = nn.LayerNorm(128, elementwise_affine=False) if USE_DESCRIPTION else None
        self.norm_image = nn.LayerNorm(128, elementwise_affine=False) if USE_IMAGE else None
        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(len(CAT_ORDER) * CAT_DIM + 128
                      + (128 if USE_DESCRIPTION else 0)
                      + (128 if USE_IMAGE else 0) + 16, HIDDEN),
            nn.ReLU(), nn.Linear(HIDDEN, OUT_DIM))

    def forward(self, idx):
        cat = torch.cat([emb(cat_t[idx, j]) for j, emb in enumerate(self.embeddings)], dim=-1)
        parts = [self.norm_cat(cat), self.norm_title(self.title(title_t[idx]))]
        if self.description is not None:
            parts.append(self.norm_description(self.description(desc_t[idx])))
        if self.image is not None:
            parts.append(self.norm_image(self.image(img_t[idx])))
        parts.append(self.norm_numeric(self.numeric(num_t[idx])))
        return self.mlp(torch.cat(parts, dim=-1))


class ComplementaryTwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = ProductEncoder()
        self.candidate_encoder = ProductEncoder()
        self.node = nn.Embedding(vocab_sizes["target_node"] + 1, NODE_DIM, padding_idx=0)
        self.query_out = nn.Linear(OUT_DIM + NODE_DIM, OUT_DIM)

    def query(self, idx, node_id):
        h = torch.cat([self.query_encoder(idx), self.node(node_id)], dim=-1)
        return F.normalize(self.query_out(h), dim=-1)

    def candidate(self, idx):
        return F.normalize(self.candidate_encoder(idx), dim=-1)


model = ComplementaryTwoTower().to(DEVICE)
model.load_state_dict(ck["state_dict"])
model.eval()
print(f"model loaded (best_epoch {cfg.get('best_epoch')}, use_description={USE_DESCRIPTION}, "
      f"use_image={USE_IMAGE})")

with torch.no_grad():
    cand_vecs = torch.empty(n_items, OUT_DIM, device=DEVICE)
    for i in range(0, n_items, 8192):
        j = min(i + 8192, n_items)
        cand_vecs[i:j] = model.candidate(torch.arange(i, j, device=DEVICE))

# ============================================================================
# Score every item as query, against the union of its licensed target
# categories' candidates
# ============================================================================
rows = []
t1 = time.time()
source_nodes = sorted(licensed.keys())
with torch.no_grad():
    for gi, source_node in enumerate(source_nodes):
        queries = cands_for_node(int(source_node))
        if len(queries) == 0:
            continue
        target_nodes = licensed[source_node]
        cand = np.unique(np.concatenate([cands_for_node(int(tn)) for tn in target_nodes]))
        if len(cand) < 2:
            continue
        cand_t = torch.tensor(cand, device=DEVICE)
        cv = cand_vecs[cand_t]

        for b in range(0, len(queries), 2048):
            batch = queries[b:b + 2048]
            q_idx = torch.tensor(batch, device=DEVICE)
            # every query in this source node shares the same licensed-target
            # merged pool, but the node embedding used is each query's OWN
            # first licensed target (a query with multiple licensed targets
            # still gets one query projection here -- see module docstring).
            node_t = torch.full((len(batch),), int(target_nodes[0]), device=DEVICE, dtype=torch.long)
            qv = model.query(q_idx, node_t)
            scores = (qv @ cv.T).cpu().numpy()
            self_mask = cand[None, :] == batch[:, None]
            scores = np.where(self_mask, -np.inf, scores)

            k = min(TOP_K, scores.shape[1])
            top_idx = np.argpartition(-scores, kth=min(k - 1, scores.shape[1] - 1), axis=1)[:, :k]
            order_k = np.argsort(-np.take_along_axis(scores, top_idx, axis=1), axis=1)
            ranked = np.take_along_axis(top_idx, order_k, axis=1)
            ranked_scores = np.take_along_axis(scores, ranked, axis=1)

            for qi, q_item in enumerate(batch):
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

        if (gi + 1) % 50 == 0 or gi + 1 == len(source_nodes):
            print(f"  source node {gi + 1}/{len(source_nodes)}  [{time.time() - t1:.0f}s]", flush=True)

out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["query_asin", "rank", "candidate_asin", "score"])
out.to_parquet(OUT_PATH)
print(f"\nqueries with >=1 recommendation: {out['query_asin'].nunique():,} of {n_items:,}")
print(f"written -> {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows, "
      f"{OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(f"total runtime: {time.time() - t0:.0f}s")
