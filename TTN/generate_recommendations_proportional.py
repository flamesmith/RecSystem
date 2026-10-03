"""Batch-generate "Complete the Look" (complements) recommendations for
every item -- same trained model and same output schema as
generate_recommendations.py, but a DIFFERENT merge rule across a query's
multiple licensed target categories.

generate_recommendations.py pools all licensed target categories' candidates
together and takes one flat top-20 by score -- one category can end up
dominating the list if it happens to score higher, with no guarantee every
licensed category is represented at all (see this session's discussion).

This script instead reproduces the `ttn_model` branch notebook's original
approach (`ttn/ttn_complementary.ipynb`, `allocate_slots`): each query's k
slots are split across its licensed target categories IN PROPORTION TO
each category's co-purchase evidence (`edges`, from
complementary_categories.pkl -- how much actual co-purchase traffic backs
that specific source-category -> target-category pair), via largest-
remainder apportionment, so the totals always sum to exactly k. The model
then independently scores candidates *within* each allocated category --
scores are only ever compared within one category, never across
categories, which is the only place the notebook found them to be
meaningful (the top-20 score span is tiny, ~0.003, an order smaller than
the gap between categories' raw evidence).

Trade-off, stated honestly: this guarantees every licensed category with a
nonzero share gets at least a chance at a slot (more diverse), at the cost
of sometimes filling a slot with a weaker same-category match instead of a
stronger match from a category that didn't get enough slots (less
"globally best"). generate_recommendations.py makes the opposite trade.
Neither is proven better here; this is the alternative to compare against.

No change needed to TTN/build_model.py -- `edges` is a property of
complementary_categories.pkl (built by data_creation/build_data.py, before
training), not something the model learns. The model only ever needs to
score "this item against this one category" (`model.query(idx, node_id)`),
which it already does; everything here is generation-time allocation on
top of that, exactly like generate_recommendations.py's flat merge is.

Output: TTN/generated/<snapshot>/recommendations/complements_proportional.parquet
-- a SEPARATE file from generate_recommendations.py's complements.parquet,
so both can be compared side by side rather than one overwriting the other.

Prerequisite: TTN/build_model.py must have already produced the given
--version under this --snapshot.

Usage: python TTN/generate_recommendations_proportional.py --snapshot w90_2017-12-09 --version 2026-09-19_v_001
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
NODE_SEP = " > "
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_ttn_arrays.py, e.g. w90_2017-12-09")
parser.add_argument("--version", required=True,
                     help="version_id from TTN/build_model.py, e.g. 2026-09-19_v_001 "
                          "(deliberately no 'latest'/'champion' default -- be explicit "
                          "about which candidate is being scored)")
args = parser.parse_args()

t0 = time.time()
SNAPSHOT_DIR = ROOT / "data" / "tower" / args.snapshot              # shared artifacts
TTN_DIR = ROOT / "TTN" / "generated" / args.snapshot                # TTN-specific artifacts
SIGLIP_DIR = ROOT / "SigLIP2" / "generated" / args.snapshot         # optional cross-read
VERSION_DIR = TTN_DIR / "models" / args.version
OUT_DIR = TTN_DIR / "recommendations"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "complements_proportional.parquet"

import json
vocabs = json.load(open(TTN_DIR / "vocabs.json"))
arrays = dict(np.load(TTN_DIR / "items.npz"))
node_of_item = np.load(SNAPSHOT_DIR / "node_of_item.npy")
asins = np.load(SNAPSHOT_DIR / "item_asins.npy", allow_pickle=False).astype(str)
n_items = len(asins)
if (TTN_DIR / "desc_emb.npy").exists():
    arrays["desc_emb"] = np.load(TTN_DIR / "desc_emb.npy")
if (SIGLIP_DIR / "siglip_img_emb.npy").exists():
    arrays["img_emb"] = np.load(SIGLIP_DIR / "siglip_img_emb.npy")
print(f"loaded {TTN_DIR.relative_to(ROOT)}/ -- {n_items:,} items, version {args.version}")

# ============================================================================
# Licensed target categories PER SOURCE NODE, with co-purchase evidence
# (`edges`) -- read directly from complementary_categories.pkl rather than
# pairs_train, since pairs_train only records WHICH (query, target_node)
# pairs were licensed, not how much co-purchase evidence backed each one.
# ============================================================================
comp_cat = pd.read_pickle(ROOT / "data" / "complementary_categories.pkl")
comp_cat["src_node_str"] = (comp_cat["src_cat_2"].astype(str) + NODE_SEP
                             + comp_cat["src_cat_3"].astype(str) + NODE_SEP
                             + comp_cat["src_cat_4"].astype(str))
comp_cat["dst_node_str"] = (comp_cat["dst_cat_2"].astype(str) + NODE_SEP
                             + comp_cat["dst_cat_3"].astype(str) + NODE_SEP
                             + comp_cat["dst_cat_4"].astype(str))
comp_cat["src_node_id"] = comp_cat["src_node_str"].map(vocabs["target_node"])
comp_cat["dst_node_id"] = comp_cat["dst_node_str"].map(vocabs["target_node"])
# Only pairs where BOTH ends exist in this snapshot's own vocab -- a category
# absent from this snapshot (too few items, or filtered upstream) can be
# neither a source nor a usable target here.
comp_cat = comp_cat.dropna(subset=["src_node_id", "dst_node_id"])
comp_cat["src_node_id"] = comp_cat["src_node_id"].astype(int)
comp_cat["dst_node_id"] = comp_cat["dst_node_id"].astype(int)
comp_cat = comp_cat[comp_cat["src_node_id"] != comp_cat["dst_node_id"]]   # self pairs excluded, as in training

licensed = {src: list(zip(grp["dst_node_id"], grp["edges"]))
            for src, grp in comp_cat.sort_values("edges", ascending=False).groupby("src_node_id")}
print(f"source categories with >=1 licensed target: {len(licensed):,}")


def allocate_slots(shares, k):
    """Largest-remainder apportionment: k slots split by share, summing to k.

    Ported from ttn_model branch's ttn/ttn_complementary.ipynb -- unchanged.
    """
    raw = np.asarray(shares, dtype=float) * k
    base = np.floor(raw).astype(int)
    remaining = k - base.sum()
    if remaining > 0:
        for j in np.argsort(-(raw - base))[:remaining]:
            base[j] += 1
    return base


# --- in-node candidate pools -------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


# ============================================================================
# Load the model -- identical to generate_recommendations.py
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
# Score every item as query, allocating its k=20 slots across its licensed
# target categories by co-purchase-evidence share, scoring each category
# independently (never comparing scores across categories, only within).
# ============================================================================
rows = []
t1 = time.time()
source_nodes = sorted(licensed.keys())
with torch.no_grad():
    for gi, source_node in enumerate(source_nodes):
        queries = cands_for_node(int(source_node))
        if len(queries) == 0:
            continue

        targets = licensed[source_node]                        # [(target_node_id, edges), ...]
        edges = np.array([e for _, e in targets], dtype=float)
        shares = edges / edges.sum()
        slots = allocate_slots(shares, TOP_K)                  # one slot-count per target, summing to TOP_K

        # Per-query accumulators across this source_node's target categories.
        per_query_rows = {int(q): [] for q in queries}

        for (target_node, edge_count), n_slots in zip(targets, slots):
            if n_slots == 0:
                continue
            cand = cands_for_node(int(target_node))
            if len(cand) == 0:
                continue
            cand_t = torch.tensor(cand, device=DEVICE)
            cv = cand_vecs[cand_t]
            k = min(int(n_slots), len(cand))

            for b in range(0, len(queries), 2048):
                batch = queries[b:b + 2048]
                q_idx = torch.tensor(batch, device=DEVICE)
                node_t = torch.full((len(batch),), int(target_node), device=DEVICE, dtype=torch.long)
                qv = model.query(q_idx, node_t)
                scores = (qv @ cv.T).cpu().numpy()
                self_mask = cand[None, :] == batch[:, None]
                scores = np.where(self_mask, -np.inf, scores)

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
                    scores_here = ranked_scores[qi, :n_valid]
                    for pos, sc in zip(cand_pos, scores_here):
                        per_query_rows[int(q_item)].append((
                            int(target_node), asins[cand[pos]], float(sc)
                        ))

        for q_item, entries in per_query_rows.items():
            if not entries:
                continue
            rows.append(pd.DataFrame({
                "query_asin": asins[q_item],
                "rank": np.arange(1, len(entries) + 1),
                "candidate_asin": [e[1] for e in entries],
                "score": pd.array([e[2] for e in entries], dtype="float32"),
                "target_node_id": [e[0] for e in entries],
            }))

        if (gi + 1) % 50 == 0 or gi + 1 == len(source_nodes):
            print(f"  source node {gi + 1}/{len(source_nodes)}  [{time.time() - t1:.0f}s]", flush=True)

out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
    columns=["query_asin", "rank", "candidate_asin", "score", "target_node_id"])
out.to_parquet(OUT_PATH)
print(f"\nqueries with >=1 recommendation: {out['query_asin'].nunique():,} of {n_items:,}")
print(f"written -> {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows, "
      f"{OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(f"total runtime: {time.time() - t0:.0f}s")
