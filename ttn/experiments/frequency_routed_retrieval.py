"""Frequency-routed retrieval: split each query's candidate pool by the
CANDIDATE's own training-target frequency, score the warm half with TTN and
the cold half with CONTENT, and merge -- instead of asking both engines to
rank the same pool and reconciling two independent top-K lists.

Mechanism, per test pair (query, target_node):
  1. All candidates in the target's category are split into WARM (target_freq
     >= THRESHOLD, scored by TTN) and COLD (target_freq < THRESHOLD, scored
     by CONTENT) -- target_freq is precomputed once from the full training
     set, so this is knowable per catalogue item ahead of any query, not a
     per-query decision.
  2. The K slots (K=10, K=100) are split between the two pools PROPORTIONAL
     to how much of that category's candidate pool is warm vs cold -- e.g. a
     category that's 90% warm items gives TTN 9 of 10 slots. If one pool has
     fewer candidates than its slot budget, the shortfall rolls over to the
     other pool so the merged list still has K candidates when the category
     supports it.
  3. hit@K is membership of the true target in the merged set -- the same
     rule used throughout this notebook's similarity work, never
     `(scores > true).sum()`.

THRESHOLD is swept over a few values to see where routing helps most, next
to plain TTN+IMAGE, plain CONTENT (desc+title+features+image, 0.25 each --
this session's best of each), and POP for reference. This is a genuinely new
comparison -- nothing in the notebook or ttn/experiments/ has tried routed
retrieval with this slot-budget rule before; §19 only flagged it as future
work ("Use the switch as list selection, not a score blend").

Usage: python ttn/experiments/frequency_routed_retrieval.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "tower"
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]
N_SAMPLE = 20_000
SEED = 0
BUCKET_ORDER = ["never", "1-5", "6-25", "26-100", ">100", "all"]
THRESHOLDS = [5, 10, 25, 50, 100]

t0 = time.time()
vocabs = json.load(open(OUT_DIR / "vocabs.json"))
arrays_base = dict(np.load(OUT_DIR / "items.npz"))
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
asins = np.load(OUT_DIR / "item_asins.npy", allow_pickle=False)
title_emb = arrays_base["title_emb"].astype("float32")
desc_emb = arrays_base["desc_emb"].astype("float32")
img_emb = np.load(OUT_DIR / "siglip_img_emb.npy").astype("float32")
img_status = np.load(OUT_DIR / "siglip_img_status.npy")
n_items = len(asins)

pairs_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
print(f"loaded in {time.time() - t0:.0f}s | train {len(pairs_train):,} | "
      f"test {len(pairs_test):,}")

has_desc = ~np.all(desc_emb == 0, axis=1)
has_title = ~np.all(title_emb == 0, axis=1)
has_img = img_status == 1


def text_for(field):
    inv = {i: v for v, i in vocabs[field].items()}
    ids = arrays_base["cat_ids"][:, CAT_ORDER.index(field)]
    return np.array([inv.get(int(i), "Missing") for i in ids], dtype=object)


def encode_categorical(text_arr, sbert):
    uniq = sorted(set(text_arr))
    vecs = np.asarray(sbert.encode(uniq, batch_size=64, show_progress_bar=False))
    lut = dict(zip(uniq, vecs))
    out = np.stack([lut[t] for t in text_arr]).astype("float32")
    absent = np.array([t.strip() == "" or t == "Missing" for t in text_arr])
    out[absent] = 0.0
    return out, ~absent


from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer("all-MiniLM-L6-v2")
features_text = text_for("features")
features_emb, has_features = encode_categorical(features_text, sbert)
print(f"title present: {has_title.mean():.1%}  desc present: {has_desc.mean():.1%}  "
      f"features present: {has_features.mean():.1%}  image present: {has_img.mean():.1%}")

SIGNALS = {
    "desc": (desc_emb, has_desc),
    "title": (title_emb, has_title),
    "features": (features_emb, has_features),
    "image": (img_emb, has_img),
}
NORM = {k: v[0] / (np.linalg.norm(v[0], axis=1, keepdims=True) + 1e-9) for k, v in SIGNALS.items()}
HAS = {k: v[1] for k, v in SIGNALS.items()}
CONTENT_PARTS = ["desc", "title", "features", "image"]

# --- in-node candidate pools -------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


# --- POP + freq bucket + per-item target frequency, from the FULL training
# pairs table -- this is the per-item property routing is based on ----------
counts = (pairs_train.groupby(["target_node_id", "target_idx"]).size()
          .rename("n").reset_index())
top10_by_node = {int(k): set(g.nlargest(10, "n")["target_idx"].astype(int))
                  for k, g in counts.groupby("target_node_id")}
top100_by_node = {int(k): set(g.nlargest(100, "n")["target_idx"].astype(int))
                   for k, g in counts.groupby("target_node_id")}
target_freq_train = pairs_train.groupby("target_idx").size()
ITEM_FREQ = np.zeros(n_items, dtype="int64")
ITEM_FREQ[target_freq_train.index.to_numpy()] = target_freq_train.to_numpy()


def bucket(f):
    if f == 0:
        return "never"
    if f <= 5:
        return "1-5"
    if f <= 25:
        return "6-25"
    if f <= 100:
        return "26-100"
    return ">100"


# --- 20,000-pair test sample, seed 0 (same sample used throughout) ---------
rng = np.random.default_rng(SEED)
take = rng.choice(len(pairs_test), min(N_SAMPLE, len(pairs_test)), replace=False)
sample = pairs_test.iloc[take].reset_index(drop=True)
sample["freq"] = sample["target_idx"].map(target_freq_train).fillna(0).astype(int)
sample["bucket"] = sample["freq"].apply(bucket)
print(f"\ntest sample: {len(sample):,} pairs (seed {SEED})")
print(sample["bucket"].value_counts().reindex(BUCKET_ORDER[:-1]))

# =============================================================================
# TTN+IMAGE -- load the checkpoint trained earlier this session
# (data/tower/ttn_complementary_image.pt), read-only, never retrained here.
# =============================================================================
CHECKPOINT = OUT_DIR / "ttn_complementary_image.pt"
ck = torch.load(CHECKPOINT, weights_only=False)
cfg = ck["config"]
DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")

title_t = torch.tensor(title_emb, device=DEVICE)
USE_DESCRIPTION = cfg["use_description"]
USE_IMAGE = cfg.get("use_image", False)
desc_t = torch.tensor(desc_emb, device=DEVICE) if USE_DESCRIPTION else torch.zeros(n_items, 1, device=DEVICE)
img_t = torch.tensor(img_emb, device=DEVICE) if USE_IMAGE else torch.zeros(n_items, 1, device=DEVICE)
cat_t = torch.tensor(arrays_base["cat_ids"], device=DEVICE)
num_t = torch.tensor(arrays_base["numeric"], device=DEVICE)
mu, sd = ck["numeric_standardisation"]["mean"].to(DEVICE), ck["numeric_standardisation"]["std"].to(DEVICE)
num_t = (num_t - mu) / sd
CAT_DIM, NODE_DIM, HIDDEN, OUT_DIM_ = cfg["cat_dim"], cfg["node_dim"], cfg["hidden"], cfg["out_dim"]
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
            nn.ReLU(), nn.Linear(HIDDEN, OUT_DIM_))

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
        self.query_out = nn.Linear(OUT_DIM_ + NODE_DIM, OUT_DIM_)

    def query(self, idx, node_id):
        h = torch.cat([self.query_encoder(idx), self.node(node_id)], dim=-1)
        return F.normalize(self.query_out(h), dim=-1)

    def candidate(self, idx):
        return F.normalize(self.candidate_encoder(idx), dim=-1)


model = ComplementaryTwoTower().to(DEVICE)
model.load_state_dict(ck["state_dict"])
model.eval()
print(f"TTN+IMAGE checkpoint loaded (best_epoch {cfg.get('best_epoch')})")

with torch.no_grad():
    cand_vecs = torch.empty(n_items, OUT_DIM_, device=DEVICE)
    for i in range(0, n_items, 8192):
        j = min(i + 8192, n_items)
        cand_vecs[i:j] = model.candidate(torch.arange(i, j, device=DEVICE))

# =============================================================================
# per-pair: TTN scores + CONTENT scores over the FULL in-node candidate pool,
# then routed merges at every threshold, all from the same two score rows.
# =============================================================================
hit10 = {"TTN": np.zeros(len(sample), dtype=bool),
         "CONTENT": np.zeros(len(sample), dtype=bool),
         "POP": np.zeros(len(sample), dtype=bool)}
hit100 = {"TTN": np.zeros(len(sample), dtype=bool),
          "CONTENT": np.zeros(len(sample), dtype=bool),
          "POP": np.zeros(len(sample), dtype=bool)}
for thr in THRESHOLDS:
    hit10[f"ROUTE@{thr}"] = np.zeros(len(sample), dtype=bool)
    hit100[f"ROUTE@{thr}"] = np.zeros(len(sample), dtype=bool)


t1 = time.time()
n_groups = sample["target_node_id"].nunique()
for gi, (node_id, grp) in enumerate(sample.groupby("target_node_id")):
    node_id = int(node_id)
    cand = cands_for_node(node_id)
    if len(cand) < 2:
        continue
    rows = grp.index.to_numpy()
    q_idx_np = grp["query_idx"].to_numpy()
    t_idx = grp["target_idx"].to_numpy()
    cand_pos = {c: i for i, c in enumerate(cand)}
    t_pos = np.array([cand_pos.get(t, -1) for t in t_idx])
    self_mask = cand[None, :] == q_idx_np[:, None]
    cand_freq = ITEM_FREQ[cand]                      # (n_cand,) -- per-candidate, query-independent

    # --- TTN scores, full row over this node's candidates ------------------
    with torch.no_grad():
        q_idx_t = torch.tensor(q_idx_np, device=DEVICE)
        node_t = torch.full((len(grp),), node_id, device=DEVICE, dtype=torch.long)
        qv = model.query(q_idx_t, node_t)
        cv = cand_vecs[cand]
        ttn_scores = (qv @ cv.T).cpu().numpy()
    ttn_scores = np.where(self_mask, -np.inf, ttn_scores)

    # --- CONTENT scores, full row over this node's candidates --------------
    mats = []
    for part in CONTENT_PARTS:
        Q, C = NORM[part][q_idx_np], NORM[part][cand]
        M = Q @ C.T
        mask = np.outer(HAS[part][q_idx_np], HAS[part][cand])
        mats.append(np.where(mask, M, np.nan))
    with np.errstate(invalid="ignore"):
        content_scores = np.nanmean(np.stack(mats, axis=0), axis=0)
    content_scores = np.where(np.isnan(content_scores), -np.inf, content_scores)
    content_scores = np.where(self_mask, -np.inf, content_scores)

    for r, tp in enumerate(t_pos):
        if tp < 0:
            continue
        ttn_row = ttn_scores[r]
        content_row = content_scores[r]
        valid = ~self_mask[r]

        ttn_rank_all = np.argsort(-ttn_row)
        ttn_rank_all = ttn_rank_all[ttn_row[ttn_rank_all] != -np.inf]
        hit10["TTN"][rows[r]] = tp in set(ttn_rank_all[:10])
        hit100["TTN"][rows[r]] = tp in set(ttn_rank_all[:100])

        content_rank_all = np.argsort(-content_row)
        content_rank_all = content_rank_all[content_row[content_rank_all] != -np.inf]
        hit10["CONTENT"][rows[r]] = tp in set(content_rank_all[:10])
        hit100["CONTENT"][rows[r]] = tp in set(content_rank_all[:100])

        for thr in THRESHOLDS:
            warm = valid & (cand_freq >= thr)
            cold = valid & (cand_freq < thr)
            n_warm, n_cold = int(warm.sum()), int(cold.sum())
            warm_rank = np.argsort(-np.where(warm, ttn_row, -np.inf))
            cold_rank = np.argsort(-np.where(cold, content_row, -np.inf))
            for k, hits in ((10, hit10), (100, hit100)):
                if n_warm + n_cold == 0:
                    continue
                k_warm = int(round(k * n_warm / (n_warm + n_cold)))
                k_cold = k - k_warm
                # roll over any shortfall so the merged list still has k slots
                if k_warm > n_warm:
                    k_cold += k_warm - n_warm
                    k_warm = n_warm
                if k_cold > n_cold:
                    k_warm += k_cold - n_cold
                    k_cold = n_cold
                routed = set(warm_rank[:k_warm]) | set(cold_rank[:k_cold])
                hits[f"ROUTE@{thr}"][rows[r]] = tp in routed

    p10, p100 = top10_by_node.get(node_id, set()), top100_by_node.get(node_id, set())
    for r, t in enumerate(t_idx):
        hit10["POP"][rows[r]] = int(t) in p10
        hit100["POP"][rows[r]] = int(t) in p100

    if (gi + 1) % 50 == 0 or gi + 1 == n_groups:
        print(f"  node {gi + 1}/{n_groups}  [{time.time() - t1:.0f}s]", flush=True)
print(f"\nscoring done in {time.time() - t1:.0f}s")

# =============================================================================
# report
# =============================================================================
COLS = ["TTN", "CONTENT", "POP"] + [f"ROUTE@{t}" for t in THRESHOLDS]


def table(hits, title):
    print(f"\n{title}")
    header = f"{'target freq':<12}{'pairs':>8}" + "".join(f"{c:>12}" for c in COLS)
    print(header)
    for b in BUCKET_ORDER:
        m = sample["bucket"] == b if b != "all" else slice(None)
        n = int(m.sum()) if b != "all" else len(sample)
        row = f"{b:<12}{n:>8}"
        for c in COLS:
            v = hits[c][sample.index[m]].mean() if b != "all" else hits[c].mean()
            row += f"{v:>12.4f}"
        print(row)


table(hit10, f"Recall@10  (test sample {len(sample):,} pairs, seed {SEED}) "
             f"-- TTN=TTN+IMAGE, CONTENT=desc+title+features+image")
table(hit100, f"Recall@100  (test sample {len(sample):,} pairs, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
