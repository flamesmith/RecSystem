"""Two new breakdowns of the five models used throughout this session (TTN,
TTN+IMAGE, DESC, POP, CONTENT), on the SAME 20k-pair test sample (seed 0)
used everywhere else:

1. Recall by the QUERY item's own training frequency (how many times this
   item was used as a query in training), instead of the target's frequency.
   Every bucketed table so far has been target-side ("will this item get
   recommended"); this asks the complementary question ("will a rarely-
   browsed item GET good recommendations"), which nothing in this session or
   the notebook has tested. Bucketed with the same never/1-5/6-25/26-100/>100
   scheme, computed from pairs_train's query_idx column.

2. Recall by the query item's top-level category (`cat_2`, 7 values) --
   a confound check on every frequency-bucketed result so far: if certain
   categories systematically have thinner co-purchase histories or are just
   harder to retrieve in, "TTN struggles below 25 observations" could partly
   be "TTN struggles in a few specific categories" wearing a frequency-shaped
   disguise.

Models: TTN and TTN+IMAGE load their committed checkpoints read-only (never
retrained here). CONTENT = desc+title+features+image (0.25 each, this
session's best). DESC = description-only cosine. POP = per-category top-10/
top-100 by training frequency. Same scoring discipline as every other script
this session: candidates restricted to the pair's target category, hit@k via
top-k membership, never `(scores > true).sum()`.

Usage: python ttn/experiments/query_side_breakdown.py
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
cat2_text = text_for("cat_2")
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


# --- POP, target frequency (unused bucketing here, kept for POP itself) ----
counts = (pairs_train.groupby(["target_node_id", "target_idx"]).size()
          .rename("n").reset_index())
top10_by_node = {int(k): set(g.nlargest(10, "n")["target_idx"].astype(int))
                  for k, g in counts.groupby("target_node_id")}
top100_by_node = {int(k): set(g.nlargest(100, "n")["target_idx"].astype(int))
                   for k, g in counts.groupby("target_node_id")}

# --- QUERY-side frequency: how often each item was used as a QUERY --------
query_freq_train = pairs_train.groupby("query_idx").size()


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


# --- the SAME 20,000-pair test sample used throughout this session ---------
rng = np.random.default_rng(SEED)
take = rng.choice(len(pairs_test), min(N_SAMPLE, len(pairs_test)), replace=False)
sample = pairs_test.iloc[take].reset_index(drop=True)
sample["qfreq"] = sample["query_idx"].map(query_freq_train).fillna(0).astype(int)
sample["qbucket"] = sample["qfreq"].apply(bucket)
sample["qcat2"] = cat2_text[sample["query_idx"].to_numpy()]
print(f"\ntest sample: {len(sample):,} pairs (seed {SEED})")
print("query-frequency bucket counts:")
print(sample["qbucket"].value_counts().reindex(BUCKET_ORDER[:-1]))
print("\nquery cat_2 counts:")
print(sample["qcat2"].value_counts())

MODELS = ["TTN", "TTN+IMAGE", "DESC", "POP", "CONTENT"]
hit10 = {m: np.zeros(len(sample), dtype=bool) for m in MODELS}
hit100 = {m: np.zeros(len(sample), dtype=bool) for m in MODELS}

# =============================================================================
# DESC, CONTENT, POP -- no model, in-node top-k membership
# =============================================================================
t1 = time.time()
n_groups = sample["target_node_id"].nunique()
for gi, (node_id, grp) in enumerate(sample.groupby("target_node_id")):
    node_id = int(node_id)
    cand = cands_for_node(node_id)
    if len(cand) < 2:
        continue
    rows = grp.index.to_numpy()
    q_idx = grp["query_idx"].to_numpy()
    t_idx = grp["target_idx"].to_numpy()
    cand_pos = {c: i for i, c in enumerate(cand)}
    t_pos = np.array([cand_pos.get(t, -1) for t in t_idx])

    combos = {"DESC": ["desc"], "CONTENT": CONTENT_PARTS}
    for combo_name, parts in combos.items():
        mats = []
        for part in parts:
            Q, C = NORM[part][q_idx], NORM[part][cand]
            M = Q @ C.T
            mask = np.outer(HAS[part][q_idx], HAS[part][cand])
            mats.append(np.where(mask, M, np.nan))
        with np.errstate(invalid="ignore"):
            combined = np.nanmean(np.stack(mats, axis=0), axis=0)
        combined = np.where(np.isnan(combined), -np.inf, combined)
        self_mask = cand[None, :] == q_idx[:, None]
        combined = np.where(self_mask, -np.inf, combined)

        k = min(100, combined.shape[1])
        top_idx = np.argpartition(-combined, kth=min(9, k - 1), axis=1)[:, :k]
        order10 = np.argsort(-np.take_along_axis(combined, top_idx, axis=1), axis=1)
        ranked = np.take_along_axis(top_idx, order10, axis=1)
        for r, tp in enumerate(t_pos):
            if tp < 0:
                continue
            in_top = ranked[r]
            if len(in_top) == 0 or combined[r, in_top[0]] == -np.inf:
                continue
            hit10[combo_name][rows[r]] = tp in set(in_top[:10])
            hit100[combo_name][rows[r]] = tp in set(in_top[:100])

    p10, p100 = top10_by_node.get(node_id, set()), top100_by_node.get(node_id, set())
    for r, t in enumerate(t_idx):
        hit10["POP"][rows[r]] = int(t) in p10
        hit100["POP"][rows[r]] = int(t) in p100

    if (gi + 1) % 100 == 0 or gi + 1 == n_groups:
        print(f"  content/pop node {gi + 1}/{n_groups}  [{time.time() - t1:.0f}s]", flush=True)
print(f"content+pop scoring done in {time.time() - t1:.0f}s")

# =============================================================================
# TTN and TTN+IMAGE -- load committed checkpoints read-only, never retrained
# =============================================================================
DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
title_t = torch.tensor(title_emb, device=DEVICE)
cat_t = torch.tensor(arrays_base["cat_ids"], device=DEVICE)
num_t_raw = torch.tensor(arrays_base["numeric"], device=DEVICE)


def load_ttn(checkpoint_path):
    ck = torch.load(checkpoint_path, weights_only=False)
    cfg = ck["config"]
    USE_DESCRIPTION = cfg["use_description"]
    USE_IMAGE = cfg.get("use_image", False)
    desc_t = torch.tensor(desc_emb, device=DEVICE) if USE_DESCRIPTION else torch.zeros(n_items, 1, device=DEVICE)
    img_t = torch.tensor(img_emb, device=DEVICE) if USE_IMAGE else torch.zeros(n_items, 1, device=DEVICE)
    mu, sd = ck["numeric_standardisation"]["mean"].to(DEVICE), ck["numeric_standardisation"]["std"].to(DEVICE)
    num_t = (num_t_raw - mu) / sd
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

    m = ComplementaryTwoTower().to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, OUT_DIM_


for tag, ckpt_name in (("TTN", "ttn_complementary.pt"), ("TTN+IMAGE", "ttn_complementary_image.pt")):
    model, OUT_DIM_ = load_ttn(OUT_DIR / ckpt_name)
    with torch.no_grad():
        cand_vecs = torch.empty(n_items, OUT_DIM_, device=DEVICE)
        for i in range(0, n_items, 8192):
            j = min(i + 8192, n_items)
            cand_vecs[i:j] = model.candidate(torch.arange(i, j, device=DEVICE))
        for node_id, grp in sample.groupby("target_node_id"):
            node_id = int(node_id)
            cand = cands_for_node(node_id)
            if len(cand) < 2:
                continue
            rows = grp.index.to_numpy()
            q_idx = torch.tensor(grp["query_idx"].to_numpy(), device=DEVICE)
            t_idx = grp["target_idx"].to_numpy()
            node_t = torch.full((len(grp),), node_id, device=DEVICE, dtype=torch.long)
            qv = model.query(q_idx, node_t)
            cv = cand_vecs[cand]
            scores = (qv @ cv.T).cpu().numpy()
            self_mask = cand[None, :] == grp["query_idx"].to_numpy()[:, None]
            scores = np.where(self_mask, -np.inf, scores)
            cand_pos = {c: i for i, c in enumerate(cand)}
            for r, t in enumerate(t_idx):
                tp = cand_pos.get(int(t), -1)
                if tp < 0:
                    continue
                k = min(100, scores.shape[1])
                top = np.argpartition(-scores[r], kth=min(9, k - 1))[:k]
                top = top[np.argsort(-scores[r, top])]
                hit10[tag][rows[r]] = tp in set(top[:10])
                hit100[tag][rows[r]] = tp in set(top[:100])
    print(f"{tag} scored on test sample")

# =============================================================================
# report
# =============================================================================
def table_by(hits, key, order_, title):
    print(f"\n{title}")
    header = f"{'group':<14}{'pairs':>8}" + "".join(f"{m:>12}" for m in MODELS)
    print(header)
    for b in order_:
        m_ = sample[key] == b if b != "all" else slice(None)
        n = int(m_.sum()) if b != "all" else len(sample)
        row = f"{str(b):<14}{n:>8}"
        for mod in MODELS:
            v = hits[mod][sample.index[m_]].mean() if b != "all" else hits[mod].mean()
            row += f"{v:>12.4f}"
        print(row)


cat2_order = sorted(sample["qcat2"].unique().tolist()) + ["all"]

table_by(hit10, "qbucket", BUCKET_ORDER, f"Recall@10 by QUERY frequency  (n={len(sample):,}, seed {SEED})")
table_by(hit100, "qbucket", BUCKET_ORDER, f"Recall@100 by QUERY frequency  (n={len(sample):,}, seed {SEED})")
table_by(hit10, "qcat2", cat2_order, f"Recall@10 by QUERY cat_2  (n={len(sample):,}, seed {SEED})")
table_by(hit100, "qcat2", cat2_order, f"Recall@100 by QUERY cat_2  (n={len(sample):,}, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
