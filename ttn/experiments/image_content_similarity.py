"""Does adding SigLIP image similarity to §19's CONTENT signal help?

§19 (`ttn/ttn_complementary.ipynb`) defines CONTENT = mean cosine over whichever
of {description, colour, material} is present on both query and candidate
(SBERT `all-MiniLM-L6-v2`, equal weights, renormalised when a term is dropped).
This adds a fourth component -- SigLIP2 image similarity (`data/tower/
siglip_img_emb.npy`) -- and reports IMAGE alone and CONTENT+IMAGE next to it,
plus TTN and POP for reference, using the exact §19 methodology:

  - N_SAMPLE test pairs, seed 0 (a *sample* of the 201,197 test pairs, same
    discipline as §18/19's 20,000 -- not the full set).
  - every ranker scored inside the pair's target category (candidates
    restricted to node_of_item == target_node_id, self excluded).
  - hit@k via `target in topk(scores)` membership, never `(scores > true).sum()`
    -- that strictly-greater rule is what inflated the original description
    number 3x when zero-vector ties got a free hit (see the notebook's §19
    writeup and ttn/experiments/ for the postmortem). A query (or a whole
    node) with zero usable signal on every component is scored a miss, not
    given a free pass.
  - recall reported by the target's training frequency bucket, matching §18/19.

Colour and material text are read back from `cat_ids` + `vocabs.json` (the
model's own folded vocabulary, "Missing" already standing in for absent) rather
than re-joining df_features -- cheaper, and exactly what the model itself sees.

Usage: python ttn/experiments/image_content_similarity.py [N_SAMPLE]
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Mean of empty slice")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "tower"
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]
N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
SEED = 0

t0 = time.time()
vocabs = json.load(open(OUT_DIR / "vocabs.json"))
arrays = dict(np.load(OUT_DIR / "items.npz"))
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
asins = np.load(OUT_DIR / "item_asins.npy", allow_pickle=False)
desc_emb = arrays["desc_emb"].astype("float32")
img_emb = np.load(OUT_DIR / "siglip_img_emb.npy").astype("float32")
img_status = np.load(OUT_DIR / "siglip_img_status.npy")
pairs_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")

n_items = len(asins)
assert desc_emb.shape[0] == n_items and img_emb.shape[0] == n_items
print(f"items: {n_items:,} | test pairs: {len(pairs_test):,} | "
      f"loaded in {time.time() - t0:.0f}s")

has_desc = ~np.all(desc_emb == 0, axis=1)
has_img = img_status == 1
print(f"description present: {has_desc.mean():.1%}   image present: {has_img.mean():.1%}")


# --- colour / material text, read back from the model's own vocabulary -----
def text_for(field):
    inv = {i: v for v, i in vocabs[field].items()}
    ids = arrays["cat_ids"][:, CAT_ORDER.index(field)]
    return np.array([inv.get(int(i), "Missing") for i in ids], dtype=object)


color_text = text_for("color")
material_text = text_for("material")


def encode_categorical(text_arr, sbert):
    """SBERT over unique values only (colour: 64, material: 209) -- cheap."""
    uniq = sorted(set(text_arr))
    vecs = np.asarray(sbert.encode(uniq, batch_size=64, show_progress_bar=False))
    lut = dict(zip(uniq, vecs))
    out = np.stack([lut[t] for t in text_arr]).astype("float32")
    absent = np.array([t.strip() == "" or t == "Missing" for t in text_arr])
    out[absent] = 0.0
    return out, ~absent


from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer("all-MiniLM-L6-v2")
color_emb, has_color = encode_categorical(color_text, sbert)
material_emb, has_material = encode_categorical(material_text, sbert)
print(f"colour present: {has_color.mean():.1%}   material present: {has_material.mean():.1%}")
print(f"encoded colour/material in {time.time() - t0:.0f}s total")

SIGNALS = {
    "desc": (desc_emb, has_desc),
    "color": (color_emb, has_color),
    "material": (material_emb, has_material),
    "image": (img_emb, has_img),
}
COMBOS = {
    "DESC": ["desc"],
    "IMAGE": ["image"],
    "CONTENT": ["desc", "color", "material"],
    "CONTENT+IMAGE": ["desc", "color", "material", "image"],
}

# --- unit-normalise once; cosine becomes a plain dot product ---------------
NORM = {k: v[0] / (np.linalg.norm(v[0], axis=1, keepdims=True) + 1e-9)
        for k, v in SIGNALS.items()}
HAS = {k: v[1] for k, v in SIGNALS.items()}

# --- in-node candidate pools ------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


# --- popularity baseline, exactly as §19/model-code's `baselines()` --------
counts = (pairs_train.groupby(["target_node_id", "target_idx"]).size()
          .rename("n").reset_index())
top10_by_node = {int(k): set(g.nlargest(10, "n")["target_idx"].astype(int))
                  for k, g in counts.groupby("target_node_id")}
top100_by_node = {int(k): set(g.nlargest(100, "n")["target_idx"].astype(int))
                   for k, g in counts.groupby("target_node_id")}

# --- target's training frequency, for the §18/19 bucket table -------------
target_freq_train = pairs_train.groupby("target_idx").size()


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


BUCKET_ORDER = ["never", "1-5", "6-25", "26-100", ">100", "all"]

# --- sample ------------------------------------------------------------------
rng = np.random.default_rng(SEED)
take = rng.choice(len(pairs_test), min(N_SAMPLE, len(pairs_test)), replace=False)
sample = pairs_test.iloc[take].reset_index(drop=True)
sample["freq"] = sample["target_idx"].map(target_freq_train).fillna(0).astype(int)
sample["bucket"] = sample["freq"].apply(bucket)
print(f"\nsampled {len(sample):,} test pairs (seed {SEED})")
print(sample["bucket"].value_counts().reindex(BUCKET_ORDER[:-1]))

# --- score every combo, in-node, top-k membership --------------------------
hit10 = {name: np.zeros(len(sample), dtype=bool) for name in list(COMBOS) + ["TTN", "POP"]}
hit100 = {name: np.zeros(len(sample), dtype=bool) for name in list(COMBOS) + ["TTN", "POP"]}

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

    # target_idx must be a candidate in its own node -- true by construction
    cand_pos = {c: i for i, c in enumerate(cand)}
    t_pos = np.array([cand_pos.get(t, -1) for t in t_idx])

    for combo_name, parts in COMBOS.items():
        mats = []
        for part in parts:
            Q = NORM[part][q_idx]           # (nq, d)
            C = NORM[part][cand]            # (nc, d)
            M = Q @ C.T                     # (nq, nc) cosine, both unit-norm
            mask = np.outer(HAS[part][q_idx], HAS[part][cand])
            M = np.where(mask, M, np.nan)
            mats.append(M)
        with np.errstate(invalid="ignore"):
            combined = np.nanmean(np.stack(mats, axis=0), axis=0)
        unrankable = np.isnan(combined)
        combined = np.where(unrankable, -np.inf, combined)
        # exclude the query itself, if it happens to sit in this node
        self_mask = cand[None, :] == q_idx[:, None]
        combined = np.where(self_mask, -np.inf, combined)

        k = min(100, combined.shape[1])
        top_idx = np.argpartition(-combined, kth=min(9, k - 1), axis=1)
        order10 = np.argsort(-np.take_along_axis(combined, top_idx[:, :k], axis=1), axis=1)
        ranked = np.take_along_axis(top_idx[:, :k], order10, axis=1)
        for r, tp in enumerate(t_pos):
            if tp < 0:
                continue
            in_top = ranked[r]
            all_neg_inf = combined[r, in_top[0]] == -np.inf if len(in_top) else True
            if all_neg_inf:
                continue        # query has zero usable signal in this node -> miss
            hit10[combo_name][rows[r]] = tp in set(in_top[:10])
            hit100[combo_name][rows[r]] = tp in set(in_top[:100])

    # --- POP: fixed per-node top10/top100, no query read -----------------
    p10, p100 = top10_by_node.get(node_id, set()), top100_by_node.get(node_id, set())
    for r, t in enumerate(t_idx):
        hit10["POP"][rows[r]] = int(t) in p10
        hit100["POP"][rows[r]] = int(t) in p100

    if (gi + 1) % 50 == 0 or gi + 1 == n_groups:
        print(f"  node {gi + 1}/{n_groups}  [{time.time() - t1:.0f}s]", flush=True)

print(f"\ncontent-similarity scoring: {time.time() - t1:.0f}s")

# --- TTN: load the trained checkpoint, score the same in-node candidates --
CHECKPOINT = OUT_DIR / "ttn_complementary.pt"
if CHECKPOINT.exists():
    ck = torch.load(CHECKPOINT, weights_only=False)
    cfg = ck["config"]
    DEVICE = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")

    title_t = torch.tensor(arrays["title_emb"], device=DEVICE)
    USE_DESCRIPTION = cfg["use_description"]
    desc_t = torch.tensor(desc_emb, device=DEVICE) if USE_DESCRIPTION else torch.zeros(n_items, 1, device=DEVICE)
    cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)
    num_t = torch.tensor(arrays["numeric"], device=DEVICE)
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
            self.numeric = nn.Linear(num_t.shape[1], 16)
            self.norm_cat = nn.LayerNorm(len(CAT_ORDER) * CAT_DIM, elementwise_affine=False)
            self.norm_title = nn.LayerNorm(128, elementwise_affine=False)
            self.norm_description = nn.LayerNorm(128, elementwise_affine=False) if USE_DESCRIPTION else None
            self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)
            self.mlp = nn.Sequential(
                nn.Linear(len(CAT_ORDER) * CAT_DIM + 128 + (128 if USE_DESCRIPTION else 0) + 16, HIDDEN),
                nn.ReLU(), nn.Linear(HIDDEN, OUT_DIM_))

        def forward(self, idx):
            cat = torch.cat([emb(cat_t[idx, j]) for j, emb in enumerate(self.embeddings)], dim=-1)
            parts = [self.norm_cat(cat), self.norm_title(self.title(title_t[idx]))]
            if self.description is not None:
                parts.append(self.norm_description(self.description(desc_t[idx])))
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
                hit10["TTN"][rows[r]] = tp in set(top[:10])
                hit100["TTN"][rows[r]] = tp in set(top[:100])
    print(f"TTN checkpoint scored (best_epoch {cfg.get('best_epoch')}, "
          f"use_description={USE_DESCRIPTION})")
else:
    print("no checkpoint at data/tower/ttn_complementary.pt -- skipping TTN column")
    del hit10["TTN"], hit100["TTN"]

# --- report ------------------------------------------------------------------
def table(hits, title):
    print(f"\n{title}")
    cols = list(hits.keys())
    header = f"{'target freq':<12}{'pairs':>8}" + "".join(f"{c:>16}" for c in cols)
    print(header)
    for b in BUCKET_ORDER:
        m = sample["bucket"] == b if b != "all" else slice(None)
        n = int(m.sum()) if b != "all" else len(sample)
        row = f"{b:<12}{n:>8}"
        for c in cols:
            v = hits[c][sample.index[m]].mean() if b != "all" else hits[c].mean()
            row += f"{v:>16.4f}"
        print(row)


table(hit10, f"Recall@10  (n={len(sample):,} sampled test pairs, seed {SEED})")
table(hit100, f"Recall@100  (n={len(sample):,} sampled test pairs, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
