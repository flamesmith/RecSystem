"""Add SigLIP2 image embeddings to the TTN model, TTN half.

Trains TTN+IMAGE on the FULL training set (2,223,995 pairs), same regime as
the committed data/tower/ttn_complementary.pt checkpoint (BATCH 4096, up to
30 epochs, early stopping PATIENCE 5, TAU 0.1, N_HARD 8, logQ correction) --
NOT the 20k-pair/50-epoch capacity-test regime used by
final_capacity_comparison_ttn.py -- so it stays directly comparable to the
existing committed TTN checkpoint: same training data, same frequency
buckets (computed from the full training set).

The image branch is patched into the model-code cell the same way
final_capacity_comparison_ttn.py does it, only without shrinking EPOCHS or
swapping in a 20k-pair sample.

Saves the new checkpoint to a SEPARATE file
(data/tower/ttn_complementary_image.pt), not over the committed
ttn_complementary.pt -- that file is relied on elsewhere in this notebook and
this is an additive comparison, not a replacement.

Originally one script with a similarity-scoring half (DESC / CONTENT / POP);
split in two so this repo's ttn_model branch carries only the model-training
half. The similarity half is image_augmented_similarity.py, on the
similarity_model branch.

Usage: python ttn/experiments/image_augmented_ttn.py
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
NEW_CHECKPOINT = OUT_DIR / "ttn_complementary_image.pt"
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]
N_SAMPLE = 20_000
SEED = 0
BUCKET_ORDER = ["never", "1-5", "6-25", "26-100", ">100", "all"]

t0 = time.time()
vocabs = json.load(open(OUT_DIR / "vocabs.json"))
arrays_base = dict(np.load(OUT_DIR / "items.npz"))
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
asins = np.load(OUT_DIR / "item_asins.npy", allow_pickle=False)
desc_emb = arrays_base["desc_emb"].astype("float32")
img_emb = np.load(OUT_DIR / "siglip_img_emb.npy").astype("float32")
n_items = len(asins)

pairs_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
print(f"loaded in {time.time() - t0:.0f}s | train {len(pairs_train):,} | "
      f"test {len(pairs_test):,}")

# --- in-node candidate pools -------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


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


# --- the SAME 20,000-pair test sample used throughout this notebook's
# similarity-model comparisons ------------------------------------------------
rng = np.random.default_rng(SEED)
take = rng.choice(len(pairs_test), min(N_SAMPLE, len(pairs_test)), replace=False)
sample = pairs_test.iloc[take].reset_index(drop=True)
sample["freq"] = sample["target_idx"].map(target_freq_train).fillna(0).astype(int)
sample["bucket"] = sample["freq"].apply(bucket)
print(f"\ntest sample: {len(sample):,} pairs (seed {SEED})")
print(sample["bucket"].value_counts().reindex(BUCKET_ORDER[:-1]))

hit10 = {}
hit100 = {}

# =============================================================================
# TTN+IMAGE -- trained HERE on the FULL training set, same regime as the
# committed checkpoint (BATCH 4096, up to 30 epochs, early stop PATIENCE 5),
# patched straight out of ttn_complementary.ipynb's `model-code` cell.
# =============================================================================
nb = json.load(open(ROOT / "ttn/ttn_complementary.ipynb"))
BASE_SRC = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])


def sub1(s, old, new, label):
    n = s.count(old)
    assert n == 1, f"anchor {label!r} matched {n} times, expected 1"
    return s.replace(old, new)


def build_source():
    src = BASE_SRC
    src = sub1(
        src,
        'print(f"description block: {\'on\' if USE_DESCRIPTION else \'OFF — no desc_emb in items.npz\'}")\n'
        'cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)',
        'print(f"description block: {\'on\' if USE_DESCRIPTION else \'OFF — no desc_emb in items.npz\'}")\n'
        'USE_IMAGE = "img_emb" in arrays\n'
        'img_t = (torch.tensor(arrays["img_emb"], device=DEVICE) if USE_IMAGE\n'
        '         else torch.zeros(len(arrays["title_emb"]), 1, device=DEVICE))\n'
        'print(f"image block: {\'on\' if USE_IMAGE else \'OFF — no img_emb in items.npz\'}")\n'
        'cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)',
        "feature tensors")
    src = sub1(
        src,
        "        self.description = nn.Linear(desc_t.shape[1], 128) if USE_DESCRIPTION else None\n"
        "        self.numeric = nn.Linear(num_t.shape[1], 16)",
        "        self.description = nn.Linear(desc_t.shape[1], 128) if USE_DESCRIPTION else None\n"
        "        self.image = nn.Linear(img_t.shape[1], 128) if USE_IMAGE else None\n"
        "        self.numeric = nn.Linear(num_t.shape[1], 16)",
        "__init__ linear")
    src = sub1(
        src,
        "        self.norm_description = (nn.LayerNorm(128, elementwise_affine=False)\n"
        "                                 if USE_DESCRIPTION else None)\n"
        "        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)",
        "        self.norm_description = (nn.LayerNorm(128, elementwise_affine=False)\n"
        "                                 if USE_DESCRIPTION else None)\n"
        "        self.norm_image = (nn.LayerNorm(128, elementwise_affine=False)\n"
        "                           if USE_IMAGE else None)\n"
        "        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)",
        "__init__ layernorm")
    src = sub1(
        src,
        "            nn.Linear(len(CAT_ORDER) * CAT_DIM + 128\n"
        "                      + (128 if USE_DESCRIPTION else 0) + 16, HIDDEN),",
        "            nn.Linear(len(CAT_ORDER) * CAT_DIM + 128\n"
        "                      + (128 if USE_DESCRIPTION else 0)\n"
        "                      + (128 if USE_IMAGE else 0) + 16, HIDDEN),",
        "mlp input dim")
    src = sub1(
        src,
        "        if self.description is not None:\n"
        "            parts.append(self.norm_description(self.description(desc_t[idx])))\n"
        "        parts.append(self.norm_numeric(self.numeric(num_t[idx])))",
        "        if self.description is not None:\n"
        "            parts.append(self.norm_description(self.description(desc_t[idx])))\n"
        "        if self.image is not None:\n"
        "            parts.append(self.norm_image(self.image(img_t[idx])))\n"
        "        parts.append(self.norm_numeric(self.numeric(num_t[idx])))",
        "forward")
    src = sub1(src, '"use_description": USE_DESCRIPTION,',
               '"use_description": USE_DESCRIPTION,\n'
               '               "use_image": USE_IMAGE,', "config")
    src = sub1(src, 'CHECKPOINT = OUT_DIR / "ttn_complementary.pt"',
               f'CHECKPOINT = Path("{NEW_CHECKPOINT}")', "checkpoint")
    src = sub1(src, "{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path")
    return src


arrays = dict(arrays_base)
arrays["img_emb"] = img_emb
src = build_source()
g = {k: v for k, v in globals().items() if not k.startswith("__")}
g.update({"pairs_train": pairs_train, "pairs_test": pairs_test, "arrays": arrays,
          "vocabs": vocabs, "node_of_item": node_of_item, "OUT_DIR": OUT_DIR,
          "ROOT": ROOT, "Path": Path, "CAT_ORDER": CAT_ORDER})
print(f"\n{'=' * 70}\n### TTN+IMAGE  (full training set, {len(pairs_train):,} pairs)\n{'=' * 70}",
      flush=True)
tt = time.time()
exec(compile("from pathlib import Path\n" + src, "<ttn_image>", "exec"), g)
print(f"[TTN+IMAGE] wall time {time.time() - tt:.0f}s")
model = g["model"]
DEVICE = g["DEVICE"]
OUT_DIM_ = g["OUT_DIM"]

# --- score TTN+IMAGE on the SAME test sample, in-node, top-k ---------------
hit10["TTN+IMAGE"] = np.zeros(len(sample), dtype=bool)
hit100["TTN+IMAGE"] = np.zeros(len(sample), dtype=bool)
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
            hit10["TTN+IMAGE"][rows[r]] = tp in set(top[:10])
            hit100["TTN+IMAGE"][rows[r]] = tp in set(top[:100])
print("TTN+IMAGE scored on test sample")

# --- TTN (original, committed checkpoint) for reference ---------------------
CHECKPOINT_ORIG = OUT_DIR / "ttn_complementary.pt"
ck = torch.load(CHECKPOINT_ORIG, weights_only=False)
cfg = ck["config"]
title_t2 = torch.tensor(arrays_base["title_emb"], device=DEVICE)
USE_DESCRIPTION2 = cfg["use_description"]
desc_t2 = torch.tensor(desc_emb, device=DEVICE) if USE_DESCRIPTION2 else torch.zeros(n_items, 1, device=DEVICE)
cat_t2 = torch.tensor(arrays_base["cat_ids"], device=DEVICE)
num_t2 = torch.tensor(arrays_base["numeric"], device=DEVICE)
mu2, sd2 = ck["numeric_standardisation"]["mean"].to(DEVICE), ck["numeric_standardisation"]["std"].to(DEVICE)
num_t2 = (num_t2 - mu2) / sd2
CAT_DIM2, NODE_DIM2, HIDDEN2, OUT_DIM2 = cfg["cat_dim"], cfg["node_dim"], cfg["hidden"], cfg["out_dim"]
vocab_sizes2 = ck["vocab_sizes"]


class ProductEncoderOrig(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_sizes2[name] + 1, CAT_DIM2, padding_idx=0)
            for name in CAT_ORDER])
        self.title = nn.Linear(title_t2.shape[1], 128)
        self.description = nn.Linear(desc_t2.shape[1], 128) if USE_DESCRIPTION2 else None
        self.numeric = nn.Linear(num_t2.shape[1], 16)
        self.norm_cat = nn.LayerNorm(len(CAT_ORDER) * CAT_DIM2, elementwise_affine=False)
        self.norm_title = nn.LayerNorm(128, elementwise_affine=False)
        self.norm_description = nn.LayerNorm(128, elementwise_affine=False) if USE_DESCRIPTION2 else None
        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(len(CAT_ORDER) * CAT_DIM2 + 128 + (128 if USE_DESCRIPTION2 else 0) + 16, HIDDEN2),
            nn.ReLU(), nn.Linear(HIDDEN2, OUT_DIM2))

    def forward(self, idx):
        cat = torch.cat([emb(cat_t2[idx, j]) for j, emb in enumerate(self.embeddings)], dim=-1)
        parts = [self.norm_cat(cat), self.norm_title(self.title(title_t2[idx]))]
        if self.description is not None:
            parts.append(self.norm_description(self.description(desc_t2[idx])))
        parts.append(self.norm_numeric(self.numeric(num_t2[idx])))
        return self.mlp(torch.cat(parts, dim=-1))


class ComplementaryTwoTowerOrig(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_encoder = ProductEncoderOrig()
        self.candidate_encoder = ProductEncoderOrig()
        self.node = nn.Embedding(vocab_sizes2["target_node"] + 1, NODE_DIM2, padding_idx=0)
        self.query_out = nn.Linear(OUT_DIM2 + NODE_DIM2, OUT_DIM2)

    def query(self, idx, node_id):
        h = torch.cat([self.query_encoder(idx), self.node(node_id)], dim=-1)
        return F.normalize(self.query_out(h), dim=-1)

    def candidate(self, idx):
        return F.normalize(self.candidate_encoder(idx), dim=-1)


model_orig = ComplementaryTwoTowerOrig().to(DEVICE)
model_orig.load_state_dict(ck["state_dict"])
model_orig.eval()
hit10["TTN"] = np.zeros(len(sample), dtype=bool)
hit100["TTN"] = np.zeros(len(sample), dtype=bool)
with torch.no_grad():
    cand_vecs2 = torch.empty(n_items, OUT_DIM2, device=DEVICE)
    for i in range(0, n_items, 8192):
        j = min(i + 8192, n_items)
        cand_vecs2[i:j] = model_orig.candidate(torch.arange(i, j, device=DEVICE))
    for node_id, grp in sample.groupby("target_node_id"):
        node_id = int(node_id)
        cand = cands_for_node(node_id)
        if len(cand) < 2:
            continue
        rows = grp.index.to_numpy()
        q_idx = torch.tensor(grp["query_idx"].to_numpy(), device=DEVICE)
        t_idx = grp["target_idx"].to_numpy()
        node_t = torch.full((len(grp),), node_id, device=DEVICE, dtype=torch.long)
        qv = model_orig.query(q_idx, node_t)
        cv = cand_vecs2[cand]
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
print("TTN (original checkpoint) scored on test sample")

# =============================================================================
# report
# =============================================================================
COLS = ["TTN", "TTN+IMAGE"]


def table(hits, title):
    print(f"\n{title}")
    header = f"{'target freq':<12}{'pairs':>8}" + "".join(f"{c:>16}" for c in COLS)
    print(header)
    for b in BUCKET_ORDER:
        m = sample["bucket"] == b if b != "all" else slice(None)
        n = int(m.sum()) if b != "all" else len(sample)
        row = f"{b:<12}{n:>8}"
        for c in COLS:
            v = hits[c][sample.index[m]].mean() if b != "all" else hits[c].mean()
            row += f"{v:>16.4f}"
        print(row)


table(hit10, f"Recall@10  (test sample {len(sample):,} pairs, seed {SEED})")
table(hit100, f"Recall@100  (test sample {len(sample):,} pairs, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
