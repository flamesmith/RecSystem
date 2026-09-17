"""§13's proposed-but-never-run capacity test, with an image-embedding arm added.

§13 ("Results log") lists as open work: "Can the model overfit a small subset?
Train on ~20k pairs for 50 epochs." That was never executed in the notebook --
the headline TTN numbers everywhere else (§13's before/after table, §18, §19)
come from the full 2.2M-pair, 30-epoch, early-stopped run. This script runs
that never-executed diagnostic for the first time, twice (image block off/on),
plus two content-similarity baselines and a popularity baseline, all under the
SAME reduced-data regime so every column in the table answers "what can be
learned from ~20k co-purchase pairs" rather than mixing data budgets.

Five columns:
  TTN               -- ComplementaryTwoTower, image block OFF, trained here
  TTN+IMAGE         -- ComplementaryTwoTower, image block ON,  trained here
  DESC+TITLE+FEAT   -- mean cosine over whichever of {description, title,
                       features} is present on both sides (SBERT / frozen
                       title_emb), §19's CONTENT recipe with color+material
                       swapped for title+features, per this session's brief
  DESC+TITLE+FEAT+IMG -- the same, with SigLIP image cosine added as a
                       fourth (optional) component
  POP               -- fixed per-node top-10/top-100 by training frequency,
                       no model, no query read (§13/§19's popularity baseline)

Everything -- TTN training data, the "target's training frequency" bucket, and
the popularity counts -- uses the SAME ~20,000-pair TRAINING sample, seed 0.
Evaluation is a separate 20,000-pair TEST sample, seed 0, scored inside the
pair's target category (candidates restricted to node_of_item ==
target_node_id, self excluded), hit@k via `target in topk(scores)` membership
-- never `(scores > true).sum()`, which is the rule that inflated the
description-similarity number 3x in an earlier pass (see §19's writeup).

Usage: python ttn/experiments/final_capacity_comparison.py
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

ROOT = Path("/Users/lazr/PycharmProjects/RecSystem")
OUT_DIR = ROOT / "data" / "tower"
SCRATCH = Path("/private/tmp/claude-501/-Users-lazr-PycharmProjects-RecSystem/"
               "9f5de6c8-4072-4157-bc73-973c9dc7569b/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

N_TRAIN_SAMPLE = 20_000
N_TEST_SAMPLE = 20_000
EPOCHS = 50
SEED = 0
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]
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
assert img_emb.shape[0] == n_items and desc_emb.shape[0] == n_items

full_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
print(f"loaded in {time.time() - t0:.0f}s | full train {len(full_train):,} | "
      f"test {len(pairs_test):,}")

# --- THE ~20k-pair training sample, used by every column below -------------
train_sample = full_train.sample(n=min(N_TRAIN_SAMPLE, len(full_train)),
                                  random_state=SEED).reset_index(drop=True)
print(f"training sample: {len(train_sample):,} pairs (seed {SEED})")

has_desc = ~np.all(desc_emb == 0, axis=1)
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
has_title = np.any(title_emb != 0, axis=1)
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
COMBOS = {
    "DESC+TITLE+FEAT": ["desc", "title", "features"],
    "DESC+TITLE+FEAT+IMG": ["desc", "title", "features", "image"],
}

# --- target's training frequency, from the 20k-pair TRAINING sample --------
target_freq = train_sample.groupby("target_idx").size()


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


# --- in-node candidate pools -------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


# --- POP, from the SAME 20k-pair training sample ----------------------------
counts = (train_sample.groupby(["target_node_id", "target_idx"]).size()
          .rename("n").reset_index())
top10_by_node = {int(k): set(g.nlargest(10, "n")["target_idx"].astype(int))
                  for k, g in counts.groupby("target_node_id")}
top100_by_node = {int(k): set(g.nlargest(100, "n")["target_idx"].astype(int))
                   for k, g in counts.groupby("target_node_id")}

# --- the test sample, bucketed by the 20k-pair training sample's frequency --
rng = np.random.default_rng(SEED)
take = rng.choice(len(pairs_test), min(N_TEST_SAMPLE, len(pairs_test)), replace=False)
sample = pairs_test.iloc[take].reset_index(drop=True)
sample["freq"] = sample["target_idx"].map(target_freq).fillna(0).astype(int)
sample["bucket"] = sample["freq"].apply(bucket)
print(f"\ntest sample: {len(sample):,} pairs (seed {SEED})")
print(sample["bucket"].value_counts().reindex(BUCKET_ORDER[:-1]))

CONTENT_NAMES = list(COMBOS) + ["POP"]
hit10 = {name: np.zeros(len(sample), dtype=bool) for name in CONTENT_NAMES}
hit100 = {name: np.zeros(len(sample), dtype=bool) for name in CONTENT_NAMES}

# --- score DESC+TITLE+FEAT[, +IMG] and POP, in-node, top-k membership ------
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

    for combo_name, parts in COMBOS.items():
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
# TTN, image block off/on -- both trained HERE on the same 20k-pair sample,
# 50 epochs, patched straight out of ttn_complementary.ipynb's `model-code`
# cell (same mechanism as ttn/experiments/temperature_test.py).
# =============================================================================
nb = json.load(open(ROOT / "ttn/ttn_complementary.ipynb"))
BASE_SRC = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])


def sub1(s, old, new, label):
    n = s.count(old)
    assert n == 1, f"anchor {label!r} matched {n} times, expected 1"
    return s.replace(old, new)


def build_source(use_image, tag):
    src = sub1(BASE_SRC, "BATCH, EPOCHS, LR = 4096, 30, 3e-3",
               f"BATCH, EPOCHS, LR = 4096, {EPOCHS}, 3e-3", "epochs")
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
               f'CHECKPOINT = Path("{SCRATCH}") / "ttn_capacity_{tag}.pt"', "checkpoint")
    src = sub1(src, "{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path")
    return src


def train_ttn(use_image):
    tag = "with_img" if use_image else "baseline"
    arrays = dict(arrays_base)
    if use_image:
        arrays["img_emb"] = img_emb
    src = build_source(use_image, tag)
    g = {k: v for k, v in globals().items() if not k.startswith("__")}
    g.update({"pairs_train": train_sample, "pairs_test": pairs_test, "arrays": arrays,
              "vocabs": vocabs, "node_of_item": node_of_item, "OUT_DIR": OUT_DIR,
              "ROOT": ROOT, "Path": Path, "CAT_ORDER": CAT_ORDER})
    print(f"\n{'=' * 70}\n### TTN {tag}  (image block {'ON' if use_image else 'OFF'}, "
          f"{len(train_sample):,} pairs, {EPOCHS} epochs)\n{'=' * 70}", flush=True)
    tt = time.time()
    exec(compile("from pathlib import Path\n" + src, f"<{tag}>", "exec"), g)
    print(f"[{tag}] wall time {time.time() - tt:.0f}s")
    return g["model"], g["ComplementaryTwoTower"], g["cand_t"] if "cand_t" in g else None, g


ttn_models = {}
for use_image in (False, True):
    tag = "TTN+IMAGE" if use_image else "TTN"
    model, _, _, g = train_ttn(use_image)
    ttn_models[tag] = (model, g)

# --- score both trained TTNs on the SAME test sample, in-node, top-k -------
for tag, (model, g) in ttn_models.items():
    DEVICE = g["DEVICE"]
    OUT_DIM_ = g["OUT_DIM"]
    hit10[tag] = np.zeros(len(sample), dtype=bool)
    hit100[tag] = np.zeros(len(sample), dtype=bool)
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
                hit10[tag][rows[r]] = tp in set(top[:10])
                hit100[tag][rows[r]] = tp in set(top[:100])
    print(f"{tag} scored on test sample")

# =============================================================================
# report
# =============================================================================
COLS = ["TTN", "TTN+IMAGE", "DESC+TITLE+FEAT", "DESC+TITLE+FEAT+IMG", "POP"]


def table(hits, title):
    print(f"\n{title}")
    header = f"{'target freq':<12}{'pairs':>8}" + "".join(f"{c:>22}" for c in COLS)
    print(header)
    for b in BUCKET_ORDER:
        m = sample["bucket"] == b if b != "all" else slice(None)
        n = int(m.sum()) if b != "all" else len(sample)
        row = f"{b:<12}{n:>8}"
        for c in COLS:
            v = hits[c][sample.index[m]].mean() if b != "all" else hits[c].mean()
            row += f"{v:>22.4f}"
        print(row)


table(hit10, f"Recall@10  (train sample {len(train_sample):,} pairs / {EPOCHS} epochs, "
             f"test sample {len(sample):,} pairs, seed {SEED})")
table(hit100, f"Recall@100  (train sample {len(train_sample):,} pairs / {EPOCHS} epochs, "
              f"test sample {len(sample):,} pairs, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
