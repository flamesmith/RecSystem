"""§13's proposed-but-never-run capacity test, similarity half.

§13 ("Results log") lists as open work: "Can the model overfit a small subset?
Train on ~20k pairs for 50 epochs." That never-executed diagnostic prompted
this comparison: two content-similarity baselines (no training at all) and a
popularity baseline, under the SAME reduced-data regime as the TTN half, so
every column answers "what can be learned from ~20k co-purchase pairs" rather
than mixing data budgets.

Two columns:
  DESC+TITLE+FEAT      -- mean cosine over whichever of {description, title,
                          features} is present on both sides (SBERT / frozen
                          title_emb), §19's CONTENT recipe with title+features
                          swapped for colour+material, per this session's brief
  DESC+TITLE+FEAT+IMG  -- the same, with SigLIP image cosine added as a
                          fourth (optional) component
  POP                  -- fixed per-node top-10/top-100 by training frequency,
                          no model, no query read (§13/§19's popularity baseline)

Originally one script with a TTN-training half; split in two so this repo's
similarity_model branch carries only the no-training halves of the
experiments that touch both. The TTN half is
final_capacity_comparison_ttn.py, on the ttn_model branch.

Everything -- the "target's training frequency" bucket and the popularity
counts -- uses the SAME ~20,000-pair TRAINING sample, seed 0. Evaluation is a
separate 20,000-pair TEST sample, seed 0, scored inside the pair's target
category (candidates restricted to node_of_item == target_node_id, self
excluded), hit@k via `target in topk(scores)` membership -- never
`(scores > true).sum()`, which is the rule that inflated the
description-similarity number 3x in an earlier pass (see §19's writeup).

Usage: python ttn/experiments/final_capacity_comparison_similarity.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice")

ROOT = Path("/Users/lazr/PycharmProjects/RecSystem")
OUT_DIR = ROOT / "data" / "tower"

N_TRAIN_SAMPLE = 20_000
N_TEST_SAMPLE = 20_000
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

full_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
print(f"loaded in {time.time() - t0:.0f}s | full train {len(full_train):,} | "
      f"test {len(pairs_test):,}")

# --- THE ~20k-pair training sample, used below -----------------------------
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
# report
# =============================================================================
COLS = ["DESC+TITLE+FEAT", "DESC+TITLE+FEAT+IMG", "POP"]


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


table(hit10, f"Recall@10  (train sample {len(train_sample):,} pairs, "
             f"test sample {len(sample):,} pairs, seed {SEED})")
table(hit100, f"Recall@100  (train sample {len(train_sample):,} pairs, "
              f"test sample {len(sample):,} pairs, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
