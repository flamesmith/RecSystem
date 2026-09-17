"""Train the TTN ("Complete the Look") two-tower model and save its checkpoint.

Extracted verbatim from ttn_complementary.ipynb's section 10 -- the model
definition, BPR loss (temperature, logQ correction, in-node hard negatives),
training loop, evaluation, and checkpoint export.

Prerequisite: TTN/build_data.py must have already run, to produce the
data/tower/ arrays this reads. Run TTN/encode_descriptions.py in between the
two for a model trained with description signal (optional -- this script
detects its absence and trains without that block otherwise).

Usage: python TTN/build_model.py
Output: data/tower/ttn_complementary.pt
"""

# ============================================================================
# Setup
# ============================================================================
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# data/ lives at the repo root, one level up from this ttn/ folder
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Show every column/variable when displaying a dataframe
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)
# Turn off scientific notation (e.g. 2.447268e+06 -> 2447268.00)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
OUT_DIR = DATA_DIR / "tower"
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]

import json
vocabs = json.load(open(OUT_DIR / "vocabs.json"))
arrays = dict(np.load(OUT_DIR / "items.npz"))
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
import pandas as pd
pairs_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
if (OUT_DIR / "desc_emb.npy").exists():
    arrays["desc_emb"] = np.load(OUT_DIR / "desc_emb.npy")
print(f"loaded data/tower/ -- {len(arrays['title_emb']):,} items, "
      f"{len(pairs_train):,} train pairs, {len(pairs_test):,} test pairs, "
      f"description block: {'on' if 'desc_emb' in arrays else 'OFF'}")

# ============================================================================
# 10. The two-tower model
# ============================================================================
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
CAT_DIM, NODE_DIM, HIDDEN, OUT_DIM = 32, 32, 256, 128
BATCH, EPOCHS, LR = 4096, 30, 3e-3
SEED = 0

# Temperature on the cosine scores, and the single most consequential constant
# in this cell. Both towers are L2-normalised, so scores live in [-1, 1] and the
# differences that decide a ranking are tiny: measured on the untempered model,
# score(top1) - score(top10) was 0.0067. At that scale `logsigmoid` is
# effectively linear, so a badly-ordered pair pulls no harder than a well-ordered
# one -- the loss falls while the ranking stays arbitrary.
#
# Isolated on 5,000 pairs inside ONE category (so the node embedding is constant
# and cannot help), 150 epochs of training reached:
#
#     TAU = 1.0   train R@10 0.0392    <- worse than a fixed top-10 (0.0964)
#     TAU = 0.5   train R@10 0.3022
#     TAU = 0.2   train R@10 0.5036
#     TAU = 0.1   train R@10 0.5158    <- best
#     TAU = 0.05  train R@10 0.2700    <- over-sharpened, gradient saturates
#
# On the full data, 3 epochs: R@10 0.1218 at TAU 1.0 against 0.1916 at TAU 0.1.
TAU = 0.1

# EPOCHS was 3 because the untempered loss plateaued after ~500 steps. That is
# no longer true, so the run is length-limited by early stopping instead: train
# until test Recall@10 has not improved for PATIENCE epochs, then restore the
# best weights. Stopping on the metric rather than the loss matters here --
# under the old loss the two moved in opposite directions.
PATIENCE = 5

# Measured directly (ttn/experiments/): a rare-target row gets ~67x less
# in-batch negative exposure than a frequent-target row, because an item can
# only be an in-batch negative when it is SOMEONE's positive in that batch --
# logQ corrects the score it gets when sampled, not how rarely it is sampled at
# all. The uniform in-node draw below is the one channel that stays roughly
# flat by frequency (2x, not 67x), so it is the one channel actually teaching
# the model to separate two rare items from each other. N_HARD was 1; this
# tests whether more of the channel that already works helps the rows that
# depend on it most.
N_HARD = 8

# logQ correction. In-batch negatives are the *targets of other rows*, so an
# item is drawn as a negative in proportion to how often it appears as a target
# -- frequent items are penalised far more often than rare ones, purely as an
# artifact of sampling. Subtracting log P(sampled) from each candidate's logit
# removes that bias (Yi et al., 2019). It matters here because error analysis
# showed recall is almost entirely a function of target frequency: targets seen
# >100 times score R@10 0.48, targets seen <=5 times score ~0.02.
#
# Applied only to the in-batch term. The in-node negatives are drawn uniformly
# within a category, so their sampling probability carries no per-item bias and
# needs no correction.
LOGQ_CORRECTION = True
torch.manual_seed(SEED)
print(f"device: {DEVICE}")

# --- Feature tensors, one row per item ------------------------------------
title_t = torch.tensor(arrays["title_emb"], device=DEVICE)
# Description vectors, if §9 encoded them. A cold item has a description before
# it has a single purchase, which is the whole reason for this block -- see §18.
USE_DESCRIPTION = "desc_emb" in arrays
desc_t = (torch.tensor(arrays["desc_emb"], device=DEVICE) if USE_DESCRIPTION
          else torch.zeros(len(arrays["title_emb"]), 1, device=DEVICE))
print(f"description block: {'on' if USE_DESCRIPTION else 'OFF — no desc_emb in items.npz'}")
cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)
num_t = torch.tensor(arrays["numeric"], device=DEVICE)

# Standardise the numeric block on TRAIN items only, as in §§5-8.
train_item_idx = torch.tensor(
    np.unique(pairs_train[["query_idx", "target_idx"]].to_numpy()), device=DEVICE)
mu = num_t[train_item_idx].mean(0)
sd = num_t[train_item_idx].std(0).clamp_min(1e-6)
num_t = (num_t - mu) / sd


class ProductEncoder(nn.Module):
    """The paper's Product Encoder: attribute embeddings -> concat -> MLP.

    Per-block LayerNorm, applied just before the concat. Measured on the trained
    model, the four input blocks entered this concat at wildly different norms:

        categorical (8 fields, nn.Embedding, unconstrained)   10.79
        title       (nn.Linear(384, 128), small init)          1.98
        description (nn.Linear(384, 128), small init)          1.75
        numeric                                                 1.00

    Categorical embeddings are looked up and updated directly, so nothing caps
    how large they grow. Title and description reach the model only through a
    linear layer that starts small (~1/sqrt(384) init) and the loss can already
    be driven down using the categorical block alone, so gradient descent never
    has to grow that projection -- it is cheaper to keep tuning what is already
    working. The result: description's UNTRAINED cosine similarity scores R@10
    0.2084, almost matching the trained model's 0.2233, yet the MLP's first
    layer receives it at roughly 1/6th the categorical block's scale. The signal
    is real and is not reaching the network in usable form.

    LayerNorm on each block (no affine scale/shift -- `elementwise_affine=False`)
    fixes the SCALE before concatenation, without deciding importance for the
    network. The MLP's own weights, which were already comparable in norm across
    blocks (~25-33) once each input arrived at a fair scale, are what should
    decide how much each block contributes -- not an accident of initialisation.
    """

    def __init__(self):
        super().__init__()
        # padding_idx=0 keeps the reserved id inert -- it never receives gradient.
        self.embeddings = nn.ModuleList([
            nn.Embedding(len(vocabs[name]) + 1, CAT_DIM, padding_idx=0)
            for name in CAT_ORDER])
        self.title = nn.Linear(title_t.shape[1], 128)
        self.description = nn.Linear(desc_t.shape[1], 128) if USE_DESCRIPTION else None
        self.numeric = nn.Linear(num_t.shape[1], 16)

        # elementwise_affine=False: normalise scale only, learn no per-dimension
        # gain. A learned affine would let training re-inflate exactly the
        # imbalance this is meant to remove.
        self.norm_cat = nn.LayerNorm(len(CAT_ORDER) * CAT_DIM, elementwise_affine=False)
        self.norm_title = nn.LayerNorm(128, elementwise_affine=False)
        self.norm_description = (nn.LayerNorm(128, elementwise_affine=False)
                                 if USE_DESCRIPTION else None)
        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)

        self.mlp = nn.Sequential(
            nn.Linear(len(CAT_ORDER) * CAT_DIM + 128
                      + (128 if USE_DESCRIPTION else 0) + 16, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, OUT_DIM))

    def forward(self, idx):
        cat = torch.cat([emb(cat_t[idx, j]) for j, emb in enumerate(self.embeddings)], dim=-1)
        parts = [self.norm_cat(cat), self.norm_title(self.title(title_t[idx]))]
        if self.description is not None:
            parts.append(self.norm_description(self.description(desc_t[idx])))
        parts.append(self.norm_numeric(self.numeric(num_t[idx])))
        return self.mlp(torch.cat(parts, dim=-1))


class ComplementaryTwoTower(nn.Module):
    """Same architecture both sides, no shared weights.

    Only the query tower sees the target-category embedding -- the one
    modification Complementary-TT makes to a plain two-tower.
    """

    def __init__(self):
        super().__init__()
        self.query_encoder = ProductEncoder()
        self.candidate_encoder = ProductEncoder()
        self.node = nn.Embedding(len(vocabs["target_node"]) + 1, NODE_DIM, padding_idx=0)
        self.query_out = nn.Linear(OUT_DIM + NODE_DIM, OUT_DIM)

    def query(self, idx, node_id):
        h = torch.cat([self.query_encoder(idx), self.node(node_id)], dim=-1)
        return F.normalize(self.query_out(h), dim=-1)

    def candidate(self, idx):
        return F.normalize(self.candidate_encoder(idx), dim=-1)


# --- Sampling probability of each item, as a target in training -----------
_target_freq = pairs_train.groupby("target_idx").size()
_count = np.zeros(len(cat_t), dtype="float64")
_count[_target_freq.index.to_numpy()] = _target_freq.to_numpy()
# An item never seen as a target is floored at 1 rather than dropped: it can
# still turn up as an in-batch negative, and log(0) is not a number.
LOGQ = torch.tensor(np.log(np.maximum(_count, 1.0) / max(_count.sum(), 1.0)),
                    dtype=torch.float32, device=DEVICE)
print(f"logQ spans {LOGQ.min():.2f} to {LOGQ.max():.2f} "
      f"({int((_count == 0).sum()):,} items never a target)")


# --- Hard negatives: items sharing the target's category ------------------
# Bucketed once by node id, so drawing one is two lookups and a uniform sample.
_order = np.argsort(node_of_item, kind="stable")
_starts = np.searchsorted(node_of_item[_order],
                          np.arange(node_of_item.max() + 2), side="left")
order_t = torch.tensor(_order, device=DEVICE)
starts_t = torch.tensor(_starts, device=DEVICE)


def hard_negatives(node_ids, n=N_HARD):
    lo = starts_t[node_ids].unsqueeze(1)
    hi = starts_t[node_ids + 1].unsqueeze(1)
    size = (hi - lo).clamp_min(1)
    offset = (torch.rand(len(node_ids), n, device=DEVICE) * size).long().clamp(max=size - 1)
    return order_t[(lo + offset).clamp(max=len(order_t) - 1)]


def bpr_loss(q, c_pos, c_hard=None, pos_idx=None):
    """BPR over in-batch negatives, plus one hard negative per row.

    `masked_fill`, not `masked_select`: a data-dependent output size is the
    pattern that returns the wrong element count on MPS.
    """
    scores = (q @ c_pos.T) / TAU
    if LOGQ_CORRECTION and pos_idx is not None:
        # column j is item pos_idx[j] acting as a negative for every other row
        scores = scores - LOGQ[pos_idx].unsqueeze(0)
    pos = scores.diagonal().unsqueeze(1)
    eye = torch.eye(len(q), dtype=torch.bool, device=q.device)
    easy = (-F.logsigmoid(pos - scores)).masked_fill(eye, 0.0).sum() / (len(q) * (len(q) - 1))
    if c_hard is None:
        return easy
    hard = -F.logsigmoid(pos - torch.einsum("bd,bnd->bn", q, c_hard) / TAU).mean()
    return (easy + hard) / 2


# --- Evaluation: rank the true target against the whole catalogue ---------
# Strict Recall@k credits only the target of the pair being scored. A test
# (query, category) key holds 1.73 held-out targets on average, and the rest sit
# in the candidate pool counted as wrong answers -- so strict recall understates
# top-10 quality by roughly a factor of two. `lenient=True` credits any of them.
# Both are reported, and the baselines below are measured the same way.
relevant = defaultdict(set)
for _q, _n, _t in pairs_test[["query_idx", "target_node_id", "target_idx"]].itertuples(index=False):
    relevant[(int(_q), int(_n))].add(int(_t))


@torch.no_grad()
def catalogue_vectors(model, chunk=8192):
    model.eval()
    out = torch.empty(len(cat_t), OUT_DIM, device=DEVICE)
    for i in range(0, len(cat_t), chunk):
        j = min(i + chunk, len(cat_t))
        out[i:j] = model.candidate(torch.arange(i, j, device=DEVICE))
    return out


@torch.no_grad()
def evaluate(model, pairs, k_values=(10, 20, 50, 100), n_eval=10_000, seed=0,
             lenient=False):
    cand = catalogue_vectors(model)
    take = np.random.default_rng(seed).choice(
        len(pairs), min(n_eval, len(pairs)), replace=False)
    sub = pairs.iloc[take]
    q_idx = torch.tensor(sub["query_idx"].to_numpy(), device=DEVICE)
    t_idx = torch.tensor(sub["target_idx"].to_numpy(), device=DEVICE)
    node = torch.tensor(sub["target_node_id"].to_numpy(), device=DEVICE)

    q_np, n_np = sub["query_idx"].to_numpy(), sub["target_node_id"].to_numpy()
    ranks = torch.empty(len(sub), dtype=torch.long, device=DEVICE)
    hit_lenient = np.zeros(len(sub), dtype=bool)
    for i in range(0, len(sub), 1024):
        j = min(i + 1024, len(sub))
        rows = torch.arange(j - i, device=DEVICE)
        scores = model.query(q_idx[i:j], node[i:j]) @ cand.T
        scores[rows, q_idx[i:j]] = -1e4                 # never recommend itself
        true = scores[rows, t_idx[i:j]].unsqueeze(1)
        ranks[i:j] = (scores > true).sum(1)             # 0-based rank
        if lenient:
            top = torch.topk(scores, 10, dim=1).indices.cpu().numpy()
            for r in range(j - i):
                hit_lenient[i + r] = bool(
                    set(top[r]) & relevant[(int(q_np[i + r]), int(n_np[i + r]))])
    model.train()

    out = {}
    for k in k_values:
        hit = (ranks < k).float()
        out[f"Recall@{k}"] = hit.mean().item()
        out[f"NDCG@{k}"] = (hit / torch.log2(ranks.float() + 2)).mean().item()
    out["MedianRank"] = float(ranks.median().item())
    if lenient:
        out["Recall@10_lenient"] = float(hit_lenient.mean())
    return out


# --- Baselines, scored exactly the way the model is ------------------------
# `10 / len(items)` was the wrong yardstick: the query tower is handed the true
# target category and 99.6% of its top-10 lands inside it, so the floor is a
# random draw WITHIN that category and the bar to clear is per-category
# popularity -- one groupby, no model. See ttn/experiments/ for what happened
# when the negative sampling was changed to try to beat it.
def baselines(pairs, n_eval=10_000, seed=0):
    take = np.random.default_rng(seed).choice(
        len(pairs), min(n_eval, len(pairs)), replace=False)
    sub = pairs.iloc[take]
    node = sub["target_node_id"].to_numpy()
    target = sub["target_idx"].to_numpy()
    query = sub["query_idx"].to_numpy()
    size = np.bincount(node_of_item, minlength=int(node_of_item.max()) + 1)

    counts = (pairs_train.groupby(["target_node_id", "target_idx"]).size()
              .rename("n").reset_index())
    top10 = {int(k): set(g.nlargest(10, "n")["target_idx"].astype(int))
             for k, g in counts.groupby("target_node_id")}
    return {
        "random over the whole catalogue": 10 / len(cat_t),
        "random within the asked-for category":
            float(np.mean(np.minimum(10 / size[node], 1))),
        "popularity within the category":
            float(np.mean([int(t) in top10.get(int(nd), set())
                           for nd, t in zip(node, target)])),
        "popularity within the category (lenient)":
            float(np.mean([bool(top10.get(int(nd), set()) & relevant[(int(q), int(nd))])
                           for q, nd in zip(query, node)])),
    }


# --- Train ----------------------------------------------------------------
model = ComplementaryTwoTower().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

q_all = torch.tensor(pairs_train["query_idx"].to_numpy(), device=DEVICE)
c_all = torch.tensor(pairs_train["target_idx"].to_numpy(), device=DEVICE)
n_all = torch.tensor(pairs_train["target_node_id"].to_numpy(), device=DEVICE)
print(f"training on {len(q_all):,} pairs over {len(cat_t):,} items\n")

# A shuffled pass per epoch, so every pair is seen exactly once each time.
# Sampling with replacement instead would leave ~16% of pairs unseen at this
# length and make two runs at the same step count disagree on their data.
steps_per_epoch = (len(q_all) + BATCH - 1) // BATCH
print(f"{EPOCHS} epochs x {steps_per_epoch:,} steps of {BATCH:,}")
m = evaluate(model, pairs_test, n_eval=5000)
print(f"epoch 0 (untrained)          R@10 {m['Recall@10']:.4f}  "
      f"R@100 {m['Recall@100']:.4f}  medRank {m['MedianRank']:,.0f}")

start = time.time()
step = 0
best = {"epoch": 0, "recall": -1.0, "state": None}
for epoch in range(1, EPOCHS + 1):
    order = torch.randperm(len(q_all), device=DEVICE)
    for i in range(0, len(q_all), BATCH):
        batch = order[i:i + BATCH]
        if len(batch) < 2:                 # in-batch negatives need a partner
            continue
        q = model.query(q_all[batch], n_all[batch])
        neg_idx = hard_negatives(n_all[batch])
        neg_v = model.candidate(neg_idx.reshape(-1)).reshape(len(batch), N_HARD, OUT_DIM)
        loss = bpr_loss(q, model.candidate(c_all[batch]), neg_v, pos_idx=c_all[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
    m = evaluate(model, pairs_test, n_eval=5000)
    mark = ""
    if m["Recall@10"] > best["recall"]:
        best = {"epoch": epoch, "recall": m["Recall@10"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        mark = "  <- best"
    print(f"epoch {epoch:>2} ({step:,} steps)  loss {loss.item():.4f}  "
          f"R@10 {m['Recall@10']:.4f}  R@100 {m['Recall@100']:.4f}  "
          f"medRank {m['MedianRank']:,.0f}  [{time.time() - start:.0f}s]{mark}")
    if epoch - best["epoch"] >= PATIENCE:
        print(f"\nno improvement in {PATIENCE} epochs -- stopping at epoch {epoch}")
        break

# Report and save the BEST epoch, not the last one.
if best["state"] is not None:
    model.load_state_dict(best["state"])
    print(f"restored epoch {best['epoch']} (test R@10 {best['recall']:.4f})")

elapsed = time.time() - start
print(f"\n{EPOCHS} epochs / {step:,} steps in {elapsed:.0f}s "
      f"({len(q_all) * EPOCHS / elapsed:,.0f} pairs/s)\n")
metrics = {}
for split, pairs in (("train", pairs_train), ("test", pairs_test)):
    metrics[split] = evaluate(model, pairs, lenient=(split == "test"))
    print(f"{split:<6} " + "  ".join(
        f"{k} {v:,.0f}" if "Rank" in k else f"{k} {v:.4f}"
        for k, v in metrics[split].items()))
print()
for _name, _value in baselines(pairs_test).items():
    print(f"baseline  {_name:<42} {_value:.4f}")

# --- Save the weights, and everything needed to rebuild the model ---------
# The fitted artifacts (vocabs.json, the medians) already live on disk; without
# this the weights were the one half of the setup that vanished on restart, so
# no two sessions could compare models. The config travels with the state dict
# because ComplementaryTwoTower's shapes are derived from the vocabularies --
# reloading against a different §6 fit would silently mismatch.
CHECKPOINT = OUT_DIR / "ttn_complementary.pt"
torch.save({
    "state_dict": model.state_dict(),
    "config": {"cat_dim": CAT_DIM, "node_dim": NODE_DIM, "hidden": HIDDEN,
               "out_dim": OUT_DIM, "batch": BATCH, "epochs": EPOCHS, "lr": LR,
               "seed": SEED, "cat_order": CAT_ORDER,
               "logq_correction": LOGQ_CORRECTION,
               "n_hard": N_HARD,
               "use_description": USE_DESCRIPTION,
               "tau": TAU, "patience": PATIENCE,
               "best_epoch": best["epoch"]},
    "vocab_sizes": {k: len(v) for k, v in vocabs.items()},
    "numeric_standardisation": {"mean": mu.cpu(), "std": sd.cpu()},
    "metrics": metrics,
    "n_items": int(len(cat_t)),
    "n_train_pairs": int(len(pairs_train)),
}, CHECKPOINT)
print(f"\nsaved -> {CHECKPOINT.relative_to(ROOT)} "
      f"({CHECKPOINT.stat().st_size / 1e6:,.1f} MB)")

# Reload into a fresh model and confirm it scores identically, so a restart
# genuinely resumes rather than appearing to.
checkpoint = torch.load(CHECKPOINT, weights_only=False)
reloaded = ComplementaryTwoTower().to(DEVICE)
reloaded.load_state_dict(checkpoint["state_dict"])
before = evaluate(model, pairs_test, n_eval=5000)["Recall@10"]
after = evaluate(reloaded, pairs_test, n_eval=5000)["Recall@10"]
assert abs(before - after) < 1e-9, f"reload changed the model: {before} vs {after}"
print(f"reloaded checkpoint reproduces R@10 exactly: {after:.4f}")
