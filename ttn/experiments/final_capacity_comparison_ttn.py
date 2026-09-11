"""§13's proposed-but-never-run capacity test, TTN half.

§13 ("Results log") lists as open work: "Can the model overfit a small subset?
Train on ~20k pairs for 50 epochs." That was never executed in the notebook --
the headline TTN numbers everywhere else (§13's before/after table, §18, §19)
come from the full 2.2M-pair, 30-epoch, early-stopped run. This runs that
never-executed diagnostic, twice (image block off/on), under a reduced-data
regime -- ~20k co-purchase training pairs, 50 epochs -- plus the no-model
popularity baseline for reference.

Originally one script with a similarity-baseline half (DESC+TITLE+FEAT[+IMG]);
split in two so this repo's ttn_model branch carries only the TTN halves of
the experiments that touch both. The similarity half is
final_capacity_comparison_similarity.py, on the similarity_model branch.

Everything -- TTN training data, the "target's training frequency" bucket, and
the popularity counts -- uses the SAME ~20,000-pair TRAINING sample, seed 0.
Evaluation is a separate 20,000-pair TEST sample, seed 0, scored inside the
pair's target category (candidates restricted to node_of_item ==
target_node_id, self excluded), hit@k via `target in topk(scores)` membership.

Usage: python ttn/experiments/final_capacity_comparison_ttn.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
img_emb = np.load(OUT_DIR / "siglip_img_emb.npy").astype("float32")
n_items = len(asins)

full_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
print(f"loaded in {time.time() - t0:.0f}s | full train {len(full_train):,} | "
      f"test {len(pairs_test):,}")

# --- THE ~20k-pair training sample, used below -----------------------------
train_sample = full_train.sample(n=min(N_TRAIN_SAMPLE, len(full_train)),
                                  random_state=SEED).reset_index(drop=True)
print(f"training sample: {len(train_sample):,} pairs (seed {SEED})")

# --- in-node candidate pools -------------------------------------------------
order = np.argsort(node_of_item, kind="stable")
starts = np.searchsorted(node_of_item[order], np.arange(int(node_of_item.max()) + 2))


def cands_for_node(node_id):
    return order[starts[node_id]:starts[node_id + 1]]


# --- POP, from the SAME 20k-pair training sample (no-model reference) ------
counts = (train_sample.groupby(["target_node_id", "target_idx"]).size()
          .rename("n").reset_index())
top10_by_node = {int(k): set(g.nlargest(10, "n")["target_idx"].astype(int))
                  for k, g in counts.groupby("target_node_id")}
top100_by_node = {int(k): set(g.nlargest(100, "n")["target_idx"].astype(int))
                   for k, g in counts.groupby("target_node_id")}

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


# --- the test sample, bucketed by the 20k-pair training sample's frequency --
rng = np.random.default_rng(SEED)
take = rng.choice(len(pairs_test), min(N_TEST_SAMPLE, len(pairs_test)), replace=False)
sample = pairs_test.iloc[take].reset_index(drop=True)
sample["freq"] = sample["target_idx"].map(target_freq).fillna(0).astype(int)
sample["bucket"] = sample["freq"].apply(bucket)
print(f"\ntest sample: {len(sample):,} pairs (seed {SEED})")
print(sample["bucket"].value_counts().reindex(BUCKET_ORDER[:-1]))

hit10 = {"POP": np.zeros(len(sample), dtype=bool)}
hit100 = {"POP": np.zeros(len(sample), dtype=bool)}
for node_id, grp in sample.groupby("target_node_id"):
    node_id = int(node_id)
    p10, p100 = top10_by_node.get(node_id, set()), top100_by_node.get(node_id, set())
    rows = grp.index.to_numpy()
    for r, t in enumerate(grp["target_idx"].to_numpy()):
        hit10["POP"][rows[r]] = int(t) in p10
        hit100["POP"][rows[r]] = int(t) in p100

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
    return g["model"], g


ttn_models = {}
for use_image in (False, True):
    tag = "TTN+IMAGE" if use_image else "TTN"
    model, g = train_ttn(use_image)
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
COLS = ["TTN", "TTN+IMAGE", "POP"]


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


table(hit10, f"Recall@10  (train sample {len(train_sample):,} pairs / {EPOCHS} epochs, "
             f"test sample {len(sample):,} pairs, seed {SEED})")
table(hit100, f"Recall@100  (train sample {len(train_sample):,} pairs / {EPOCHS} epochs, "
              f"test sample {len(sample):,} pairs, seed {SEED})")

print(f"\ntotal runtime: {time.time() - t0:.0f}s")
