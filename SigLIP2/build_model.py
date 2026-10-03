"""Train the SigLIP2 image/text adapter from scratch -- the actual training
step `encode_siglip2_images.py` does NOT do (that script only runs a frozen
model for inference). Mirrors TTN/build_model.py's shape: trains on a
snapshot, saves a versioned checkpoint, never overwrites a previous run.

SCOPE (a deliberate choice, not the full training system): this is a small,
self-contained script -- the same ResidualAdapter/ContrastiveAdapters
architecture already in adapter.py, trained with a straightforward
contrastive loss on a sampled subset of this snapshot's items. It is NOT a
port of the `SigLIP2` branch's SigLIP2_training/ package (no hard-negative
mining, no multi-phase guardrails, no separate feature-shard caching layer).
Starts from a RANDOMLY INITIALIZED adapter every run -- it does not continue
training from checkpoints/pilot_200k/taxonomy_w20_random_200k.pt.

Both image and text embeddings for the sampled subset are computed inline,
not read from a precomputed catalogue-wide file -- cheap enough at subset
scale (text encoding is fast; images are only read from the already-built
per-snapshot embeddings, not re-downloaded) that a separate catalogue-wide
text-embedding artifact isn't needed just to run this.

Prerequisite: SigLIP2/encode_siglip2_images.py must have already run for
the given --snapshot, to produce that snapshot's siglip_img_emb.npy /
_status.npy this script samples images from.

Usage: python SigLIP2/build_model.py --snapshot w90_2017-12-09 [--n-items 5000]
Output: SigLIP2/generated/<snapshot_id>/models/<date>_v_00x/
          adapter.pt             state_dict, config, metrics
          version_manifest.json  same identity/config/metrics, readable without loading torch
"""
import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModel, AutoTokenizer

from SigLIP2.adapter import ContrastiveAdapters
from SigLIP2.description_pipeline import DescriptionProcessor, ProductInput, load_yaml
from SigLIP2.text_embeddings import encode_texts

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_snapshot.py, e.g. w90_2017-12-09")
parser.add_argument("--n-items", type=int, default=5000,
                     help="how many items to sample for this training run (default 5000 -- "
                          "a fast first test, not the full catalogue)")
ARGS = parser.parse_args()
SNAPSHOT_ID = ARGS.snapshot

DATA_DIR = ROOT / "data"
SNAP_DIR = DATA_DIR / "tower" / SNAPSHOT_ID                    # shared artifacts
SIGLIP_DIR = ROOT / "SigLIP2" / "generated" / SNAPSHOT_ID       # SigLIP2-specific artifacts

MODEL_ID = "google/siglip2-base-patch16-224"   # same checkpoint encode_siglip2_images.py uses

# --- adapter hyperparameters -- reused from the imported checkpoint's own
# training_config (checkpoints/pilot_200k/README.md), since those are already
# proven reasonable for this exact architecture/domain, not arbitrary picks.
BOTTLENECK = 128
DROPOUT = 0.05
MAXIMUM_LOGIT_SCALE = 200.0
# The adapter's logit_scale/logit_bias MUST start from the base model's own
# native calibration, not an arbitrary guess -- the untrained adapter is
# ~identity (zero-init up-projection), so at init its output IS essentially
# the raw SigLIP2 embedding space, which was pretrained/calibrated at this
# exact scale. An arbitrary init here (e.g. 10.0) was tried first and made
# training actively WORSE than the untrained baseline (0.34 vs 0.385
# bidirectional_recall@10) -- this fixes that.
INIT_LOGIT_SCALE = 112.66889953613281   # = google/siglip2-base-patch16-224's own logit_scale.exp()
INIT_LOGIT_BIAS = -16.771724700927734   # = that same model's own logit_bias
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
EPOCHS = 30
PATIENCE = 5
VAL_FRACTION = 0.1
SEED = 0

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
print(f"device: {DEVICE} | snapshot: {SNAPSHOT_ID} | target items: {ARGS.n_items:,}")

# ============================================================================
# 1. Sample items with BOTH a usable image embedding AND usable text
# ============================================================================
asins = np.load(SNAP_DIR / "item_asins.npy", allow_pickle=False).astype(str)
img_emb = np.load(SIGLIP_DIR / "siglip_img_emb.npy").astype("float32")
img_status = np.load(SIGLIP_DIR / "siglip_img_status.npy")
has_img = img_status == 1
print(f"items with a usable image: {has_img.sum():,} of {len(asins):,}")

feat = pd.read_pickle(DATA_DIR / "df_features.pkl")
feat = feat[feat["asin"].isin(set(asins[has_img]))].drop_duplicates("asin").set_index("asin")

rng = np.random.default_rng(SEED)
candidate_asins = asins[has_img]
candidate_asins = candidate_asins[np.isin(candidate_asins, feat.index)]
rng.shuffle(candidate_asins)

# ============================================================================
# 2. Build text views for the candidates, stop once n-items have usable text
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = DescriptionProcessor(load_yaml(), tokenizer=tokenizer)

rows, texts = [], []
for asin in candidate_asins:
    if len(rows) >= ARGS.n_items:
        break
    r = feat.loc[asin]
    product = ProductInput(
        product_id=asin,
        title=str(r.get("title", "")) or "",
        descriptions=[str(r["description"])] if pd.notna(r.get("description")) else [],
        features=[str(r["feature"])] if pd.notna(r.get("feature")) else [],
        details={"Brand": str(r.get("brand", "")) or "", "Color": str(r.get("Color", "")) or ""},
        categories=[str(r.get("cat_2", "")), str(r.get("cat_3", "")), str(r.get("cat_4", ""))],
    )
    result = processor.transform(product)
    if not result.description_canonical:
        continue   # no usable text for this item -- skip it, don't pad with empties
    rows.append(asin)
    texts.append(result.description_canonical)

n_items = len(rows)
if n_items < 50:
    raise RuntimeError(f"only {n_items} items had both a usable image and usable text -- "
                        f"too few to train on; try a larger --n-items or a snapshot with more coverage")
print(f"sampled {n_items:,} items with both a usable image and usable text")

idx_of_asin = {a: i for i, a in enumerate(asins)}
image_arr = img_emb[[idx_of_asin[a] for a in rows]]
text_arr = encode_texts(model, tokenizer, texts, device=DEVICE, max_length=64)
del model, tokenizer   # frees the base model; only the small adapter trains from here

# ============================================================================
# 3. Train/val split, same bidirectional-recall evaluation family the
#    imported checkpoint's own training used
# ============================================================================
perm = rng.permutation(n_items)
n_val = max(1, int(n_items * VAL_FRACTION))
val_idx, train_idx = perm[:n_val], perm[n_val:]
print(f"train {len(train_idx):,} | val {len(val_idx):,}")

image_t = torch.tensor(image_arr, device=DEVICE)
text_t = torch.tensor(text_arr, device=DEVICE)


def _sigmoid_contrastive_loss(logits):
    labels = -torch.ones_like(logits)
    labels.fill_diagonal_(1.0)
    return -F.logsigmoid(labels * logits).sum(dim=1).mean()


@torch.no_grad()
def bidirectional_recall_at_k(adapter, idx, k=10):
    adapter.eval()
    img_v, txt_v = adapter.embeddings(image_t[idx], text_t[idx])
    scores = img_v @ txt_v.T
    n = len(idx)
    labels = torch.arange(n, device=DEVICE)
    i2t = (scores.topk(min(k, n), dim=1).indices == labels[:, None]).any(1).float().mean().item()
    t2i = (scores.T.topk(min(k, n), dim=1).indices == labels[:, None]).any(1).float().mean().item()
    adapter.train()
    return {"image_to_text_recall@10": i2t, "text_to_image_recall@10": t2i,
            "bidirectional_recall@10": (i2t + t2i) / 2}


# ============================================================================
# 4. Train -- randomly-initialized adapter, every run
# ============================================================================
adapter = ContrastiveAdapters(
    dimension=image_t.shape[1], bottleneck=BOTTLENECK, dropout=DROPOUT,
    logit_scale=INIT_LOGIT_SCALE, logit_bias=INIT_LOGIT_BIAS, maximum_logit_scale=MAXIMUM_LOGIT_SCALE,
).to(DEVICE)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

m0 = bidirectional_recall_at_k(adapter, val_idx)
print(f"epoch 0 (untrained)  bidirectional_recall@10 {m0['bidirectional_recall@10']:.4f}")

start = time.time()
best = {"epoch": 0, "recall": -1.0, "state": None}
for epoch in range(1, EPOCHS + 1):
    order = train_idx[rng.permutation(len(train_idx))]
    for i in range(0, len(order), BATCH):
        batch = torch.tensor(order[i:i + BATCH], device=DEVICE)
        if len(batch) < 2:
            continue
        img_v, txt_v = adapter.embeddings(image_t[batch], text_t[batch])
        scale = adapter.logit_scale.exp().clamp(max=adapter.maximum_logit_scale)
        logits = scale * img_v @ txt_v.T + adapter.logit_bias
        loss = _sigmoid_contrastive_loss(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    m = bidirectional_recall_at_k(adapter, val_idx)
    mark = ""
    if m["bidirectional_recall@10"] > best["recall"]:
        best = {"epoch": epoch, "recall": m["bidirectional_recall@10"],
                "state": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}}
        mark = "  <- best"
    print(f"epoch {epoch:>2}  loss {loss.item():.4f}  "
          f"bidirectional_recall@10 {m['bidirectional_recall@10']:.4f}  "
          f"[{time.time() - start:.0f}s]{mark}")
    if epoch - best["epoch"] >= PATIENCE:
        print(f"\nno improvement in {PATIENCE} epochs -- stopping at epoch {epoch}")
        break

adapter.load_state_dict(best["state"])
elapsed = time.time() - start
final_metrics = bidirectional_recall_at_k(adapter, val_idx)
print(f"\nrestored epoch {best['epoch']} (val bidirectional_recall@10 {best['recall']:.4f}) "
      f"in {elapsed:.0f}s")

# ============================================================================
# 5. Save -- same versioning scheme as TTN/build_model.py
# ============================================================================
MODELS_DIR = SIGLIP_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
today = date.today().isoformat()
existing = sorted(int(p.name.rsplit("_v_", 1)[1]) for p in MODELS_DIR.glob(f"{today}_v_*")
                   if p.is_dir() and p.name.rsplit("_v_", 1)[1].isdigit())
VERSION_ID = f"{today}_v_{(existing[-1] + 1 if existing else 1):03d}"
VERSION_DIR = MODELS_DIR / VERSION_ID
VERSION_DIR.mkdir()

try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    git_commit = None

CONFIG = {"dimension": image_t.shape[1], "bottleneck": BOTTLENECK, "dropout": DROPOUT,
          "maximum_logit_scale": MAXIMUM_LOGIT_SCALE, "lr": LR, "weight_decay": WEIGHT_DECAY,
          "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE, "seed": SEED,
          "n_items_requested": ARGS.n_items, "n_items_used": n_items,
          "text_view": "description_canonical", "base_model": MODEL_ID,
          "best_epoch": best["epoch"]}
CHECKPOINT = VERSION_DIR / "adapter.pt"
torch.save({
    "state_dict": adapter.state_dict(),
    "config": CONFIG,
    "metrics": {"untrained": m0, "val": final_metrics},
    "snapshot_id": SNAPSHOT_ID,
    "version_id": VERSION_ID,
    "git_commit": git_commit,
    "trained_from_scratch": True,   # never continues from checkpoints/pilot_200k/
}, CHECKPOINT)
print(f"\nsaved -> {CHECKPOINT.relative_to(ROOT)}")

import json
json.dump({
    "version_id": VERSION_ID, "snapshot_id": SNAPSHOT_ID, "git_commit": git_commit,
    "config": CONFIG, "metrics": {"untrained": m0, "val": final_metrics},
}, open(VERSION_DIR / "version_manifest.json", "w"), indent=1)

# Reload and confirm it reproduces the same metric -- same self-check TTN/build_model.py does
checkpoint = torch.load(CHECKPOINT, weights_only=False)
reloaded = ContrastiveAdapters(
    dimension=CONFIG["dimension"], bottleneck=CONFIG["bottleneck"], dropout=CONFIG["dropout"],
    logit_scale=INIT_LOGIT_SCALE, logit_bias=INIT_LOGIT_BIAS, maximum_logit_scale=CONFIG["maximum_logit_scale"],
).to(DEVICE)
reloaded.load_state_dict(checkpoint["state_dict"])
before = bidirectional_recall_at_k(adapter, val_idx)["bidirectional_recall@10"]
after = bidirectional_recall_at_k(reloaded, val_idx)["bidirectional_recall@10"]
assert abs(before - after) < 1e-9, f"reload changed the model: {before} vs {after}"
print(f"reloaded checkpoint reproduces bidirectional_recall@10 exactly: {after:.4f}")
print(f"\nversion {VERSION_ID} is a CANDIDATE -- not wired into generate_recommendations.py yet, "
      f"and not automatically used anywhere.")
