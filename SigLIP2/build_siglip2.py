"""Build SigLIP2 image embeddings for "Visually Similar Products".

Encodes each tower item's product image into a 768-d SigLIP2 vector
(google/siglip2-base-patch16-224, F.normalize'd). Streaming pipeline: each
image is downloaded, encoded, and deleted immediately, so peak disk use is a
few hundred images at a time, not the ~5GB the full catalogue's images would
take.

Two-level storage, because item coverage/order changes across snapshots
(different --window-days can change which items even appear) but a given
item's photo doesn't:
  - data/tower/_siglip_cache/ -- ASIN-KEYED, shared across every snapshot.
    Grows over time; an item encoded once is never re-fetched for a later
    snapshot just because that snapshot orders items differently.
  - data/tower/<snapshot_id>/siglip_img_{emb,status}.npy -- ROW-INDEX-KEYED,
    aligned to that snapshot's item_asins.npy, projected from the cache.
    This is what downstream consumers (TTN's image-augmented model,
    results.ipynb, recommendation generation) actually read.

Status codes: 0 pending, 1 encoded, 2 no URL, 3 download failed, 4
decode/encode failed. Re-running only fetches cache items still at 0, so
interrupting and restarting is always safe. About 28% of items have no
image and get a zero vector, the same convention TTN uses for blank
descriptions.

Prerequisite: TTN/build_data.py must have already run for the given
--snapshot, to produce that snapshot's item_asins.npy (the item list this
run needs embeddings for).

Usage: python SigLIP2/build_siglip2.py --snapshot w90_2017-12-09
  Runs a full pass by default (MAX_IMAGES = None below). Edit that constant
  to a small number for a quick smoke test before committing to the full
  run -- the pipeline resumes either way, so a small first pass costs
  nothing.
"""

# ============================================================================
# 1. Configuration
# ============================================================================
import argparse
from pathlib import Path

MODEL_ID = "google/siglip2-base-patch16-224"     # same checkpoint as the SigLIP2 branch

# --- how much to run ----------------------------------------------------------
# The first pass processes only this many images. Set to None (or a large int)
# for the full run once the smoke test looks right; the pipeline resumes.
MAX_IMAGES = None       # None = full run; a small int = quick smoke test

# --- pipeline knobs ---------------------------------------------------------
ENCODE_BATCH_SIZE = 16          # images per SigLIP2 forward pass
FETCH_WORKERS     = 32          # concurrent image downloads
FETCH_CHUNK       = 256         # images fetched before each encode + delete cycle
SAVE_EVERY        = 1024        # flush embeddings to disk every N processed items
IMAGE_SIZE_TAG    = "._SX224_"  # Amazon URL size suffix (smaller download); "" = original
KEEP_IMAGES       = False       # True -> leave the downloaded files on disk
REQUEST_TIMEOUT   = 15          # seconds per image
SEED = 42

# --- paths ----------------------------------------------------------------
def _find_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "data" / "tower").is_dir():
            return p
    raise FileNotFoundError("run this from inside the RecSystem repo")

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from TTN/build_data.py, e.g. w90_2017-12-09")
SNAPSHOT_ID = parser.parse_args().snapshot

ROOT         = _find_root(Path.cwd())
DATA_DIR     = ROOT / "data"
TOWER_DIR    = DATA_DIR / "tower"
SNAPSHOT_DIR = TOWER_DIR / SNAPSHOT_ID
ITEM_ASINS   = SNAPSHOT_DIR / "item_asins.npy"           # this snapshot's items, in order

CACHE_DIR    = TOWER_DIR / "_siglip_cache"               # shared, asin-keyed, grows over time
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ASINS  = CACHE_DIR / "cache_asins.npy"
CACHE_EMB    = CACHE_DIR / "cache_emb.npy"
CACHE_STATUS = CACHE_DIR / "cache_status.npy"
IMAGE_CACHE  = TOWER_DIR / "_siglip_image_cache"         # transient; emptied as we go

EMB_PATH     = SNAPSHOT_DIR / "siglip_img_emb.npy"       # per-snapshot output, projected from the cache
STATUS_PATH  = SNAPSHOT_DIR / "siglip_img_status.npy"
print("repo root:", ROOT, "| snapshot:", SNAPSHOT_ID)

# ============================================================================
# 2. Imports and device
# ============================================================================
import shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(ROOT))
from complementary_cats_pairs import first_image_url

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(SEED)
print({"device": str(DEVICE), "torch": torch.__version__})

# ============================================================================
# 3. Resolve the item list and image URLs
# ============================================================================
asins = np.load(ITEM_ASINS, allow_pickle=False).astype(str)
n_items = len(asins)
idx_of = {a: i for i, a in enumerate(asins)}

feat = pd.read_pickle(DATA_DIR / "df_features.pkl")[["asin", "imageURL", "imageURLHighRes"]]
feat = feat[feat["asin"].isin(idx_of)].drop_duplicates("asin")
feat["url"] = first_image_url(feat["imageURLHighRes"], feat["imageURL"])
url_of = dict(zip(feat["asin"], feat["url"]))

urls = np.array([url_of.get(a) if isinstance(url_of.get(a), str) else "" for a in asins], dtype=object)
have_url = np.array([bool(u) for u in urls])
print(f"tower items      : {n_items:,}")
print(f"with an image URL: {have_url.sum():,} ({have_url.mean():.1%})")

# ============================================================================
# 4. Load SigLIP2
# ============================================================================
# use_fast=False -> the PIL/numpy image processor (no torchvision dependency)
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

EMB_DIM = model.config.vision_config.hidden_size
n_params = sum(p.numel() for p in model.parameters())
print(f"{model.__class__.__name__}: {n_params/1e6:.0f}M params on {DEVICE} | image dim {EMB_DIM}")

# ============================================================================
# 5. Embedding store -- resumable, asin-keyed, shared across every snapshot
# ============================================================================
if CACHE_ASINS.exists():
    cache_asins = np.load(CACHE_ASINS, allow_pickle=False).astype(str)
    cache_emb = np.load(CACHE_EMB)
    cache_status = np.load(CACHE_STATUS)
    assert cache_emb.shape == (len(cache_asins), EMB_DIM), cache_emb.shape
    assert cache_status.shape == (len(cache_asins),), cache_status.shape
    print(f"cache: {len(cache_asins):,} asins tracked, "
          f"{int((cache_status == 1).sum()):,} already encoded")
else:
    cache_asins = np.array([], dtype=str)
    cache_emb = np.zeros((0, EMB_DIM), dtype="float32")
    cache_status = np.zeros(0, dtype="int8")
    print("fresh cache")

cache_idx_of = {a: i for i, a in enumerate(cache_asins)}

# Extend the cache with any of THIS snapshot's asins it hasn't seen before --
# an item already cached (from any other snapshot) is never re-added.
new_asins = [a for a in asins if a not in cache_idx_of]
if new_asins:
    new_status = np.array([0 if have_url[idx_of[a]] else 2 for a in new_asins], dtype="int8")
    base = len(cache_asins)
    cache_asins = np.concatenate([cache_asins, np.array(new_asins, dtype=str)])
    cache_emb = np.concatenate([cache_emb, np.zeros((len(new_asins), EMB_DIM), dtype="float32")])
    cache_status = np.concatenate([cache_status, new_status])
    cache_idx_of.update({a: base + k for k, a in enumerate(new_asins)})
print(f"cache after extending for this snapshot: {len(cache_asins):,} asins "
      f"({len(new_asins):,} newly added)")
print("cache status:", {int(k): int(v) for k, v in zip(*np.unique(cache_status, return_counts=True))})

# URL for each of this run's pending cache indices -- only known for THIS
# snapshot's asins, which is fine: we only ever fetch cache indices reachable
# from this run's own asin set below.
cache_url_of = {cache_idx_of[a]: url_of.get(a, "") for a in asins}

# ============================================================================
# 6. Fetch / encode / delete helpers
# ============================================================================
IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (RecSystem research image fetch)"})


def sized(url: str) -> str:
    """Insert Amazon's size suffix so the download is a ~224px thumbnail."""
    if not IMAGE_SIZE_TAG or "/images/I/" not in url:
        return url
    stem, dot, ext = url.rpartition(".")
    return f"{stem}{IMAGE_SIZE_TAG}{dot}{ext}" if dot and "/" not in ext else url


def fetch(ci: int):
    """Download cache index ci's image to the cache dir. Returns (ci, path) or (ci, None)."""
    path = IMAGE_CACHE / f"{cache_asins[ci]}.img"
    try:
        r = _session.get(sized(cache_url_of[ci]), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        path.write_bytes(r.content)
        return ci, path
    except Exception:
        return ci, None


@torch.inference_mode()
def encode(items):
    """items: list of (cache_idx, path). Writes normalised vectors into cache_emb, sets cache_status."""
    imgs, keep = [], []
    for ci, p in items:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
            keep.append(ci)
        except Exception:
            cache_status[ci] = 4
    if not imgs:
        return
    pixel_values = image_processor(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
    out = model.get_image_features(pixel_values=pixel_values)
    feats = getattr(out, "pooler_output", out)          # this transformers version wraps it
    feats = F.normalize(feats, dim=-1).float().cpu().numpy()
    for j, ci in enumerate(keep):
        cache_emb[ci] = feats[j]
        cache_status[ci] = 1


def flush_cache():
    np.save(CACHE_ASINS, cache_asins)
    np.save(CACHE_EMB, cache_emb)
    np.save(CACHE_STATUS, cache_status)

# ============================================================================
# 7. Run the pipeline
# ============================================================================
pending = np.array([cache_idx_of[a] for a in asins if cache_status[cache_idx_of[a]] == 0])
todo = pending if MAX_IMAGES is None else pending[:int(MAX_IMAGES)]
print(f"pending with URL (this snapshot): {len(pending):,}   |   this run: {len(todo):,}")

t0 = time.time()
since_save = 0
with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
    for c in tqdm(range(0, len(todo), FETCH_CHUNK), desc="chunks"):
        chunk = todo[c:c + FETCH_CHUNK]
        fetched = list(pool.map(fetch, chunk))
        ok = [(ci, p) for ci, p in fetched if p is not None]
        for ci, p in fetched:
            if p is None:
                cache_status[ci] = 3
        for b in range(0, len(ok), ENCODE_BATCH_SIZE):
            encode(ok[b:b + ENCODE_BATCH_SIZE])
        if not KEEP_IMAGES:
            for _, p in ok:
                p.unlink(missing_ok=True)
        since_save += len(chunk)
        if since_save >= SAVE_EVERY:
            flush_cache()
            since_save = 0

flush_cache()
if not KEEP_IMAGES:
    shutil.rmtree(IMAGE_CACHE, ignore_errors=True)

dt = time.time() - t0
codes = {0: "pending", 1: "encoded", 2: "no url", 3: "fetch fail", 4: "decode fail"}
counts = {codes[int(k)]: int(v) for k, v in zip(*np.unique(cache_status, return_counts=True))}
rate = len(todo) / max(dt, 1e-9)
print(f"\nthis run: {len(todo):,} items in {dt:.1f}s  ({rate:.1f} img/s)")
print("cache status:", counts)
rem = len(pending) - len(todo)
if rem and len(todo) and dt > 0:
    print(f"est. for the remaining {rem:,} in this snapshot: ~{rem / rate / 60:.0f} min "
          f"(a small run overstates this — model load and MPS warm-up are one-time)")

# ============================================================================
# 7b. Project the cache onto this snapshot's own item order
# ============================================================================
# What everything downstream (TTN's image-augmented model, results.ipynb,
# recommendation generation) actually reads -- row-index-keyed, aligned to
# THIS snapshot's item_asins.npy, same convention as items.npz/node_of_item.npy.
snap_idx = np.array([cache_idx_of[a] for a in asins])
emb = cache_emb[snap_idx]
status = cache_status[snap_idx]
np.save(EMB_PATH, emb)
np.save(STATUS_PATH, status)
print(f"\nprojected -> {EMB_PATH.relative_to(ROOT)}, {STATUS_PATH.relative_to(ROOT)} "
      f"({len(asins):,} items)")

# ============================================================================
# 8. Sanity check
# ============================================================================
done = np.where(status == 1)[0]
print(f"encoded: {len(done):,}   saved to {EMB_PATH.relative_to(ROOT)}")
if len(done) >= 2:
    with np.errstate(all="ignore"):                 # quiet a spurious Accelerate BLAS warning
        V = emb[done]
        norms = (V ** 2).sum(1)
        idx = done[:min(len(done), 12)]
        S = np.clip(V[:len(idx)] @ V[:len(idx)].T, -1.0, 1.0)
        tri = S[np.triu_indices(len(idx), 1)]
    print(f"vector norms: {norms.min():.3f}–{norms.max():.3f} (expect ~1.0)")
    print(f"pairwise cosine among the first {len(idx)}: {tri.min():.3f}–{tri.max():.3f}")
    for i in idx[:10]:
        print(f"  {asins[i]}  {str(urls[i])[:70]}")
