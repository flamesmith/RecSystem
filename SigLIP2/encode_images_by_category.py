"""Encode SigLIP2 image embeddings for items OUTSIDE the existing snapshots
-- same approach as encode_siglip2_images.py (frozen base model, streaming
download/encode/delete, resumable), but driven by a category slice of
df_features.pkl's full 1,134,566 items instead of a --snapshot's item
list. Lets the ~460K items never attempted (because they're not in any
current snapshot -- see this session's discussion) get encoded
incrementally, one category at a time, rather than one multi-hour run.

Writes into the SAME shared, asin-keyed cache
(SigLIP2/image_cache/cache_{asins,emb,status}.npy) encode_siglip2_images.py
already uses and grows over time -- an item encoded here is immediately
available to anything that reads that cache later, snapshot-scoped or not.
This script does NOT write a per-category "projected" file the way
encode_siglip2_images.py projects onto a snapshot's item order -- there's
no snapshot here to project onto. The shared cache itself is the
deliverable; project it onto whatever population you actually need later
(a df_features.pkl-scoped file, say) once enough categories are done.

Status codes, resumability, and all pipeline knobs (FETCH_WORKERS,
ENCODE_BATCH_SIZE, IMAGE_SIZE_TAG, REQUEST_TIMEOUT) are identical to
encode_siglip2_images.py -- see that file's docstring for what each means.

Usage: python SigLIP2/encode_images_by_category.py --category Furniture
       python SigLIP2/encode_images_by_category.py --category Furniture --max-images 500   # smoke test
"""

# ============================================================================
# 1. Configuration
# ============================================================================
import argparse
from pathlib import Path

MODEL_ID = "google/siglip2-base-patch16-224"

ENCODE_BATCH_SIZE = 16
FETCH_WORKERS     = 32
FETCH_CHUNK       = 256
SAVE_EVERY        = 1024
IMAGE_SIZE_TAG    = "._SX224_"
KEEP_IMAGES       = False
REQUEST_TIMEOUT   = 15
SEED = 42


def _find_root(start: Path) -> Path:
    for p in (start, *start.parents):
        if (p / "data" / "tower").is_dir():
            return p
    raise FileNotFoundError("run this from inside the RecSystem repo")


parser = argparse.ArgumentParser()
parser.add_argument("--category", required=True,
                     help="cat_2 value from df_features.pkl, e.g. Furniture, Bedding")
parser.add_argument("--max-images", type=int, default=None,
                     help="process only this many pending images (smoke test); default: all pending")
ARGS = parser.parse_args()
CATEGORY = ARGS.category

ROOT      = _find_root(Path.cwd())
DATA_DIR  = ROOT / "data"

SIGLIP_ROOT  = ROOT / "SigLIP2"
CACHE_DIR    = SIGLIP_ROOT / "image_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ASINS  = CACHE_DIR / "cache_asins.npy"
CACHE_EMB    = CACHE_DIR / "cache_emb.npy"
CACHE_STATUS = CACHE_DIR / "cache_status.npy"
IMAGE_CACHE  = SIGLIP_ROOT / "_image_download_cache"

print("repo root:", ROOT, "| category:", CATEGORY)

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
from data_creation.complementary_cats_pairs import first_image_url

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
torch.manual_seed(SEED)
print({"device": str(DEVICE), "torch": torch.__version__})

# ============================================================================
# 3. Resolve this category's items and image URLs -- from df_features.pkl
# directly, not any --snapshot's item_asins.npy.
# ============================================================================
feat = pd.read_pickle(DATA_DIR / "df_features.pkl")
feat = feat[feat["cat_2"] == CATEGORY][["asin", "imageURL", "imageURLHighRes"]].drop_duplicates("asin")
if feat.empty:
    raise ValueError(f"no df_features.pkl rows with cat_2 == {CATEGORY!r} -- check spelling/casing")
feat["url"] = first_image_url(feat["imageURLHighRes"], feat["imageURL"])

asins = feat["asin"].to_numpy().astype(str)
urls = np.array([u if isinstance(u, str) else "" for u in feat["url"]], dtype=object)
have_url = np.array([bool(u) for u in urls])
url_of = dict(zip(asins, urls))
n_items = len(asins)
print(f"{CATEGORY} items in df_features.pkl: {n_items:,}")
print(f"with an image URL: {have_url.sum():,} ({have_url.mean():.1%})")

# ============================================================================
# 4. Load SigLIP2
# ============================================================================
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

EMB_DIM = model.config.vision_config.hidden_size
n_params = sum(p.numel() for p in model.parameters())
print(f"{model.__class__.__name__}: {n_params/1e6:.0f}M params on {DEVICE} | image dim {EMB_DIM}")

# ============================================================================
# 5. Embedding store -- same shared, asin-keyed cache encode_siglip2_images.py uses
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
idx_of = {a: i for i, a in enumerate(asins)}

new_asins = [a for a in asins if a not in cache_idx_of]
if new_asins:
    new_status = np.array([0 if have_url[idx_of[a]] else 2 for a in new_asins], dtype="int8")
    base = len(cache_asins)
    cache_asins = np.concatenate([cache_asins, np.array(new_asins, dtype=str)])
    cache_emb = np.concatenate([cache_emb, np.zeros((len(new_asins), EMB_DIM), dtype="float32")])
    cache_status = np.concatenate([cache_status, new_status])
    cache_idx_of.update({a: base + k for k, a in enumerate(new_asins)})
print(f"cache after extending for {CATEGORY}: {len(cache_asins):,} asins "
      f"({len(new_asins):,} newly added)")
print("cache status:", {int(k): int(v) for k, v in zip(*np.unique(cache_status, return_counts=True))})

cache_url_of = {cache_idx_of[a]: url_of.get(a, "") for a in asins}

# ============================================================================
# 6. Fetch / encode / delete helpers -- identical to encode_siglip2_images.py
# ============================================================================
IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (RecSystem research image fetch)"})


def sized(url: str) -> str:
    if not IMAGE_SIZE_TAG or "/images/I/" not in url:
        return url
    stem, dot, ext = url.rpartition(".")
    return f"{stem}{IMAGE_SIZE_TAG}{dot}{ext}" if dot and "/" not in ext else url


def fetch(ci: int):
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
    feats = getattr(out, "pooler_output", out)
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
todo = pending if ARGS.max_images is None else pending[:ARGS.max_images]
print(f"pending with URL (this category): {len(pending):,}   |   this run: {len(todo):,}")

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
print("full cache status:", counts)
rem = len(pending) - len(todo)
if rem and len(todo) and dt > 0:
    print(f"est. for the remaining {rem:,} in {CATEGORY}: ~{rem / rate / 60:.0f} min "
          f"(a small run overstates this -- model load and MPS warm-up are one-time)")

print(f"\n{CATEGORY} status within this run's cache indices:",
      {codes[int(k)]: int(v) for k, v in
       zip(*np.unique(cache_status[[cache_idx_of[a] for a in asins]], return_counts=True))})
