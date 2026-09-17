"""Build SigLIP2 image embeddings for "Visually Similar Products".

Encodes each tower item's product image into a 768-d SigLIP2 vector
(google/siglip2-base-patch16-224, F.normalize'd). Streaming pipeline: each
image is downloaded, encoded, and deleted immediately, so peak disk use is a
few hundred images at a time, not the ~5GB the full catalogue's images would
take.

Resumable: data/tower/siglip_img_emb.npy and siglip_img_status.npy persist
between runs. Status codes: 0 pending, 1 encoded, 2 no URL, 3 download
failed, 4 decode/encode failed. Re-running only touches items still at 0, so
interrupting and restarting is always safe.

Row-aligned to data/tower/item_asins.npy, so build downstream consumers
(TTN's image-augmented model, results.ipynb) can index it the same way they
index items.npz. About 28% of items have no image and get a zero vector,
the same convention TTN uses for blank descriptions.

Prerequisite: TTN/build_data.py must have already run, to produce
data/tower/item_asins.npy (the item list and order this reads).

Usage: python SigLIP2/build_siglip2.py
  Runs a full pass by default (MAX_IMAGES = None below). Edit that constant
  to a small number for a quick smoke test before committing to the full
  run -- the pipeline resumes either way, so a small first pass costs
  nothing.
"""

# ============================================================================
# 1. Configuration
# ============================================================================
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
        if (p / "data" / "tower" / "item_asins.npy").exists():
            return p
    raise FileNotFoundError("run this from inside the RecSystem repo")

ROOT        = _find_root(Path.cwd())
DATA_DIR    = ROOT / "data"
TOWER_DIR   = DATA_DIR / "tower"
ITEM_ASINS  = TOWER_DIR / "item_asins.npy"              # 137,362 tower items, in order
IMAGE_CACHE = TOWER_DIR / "_siglip_image_cache"         # transient; emptied as we go
EMB_PATH    = TOWER_DIR / "siglip_img_emb.npy"          # (n_items, 768) float32, aligned to ITEM_ASINS
STATUS_PATH = TOWER_DIR / "siglip_img_status.npy"       # (n_items,) int8 status codes
print("repo root:", ROOT)

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
# 5. Embedding store -- resumable
# ============================================================================
if EMB_PATH.exists() and STATUS_PATH.exists():
    emb = np.load(EMB_PATH)
    status = np.load(STATUS_PATH)
    assert emb.shape == (n_items, EMB_DIM), emb.shape
    assert status.shape == (n_items,), status.shape
    print(f"resuming — already encoded: {int((status == 1).sum()):,}")
else:
    emb = np.zeros((n_items, EMB_DIM), dtype="float32")
    status = np.zeros(n_items, dtype="int8")
    print("fresh store")

status[(status == 0) & ~have_url] = 2      # items with no URL: nothing to do
print("status:", {int(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))})

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


def fetch(i: int):
    """Download item i's image to the cache dir. Returns (i, path) or (i, None)."""
    path = IMAGE_CACHE / f"{asins[i]}.img"
    try:
        r = _session.get(sized(urls[i]), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        path.write_bytes(r.content)
        return i, path
    except Exception:
        return i, None


@torch.inference_mode()
def encode(items):
    """items: list of (i, path). Writes normalised vectors into `emb`, sets status."""
    imgs, keep = [], []
    for i, p in items:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
            keep.append(i)
        except Exception:
            status[i] = 4
    if not imgs:
        return
    pixel_values = image_processor(images=imgs, return_tensors="pt")["pixel_values"].to(DEVICE)
    out = model.get_image_features(pixel_values=pixel_values)
    feats = getattr(out, "pooler_output", out)          # this transformers version wraps it
    feats = F.normalize(feats, dim=-1).float().cpu().numpy()
    for j, i in enumerate(keep):
        emb[i] = feats[j]
        status[i] = 1


def flush():
    np.save(EMB_PATH, emb)
    np.save(STATUS_PATH, status)

# ============================================================================
# 7. Run the pipeline
# ============================================================================
pending = np.where((status == 0) & have_url)[0]
todo = pending if MAX_IMAGES is None else pending[:int(MAX_IMAGES)]
print(f"pending with URL: {len(pending):,}   |   this run: {len(todo):,}")

t0 = time.time()
since_save = 0
with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
    for c in tqdm(range(0, len(todo), FETCH_CHUNK), desc="chunks"):
        chunk = todo[c:c + FETCH_CHUNK]
        fetched = list(pool.map(fetch, chunk))
        ok = [(i, p) for i, p in fetched if p is not None]
        for i, p in fetched:
            if p is None:
                status[i] = 3
        for b in range(0, len(ok), ENCODE_BATCH_SIZE):
            encode(ok[b:b + ENCODE_BATCH_SIZE])
        if not KEEP_IMAGES:
            for _, p in ok:
                p.unlink(missing_ok=True)
        since_save += len(chunk)
        if since_save >= SAVE_EVERY:
            flush()
            since_save = 0

flush()
if not KEEP_IMAGES:
    shutil.rmtree(IMAGE_CACHE, ignore_errors=True)

dt = time.time() - t0
codes = {0: "pending", 1: "encoded", 2: "no url", 3: "fetch fail", 4: "decode fail"}
counts = {codes[int(k)]: int(v) for k, v in zip(*np.unique(status, return_counts=True))}
rate = len(todo) / max(dt, 1e-9)
print(f"\nthis run: {len(todo):,} items in {dt:.1f}s  ({rate:.1f} img/s)")
print("store status:", counts)
rem = int((status == 0).sum())
if rem and len(todo) and dt > 0:
    print(f"est. for the remaining {rem:,}: ~{rem / rate / 60:.0f} min "
          f"(a small run overstates this — model load and MPS warm-up are one-time)")

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
