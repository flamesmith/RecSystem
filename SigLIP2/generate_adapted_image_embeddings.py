"""Apply the current SigLIP2 champion adapter to a snapshot's existing raw
image embeddings, catalogue-wide -- closes the gap flagged repeatedly this
session: nothing had ever run adapt_image() across a full snapshot and
saved the result, only on individual items as a verification check.

Scope: IMAGE side only, by deliberate choice -- for image-to-image
"visually similar" recommendations, not cross-modal text search (the
text side would need raw SigLIP2 text embeddings, which don't exist at
this scale yet and aren't needed for this comparison).

Reads the champion from SigLIP2/champion_selection_siglip.json rather than
a hardcoded path, so this always applies whichever adapter is currently
recorded as champion -- today that's checkpoints/pilot_200k/
taxonomy_w20_random_200k.pt, trained on 200K products, not this
snapshot's own items.

This is fast, not resumable like encode_siglip2_images.py: the raw image
embeddings already exist (that script's own output), so this is just a
small matrix operation through a 396K-parameter adapter, not network-bound
image downloads. A few seconds for the whole snapshot, not minutes.

Items with no usable image (status != 1 in siglip_img_status.npy, about
28% of the catalogue) keep their zero vector -- adapting a zero vector
through a zero-initialized-up-projection residual adapter still returns
(approximately) a zero vector, so the same status array still correctly
flags them; no separate status file is written.

Output: SigLIP2/generated/<snapshot_id>/adapted_img_emb.npy (one 768-d
vector per item, same order as item_asins.npy) +
adapted_img_emb_manifest.json (which checkpoint produced it, for
traceability -- the champion file can change later).

Prerequisite: SigLIP2/encode_siglip2_images.py must have already run for
the given --snapshot.

Usage: python SigLIP2/generate_adapted_image_embeddings.py --snapshot w60_2017-12-09
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SigLIP2.adapter import load_adapter, adapt_image

BATCH = 8192

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_snapshot.py, e.g. w60_2017-12-09")
args = parser.parse_args()
SNAPSHOT_ID = args.snapshot

SIGLIP_DIR = ROOT / "SigLIP2" / "generated" / SNAPSHOT_ID
OUT_PATH = SIGLIP_DIR / "adapted_img_emb.npy"
MANIFEST_PATH = SIGLIP_DIR / "adapted_img_emb_manifest.json"

t0 = time.time()
img_emb_path = SIGLIP_DIR / "siglip_img_emb.npy"
assert img_emb_path.exists(), (
    f"missing: {img_emb_path} -- run "
    f"'python SigLIP2/encode_siglip2_images.py --snapshot {SNAPSHOT_ID}' first")
img_emb = np.load(img_emb_path).astype("float32")
img_status = np.load(SIGLIP_DIR / "siglip_img_status.npy")
n_items = len(img_emb)
print(f"snapshot: {SNAPSHOT_ID} | items: {n_items:,} | with usable image: "
      f"{(img_status == 1).sum():,} ({(img_status == 1).mean():.1%})")

champion_path = ROOT / "SigLIP2" / "champion_selection_siglip.json"
champion = json.loads(champion_path.read_text())
checkpoint_path = ROOT / champion["checkpoint_path"]
assert checkpoint_path.exists(), f"missing: {checkpoint_path}"
print(f"champion checkpoint: {checkpoint_path.relative_to(ROOT)}")

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")
model = load_adapter(checkpoint_path=checkpoint_path, device=DEVICE)
print(f"adapter loaded on {DEVICE}")

adapted = np.empty_like(img_emb)
for i in range(0, n_items, BATCH):
    j = min(i + BATCH, n_items)
    adapted[i:j] = adapt_image(model, img_emb[i:j])

np.save(OUT_PATH, adapted)
MANIFEST_PATH.write_text(json.dumps({
    "snapshot_id": SNAPSHOT_ID,
    "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    "n_items": n_items,
    "n_with_usable_image": int((img_status == 1).sum()),
}, indent=1))

print(f"\nwritten -> {OUT_PATH.relative_to(ROOT)} "
      f"({OUT_PATH.stat().st_size / 1e6:,.1f} MB)")
print(f"written -> {MANIFEST_PATH.relative_to(ROOT)}")
print(f"total runtime: {time.time() - t0:.1f}s")
