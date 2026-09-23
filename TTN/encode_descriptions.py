"""Encode `description_cleaned` for the tower's items, in a lean process.

Run AFTER data_processing/build_ttn_arrays.py has written
data/tower/<snapshot_id>/items.npz, and BEFORE build_model.py trains the
model -- build_model.py detects whether desc_emb.npy exists in that same
snapshot and trains without the description block if it doesn't, so running
this in between is what gives the model description signal at all:

    python data_processing/build_snapshot.py --window-days 90
    python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
    python TTN/encode_descriptions.py --snapshot w90_2017-12-09
    python TTN/build_model.py --snapshot w90_2017-12-09   # now finds desc_emb.npy

Doing this inside build_ttn_arrays.py holds df_features (2.9 GB) and the
embeddings pickle (4.3 GB) in memory alongside SBERT; on 16 GB the batches
stall for minutes at a time. This process loads two columns and nothing else.

Writes data/tower/<snapshot_id>/desc_emb.npy, aligned row-for-row with that
snapshot's items.npz. Re-encodes per snapshot even though the description
text itself doesn't change with window_days -- only item coverage/order can
differ between snapshots, and this hasn't been made snapshot-independent
(asin-keyed) the way SigLIP2's cache is. Known inefficiency, not a bug.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", required=True,
                     help="snapshot_id from data_processing/build_ttn_arrays.py, e.g. w90_2017-12-09")
OUT = ROOT / "data" / "tower" / parser.parse_args().snapshot
MAX_CHARS = 1200          # MiniLM caps at 256 word-pieces; the tokeniser still
                          # reads everything past it, and a few 50 kB blurbs
                          # otherwise dominate the batch time
BATCH = 64

asins = np.load(OUT / "item_asins.npy", allow_pickle=False)
print(f"items: {len(asins):,}")

desc = pd.read_pickle(ROOT / "data" / "df_features.pkl")[["asin", "description_cleaned"]]
desc = (desc.drop_duplicates("asin").set_index("asin")["description_cleaned"]
        .reindex(pd.Index(asins)).fillna("").astype(str))
blank = (desc.str.strip() == "").to_numpy()
dup = float(desc[~blank].duplicated(keep=False).mean()) if (~blank).any() else 0.0
print(f"non-empty: {int((~blank).sum()):,}   blank: {int(blank.sum()):,}")
print(f"share sharing their text with another item: {dup:.1%}")
print("  (boilerplate cannot separate items — read that as a ceiling on what this adds)")
print(f"median length: {int(desc.str.len().median()):,} chars, "
      f"p95 {int(desc.str.len().quantile(0.95)):,}, truncating at {MAX_CHARS:,}")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
out = np.asarray(model.encode(desc.str.slice(0, MAX_CHARS).tolist(),
                              batch_size=BATCH, show_progress_bar=True)).astype("float32")
out[blank] = 0.0
np.save(OUT / "desc_emb.npy", out)
print(f"\nwrote {OUT / 'desc_emb.npy'}  {out.shape}")
