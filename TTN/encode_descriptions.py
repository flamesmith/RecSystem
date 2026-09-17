"""Encode `description_cleaned` for the tower's items, in a lean process.

Run AFTER §9 of ttn_complementary.ipynb has written data/tower/item_asins.npy:

    python ttn/encode_descriptions.py

Doing this inside §9 holds df_features (2.9 GB) and the embeddings pickle
(4.3 GB) in memory alongside SBERT; on 16 GB the batches stall for minutes at a
time. This process loads two columns and nothing else.

Writes data/tower/desc_emb.npy, aligned row-for-row with items.npz.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tower"
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
