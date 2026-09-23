# SigLIP2 — Visually Similar Products

Retrieves by **cosine similarity over frozen SigLIP2 image embeddings** —
not a trained model. Two items with visually similar product photos score
high regardless of whether anyone has ever bought them together: it answers
"what looks like this" (a substitute), not "what goes with this" (the
complement `../TTN/` retrieves).

## Run

```
# after data_processing/build_snapshot.py --window-days 90 has run at least once
python SigLIP2/build_siglip2.py --snapshot w90_2017-12-09
```

Resumable — safe to interrupt and re-run; only touches items not yet
encoded. Defaults to a full run (`MAX_IMAGES = None` in the script); edit
that constant to a small number for a quick smoke test first.

Two-level storage: `data/tower/_siglip_cache/` is an **asin-keyed** cache
shared across every snapshot (an item's photo doesn't change just because a
different `--window-days` snapshot orders items differently, so it's never
re-fetched once cached). Each run then **projects** that cache onto the
given snapshot's own item order, writing
`data/tower/<snapshot_id>/siglip_img_emb.npy` (one 768-d vector per item)
and `siglip_img_status.npy` (flags items with no usable image — about 28%
of the catalogue, left as a zero vector) — that per-snapshot pair is what
everything downstream actually reads.

Depends on `data_processing/build_snapshot.py` having already run for the given snapshot:
it reads that snapshot's `item_asins.npy` for the item order and list.

See `results.ipynb` (repo root) for its evaluated Recall@10/@100.
