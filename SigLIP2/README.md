# SigLIP2 — Visually Similar Products

Retrieves by **cosine similarity over frozen SigLIP2 image embeddings** —
not a trained model. Two items with visually similar product photos score
high regardless of whether anyone has ever bought them together: it answers
"what looks like this" (a substitute), not "what goes with this" (the
complement `../TTN/` retrieves).

## Run

```
# after TTN/build_data.py has run at least once
python SigLIP2/build_siglip2.py
```

Resumable — safe to interrupt and re-run; only touches items not yet
encoded. Defaults to a full run (`MAX_IMAGES = None` in the script); edit
that constant to a small number for a quick smoke test first.

Produces `data/tower/siglip_img_emb.npy` (one 768-d vector per item) and
`data/tower/siglip_img_status.npy` (flags items with no usable image — about
28% of the catalogue, left as a zero vector).

Depends on `TTN/build_data.py` having already run: it reads
`data/tower/item_asins.npy` for the item order and list.

See `results.ipynb` (repo root) for its evaluated Recall@10/@100.
