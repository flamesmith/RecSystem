# SigLIP2 — Visually Similar Products

Retrieves by **cosine similarity over frozen SigLIP2 image embeddings** —
not a trained model. Two items with visually similar product photos score
high regardless of whether anyone has ever bought them together: it answers
"what looks like this" (a substitute), not "what goes with this" (the
complement `../TTN/` retrieves).

## Run

```
# after data_processing/build_snapshot.py --window-days 90 has run at least once
python SigLIP2/encode_siglip2_images.py --snapshot w90_2017-12-09
```

Resumable — safe to interrupt and re-run; only touches items not yet
encoded. Defaults to a full run (`MAX_IMAGES = None` in the script); edit
that constant to a small number for a quick smoke test first.

Two-level storage, both under this folder, not `data/`: `SigLIP2/image_cache/`
is an **asin-keyed** cache shared across every snapshot (an item's photo
doesn't change just because a different `--window-days` snapshot orders
items differently, so it's never re-fetched once cached). Each run then
**projects** that cache onto the given snapshot's own item order, writing
`SigLIP2/generated/<snapshot_id>/siglip_img_emb.npy` (one 768-d vector per
item) and `siglip_img_status.npy` (flags items with no usable image — about
28% of the catalogue, left as a zero vector) — that per-snapshot pair is what
everything downstream actually reads.

Depends on `data_processing/build_snapshot.py` having already run for the given snapshot:
it reads that snapshot's `item_asins.npy` for the item order and list.

See `results.ipynb` (repo root) for its evaluated Recall@10/@100.

## Trained adapter — imported from the `SigLIP2` branch

Everything above uses the SigLIP2 backbone **frozen, as-is** — no training.
A more advanced approach was developed on the `SigLIP2` branch (package
`SigLIP2_training/`): lightweight **adapter** layers (image-side and
text-side) trained on top of frozen SigLIP2 features, rather than full
model fine-tuning. The trained checkpoint and a minimal loader for it are
now imported into this folder:

- `checkpoints/pilot_200k/` — the full deployment package, copied
  byte-for-byte from that branch (SHA-256 verified), same folder name and
  same files: `taxonomy_w20_random_200k.pt` (the trained weights),
  `results.json` (validation/test metrics and the promotion decision),
  `manifest.json` (run environment and lineage), and `README.md` (that
  branch's own provenance notes, unchanged).
- `adapter.py` — a small, self-contained loader (not the full training
  package): `load_adapter()` reconstructs the architecture from the
  checkpoint's own config and loads the weights; `adapt_image()` /
  `adapt_text()` apply it to raw frozen SigLIP2 embeddings.
- `description_pipeline.py` + `description_v1.yaml` — the exact
  deterministic text preprocessing the adapter was trained on, ported
  from that branch's `text.py`/`schema.py`/`config.py`: HTML stripping,
  sentence dedup, boilerplate removal, taxonomy text, structured
  attributes, and a scored "canonical description" (max 54 words).
- `text_embeddings.py` — generates SigLIP2 **text** embeddings, which
  nothing in this repo did before (`encode_siglip2_images.py` only ever calls
  `get_image_features()`; this is `get_text_features()`, the other half
  of the same dual encoder, same checkpoint).

Both the image and text paths are verified working end to end against
real data already in this repo: `adapt_image()` against the `w60`
snapshot's generated embeddings (median cosine to original 0.91, over
2,000 real items), and the full chain `raw catalogue fields →
description_pipeline → text_embeddings → adapt_text()` against a real
catalogue item from `df_features.pkl` (cosine to original 0.81). Both
confirm the adapter does real, bounded work — not a no-op, not
destructive to the original signal.

**What it is:**
- **Trained on both images and descriptions**, via the preprocessing
  above — not raw description text.
- **Real, measured improvement**: on held-out cross-modal retrieval,
  bidirectional Recall@10 went from 0.579 (frozen backbone alone) to 0.806
  (after adapter training).
- **Pilot scope**: trained on 200K products, not the full catalogue —
  "no full-catalogue image download has been started" per the source
  branch's own status notes.

**What's still missing, to go from "callable" to "actually serving recommendations":**
- **No precomputed catalogue index.** Nothing here has run every item's
  embedding through `adapt_image()`/`adapt_text()` and stored the result
  — that's the same batch-scoring step `SigLIP2/generate_recommendations.py`
  already does for the *un*adapted image embeddings; it would need a
  parallel version calling this adapter instead.
- A `transformers` version difference from when this checkpoint's ecosystem
  was built meant `get_text_features()` (and, in the existing
  `encode_siglip2_images.py`, `get_image_features()`) now returns a wrapped output
  object rather than a plain tensor — handled in both, via the same
  `getattr(out, "pooler_output", out)` fallback.
- The source branch's own inference notebook
  (`SigLIP2_training/notebooks/siglip2_200k_inference_and_api_flow.ipynb`)
  is the fuller reference for the multichunk text view and a sketch of an
  actual API layer, neither of which is imported here.
