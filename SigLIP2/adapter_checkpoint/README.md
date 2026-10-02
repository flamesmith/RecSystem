# Promoted SigLIP2 adapter — imported from the `SigLIP2` branch

`taxonomy_w20_random_200k.pt` is copied, byte-for-byte, from that branch's
`SigLIP2_training/checkpoints/pilot_200k/taxonomy_w20_random_200k.pt`.

- **Source branch**: `SigLIP2`
- **Source commit**: `43327cad712038579d06ac53996f3c8133b953e9` ("Add promoted 200K SigLIP2 adapter and inference demo")
- **SHA-256** (verified identical to the source branch's copy):
  `bbbedcfa9c7d8bdb74552e087dca85c2d031f133b217914ffc8fac20dbcc34d8`

## What's in the file

A dict: `candidate`, `strategy`, `state_dict` (the trained weights — two
small residual adapters, image and text, plus a logit scale/bias),
`dimension` (768), `training_config` (the exact hyperparameters used —
bottleneck 128, dropout 0.05, 30 epochs max, patience 6), and
`feature_index` (provenance of the 200K-product feature shards it was
trained against — not included here, only its metadata).

## How to load and call it

See `../adapter.py` — `load_adapter()` reconstructs the architecture from
this checkpoint's own `training_config` and loads the weights;
`adapt_image()`/`adapt_text()` apply it to raw frozen SigLIP2 embeddings.

## What this is *not*

- Not a standalone SigLIP2 model — it only adapts embeddings that already
  came from `google/siglip2-base-patch16-224`'s own image/text towers.
- Not trained on this repo's full catalogue — the source branch's own
  notes say "no full-catalogue image download has been started"; this was
  trained on a 200K-product pilot sample.
- Not accompanied by a precomputed catalogue index — nothing here scores
  your actual catalogue yet, it just makes the adapter callable. Building
  a real index means running every catalogue item's frozen SigLIP2
  embedding through `adapt_image()` (and, for text, through
  `adapt_text()` once something produces SigLIP2 text embeddings) and
  storing the results — not done by this import.
