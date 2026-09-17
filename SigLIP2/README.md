# SigLIP2 — visually similar products

**This is the "Visually Similar Products" carousel of the `v1-recommendations`
structure.** The complementary carousel ("Complete the Look") lives in
[`../TTN/`](../TTN/), the popularity fallback in
[`../Popularity/`](../Popularity/); cross-model comparisons that measure this
signal against those two live in [`../analysis/`](../analysis/).

Unlike TTN, this is not a trained model — it retrieves by **cosine similarity
over frozen SigLIP2 image embeddings**, one vector per product image. Two
items with visually similar product photos score high regardless of whether
anyone has ever bought them together, which is the point: it answers "what
looks like this" (a substitute), not "what goes with this" (a complement) —
the same phone-case-vs-another-phone distinction `../TTN/README.md` draws,
from the other side.

## What's here

[`SigLip embeddings.ipynb`](./SigLip%20embeddings.ipynb) builds the
embeddings: one SigLIP2 vector per product image, saved to
`data/tower/siglip_img_emb.npy` (`data/tower/siglip_img_status.npy` flags
which items have no usable image — about 28.5% of the catalogue, per this
session's measurements). That `.npy` file is what every other SigLIP2-reading
script in this repo, including `../TTN/`'s image-augmented model and
`../analysis/`'s comparisons, actually loads.

## Where the retrieval itself is measured

No script in this folder scores "visually similar" retrieval in isolation —
every measurement of this signal in this repo scores it *against* TTN,
CONTENT (a blend of description + title + features + image), or popularity,
because that comparison is what answered the questions this project actually
asked (does image similarity help the cold-start problem; does it help TTN
directly). See:

- `../analysis/image_content_similarity.py` — the `IMAGE` column is pure
  SigLIP2 cosine similarity, this carousel's actual retrieval logic, reported
  next to `DESC`, `CONTENT`, `TTN`, and `POP` for comparison. Headline: image
  similarity alone beats description-only similarity in every bucket.
- `../TTN/experiments/image_in_model.py` and `image_augmented_ttn.py` — add a
  SigLIP2 branch to the TTN *model* itself (a different question: does image
  signal help the complementary model, not "is image similarity itself a
  good substitute retriever").
- `../analysis/frequency_routed_retrieval*.py` — test blending this signal's
  cold-start strength with TTN's warm-target strength.

## Dependencies

- `data/tower/siglip_img_emb.npy`, `data/tower/siglip_img_status.npy` — built
  by the notebook above; not tracked in git (`data/tower/` is gitignored).
- The item catalogue, in the same `data/tower/items.npz` / `item_asins.npy`
  form that `../TTN/ttn_complementary.ipynb` §9 exports, so an item's row
  index means the same thing in both places.
