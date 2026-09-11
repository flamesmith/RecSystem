# Experiments behind the similarity baselines

No model is trained by anything in this folder on the `similarity_model`
branch -- every script here scores a retrieval candidate by cosine similarity
over frozen embeddings (SBERT text, SigLIP2 image) or by training-set
popularity, and compares the result against the committed TTN checkpoint
(`data/tower/ttn_complementary.pt`, read-only, for reference -- never
retrained here). The model-training counterpart of this work lives on the
`ttn_model` branch. `final_capacity_comparison_similarity.py` and
`image_augmented_similarity.py` are each one half of a script that used to
also train TTN (`final_capacity_comparison.py`, `image_augmented_models.py`);
the other half is over there. Neither split script has been re-run yet, so
there's no log of its own -- `final_capacity_comparison.log` and
`image_augmented_models.log` are the original combined runs (same
computation, same numbers, just also carrying the TTN columns this branch
dropped).

| file | question it answers | headline |
| --- | --- | --- |
| `image_content_similarity.py` | does adding SigLIP image similarity to CONTENT (desc+colour+material) help? | yes, in every bucket; IMAGE alone also beats DESC alone |
| `reproduce_section19.py` | does the notebook's §19 table (TTN / DESC / POP / CONTENT by target frequency) still reproduce on current data? | TTN/POP/DESC close; CONTENT (desc+colour+material) ~30% lower than originally reported |
| `content_desc_title_features.py` | redefine CONTENT as desc+title+features instead of desc+colour+material -- does it do better? | yes, beats desc+colour+material in every bucket (R@10 all: 0.0688 vs 0.0559) |
| `feature_variance.py` | do item features discriminate *within* a category? | yes: title cosine 0.390 within a category vs 0.187 across -- the reason title/features are useful CONTENT components |
| `final_capacity_comparison_similarity.py` | §13's never-executed "~20k pairs" test, similarity half -- DESC+TITLE+FEAT(+IMG) and POP, matched to the 20k-pair training-sample regime | see `final_capacity_comparison.log` (combined run) |
| `image_augmented_similarity.py` | add image to CONTENT (desc+title+features+image, 0.25 weight each) -- does it help further? | yes, in every bucket (R@10 all: 0.0775 vs 0.0688); see `image_augmented_models.log` (combined run) |

## What CONTENT means, and why it changed twice

The notebook's own §19 defines `CONTENT` as description + colour + material.
This session redefined it as **description + title + features** instead
(`content_desc_title_features.py`), which scored higher in every bucket, then
added a fourth, optional component -- SigLIP2 image similarity
(`image_augmented_similarity.py`) -- for a further gain everywhere except the
`never`-frequency bucket at R@100, where it's flat. All four scripts share
one scoring discipline, carried over unchanged from the notebook's §19
writeup:

- Every ranker is scored **inside the pair's target category** (candidates
  restricted to `node_of_item == target_node_id`, self excluded).
- Hit@k is **top-k membership** (`target in topk(scores)`), never
  `(scores > true_score).sum()` -- that rank-based rule is what inflated the
  original description-similarity number ~3x, by giving a free hit@10/@100 to
  every zero-vector-vs-zero-vector tie (roughly 15% of test queries have no
  description text).
- Recall is reported **by the target's training frequency** (`never`, `1-5`,
  `6-25`, `26-100`, `>100`), because a cold target with a real description
  needs a completely different signal than a target the model has already
  seen 200 times -- CONTENT wins below ~25 observations, TTN and popularity
  dominate above it, and the "which signal, when" call this data supports is
  routing by frequency at serving time, not blending scores (an RRF blend of
  TTN and DESC scored *below* either alone -- see the notebook's §19).
