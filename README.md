# RecSystem — similarity model

This branch (`similarity_model`) carries the **no-training cosine-similarity
baselines** used to retrieve complementary products for items the two-tower
model (`ttn_model` branch) handles badly: items whose target category has
few or no training co-purchases. It's one of two branches split off
`add-bpr` for review as separate, self-contained pieces of work -- the other,
`ttn_model`, carries the trained two-tower model these baselines are compared
against.

## What's here

**Data pipeline** (identical on both branches -- this is the shared
prerequisite both the similarity baselines and the TTN model read):

1. `feature_extraction_workflow/` -- turns raw item title/description/feature
   text into a structured `extracted_features` dict per item
   (`data/df_features.pkl`).
2. `embedding_analysis/` -- embeds `title_cleaned` with SBERT
   (`data/df_features_with_embeddings.pkl`).
3. `complementary_cats_pairs/` -- two independent halves: `categories.*`
   builds the complementary-category mapping from Amazon's `also_buy` edges
   (`data/complementary_categories.pkl`); `pairs.*` builds the co-purchase
   item pairs from the review log (`data/co_purchase_pairs_{train,test}.pkl`).
4. `ttn/encode_descriptions.py` -- frozen SBERT description embeddings.
5. `ttn/SigLip embeddings.ipynb` -- frozen SigLIP2 product-image embeddings.
6. `ttn/ttn_complementary.ipynb`, sections 1–9 -- joins everything into the
   model-ready arrays in `data/tower/` (`items.npz`, `pairs_{train,test}.parquet`,
   `vocabs.json`, `node_of_item.npy`) that every baseline below scores
   against.

**The baselines** (`ttn/ttn_complementary.ipynb` §18–20, and
`ttn/experiments/`, documented in full in
`ttn/experiments/README_similarity_model.md`):

- **DESC** -- SBERT cosine over the frozen description embedding alone.
- **CONTENT** -- mean cosine over whichever of {description, title, features}
  is present on both the query and candidate item (equal weight,
  renormalised when a term is dropped), optionally with a fourth component,
  SigLIP2 image cosine, at 0.25 weight each. (The notebook's own §19 defines
  CONTENT as description + colour + material instead; this branch's scripts
  use description + title + features, which scored higher in every bucket on
  the same test sample -- see the experiments README for both numbers.)
- **POP** -- fixed per-category top-10/top-100 by training-set frequency, no
  query read at all. The honest no-model floor, since the query tower is
  handed the true target category, so category popularity (not the whole
  catalogue) is what any model has to beat.

All three are scored **inside the pair's target category** (candidates
restricted to `node_of_item == target_node_id`, self excluded), by top-k
**membership** (`target in topk(scores)`) -- never `(scores > true_score).sum()`,
which is the rule that inflated an earlier description-similarity number ~3x
by crediting every zero-vector tie as a hit.

## Headline result: CONTENT wins cold, TTN and POP win warm

On a 20,000-pair held-out test sample (seed 0), broken out by how often the
target was seen in training:

| target freq | pairs | TTN | POP | CONTENT (desc+title+feat) | CONTENT+image |
| --- | --- | --- | --- | --- | --- |
| never | 668 | 0.0389 | 0.0000 | 0.0838 | 0.0823 |
| 1–5 | 2,176 | 0.0133 | 0.0051 | 0.0758 | 0.0864 |
| 6–25 | 4,595 | 0.0276 | 0.0324 | 0.0805 | 0.0899 |
| 26–100 | 5,722 | 0.1075 | 0.1003 | 0.0736 | 0.0869 |
| >100 | 6,839 | 0.5210 | 0.5207 | 0.0534 | 0.0580 |
| all | 20,000 | 0.2180 | 0.2147 | 0.0688 | 0.0775 |

(Recall@10; see `ttn/experiments/README_similarity_model.md` for Recall@100
and the image-augmented TTN comparison.) CONTENT is the only signal that
scores at all on `never`-frequency targets (POP is 0 by construction; TTN has
nothing to learn from). Above ~25 observations, TTN and popularity dominate
and CONTENT flattens out. The practical implication (§19 of the notebook):
route by frequency at serving time -- build the candidate pool from CONTENT
below ~25 observations and from TTN/popularity above it. A score-level blend
(reciprocal rank fusion of TTN and DESC) was tried and scored *worse* than
either alone, because RRF compares by rank position and a rare item's #1 ties
a popular item's #1.

## Setup

### Clone repo
git clone https://github.com/flamesmith/RecSystem.git

### Dataset

Download dataset from Google Drive and place in:

data/

Do NOT commit datasets to GitHub.

### Install dependencies

pip install -r requirements.txt
