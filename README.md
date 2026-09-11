# RecSystem — TTN model

This branch (`ttn_model`) carries the **two-tower neural network** that
retrieves complementary products: given an item, it finds the items bought
*alongside* it (a phone case for a phone), not the items most similar to it.
It's one of two branches split off `add-bpr` for review as separate,
self-contained pieces of work -- the other, `similarity_model`, carries the
no-training cosine-similarity baselines this model is compared against.
Despite the branch name, the *original* BPR recommender (`functions/bpr.py`,
a plain user-item model) is not part of either branch; only the loss
function it left behind -- BPR pairwise ranking, reused for the two-tower
model below -- survived the branch that bears its name.

## What's here

**Data pipeline** (identical on both branches -- this is the shared
prerequisite both the TTN model and the similarity baselines read):

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
   `vocabs.json`, `node_of_item.npy`), fitting every statistic (brand folds,
   categorical vocabularies, category price medians, `target_node` vocabulary)
   on the train slice only and applying it unchanged to test.

**The model** (`ttn/ttn_complementary.ipynb`, sections 10 onward, and
`ttn/experiments/`):

- A two-tower (dual-encoder) model, both towers the same Product Encoder over
  item attributes (title, description, categoricals, numeric features).
  Only the query tower is modified: the query item's embedding is
  concatenated with a **target-category embedding**, looked up from the
  complementary-category mapping, before the final projection.
- Trained with a **BPR** pairwise loss (the branch's namesake objective,
  repurposed for this model): in-batch negatives plus `N_HARD` uniform
  in-node hard negatives per row, a temperature (`TAU`) on the cosine scores,
  and a logQ correction for negative-sampling frequency bias.
- Retrieval is evaluated **inside the pair's target category** (candidates
  restricted to the same `target_node_id`), never against the whole catalogue
  -- the query tower is handed the true target category, so the honest floor
  to beat is per-category popularity, not `10 / n_items`.
- `ttn/experiments/README_ttn_model.md` documents every architecture/training
  experiment that shaped the current design (negative sampling, temperature,
  logQ, hard negatives, image embeddings) and what each one found.

## Current model artifacts

- `data/tower/ttn_complementary.pt` -- the committed checkpoint, BPR loss,
  trained on the full ~2.2M-pair training set (30 epochs, early-stopped).
- `data/tower/ttn_complementary_image.pt` -- the same, plus a SigLIP2 image
  branch on the Product Encoder, trained separately so the original
  checkpoint stays untouched. R@10 (all buckets): 0.2180 → 0.2245 with image;
  helps most on frequently-seen targets, is flat-to-slightly-worse on
  never/rarely-seen ones.

Both are compared against the `similarity_model` branch's DESC / CONTENT /
POP baselines on the same held-out 20,000-pair test sample -- see that
branch's README for the numbers.

## Setup

### Clone repo
git clone https://github.com/flamesmith/RecSystem.git

### Dataset

Download dataset from Google Drive and place in:

data/

Do NOT commit datasets to GitHub.

### Install dependencies

pip install -r requirements.txt
