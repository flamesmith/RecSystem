# Two-Tower Model (TTN)

The goal of this folder is to build a **two-tower neural network** in
**PyTorch** for recommendation, trained on the Home & Kitchen review data
together with the **extracted item features** produced by the
`feature_extraction_workflow/` and `embedding_analysis/` modules.

The model is implemented in [`ttn.ipynb`](./ttn.ipynb).

## What a two-tower model is

A two-tower (a.k.a. dual-encoder) model learns two separate neural
encoders that map users and items into the **same embedding space**, so
that the relevance of an item to a user is just the dot product (or cosine
similarity) of their vectors.

```
   user features                         item features
        │                                     │
   ┌────▼────┐                           ┌────▼────┐
   │  USER   │   (MLP / embedding)       │  ITEM   │   (MLP / embedding)
   │  TOWER  │                           │  TOWER  │
   └────┬────┘                           └────┬────┘
        │                                     │
   u ∈ ℝ^d  ──────────  score = u · v  ──────  v ∈ ℝ^d
```

- Each tower is typically an embedding layer (for IDs) followed by a small
  MLP that projects to a shared `d`-dimensional space (e.g. 64 or 128).
- The two towers are independent — they share **no weights** — but their
  outputs live in the same space so they can be compared directly.
- At serving time the item tower can be run **offline** over the whole
  catalog and the vectors stored in an ANN index (FAISS, etc.). Retrieval
  then becomes a nearest-neighbor lookup against the user vector, which is
  what makes this architecture scale to large catalogs.

### Training objective

Because this branch is `add-bpr`, the natural loss is **BPR (Bayesian
Personalized Ranking)** / sampled softmax style pairwise ranking:

For a user `u`, a positive item `i` (one they interacted with) and a
negative item `j` (sampled, not interacted with):

```
L = -log σ( score(u, i) - score(u, j) )
```

This pushes the score of items the user actually reviewed above randomly
sampled items, which is exactly the retrieval behavior we want.

## How the data is used

There are two complementary data sources in `data/`:

1. **Interactions** — the review records (`reviewerID`, `asin`,
   `unixReviewTime`). `data/interaction_matrix.py`
   (`InteractionMatrixBuilder`) already filters to items with at least
   `min_users` distinct reviewers and builds a sparse user × item matrix.
   These (user, item) pairs are the **positive examples**; everything else
   is a candidate **negative** to sample from.

2. **Item features with embeddings** — `data/df_features_with_embeddings.pkl`
   (produced by `embedding_analysis/`). One row per `asin`, including:
   - `title_cleaned` and a 384-dim `title_embedding` (SBERT
     `all-MiniLM-L6-v2`, already unit-norm),
   - categorical fields: `cat_2`, `cat_3`, `cat_4`, `brand`,
     `Product_Type`, `Material`, `Color`,
   - `extracted_features` (the raw LLM-extracted attribute dict).

### Feeding the towers

- **Item tower** consumes the rich item features:
  - the precomputed `title_embedding` as a dense input,
  - categorical features (`cat_*`, `brand`, `Product_Type`, `Material`,
    `Color`) each through their own embedding table,
  - concatenate → MLP → `d`-dim item vector.

  Using the extracted features here gives the model **content signal**, so
  it can generalize to cold / sparsely-reviewed items instead of relying on
  the item-ID embedding alone.

- **User tower** — the simplest version is a learned embedding per
  `reviewerID`. A stronger version represents a user by **aggregating the
  item features of the products they reviewed** (e.g. mean of the item
  vectors / feature embeddings of their history), which lets the user tower
  also benefit from the extracted features and handle new users.

### Suggested flow in `ttn.ipynb`

1. Load `df_features_with_embeddings.pkl` and build the interaction matrix.
2. Build index maps for users, items, and each categorical feature.
3. Create a `Dataset` that yields `(user, pos_item, neg_item)` triples
   (BPR-style negative sampling).
4. Define `UserTower`, `ItemTower`, and a `TwoTower` module that scores a
   (user, item) pair by dot product.
5. Train with the BPR loss above; evaluate with ranking metrics
   (Recall@K, NDCG@K) on a held-out set of interactions.
6. Export item vectors for retrieval.

## Dependencies

- `torch`
- `pandas`, `numpy`, `scipy`
- (optional) `faiss` for nearest-neighbor retrieval at serving time
