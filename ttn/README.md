# Two-Tower Model — complementary products

The goal of this folder is to build a **two-tower neural network** in
**PyTorch** that finds **complementary products** — given an item, retrieve the
items bought *alongside* it rather than the items most similar to it. A phone
case complements a phone; another phone does not.

It is trained on the Home & Kitchen review data together with the **extracted
item features** produced by the `feature_extraction_workflow/` and
`embedding_analysis/` modules, and on the co-purchase signals built by
`complementary_cats_pairs/`.

The model is implemented in
[`ttn_complementary.ipynb`](./ttn_complementary.ipynb).

## The paper this is based on

> **Suggest, complement, inspire: story of Two Tower recommendations at
> Allegro.com** — Aleksandra Osowska-Kurczab, Klaudia Nazarko, Mateusz Marzec,
> Lidia Wojciechowska, Eliška Kremeňová. *Proceedings of the Nineteenth ACM
> Conference on Recommender Systems (RecSys '25)*, Prague, 22–26 September 2025.
> <https://arxiv.org/html/2508.03702v1>

The paper describes one two-tower architecture serving several recommendation
placements at Allegro, with a **Complementary-TT** variant that is the design
this folder follows. The parts that matter here:

- **Both towers are the same Product Encoder.** There is no user tower. Each
  item attribute — the paper uses title, price and category with a hierarchical
  taxonomy — goes through its own embedding table; the vectors are
  concatenated, passed through an MLP, and L2-normalised.
- **Only the query tower is modified for the complementary task.** "The
  modifications are implemented in the query tower, with the target tower
  remaining unchanged": the query product embedding is concatenated with a
  **target category embedding** and that becomes the query representation.
- **The target category comes from a complementary-category mapping**, one-to-
  many, "derived from the statistical models fit on co-purchase data, external
  annotations and domain knowledge".
- **Training pairs are co-occurring product pairs** above a minimum
  co-occurrence threshold, filtered "to examples that follow a complementarity
  relation heuristic".
- **Loss** is a sampled softmax with mixed negative sampling; the complementary
  variant adds a **target category reconstruction error** "to enforce the
  correctness of the target category embedding".

Two things the paper leaves open, so they are decisions made in this repo
rather than facts taken from it: which statistical measures build the
complementary-category mapping, and the formulation of the reconstruction
error. This project uses **support and lift** over Amazon's `also_buy` edges
for the first — see `complementary_cats_pairs/README.md`.

## How the paper's pieces map onto this repo

| Paper | Here |
| --- | --- |
| complementary-category mapping (one-to-many, from co-purchase data) | `data/complementary_categories.pkl`, built by `complementary_cats_pairs/categories.ipynb`, scored by support and lift |
| co-occurring product pairs above a minimum threshold | `data/co_purchase_pairs.pkl`, built by `complementary_cats_pairs/pairs.ipynb` |
| product-encoder input attributes | `data/df_features.pkl` / `df_features_with_embeddings.pkl` — `title_cleaned`, `cat_*`, `brand`, and the extracted attributes |
| train/test split point | `date_threshold` in [`constants.json`](./constants.json), shared with `pairs.ipynb` so both read one value |

## What a two-tower model is

A two-tower (a.k.a. dual-encoder) model learns two separate neural
encoders that map their inputs into the **same embedding space**, so that
relevance is just the dot product (or cosine similarity) of their vectors.

In the classic retrieval setup one tower encodes users and the other items. **In
the complementary setup both towers encode items**: the query tower takes the
item being looked at, the candidate tower takes a possible complement, and the
score says how well the second completes the first.

```
   query ITEM features                   candidate ITEM features
        │                                          │
   ┌────▼────┐                                ┌────▼────┐
   │  QUERY  │  product encoder               │CANDIDATE│  product encoder
   │  TOWER  │  (+ target category)           │  TOWER  │
   └────┬────┘                                └────┬────┘
        │                                          │
   q ∈ ℝ^d  ──────────  score = q · c  ──────────  c ∈ ℝ^d
```

- Each tower is an embedding table per attribute, concatenated and followed
  by a small MLP that projects to a shared `d`-dimensional space (e.g. 64 or
  128), then L2-normalised.
- The two towers share the same *architecture* (the paper's Product Encoder)
  but are independent — they share **no weights** — and their outputs live in
  the same space so they can be compared directly. Only the query tower gets
  the target-category embedding concatenated in.
- At serving time the item tower can be run **offline** over the whole
  catalog and the vectors stored in an ANN index (FAISS, etc.). Retrieval
  then becomes a nearest-neighbor lookup against the user vector, which is
  what makes this architecture scale to large catalogs.

### Training objective

The paper trains with a **sampled softmax** loss and a mixed negative sampling
strategy, and the complementary variant adds a **target category reconstruction
error** to keep the target-category embedding honest.

Because this branch is `add-bpr`, the loss actually used here is **BPR
(Bayesian Personalized Ranking)** — the same pairwise ranking idea, over the
same positives. Note the operands change from the classic setup: the anchor is
a **query item**, not a user.

For a query item `q`, a positive candidate `c⁺` (one actually co-purchased with
`q`) and a negative candidate `c⁻` (sampled, not co-purchased with it):

```
L = -log σ( score(q, c⁺) - score(q, c⁻) )
```

This pushes items genuinely bought alongside `q` above sampled ones, which is
the retrieval behaviour wanted. Swapping in sampled softmax later changes the
loss, not the data or the towers.

## How the data is used

Three sources in `data/`, and the first is what makes this a *complementary*
model rather than a similarity one:

1. **Co-purchase pairs** — `data/co_purchase_pairs.pkl`, built by
   `complementary_cats_pairs/pairs.ipynb`. Every unordered pair of items the
   same user bought within `window_days` of each other, using only
   interactions before `date_threshold`. These pairs are the **positive
   examples**: `(query item, candidate item)`. Other items are candidate
   **negatives** to sample from. The window matters — two purchases four years
   apart are not one shopping intent, and without it every item a long-lived
   account ever bought would count as a complement of every other.

2. **The complementary-category mapping** — `data/complementary_categories.pkl`,
   built by `complementary_cats_pairs/categories.ipynb` from Amazon's
   `also_buy` edges, scored by support and lift. One-to-many: source category
   path → the target category paths worth recommending alongside it. This is
   the paper's mapping, and it supplies the **target category** whose embedding
   is concatenated into the query tower.

3. **Item features with embeddings** — `data/df_features_with_embeddings.pkl`
   (produced by `embedding_analysis/`). One row per `asin`, including:
   - `title_cleaned` and a 384-dim `title_embedding` (SBERT
     `all-MiniLM-L6-v2`, already unit-norm),
   - categorical fields: `cat_2`, `cat_3`, `cat_4`, `brand`,
     `Product_Type`, `Material`, `Color`,
   - `extracted_features` (the raw LLM-extracted attribute dict).

   These are the product-encoder inputs, and **both towers read them**.

The raw interaction log is still needed, but as the thing co-purchase pairs are
derived *from* rather than as `(user, item)` positives.

### Feeding the towers

Both towers consume item information — there is no user tower. They share the
same architecture and no weights.

- **Candidate tower** is a plain product encoder:
  - the precomputed `title_embedding` as a dense input,
  - categorical features (`cat_*`, `brand`, `Product_Type`, `Material`,
    `Color`) each through their own embedding table,
  - concatenate → MLP → L2-normalise → `d`-dim candidate vector.

  Using the extracted features gives the model **content signal**, so it can
  generalise to cold / sparsely-reviewed items instead of relying on an
  item-ID embedding alone — which matters more here than in the user/item
  setup, because a query item may be one the model has never seen paired with
  anything.

- **Query tower** is the same product encoder over the *query* item, plus the
  one modification the paper makes: the query product embedding is
  concatenated with the **target category embedding** looked up from the
  complementary-category mapping, and that concatenation is the final query
  representation.

Because both sides are items, the user features assembled in the notebook's §4
are not tower inputs. They are kept because the same table backs the
user/item variant, and because a user's history is what defines which pairs
count as co-purchased at all.

### Suggested flow in `ttn_complementary.ipynb`

1. Load `df_features_with_embeddings.pkl`, `co_purchase_pairs.pkl` and
   `complementary_categories.pkl`.
2. Build index maps for items, each categorical feature, and the category
   paths used as targets.
3. Create a `Dataset` that yields `(query_item, pos_candidate, neg_candidate)`
   triples, sampling negatives per query.
4. Define a `ProductEncoder`, then `QueryTower` (encoder + target-category
   embedding) and `CandidateTower` (encoder alone), and a `TwoTower` module
   scoring a pair by dot product.
5. Train with the loss above; evaluate with ranking metrics (Recall@K,
   NDCG@K) on co-purchase pairs held out after `date_threshold`.
6. Export candidate vectors for retrieval.

## Dependencies

- `torch`
- `pandas`, `numpy`, `scipy`
- (optional) `faiss` for nearest-neighbor retrieval at serving time
