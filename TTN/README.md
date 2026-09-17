# TTN — Complete the Look

A two-tower neural network that retrieves **complementary** products: given
an item, the items bought *alongside* it (a phone case for a phone), not the
items most similar to it.

## Run

```
python TTN/build_data.py           # -> data/tower/ (items, pairs, vocabs)
python TTN/encode_descriptions.py  # optional, recommended -- see its docstring
python TTN/build_model.py          # -> data/tower/ttn_complementary.pt
```

`build_data.py` reads the shared data pipeline's output
(`feature_extraction_workflow/`, `embedding_analysis/`,
`complementary_cats_pairs/`, and the raw CSVs/JSONs in `data/`) and needs
`TTN/constants.json` (the train/test split date). `build_model.py` reads only
what `build_data.py` produced.

`SigLIP2/` and `Popularity/`'s own build steps both depend on
`build_data.py` having already run — they read `data/tower/items.npz` and
`item_asins.npy`.

## What it does

Both towers are the same product encoder (categorical attributes + title +
description, each through their own embedding, concatenated, MLP,
L2-normalised). Only the query tower differs: the query item's embedding is
concatenated with a **target-category embedding** before the final
projection, so the model retrieves *for a specific kind of complement*, not
just "similar items." Trained with a BPR pairwise loss over co-purchase
pairs (`data/co_purchase_pairs_{train,test}.pkl`).

See `results.ipynb` (repo root) for its evaluated Recall@10/@100.
