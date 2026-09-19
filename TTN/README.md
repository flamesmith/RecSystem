# TTN — Complete the Look

A two-tower neural network that retrieves **complementary** products: given
an item, the items bought *alongside* it (a phone case for a phone), not the
items most similar to it.

## Run

```
python TTN/build_data.py --window-days 90               # -> data/tower/w90_2017-12-09/
python TTN/encode_descriptions.py --snapshot w90_2017-12-09   # optional, recommended
python TTN/build_model.py --snapshot w90_2017-12-09      # -> .../models/ttn/<date>_v_00x/
```

`--window-days` is the co-purchase pairing window (how many days apart two
purchases can still count as "bought together") -- see `build_data.py`'s
docstring. Different values produce distinct, coexisting data snapshots
(`data/tower/w{window_days}_{date_threshold}/`) rather than overwriting
each other, so it's safe to try several without losing earlier ones.

`build_data.py` reads the shared data pipeline's output
(`feature_extraction_workflow/`, `embedding_analysis/`,
`complementary_cats_pairs/`, and the raw CSVs/JSONs in `data/`) and needs
`TTN/constants.json` (the train/test split date). `build_model.py` reads only
what `build_data.py` produced for that same snapshot, and saves a new
**version** each run (`<date>_v_00x/model.pt` + `version_manifest.json`)
rather than overwriting the last one -- a candidate version isn't the
champion until something explicitly promotes it.

`SigLIP2/` and `Popularity/`'s own build steps both depend on
`build_data.py` having already run for the snapshot they're given — they
read that snapshot's `items.npz` and `item_asins.npy`.

## What it does

Both towers are the same product encoder (categorical attributes + title +
description, each through their own embedding, concatenated, MLP,
L2-normalised). Only the query tower differs: the query item's embedding is
concatenated with a **target-category embedding** before the final
projection, so the model retrieves *for a specific kind of complement*, not
just "similar items." Trained with a BPR pairwise loss over co-purchase
pairs (`data/co_purchase_pairs_{train,test}.pkl`).

See `results.ipynb` (repo root) for its evaluated Recall@10/@100.
