# TTN — Complete the Look

A two-tower neural network that retrieves **complementary** products: given
an item, the items bought *alongside* it (a phone case for a phone), not the
items most similar to it.

## Run

```
python data_processing/build_snapshot.py --window-days 90             # SHARED, run once per snapshot
python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09  # TTN-specific arrays
python TTN/encode_descriptions.py --snapshot w90_2017-12-09   # optional, recommended
python TTN/build_model.py --snapshot w90_2017-12-09      # -> .../models/ttn/<date>_v_00x/
```

`--window-days` is the co-purchase pairing window (how many days apart two
purchases can still count as "bought together") -- see `build_snapshot.py`'s
docstring. Different values produce distinct, coexisting data snapshots
(`data/tower/w{window_days}_{date_threshold}/`) rather than overwriting
each other, so it's safe to try several without losing earlier ones.

**`data_processing/build_snapshot.py` (not this folder) is the shared step** —
`SigLIP2/` and `Popularity/` depend on it too, reading that snapshot's
`item_asins.npy` and `node_of_item.npy` directly. It also writes
`tower_pairs_{train,test}.parquet` (the cleaned, joined, wide pair tables) as
a TTN-only handoff. `data_processing/build_ttn_arrays.py` picks up from there and does only
what's actually TTN-specific: fitting the model's own embedding-table
vocabularies and encoding `items.npz` / `vocabs.json` /
`pairs_{train,test}.parquet` (the slim query_idx/target_idx form used for
training). `build_model.py` reads only what `build_ttn_arrays.py` produced for
that same snapshot, and saves a new **version** each run
(`<date>_v_00x/model.pt` + `version_manifest.json`) rather than overwriting
the last one -- a candidate version isn't the champion until something
explicitly promotes it.

## What it does

Both towers are the same product encoder (categorical attributes + title +
description, each through their own embedding, concatenated, MLP,
L2-normalised). Only the query tower differs: the query item's embedding is
concatenated with a **target-category embedding** before the final
projection, so the model retrieves *for a specific kind of complement*, not
just "similar items." Trained with a BPR pairwise loss over co-purchase
pairs (`data/co_purchase_pairs_{train,test}.pkl`).

See `results.ipynb` (repo root) for its evaluated Recall@10/@100.
