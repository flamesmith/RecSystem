# Orchestration — Run Order

Which files to run, in order, to go from the two raw files to all three
trained/served recommenders. For *what* each step does, see the linked
`DATA_PIPELINE.md` in each folder — this file is just the sequence.

## 0. Prerequisites (already on disk, once)

- `data/meta_Home_and_Kitchen_filtered.csv`, `data/Home_and_Kitchen_filtered.csv` — raw, gitignored, must exist locally
- `data/master_metadata.json`, `data/global_filters.json`, `data/category_taxonomy.json` — tracked config
- `TTN/constants.json` — `date_threshold`

## 1. Shared data creation — run once, reused by everything after it

```bash
python data_creation/build_data.py
```
→ `data/df_features.pkl`, `df_features_with_embeddings.pkl`, `pair_stats.pkl`, `complementary_categories.pkl`
Details: [`TTN/DATA_PIPELINE.md`](TTN/DATA_PIPELINE.md) (Stage 1)

## 2. Shared snapshot — once per `--window-days` you want to try

```bash
python data_processing/build_snapshot.py --window-days 90
```
→ `data/tower/w90_<date_threshold>/{item_asins.npy, node_of_item.npy, tower_pairs_train.parquet, tower_pairs_test.parquet}`

Note the resulting **snapshot id** (`w90_2017-12-09`-style) — every command
below takes it as `--snapshot`. Everything from here on can run in **any
order** — TTN, SigLIP2, and Popularity are independent branches that only
share step 1 and 2's output.

## 3a. TTN — "Complete the Look" (complementary)

```bash
python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
python TTN/encode_descriptions.py --snapshot w90_2017-12-09   # optional, recommended
python TTN/build_model.py --snapshot w90_2017-12-09
```
→ trained checkpoint at `data/tower/w90_.../models/ttn/<date>_v_00x/model.pt`
Details: [`TTN/DATA_PIPELINE.md`](TTN/DATA_PIPELINE.md)

## 3b. SigLIP2 — "Visually Similar Products" (substitute)

```bash
python SigLIP2/build_siglip2.py --snapshot w90_2017-12-09
python SigLIP2/generate_recommendations.py --snapshot w90_2017-12-09
```
→ `data/tower/w90_.../recommendations/substitutes.parquet`
Details: [`SigLIP2/DATA_PIPELINE.md`](SigLIP2/DATA_PIPELINE.md)

## 3c. Popularity — "Trending in Category"

```bash
python Popularity/build_popularity.py --snapshot w90_2017-12-09
```
→ `data/tower/w90_.../recommendations/popularity.parquet`
Details: [`Popularity/DATA_PIPELINE.md`](Popularity/DATA_PIPELINE.md)

## Minimal end-to-end example

```bash
python data_creation/build_data.py
python data_processing/build_snapshot.py --window-days 90
python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
python TTN/encode_descriptions.py --snapshot w90_2017-12-09
python TTN/build_model.py --snapshot w90_2017-12-09
python SigLIP2/build_siglip2.py --snapshot w90_2017-12-09
python SigLIP2/generate_recommendations.py --snapshot w90_2017-12-09
python Popularity/build_popularity.py --snapshot w90_2017-12-09
```

Every step is resumable/safe to re-run — each one either skips work already
on disk (`build_data.py`, unless `--force`) or picks up where it left off
(`build_siglip2.py`'s cache).
