# Orchestration — Run Order

Which files to run, in order, to go from the two raw files to a running
serving API for all three recommenders. For *what* each step does, see the
linked `DATA_PIPELINE.md` in each folder — this file is just the sequence.

**Where generated output lives**: only what's genuinely shared across all
three models (`item_asins.npy`, `node_of_item.npy`, `tower_pairs_*.parquet`,
plus the cross-model serving layer) stays under `data/tower/<snapshot_id>/`.
Everything model-specific — TTN's arrays and trained checkpoints, SigLIP2's
embeddings, Popularity's counts, each model's own recommendations — lives
under that model's own folder (`TTN/generated/<snapshot_id>/`,
`SigLIP2/generated/<snapshot_id>/`), not under `data/`. Popularity's is the
one exception: `Popularity/generated/recommendations/popularity.parquet`
has no `<snapshot_id>` in its path at all, since it no longer depends on
any snapshot (see step 3c). Come back later and you'll find each model's
own output sitting right next to that model's code.

## 0. Prerequisites (already on disk, once)

- `data/meta_Home_and_Kitchen_filtered.csv`, `data/Home_and_Kitchen_filtered.csv` — raw, gitignored, must exist locally
- `data/master_metadata.json`, `data/global_filters.json`, `data/category_taxonomy.json` — tracked config
- `TTN/constants.json` — `date_threshold`

## 1. Shared data creation — run once, reused by everything after it

```bash
python data_creation/build_data.py
```
→ `data/df_features.pkl`, `df_features_with_embeddings.pkl`, `pair_stats.pkl`, `complementary_categories.pkl`, `asin_image_urls.json`
Details: [`TTN/DATA_PIPELINE.md`](TTN/DATA_PIPELINE.md) (Stage 1)

## 2. Shared snapshot — once per `--window-days` you want to try

```bash
python data_processing/build_snapshot.py --window-days 90
```
→ `data/tower/w90_<date_threshold>/{item_asins.npy, node_of_item.npy, tower_pairs_train.parquet, tower_pairs_test.parquet}`

Note the resulting **snapshot id** (`w90_2017-12-09`-style) — every command
below takes it as `--snapshot`, except Popularity's (3c), which doesn't
depend on a snapshot at all. Everything from here on can run in **any
order** — TTN, SigLIP2, and Popularity are independent branches; TTN and
SigLIP2 share step 1 and 2's output, Popularity only needs step 1's.

## 3a. TTN — "Complete the Look" (complementary)

```bash
python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
python TTN/encode_descriptions.py --snapshot w90_2017-12-09   # optional, recommended
python TTN/build_model.py --snapshot w90_2017-12-09
```
→ trained checkpoint at `TTN/generated/w90_.../models/<date>_v_00x/model.pt` — note
the printed `version_id` (e.g. `2026-09-19_v_001`), you need it next.

```bash
python TTN/generate_recommendations.py --snapshot w90_2017-12-09 --version 2026-09-19_v_001
```
→ `TTN/generated/w90_.../recommendations/complements.parquet` — this is the step
that actually produces usable recommendations; `build_model.py` alone only
saves a checkpoint. `--version` is deliberately explicit, no "latest" or
"champion" default — promoting a version to **champion** (the one a serving
API would use) is a separate, manual decision, not a script in this repo.

Details: [`TTN/DATA_PIPELINE.md`](TTN/DATA_PIPELINE.md)

## 3b. SigLIP2 — "Visually Similar Products" (substitute)

```bash
python SigLIP2/encode_siglip2_images.py --snapshot w90_2017-12-09
python SigLIP2/generate_recommendations.py --snapshot w90_2017-12-09
```
→ `SigLIP2/generated/w90_.../recommendations/substitutes.parquet`
Details: [`SigLIP2/DATA_PIPELINE.md`](SigLIP2/DATA_PIPELINE.md)

## 3c. Popularity — "Trending in Category"

```bash
python Popularity/build_popularity.py
```
No `--snapshot` — unlike TTN and SigLIP2, this doesn't depend on any
`--window-days` snapshot at all (see `Popularity/README.md` for why); run
once, reused by every snapshot.
→ `Popularity/generated/recommendations/popularity.parquet`
Details: [`Popularity/DATA_PIPELINE.md`](Popularity/DATA_PIPELINE.md)

## 4. Serving prep — precompute everything the API will ever need, once

This is the step that means **the API never runs a model, ever** — not
TTN, not SigLIP2, nothing. Every request it serves is a plain indexed
SQLite lookup against work already done in steps 1–3c. Requires all three
of 3a/3b/3c to have already run for this snapshot.

```bash
python website/prepare_serving.py --snapshot w90_2017-12-09
```
→ `data/tower/w90_.../serving/recommendations.parquet` — unifies TTN's
`complements` and `complements_proportional`, SigLIP2's `substitutes`, and
Popularity's `popular` (both variants) into one table, one row per
`(query_asin, carousel, rank)`, truncated to `DISPLAY_K` (10) -- except
`complements_proportional`, which keeps all 20 slots (truncating would drop
whole categories from its category-blocked ranking).

```bash
python website/load_serving_db.py --snapshot w90_2017-12-09
```
→ `data/tower/w90_.../serving/recommendations.db` — that table loaded into
SQLite, indexed on `(query_asin, carousel, variant)`.

## 5. Serve

```bash
SNAPSHOT=w90_2017-12-09 uvicorn website.api:app --reload
```
Reads only `recommendations.db` from step 4. The snapshot is fixed at
process startup (`SNAPSHOT` env var) — there's no live snapshot-swap, so
promoting a new snapshot or a new TTN version means re-running steps 4–5
and restarting this process. `website/index.html` is a static page that calls
this API; open it directly, no build step.

## Minimal end-to-end example

```bash
python data_creation/build_data.py
python data_processing/build_snapshot.py --window-days 90
python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09
python TTN/encode_descriptions.py --snapshot w90_2017-12-09
python TTN/build_model.py --snapshot w90_2017-12-09              # note the printed version_id
python TTN/generate_recommendations.py --snapshot w90_2017-12-09 --version 2026-09-19_v_001
python SigLIP2/encode_siglip2_images.py --snapshot w90_2017-12-09
python SigLIP2/generate_recommendations.py --snapshot w90_2017-12-09
python Popularity/build_popularity.py
python website/prepare_serving.py --snapshot w90_2017-12-09
python website/load_serving_db.py --snapshot w90_2017-12-09
SNAPSHOT=w90_2017-12-09 uvicorn website.api:app --reload
```

Every step is resumable/safe to re-run — each one either skips work already
on disk (`build_data.py`, unless `--force`), picks up where it left off
(`encode_siglip2_images.py`'s cache), or wholesale-replaces its output
(`website/load_serving_db.py`).

## Out of scope here

"Champion" promotion — deciding which TTN `--version` steps 4 onward should
use — is a deliberate manual decision in this repo, not a script. There's
no live multi-snapshot routing either: serving one snapshot means one
running `website/api.py` process pointed at it.
