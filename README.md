# RecSystem Project

## v1-recommendations structure

```
v1-recommendations
│
├── data_processing/       SHARED data prep + TTN-specific array building
│    build_snapshot.py      item list, item categories, cleaned pairs
│    build_ttn_arrays.py    TTN-only: items.npz, vocabs.json, slim pairs
├── TTN/                   Complements ("Complete the Look")
├── SigLIP2/                Substitutes ("Visually Similar Products")
├── Popularity/              Popular in category (all-time + recency)
├── prepare_serving.py       unifies all three into one table
├── load_serving_db.py       loads that table into SQLite
├── api.py                   FastAPI serving layer
└── demo/                    static demo page calling the API
```

Each folder has a `README.md` with what it does and the exact commands to
run it. Full pipeline, in order, for one snapshot:

```
# 1. Shared data prep -- must run first, all three models read its output
python data_processing/build_snapshot.py --window-days 90

# 2. Models
python data_processing/build_ttn_arrays.py --snapshot w90_2017-12-09   # TTN-specific arrays
python TTN/encode_descriptions.py --snapshot w90_2017-12-09     # optional, recommended
python TTN/build_model.py --snapshot w90_2017-12-09             # -> a candidate version
python SigLIP2/build_siglip2.py --snapshot w90_2017-12-09
python Popularity/build_popularity.py --snapshot w90_2017-12-09

# 3. Recommendation generation (needs a specific TTN --version from step 2's output)
python TTN/generate_recommendations.py --snapshot w90_2017-12-09 --version <date>_v_00x
python SigLIP2/generate_recommendations.py --snapshot w90_2017-12-09

# 4. Serving
python prepare_serving.py --snapshot w90_2017-12-09
python load_serving_db.py --snapshot w90_2017-12-09
SNAPSHOT=w90_2017-12-09 uvicorn api:app --reload

# 5. results.ipynb (this directory) evaluates Recall@10/@100 for TTN/SigLIP2/POP;
#    demo/index.html is a browsable page over the live API
```

Different `--window-days` values produce distinct, coexisting snapshots
(`data/tower/w{window_days}_{date_threshold}/`) rather than overwriting
each other. TTN versions candidate checkpoints under each snapshot
(`models/ttn/<date>_v_00x/`) rather than overwriting the last training run —
promotion to "champion" (which version `generate_recommendations.py` and
the API actually use) is a deliberate manual step, not automatic.

The shared data pipeline TTN and SigLIP2 both read from —
`feature_extraction_workflow/`, `embedding_analysis/`,
`complementary_cats_pairs/` — is unchanged at the repo root.

## Setup

### Clone repo
git clone https://github.com/flamesmith/RecSystem.git

### Dataset

Download dataset from Google Drive and place in:

data/

Do NOT commit datasets to GitHub.

### Install dependencies

pip install -r requirements.txt
