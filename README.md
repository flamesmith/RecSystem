# RecSystem Project

## v1-recommendations structure

```
v1-recommendations
│
├── TTN/                Complete the Look (complementary products)
├── SigLIP2/             Visually Similar Products (substitutes)
└── Popularity/           baseline / fallback for Complete the Look
```

Each folder has a `README.md` with what it does and the exact commands to
run it. Build order: `TTN/build_data.py` first (both `SigLIP2/` and
`Popularity/` read what it produces), then the rest in any order. Once all
three have been built, `results.ipynb` (this directory) loads them and
reports Recall@10/@100.

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
