# RecSystem Project

## v1-recommendations structure

This branch organizes the project's recommendation models by which product
carousel each one backs:

```
v1-recommendations
│
├── TTN/                    Complete the Look
│    Two-tower neural network, complementary products (what goes WITH this).
│
├── SigLIP2/                 Visually Similar Products
│    Frozen image-embedding cosine similarity, substitute products (what
│    looks LIKE this).
│
├── Popularity/               baseline / fallback for Complete the Look
│    Documentation only — see Popularity/README.md for where the existing
│    popularity-scoring logic actually lives and how it backs up TTN.
│
└── analysis/                 (not part of the diagram above)
     Cross-model comparisons that score TTN, SigLIP2/CONTENT, and Popularity
     together — where "which one wins for this item" and "does combining
     them help" get answered.
```

Each folder has its own `README.md` with the detail. The shared data
pipeline both TTN and SigLIP2 read from — `feature_extraction_workflow/`,
`embedding_analysis/`, `complementary_cats_pairs/` — is unchanged at the
repo root and documented in its own folders as before.

## Setup

### Clone repo
git clone https://github.com/flamesmith/RecSystem.git

### Dataset

Download dataset from Google Drive and place in:

data/

Do NOT commit datasets to GitHub.

### Install dependencies

pip install -r requirements.txt
