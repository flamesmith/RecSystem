# TTN — Data Pipeline

How the two raw files become the arrays `TTN/build_model.py` trains on.
For what the model itself does, see `TTN/README.md`.

## Overview

One chart: the tables/files, left to right by stage, but each table is broken
open to show the **variables** it actually holds, color-coded by variable
group so the same color can be followed across stages. Config files sit in
their own row at the **top**, and every edge out of one names exactly which
variable(s) it affects — not just "used here". Raw files sit at the far left,
listing only the columns that matter. `description`/`desc_emb` and anything
past "what feeds the model" (i.e. `TTN/build_model.py`'s own internals) are
left out — this stops exactly at the arrays the model reads.

**Color = variable group** (same color, same variable, wherever it appears):
🟦 category path (`cat_2/3/4`) · 🌸 attributes (`brand/color/material/product_type/features`) · 🟪 title text/embedding · 🟧 price · 🟩 also_buy / complementary categories · 🟢 target_node · ⬜ plumbing (join keys, vocab sizes — not a feature) · 🟥 computed but discarded

```mermaid
flowchart LR
    classDef rawc fill:#fde68a,stroke:#b45309,color:#3f2d05
    classDef config fill:#e0e7ff,stroke:#4338ca,color:#241d5c
    classDef catpath fill:#bfdbfe,stroke:#1d4ed8,color:#0f2a63
    classDef attrs fill:#fbcfe8,stroke:#be185d,color:#5c0d31
    classDef text fill:#e9d5ff,stroke:#7e22ce,color:#3b0764
    classDef price fill:#fed7aa,stroke:#c2410c,color:#5c1a06
    classDef alsobuy fill:#bbf7d0,stroke:#15803d,color:#0b3a1e
    classDef targetnode fill:#99f6e4,stroke:#0f766e,color:#04312c
    classDef neutral fill:#e5e7eb,stroke:#6b7280,color:#1f2328
    classDef discard fill:#fecaca,stroke:#b91c1c,color:#450a0a

    subgraph CFG[" STATIC CONFIG — tracked in git, hand-authored "]
        direction LR
        C1["master_metadata.json<br/>per-category field-extraction schema"]:::config
        C2["global_filters.json<br/>marketing phrases stripped from text"]:::config
        C3["category_taxonomy.json<br/>reviewed (cat_3, cat_4) whitelist"]:::config
        C4["TTN/constants.json<br/>shared train/test date_threshold"]:::config
    end

    R1["meta_Home_and_Kitchen_filtered.csv (RAW)<br/>key columns: asin, category, title,<br/>description, feature, brand, price, also_buy"]:::rawc
    R2["Home_and_Kitchen_filtered.csv (RAW)<br/>key columns: asin, reviewerID, unixReviewTime"]:::rawc

    subgraph S1[" STAGE 1 — data_creation/build_data.py "]
        direction TB
        subgraph F1FILE["df_features.pkl"]
            direction LR
            F1_CAT["cat_2, cat_3, cat_4<br/>(category path)"]:::catpath
            F1_TXT["title_cleaned"]:::text
            F1_ATTR["Color, Material,<br/>Product_Type, Features"]:::attrs
            F1_BRAND_V1["brand_clean (v1)<br/>clean_brand(): spelling-merge<br/>+ rare-fold"]:::discard
            F1_PRICE["price (untouched,<br/>still a raw string)"]:::price
            F1_AB["also_buy (parsed list)"]:::alsobuy
        end
        subgraph F2FILE["df_features_with_embeddings.pkl"]
            F2_TXT["title_embedding<br/>(384-d SBERT)"]:::text
        end
        subgraph F3FILE["pair_stats.pkl"]
            F3_AB["src/dst category path,<br/>edges, support, lift<br/>(every scored pair)"]:::alsobuy
        end
        subgraph F4FILE["complementary_categories.pkl"]
            F4_AB["same columns,<br/>pairs clearing min_edges + min_lift"]:::alsobuy
        end
        F1_TXT ==>|SBERT| F2_TXT
        F1_AB ==>|score support + lift| F3_AB
        F3_AB ==>|threshold| F4_AB
        F1_BRAND_V1 -.discarded, never read downstream.-> X1["✕"]:::discard
    end

    subgraph S2[" STAGE 2 — data_processing/build_snapshot.py --window-days N "]
        direction TB
        subgraph N12["item_asins.npy / node_of_item.npy"]
            N_ASIN["item_asins.npy<br/>asin, fixed order"]:::neutral
            N_NODE["node_of_item.npy<br/>item's OWN category → node id"]:::targetnode
        end
        subgraph N34["tower_pairs_train.parquet / tower_pairs_test.parquet"]
            direction LR
            N_CAT["query_/target_cat_2,3,4<br/>(cat_4 folded + vocab-fit)"]:::catpath
            N_ATTR["query_/target_brand_clean (v2),<br/>color, material, product_type, features"]:::attrs
            N_PRICE["query_/target_price (imputed)<br/>+ price_imputed flag"]:::price
            N_TXT["query_/target_title_cleaned<br/>(carried, not reused later)"]:::text
            N_TNODE["target_node string<br/>= cat_2 &gt; cat_3 &gt; cat_4"]:::targetnode
        end
        N_TNODE -.sorted unique strings.-> N_NODE
    end

    subgraph S3[" STAGE 3 — data_processing/build_ttn_arrays.py "]
        direction TB
        subgraph T1FILE["items.npz"]
            direction LR
            T_CAT["cat_ids[:,0:3]<br/>cat_2, cat_3, cat_4"]:::catpath
            T_ATTR["cat_ids[:,3:8]<br/>brand, color, material,<br/>product_type, features"]:::attrs
            T_PRICE["numeric[:,0:3]<br/>log1p(price), decile, imputed flag"]:::price
            T_TXT["title_emb (384-d)"]:::text
        end
        subgraph T2FILE["vocabs.json"]
            T_VOCAB["embedding-table sizes for<br/>cat_2/3/4, brand, color, material,<br/>product_type, features, target_node"]:::neutral
        end
        subgraph T34FILE["pairs_train.parquet / pairs_test.parquet"]
            T_JOIN["query_idx, target_idx<br/>(join key, not a feature)"]:::neutral
            T_TNODE["target_node_id"]:::targetnode
            T_WEIGHT["weight (always 1.0 today)"]:::neutral
        end
    end

    %% ---- static config: exactly which variables each one touches ----
    C1 -.extracts: Color, Material,<br/>Product_Type, Features.-> F1_ATTR
    C2 -.strips noise from: title_cleaned.-> F1_TXT
    C3 -.whitelist for valid src/dst pairs.-> F3_AB
    C3 -.folds cat_4 outside whitelist to '_Other'.-> N_CAT
    C4 -.splits ALL rows train/test by review date.-> S2

    %% ---- raw -> stage 1, one edge per variable ----
    R1 -->|category| F1_CAT
    R1 -->|title, description, feature: cleaned| F1_TXT
    R1 -->|title, description, feature: extracted| F1_ATTR
    R1 -->|brand| F1_BRAND_V1
    R1 -->|brand, again — Stage 2 re-reads it| N_ATTR
    R1 -->|price, unchanged| F1_PRICE
    R1 -->|also_buy| F1_AB
    R2 -->|reviewerID, unixReviewTime: forms co-purchase pairs| N34

    %% ---- stage 1 -> stage 2, same variable group ----
    F1_CAT ==>|attach to pairs| N_CAT
    F1_ATTR ==>|attach to pairs| N_ATTR
    F1_PRICE ==>|impute + flag| N_PRICE
    F1_TXT -.carried, unused past this point.-> N_TXT
    F4_AB -.licenses which pairs survive.-> N34

    %% ---- stage 2 -> stage 3, same variable group ----
    N_CAT ==>|own integer vocab| T_CAT
    N_ATTR ==>|own integer vocab| T_ATTR
    N_PRICE ==>|log1p + decile| T_PRICE
    F2_TXT ==>|reused by asin, not recomputed| T_TXT
    N_TNODE -.separate vocab fit again.-> T_TNODE
    N34 ==>|slim to idx form| T34FILE
```

**Two things this chart surfaces that a plain file list wouldn't:**
- **`brand` is computed twice, and the second computation wins.** Stage 1's `clean_brand()` (spelling-merge + rare-fold) produces `brand_clean (v1)` — shown in red, a dead end. `build_snapshot.py` looks for a column named `"brand_norm"`, doesn't find one, and silently recomputes `brand_clean` from the *raw* `brand` string instead (no spelling-merge, refit on train items only). Only that second version reaches `vocabs['brand']`.
- **`target_node` is encoded twice, independently** — once in Stage 2 (`node_of_item.npy`, the item's own category) and once in Stage 3 (`vocabs['target_node']` / `target_node_id`, the pair's target category). Both sort the same underlying strings the same way, so the ids match today, but they're two separate computations — a change to one without the other would silently desync them.

## Stage 1 — `data_creation/build_data.py`

Item-level, **window_days-independent** — run once, reused by every snapshot.
Resumable: each step is skipped (existing file reused) unless `--force`.

| # | Reads | Modifications | Writes |
|---|---|---|---|
| 1a | `meta_Home_and_Kitchen_filtered.csv` | Parse `title`/`description`/`feature` per `master_metadata.json`'s per-category schema → `title_cleaned` etc. Cleaning: lowercase, spell out numbers, space digit-unit pairs, strip non-`[a-z0-9\s.\-]` chars, drop stopwords, lemmatize, strip `global_filters.json` boilerplate phrases. Structured fields (brand, dimensions, numeric ranges) extracted from the cleaned text and flattened/canonicalized. | `df_features.pkl` — one row per item: cleaned text, structured features, category path, `also_buy` |
| 1b | `df_features.pkl` | SBERT (`all-MiniLM-L6-v2`) encodes `title_cleaned` → 384-d unit-norm vector per item | `df_features_with_embeddings.pkl` |
| 1c | `df_features.pkl`'s `also_buy` lists, `category_taxonomy.json`, `meta_Home_and_Kitchen_filtered.csv` (for target-category lookup) | Explode each item's `also_buy` into (source category → target category) edges, resolving each target's category or marking it `"Not in catalogue"`. Aggregate into distinct category pairs, score `support` and `lift` (both directions kept separately). | `pair_stats.pkl` — every scored pair, unfiltered (cached; expensive step) |
| 1d | `pair_stats.pkl` | Keep pairs where `edges >= min_edges` **and** `lift >= min_lift` (both required — support alone rewards popular categories, lift alone rewards one-off flukes). Recomputed every run (cheap) so thresholds can be retried without rescoring. | `complementary_categories.pkl` |

## Stage 2 — `data_processing/build_snapshot.py --window-days N`

Shared by TTN, SigLIP2, and Popularity. Re-reads the raw review log directly
(separately from stage 1) to build the co-purchase pairs.

| Step | Modification |
|---|---|
| 1 | Split `Home_and_Kitchen_filtered.csv` into train/test by `unixReviewTime` vs. `TTN/constants.json`'s `date_threshold` |
| 2 | `co_purchase_pairs(window_days=N)` per split — two purchases by the same reviewer within `N` days count as "bought together" |
| 3 | Join both ends of each pair to `df_features.pkl`'s `cat_2/3/4` |
| 4 | Fold `cat_4` values outside `category_taxonomy.json`'s whitelist into `"<cat_3>_Other"` |
| 5 | **License direction**: keep only `(query→target)` rows whose category pair clears `complementary_categories.pkl` (A→B and B→A checked independently) |
| 6 | Fold rare brands, **fit on train only** (< 10 items → `"other_brands"`) |
| 7 | Attach full item attributes to both sides (title, price, brand, color, material, product_type, features) |
| 8 | Fit categorical vocabularies **on train only**, cast both splits |
| 9 | Impute missing prices hierarchically (median by cat_2/3/4 → cat_2/3 → cat_2 → global), **fit on train-reviewed items only**; flag imputed rows |
| 10 | Fill missing categoricals with `"Missing"` |
| 11 | Build `target_node` = `target_cat_2 > target_cat_3 > target_cat_4`; vocabulary fit on train, unseen test values → `"Missing"` |
| 12 | **Drop same-category pairs** (`query_cat_4 == target_cat_4`) — those are substitutes, not complements |

Outputs land in `data/tower/w{N}_{date_threshold}/`:
`item_asins.npy`, `node_of_item.npy` (shared), `tower_pairs_{train,test}.parquet` (TTN-only), `snapshot_manifest.json`.

## Stage 3 — `data_processing/build_ttn_arrays.py --snapshot ...`

TTN-only encoding, picks up stage 2's `tower_pairs_{train,test}.parquet`.

| Step | Modification |
|---|---|
| 1 | Dedup both sides/splits into one items table, keyed by asin |
| 2 | **Fit TTN's own integer-coded embedding vocabularies, train only** (distinct from stage 2's categorical dtypes) — id `0` reserved for padding/unseen |
| 3 | Encode categorical attrs → `cat_ids` int matrix |
| 4 | Encode price → `numeric` block: `log1p(price)`, price decile (normalized), imputed-flag |
| 5 | **Reuse** title vectors from `df_features_with_embeddings.pkl` (matched by asin, zero-vector if missing) — not recomputed |
| 6 | Slim wide pairs down to `(query_idx, target_idx, target_node_id, weight)` |
| 7 | Assert no *training* item/node lands on reserved id 0 (would mean a vocabulary bug); report how many *test-only* items/nodes do (expected) |

Outputs (same snapshot dir): `items.npz` (`title_emb`, `cat_ids`, `numeric`), `vocabs.json`, `pairs_{train,test}.parquet`.

## Stage 3b (optional, recommended) — `TTN/encode_descriptions.py --snapshot ...`

Runs standalone rather than inside stage 3 — SBERT alongside `df_features` (2.9 GB) and the embeddings pickle (4.3 GB) stalls badly on 16 GB machines.
SBERT-encodes `description_cleaned` (from `df_features.pkl`, truncated at 1200 chars), aligned row-for-row with `items.npz`. → `desc_emb.npy`.
`build_ttn_arrays.py` and `build_model.py` both pick this file up automatically if present; if absent, the model trains without the description block.

## Stage 4 — `TTN/build_model.py --snapshot ...`

No further data transformation — reads only what stages 2-3b produced
(`items.npz`, `vocabs.json`, `node_of_item.npy`, `pairs_{train,test}.parquet`, `desc_emb.npy` if present), trains `ComplementaryTwoTower` with a BPR pairwise loss over co-purchase pairs, and saves a versioned checkpoint:

```
data/tower/w{N}_{date_threshold}/models/ttn/<date>_v_00x/
  model.pt                state_dict, config, vocab_sizes, price standardisation, metrics
  version_manifest.json   same identity/config/metrics, readable without loading torch
```

A saved version is a **candidate**, not the champion — promotion is a separate, deliberate step.

## Design principle running through every stage

Anything **fitted** (vocabularies, brand folding, price-imputation medians, embedding-table ids) is fit on **train only** and applied to test — never the reverse. This is what makes `test-only items` / `unseen values falling on id 0` an expected, printed number rather than a silent leak.
