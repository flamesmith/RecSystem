# Popularity — Data Pipeline

How reviews become the "Trending in Category" carousel. For what
Popularity does conceptually, see `Popularity/README.md`. Same diagram
style as `TTN/DATA_PIPELINE.md`: white box = transformation step, colored
box right after it = the variable(s) that step produces, amber box = the
orchestrator script, blue box = a file on disk.

The simplest of the three pipelines — one script, no separate
"model," no functions imported from anywhere else, and — unlike TTN and
SigLIP2 — no `--snapshot` argument at all.

## `Popularity/build_popularity.py`

```mermaid
flowchart TB
    classDef raw fill:#fde68a,stroke:#b45309,color:#3f2d05
    classDef config fill:#e0e7ff,stroke:#4338ca,color:#241d5c
    classDef proc fill:#ffffff,stroke:#374151,color:#111827
    classDef vars fill:#bbf7d0,stroke:#15803d,color:#0b3a1e
    classDef file fill:#93c5fd,stroke:#1d4ed8,color:#0f2a63,stroke-width:2px
    classDef orch fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px

    ORCH["Popularity/build_popularity.py<br/>ORCHESTRATOR — one self-contained script,<br/>no functions imported from elsewhere"]:::orch

    RAWREV["Home_and_Kitchen_filtered.csv (RAW)<br/>asin, unixReviewTime"]:::raw
    FEAT[["df_features.pkl<br/>(from data_creation/build_data.py)"]]:::file
    CFG["TTN/constants.json<br/>date_threshold"]:::config

    P_SPLIT["split reviews by date_threshold"]:::proc
    V_SPLIT["all_time = every review before the cutoff<br/>recency = reviews within RECENCY_WINDOW_DAYS<br/>(default 60) immediately before it"]:::vars

    P_COUNT["top_by_category(), run once per variant:<br/>count reviews per asin (one review = one purchase proxy);<br/>map asin → its own cat_2/cat_3/cat_4 (cat_4 nulls → 'Missing')"]:::proc
    V_COUNT["a review count per item,<br/>tagged with its own (cat_2, cat_3, cat_4)"]:::vars

    P_TOPK["rank within each (cat_2, cat_3, cat_4),<br/>keep the top 20"]:::proc
    V_TOPK["cat_2, cat_3, cat_4, variant,<br/>rank, candidate_asin, score"]:::vars

    OUT[["popularity.parquet"]]:::file

    ORCH ==>|drives the pipeline below| P_SPLIT

    RAWREV --> P_SPLIT
    CFG --> P_SPLIT --> V_SPLIT --> P_COUNT
    FEAT -.cat_2, cat_3, cat_4.-> P_COUNT
    P_COUNT --> V_COUNT --> P_TOPK --> V_TOPK --> OUT
```

## Two things worth knowing

- **`RECENCY_WINDOW_DAYS` (60) is unrelated to `window_days`** (the
  co-purchase *pairing* window `data_processing/build_snapshot.py --window-days`
  uses to build TTN's training pairs) — same word, two different concepts,
  living in two different scripts.
- **No longer reads `item_asins.npy`/`node_of_item.npy`.** An earlier
  version restricted counts to TTN's co-purchase-derived item list and
  category vocabulary — an unjustified restriction inherited from reusing
  TTN's infrastructure, not something Popularity's own logic needs.
  Categories now come straight from `df_features.pkl`.

## Prerequisites

Requires `data_creation/build_data.py` to have already produced
`data/df_features.pkl` — no `data_processing/build_snapshot.py` step, no
`--snapshot` argument, no dependency on any `--window-days` choice at all.
Output lands in `Popularity/generated/recommendations/popularity.parquet`
— one universal file, under this model's own folder, not `data/` and not
scoped to any snapshot.
