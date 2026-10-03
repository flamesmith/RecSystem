# Popularity — Data Pipeline

How a snapshot's reviews become the "Trending in Category" carousel. For
what Popularity does conceptually, see `Popularity/README.md`. Same diagram
style as `TTN/DATA_PIPELINE.md`: white box = transformation step, colored
box right after it = the variable(s) that step produces, amber box = the
orchestrator script, blue box = a file on disk.

The simplest of the three pipelines — one script, no separate
"model," no functions imported from anywhere else.

## `Popularity/build_popularity.py --snapshot ...`

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
    SNAP[["item_asins.npy, node_of_item.npy<br/>(from data_processing/build_snapshot.py)"]]:::file
    CFG["TTN/constants.json<br/>date_threshold"]:::config

    P_SPLIT["split reviews by date_threshold"]:::proc
    V_SPLIT["all_time = every review before the cutoff<br/>recency = reviews within RECENCY_WINDOW_DAYS<br/>(default 60) immediately before it"]:::vars

    P_COUNT["top_by_node(), run once per variant:<br/>count reviews per asin (one review = one purchase proxy);<br/>map asin → tower item_idx → its own category_node_id"]:::proc
    V_COUNT["a review count per item,<br/>tagged with its own category_node_id"]:::vars

    P_TOPK["rank within each category_node_id,<br/>keep the top 20"]:::proc
    V_TOPK["category_node_id, variant,<br/>rank, candidate_asin, score"]:::vars

    OUT[["popularity.parquet"]]:::file

    ORCH ==>|drives the pipeline below| P_SPLIT

    RAWREV --> P_SPLIT
    CFG --> P_SPLIT --> V_SPLIT --> P_COUNT
    SNAP -.item_idx, category_node_id.-> P_COUNT
    P_COUNT --> V_COUNT --> P_TOPK --> V_TOPK --> OUT
```

## Two things worth knowing

- **`RECENCY_WINDOW_DAYS` (60) is unrelated to `window_days`** (the
  co-purchase *pairing* window `data_processing/build_snapshot.py --window-days`
  uses to build TTN's training pairs) — same word, two different concepts,
  living in two different scripts.
- **The README is stale on the output filename.** `Popularity/README.md`
  says the output is `data/tower/<snapshot_id>/popularity_top100.json`; the
  actual script writes `popularity.parquet`, and not even under `data/` —
  see Prerequisites below.

## Prerequisites

Requires `data_processing/build_snapshot.py --window-days N` to have already
run for the given snapshot — reads that snapshot's `item_asins.npy` and
`node_of_item.npy` directly from the shared `data/tower/<snapshot_id>/`.
Output lands in `Popularity/generated/<snapshot_id>/recommendations/popularity.parquet`
— under this model's own folder, not `data/`.
