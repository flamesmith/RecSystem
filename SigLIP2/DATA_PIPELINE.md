# SigLIP2 — Data Pipeline

How a snapshot's items become "Visually Similar Products" recommendations.
For what SigLIP2 does conceptually, see `SigLIP2/README.md`. Same diagram
style as `TTN/DATA_PIPELINE.md`: white box = transformation step, colored
box right after it = the variable(s) that step produces, amber box = the
orchestrator script, blue box = a file on disk.

Much shorter pipeline than TTN's — no text/attribute extraction, just:
item → image URL → image bytes → embedding → similarity ranking.

## Stage A — `SigLIP2/build_siglip2.py --snapshot ...`

Resumable, asin-keyed cache shared across every snapshot (an item's photo
doesn't change just because a different `--window-days` snapshot orders
items differently).

```mermaid
flowchart TB
    classDef raw fill:#fde68a,stroke:#b45309,color:#3f2d05
    classDef proc fill:#ffffff,stroke:#374151,color:#111827
    classDef vars fill:#bbf7d0,stroke:#15803d,color:#0b3a1e
    classDef file fill:#93c5fd,stroke:#1d4ed8,color:#0f2a63,stroke-width:2px
    classDef cache fill:#ddd6fe,stroke:#6d28d9,color:#2e1065,stroke-width:2px
    classDef orch fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px

    ORCH["SigLIP2/build_siglip2.py<br/>ORCHESTRATOR — self-contained,<br/>one helper imported from categories.py"]:::orch

    SNAP[["item_asins.npy<br/>(from data_processing/build_snapshot.py)"]]:::file
    DFFEAT[["df_features.pkl<br/>(from data_creation/build_data.py)"]]:::file

    P_URL["first_image_url() — imported from<br/>complementary_cats_pairs/categories.py<br/>picks imageURLHighRes, falls back to imageURL"]:::proc
    V_URL["urls[i] — one URL or empty, per item"]:::vars

    P_FETCH["fetch()<br/>download a ~224px thumbnail,<br/>32 concurrent workers"]:::proc
    V_FETCH["downloaded image file,<br/>or a fetch failure (status 3)"]:::vars

    P_ENCODE["encode()<br/>SigLIP2 image encoder<br/>(google/siglip2-base-patch16-224), L2-normalized"]:::proc
    V_ENCODE["768-d vector,<br/>status 1 = encoded"]:::vars

    CACHE[["_siglip_cache/ — cache_asins.npy, cache_emb.npy, cache_status.npy<br/>ASIN-KEYED, shared across every snapshot, grows over time.<br/>Status: 0 pending · 1 encoded · 2 no URL · 3 fetch failed · 4 decode failed"]]:::cache

    P_PROJECT["project the cache onto<br/>THIS snapshot's own item order"]:::proc

    OUT[["siglip_img_emb.npy (768-d per item)<br/>siglip_img_status.npy<br/>— per-snapshot; everything downstream reads this, not the cache"]]:::file

    ORCH ==>|drives the pipeline below| P_URL

    SNAP --> P_URL
    DFFEAT -.imageURL, imageURLHighRes.-> P_URL
    P_URL --> V_URL --> P_FETCH --> V_FETCH --> P_ENCODE --> V_ENCODE --> CACHE
    CACHE --> P_PROJECT
    SNAP -.this snapshot's item order.-> P_PROJECT
    P_PROJECT --> OUT
```

~28% of the catalogue has no usable image and ends up as a zero vector
(status 2, 3, or 4) — the same convention TTN uses for items with no
description.

## Stage B — `SigLIP2/generate_recommendations.py --snapshot ...`

```mermaid
flowchart TB
    classDef proc fill:#ffffff,stroke:#374151,color:#111827
    classDef vars fill:#fbcfe8,stroke:#be185d,color:#5c0d31
    classDef file fill:#93c5fd,stroke:#1d4ed8,color:#0f2a63,stroke-width:2px
    classDef orch fill:#fef3c7,stroke:#d97706,color:#78350f,stroke-width:2px

    ORCH["SigLIP2/generate_recommendations.py<br/>ORCHESTRATOR — fully self-contained"]:::orch

    SNAP[["item_asins.npy, node_of_item.npy<br/>(from build_snapshot.py)"]]:::file
    EMBIN[["siglip_img_emb.npy, siglip_img_status.npy<br/>(from Stage A)"]]:::file

    P_POOL["group items by node_of_item —<br/>each item's OWN category"]:::proc
    V_POOL["candidate pool per node =<br/>every item sharing that category"]:::vars

    P_SCORE["cosine similarity, each query against<br/>its own node's candidates;<br/>self excluded; no-image items masked out"]:::proc
    V_SCORE["scores[query, candidate]"]:::vars

    P_TOPK["top-20 per query, ranked"]:::proc
    V_TOPK["query_asin, rank, candidate_asin, score"]:::vars

    OUT[["substitutes.parquet"]]:::file

    ORCH ==>|drives the pipeline below| P_POOL

    SNAP --> P_POOL --> V_POOL --> P_SCORE
    EMBIN --> P_SCORE --> V_SCORE --> P_TOPK --> V_TOPK --> OUT
```

An item with no usable image (~28% of the catalogue) can't be ranked by
this signal and is skipped as a query entirely — it can still appear as a
*candidate* for other queries in its category, just never as the query
itself.

## Prerequisites

Both stages require `data_processing/build_snapshot.py --window-days N` to
have already run for the given snapshot — Stage A reads `item_asins.npy`,
Stage B additionally reads `node_of_item.npy`. Stage A also reads
`df_features.pkl` (from `data_creation/build_data.py`) for image URLs.

Outputs land in `data/tower/<snapshot_id>/` (Stage A) and
`data/tower/<snapshot_id>/recommendations/` (Stage B).
