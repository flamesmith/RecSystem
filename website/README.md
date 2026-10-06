# Local demo website (`website/`)

A static page (`index.html`, vanilla JS, no build step) that calls the
FastAPI serving layer and shows every carousel (TTN complements x2, SigLIP2 complements, association rules, SigLIP2 substitutes, popular) for an item.

## Run

```
# 1. Build the pipeline for a snapshot (see repo root README for the full order)
python website/prepare_serving.py --snapshot w90_2017-12-09
python website/load_serving_db.py --snapshot w90_2017-12-09   # also loads titles/images

# 2. Start the API -- it also serves this page
SNAPSHOT=w90_2017-12-09 uvicorn website.api:app     # from the repo root

# 3. Open http://127.0.0.1:8000/
```

The page is served by the API at `/`, so both share one origin and no CORS
is configured. Opening `index.html` as a `file://` page no longer works.
Product images load straight from Amazon's image URLs in the browser, so
they need an internet connection.
