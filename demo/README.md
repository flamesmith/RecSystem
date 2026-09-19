# Local demo website

A static page (`index.html`, vanilla JS, no build step) that calls the
FastAPI serving layer and shows all three carousels for an item.

## Run

```
# 1. Build the pipeline for a snapshot (see repo root README for the full order)
# 2. Start the API
SNAPSHOT=w90_2017-12-09 uvicorn api:app --reload   # from the repo root

# 3. Open the page
open demo/index.html        # macOS; or just open the file in a browser
```

The page calls `http://127.0.0.1:8000` (uvicorn's default port) — if you
run the API on a different port, edit the `API` constant at the top of
`index.html`'s script.

CORS is enabled on the API (`allow_origins=["*"]`) specifically so this
file:// page can call it cross-origin — that's a local-demo convenience,
not something to carry into a real deployment.
