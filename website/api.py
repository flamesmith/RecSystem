"""FastAPI serving layer -- reads the SQLite DB load_serving_db.py built,
never runs a model. No model weights, no training-pipeline imports: this
process only needs sqlite3 and FastAPI, by design (see load_serving_db.py's
docstring for why the DB carries all the weight this needs).

Champion snapshot is fixed at process startup (SNAPSHOT env var, defaulting
to the constant below) -- there is no live "swap snapshots without
restarting" mechanism yet; promoting a new snapshot means restarting this
process pointed at the new one.

Endpoints:
  GET /items/{asin}/recommendations
      -> all three carousels for one item, DISPLAY_K each
      Query params: none yet -- carousel/variant filtering is a fast
      follow if a caller ever needs just one carousel.
  GET /items/{asin}/recommendations/{carousel}
      -> one carousel only (carousel in complements, substitutes, popular).
      For "popular", pass ?variant=all_time or ?variant=recency
      (default: all_time).
  GET /
      -> the demo page (website/index.html), same origin as the API
  GET /health
      -> {"status": "ok", "snapshot": ...} -- confirms the DB is reachable.

Usage:
  python website/load_serving_db.py --snapshot w90_2017-12-09   # once, or after promotion
  uvicorn website.api:app --reload      # from the repo root; then open http://127.0.0.1:8000/
"""
import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]   # repo root; this file lives in website/
SNAPSHOT_ID = os.environ.get("SNAPSHOT", "w90_2017-12-09")
DB_PATH = ROOT / "data" / "tower" / SNAPSHOT_ID / "serving" / "recommendations.db"

VALID_CAROUSELS = {"complements", "substitutes", "popular"}
VALID_VARIANTS = {"all_time", "recency"}

app = FastAPI(title="RecSystem serving API")
# website/index.html is served by this same app at "/", so the page and the API
# share an origin -- no CORS needed.

# One SELECT shape for every recommendations query: the rec row plus the
# item's display title/image from the `items` table load_serving_db.py builds.
REC_SELECT = """SELECT r.rank, r.candidate_asin, r.score, i.title, i.image_url
                FROM recommendations r LEFT JOIN items i ON i.asin = r.candidate_asin"""


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).resolve().parent / "index.html")


def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"serving DB not found for snapshot {SNAPSHOT_ID!r} -- "
                    f"run website/load_serving_db.py --snapshot {SNAPSHOT_ID} first",
        )
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/items/sample")
def sample_items(n: int = 12):
    """A few query items, preferring ones with recommendations in the most
    carousels, for the demo page to offer instead of asking someone to
    already know an asin."""
    conn = get_conn()
    rows = conn.execute(
        """SELECT query_asin, COUNT(DISTINCT carousel) AS n_carousels
           FROM recommendations GROUP BY query_asin
           ORDER BY n_carousels DESC LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    return {"snapshot": SNAPSHOT_ID, "asins": [r["query_asin"] for r in rows]}


@app.get("/health")
def health():
    exists = DB_PATH.exists()
    return {"status": "ok" if exists else "db_missing", "snapshot": SNAPSHOT_ID}


def _query_exists(conn, asin: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM recommendations WHERE query_asin = ? LIMIT 1", (asin,)
    ).fetchone()
    return row is not None


@app.get("/items/{asin}/recommendations")
def all_recommendations(asin: str):
    conn = get_conn()
    if not _query_exists(conn, asin):
        conn.close()
        raise HTTPException(status_code=404, detail=f"no recommendations for {asin!r}")

    out = {}
    for carousel in ("complements", "substitutes"):
        rows = conn.execute(
            REC_SELECT + """ WHERE r.query_asin = ? AND r.carousel = ? ORDER BY r.rank""",
            (asin, carousel),
        ).fetchall()
        out[carousel] = [dict(r) for r in rows]

    for variant in VALID_VARIANTS:
        rows = conn.execute(
            REC_SELECT + """ WHERE r.query_asin = ? AND r.carousel = 'popular'
               AND r.variant = ? ORDER BY r.rank""",
            (asin, variant),
        ).fetchall()
        out[f"popular_{variant}"] = [dict(r) for r in rows]

    q = conn.execute("SELECT title, image_url FROM items WHERE asin = ?", (asin,)).fetchone()
    conn.close()
    return {"query_asin": asin, "snapshot": SNAPSHOT_ID,
            "query_title": q["title"] if q else None,
            "query_image_url": q["image_url"] if q else None,
            "carousels": out}


@app.get("/items/{asin}/recommendations/{carousel}")
def one_carousel(asin: str, carousel: str, variant: Optional[str] = None):
    if carousel not in VALID_CAROUSELS:
        raise HTTPException(
            status_code=400,
            detail=f"carousel must be one of {sorted(VALID_CAROUSELS)}, got {carousel!r}",
        )
    conn = get_conn()
    if not _query_exists(conn, asin):
        conn.close()
        raise HTTPException(status_code=404, detail=f"no recommendations for {asin!r}")

    if carousel == "popular":
        variant = variant or "all_time"
        if variant not in VALID_VARIANTS:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail=f"variant must be one of {sorted(VALID_VARIANTS)}, got {variant!r}",
            )
        rows = conn.execute(
            REC_SELECT + """ WHERE r.query_asin = ? AND r.carousel = 'popular'
               AND r.variant = ? ORDER BY r.rank""",
            (asin, variant),
        ).fetchall()
    else:
        rows = conn.execute(
            REC_SELECT + """ WHERE r.query_asin = ? AND r.carousel = ? ORDER BY r.rank""",
            (asin, carousel),
        ).fetchall()

    conn.close()
    return {
        "query_asin": asin,
        "snapshot": SNAPSHOT_ID,
        "carousel": carousel,
        "variant": variant if carousel == "popular" else None,
        "recommendations": [dict(r) for r in rows],
    }
