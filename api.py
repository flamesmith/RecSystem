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
  GET /health
      -> {"status": "ok", "snapshot": ...} -- confirms the DB is reachable.

Usage:
  python load_serving_db.py --snapshot w90_2017-12-09   # once, or after promotion
  uvicorn api:app --reload
"""
import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
SNAPSHOT_ID = os.environ.get("SNAPSHOT", "w90_2017-12-09")
DB_PATH = ROOT / "data" / "tower" / SNAPSHOT_ID / "serving" / "recommendations.db"

VALID_CAROUSELS = {"complements", "substitutes", "popular"}
VALID_VARIANTS = {"all_time", "recency"}

app = FastAPI(title="RecSystem serving API")
# Local demo only: demo/index.html is opened as a file:// page or a separate
# dev server, so the browser treats it as a different origin from this API.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)


def get_conn():
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"serving DB not found for snapshot {SNAPSHOT_ID!r} -- "
                    f"run load_serving_db.py --snapshot {SNAPSHOT_ID} first",
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
            """SELECT rank, candidate_asin, score FROM recommendations
               WHERE query_asin = ? AND carousel = ? ORDER BY rank""",
            (asin, carousel),
        ).fetchall()
        out[carousel] = [dict(r) for r in rows]

    for variant in VALID_VARIANTS:
        rows = conn.execute(
            """SELECT rank, candidate_asin, score FROM recommendations
               WHERE query_asin = ? AND carousel = 'popular' AND variant = ?
               ORDER BY rank""",
            (asin, variant),
        ).fetchall()
        out[f"popular_{variant}"] = [dict(r) for r in rows]

    conn.close()
    return {"query_asin": asin, "snapshot": SNAPSHOT_ID, "carousels": out}


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
            """SELECT rank, candidate_asin, score FROM recommendations
               WHERE query_asin = ? AND carousel = 'popular' AND variant = ?
               ORDER BY rank""",
            (asin, variant),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT rank, candidate_asin, score FROM recommendations
               WHERE query_asin = ? AND carousel = ? ORDER BY rank""",
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
