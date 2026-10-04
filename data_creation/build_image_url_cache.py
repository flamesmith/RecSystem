"""Build a small asin -> product image URL cache.

Why this exists: df_features.pkl has ~90 columns and 1.13M rows, so
loading it just to reach its 2 image-url columns (imageURL,
imageURLHighRes) costs ~7.4s on its own, even though the actual url
extraction (first_image_url) takes under 1s. Nothing else in that load is
needed for an asin -> picture lookup -- demo/recommendations_demo.ipynb is
the first consumer that wants exactly that and nothing more. This cache
precomputes it once; the result loads in ~0.15s, about 49x faster.

Items with no usable image (neither imageURLHighRes nor imageURL resolves
to a real url -- about half the catalogue) are left out of the output
entirely, not stored as null, since a lookup miss is already the correct
signal for "no image" (see first_image_url's own docstring).

Called as stage [4/4] of data_creation/build_data.py -- not a separate,
unintegrated step. Running this file directly also works standalone (e.g.
to rebuild just this cache without re-running the rest of build_data.py).

Output: data/asin_image_urls.json -- {asin: image_url}, one entry per item
that has a real image. Gitignored like every other data-derived artifact
in this project; regenerate by re-running build_data.py or this file.

Usage: python data_creation/build_image_url_cache.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_creation.complementary_cats_pairs import first_image_url


def build_image_url_cache(df_features: pd.DataFrame, out_path: Path) -> int:
    """Write {asin: image_url} to out_path for every item with a real image.
    Returns how many entries were written."""
    import json

    image_by_asin = pd.Series(
        first_image_url(df_features["imageURLHighRes"], df_features["imageURL"]).to_numpy(),
        index=df_features["asin"].to_numpy(),
    ).dropna()
    out_path.write_text(json.dumps(image_by_asin.to_dict()))
    return len(image_by_asin)


if __name__ == "__main__":
    import time

    OUT_PATH = ROOT / "data" / "asin_image_urls.json"
    t0 = time.time()
    df_features = pd.read_pickle(ROOT / "data" / "df_features.pkl")
    print(f"loaded df_features.pkl -- {len(df_features):,} rows [{time.time() - t0:.1f}s]")

    n_written = build_image_url_cache(df_features, OUT_PATH)
    print(f"items with a real image: {n_written:,} of {len(df_features):,} "
          f"({n_written / len(df_features):.1%})")
    print(f"written -> {OUT_PATH.relative_to(ROOT)} "
          f"({OUT_PATH.stat().st_size / 1e6:,.1f} MB) [{time.time() - t0:.1f}s total]")
