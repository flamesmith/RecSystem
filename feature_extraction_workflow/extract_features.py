"""Feature extraction pipeline for item text.

Turns one or more raw text columns (title / description / feature bullets) plus
a category schema into a per-row `extracted_features` dict.

Entry point: `run_feature_extraction`.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PathLike = Union[str, Path]

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18",
    "nineteen": "19", "twenty": "20",
}
NUMBER_PATTERN = re.compile(r"\b(" + "|".join(NUMBER_WORDS.keys()) + r")\b")

STOPWORDS_TO_KEEP = {"no", "not", "non", "off", "self"}

_DIMENSION_UNIT_RE = re.compile(
    r'[\s-]*(inch|inchs|in|cm|mm|ft|foot|feet|meter|m|quot|"|\")'
)

# Default schema for numeric parsing (matches notebooks/create_features.ipynb).
NUMERIC_UNIT_FIELDS = (
    "Capacity",
    "Capacity_Volume",
    "Piece_Count",
    "Thread_Count",
    "Weight",
)
NUMERIC_ONLY_FIELDS = {
    "Bar_Pressure": "bar_pressure_numeric",
    "Capacity_Cups": "capacity_cups_numeric",
    "Density_Weight": "density_weight_lb",
    "Pocket_Depth": "pocket_depth_in",
    "Power_Rating": "power_rating_w",
    "Stage_Count": "stage_count_numeric",
    "Voltage": "voltage_numeric",
}
DIMENSION_FIELD = "Dimensions"

# Only single-unit numeric fields are range-cleaned. Fields that carry a
# `*_unit` column (the NUMERIC_UNIT_FIELDS: Weight, Thread_Count, ...) are
# intentionally excluded — their raw numbers can mix units (e.g. Weight in
# both grams and pounds), so a single [low, high] range would be applied
# inconsistently. Those keep their `_numeric` + `_unit` columns but get no
# `_cleaned` version.
VALID_RANGES = {
    "bar_pressure_numeric": (1, 25),
    "capacity_cups_numeric": (1, 30),
    "density_weight_lb": (0.5, 30),
    "pocket_depth_in": (5, 25),
    "power_rating_w": (1, 5000),
    "stage_count_numeric": (1, 10),
    "voltage_numeric": (110, 240),
    # Excluded — these carry a `*_unit` column and can mix units, so range
    # cleaning would be applied inconsistently. Uncomment to re-enable.
    # "thread_count_numeric": (80, 2000),
    # "weight_numeric": (0.1, 300),
}

UNIT_MAP = {
    "feet": "ft", "foot": "ft",
    "ounce": "oz",
    "piece": "pc", "pieces": "pc", "pcs": "pc",
    "quart": "qt",
    "liter": "l",
    "inch": "in", "inchs": "in", "inch deep": "in",
    "pound": "lb",
    "gallon": "gal",
    "gram": "g", "gm": "g",
    "centimeter": "cm",
    "millimeter": "mm",
    "watt": "w",
    "volt": "v",
    "dozen": "dz",
    "cu ft": "cubic foot", "cuft": "cubic foot",
    "tc": "thread count", "thread": "thread count", "threadcount": "thread count",
    "pk": "pocket",
    "count": "count", "ct": "count",
    "pillowcase": "pillow case",
}


def _ensure_nltk_resources() -> None:
    for pkg in ("stopwords", "wordnet"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def _load_json_if_path(value):
    if isinstance(value, (str, Path)):
        with open(value) as f:
            return json.load(f)
    return value


def build_stopwords(extra_keep: Optional[Iterable[str]] = None) -> set:
    """English stopwords minus a kept set useful for product features."""
    _ensure_nltk_resources()
    keep = set(STOPWORDS_TO_KEEP)
    if extra_keep:
        keep.update(extra_keep)
    return set(stopwords.words("english")) - keep


def parse_list_string(text) -> str:
    """Flatten a stringified list (e.g. "['a', 'b']") into 'a b'.

    Returns '' for NaN / empty list, falls back to str(text) when it's not a list literal.
    """
    if pd.isna(text) or text == "[]":
        return ""
    try:
        items = ast.literal_eval(text)
        if isinstance(items, list):
            return " ".join(str(item) for item in items)
    except (ValueError, SyntaxError):
        pass
    return str(text)


def clean_text(text, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """Lowercase, normalize numbers/units, strip noise, drop stopwords, lemmatize."""
    if pd.isna(text) or text == "":
        return ""
    text = text.lower()
    text = NUMBER_PATTERN.sub(lambda m: NUMBER_WORDS[m.group()], text)
    text = re.sub(r"(\d+)-(pc|inch|oz|lb|piece)", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9\s.\-]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = " ".join(text.split())
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def build_global_filter_regex(global_filters: Sequence[str]) -> Optional[re.Pattern]:
    """Compile a single regex that matches any phrase in `global_filters` (longest first)."""
    if not global_filters:
        return None
    sorted_filters = sorted(global_filters, key=len, reverse=True)
    pattern = "|".join(r"\b" + re.escape(expr) + r"\b" for expr in sorted_filters)
    return re.compile(pattern)


def remove_global_filters(text: str, filter_regex: Optional[re.Pattern]) -> str:
    if not text or filter_regex is None:
        return text
    return filter_regex.sub("", text).strip()


def _safe_parse_list(x):
    if pd.isna(x):
        return None
    if isinstance(x, list):
        return x
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else None
    except (ValueError, SyntaxError):
        return None


def ensure_cat_columns(
    df: pd.DataFrame,
    category_col: str = "category",
    n_levels: int = 6,
) -> pd.DataFrame:
    """Parse `category` into `cat_1..cat_{n_levels}` columns if they don't already exist."""
    cat_cols = [f"cat_{i+1}" for i in range(n_levels)]
    if all(c in df.columns for c in cat_cols):
        return df
    if category_col not in df.columns:
        raise KeyError(
            f"Need either {cat_cols} or '{category_col}' column to derive cat_3."
        )
    df = df.copy()
    cat_lists = df[category_col].apply(_safe_parse_list)
    for i in range(n_levels):
        df[f"cat_{i+1}"] = cat_lists.apply(
            lambda x, i=i: x[i] if isinstance(x, list) and len(x) > i else None
        )
    return df


def filter_by_cat_3(
    df: pd.DataFrame,
    master_metadata: dict,
    cat_3_col: str = "cat_3",
) -> pd.DataFrame:
    """Keep only rows whose cat_3 has an extraction schema."""
    valid = set(master_metadata.keys())
    return df[df[cat_3_col].isin(valid)].copy()


def extract_features(text: str, cat_3: Optional[str], master_metadata: dict) -> dict:
    """Extract a {field: matched_value} dict from `text` using master_metadata[cat_3]."""
    if (
        pd.isna(cat_3)
        or cat_3 not in master_metadata
        or pd.isna(text)
        or text == ""
    ):
        return {}

    features: dict = {}
    text_lower = text.lower()
    for field, spec in master_metadata[cat_3].items():
        if spec["type"] == "dictionary":
            for val in sorted(spec["values"], key=len, reverse=True):
                if re.search(r"\b" + re.escape(val) + r"\b", text_lower):
                    features[field] = val
                    break
        elif spec["type"] == "regex":
            for pattern in spec["patterns"]:
                match = re.search(pattern, text_lower)
                if not match:
                    continue
                result = match.group(0)
                if field == "Dimensions":
                    tail = text_lower[match.end():]
                    unit_match = _DIMENSION_UNIT_RE.match(tail)
                    if unit_match:
                        result = text_lower[match.start(): match.end() + unit_match.end()]
                features[field] = result
                break
    return features


def merge_extracted(row, priority: Sequence[str]) -> dict:
    """Merge per-column extracted dicts. priority[0] wins over priority[-1]."""
    merged: dict = {}
    for col in reversed(priority):
        d = row.get(f"extracted_features_{col}")
        if d:
            merged.update(d)
    return merged


def run_feature_extraction(
    df: pd.DataFrame,
    text_columns: Sequence[str],
    master_metadata: Union[dict, PathLike],
    global_filters: Union[Sequence[str], PathLike] = (),
    category_col: str = "category",
    cat_3_col: str = "cat_3",
    priority: Optional[Sequence[str]] = None,
    list_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """End-to-end pipeline: cat parsing → filter → clean → extract → merge.

    Parameters
    ----------
    df : DataFrame
        One row per item. Must contain either `cat_1..cat_6` or `category_col`.
    text_columns : sequence of str
        Columns to clean and extract features from (e.g. ['title', 'description']).
    master_metadata : dict or path
        Schema {cat_3: {field: {type, values|patterns}}}. Path is loaded via JSON.
    global_filters : sequence of str or path
        Marketing phrases to strip before extraction.
    category_col : str
        Source column for cat_1..cat_6 if those don't exist yet.
    cat_3_col : str
        Column name to use as the schema key.
    priority : sequence of str, optional
        Highest-to-lowest priority order for merging extracted dicts.
        Defaults to the order of `text_columns`.
    list_columns : sequence of str, optional
        Subset of `text_columns` whose raw values are stringified lists
        (e.g. 'description', 'feature') and need parse_list_string applied.

    Returns
    -------
    DataFrame filtered to rows with valid cat_3, with these new columns:
      - cat_1..cat_6 (if not already present)
      - <col>_cleaned for each col in text_columns
      - extracted_features_<col> (dict) for each col in text_columns
      - extracted_features (dict) — merged result, priority-ordered
    """
    master_metadata = _load_json_if_path(master_metadata)
    if isinstance(global_filters, (str, Path)):
        global_filters = _load_json_if_path(global_filters)

    df = ensure_cat_columns(df, category_col=category_col)
    df = filter_by_cat_3(df, master_metadata, cat_3_col=cat_3_col)

    stop_words = build_stopwords()
    lemmatizer = WordNetLemmatizer()
    filter_regex = build_global_filter_regex(global_filters)
    list_columns_set = set(list_columns or ())

    for col in text_columns:
        raw = df[col].fillna("")
        if col in list_columns_set:
            raw = raw.apply(parse_list_string)
        cleaned = (
            raw.apply(lambda t: clean_text(t, stop_words, lemmatizer))
               .apply(lambda t: remove_global_filters(t, filter_regex))
               .str.replace(r"\s+", " ", regex=True)
               .str.strip()
        )
        df[f"{col}_cleaned"] = cleaned
        df[f"extracted_features_{col}"] = df.apply(
            lambda r, c=col: extract_features(
                r[f"{c}_cleaned"], r.get(cat_3_col), master_metadata
            ),
            axis=1,
        )

    if priority is None:
        priority = list(text_columns)
    df["extracted_features"] = df.apply(lambda r: merge_extracted(r, priority), axis=1)

    return df


def parse_numeric_and_unit(value):
    """Parse a string like '12 oz' or '16-oz' into (numeric, unit). NaN/NaN if unparseable."""
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return np.nan, np.nan
    value = value.strip().lower().replace("-", " ")
    match = re.match(r"^([\d.]+)\s*([a-z%]+.*)?$", value)
    if match:
        num = float(match.group(1))
        unit = match.group(2).strip() if match.group(2) else np.nan
        return num, unit
    return np.nan, np.nan


def parse_dimensions(value):
    """Parse '25x20x10 in' / '12 in x 18 in' / '32x-30' into (dim1, dim2, dim3, unit).

    dim1 >= dim2 >= dim3. Missing dims are NaN.
    """
    if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        return np.nan, np.nan, np.nan, np.nan
    value = value.strip().lower()
    match = re.match(
        r"^([\d.]+)\s*([a-z]*)\s*[x×]\s*-?\s*([\d.]+)\s*([a-z]*)\s*"
        r"(?:[x×]\s*-?\s*([\d.]+)\s*([a-z]*))?$",
        value,
    )
    if match:
        dims = [float(match.group(1)), float(match.group(3))]
        if match.group(5):
            dims.append(float(match.group(5)))
        dims.sort(reverse=True)
        unit = match.group(6) or match.group(4) or match.group(2) or np.nan
        unit = unit.strip() if isinstance(unit, str) and unit.strip() else np.nan
        dim1 = dims[0]
        dim2 = dims[1] if len(dims) >= 2 else np.nan
        dim3 = dims[2] if len(dims) >= 3 else np.nan
        return dim1, dim2, dim3, unit
    match = re.match(r"^([\d.]+)\s*([a-z]+.*)?$", value)
    if match:
        dim1 = float(match.group(1))
        unit = match.group(2).strip() if match.group(2) else np.nan
        return dim1, np.nan, np.nan, unit
    return np.nan, np.nan, np.nan, np.nan


def expand_features(
    df: pd.DataFrame,
    features_col: str = "extracted_features",
    numeric_unit_fields: Sequence[str] = NUMERIC_UNIT_FIELDS,
    numeric_only_fields: Mapping[str, str] = NUMERIC_ONLY_FIELDS,
    dimension_field: str = DIMENSION_FIELD,
    unit_map: Mapping[str, str] = UNIT_MAP,
) -> pd.DataFrame:
    """Expand the merged dict into typed columns — the table you saw in create_features.ipynb.

    For each row's `extracted_features` dict, this:
      1. Creates one column per field key (e.g. `Color`, `Material`, `Dimensions`).
      2. For `numeric_unit_fields` (e.g. `Capacity_Volume='16 oz'`), splits into
         `<field>_numeric` (16.0) and `<field>_unit` ('oz').
      3. For `numeric_only_fields`, parses numeric only into the named output column
         (single-unit fields like `power_rating_w`).
      4. For `dimension_field` (`Dimensions='12x18 in'`), produces sorted-descending
         `dimension_1`, `dimension_2`, `dimension_3`, `dimension_unit`.
      5. Normalizes every `_unit` column through `unit_map` (e.g. 'feet'→'ft').

    Fields configured but not present in any row's dict are skipped silently.
    Returns a copy of `df` with the new columns added.
    """
    df = df.copy()

    all_keys: set = set()
    for d in df[features_col]:
        if isinstance(d, dict):
            all_keys.update(d.keys())
    for key in sorted(all_keys):
        df[key] = df[features_col].apply(
            lambda x, k=key: x.get(k) if isinstance(x, dict) else None
        )

    parsed_unit_cols = []
    for field in numeric_unit_fields:
        if field not in df.columns:
            continue
        col_numeric = field.lower() + "_numeric"
        col_unit = field.lower() + "_unit"
        df[[col_numeric, col_unit]] = df[field].apply(
            lambda x: pd.Series(parse_numeric_and_unit(x))
        )
        parsed_unit_cols.append(col_unit)

    for field, col_name in numeric_only_fields.items():
        if field not in df.columns:
            continue
        df[col_name] = df[field].apply(lambda x: parse_numeric_and_unit(x)[0])

    if dimension_field in df.columns:
        df[["dimension_1", "dimension_2", "dimension_3", "dimension_unit"]] = (
            df[dimension_field].apply(lambda x: pd.Series(parse_dimensions(x)))
        )
        parsed_unit_cols.append("dimension_unit")

    for col in parsed_unit_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: unit_map.get(x, x) if isinstance(x, str) else x
            )

    return df


def save_features(df: pd.DataFrame, path: PathLike) -> None:
    """Persist the final feature dataframe to disk as a pickle.

    Pickle is used because several columns hold dicts/lists (e.g.
    `extracted_features`, `extracted_features_<col>`) which CSV can't
    round-trip without serialization gymnastics.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def clean_numeric_ranges(
    df: pd.DataFrame,
    valid_ranges: Mapping[str, Sequence[float]] = VALID_RANGES,
    suffix: str = "_cleaned",
) -> pd.DataFrame:
    """Add `<col><suffix>` columns with out-of-range values set to NaN.

    The original columns are preserved untouched — this just adds new columns
    so callers can compare or fall back. Columns named in `valid_ranges` but
    not present in `df` are skipped silently (so the call is safe even when
    the upstream extractor produced a subset of fields).
    """
    df = df.copy()
    for col, (low, high) in valid_ranges.items():
        if col not in df.columns:
            continue
        df[f"{col}{suffix}"] = df[col].where(df[col].between(low, high), np.nan)
    return df
