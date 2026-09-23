"""Validation checks for `data/df_features.pkl`.

These are data checks, not unit tests: they run against a produced feature
table and report what is wrong with it. `run_all` returns a DataFrame of
findings and, with `strict=True`, raises if anything failed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import pandas as pd

from ..extract_features import VALID_RANGES, DIMENSION_UNKNOWN_UNIT
from .expected_columns import OPTIONAL_COLUMNS, expected_columns, flatten

PathLike = Union[str, Path]

# Share of rows that must carry a value, from the coverage measured on the
# 1.13M-item run. Set a few points below observed so ordinary drift passes and
# a real regression does not.
COVERAGE_FLOORS = {
    "asin": 1.00,
    "cat_3": 1.00,
    "Product_Type": 0.65,
    "Material": 0.60,
    "Features": 0.50,
    "Dimensions": 0.45,
    "Color": 0.40,
    "Piece_Count": 0.18,
    "Theme": 0.12,
    "Brand": 0.10,
    "Capacity_Volume": 0.07,
    "Size": 0.05,
}

# A unit column may only take these values (plus NA).
ALLOWED_UNITS = {
    "dimension_unit_clean": {"in", DIMENSION_UNKNOWN_UNIT},
    "weight_unit": {"lb", "g", "oz", "pound", "gram"},
    "thread_count_unit": {"thread count", "tc", "count", "series"},
}


def _finding(check, level, subject, detail):
    return {"check": check, "level": level, "subject": subject, "detail": detail}


def check_columns(df: pd.DataFrame, master_metadata: Mapping) -> list[dict]:
    """No column may appear that the contract does not describe, and none may vanish."""
    groups = expected_columns(master_metadata)
    expected = flatten(groups)
    actual = set(df.columns)

    out = []
    for col in sorted(actual - expected):
        out.append(_finding("columns", "FAIL", col,
                            "present in the table but not in the contract"))
    for col in sorted(expected - actual - OPTIONAL_COLUMNS):
        out.append(_finding("columns", "FAIL", col,
                            "in the contract but missing from the table"))
    for col in sorted((expected - actual) & OPTIONAL_COLUMNS):
        out.append(_finding("columns", "WARN", col, "optional column absent"))
    return out


def check_grain(df: pd.DataFrame, id_col: str = "asin") -> list[dict]:
    """One row per item."""
    if id_col not in df.columns:
        return [_finding("grain", "FAIL", id_col, "id column missing")]
    dupes = int(df[id_col].duplicated().sum())
    nulls = int(df[id_col].isna().sum())
    out = []
    if dupes:
        out.append(_finding("grain", "FAIL", id_col, f"{dupes:,} duplicate ids"))
    if nulls:
        out.append(_finding("grain", "FAIL", id_col, f"{nulls:,} null ids"))
    return out


def check_dtypes(df: pd.DataFrame) -> list[dict]:
    """Numeric columns must actually be numeric."""
    out = []
    for col in df.columns:
        numericish = (col.endswith(("_numeric", "_cleaned", "_in"))
                      or col in VALID_RANGES
                      or col.startswith("dimension_") and col[-1].isdigit())
        if numericish and col.split("_")[0] not in ("title", "description", "feature"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                out.append(_finding("dtype", "FAIL", col,
                                    f"expected numeric, found {df[col].dtype}"))
    return out


def check_coverage(df: pd.DataFrame, floors: Mapping[str, float] = COVERAGE_FLOORS) -> list[dict]:
    """A field that suddenly covers far fewer items usually means a broken pattern."""
    out = []
    for col, floor in floors.items():
        if col not in df.columns:
            continue
        got = float(df[col].notna().mean())
        if got < floor:
            out.append(_finding("coverage", "FAIL", col,
                                f"{got:.1%} of rows, floor is {floor:.0%}"))
    return out


def check_units(df: pd.DataFrame, allowed: Mapping[str, set] = ALLOWED_UNITS) -> list[dict]:
    """Unit columns must not grow new values."""
    out = []
    for col, ok in allowed.items():
        if col not in df.columns:
            continue
        seen = set(df[col].dropna().unique())
        for bad in sorted(seen - set(ok)):
            n = int((df[col] == bad).sum())
            out.append(_finding("units", "FAIL", col,
                                f"unexpected value {bad!r} on {n:,} rows"))
    return out


def check_ranges(df: pd.DataFrame, valid_ranges: Mapping = VALID_RANGES) -> list[dict]:
    """Every `_cleaned` column must actually respect its bound."""
    out = []
    for col, bounds in valid_ranges.items():
        cleaned = f"{col}_cleaned"
        if cleaned not in df.columns:
            continue
        low, high = bounds[0], bounds[1]
        inclusive = bounds[2] if len(bounds) > 2 else "both"
        v = df[cleaned].dropna()
        bad = int((~v.between(low, high, inclusive=inclusive)).sum())
        if bad:
            out.append(_finding("ranges", "FAIL", cleaned,
                                f"{bad:,} values outside ({low}, {high}) [{inclusive}]"))
    return out


def run_all(
    df: pd.DataFrame,
    master_metadata: Union[Mapping, PathLike] = "data/master_metadata.json",
    strict: bool = False,
) -> pd.DataFrame:
    """Run every check and return the findings, worst first."""
    if not isinstance(master_metadata, Mapping):
        with open(master_metadata) as f:
            master_metadata = json.load(f)

    findings: list[dict] = []
    findings += check_columns(df, master_metadata)
    findings += check_grain(df)
    findings += check_dtypes(df)
    findings += check_coverage(df)
    findings += check_units(df)
    findings += check_ranges(df)

    report = pd.DataFrame(findings, columns=["check", "level", "subject", "detail"])
    if not report.empty:
        report = report.sort_values(
            ["level", "check", "subject"], key=lambda s: s.map(
                {"FAIL": 0, "WARN": 1}).fillna(2) if s.name == "level" else s
        ).reset_index(drop=True)

    n_fail = int((report["level"] == "FAIL").sum()) if not report.empty else 0
    n_warn = int((report["level"] == "WARN").sum()) if not report.empty else 0
    print(f"{len(df):,} rows x {df.shape[1]} columns — "
          f"{n_fail} failure(s), {n_warn} warning(s)")
    if strict and n_fail:
        raise AssertionError(f"{n_fail} validation failure(s)\n{report.to_string()}")
    return report
