"""The column contract for `data/df_features.pkl`.

The point of this file is that **nothing should appear in the feature table
that is not described here**. The extracted half is derived from
`master_metadata.json` and the module constants rather than hard-coded, so a
schema edit updates the contract automatically — but anything the pipeline
produces that the schema does not describe is a finding, not a silent addition.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ..extract_features import (
    DIMENSION_COLS,
    DIMENSION_FIELD,
    NUMERIC_ONLY_FIELDS,
    NUMERIC_UNIT_FIELDS,
    VALID_RANGES,
)

# Passed straight through from meta_Home_and_Kitchen_filtered.csv.
SOURCE_COLUMNS = frozenset({
    "asin", "title", "description", "feature", "category", "brand",
    "rank", "main_cat", "price", "date", "tech1", "tech2",
    "imageURL", "imageURLHighRes",
})

TAXONOMY_COLUMNS = frozenset(f"cat_{i}" for i in range(1, 7))

# Added outside the extraction schema.
DERIVED_COLUMNS = frozenset({
    "brand_clean",                                    # Filter 5
    "dimension_unit", "dimension_unit_src", "dimension_unit_clean",   # Filter 6
})

TEXT_COLUMNS = ("title", "description", "feature")

# A column is allowed to be absent only if it is genuinely optional.
OPTIONAL_COLUMNS = frozenset({"tech1", "tech2", "imageURL", "imageURLHighRes"})


def expected_columns(
    master_metadata: Mapping,
    text_columns: Iterable[str] = TEXT_COLUMNS,
) -> dict[str, frozenset[str]]:
    """Every column the pipeline should produce, grouped by where it comes from."""
    fields = {f for schema in master_metadata.values() for f in schema}

    cleaned_text = {f"{c}_cleaned" for c in text_columns}
    per_source = {f"extracted_features_{c}" for c in text_columns}

    measures: set[str] = set()
    for f in NUMERIC_UNIT_FIELDS:
        if f in fields:
            measures |= {f"{f.lower()}_numeric", f"{f.lower()}_unit"}
    for f, col in NUMERIC_ONLY_FIELDS.items():
        if f in fields:
            measures.add(col)
    if DIMENSION_FIELD in fields:
        measures |= set(DIMENSION_COLS) | {f"{c}_in" for c in DIMENSION_COLS}

    ranged = {f"{c}_cleaned" for c in VALID_RANGES}

    return {
        "source": frozenset(SOURCE_COLUMNS),
        "taxonomy": frozenset(TAXONOMY_COLUMNS),
        "cleaned_text": frozenset(cleaned_text),
        "extracted_dicts": frozenset(per_source | {"extracted_features"}),
        "fields": frozenset(fields),
        "measures": frozenset(measures),
        "range_cleaned": frozenset(ranged),
        "derived": frozenset(DERIVED_COLUMNS),
    }


def flatten(groups: Mapping[str, Iterable[str]]) -> frozenset[str]:
    return frozenset().union(*groups.values())
