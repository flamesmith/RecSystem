from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProductInput:
    product_id: str
    title: str = ""
    descriptions: Sequence[str] = field(default_factory=tuple)
    features: Sequence[str] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)
    categories: Sequence[str] = field(default_factory=tuple)
    image_url: str | None = None


@dataclass(frozen=True)
class TransformResult:
    product_id: str
    title: str
    image_url: str | None
    category_path: str
    taxonomy_text: str
    description_clean: str
    description_canonical: str
    description_chunks: tuple[str, ...]
    pipeline_version: str
    config_hash: str
    source_hash: str
    cleaning_stats: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["description_chunks"] = list(self.description_chunks)
        value["cleaning_stats"] = dict(self.cleaning_stats)
        return value
