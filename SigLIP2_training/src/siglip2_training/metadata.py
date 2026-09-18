from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .schema import ProductInput


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(f"{key}: {item}" for key, item in value.items() if item not in (None, "", [], {}))
    if isinstance(value, Iterable):
        output: list[str] = []
        for item in value:
            output.extend(_strings(item))
        return tuple(output)
    return (str(value),)


def select_main_image_url(images: Any, settings: Mapping[str, Any]) -> str | None:
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        return None
    required_variant = str(settings.get("required_variant", "MAIN")).casefold()
    preferred_sizes = [str(value) for value in settings.get("preferred_sizes", ["large", "hi_res", "thumb"])]
    mappings = [item for item in images if isinstance(item, Mapping)]
    main = [item for item in mappings if str(item.get("variant", "")).casefold() == required_variant]
    candidates = main or mappings
    for image in candidates:
        for field in preferred_sizes:
            value = image.get(field)
            if isinstance(value, str) and value.startswith(("https://", "http://")):
                return value
    return None


def amazon_record_to_product(record: Mapping[str, Any], config: Mapping[str, Any]) -> ProductInput | None:
    product_id = str(record.get("parent_asin") or record.get("asin") or "").strip()
    if not product_id:
        return None
    image_url = select_main_image_url(record.get("images"), config.get("image_url", {}))
    return ProductInput(
        product_id=product_id,
        title=str(record.get("title") or ""),
        descriptions=_strings(record.get("description")),
        features=_strings(record.get("features")),
        details=record.get("details") if isinstance(record.get("details"), Mapping) else {},
        categories=_strings(record.get("categories")),
        image_url=image_url,
    )


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source_path = Path(path)
    opener = gzip.open if source_path.suffix == ".gz" else open
    with opener(source_path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {source_path}") from error
            if isinstance(value, dict):
                yield value


def open_jsonl_writer(path: str | Path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(output_path, "wt", encoding="utf-8") if output_path.suffix == ".gz" else output_path.open("w", encoding="utf-8")
