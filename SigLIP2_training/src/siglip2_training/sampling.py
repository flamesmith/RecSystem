from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .metadata import amazon_record_to_product
from .text import DescriptionProcessor


def stable_priority(product_id: str, seed: str) -> int:
    payload = f"{seed}\0{product_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


@dataclass(frozen=True)
class SampleSelection:
    records: tuple[dict[str, Any], ...]
    source_records: int
    eligible_records: int
    missing_product_id: int
    missing_image: int
    missing_text: int


def select_deterministic_sample(
    records: Iterable[dict[str, Any]],
    *,
    description_config: Mapping[str, Any],
    sample_size: int,
    seed: str,
    require_image: bool = True,
    require_text: bool = True,
) -> SampleSelection:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    heap: list[tuple[int, str, int, dict[str, Any]]] = []
    source_records = eligible_records = missing_product_id = missing_image = missing_text = 0
    for ordinal, record in enumerate(records):
        source_records += 1
        product = amazon_record_to_product(record, description_config)
        if product is None:
            missing_product_id += 1
            continue
        if require_image and not product.image_url:
            missing_image += 1
            continue
        has_text = bool(product.title.strip() or product.descriptions or product.features)
        if require_text and not has_text:
            missing_text += 1
            continue
        eligible_records += 1
        priority = stable_priority(product.product_id, seed)
        item = (-priority, product.product_id, ordinal, record)
        if len(heap) < sample_size:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [(-negative_priority, product_id, record) for negative_priority, product_id, _, record in heap]
    selected.sort(key=lambda item: (item[0], item[1]))
    return SampleSelection(
        records=tuple(record for _, _, record in selected),
        source_records=source_records,
        eligible_records=eligible_records,
        missing_product_id=missing_product_id,
        missing_image=missing_image,
        missing_text=missing_text,
    )


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _content_hash(value: str) -> str:
    normalized = " ".join(value.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_for_group(group_id: str, seed: str, train_fraction: float, validation_fraction: float) -> str:
    score = stable_priority(group_id, seed) / float(2**256)
    if score < train_fraction:
        return "train"
    if score < train_fraction + validation_fraction:
        return "validation"
    return "test"


def transform_and_split(
    raw_records: Iterable[dict[str, Any]],
    *,
    description_config: Mapping[str, Any],
    processor: DescriptionProcessor,
    seed: str,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("Invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be below one")
    output: list[dict[str, Any]] = []
    duplicate_text_keys: list[str] = []
    for raw_record in raw_records:
        product = amazon_record_to_product(raw_record, description_config)
        if product is None:
            continue
        value = processor.transform(product).to_dict()
        value["sampling_priority"] = f"{stable_priority(product.product_id, seed):064x}"
        value["leaf_category"] = value["category_path"].split(" > ")[-1] if value["category_path"] else ""
        output.append(value)
        duplicate_source = " ".join(
            processor.normalize(part) for part in (*product.descriptions, *product.features) if str(part).strip()
        )
        duplicate_text_keys.append(_content_hash(duplicate_source or processor.normalize(product.title)))

    groups = _DisjointSet(len(output))
    image_owner: dict[str, int] = {}
    text_owner: dict[str, int] = {}
    for index, value in enumerate(output):
        image_key = str(value.get("image_url") or "")
        text_key = duplicate_text_keys[index]
        if image_key:
            if image_key in image_owner:
                groups.union(index, image_owner[image_key])
            else:
                image_owner[image_key] = index
        if text_key:
            if text_key in text_owner:
                groups.union(index, text_owner[text_key])
            else:
                text_owner[text_key] = index

    members: dict[int, list[int]] = {}
    for index in range(len(output)):
        members.setdefault(groups.find(index), []).append(index)
    split_counts: Counter[str] = Counter()
    duplicate_groups = 0
    for indices in members.values():
        product_ids = sorted(str(output[index]["product_id"]) for index in indices)
        group_id = hashlib.sha256("\0".join(product_ids).encode("utf-8")).hexdigest()[:20]
        split = _split_for_group(group_id, seed, train_fraction, validation_fraction)
        if len(indices) > 1:
            duplicate_groups += 1
        for index in indices:
            output[index]["duplicate_group_id"] = group_id
            output[index]["split"] = split
            split_counts[split] += 1

    output.sort(key=lambda value: (value["sampling_priority"], value["product_id"]))
    leaf_counts = Counter(str(value["leaf_category"] or "(missing)") for value in output)
    statistics = {
        "written_records": len(output),
        "duplicate_groups": duplicate_groups,
        "split_counts": dict(sorted(split_counts.items())),
        "unique_leaf_categories": len(leaf_counts),
        "top_leaf_categories": dict(leaf_counts.most_common(20)),
    }
    return output, statistics
