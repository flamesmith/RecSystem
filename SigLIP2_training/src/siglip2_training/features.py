from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from .cache import ImageCache


@dataclass(frozen=True)
class FeatureExtractionResult:
    source_records: int
    written_records: int
    reused_records: int
    failed_records: int
    shards: tuple[str, ...]
    embedding_dimension: int
    device: str
    pretrained_logit_scale: float
    pretrained_logit_bias: float
    elapsed_seconds: float


def choose_device(preferences: Sequence[str]) -> str:
    import torch

    for preference in preferences:
        if preference == "mps" and torch.backends.mps.is_available():
            return "mps"
        if preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        if preference == "cpu":
            return "cpu"
    return "cpu"


def _batched(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _encode_texts(model: Any, tokenizer: Any, texts: Sequence[str], device: str, batch_size: int, maximum_tokens: int):
    import torch
    import torch.nn.functional as functional

    parts = []
    for batch in _batched(texts, batch_size):
        inputs = tokenizer(
            text=list(batch),
            padding="max_length",
            truncation=True,
            max_length=maximum_tokens,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            embeddings = model.get_text_features(**inputs)
            embeddings = functional.normalize(embeddings, dim=-1)
        parts.append(embeddings.cpu().to(torch.float16).numpy())
    return np.concatenate(parts, axis=0)


def _encode_images(model: Any, image_processor: Any, images: Sequence[Image.Image], device: str):
    import torch
    import torch.nn.functional as functional

    inputs = image_processor(images=list(images), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embeddings = model.get_image_features(**inputs)
        embeddings = functional.normalize(embeddings, dim=-1)
    return embeddings.cpu().to(torch.float16).numpy()


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part_path = path.with_suffix(path.suffix + ".part")
    with part_path.open("wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    part_path.replace(path)


def _safe_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key, "")
    return value if isinstance(value, str) else ""


def _completed_shards(output_directory: Path) -> tuple[list[str], set[str]]:
    names = sorted(path.name for path in output_directory.glob("features-*.npz"))
    completed: set[str] = set()
    for expected_index, name in enumerate(names):
        if name != f"features-{expected_index:05d}.npz":
            raise ValueError("Existing feature shards are not contiguous")
        with np.load(output_directory / name, allow_pickle=False) as shard:
            product_ids = [str(value) for value in shard["product_id"]]
        overlap = completed.intersection(product_ids)
        if overlap:
            raise ValueError(f"Duplicate product IDs across existing shards: {sorted(overlap)[:3]}")
        completed.update(product_ids)
    return names, completed


def _load_reuse_features(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    parts: dict[str, list[np.ndarray]] = {}
    for shard_name in index["shards"]:
        with np.load(directory / shard_name, allow_pickle=False) as shard:
            for key in shard.files:
                parts.setdefault(key, []).append(shard[key])
    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    positions = {str(product_id): row for row, product_id in enumerate(arrays["product_id"])}
    if len(positions) != len(arrays["product_id"]):
        raise ValueError("Reuse feature directory contains duplicate product IDs")
    return arrays, positions


def _reuse_feature_keys(enabled_views: dict[str, bool]) -> set[str]:
    keys = {"image"}
    if enabled_views.get("raw_first_64", True):
        keys.add("text_raw")
    if enabled_views.get("canonical_64", True):
        keys.add("text_canonical")
    if enabled_views.get("taxonomy_64", True):
        keys.add("text_taxonomy")
    if enabled_views.get("multichunk_64", True):
        keys.update({"text_chunks", "text_chunk_mask"})
    return keys


def _metadata_arrays(records: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "product_id": np.asarray([str(record["product_id"]) for record in records]),
        "split": np.asarray([str(record.get("split", "")) for record in records]),
        "duplicate_group_id": np.asarray(
            [str(record.get("duplicate_group_id", "")) for record in records]
        ),
        "category_path": np.asarray([str(record.get("category_path", "")) for record in records]),
        "leaf_category": np.asarray([str(record.get("leaf_category", "")) for record in records]),
        "taxonomy_mask": np.asarray([bool(record.get("taxonomy_text")) for record in records]),
    }


def _extract_batch(
    *,
    records: Sequence[dict[str, Any]],
    images: Sequence[Image.Image],
    model: Any,
    tokenizer: Any,
    image_processor: Any,
    device: str,
    text_batch_size: int,
    maximum_tokens: int,
    maximum_chunks: int,
    enabled_views: dict[str, bool],
    image_batch_size: int,
    taxonomy_feature_cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    arrays = _metadata_arrays(records)
    arrays["image"] = np.concatenate(
        [
            _encode_images(model, image_processor, image_batch, device)
            for image_batch in _batched(images, image_batch_size)
        ],
        axis=0,
    )
    if enabled_views.get("raw_first_64", True):
        arrays["text_raw"] = _encode_texts(
            model,
            tokenizer,
            [_safe_text(record, "description_clean") for record in records],
            device,
            text_batch_size,
            maximum_tokens,
        )
    if enabled_views.get("canonical_64", True):
        arrays["text_canonical"] = _encode_texts(
            model,
            tokenizer,
            [_safe_text(record, "description_canonical") for record in records],
            device,
            text_batch_size,
            maximum_tokens,
        )
    if enabled_views.get("taxonomy_64", True):
        taxonomy_texts = [_safe_text(record, "taxonomy_text") for record in records]
        unseen_taxonomy = list(
            dict.fromkeys(text for text in taxonomy_texts if text not in taxonomy_feature_cache)
        )
        if unseen_taxonomy:
            unseen_features = _encode_texts(
                model,
                tokenizer,
                unseen_taxonomy,
                device,
                text_batch_size,
                maximum_tokens,
            )
            taxonomy_feature_cache.update(zip(unseen_taxonomy, unseen_features, strict=True))
        arrays["text_taxonomy"] = np.stack(
            [taxonomy_feature_cache[text] for text in taxonomy_texts]
        )
    if enabled_views.get("multichunk_64", True):
        chunks_by_record = [record.get("description_chunks") or [] for record in records]
        if any(not chunks for chunks in chunks_by_record):
            raise ValueError("Multichunk view requested, but at least one record has no description_chunks")
        if any(len(chunks) > maximum_chunks for chunks in chunks_by_record):
            raise ValueError("A record exceeds the configured maximum_chunks")
        flat_chunks = [chunk for chunks in chunks_by_record for chunk in chunks]
        flat_embeddings = _encode_texts(
            model, tokenizer, flat_chunks, device, text_batch_size, maximum_tokens
        )
        dimension = flat_embeddings.shape[1]
        chunk_embeddings = np.zeros((len(records), maximum_chunks, dimension), dtype=np.float16)
        chunk_mask = np.zeros((len(records), maximum_chunks), dtype=np.bool_)
        offset = 0
        for row, chunks in enumerate(chunks_by_record):
            count = len(chunks)
            chunk_embeddings[row, :count] = flat_embeddings[offset : offset + count]
            chunk_mask[row, :count] = True
            offset += count
        arrays["text_chunks"] = chunk_embeddings
        arrays["text_chunk_mask"] = chunk_mask
    return arrays


def extract_feature_shards(
    *,
    records: Iterable[dict[str, Any]],
    cache: ImageCache,
    output_directory: Path,
    config: dict[str, Any],
    limit: int | None = None,
    resume: bool = False,
    reuse_feature_directory: Path | None = None,
) -> FeatureExtractionResult:
    started = time.perf_counter()
    output_directory.mkdir(parents=True, exist_ok=True)
    if (output_directory / "index.json").exists():
        raise FileExistsError(f"Feature output is already complete: {output_directory}")
    existing_shards, completed_product_ids = _completed_shards(output_directory)
    if existing_shards and not resume:
        raise FileExistsError(
            f"Partial feature output exists; rerun with resume enabled: {output_directory}"
        )
    model_config = config["model"]
    runtime = config["runtime"]
    local_only = bool(model_config.get("local_files_only", True))
    if local_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    device = choose_device(runtime.get("device_preference", ["mps", "cpu"]))
    model = AutoModel.from_pretrained(model_config["id"], local_files_only=local_only).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        local_files_only=local_only,
        use_fast=bool(runtime.get("use_fast_tokenizer", True)),
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_config["id"],
        local_files_only=local_only,
        use_fast=bool(runtime.get("use_fast_image_processor", False)),
    )
    image_batch_size = int(runtime.get("image_batch_size", 4))
    record_batch_size = int(runtime.get("record_batch_size", image_batch_size))
    text_batch_size = int(runtime.get("text_batch_size", 16))
    shard_size = int(runtime.get("shard_size", 1000))
    maximum_tokens = int(runtime.get("maximum_text_tokens", 64))
    maximum_chunks = int(runtime.get("maximum_chunks", 4))
    if str(runtime.get("output_dtype", "float16")) != "float16":
        raise ValueError("Only float16 feature output is currently supported")
    enabled_views = {key: bool(value) for key, value in config.get("views", {}).items()}
    required_reuse_keys = _reuse_feature_keys(enabled_views)
    if image_batch_size < 1 or record_batch_size < 1 or shard_size < 1:
        raise ValueError("image_batch_size, record_batch_size, and shard_size must be positive")

    reuse_arrays: dict[str, np.ndarray] = {}
    reuse_positions: dict[str, int] = {}
    if reuse_feature_directory is not None:
        reuse_arrays, reuse_positions = _load_reuse_features(Path(reuse_feature_directory))
        missing_reuse_keys = required_reuse_keys - set(reuse_arrays)
        if missing_reuse_keys:
            raise ValueError(f"Reuse feature directory is missing: {sorted(missing_reuse_keys)}")

    source_records = 0
    written_records = len(completed_product_ids)
    reused_records = 0
    failed_records = 0
    shard_names: list[str] = list(existing_shards)
    shard_arrays: dict[str, list[np.ndarray]] = {}
    shard_rows = 0

    def flush_shard() -> None:
        nonlocal shard_arrays, shard_rows
        if shard_rows == 0:
            return
        combined = {key: np.concatenate(parts, axis=0) for key, parts in shard_arrays.items()}
        shard_name = f"features-{len(shard_names):05d}.npz"
        _write_npz_atomic(output_directory / shard_name, combined)
        shard_names.append(shard_name)
        print(
            json.dumps(
                {
                    "feature_shard": shard_name,
                    "shard_rows": shard_rows,
                    "written_records": written_records,
                }
            ),
            flush=True,
        )
        shard_arrays = {}
        shard_rows = 0

    pending_records: list[dict[str, Any]] = []
    pending_images: list[Image.Image] = []
    taxonomy_feature_cache: dict[str, np.ndarray] = {}
    failure_part = output_directory / "failures.jsonl.part"
    failure_log = failure_part.open("w", encoding="utf-8")

    def append_arrays(arrays: dict[str, np.ndarray]) -> None:
        nonlocal written_records, shard_rows, shard_arrays
        for key, value in arrays.items():
            shard_arrays.setdefault(key, []).append(value)
        batch_rows = len(arrays["product_id"])
        written_records += batch_rows
        shard_rows += batch_rows
        if shard_rows >= shard_size:
            flush_shard()

    def process_pending() -> None:
        nonlocal pending_records, pending_images
        if not pending_records:
            return
        arrays = _extract_batch(
            records=pending_records,
            images=pending_images,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            text_batch_size=text_batch_size,
            maximum_tokens=maximum_tokens,
            maximum_chunks=maximum_chunks,
            enabled_views=enabled_views,
            image_batch_size=image_batch_size,
            taxonomy_feature_cache=taxonomy_feature_cache,
        )
        append_arrays(arrays)
        pending_records = []
        pending_images = []

    try:
        for record in records:
            if limit is not None and source_records >= limit:
                break
            source_records += 1
            product_id = str(record.get("product_id"))
            if product_id in completed_product_ids:
                continue
            reuse_row = reuse_positions.get(product_id)
            if reuse_row is not None:
                process_pending()
                arrays = _metadata_arrays([record])
                for key in required_reuse_keys:
                    arrays[key] = reuse_arrays[key][reuse_row : reuse_row + 1]
                append_arrays(arrays)
                reused_records += 1
                continue
            try:
                cached_image = cache.get_or_download(str(record["image_url"]))
                with Image.open(cached_image.path) as source_image:
                    image = source_image.convert("RGB")
                    image.load()
            except Exception as error:
                failed_records += 1
                failure_log.write(
                    json.dumps(
                        {
                            "product_id": record.get("product_id"),
                            "image_url": record.get("image_url"),
                            "error": f"{type(error).__name__}: {error}"[:1000],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            pending_records.append(record)
            pending_images.append(image)
            if len(pending_records) >= record_batch_size:
                process_pending()
        process_pending()
        flush_shard()
    finally:
        failure_log.close()
    if failed_records:
        failure_part.replace(output_directory / "failures.jsonl")
    else:
        failure_part.unlink(missing_ok=True)
    dimension = 0
    if shard_names:
        with np.load(output_directory / shard_names[0], allow_pickle=False) as first:
            dimension = int(first["image"].shape[1])
    result = FeatureExtractionResult(
        source_records=source_records,
        written_records=written_records,
        reused_records=reused_records,
        failed_records=failed_records,
        shards=tuple(shard_names),
        embedding_dimension=dimension,
        device=device,
        pretrained_logit_scale=float(model.logit_scale.detach().exp().cpu()),
        pretrained_logit_bias=float(model.logit_bias.detach().cpu()),
        elapsed_seconds=round(time.perf_counter() - started, 4),
    )
    (output_directory / "index.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )
    return result
