from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .cache import ImageCache
from .config import config_hash, load_yaml
from .features import extract_feature_shards
from .manifest import build_manifest, write_manifest
from .metadata import amazon_record_to_product, iter_jsonl, open_jsonl_writer
from .text import DescriptionProcessor


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tokenizer(model_id: str, local_files_only: bool):
    if local_files_only:
        # Transformers 4.57 can still call model_info() while resolving tokenizer
        # compatibility. Set offline mode before importing Transformers so a
        # local-only run never attempts DNS or HTTP access.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)


def command_hardware(_: argparse.Namespace) -> int:
    root = _project_root()
    disk = shutil.disk_usage(root)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "disk_total_gib": round(disk.total / 1024**3, 2),
        "disk_used_gib": round(disk.used / 1024**3, 2),
        "disk_free_gib": round(disk.free / 1024**3, 2),
    }
    try:
        import torch

        report.update(
            {
                "torch": torch.__version__,
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "cuda_available": torch.cuda.is_available(),
            }
        )
    except ImportError:
        report["torch"] = None
    print(json.dumps(report, indent=2))
    return 0


def command_preprocess(arguments: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = _project_root()
    config = load_yaml(arguments.config)
    tokenizer = None
    if arguments.tokenizer:
        tokenizer = _load_tokenizer(arguments.tokenizer, arguments.local_files_only)
    processor = DescriptionProcessor(config, tokenizer=tokenizer)
    input_path = Path(arguments.input).resolve()
    output_path = Path(arguments.output).resolve()
    part_path = output_path.with_name(f"{output_path.stem}.part{output_path.suffix}")
    part_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "source_records": 0,
        "written_records": 0,
        "missing_product_id": 0,
        "missing_image_url": 0,
        "empty_clean_description": 0,
        "records_with_chunks": 0,
        "duplicate_sentences_removed": 0,
        "boilerplate_sentences_removed": 0,
        "source_characters": 0,
        "clean_characters": 0,
        "canonical_characters": 0,
        "description_chunks": 0,
        "multi_chunk_records": 0,
        "taxonomy_depth": 0,
    }
    try:
        with open_jsonl_writer(part_path) as output:
            for source_record in iter_jsonl(input_path):
                if arguments.limit is not None and counts["source_records"] >= arguments.limit:
                    break
                counts["source_records"] += 1
                product = amazon_record_to_product(source_record, config)
                if product is None:
                    counts["missing_product_id"] += 1
                    continue
                transformed = processor.transform(product)
                if not transformed.image_url:
                    counts["missing_image_url"] += 1
                if not transformed.description_clean:
                    counts["empty_clean_description"] += 1
                if transformed.description_chunks:
                    counts["records_with_chunks"] += 1
                counts["duplicate_sentences_removed"] += transformed.cleaning_stats[
                    "duplicate_sentences_removed"
                ]
                counts["boilerplate_sentences_removed"] += transformed.cleaning_stats[
                    "boilerplate_sentences_removed"
                ]
                counts["source_characters"] += transformed.cleaning_stats["source_characters"]
                counts["clean_characters"] += len(transformed.description_clean)
                counts["canonical_characters"] += len(transformed.description_canonical)
                counts["description_chunks"] += len(transformed.description_chunks)
                if len(transformed.description_chunks) > 1:
                    counts["multi_chunk_records"] += 1
                if transformed.category_path:
                    counts["taxonomy_depth"] += transformed.category_path.count(" > ") + 1
                output.write(json.dumps(transformed.to_dict(), ensure_ascii=False) + "\n")
                counts["written_records"] += 1
        part_path.replace(output_path)
    except Exception:
        part_path.unlink(missing_ok=True)
        raise
    written = counts["written_records"]
    counts["mean_clean_retention"] = round(
        counts["clean_characters"] / max(counts["source_characters"], 1), 4
    )
    counts["mean_clean_characters"] = round(counts["clean_characters"] / max(written, 1), 2)
    counts["mean_canonical_characters"] = round(
        counts["canonical_characters"] / max(written, 1), 2
    )
    counts["mean_chunks_per_record"] = round(counts["description_chunks"] / max(written, 1), 3)
    counts["mean_taxonomy_depth"] = round(counts["taxonomy_depth"] / max(written, 1), 3)
    counts["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    summary_path = output_path.with_name(output_path.name + ".summary.json")
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    summary_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    manifest = build_manifest(
        project_root=project_root.parent,
        command=sys.argv,
        inputs={"metadata": str(input_path)},
        outputs={"processed_metadata": str(output_path), "summary": str(summary_path)},
        configuration={
            "path": str(Path(arguments.config).resolve()),
            "hash": config_hash(config),
            "pipeline_version": config["pipeline"]["version"],
            "tokenizer": arguments.tokenizer,
            "local_files_only": arguments.local_files_only,
        },
        statistics=counts,
    )
    write_manifest(manifest_path, manifest)
    print(json.dumps({"summary": counts, "output": str(output_path), "manifest": str(manifest_path)}, indent=2))
    return 0


def command_cache_status(arguments: argparse.Namespace) -> int:
    config = load_yaml(arguments.config)
    cache = ImageCache.from_config(config, _project_root())
    status = cache.status()
    status["ready_gib"] = round(status["ready_bytes"] / 1024**3, 4)
    print(json.dumps(status, indent=2))
    return 0


def command_cache_prefetch(arguments: argparse.Namespace) -> int:
    started = time.perf_counter()
    project_root = _project_root()
    config = load_yaml(arguments.config)
    cache = ImageCache.from_config(config, project_root)
    configured_workers = int(config.get("prefetch", {}).get("download_workers", 4))
    workers = int(arguments.workers or configured_workers)
    if workers < 1 or workers > configured_workers:
        raise ValueError(f"workers must be between 1 and configured maximum {configured_workers}")
    counts = {
        "source_records": 0,
        "unique_urls": 0,
        "duplicate_urls_skipped": 0,
        "cache_hits": 0,
        "downloaded": 0,
        "failed": 0,
    }
    seen: set[str] = set()

    def urls():
        for record in iter_jsonl(arguments.input):
            if arguments.limit is not None and counts["source_records"] >= arguments.limit:
                break
            counts["source_records"] += 1
            url = record.get("image_url")
            if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                continue
            if url in seen:
                counts["duplicate_urls_skipped"] += 1
                continue
            seen.add(url)
            counts["unique_urls"] += 1
            yield url

    def fetch(url: str) -> str:
        if cache.get(url) is not None:
            return "cache_hits"
        cache.get_or_download(url)
        return "downloaded"

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending: set[concurrent.futures.Future[str]] = set()
        for url in urls():
            pending.add(executor.submit(fetch, url))
            if len(pending) >= workers * 2:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    try:
                        counts[future.result()] += 1
                    except Exception:
                        counts["failed"] += 1
        for future in concurrent.futures.as_completed(pending):
            try:
                counts[future.result()] += 1
            except Exception:
                counts["failed"] += 1
    counts["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    counts["cache_status"] = cache.status()
    report_path = Path(arguments.report).resolve()
    manifest = build_manifest(
        project_root=project_root.parent,
        command=sys.argv,
        inputs={"processed_metadata": str(Path(arguments.input).resolve())},
        outputs={"cache_root": str(cache.root)},
        configuration={
            "path": str(Path(arguments.config).resolve()),
            "hash": config_hash(config),
            "workers": workers,
        },
        statistics=counts,
    )
    write_manifest(report_path, manifest)
    print(json.dumps({"summary": counts, "manifest": str(report_path)}, indent=2))
    return 0


def command_extract_features(arguments: argparse.Namespace) -> int:
    project_root = _project_root()
    feature_config = load_yaml(arguments.config)
    cache_config = load_yaml(arguments.cache_config)
    cache = ImageCache.from_config(cache_config, project_root)
    output_directory = Path(arguments.output).resolve()
    result = extract_feature_shards(
        records=iter_jsonl(arguments.input),
        cache=cache,
        output_directory=output_directory,
        config=feature_config,
        limit=arguments.limit,
    )
    statistics = dict(result.__dict__)
    manifest = build_manifest(
        project_root=project_root.parent,
        command=sys.argv,
        inputs={"processed_metadata": str(Path(arguments.input).resolve())},
        outputs={"feature_directory": str(output_directory)},
        configuration={
            "features_path": str(Path(arguments.config).resolve()),
            "features_hash": config_hash(feature_config),
            "cache_path": str(Path(arguments.cache_config).resolve()),
            "cache_hash": config_hash(cache_config),
        },
        statistics=statistics,
    )
    manifest_path = output_directory / "manifest.json"
    write_manifest(manifest_path, manifest)
    print(json.dumps({"summary": statistics, "manifest": str(manifest_path)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SigLIP2 full-catalog training utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hardware = subparsers.add_parser("hardware", help="Report local training capabilities")
    hardware.set_defaults(function=command_hardware)

    preprocess = subparsers.add_parser("preprocess", help="Transform Amazon metadata into versioned text records")
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--limit", type=int)
    preprocess.add_argument("--tokenizer")
    preprocess.add_argument("--local-files-only", action="store_true")
    preprocess.set_defaults(function=command_preprocess)

    cache_status = subparsers.add_parser("cache-status", help="Inspect the bounded image cache")
    cache_status.add_argument("--config", required=True)
    cache_status.set_defaults(function=command_cache_status)

    cache_prefetch = subparsers.add_parser(
        "cache-prefetch", help="Download processed-record images through the bounded cache"
    )
    cache_prefetch.add_argument("--input", required=True)
    cache_prefetch.add_argument("--config", required=True)
    cache_prefetch.add_argument("--limit", type=int)
    cache_prefetch.add_argument("--workers", type=int)
    cache_prefetch.add_argument(
        "--report", default="artifacts/cache_prefetch_manifest.json"
    )
    cache_prefetch.set_defaults(function=command_cache_prefetch)

    extract_features = subparsers.add_parser(
        "extract-features", help="Stream images and write frozen SigLIP2 embedding shards"
    )
    extract_features.add_argument("--input", required=True)
    extract_features.add_argument("--output", required=True)
    extract_features.add_argument("--config", required=True)
    extract_features.add_argument("--cache-config", required=True)
    extract_features.add_argument("--limit", type=int)
    extract_features.set_defaults(function=command_extract_features)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.function(arguments))
