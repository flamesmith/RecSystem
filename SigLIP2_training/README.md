# SigLIP2 full-catalog training

This directory contains the reproducible pipeline for training SigLIP2 on the full Home & Kitchen catalog. It is intentionally independent from the exploratory notebooks in `../notebooks`.

## Design goals

- Stream images through a bounded on-disk cache instead of retaining the full image corpus.
- Transform product descriptions with one deterministic, versioned implementation.
- Use the same product-text transformation during experiments, catalog ingestion, and production inference.
- Extract frozen SigLIP2 backbone features once, then perform most head tuning without downloading images again.
- Record configuration hashes and run manifests with every generated artifact.
- Keep all generated metadata, caches, image files, features, and checkpoints out of Git.

## Confirmed local hardware

- MacBook Pro 16-inch, 2019
- Intel Core i7, 6 cores
- 16 GiB RAM
- AMD Radeon Pro 5300M, 4 GiB VRAM
- PyTorch Metal device available as `mps:0` outside the workspace sandbox
- Approximately 101 GiB free disk space at project initialization

The local design therefore prioritizes frozen-feature extraction and projection-head training. Full-backbone fine-tuning is not the default.

## Project structure

```text
SigLIP2_training/
├── configs/                 Versioned text, cache, and phase settings
├── docs/                    Decision log and operating notes
├── src/siglip2_training/    Reusable production code
├── tests/                   Deterministic unit tests
├── artifacts/               Generated manifests and small reports (ignored)
├── cache/                   Bounded image cache (ignored)
├── features/                Frozen feature shards (ignored)
└── checkpoints/             Model checkpoints (ignored)
```

## Description pipeline contract

One raw product becomes several explicit text fields:

- `description_clean`: normalized description with duplicate and boilerplate sentences removed.
- `description_canonical`: compact taxonomy, structured attributes, and high-signal sentences.
- `taxonomy_text`: informative tail of the category path, excluding constant roots.
- `description_chunks`: overlapping chunks created with the exact SigLIP2 tokenizer.
- `pipeline_version`, `config_hash`, and `source_hash`: lineage fields required by training and inference.

The pipeline is deterministic. Changes to cleaning behavior require a new semantic version in `configs/description_v1.yaml`.

## Local cache budget

The default host-side budget is intentionally conservative:

- Training cache high-water mark: 8 GiB
- Eviction target: 6 GiB
- Permanent validation cache allowance: 2 GiB
- Active download/prefetch allowance: 1 GiB
- Stop new downloads below 30 GiB free disk

The cache stores validated image bytes under content-independent URL hashes, uses atomic downloads, and tracks access/failure state in SQLite.

## Four phases

1. **Text strategy pilot — 50K products:** raw, canonical, multi-chunk, and taxonomy-aware views.
2. **Optimization pilot — up to 200K products:** true batch size, hard negatives, balanced sampling, and bounded unfreezing.
3. **Confirmation:** finalists on fixed cached validation data and repeated seeds.
4. **Full catalog:** one resumable streaming feature-extraction pass followed by head training on stored features.

Phase limits are ceilings and success gates, not automatic workload commitments.

## Current status

The reusable foundation is implemented and smoke-tested against the real Home & Kitchen metadata and image URLs. See `docs/foundation_validation.md` for the measured gate. The next workload is a fixed, stratified Phase 1 sample; no full-catalog image download has been started.

## Run locally without installation

From this directory:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m unittest discover -s tests -v
```

Inspect the environment:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m siglip2_training hardware
```

Run a metadata-only preprocessing sample:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m siglip2_training preprocess \
  --input ../full_metadata/meta_Home_and_Kitchen.jsonl.gz \
  --output artifacts/description_v1_sample.jsonl.gz \
  --config configs/description_v1.yaml \
  --limit 1000
```

Add `--tokenizer google/siglip2-base-patch16-224 --local-files-only` to generate exact 64-token chunks from an already cached tokenizer.

Prefetch a bounded sample through the image cache:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m siglip2_training cache-prefetch \
  --input artifacts/description_v1_sample.jsonl.gz \
  --config configs/cache_local.yaml \
  --limit 100 \
  --workers 4 \
  --report artifacts/cache_prefetch_100.manifest.json
```

## Container scope

The Docker image packages CPU preprocessing for reproducibility and future production ingestion. Local model training remains native so PyTorch can access the Mac's Metal device.

## Generated-data policy

Never commit:

- Raw or cleaned full datasets
- Downloaded images
- SQLite cache databases
- Frozen feature matrices
- Model weights or optimizer checkpoints

Commit code, configuration, tests, schemas, decision records, and small aggregate reports only.
