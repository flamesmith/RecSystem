# SigLIP2 full-catalog training

This directory contains the reproducible pipeline for training SigLIP2 on the full Home & Kitchen catalog. It is intentionally independent from the exploratory notebooks in `../notebooks`.

## Design goals

- Stream images through a bounded on-disk cache instead of retaining the full image corpus.
- Transform product descriptions with one deterministic, versioned implementation.
- Use the same product-text transformation during experiments, catalog ingestion, and production inference.
- Extract frozen normalized SigLIP2 embeddings once, then perform most adapter tuning without downloading images again.
- Record configuration hashes and run manifests with every generated artifact.
- Keep all generated metadata, caches, image files, features, and checkpoints out of Git.

## Confirmed local hardware

- MacBook Pro 16-inch, 2019
- Intel Core i7, 6 cores
- 16 GiB RAM
- AMD Radeon Pro 5300M, 4 GiB VRAM
- PyTorch Metal device available as `mps:0` outside the workspace sandbox
- Approximately 101 GiB free disk space at project initialization

The local design therefore prioritizes frozen-feature extraction and post-embedding adapter training. Full-backbone fine-tuning is not the default.

The trainable components in the frozen-feature stages are new post-embedding adapters. They are distinct from SigLIP2's native vision pooling head, whose inputs would be prohibitively large to retain for the full catalog.

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
4. **Full catalog:** one resumable streaming feature-extraction pass followed by adapter training on stored features.

Phase limits are ceilings and success gates, not automatic workload commitments.

The first pilot uses a deterministic hash sample rather than the first 5,000 rows. Exact duplicate descriptions and shared image URLs are grouped before assigning train, validation, or test membership.

The 50K expansion preserves the exact 5K validation/test IDs and removes any additional training row connected to them by an exact normalized description or shared image URL. This fixed benchmark is required for the 5K-versus-50K deterioration guardrail.

## Current status

The reusable foundation and 5K pilot are complete. The selected trained multi-chunk adapter improved sealed-test bidirectional Recall@10 from 58.50% to 70.90%, with its 95% paired-bootstrap gain interval entirely above zero. See `docs/pilot_5k_results.md` for the full comparison and the fixed-evaluation 50K protocol. No full-catalog image download has been started.

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

Extract pretrained embedding shards while using the same bounded cache:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m siglip2_training extract-features \
  --input artifacts/description_v1_tokenized_sample.jsonl.gz \
  --output features/pilot \
  --config configs/features_local.yaml \
  --cache-config configs/cache_local.yaml \
  --limit 100
```

Create the fixed 5K pilot from the complete metadata file:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m siglip2_training sample-catalog \
  --input ../full_metadata/meta_Home_and_Kitchen.jsonl.gz \
  --output artifacts/pilot_5k.jsonl.gz \
  --description-config configs/description_v1.yaml \
  --pilot-config configs/pilot_5k.yaml
```

The 5K extraction uses `configs/features_pilot_5k.yaml` and supports `--resume`. Train all guarded adapter candidates after the feature index is complete:

```bash
PYTHONPATH=src ../.venv-siglip2/bin/python -m siglip2_training train-adapters \
  --features features/pilot_5k \
  --output checkpoints/pilot_5k \
  --config configs/adapter_pilot.yaml \
  --guardrails configs/guardrails.yaml
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
