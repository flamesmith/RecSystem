# Promoted SigLIP2 200K adapter

This directory is the deliberately versioned deployment package for the selected 200K Home &
Kitchen pilot.

## Files

- `taxonomy_w20_random_200k.pt`: learned image and text residual adapters, logit parameters,
  training configuration, and base-feature metadata.
- `results.json`: fixed validation/test metrics and promotion decision.
- `manifest.json`: run environment and lineage.

## Runtime contract

The checkpoint is not a standalone SigLIP2 backbone. Inference also requires:

- `google/siglip2-base-patch16-224`;
- `configs/description_v1.yaml` for catalog text preprocessing; and
- an external catalog vector index for retrieval.

The 1.5 GB pilot feature shards, processed product sample, images, cache, and Hugging Face base-model
files remain outside Git.

Checkpoint SHA-256:

```text
bbbedcfa9c7d8bdb74552e087dca85c2d031f133b217914ffc8fac20dbcc34d8
```
