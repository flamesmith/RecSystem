# Processed product data contract

## Input

The initial adapter targets Amazon Home & Kitchen metadata with these fields:

- `parent_asin`
- `title`
- `description`
- `features`
- `details`
- `categories`
- `images`

The adapter selects the `MAIN` image and prefers `large`, then `hi_res`, then `thumb`. The preference avoids downloading unnecessarily large source images for a 224px model while retaining fallbacks.

## Output

Every output line conforms to `schemas/processed_product.schema.json`.

Lineage fields are mandatory:

- `pipeline_version` identifies semantic behavior.
- `config_hash` identifies the exact serialized configuration.
- `source_hash` identifies all source fields that influence the output.

Downstream feature shards and model checkpoints must repeat the pipeline version and configuration hash in their manifests.

## Training versus inference

Catalog products use the complete product transformation. Natural user queries receive only Unicode and whitespace normalization; they must not be assigned invented taxonomy or have query terms removed as marketing boilerplate.

## Duplicate policy

- Duplicate sentences inside one product are removed conservatively.
- Products with duplicate descriptions or images are retained but must be grouped before train/validation/test splitting.
- Product variants must not leak near-identical images or text across evaluation boundaries.
