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

Pilot records add deterministic experiment fields:

- `sampling_priority`: seeded catalog-sampling hash.
- `duplicate_group_id`: connected component formed by exact normalized description or shared image URL.
- `split`: `train`, `validation`, or `test`; one duplicate group can belong to only one split.
- `leaf_category`: final source taxonomy level, or an empty string when unavailable.

## Training versus inference

Catalog products use the complete product transformation. Natural user queries receive only Unicode and whitespace normalization; they must not be assigned invented taxonomy or have query terms removed as marketing boilerplate.

## Duplicate policy

- Duplicate sentences inside one product are removed conservatively.
- Products with duplicate descriptions or images are retained but must be grouped before train/validation/test splitting.
- Product variants must not leak near-identical images or text across evaluation boundaries.

## Frozen feature shards

Each `features-*.npz` shard is atomic and contains aligned rows:

- `product_id`: stable product identifiers.
- `image`: normalized pretrained image embeddings, shape `[N, 768]`.
- `text_raw`: normalized first-64-token text embeddings, shape `[N, 768]`.
- `text_canonical`: normalized canonical text embeddings, shape `[N, 768]`.
- `text_taxonomy`: normalized taxonomy embeddings, shape `[N, 768]`.
- `text_chunks`: up to four normalized chunk embeddings, shape `[N, 4, 768]`.
- `text_chunk_mask`: valid-chunk mask, shape `[N, 4]`.
- `taxonomy_mask`: indicates whether a real taxonomy string was present; missing taxonomy is never replaced with an invented label.
- `split`, `duplicate_group_id`, `category_path`, and `leaf_category`: aligned evaluation metadata.

Floating-point arrays are finite `float16` values. The manifest binds each shard set to its source metadata, feature configuration, cache configuration, packages, and Git revision. These are final pretrained embeddings, not the 196 vision patch tokens consumed by SigLIP2's native pooling head.
