# Foundation validation — 2026-09-18

## Outcome

The local-first preprocessing and bounded-cache foundation is ready for the Phase 1 text-strategy pilot. This gate does not claim that a model has been trained yet. It confirms that the inputs can be transformed, tokenized, downloaded, validated, and reproduced safely before starting an expensive run.

## Environment

| Check | Observed result | Decision |
|---|---:|---|
| Free disk at initialization | approximately 101 GiB | Keep an 8 GiB training cache and stop new downloads below 30 GiB free. |
| System memory | 16 GiB | Stream metadata and images; do not materialize the catalog in memory. |
| GPU | AMD Radeon Pro 5300M, 4 GiB | Use PyTorch MPS natively; default to frozen-feature extraction and small micro-batches. |
| PyTorch Metal test | allocation on `mps:0` succeeded | Native macOS training is viable. Docker remains CPU preprocessing only. |
| Raw metadata | `meta_Home_and_Kitchen.jsonl.gz`, approximately 2.8 GiB | Read compressed JSONL sequentially. |

## Automated checks

Nine unit tests passed. They cover deterministic text output and hashes, conservative sentence cleanup, tail-aware chunk selection, Amazon metadata adaptation, main-image selection, valid image writes, invalid image rejection, and least-recently-used cache eviction.

## Real-data smoke tests

| Test | Result |
|---|---|
| Metadata-only preprocessing | 1,000 rows written; no missing IDs, images, or clean descriptions. |
| Duplicate cleanup | 440 duplicate sentences and 23 boilerplate sentences removed. |
| Cleanup performance | approximately 1.6 seconds for 1,000 rows in the final gate, down from approximately 29.5 seconds before optimization. |
| Description retention | cleaned text retained 97.24% of source characters; the canonical view averaged 338.23 characters. |
| Taxonomy coverage | mean source hierarchy depth was 4.081 levels. |
| Exact SigLIP2 tokenizer | 100 products produced 341 chunks. |
| Token-length contract | minimum 14, maximum 63, mean 61.02 tokens; zero chunks over the 64-token limit. |
| Long-description coverage | selection included later portions instead of retaining only leading chunks. |
| Real image cache | five unique URLs processed: one hit, four downloads, zero failures. |
| Cached images after smoke test | five validated JPEG entries, 91,761 bytes total. |

## Known data-quality observation

Some listings contain conflicts between a title and description—for example, a title may advertise one set quantity while prose mentions another. Version 1 preserves this evidence rather than inventing a correction. Before production ranking, conflict flags should be measured and potentially exposed as structured quality features.

## Phase 1 entry gate

The following must remain true before increasing sample size:

- The same `description_v1` configuration and tokenizer revision are used for every compared strategy.
- Train, validation, and test membership is fixed by product ID before experiments begin.
- Validation image URLs are pinned and cached separately from the evictable training cache.
- Candidate strategies are compared on retrieval metrics and slice metrics, not only training loss.
- Failed downloads are logged and excluded consistently across candidates.
- A small end-to-end run completes before the planned 50,000-product ceiling is attempted.

## Recommended next run

Build a deterministic category-aware sample and start with 5,000 products. Extract frozen image/text features, verify restart behavior and validation metrics, then scale the identical run definition toward 50,000 only if runtime, failure rate, and memory remain inside budget.
