# SigLIP2 Product Exploration — Decision Journal

This journal records the reasoning behind the local environment and notebook design for `siglip2_product_exploration.ipynb`. It is written as a running engineering log so later changes can be compared with the original constraints.

## 2026-08-26 — Initial repository and data audit

### Decision: treat the CSV manifest as the source of truth

The notebook reads `data/meta_Home_and_Kitchen_siglip2_pairs_10k.csv`. The file has exactly 10,000 data rows and three columns: `asin`, `description`, and `image_path`. The image directory also contains exactly 10,000 files.

I chose to resolve each CSV `image_path` from the repository root instead of reconstructing a path from the ASIN. This keeps the notebook coupled to the explicit image-description pairs in the manifest and allows a future dataset to use a different image layout without changing the Dataset class.

What this optimizes: correctness, portability between running Jupyter from the repository root or the `notebooks` directory, and early failure when the data contract changes.

Alternative not selected: silently discard rows with missing descriptions or images. That would let experiments run on an unknown subset and make metrics difficult to reproduce. The notebook instead reports quality counts and stops if the expected 10,000 clean pairs are not present.

### Decision: use Python 3.11.6 in a repository-local virtual environment

The virtual environment is `.venv-siglip2` at the repository root and uses the already-installed pyenv Python 3.11.6 interpreter.

I selected Python 3.11 rather than the system Python 3.9 because current Transformers requires Python 3.10 or newer, while Python 3.11 has broad wheel support across the chosen scientific stack. Reusing the installed interpreter avoids changing the user's system Python or Conda environments.

What this optimizes: isolation from the existing recommender-system environment, compatibility with SigLIP2 support in Transformers, and low setup risk.

Alternatives not selected:

- System Python 3.9: too old for the current Transformers package line and would mix notebook dependencies with a global interpreter.
- Python 3.12 or 3.13: no suitable interpreter is already installed here, and they do not improve this experiment enough to justify another runtime installation.
- Conda environment: workable, but a standard `venv` is smaller and sufficient because all selected packages have compatible wheels.

### Decision: pin PyTorch 2.2.2 and NumPy 1.26.4

This laptop is an Intel `x86_64` Mac. PyTorch deprecated macOS x86_64 binary builds after the 2.2 release line, so a current PyTorch release cannot be installed from an official Intel-macOS wheel. PyTorch 2.2.2 is the latest patch release with a CPython 3.11 Intel-macOS wheel.

NumPy is pinned to 1.26.4 because PyTorch 2.2.2's compiled NumPy bridge is not compatible with NumPy 2.x. The notebook contains a direct `torch.from_numpy` smoke check so this failure is caught immediately.

What this optimizes: a working native installation on the actual machine and stable tensor interchange between NumPy and PyTorch.

Alternative not selected: newest PyTorch. It targets Apple Silicon on current macOS releases and does not publish the required Intel wheel. Full training should eventually move to a CUDA Linux machine with a newer PyTorch build; the local Intel Mac is appropriate for data validation, code plumbing, and small inference smoke tests.

### Decision: pin Transformers 4.57.6

SigLIP2 support entered Transformers after the model's 2025 release. Version 4.57.6 includes the SigLIP2 model and processor while remaining compatible with the older PyTorch runtime required by this Intel Mac.

I selected the mature 4.x line rather than the newest Transformers 5.x line because current Transformers 5.x declares PyTorch 2.5 or newer, which conflicts with the last available Intel-macOS PyTorch wheel.

What this optimizes: access to the SigLIP2 checkpoint through `AutoModel` and `AutoProcessor` without forcing an impossible PyTorch upgrade on this hardware.

The official FixRes checkpoint declares the backwards-compatible `siglip` configuration type, so `AutoModel` correctly instantiates the class named `SiglipModel`. This is still the official `google/siglip2-base-patch16-224` checkpoint; the compatibility class name is expected for FixRes and is not evidence that the older SigLIP weights were loaded.

### Decision: keep a dedicated, fully pinned notebook requirements file

The environment is described by `notebooks/requirements-siglip2.txt`. Direct notebook dependencies are exact-pinned. Transitive dependencies are resolved by pip and can be captured with `pip freeze` if the experiment later needs a byte-for-byte archival lock.

I did not reuse the root `requirements.txt` because it describes the older recommender-system environment (including Python 3.8/3.9-era scientific packages) and has no Transformers or PyTorch dependency.

What this optimizes: separation of concerns and repeatable notebook setup without destabilizing existing project scripts.

### Decision: remove package installation from notebook execution

The notebook now begins with environment verification rather than `%pip install`. Installing or force-reinstalling core packages inside a live kernel can leave old modules loaded next to new files and creates misleading ABI errors until the kernel is restarted.

The one-time shell setup belongs in this journal and the notebook's opening markdown. Normal `Run All` execution is then deterministic and does not mutate its own environment.

What this optimizes: reliable reruns and a clear separation between setup and experiment logic.

### Decision: keep the checkpoint-default processor path

With pinned Transformers 4.57.6, `AutoProcessor.from_pretrained(MODEL_ID)` selects the checkpoint's legacy image processor and bundled fast tokenizer. It prints an informational warning that a later Transformers default may use the fast image processor, but the exact package pin prevents that behavior from changing in this environment.

I tested both forced alternatives. `use_fast=False` also forces the slow Gemma tokenizer and therefore adds SentencePiece. `use_fast=True` adds torchvision, whose eager import fails because this existing pyenv Python was built without the optional `_lzma` standard-library extension. Neither expansion improves the base experiment enough to justify extra host-runtime work.

What this optimizes: the smallest working environment, use of the processor configuration shipped with the checkpoint, and avoidance of dependencies that are unnecessary for the current paired-data baseline.

### Decision: include `nbconvert` as execution tooling

`nbconvert` is included so the notebook can be executed end-to-end from the command line using its registered kernel. This makes validation repeatable and lets automated checks fail on the exact cell that breaks instead of testing copied snippets outside the notebook.

What this optimizes: confidence that the `.ipynb` artifact itself runs in order and records its outputs. It is development tooling, not a model runtime dependency.

## 2026-08-26 — Notebook experiment design

### Decision: start with the official FixRes base checkpoint

The checkpoint is `google/siglip2-base-patch16-224`. It uses fixed 224 × 224 image inputs and is appropriate for a base-level image-text retrieval implementation. The checkpoint is about 1.5 GB, so the notebook warns that first load downloads model assets.

I selected the base FixRes model over NaFlex or a larger checkpoint because the goal is to establish correct local image-description pairing, batching, embeddings, and retrieval metrics before introducing variable-resolution packing or larger compute requirements.

What this optimizes: simplicity, documented preprocessing, and a realistic chance of running inference locally.

### Decision: use the checkpoint processor for both modalities

`AutoProcessor` performs model-specific RGB image resizing and normalization plus text tokenization. Text is truncated and padded to 64 tokens, matching the established SigLIP preprocessing length.

I did not implement custom torchvision transforms or a separate tokenizer. Those alternatives risk drifting from checkpoint preprocessing and add dependencies without improving this baseline.

What this optimizes: correctness relative to pretrained weights and fewer hand-maintained preprocessing assumptions.

### Decision: deterministic 80/10/10 split saved by ASIN

The notebook creates 8,000 training, 1,000 validation, and 1,000 test pairs with seed 42, saves only the `asin,split` mapping, and validates an existing mapping before reuse.

The validation split is used for baseline checks and iteration. The test split stays untouched until final evaluation. ASIN-level separation prevents the exact product ID from crossing splits.

What this optimizes: experiment reproducibility and reduced evaluation leakage.

Known limitation: semantically identical or near-duplicate listings can still cross splits. A future production benchmark should group perceptual-image hashes, product families, or duplicate descriptions before splitting.

### Decision: make training opt-in and inference the default

`RUN_TRAINING` defaults to `False`, and the first fine-tuning run is capped at 512 rows. The base model has 375,187,970 parameters. This Intel Mac exposes PyTorch's MPS backend, which is useful for the small inference smoke test, but full fine-tuning is still not a sensible default on this local environment.

What this optimizes: a safe `Run All` path that validates data, processor tensors, pretrained retrieval, and artifact generation without accidentally launching a very long CPU training job.

For full training, use a CUDA Linux environment, increase the effective batch size, and revisit the PyTorch pin for that platform.

### Decision: use paired bidirectional Recall@K as the first metric

For every held-out image, the CSV description at the same row is treated as the positive among all descriptions in the split; the reverse direction is evaluated too. Recall@1, Recall@5, and Recall@10 provide an easy-to-interpret retrieval plumbing baseline.

I did not treat this as a complete recommendation metric. Home products can have multiple valid semantic matches, so paired Recall@K can label reasonable retrievals as incorrect. Human relevance judgments and recommendation-specific offline/online metrics are future work.

What this optimizes: a fast, deterministic test that the two embedding towers and row alignment work as intended.

## Reproduction commands

Run these commands from the repository root:

```bash
/Users/vachemacbook/.pyenv/versions/3.11.6/bin/python -m venv .venv-siglip2
.venv-siglip2/bin/python -m pip install --upgrade pip
.venv-siglip2/bin/python -m pip install -r notebooks/requirements-siglip2.txt
.venv-siglip2/bin/python -m ipykernel install --user --name recsystem-siglip2 --display-name "RecSystem — SigLIP2 (Python 3.11)"
```

Then select **RecSystem — SigLIP2 (Python 3.11)** as the notebook kernel and run the notebook from top to bottom.

## Installed environment verification

This section is updated after installation and runtime smoke tests.

- Environment path: `.venv-siglip2` in the repository root
- Python: 3.11.6 on macOS 15.4.1 x86_64
- Kernel registration: `recsystem-siglip2`, displayed as **RecSystem — SigLIP2 (Python 3.11)**
- Direct package versions: torch 2.2.2, NumPy 1.26.4, Transformers 4.57.6, pandas 2.2.3, Pillow 11.3.0, matplotlib 3.10.6, scikit-learn 1.7.2, tqdm 4.67.1, ipykernel 6.30.1, ipywidgets 8.1.7, and nbconvert 7.16.6
- Dependency consistency: `pip check` reported no broken requirements
- Data validation: 10,000 rows, 10,000 existing image paths, zero duplicate ASINs, zero empty fields, and deterministic 8,000/1,000/1,000 split counts
- Model cache: official checkpoint revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, approximately 1.4 GB locally
- Model/processor smoke test: passed on MPS with four real manifest pairs; pixel tensors were `(4, 3, 224, 224)`, token tensors were `(4, 64)`, image/text embeddings were `(4, 768)`, pairwise logits were `(4, 4)`, and all logits were finite
- Notebook execution: all 10 code cells completed with no error outputs; full validation, training, final test, and artifact writing remained disabled by design

## 2026-08-26 — Training and retrieval playground

### Decision: create a separate training notebook

The training work lives in `siglip2_product_training_and_retrieval.ipynb` instead of changing the original exploration notebook. The exploration notebook remains a clean pretrained baseline, while the new notebook is allowed to mutate model weights and write artifacts.

What this optimizes: a stable reference point, easier comparison between pretrained and tuned behavior, and less risk that changing a playground control makes the baseline notebook difficult to reproduce.

### Decision: train the two pooling heads and similarity calibration locally

The default `TRAINING_MODE="heads"` freezes both 12-layer transformer backbones. It trains the text projection head, vision attention-pooling head, learned logit scale, and learned logit bias. This is 7,677,698 trainable parameters out of 375,187,970 total, or 2.046%.

I chose this over full-backbone fine-tuning because the local machine has 16 GB of system memory and a 4 GB AMD GPU exposed through MPS. Head tuning still changes the final image and text embeddings, but it avoids optimizer states and gradients for roughly 367 million backbone parameters. The notebook blocks `full` mode unless CUDA is active.

What this optimizes: completing a real backward/optimizer training loop locally with a low out-of-memory risk. The tradeoff is lower adaptation capacity than full fine-tuning.

Alternative not selected: LoRA. It is a strong next option on CUDA, but it adds PEFT configuration and target-module choices before the basic training and retrieval plumbing has been proven. Head tuning is easier to inspect and save as a compact partial state dictionary.

### Decision: use 64 training pairs, 64 validation pairs, and a 250-product test catalog for the first run

The samples use seed 42 and come from the existing 8,000/1,000/1,000 split. Training rows come only from `train`; before/after metrics use `validation`; the searchable catalog uses unseen `test` products.

The small sizes are deliberate smoke-test settings. They prove loss computation, gradients, optimizer updates, held-out evaluation, embedding generation, row alignment, and retrieval visualizations in about a minute on this laptop. They are not presented as a statistically reliable quality experiment.

What this optimizes: fast iteration and leakage-resistant mechanics. The next useful scale change is `CATALOG_ROWS=None` for all 1,000 test products, followed by a larger training run on CUDA.

### Decision: compute both image and description embeddings for each catalog row

The notebook stores one normalized 768-dimensional image vector and one normalized 768-dimensional description vector per catalog product. Both matrices use the exact row order in `embedding_catalog.csv`.

This enables three distinct behaviors without re-encoding the catalog:

- image vector versus image matrix for visually similar products;
- free-text vector versus image matrix for cross-modal product discovery;
- free-text vector versus description matrix for semantic description search.

What this optimizes: flexibility for product discovery experiments and a clear comparison between what visual content retrieves and what product copy retrieves.

### Decision: use exact cosine search for the playground

All vectors are L2-normalized, so cosine similarity is computed by a simple dot product. The current 250–1,000 product search space is small enough for exact PyTorch matrix operations.

Alternative not selected: FAISS or another approximate nearest-neighbor index. Approximate indexing becomes valuable at a much larger catalog size, but it would add complexity without improving latency meaningfully for this sample.

What this optimizes: transparent rankings, no additional dependency, and easy verification of embedding/catalog alignment.

### Decision: save a partial head checkpoint and aligned embedding artifacts

Artifacts live in `models/siglip2_playground/`, which is ignored by Git. `trained_heads.pt` contains only the trainable tensors and is applied over the original Hugging Face checkpoint. The directory also contains `image_embeddings.npy`, `description_embeddings.npy`, `embedding_catalog.csv`, and `experiment.json`.

An inference-only run cannot overwrite `trained_heads.pt` unless training actually ran. This protects the tuned head state while still allowing embeddings to be refreshed.

What this optimizes: a compact 29 MB tuned checkpoint instead of duplicating the roughly 1.4 GB base checkpoint, plus an explicit row contract for reusable vectors.

### Observed first-run results

The first bounded run completed all 13 code cells with no errors.

- Training: one epoch, 64 pairs, batch size 4, learning rate `1e-5`, 14.2 seconds on MPS
- Loss: first batch 3.7578, last batch 1.3860, epoch mean 4.1860
- Validation image-to-text Recall@1: 0.5781 before and 0.6406 after
- Validation text-to-image Recall@1: 0.5625 before and 0.5938 after
- Some Recall@5/10 values moved down while others moved up; with only 64 validation pairs this is expected noise and is a warning not to over-interpret the smoke test
- Catalog: 250 unseen products with image and description matrices of shape `(250, 768)`; mean row norm was exactly 1.0 for both
- Text query `modern wooden coffee table for a living room`: the top cross-modal image result and top description result were both an actual coffee table (`B00869NVRQ`)

The process makes sense end-to-end. The next experiment should increase the held-out catalog to all 1,000 products before spending substantially more compute on training, because retrieval quality is easier to judge in a realistic candidate pool.

## 2026-08-27 — Furniture-only 20,000-product dataset

### Decision: filter with the official category hierarchy

The furniture notebook accepts a record only when its category path begins with `Home & Kitchen > Furniture`. The raw metadata stores this hierarchy in the `categories` field. The broader `main_category` field usually says only `Amazon Home`, so it cannot separate furniture from kitchenware, bath products, linens, or décor.

I selected the strict first-two-category rule over searching titles, descriptions, or category labels for the word "furniture." Keyword matching brought in promotional and mixed categories such as `Furniture & Decor`, while the strict branch describes the intended product taxonomy directly. A complete source scan found 270,345 records in this branch and 173,469 unique products with both a description and a primary image URL.

What this optimizes: category precision and an auditable definition of what counts as furniture.

### Decision: draw a deterministic uniform sample of 20,000 eligible products

`data_exploration_furniture.ipynb` uses streaming reservoir sampling with seed 42. It considers only strict furniture records with a unique ASIN, a non-empty description, and a primary image URL. This gives every eligible product the same probability of selection without loading the 2.8 GB compressed metadata file into memory.

I selected 20,000 rather than the earlier 10,000 because furniture retrieval benefits from a larger variety of product types and styles, while the resulting image collection remains manageable on this machine. I kept uniform sampling instead of forcing equal subcategory sizes so this first furniture dataset reflects the source distribution. A later benchmark can add a stratified evaluation set if smaller categories need equal representation.

What this optimizes: reproducibility, bounded memory use, broader retrieval coverage, and a realistic category mix.

### Decision: retain taxonomy provenance in addition to the requested pair fields

The sample keeps `asin`, `description`, `image_url`, and `image_url_high_res`. It also keeps `furniture_subcategory` and the complete `category_path`. The final training CSV keeps the ASIN, description, subcategory, category path, and local image path.

The extra taxonomy fields are small compared with the descriptions and images, and they make it possible to audit the sample or measure retrieval behavior by furniture group later.

What this optimizes: traceability and future slice-based evaluation without requiring another full metadata scan.

### Decision: isolate furniture images and make downloading resumable

Furniture images live in `data/images_furniture`, separate from the original general Home & Kitchen images. Downloads use 24 bounded worker threads, three attempts per URL, a high-resolution URL fallback, file-signature checks, and atomic `.part` replacement. Existing files are indexed once and reused on a rerun.

I selected a separate flat ASIN-named folder because the ASIN is already unique and is the join key in every table. This avoids mixing experiments while keeping image lookup simple.

What this optimizes: safe resumption, fast joins, clear dataset boundaries, and protection from incomplete files.

The generated image directories and full raw metadata directory are ignored by Git, matching the existing policy for generated CSV data. The notebooks and decision journal remain trackable, while large reproducible artifacts stay local.

### Decision: backfill exhausted downloads to preserve exactly 20,000 usable pairs

One sampled product, `B000VNPWDQ`, returned a permanent HTTP 404 after all retries. The notebook records the failed source URL, removes that product from the final sample, and deterministically scans for an eligible furniture product outside the sampled ASINs. It downloads replacements until the final sample contains exactly 20,000 usable image-description pairs.

I selected backfilling over accepting 19,999 pairs because the requested dataset size should describe the rows that can actually enter training, not merely the URLs attempted. The single replacement has a negligible effect on the otherwise uniform sample, and the original failure remains auditable.

What this optimizes: an exact training-data contract and transparent failure handling.

### Observed furniture dataset results

The furniture notebook completed all eight cells with no error outputs.

- Eligible source pool: 173,469 unique strict-furniture products with descriptions and images
- Final sample: 20,000 rows and 20,000 unique ASINs
- Final image manifest: 20,000 rows
- Final SigLIP2-ready pairs: 20,000 rows
- Taxonomy validation: all 20,000 paths begin with `Home & Kitchen > Furniture`
- Data validation: zero empty descriptions, zero missing local image paths, and zero duplicate ASINs
- Image validation: all 20,000 files are JPEGs; Pillow structural verification found zero invalid images
- Local image storage: approximately 446 MB in `data/images_furniture`
- Download audit: one exhausted original source URL, followed by one successful backfill

The largest groups in the final sampled data are Living Room Furniture (6,949), Bedroom Furniture (4,088), Home Office Furniture (2,335), Dining Room Furniture (2,046), and Game & Recreation Room Furniture (1,963). This broadly follows the source distribution rather than artificially balancing the groups.

## 2026-08-27 — Furniture-specific SigLIP2 training and retrieval

### Decision: create an independent furniture training notebook and artifact directory

The furniture experiment lives in `siglip2_furniture_training_and_retrieval.ipynb`. It reads only `meta_Home_and_Kitchen_furniture_siglip2_pairs_20k.csv` and saves outputs under `models/siglip2_furniture_playground`. The general Home & Kitchen training notebook and its artifacts remain unchanged.

What this optimizes: clean comparison between the broad-domain and furniture-domain experiments, with no risk of accidentally loading or overwriting the wrong trained heads or embedding catalog.

### Decision: use a deterministic subcategory-stratified 80/10/10 split

The notebook creates and then reuses `meta_Home_and_Kitchen_furniture_siglip2_splits_20k.csv`. Seed 42 assigns 16,000 products to training, 2,000 to validation, and 2,000 to test. Splitting is stratified on `furniture_subcategory`, so major and smaller furniture groups retain approximately the same proportions in all three splits.

I selected stratification over a plain random split because the new dataset has an explicit taxonomy and is dominated by living room and bedroom furniture. A random split would usually be acceptable at 20,000 rows, but stratification makes category coverage intentional and reproducible at essentially no added runtime cost.

What this optimizes: stable category representation and fairer held-out evaluation slices.

### Decision: keep head tuning as the local default

The furniture notebook uses the same `heads` method as the earlier playground: both 12-layer transformer backbones are frozen while the text projection head, vision attention-pooling head, logit scale, and logit bias are trained. This is 7,677,698 trainable parameters out of 375,187,970 total, or 2.046%.

I kept the training method constant so changes between experiments can be attributed primarily to the furniture-only data. Full-backbone training remains blocked without CUDA because it is not appropriate for the memory available on this laptop.

What this optimizes: an interpretable domain comparison and a low-risk local training run.

### Decision: keep the first furniture run deliberately bounded

The executed default trains on 64 products, evaluates on 64 validation products, and embeds a 250-product sample from the unseen 2,000-product test split. Batch size is four, the learning rate is `1e-5`, and the run uses one epoch.

This is a system smoke test, not a final quality claim. It verifies data loading, split isolation, contrastive loss, gradients, updated embeddings, retrieval, visualizations, and artifact persistence before committing more compute.

What this optimizes: fast feedback and complete end-to-end validation.

### Decision: keep taxonomy fields in retrieval results and saved catalogs

Image and text search results include the furniture subcategory and complete category path as well as ASIN, score, description, and image path. The saved embedding catalog retains the same fields.

What this optimizes: easy inspection of whether neighbors stay within a sensible furniture family and future evaluation by subcategory.

### Observed furniture smoke-test results

The final notebook execution completed all 27 cells with no error outputs on MPS.

- Training: one epoch, 64 pairs, batch size 4, 17.1 seconds
- Training loss: first batch 2.8229, last batch 1.6854, epoch mean 3.9082
- Image-to-text Recall@1: 0.5469 before and 0.5156 after
- Text-to-image Recall@1: 0.5000 before and 0.5156 after
- Image-to-text Recall@5: 0.6719 before and 0.7188 after
- Text-to-image Recall@5: 0.7031 before and 0.7031 after
- Image-to-text Recall@10: 0.7656 before and 0.7812 after
- Text-to-image Recall@10: 0.7812 before and 0.7656 after
- Embedding catalog: 250 unseen test products with image and description matrices of shape `(250, 768)`
- Embedding validation: all values finite; every row has unit norm within floating-point tolerance; catalog ASINs belong only to the test split
- Saved checkpoint: 15 finite trainable tensors, approximately 29 MB

The small metric movements go in both directions and should not be interpreted as a stable quality improvement from only 64 training pairs. The retrieval behavior is nevertheless coherent: an ottoman query image returned mostly ottomans and storage benches, while the text query `mid-century modern walnut coffee table for a living room` returned coffee tables and closely related living-room tables near the top.

## 2026-08-31 — Taxonomy-enriched furniture descriptions A/B test

### Decision: preserve the original dataset and create a separate enriched variant

The original 20,000-pair CSV remains unchanged. `data_exploration_furniture_taxonomy_enrichment.ipynb` creates `meta_Home_and_Kitchen_furniture_siglip2_pairs_20k_taxonomy_enriched.csv` with both `description_original` and `description_enriched`, plus `taxonomy_text`, the existing taxonomy columns, and the same image path.

What this optimizes: reversible experimentation and exact row-level reconciliation. No result requires reconstructing or overwriting the original text.

### Taxonomy audit before enrichment

The 20,000-product sample contains 142 distinct category paths and 133 distinct leaf labels. After `Home & Kitchen > Furniture`, products have zero to four more taxonomy levels. Most products have two or three varying levels, such as `Living Room Furniture > Tables > Coffee Tables`.

Only 7.8% of descriptions contain the exact normalized leaf label as written, and only 9.7% contain any exact varying hierarchy label. This exact-label check is conservative because singular/plural variants such as `barstool` versus `Barstools` are counted as different, but it confirms that taxonomy often supplies explicit wording not present verbatim in the description.

### Decision: use a compact semicolon hierarchy prefix

The enriched text format is:

`Furniture; Living Room Furniture; Tables; Coffee Tables. <original description>`

The constant `Home & Kitchen` root is omitted from the text because it adds no discrimination inside this furniture-only experiment. `Furniture` and every more-specific level are retained, and the complete original `category_path` remains in its own column.

I selected this over a verbose `Category: ... Description: ...` wrapper. With the actual SigLIP2 tokenizer, the compact format adds a median 11 tokens versus 15 for the verbose form.

What this optimizes: maximum hierarchy signal for a limited text-token budget.

### Identified tradeoff: taxonomy displaces description tokens

The model uses a 64-token text limit. Before enrichment, 14,867 rows, or 74.3%, already exceed that limit. The compact taxonomy prefix raises the truncated share to 80.7%, with 1,268 products, or 6.3%, newly crossing the limit because of the prefix.

This is not automatically harmful: the prepended product type may be more useful than later marketing copy. It is nevertheless a real treatment effect and one reason to evaluate against natural descriptions rather than assuming more text is always better.

### Decision: run a controlled two-model comparison

`siglip2_furniture_taxonomy_ab_comparison.ipynb` trains two fresh models sequentially:

- Model A trains on `description_original`.
- Model B trains on `description_enriched`.

Both runs use the same pretrained checkpoint, seed 42, 64 training ASINs, batch order, images, 64 validation ASINs, 250-product test catalog, optimizer, learning rate, batch size, and one epoch. Each model saves its own 15-tensor head checkpoint and aligned embedding artifacts under `models/siglip2_furniture_taxonomy_ab`.

Primary evaluation uses original natural descriptions for both models. Secondary evaluation uses enriched descriptions and is explicitly labeled as a diagnostic, because real user queries will not necessarily contain the catalog taxonomy.

What this optimizes: isolating the text treatment rather than accidentally comparing different data or initialization.

### Decision: add category-neighbor metrics that exclude the exact pair

Exact-pair Recall@K may miss the behavior taxonomy is intended to improve. The comparison therefore also removes each query's exact paired/self item and checks whether any retrieved neighbor shares its broad furniture subcategory. It reports image-to-text, text-to-image, image-to-image, and text-to-text category-neighbor rates at 1, 5, and 10.

What this optimizes: measuring category consistency separately from exact listing alignment.

### Observed bounded A/B results

Both notebooks completed without error outputs. The enriched CSV has 20,000 unique ASINs and reconciles exactly with the original descriptions and image paths. Both saved embedding pairs have shape `(250, 768)`, finite values, unit-normalized rows, and test-only catalog ASINs. Both checkpoints contain 15 finite tensors.

On the primary original-description evaluation, Model B versus Model A produced:

- Image-to-text Recall@1: 0.5156 versus 0.5156
- Text-to-image Recall@1: 0.4844 versus 0.5000
- Image-to-text Recall@5: 0.7188 versus 0.7188
- Text-to-image Recall@5: 0.7031 versus 0.7031
- Image-to-text Recall@10: 0.7969 versus 0.7656
- Text-to-image Recall@10: 0.7812 versus 0.7812

Category-neighbor differences were also small. The largest positive change was +0.8 percentage points for image-to-text same-subcategory@1; the largest negative change was -1.2 points for image-to-text same-subcategory@10. The natural coffee-table query returned the same ten products for both models, in the same order at the top of the ranking.

The taxonomy model's epoch mean loss was 4.5907 versus 3.9311 for the original model. Loss values are not directly a model-selection metric here, but the taxonomy treatment was not obviously easier to optimize in this tiny run.

### Current conclusion and next trustworthy experiment

The hypothesis is sensible, but this 64-pair, one-seed smoke test does not demonstrate a meaningful improvement. Results are effectively tied with small movements in both directions. That is a successful experiment outcome: the data and A/B machinery work, and there is no basis yet for replacing natural descriptions.

The next useful test should use at least 512–2,000 training pairs, the full 2,000-product test catalog, and multiple seeds. It should keep original-description evaluation primary and compare the current full hierarchy against a shorter leaf-focused prefix such as `Coffee Tables. <description>`. That would test whether the most specific category provides the benefit without consuming as much of the 64-token budget.

## 2026-08-31 — Furniture long-text strategy benchmark

### Decision: compare five text strategies under one controlled harness

I created `siglip2_furniture_text_strategy_benchmark.ipynb` and trained five fresh models from the same `google/siglip2-base-patch16-224` checkpoint:

- `baseline64`: ordinary right truncation at the model's native 64-token limit.
- `compact64`: a compact taxonomy prefix plus sentence selection weighted toward rare, visually meaningful furniture terms.
- `multichunk64`: up to four overlapping native-length chunks, a different chunk used in each training epoch, and normalized mean pooling across chunks at retrieval time.
- `dual64`: separate description and taxonomy losses during training, followed by a weighted description/taxonomy embedding fusion.
- `extended128`: interpolation of the learned 64-position text table to 128 positions, with the new position table trainable.

Every arm used seed 42, the same 256 training ASINs, 128 held-out exact-pair examples, 500-product retrieval catalog, two epochs, batch size 4, learning rate `1e-5`, image augmentations, and trainable projection heads. Only the text representation changed.

What this optimizes: a fair, laptop-sized directional screen in which differences can be attributed mainly to the text strategy.

### Decision: select on retrieval behavior rather than training loss

The balanced selection uses eight views: mean natural exact Recall@1 and Recall@10, mean strategy-native catalog exact Recall@1 and Recall@10, category-query precision@10 against images and descriptions, image-neighbor category accuracy@1, and cross-modal category accuracy@1. Each strategy is ranked on every view, and the mean rank determines the screening winner.

What this optimizes: both requested use cases—similar-image discovery and free-text product search—without allowing one easy or noisy metric to dominate the decision.

### Observed screening results

`multichunk64` ranked first with a mean rank of 1.875. The next strategies were `dual64` at 2.750, `compact64` at 2.8125, `baseline64` at 2.9375, and `extended128` at 4.625.

The strongest `multichunk64` results were:

- Category query to image precision@10: 0.1611, best of the five arms.
- Category query to description precision@10: 0.1500, best of the five arms.
- Cross-modal same-subcategory@1: 0.6800, best of the five arms.
- Mean natural exact-pair Recall@10: 0.7812, tied with `compact64` for best.

It did not win every metric. Its mean natural exact Recall@1 was 0.4531 versus 0.4648 for `compact64`, and image-neighbor category@1 was 0.694 versus 0.700 for `baseline64`. The selection is therefore based on breadth of performance, especially the stronger broad text-query behavior, rather than claiming universal dominance.

The direct 128-token extension finished cleanly but ranked last. Its new positional parameters had only 256 training examples to adapt, and merely exposing more product copy did not make that copy more useful. This result argues against changing the pretrained architecture at this stage.

All five saved catalog embedding pairs have shape `(500, 768)`, contain only finite values, and have unit-normalized rows. All five saved checkpoints contain only finite tensors. The executed notebook completed without errors after making tokenizer attention masks optional, because this SigLIP2 tokenizer does not emit one.

### Working choice: promote multi-chunk 64, with a confirmation gate

The current working choice is `multichunk64`. It preserves the pretrained model's native 64-position architecture, sees information beyond the opening window, and avoids permanently discarding later attributes. Its cost is extra text-encoding work—up to four text forward passes per catalog item—and a slightly weaker Recall@1 result in this run.

What this optimizes: better use of long furniture descriptions without retraining a new text architecture or relying on a brittle hand-written summary.

This remains a directional screen, not final proof. Before replacing the main furniture pipeline, the confirmation run should use the full 2,000-product test catalog and at least three seeds. If the lead survives, the next full training notebook should use rotated chunks during training and precompute mean-pooled chunk embeddings once for the catalog.
