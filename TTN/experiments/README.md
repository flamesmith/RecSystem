# Experiments behind the TTN model ("Complete the Look")

Raw logs and the scripts that produced them. Kept so the dead ends do not get
re-run. Every script reads the artifacts in `data/tower/` and, where it needs the
model, the cell source straight out of `../ttn_complementary.ipynb` — so none
of them can drift away from what the notebook actually says.

This folder holds only experiments that train or inspect the TTN model
itself. Scripts that score TTN against SigLIP2/CONTENT and Popularity
together — the frequency-routing experiments, the Furniture/Bedding and
query-frequency breakdowns, the `final_capacity_comparison_ttn.py` /
`image_augmented_ttn.py` files' similarity-scoring counterparts — live in
[`../../analysis/`](../../analysis/) instead, since they don't belong to just
this carousel. `final_capacity_comparison.log` and `image_augmented_models.log`
are duplicated in both folders: the original combined runs, before those two
scripts were split into a TTN half (here) and an analysis half.

| file | question it answers | headline |
| --- | --- | --- |
| `negative_sampling_sweep.*` | do negatives mined from the model's own in-node top-K help? | no: R@10 0.1240 → 0.0636; temperature and LayerNorm do not rescue it |
| `why_mining_fails.py` | why not? | mined negatives sit at popularity percentile 0.836; the true targets sit at 0.832 |
| `all_six_changes.log` | the first attempt, six changes at once | R@10 0.0477 — the run that prompted the ablation |
| `negatives_count_sweep.*` | does the *number* of uniform in-node negatives matter? | no: 1/4/16/64 gives 0.1191/0.1159/0.1143/0.1205, diversity flat at 0.042–0.044 |
| `remove_target_category.*` | is the query tower better off without the node embedding? | no: R@10 0.0037, median rank 4,058, only 7.2% of the top-10 in the asked-for category |
| `feature_variance.*` | do item features discriminate *within* a category? | yes: title cosine 0.390 within a category vs 0.187 across; 142–533 brands per category |
| `query_conditional_signal.*` | does conditioning on the query beat category popularity? | flat (0.2147 → 0.2039) — but the test is sparsity-limited, see §13 |
| `brand_signal.*` | is the query-conditional signal in `also_buy` learnable? | yes: 36.8% of raw edges are same-brand at 161x chance; §19 filtering cuts it to 9.2% |
| `same_category_by_product.*` | does the same-category rate depend on the product? | enormously: 0.136 (Knife Sets) to 0.961 (Incense); price effect is real but small (r = −0.123) |

`parse_sweep.py` renders `negative_sampling_sweep.log` as a comparison table.

Scripts load pair tables from a scratch `.npz`; point `_ld` at
`data/tower/pairs_{train,test}.parquet` to run them against the repo directly.

## The one that worked

| file | question | headline |
| --- | --- | --- |
| `overfit_capacity_test.*` | can the model fit 5,000 pairs it sees 150 times? | no — R@10 0.0398 against a 0.0964 ceiling for a *constant*; loss fell while recall reversed |
| `temperature_overfit_sweep.log` | is score compression the cause? | yes — same setup, TAU 0.1 reaches 0.5158, a 13x gain, inverted-U with 0.05 over-sharpening |
| `temperature_full_sweep.log` | does it transfer to 2.2M pairs? | yes — R@10 0.1218 (TAU 1.0) → 0.1916 (TAU 0.1) at 3 epochs |
| `temperature_final_run.log` | 30 epochs with early stopping | best epoch 9: **R@10 0.2174, lenient 0.4115**, both above the popularity baseline (0.2136 / 0.4034) |

Diversity did **not** improve (distinct share 0.048 against 0.044). Temperature
fixed ranking, not collapse.

## Adding SigLIP2 image embeddings to TTN

| file | question it answers | headline |
| --- | --- | --- |
| `image_in_model.py` | does adding a SigLIP image block to the ProductEncoder help, A/B on a subsample? | see `image_in_model.log` |
| `final_capacity_comparison_ttn.py` | §13's never-executed "~20k pairs, 50 epochs" test — TTN and TTN+IMAGE trained on a 20k-pair sample | see `final_capacity_comparison.log` (combined run, includes the similarity columns now split out to `analysis/`) |
| `image_augmented_ttn.py` | TTN+IMAGE trained on the FULL training set, same regime as the committed checkpoint | R@10 all buckets: 0.2180 → 0.2245 with image; see `image_augmented_models.log` (combined run). Saves `data/tower/ttn_complementary_image.pt`, separate from the committed `ttn_complementary.pt` |
