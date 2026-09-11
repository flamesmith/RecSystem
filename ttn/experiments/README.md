# Experiments behind the TTN two-tower model

Raw logs and the scripts that produced them, kept so the dead ends do not get
re-run. Every script reads the artifacts in `data/tower/` and, where it needs
the model, the cell source straight out of `ttn/ttn_complementary.ipynb`'s
`model-code` cell -- so none of them can drift away from what the notebook
actually trains.

This is the `ttn_model` branch: it carries only the experiments that train or
inspect the two-tower model itself. The no-training similarity baselines
(DESC / CONTENT / POP) live on the `similarity_model` branch instead --
`final_capacity_comparison_ttn.py` and `image_augmented_ttn.py` here are each
one half of a script that used to do both (`final_capacity_comparison.py`,
`image_augmented_models.py`); the other half
(`final_capacity_comparison_similarity.py`, `image_augmented_similarity.py`)
is over there. Neither split script has been re-run yet, so there's no log of
its own -- `final_capacity_comparison.log` and `image_augmented_models.log`
are the original combined runs (same computation, same numbers, just also
carrying the similarity columns this branch dropped).

| file | question it answers | headline |
| --- | --- | --- |
| `negative_sampling_sweep.py` | do negatives mined from the model's own in-node top-K help? | no: R@10 0.1240 → 0.0636; temperature and LayerNorm do not rescue it |
| `why_mining_fails.py` | why not? | mined negatives sit at popularity percentile 0.836; the true targets sit at 0.832 |
| `negatives_count_sweep.py` | does the *number* of uniform in-node negatives matter? | no: 1/4/16/64 gives 0.1191/0.1159/0.1143/0.1205, diversity flat at 0.042–0.044 |
| `remove_target_category.py` | is the query tower better off without the node embedding? | no: R@10 0.0037, median rank 4,058, only 7.2% of the top-10 in the asked-for category |
| `feature_variance.py` | do item features discriminate *within* a category? | yes: title cosine 0.390 within a category vs 0.187 across; 142–533 brands per category |
| `query_conditional_signal.py` | does conditioning on the query beat category popularity? | flat (0.2147 → 0.2039) -- sparsity-limited, see notebook §13 |
| `brand_signal.py` | is the query-conditional signal in `also_buy` learnable? | yes: 36.8% of raw edges are same-brand at 161x chance |
| `same_category_by_product.py` | does the same-category rate depend on the product? | enormously: 0.136 (Knife Sets) to 0.961 (Incense); price effect is real but small (r = −0.123) |
| `overfit_capacity_test.py` | can the model fit 5,000 pairs it sees 150 times? | no -- R@10 0.0398 against a 0.0964 ceiling for a *constant*; loss fell while recall reversed |
| `temperature_test.py` | is score compression the cause? | yes -- TAU 0.1 fixes it, see `temperature_*` logs |
| `image_in_model.py` | does adding a SigLIP image block to the ProductEncoder help, A/B on a subsample? | see `image_in_model.log` |
| `final_capacity_comparison_ttn.py` | §13's never-executed "~20k pairs, 50 epochs" test -- TTN and TTN+IMAGE trained on a 20k-pair sample | see `final_capacity_comparison.log` (combined run) |
| `image_augmented_ttn.py` | TTN+IMAGE trained on the FULL training set, same regime as the committed checkpoint | R@10 all: 0.2180 → 0.2245 with image; see `image_augmented_models.log` (combined run). Saves `data/tower/ttn_complementary_image.pt`, separate from the committed `ttn_complementary.pt` |

`parse_sweep.py` renders `negative_sampling_sweep.log` as a comparison table.

Scripts load pair tables from a scratch `.npz`; point `_ld` at
`data/tower/pairs_{train,test}.parquet` to run them against the repo directly.

## The one that worked

`overfit_capacity_test.py` showed the model *can* memorise 5,000 pairs seen
150 times -- but recall got worse while loss fell, because score compression
made a badly-ordered pair cost the same as a well-ordered one. `TAU = 0.1` on
the BPR scores fixed it: 30 epochs with early stopping reaches R@10 0.2174,
above the popularity baseline (0.2136). Diversity did **not** improve
(distinct share 0.048 against 0.044) -- temperature fixed ranking, not
collapse.
