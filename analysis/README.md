# Cross-model analysis

Scripts that score TTN, SigLIP2/CONTENT, and Popularity **together**, to
answer questions no single carousel folder can answer on its own — which one
wins for a given item, and whether combining them beats any one alone. None
of these belong to just `../TTN/`, `../SigLIP2/`, or `../Popularity/`, which
is why they're a fourth, sibling folder rather than filed under one of the
three.

All reads/write `data/tower/` artifacts (gitignored, built by
`../TTN/ttn_complementary.ipynb` §1–9 and `../SigLIP2/SigLip embeddings.ipynb`)
and the committed TTN checkpoints (`ttn_complementary.pt`,
`ttn_complementary_image.pt`) read-only — nothing here retrains TTN.

| file | question it answers | headline |
| --- | --- | --- |
| `reproduce_section19.py` | does the notebook's §19 table (TTN / DESC / POP / CONTENT by target frequency) still reproduce? | TTN/POP/DESC close; CONTENT (desc+colour+material, §19's original definition) ~30% lower than originally reported |
| `content_desc_title_features.py` | redefine CONTENT as desc+title+features instead of desc+colour+material — does it do better? | yes, beats desc+colour+material in every bucket (R@10 all: 0.0688 vs 0.0559) |
| `image_content_similarity.py` | does adding SigLIP2 image similarity to CONTENT help? | yes, in every bucket; `IMAGE` alone (pure SigLIP2 cosine — see `../SigLIP2/README.md`) also beats `DESC` alone |
| `final_capacity_comparison_similarity.py` | §13's never-executed "~20k pairs" test, similarity half — DESC+TITLE+FEAT(+IMG) and POP, matched to the TTN-training 20k-pair sample (`../TTN/experiments/final_capacity_comparison_ttn.py`) | see `final_capacity_comparison.log` (combined run, before the TTN/similarity split) |
| `image_augmented_similarity.py` | add image to CONTENT (desc+title+features+image, 0.25 each) — does it help further? | yes, in every bucket (R@10 all: 0.0775 vs 0.0688); see `image_augmented_models.log` (combined run, before the TTN/similarity split) |
| `query_side_breakdown.py` | does the *query* item's own training frequency matter, the way the *target*'s does? | barely — TTN's spread across query-frequency buckets is ~1.5x, against ~45x across target-frequency buckets. Cold start is a target-side problem, not a query-side one |
| `furniture_bedding_breakdown.py` | drills the target- and query-frequency breakdowns into just the Furniture and Bedding categories | same crossover holds inside each category; Bedding's target-frequency 1–5 bucket scores **TTN R@10 = 0.0000** (225 pairs) |
| `frequency_routed_retrieval.py` | route each candidate to TTN or CONTENT by its own target frequency, slots split **proportional** to category pool composition, threshold swept | loses to plain TTN at every threshold — the count-based split starves TTN's slot budget even in categories where the true answer is warm |
| `frequency_routed_retrieval_ttn_first.py` | same idea, but TTN claims all slots first and CONTENT only backfills if the category's warm pool itself runs short | fixes the aggregate but makes cold-bucket recall **worse** than plain TTN — backfill essentially never fires for a query whose own answer is cold, only when the whole category is thin |
| `frequency_routed_retrieval_reserved.py` | fixes that: CONTENT always gets a fixed slot reservation (20% tested), regardless of category composition | cold buckets clearly improve over plain TTN; aggregate recall drops a little at every threshold — a real tradeoff, not a free win |

## The tradeoff, in one place

None of the three routing variants beat plain TTN on blended Recall@10/@100.
That is expected, not a bug: ~63% of test pairs already have a popular
correct answer, where TTN is dramatically better than random (up to 24x) —
there is no way to give CONTENT real influence on cold targets without
costing some of that majority case. Whether the tradeoff is worth it depends
on the product goal (pure aggregate recall says no; cold/long-tail item
exposure says yes) — see `../Popularity/README.md` for how this plugs into
"Complete the Look"'s fallback design, and the notebook's §19–21 for the full
write-up.
