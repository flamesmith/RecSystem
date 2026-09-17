# Popularity — baseline / fallback for "Complete the Look"

**This is the Popularity folder of the `v1-recommendations` structure.**
Per this branch's scope (reorganizing existing artifacts, not writing new
serving code), this folder is documentation only — it does not duplicate or
reimplement the popularity-scoring logic, which already exists elsewhere in
the repo. See below for exactly where.

## Its role

Throughout this project's evaluation (`../TTN/ttn_complementary.ipynb` §13
onward, and every script in `../TTN/experiments/` and `../analysis/`),
popularity is the baseline TTN has to beat, and the two stay close in
aggregate — but TTN's advantage is not uniform. This session's analysis
(notebook §19–21) found a clear split by how often the *target* item was
seen in training:

- **Below ~25 observations**: TTN is at or below random chance — in one
  measured case (Bedding, target frequency 1–5) TTN scored **R@10 = 0.0000**
  across 225 pairs. Popularity is not much better there either (it is 0 by
  construction for never-seen targets), but it never actively underperforms
  random the way TTN can.
- **Above ~25 observations**: TTN clearly wins, up to ~24x random chance.

So popularity's role as "Complete the Look"'s fallback is specifically for
the cold-target case: when a candidate item has too little co-purchase
history for TTN to have learned anything useful about it, popularity within
its category is a safer default than trusting TTN's output. See
`../analysis/frequency_routed_retrieval*.py` for three tested ways to
combine the two (none beat plain TTN in aggregate; the reserved-slots variant
is the one that improves cold-target recall without making it worse than
plain TTN — see that folder's README for the numbers and the tradeoff).

## Where the existing implementation lives

Not duplicated here. The canonical definition — "popularity within the
category": the target's membership in its category's top-10 (or top-100)
most frequent training targets — is:

- `../TTN/ttn_complementary.ipynb`'s `model-code` cell, in the `baselines()`
  function (`"popularity within the category"` / `"... (lenient)"`).
- Recomputed identically (same `groupby(["target_node_id", "target_idx"])
  .size()` → `nlargest` pattern) in nearly every script under
  `../TTN/experiments/` and `../analysis/` that reports a `POP` column,
  rather than imported from one shared module — it was never factored out
  into its own reusable function anywhere in this repo.

## An open gap, not yet resolved

Every popularity number in this repo, including all of this session's
tables, uses **all-time training-window popularity** — a target's raw count
across the full training pairs table, no time decay. That is a different
definition from the "popular in the past 60 days" carousel mentioned
alongside this one in planning — nothing in this repo currently computes a
time-windowed popularity baseline. Building that (and deciding whether it
should also become "Complete the Look"'s fallback, replacing the all-time
version used above) is out of scope for this branch's reorganization.
