"""Can the model fit a SMALL training set it sees many times?

Train/test parity on 2.2M pairs cannot separate "too little capacity" from
"architecturally unable to answer per-query", because 2.2M mappings are too many
to memorise at this size either way. Shrinking the data removes that ambiguity.

Two subsets:
  A  20,000 pairs drawn from everywhere
  B   5,000 pairs from ONE target category, so the node embedding is constant
      across every row and cannot carry any information at all

The ceiling matters as much as the curve. If the query tower collapses to one
direction per category, the best it can do is emit a single top-10 per node --
so its recall is capped by how much of the subset those 10 items cover. That
number is computed below, in advance.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd

S = Path("/private/tmp/claude-501/-Users-lazr-PycharmProjects-RecSystem/4d057b13-3828-4b13-bed8-0ded1333e1f6/scratchpad")
ROOT = Path("/Users/lazr/PycharmProjects/RecSystem")
OUT_DIR = ROOT / "data/tower"
vocabs = json.load(open(OUT_DIR / "vocabs.json"))
arrays = np.load(OUT_DIR / "items.npz")
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
_ld = lambda n: pd.DataFrame({k: v for k, v in np.load(S / f"{n}.npz").items()})
full_train, pairs_test = _ld("pairs_train"), _ld("pairs_test")
CAT_ORDER = ["cat_2","cat_3","cat_4","brand","color","material","product_type","features"]
rng = np.random.default_rng(0)

variant = sys.argv[1]
if variant == "A":
    sub = full_train.iloc[rng.choice(len(full_train), 20_000, replace=False)].reset_index(drop=True)
    title = "A: 20,000 pairs from everywhere"
else:
    big = full_train.target_node_id.value_counts().idxmax()
    pool = full_train[full_train.target_node_id == big]
    sub = pool.iloc[rng.choice(len(pool), min(5000, len(pool)), replace=False)].reset_index(drop=True)
    name = [k for k, v in vocabs["target_node"].items() if v == int(big)]
    title = f"B: 5,000 pairs inside ONE category -- {name[0] if name else big}"

# --- the ceiling for a model that emits one fixed list per category --------
cap = 0
for nd, g in sub.groupby("target_node_id"):
    top10 = g.target_idx.value_counts().index[:10]
    cap += g.target_idx.isin(top10).sum()
ceiling = cap / len(sub)
print(f"\n{'='*78}\n### {title}\n{'='*78}")
print(f"pairs {len(sub):,}   distinct queries {sub.query_idx.nunique():,}   "
      f"distinct targets {sub.target_idx.nunique():,}   categories {sub.target_node_id.nunique():,}")
print(f"CEILING if the model emits one fixed top-10 per category: R@10 = {ceiling:.4f}")
print(f"(a model that genuinely reads the query should blow past this)\n", flush=True)

nb = json.load(open(ROOT / "ttn/ttn_complementary.ipynb"))
src = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])
def sub_once(s, old, new, label):
    assert s.count(old) == 1, f"anchor {label}: {s.count(old)}"
    return s.replace(old, new)
src = sub_once(src, "BATCH, EPOCHS, LR = 4096, 3, 3e-3",
               "BATCH, EPOCHS, LR = 512, 150, 3e-3", "hyperparams")
# the printed curve should be TRAIN recall -- that is the whole point
src = sub_once(src, "    m = evaluate(model, pairs_test, n_eval=5000)\n    print(f\"epoch {epoch}",
               "    m = evaluate(model, pairs_train, n_eval=5000)\n    print(f\"epoch {epoch}", "curve")
src = sub_once(src, 'CHECKPOINT = OUT_DIR / "ttn_complementary.pt"',
               f'CHECKPOINT = Path("{S}") / "overfit.pt"', "checkpoint")
src = sub_once(src, "{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path")

g = {k: v for k, v in globals().items() if not k.startswith("__")}
g["pairs_train"] = sub
g["Path"] = Path
exec(compile("from pathlib import Path\n" + src, f"overfit{variant}", "exec"), g)
print(f"\nCEILING was {ceiling:.4f}; final train R@10 above is the answer.")
