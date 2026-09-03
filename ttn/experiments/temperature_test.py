"""Temperature ALONE on the unchanged cell. Nothing else touched.

Every temperature config in the earlier sweep was bundled with mined negatives,
which were independently doing the damage. This isolates it.

  mode=overfit : 5,000 pairs inside ONE category, 150 epochs. The node embedding
                 is constant, so this is pure within-category ranking against a
                 known ceiling (0.0964 = what a single fixed top-10 would score).
  mode=full    : the ordinary 3-epoch run on all 2.2M pairs.
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

TAU, mode = float(sys.argv[1]), sys.argv[2]
nb = json.load(open(ROOT / "ttn/ttn_complementary.ipynb"))
src = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])
DIV = "".join(next(c for c in nb["cells"] if c.get("id") == "diversity-audit")["source"])

def sub1(s, old, new, label):
    assert s.count(old) == 1, f"anchor {label}: {s.count(old)}"
    return s.replace(old, new)

# --- the only change: divide both score terms by TAU -----------------------
src = sub1(src, "SEED = 0", f"SEED = 0\nTAU = {TAU}", "tau const")
src = sub1(src, "    scores = q @ c_pos.T", "    scores = (q @ c_pos.T) / TAU", "easy scores")
src = sub1(src, "    hard = -F.logsigmoid(pos.squeeze(1) - (q * c_hard).sum(-1)).mean()",
           "    hard = -F.logsigmoid(pos.squeeze(1) - (q * c_hard).sum(-1) / TAU).mean()", "hard score")
src = sub1(src, 'CHECKPOINT = OUT_DIR / "ttn_complementary.pt"',
           f'CHECKPOINT = Path("{S}") / "temp.pt"', "checkpoint")
src = sub1(src, "{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path")

rng = np.random.default_rng(0)
if mode == "overfit":
    big = full_train.target_node_id.value_counts().idxmax()
    pool = full_train[full_train.target_node_id == big]
    sub = pool.iloc[rng.choice(len(pool), min(5000, len(pool)), replace=False)].reset_index(drop=True)
    cap = sum(g.target_idx.isin(g.target_idx.value_counts().index[:10]).sum()
              for _, g in sub.groupby("target_node_id")) / len(sub)
    src = sub1(src, "BATCH, EPOCHS, LR = 4096, 3, 3e-3", "BATCH, EPOCHS, LR = 512, 150, 3e-3", "hyper")
    src = sub1(src, '    m = evaluate(model, pairs_test, n_eval=5000)\n    print(f"epoch {epoch}',
               '    m = evaluate(model, pairs_train, n_eval=5000)\n    print(f"epoch {epoch}', "curve")
    print(f"\n{'='*70}\n### TAU = {TAU}  (overfit: 5,000 pairs, one category)\n{'='*70}")
    print(f"CEILING for a single fixed top-10 = {cap:.4f}   (curve below is TRAIN recall)", flush=True)
else:
    sub = full_train
    print(f"\n{'='*70}\n### TAU = {TAU}  (full 2.2M pairs, 3 epochs)\n{'='*70}", flush=True)

g = {k: v for k, v in globals().items() if not k.startswith("__")}
g["pairs_train"] = sub; g["Path"] = Path
exec(compile("from pathlib import Path\n" + src, f"tau{TAU}", "exec"), g)
if mode == "full":
    g["catalogue"] = g["catalogue_vectors"](g["model"])
    exec(compile(DIV, "diversity", "exec"), g)
