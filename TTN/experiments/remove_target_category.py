"""Does the query tower do better WITHOUT the target-category embedding?

Two runs, identical but for one line: `self.node(node_id)` is zeroed, so the
query tower cannot see which category is being asked for. Everything else --
seed, epochs, negatives, loss -- is whatever the notebook cell currently says.
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
pairs_train, pairs_test = _ld("pairs_train"), _ld("pairs_test")
CAT_ORDER = ["cat_2","cat_3","cat_4","brand","color","material","product_type","features"]

nb = json.load(open(ROOT / "ttn/ttn_complementary.ipynb"))
CELL = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])
DIV  = "".join(next(c for c in nb["cells"] if c.get("id") == "diversity-audit")["source"])

EXTRA = '''
# Does the top-10 still land in the category that was asked for?
_rng = np.random.default_rng(0)
_sub = pairs_test.iloc[_rng.choice(len(pairs_test), 5000, replace=False)]
_q = _sub["query_idx"].to_numpy(); _n = _sub["target_node_id"].to_numpy()
_tops = top_k_batch(_q, _n)
print(f"\\nshare of top-10 landing in the asked-for category: "
      f"{float(np.mean(node_of_item[_tops] == _n[:, None])):.3f}")
print(f"distinct items over those 5,000 queries x 10: {len(np.unique(_tops)):,} of 50,000 slots")
'''

def patched(drop_node):
    s = CELL
    def sub(old, new, label):
        nonlocal s
        assert s.count(old) == 1, f"anchor {label}: {s.count(old)} matches"
        s = s.replace(old, new)
    if drop_node:
        sub("h = torch.cat([self.query_encoder(idx), self.node(node_id)], dim=-1)",
            "h = torch.cat([self.query_encoder(idx),\n"
            "                       torch.zeros_like(self.node(node_id))], dim=-1)",
            "drop node")
    sub('CHECKPOINT = OUT_DIR / "ttn_complementary.pt"',
        f'CHECKPOINT = Path("{S}") / "nonode.pt"', "checkpoint")
    sub("{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path print")
    return s

label = sys.argv[1]
drop = label == "nonode"
print(f"\n{'='*78}\n### {'WITHOUT' if drop else 'WITH'} the target-category embedding\n{'='*78}", flush=True)
g = {k: v for k, v in globals().items() if not k.startswith("__")}
g["Path"] = Path
exec(compile("from pathlib import Path\n" + patched(drop), label, "exec"), g)
g["catalogue"] = g["catalogue_vectors"](g["model"])
print(f"\n--- diversity audit ---", flush=True)
exec(compile(DIV + EXTRA, "diversity", "exec"), g)
