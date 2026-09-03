"""Run the notebook's cell 23 with one knob changed at a time."""
import json, os, sys
from pathlib import Path
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
BASE = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])
BASE = BASE.replace('CHECKPOINT = OUT_DIR / "ttn_complementary.pt"', f'CHECKPOINT = Path("{S}") / "sweep.pt"')
BASE = BASE.replace("{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}")
BASE = BASE.replace("from pathlib import Path\n", "")
BASE = "from pathlib import Path\n" + BASE

def run(name, **kw):
    src = BASE
    for k, v in kw.items():
        src = re.sub(rf"^{k}(\s*)=(\s*)[^#\n]+", lambda m: f"{k}{m.group(1)}={m.group(2)}{v!r} ", src, count=1, flags=re.M)
    if kw.get("N_MINED") == 0:
        src = src.replace("""negs = torch.cat([mine(model, q_i, n_i, cache, N_MINED),
                          uniform_in_node(n_i, N_UNIFORM)], dim=1)""",
                          "negs = uniform_in_node(n_i, N_UNIFORM)")
    if kw.get("LAYERNORM") is False:
        src = src.replace("item = self.item_norm(self.query_encoder(idx))", "item = self.query_encoder(idx)")
        src = src.replace("self.node_norm(node)", "node")
    src = re.sub(r"^LAYERNORM.*\n", "", src, flags=re.M)
    g = {k: v for k, v in globals().items() if not k.startswith("_")}
    print(f"\n{'='*78}\n### {name}\n{'='*78}", flush=True)
    exec(compile(src, name, "exec"), g)

import re
CFG = sys.argv[1]
if CFG == "1": run("1. reproduce the original: 1 uniform negative, no temperature",
                   N_MINED=0, N_UNIFORM=1, POOL=8, TAU=1.0, W_EASY=0.5, NODE_DROPOUT=0.0, LAYERNORM=False)
if CFG == "2": run("2. + mined negatives only (no temperature, no LN, no dropout)",
                   N_MINED=4, N_UNIFORM=4, TAU=1.0, W_EASY=0.5, NODE_DROPOUT=0.0, LAYERNORM=False)
if CFG == "3": run("3. + temperature 0.2",
                   N_MINED=4, N_UNIFORM=4, TAU=0.2, W_EASY=0.5, NODE_DROPOUT=0.0, LAYERNORM=False)
if CFG == "4": run("4. + temperature 0.05",
                   N_MINED=4, N_UNIFORM=4, TAU=0.05, W_EASY=0.5, NODE_DROPOUT=0.0, LAYERNORM=False)
if CFG == "5": run("5. temperature 0.2 + LayerNorm, no node dropout",
                   N_MINED=4, N_UNIFORM=4, TAU=0.2, W_EASY=0.5, NODE_DROPOUT=0.0, LAYERNORM=True)
if CFG == "6": run("6. temperature 0.2 + LayerNorm + node dropout 0.3",
                   N_MINED=4, N_UNIFORM=4, TAU=0.2, W_EASY=0.5, NODE_DROPOUT=0.3, LAYERNORM=True)
