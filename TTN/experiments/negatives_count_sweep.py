"""One knob: how many uniform in-node negatives per row. Nothing else changes.

Source of truth is the notebook cell itself, patched in three places so the
single hard negative becomes N of them.
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

def patched(n_uniform):
    s = CELL
    def sub(old, new, label):
        nonlocal s
        assert s.count(old) == 1, f"anchor {label}: {s.count(old)} matches"
        s = s.replace(old, new)

    sub("""def hard_negatives(node_ids):
    lo, hi = starts_t[node_ids], starts_t[node_ids + 1]
    size = (hi - lo).clamp_min(1)
    offset = (torch.rand(len(node_ids), device=DEVICE) * size).long().clamp(max=size - 1)
    return order_t[(lo + offset).clamp(max=len(order_t) - 1)]""",
f"""N_UNIFORM = {n_uniform}


def hard_negatives(node_ids, n=N_UNIFORM):
    lo, hi = starts_t[node_ids].unsqueeze(1), starts_t[node_ids + 1].unsqueeze(1)
    size = (hi - lo).clamp_min(1)
    offset = (torch.rand(len(node_ids), n, device=DEVICE) * size).long().clamp(max=size - 1)
    return order_t[(lo + offset).clamp(max=len(order_t) - 1)]""", "hard_negatives")

    sub("    hard = -F.logsigmoid(pos.squeeze(1) - (q * c_hard).sum(-1)).mean()",
        '    hard = -F.logsigmoid(pos - torch.einsum("bd,bnd->bn", q, c_hard)).mean()',
        "hard term")

    sub("""        loss = bpr_loss(q, model.candidate(c_all[batch]),
                        model.candidate(hard_negatives(n_all[batch])))""",
"""        _negs = hard_negatives(n_all[batch])
        _nv = model.candidate(_negs.reshape(-1)).reshape(len(batch), _negs.shape[1], OUT_DIM)
        loss = bpr_loss(q, model.candidate(c_all[batch]), _nv)""", "call site")

    sub('CHECKPOINT = OUT_DIR / "ttn_complementary.pt"',
        f'CHECKPOINT = Path("{S}") / "uniform_sweep.pt"', "checkpoint")
    sub("{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path print")
    return s

N = int(sys.argv[1])
print(f"\n{'='*78}\n### {N} uniform in-node negative(s) per row\n{'='*78}", flush=True)
g = {k: v for k, v in globals().items() if not k.startswith("__")}
g["Path"] = Path
exec(compile("from pathlib import Path\n" + patched(N), f"n{N}", "exec"), g)
g["catalogue"] = g["catalogue_vectors"](g["model"])
g["DEVICE"] = g["DEVICE"]
print(f"\n--- diversity audit, {N} negatives ---", flush=True)
exec(compile(DIV, "diversity", "exec"), g)
