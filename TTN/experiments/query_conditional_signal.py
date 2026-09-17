"""Does knowing the QUERY buy anything over per-category popularity? No model."""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
S = Path(sys.argv[1]); ROOT = Path("/Users/lazr/PycharmProjects/RecSystem"); OUT = ROOT/"data/tower"
arrays = np.load(OUT/"items.npz")
_ld = lambda n: pd.DataFrame({k: v for k, v in np.load(S/f"{n}.npz").items()})
tr, te_all = _ld("pairs_train"), _ld("pairs_test")
CO = ["cat_2","cat_3","cat_4","brand","color","material","product_type","features"]
ci = arrays["cat_ids"]; col = lambda f: ci[:, CO.index(f)].astype(np.int64)

rng = np.random.default_rng(0)
te = te_all.iloc[rng.choice(len(te_all), 20000, replace=False)].reset_index(drop=True)
relevant = defaultdict(set)
for a,b,c in te_all[["query_idx","target_node_id","target_idx"]].itertuples(index=False):
    relevant[(int(a),int(b))].add(int(c))

tgt = te.target_idx.to_numpy(); qte = te.query_idx.to_numpy(); ndte = te.target_node_id.to_numpy().astype(np.int64)
qtr = tr.query_idx.to_numpy(); ndtr = tr.target_node_id.to_numpy().astype(np.int64); ttr = tr.target_idx.to_numpy()

def top10_by(key_tr):
    d = pd.DataFrame({"k": key_tr, "t": ttr})
    g = d.groupby(["k", "t"]).size().rename("n").reset_index()
    g = g.sort_values(["k", "n"], ascending=[True, False])
    g["r"] = g.groupby("k").cumcount()
    g = g[g.r < 10]
    sizes = g.groupby("k").size()                     # index is sorted by k
    chunks = np.split(g.t.to_numpy(), np.cumsum(sizes.to_numpy())[:-1])
    top = dict(zip(sizes.index.to_numpy(), chunks))   # one key per GROUP, not per row
    obs = d.groupby("k").size()
    return top, dict(zip(obs.index.to_numpy(), obs.to_numpy()))


back, _ = top10_by(ndtr)

def score(key_tr, key_te, min_obs=5):
    top, obs = top10_by(key_tr)
    hit = np.zeros(len(te), bool); leni = np.zeros(len(te), bool); used = 0
    for i in range(len(te)):
        k = key_te[i]
        c = top.get(k) if obs.get(k, 0) >= min_obs else None
        used += c is not None
        if c is None: c = back.get(int(ndte[i]), np.array([], np.int64))
        s = set(c.tolist())
        hit[i] = int(tgt[i]) in s
        leni[i] = bool(s & relevant[(int(qte[i]), int(ndte[i]))])
    return hit.mean(), leni.mean(), used/len(te)

M = np.int64(200003)
specs = [("target category only (the baseline)", ndtr, ndte)]
for f in ("cat_3", "cat_4", "brand", "product_type"):
    a = col(f)
    specs.append((f"+ query {f}", ndtr*M + a[qtr], ndte*M + a[qte]))
specs.append(("+ the exact query item (memorisation)", ndtr*M + qtr.astype(np.int64), ndte*M + qte.astype(np.int64)))

print(f"{'top-10 built from train counts, grouped by':<44} {'R@10':>7} {'lenient':>8} {'group used':>11}", flush=True)
for name, a, b in specs:
    h, l, u = score(a, b)
    print(f"{name:<44} {h:>7.4f} {l:>8.4f} {u:>10.1%}", flush=True)
