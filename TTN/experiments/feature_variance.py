"""Do the input features discriminate BETWEEN items inside one category?

No model, no training -- this is a property of the data. If a grey table and a
walnut table have near-identical inputs, nothing downstream can separate them.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/lazr/PycharmProjects/RecSystem"); OUT = ROOT / "data/tower"
vocabs = json.load(open(OUT / "vocabs.json"))
arrays = np.load(OUT / "items.npz")
node_of_item = np.load(OUT / "node_of_item.npy")
CAT_ORDER = ["cat_2","cat_3","cat_4","brand","color","material","product_type","features"]
title = arrays["title_emb"].astype(np.float32)
title /= np.linalg.norm(title, axis=1, keepdims=True).clip(1e-9)
ci = arrays["cat_ids"]; price = np.expm1(arrays["numeric"][:, 0])
rng = np.random.default_rng(0)
inv = {k: {i: v for v, i in d.items()} for k, d in vocabs.items()}

sizes = pd.Series(node_of_item).value_counts()
nodes = sizes[sizes >= 400].index.to_numpy()[:12]
S = 300                                   # items sampled per node

rows = []
for nd in nodes:
    m = np.flatnonzero(node_of_item == nd)
    take = rng.choice(m, min(S, len(m)), replace=False)
    T = title[take]
    C = T @ T.T
    off = C[~np.eye(len(C), dtype=bool)]
    row = {"category": str(inv["target_node"].get(int(nd), "?")).split(" > ")[-1][:24],
           "items": len(m), "title cos": off.mean()}
    for f in ("brand", "color", "material", "product_type"):
        v = ci[m, CAT_ORDER.index(f)]
        p = np.bincount(v)[np.bincount(v) > 0] / len(v)
        row[f"{f} distinct"] = int(len(np.unique(v)))
        row[f"{f} entropy"] = float(-(p * np.log2(p)).sum())
    pm = price[m]; pm = pm[np.isfinite(pm) & (pm > 0)]
    row["price CV"] = float(pm.std() / pm.mean()) if len(pm) > 2 else np.nan
    rows.append(row)

d = pd.DataFrame(rows)
show = ["category","items","title cos","brand distinct","brand entropy",
        "color distinct","color entropy","material entropy","price CV"]
print("WITHIN each category -- how different are items from each other?\n")
print(d[show].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

r = rng.choice(len(title), 3000, replace=False)
Cr = title[r] @ title[r].T
print(f"\nreference: mean title cosine between two RANDOM items anywhere: "
      f"{Cr[~np.eye(len(Cr), dtype=bool)].mean():.3f}")
print(f"           mean title cosine WITHIN a category:                 {d['title cos'].mean():.3f}")
print(f"\nmax possible entropy at {S} sampled items is {np.log2(S):.1f} bits;")
print("an entropy near 0 would mean every item in the category shares that value.")
