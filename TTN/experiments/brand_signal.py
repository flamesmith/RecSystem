"""Is the query-conditional signal in also_buy LEARNABLE from features?

Raw also_buy is item-specific (jaccard 0.005). But a model can only use
specificity that its features can express. Same-brand is the obvious candidate:
brand IS a feature. So how much of the signal is brand-shaped, and how much
survives the TTN pipeline's filtering?
"""
import ast, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/lazr/PycharmProjects/RecSystem"); OUT = ROOT / "data/tower"
S = Path("/private/tmp/claude-501/-Users-lazr-PycharmProjects-RecSystem/4d057b13-3828-4b13-bed8-0ded1333e1f6/scratchpad")

# ---- 1. The TTN's own filtered pairs -------------------------------------
arrays = np.load(OUT / "items.npz"); vocabs = json.load(open(OUT / "vocabs.json"))
node_of_item = np.load(OUT / "node_of_item.npy")
CAT_ORDER = ["cat_2","cat_3","cat_4","brand","color","material","product_type","features"]
ci = arrays["cat_ids"]
brand = ci[:, CAT_ORDER.index("brand")]; c4 = ci[:, CAT_ORDER.index("cat_4")]
other = vocabs["brand"].get("other_brands")
_ld = lambda n: pd.DataFrame({k: v for k, v in np.load(S / f"{n}.npz").items()})
tr = _ld("pairs_train")
q, t = tr.query_idx.to_numpy(), tr.target_idx.to_numpy()
real = (brand[q] != other) & (brand[t] != other)      # 'other_brands' is a fold, not a brand
print("--- TTN filtered pairs (what the model trains on) ---")
print(f"pairs: {len(q):,}   with a real brand on both sides: {real.mean():.1%}")
print(f"same brand : {(brand[q][real] == brand[t][real]).mean():.3%}")
print(f"same cat_4 : {(c4[q] == c4[t]).mean():.3%}")
print(f"same node  : {(node_of_item[q] == node_of_item[t]).mean():.3%}")
# chance rate: brand match if targets were drawn at random within the node
rng = np.random.default_rng(0)
_order = np.argsort(node_of_item, kind="stable")
_starts = np.searchsorted(node_of_item[_order], np.arange(node_of_item.max()+2))
samp = rng.integers(0, 1 << 30, len(q)) % np.maximum(
    (_starts[node_of_item[t]+1] - _starts[node_of_item[t]]), 1)
rand_t = _order[np.minimum(_starts[node_of_item[t]] + samp, len(_order)-1)]
rr = real & (brand[rand_t] != other)
print(f"chance same-brand (random item in the same target node): "
      f"{(brand[q][rr] == brand[rand_t][rr]).mean():.3%}")

# ---- 2. Raw also_buy, before any of the pipeline --------------------------
print("\n--- raw also_buy from the metadata ---")
meta = pd.read_csv(ROOT/"data/meta_Home_and_Kitchen_filtered.csv",
                   usecols=["asin","also_buy","brand","category"],
                   dtype={"asin":str,"also_buy":str,"brand":str,"category":str},
                   low_memory=False).drop_duplicates("asin")
def as_list(v):
    if not isinstance(v,str) or v in ("","[]"): return []
    try: p = ast.literal_eval(v)
    except (ValueError,SyntaxError): return []
    return p if isinstance(p,list) else []
def cat4(v):
    p = as_list(v)
    return p[3] if len(p) > 3 else None
sub = meta[meta.also_buy.fillna("[]").ne("[]")].copy()
sub["targets"] = sub.also_buy.apply(as_list)
e = sub[["asin","brand","category","targets"]].explode("targets").dropna(subset=["targets"])
e = e.rename(columns={"asin":"src","targets":"tgt","brand":"src_brand","category":"src_cat"})
info = meta.set_index("asin")[["brand","category"]]
e = e.join(info.rename(columns={"brand":"tgt_brand","category":"tgt_cat"}), on="tgt")
e = e[e.tgt_brand.notna()]
bn = e.src_brand.notna() & e.tgt_brand.notna() & (e.src_brand.str.lower() != "")
print(f"edges with brand on both sides: {bn.sum():,}")
print(f"same brand : {(e.src_brand[bn] == e.tgt_brand[bn]).mean():.3%}")
sc, tc = e.src_cat.map(cat4), e.tgt_cat.map(cat4)
ok = sc.notna() & tc.notna()
print(f"same cat_4 : {(sc[ok] == tc[ok]).mean():.3%}")
pb = e.tgt_brand[bn].value_counts(normalize=True)
print(f"chance same-brand (brand marginal): {(pb**2).sum():.3%}")
