"""Does the same-category rate of also_buy depend on WHAT the product is?

Hypothesis under test: big-ticket items (a sofa) get cross-category complements,
small cheap items (a rug, an accessory) get same-category substitutes.
"""
import ast, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/Users/lazr/PycharmProjects/RecSystem")
meta = pd.read_csv(ROOT/"data/meta_Home_and_Kitchen_filtered.csv",
                   usecols=["asin","also_buy","brand","category","price","title"],
                   dtype={"asin":str,"also_buy":str,"brand":str,"category":str,"price":str},
                   low_memory=False).drop_duplicates("asin")

def as_list(v):
    if not isinstance(v,str) or v in ("","[]"): return []
    try: p = ast.literal_eval(v)
    except (ValueError,SyntaxError): return []
    return p if isinstance(p,list) else []

path = meta.category.apply(as_list)
meta["c3"] = path.apply(lambda p: p[2] if len(p) > 2 else None)
meta["c4"] = path.apply(lambda p: p[3] if len(p) > 3 else None)
def money(v):
    if not isinstance(v,str): return np.nan
    m = re.match(r"^\$?([\d,]+\.?\d*)$", v.strip())
    return float(m.group(1).replace(",","")) if m else np.nan
meta["p"] = meta.price.apply(money)

sub = meta[meta.also_buy.fillna("[]").ne("[]")].copy()
sub["targets"] = sub.also_buy.apply(as_list)
e = sub[["asin","c3","c4","brand","p","targets"]].explode("targets").dropna(subset=["targets"])
e = e.rename(columns={"asin":"src","targets":"tgt","c3":"s3","c4":"s4","brand":"sb","p":"sp"})
info = meta.set_index("asin")[["c3","c4","brand","p"]].rename(
    columns={"c3":"t3","c4":"t4","brand":"tb","p":"tp"})
e = e.join(info, on="tgt")
e = e[e.t4.notna() & e.s4.notna()]
e["same4"] = (e.s4 == e.t4)
e["same3"] = (e.s3 == e.t3)
e["sameb"] = e.sb.notna() & (e.sb == e.tb)
print(f"edges scored: {len(e):,}\n")

g = (e.groupby("s4").agg(edges=("same4","size"), same_cat4=("same4","mean"),
                         same_cat3=("same3","mean"), same_brand=("sameb","mean"),
                         median_price=("sp","median"))
     .loc[lambda d: d.edges >= 3000].sort_values("same_cat4"))
pd.set_option("display.width", None)
print("MOST cross-category source categories (complement-like):")
print(g.head(12).to_string(float_format=lambda x: f"{x:,.3f}"))
print("\nMOST same-category source categories (substitute-like):")
print(g.tail(12).to_string(float_format=lambda x: f"{x:,.3f}"))

print("\n\nBy source price decile:")
q = e[e.sp.notna() & (e.sp > 0)].copy()
q["band"] = pd.qcut(q.sp, 10, labels=False, duplicates="drop")
pb = q.groupby("band").agg(edges=("same4","size"), median_price=("sp","median"),
                           same_cat4=("same4","mean"), same_brand=("sameb","mean"))
print(pb.to_string(float_format=lambda x: f"{x:,.3f}"))
print(f"\ncorrelation(log source price, same_cat4): "
      f"{np.corrcoef(np.log1p(q.sp), q.same4.astype(float))[0,1]:+.3f}")

print("\n\nNamed lookups:")
for pat in ("Sofa", "Area Rug", "Mattress", "Dinner Plate", "Nightstand", "Throw Pillow"):
    m = g.index.str.contains(pat, case=False, na=False)
    for name in g.index[m][:2]:
        r = g.loc[name]
        print(f"  {name[:34]:<34} edges {r.edges:>7,.0f}  same_cat4 {r.same_cat4:.3f}  "
              f"same_brand {r.same_brand:.3f}  median $ {r.median_price:,.2f}")
