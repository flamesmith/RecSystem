"""Parse the sweep log into a comparison table. Re-run any time; the log grows."""
import re, sys
from pathlib import Path

LOG = Path(__file__).with_name("negative_sampling_sweep.log")
rows, name, base = [], None, {}
for line in LOG.read_text().splitlines():
    if line.startswith("### "):
        name = line[4:].strip()
    elif line.startswith("test ") and name:
        g = dict(re.findall(r"(\w+@?\d*(?:_lenient)?|MedianRank)\s+([\d.,]+)", line))
        rows.append((name, g))
        name = None
    elif line.startswith("baseline  popularity within category R@10  "):
        base["pop"] = line.split()[-1]

hdr = ["Recall@10", "Recall@100", "MedianRank", "Recall@10_lenient"]
print(f"{'config':<62} " + "  ".join(f"{h:>17}" for h in hdr))
print("-" * 62 + "  " + "  ".join("-" * 17 for _ in hdr))
print(f"{'your original run (commit 9280a29)':<62} " +
      "  ".join(f"{v:>17}" for v in ["0.1138", "0.4645", "119", "0.2418"]))
for n, g in rows:
    print(f"{n[:62]:<62} " + "  ".join(f"{g.get(h, '-'):>17}" for h in hdr))
if base.get("pop"):
    print(f"{'in-node popularity baseline (no model)':<62} " +
          "  ".join(f"{v:>17}" for v in [base['pop'], "-", "-", "0.4034"]))
