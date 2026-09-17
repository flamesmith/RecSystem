"""Does adding a SigLIP image block to the ProductEncoder help the TTN model?

Patches `model-code` straight out of `ttn/ttn_complementary.ipynb` (same
mechanism as ttn/experiments/temperature_test.py -- string-substitution on the
committed cell source, asserted to match exactly once, so this cannot drift
from what the notebook actually trains) to add one more per-block
Linear -> LayerNorm(affine=False) branch, mirroring how `desc_emb` was added:
zero-fill missing image vectors, same concat-then-MLP treatment.

Run on a SAMPLE of the training pairs, not the full 2.2M -- one A/B pair
(image block on vs off), same subsample and seed both times, so the only
thing that differs is the architecture. Checkpoints go to a scratch path;
the committed data/tower/ttn_complementary.pt is never touched.

Usage: python ttn/experiments/image_in_model.py [train_frac] [epochs]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
EPOCHS_OVERRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else None

ROOT = Path("/Users/lazr/PycharmProjects/RecSystem")
OUT_DIR = ROOT / "data" / "tower"
SCRATCH = Path("/private/tmp/claude-501/-Users-lazr-PycharmProjects-RecSystem/"
               "be4f830b-9fa8-49ab-9da9-6eb8a3a7f9c2/scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)

vocabs = json.load(open(OUT_DIR / "vocabs.json"))
arrays_base = dict(np.load(OUT_DIR / "items.npz"))
node_of_item = np.load(OUT_DIR / "node_of_item.npy")
img_emb = np.load(OUT_DIR / "siglip_img_emb.npy").astype("float32")
assert img_emb.shape[0] == arrays_base["title_emb"].shape[0]

full_train = pd.read_parquet(OUT_DIR / "pairs_train.parquet")
pairs_test = pd.read_parquet(OUT_DIR / "pairs_test.parquet")
CAT_ORDER = ["cat_2", "cat_3", "cat_4", "brand", "color", "material", "product_type", "features"]

sub = (full_train.sample(frac=TRAIN_FRAC, random_state=0).reset_index(drop=True)
       if TRAIN_FRAC < 1.0 else full_train)
print(f"training sample: {len(sub):,} of {len(full_train):,} pairs "
      f"({TRAIN_FRAC:.0%}) | test: {len(pairs_test):,}")

nb = json.load(open(ROOT / "ttn/ttn_complementary.ipynb"))
BASE_SRC = "".join(next(c for c in nb["cells"] if c.get("id") == "model-code")["source"])


def sub1(s, old, new, label):
    n = s.count(old)
    assert n == 1, f"anchor {label!r} matched {n} times, expected 1"
    return s.replace(old, new)


def build_source(use_image, tag):
    src = BASE_SRC
    if EPOCHS_OVERRIDE is not None:
        src = sub1(src, "BATCH, EPOCHS, LR = 4096, 30, 3e-3",
                   f"BATCH, EPOCHS, LR = 4096, {EPOCHS_OVERRIDE}, 3e-3", "epochs")
    src = sub1(
        src,
        'print(f"description block: {\'on\' if USE_DESCRIPTION else \'OFF — no desc_emb in items.npz\'}")\n'
        'cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)',
        'print(f"description block: {\'on\' if USE_DESCRIPTION else \'OFF — no desc_emb in items.npz\'}")\n'
        'USE_IMAGE = "img_emb" in arrays\n'
        'img_t = (torch.tensor(arrays["img_emb"], device=DEVICE) if USE_IMAGE\n'
        '         else torch.zeros(len(arrays["title_emb"]), 1, device=DEVICE))\n'
        'print(f"image block: {\'on\' if USE_IMAGE else \'OFF — no img_emb in items.npz\'}")\n'
        'cat_t = torch.tensor(arrays["cat_ids"], device=DEVICE)',
        "feature tensors")
    src = sub1(
        src,
        "        self.description = nn.Linear(desc_t.shape[1], 128) if USE_DESCRIPTION else None\n"
        "        self.numeric = nn.Linear(num_t.shape[1], 16)",
        "        self.description = nn.Linear(desc_t.shape[1], 128) if USE_DESCRIPTION else None\n"
        "        self.image = nn.Linear(img_t.shape[1], 128) if USE_IMAGE else None\n"
        "        self.numeric = nn.Linear(num_t.shape[1], 16)",
        "__init__ linear")
    src = sub1(
        src,
        "        self.norm_description = (nn.LayerNorm(128, elementwise_affine=False)\n"
        "                                 if USE_DESCRIPTION else None)\n"
        "        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)",
        "        self.norm_description = (nn.LayerNorm(128, elementwise_affine=False)\n"
        "                                 if USE_DESCRIPTION else None)\n"
        "        self.norm_image = (nn.LayerNorm(128, elementwise_affine=False)\n"
        "                           if USE_IMAGE else None)\n"
        "        self.norm_numeric = nn.LayerNorm(16, elementwise_affine=False)",
        "__init__ layernorm")
    src = sub1(
        src,
        "            nn.Linear(len(CAT_ORDER) * CAT_DIM + 128\n"
        "                      + (128 if USE_DESCRIPTION else 0) + 16, HIDDEN),",
        "            nn.Linear(len(CAT_ORDER) * CAT_DIM + 128\n"
        "                      + (128 if USE_DESCRIPTION else 0)\n"
        "                      + (128 if USE_IMAGE else 0) + 16, HIDDEN),",
        "mlp input dim")
    src = sub1(
        src,
        "        if self.description is not None:\n"
        "            parts.append(self.norm_description(self.description(desc_t[idx])))\n"
        "        parts.append(self.norm_numeric(self.numeric(num_t[idx])))",
        "        if self.description is not None:\n"
        "            parts.append(self.norm_description(self.description(desc_t[idx])))\n"
        "        if self.image is not None:\n"
        "            parts.append(self.norm_image(self.image(img_t[idx])))\n"
        "        parts.append(self.norm_numeric(self.numeric(num_t[idx])))",
        "forward")
    src = sub1(src, '"use_description": USE_DESCRIPTION,',
               '"use_description": USE_DESCRIPTION,\n'
               '               "use_image": USE_IMAGE,', "config")
    src = sub1(src, 'CHECKPOINT = OUT_DIR / "ttn_complementary.pt"',
               f'CHECKPOINT = Path("{SCRATCH}") / "ttn_image_{tag}.pt"', "checkpoint")
    src = sub1(src, "{CHECKPOINT.relative_to(ROOT)}", "{CHECKPOINT}", "path")
    return src


def run(use_image):
    tag = "with_img" if use_image else "baseline"
    arrays = dict(arrays_base)
    if use_image:
        arrays["img_emb"] = img_emb
    src = build_source(use_image, tag)
    g = {k: v for k, v in globals().items() if not k.startswith("__")}
    g.update({"pairs_train": sub, "pairs_test": pairs_test, "arrays": arrays,
              "vocabs": vocabs, "node_of_item": node_of_item, "OUT_DIR": OUT_DIR,
              "ROOT": ROOT, "Path": Path, "CAT_ORDER": CAT_ORDER})
    print(f"\n{'=' * 70}\n### {tag}  (image block {'ON' if use_image else 'OFF'})\n{'=' * 70}", flush=True)
    t0 = time.time()
    exec(compile("from pathlib import Path\n" + src, f"<{tag}>", "exec"), g)
    print(f"[{tag}] wall time {time.time() - t0:.0f}s")
    return g["metrics"], g["best"]["epoch"]


if __name__ == "__main__":
    results = {}
    for use_image in (False, True):
        metrics, best_epoch = run(use_image)
        results[use_image] = (metrics, best_epoch)

    print(f"\n{'=' * 70}\nSUMMARY  (train sample {TRAIN_FRAC:.0%} = {len(sub):,} pairs)\n{'=' * 70}")
    print(f"{'':<12}{'best_epoch':>12}{'train R@10':>14}{'test R@10':>14}"
          f"{'test R@100':>14}{'test R@10_len':>16}")
    for use_image, (metrics, best_epoch) in results.items():
        tag = "with_img" if use_image else "baseline"
        print(f"{tag:<12}{best_epoch:>12}{metrics['train']['Recall@10']:>14.4f}"
              f"{metrics['test']['Recall@10']:>14.4f}{metrics['test']['Recall@100']:>14.4f}"
              f"{metrics['test'].get('Recall@10_lenient', float('nan')):>16.4f}")
