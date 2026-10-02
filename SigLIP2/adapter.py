"""Load and call the trained SigLIP2 image/text adapter.

This is a self-contained port of just the model architecture and loading
logic needed to use `adapter_checkpoint/taxonomy_w20_random_200k.pt` --
not the full training pipeline that produced it.

Origin: the `SigLIP2` branch, `SigLIP2_training/src/siglip2_training/
training.py` (`_torch_modules`), at commit 43327cad712038579d06ac53996f3c
8133b953e9. That branch trains the adapter from scratch over a 200K-product
pilot; this file only loads the already-trained result. See
`adapter_checkpoint/README.md` for the checkpoint's own provenance, and
`../SigLIP2/README.md`'s "Trained adapter" section for what this is and
its known limitations (pilot-scale, no precomputed catalogue index here).

What the adapter does: SigLIP2's frozen base embeddings (768-d, the same
ones `build_siglip2.py` already produces for images) are *not* trained for
this catalogue's text -- the adapter is two small residual MLP bottlenecks
(one for images, one for text) learned on top of those frozen embeddings,
pulling matching image/text pairs closer in the shared space. Both base
embeddings must come from the same SigLIP2 checkpoint this was trained
against: `google/siglip2-base-patch16-224`.

Usage:
    from SigLIP2.adapter import load_adapter, adapt_image, adapt_text

    model = load_adapter()                    # loads the default checkpoint
    adapted = adapt_image(model, raw_image_embeddings)   # (N, 768) -> (N, 768)
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "adapter_checkpoint" / "taxonomy_w20_random_200k.pt"


class ResidualAdapter(nn.Module):
    """A small bottleneck MLP added as a residual, then re-normalized.

    `up`'s weights are zero-initialized, so an untrained adapter is the
    identity function -- training only ever learns a *correction* to the
    frozen base embedding, never replaces it outright.
    """

    def __init__(self, dimension: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, values):
        residual = self.up(self.dropout(self.activation(self.down(values))))
        return F.normalize(values + residual, dim=-1)


class ContrastiveAdapters(nn.Module):
    """Two independent `ResidualAdapter`s (image, text) trained jointly
    with a CLIP-style contrastive loss. `image_adapter` and `text_adapter`
    can each be called on their own -- there's no cross-dependency between
    them at inference time, only during training."""

    def __init__(
        self,
        dimension: int,
        bottleneck: int,
        dropout: float,
        logit_scale: float,
        logit_bias: float,
        maximum_logit_scale: float,
    ) -> None:
        super().__init__()
        self.image_adapter = ResidualAdapter(dimension, bottleneck, dropout)
        self.text_adapter = ResidualAdapter(dimension, bottleneck, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(max(logit_scale, 1e-6))))
        self.logit_bias = nn.Parameter(torch.tensor(logit_bias))
        self.maximum_logit_scale = maximum_logit_scale

    def embeddings(self, images, texts):
        return self.image_adapter(images), self.text_adapter(texts)


def load_adapter(checkpoint_path=DEFAULT_CHECKPOINT, device: str = "cpu") -> ContrastiveAdapters:
    """Reconstruct the adapter and load the trained weights. Returns it in
    eval() mode -- dropout is a training-only regularizer, never applied here."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter_cfg = checkpoint["training_config"]["adapter"]
    model = ContrastiveAdapters(
        dimension=checkpoint["dimension"],
        bottleneck=adapter_cfg["bottleneck_dimension"],
        dropout=adapter_cfg["dropout"],
        logit_scale=1.0,   # placeholder -- overwritten by load_state_dict below
        logit_bias=0.0,    # placeholder -- overwritten by load_state_dict below
        maximum_logit_scale=adapter_cfg["maximum_logit_scale"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def _to_tensor(embeddings, device):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings).float()
    return embeddings.to(device)


@torch.inference_mode()
def adapt_image(model: ContrastiveAdapters, embeddings) -> np.ndarray:
    """Apply the trained image-side adapter to raw frozen SigLIP2 image
    embeddings (shape (N, 768) or (768,)) -- e.g. straight out of
    `siglip_img_emb.npy`, or a fresh `model.get_image_features(...)` call
    on a never-before-seen image via the same SigLIP2 checkpoint."""
    device = next(model.parameters()).device
    values = _to_tensor(embeddings, device)
    single = values.ndim == 1
    if single:
        values = values.unsqueeze(0)
    out = model.image_adapter(values).cpu().numpy()
    return out[0] if single else out


@torch.inference_mode()
def adapt_text(model: ContrastiveAdapters, embeddings) -> np.ndarray:
    """Apply the trained text-side adapter to raw frozen SigLIP2 TEXT
    embeddings (shape (N, 768) or (768,)) -- note this is SigLIP2's own
    text tower (`model.get_text_features(...)`), not SBERT. Nothing in
    this repo currently produces SigLIP2 text embeddings; you'd need to
    add that call yourself, using the same google/siglip2-base-patch16-224
    checkpoint and its tokenizer."""
    device = next(model.parameters()).device
    values = _to_tensor(embeddings, device)
    single = values.ndim == 1
    if single:
        values = values.unsqueeze(0)
    out = model.text_adapter(values).cpu().numpy()
    return out[0] if single else out
