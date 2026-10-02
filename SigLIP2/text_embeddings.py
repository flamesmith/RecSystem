"""Generate SigLIP2 TEXT embeddings — the piece that didn't exist anywhere
in this repo before. `SigLIP2/build_siglip2.py` only ever calls
`model.get_image_features()`; this is the equivalent for
`model.get_text_features()`, ported from the `SigLIP2` branch's
`SigLIP2_training/src/siglip2_training/features.py::_encode_texts`
(commit 43327cad712038579d06ac53996f3c8133b953e9) — same tokenization
(`padding="max_length"`, `truncation=True`, 64 tokens — matching
`description_v1.yaml`'s `chunking.maximum_tokens`) and the same
L2-normalization `build_siglip2.py` already applies to image embeddings.

Uses the SAME `google/siglip2-base-patch16-224` checkpoint
`build_siglip2.py` already downloads — no new model, just the text half
of the same dual encoder that script never calls.

Usage:
    from transformers import AutoModel, AutoTokenizer
    from SigLIP2.description_pipeline import DescriptionProcessor, ProductInput, load_yaml
    from SigLIP2.text_embeddings import encode_texts, embed_product_text

    tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").eval()
    processor = DescriptionProcessor(load_yaml(), tokenizer=tokenizer)

    product = ProductInput(product_id="...", title="...", descriptions=["..."], categories=["...", "..."])
    raw_text_embedding = embed_product_text(model, tokenizer, processor, product)   # (768,)

    # Then, optionally, apply the trained adapter:
    from SigLIP2.adapter import load_adapter, adapt_text
    adapted = adapt_text(load_adapter(), raw_text_embedding)
"""
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from SigLIP2.description_pipeline import DescriptionProcessor, ProductInput

MAX_TOKENS = 64  # must match description_v1.yaml's chunking.maximum_tokens -- the
                  # adapter was trained on embeddings produced at this exact length


def _batched(values: Sequence, size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


@torch.inference_mode()
def encode_texts(model, tokenizer, texts: Sequence[str], device: str = "cpu",
                  batch_size: int = 64, max_length: int = MAX_TOKENS) -> np.ndarray:
    """Raw SigLIP2 text embeddings, L2-normalized -- the text-side equivalent
    of build_siglip2.py's image encoding. One row per input string."""
    parts = []
    for batch in _batched(list(texts), batch_size):
        inputs = tokenizer(
            text=batch, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        out = model.get_text_features(**inputs)
        embeddings = getattr(out, "pooler_output", out)   # this transformers version wraps it
        embeddings = F.normalize(embeddings, dim=-1)
        parts.append(embeddings.cpu().numpy())
    return np.concatenate(parts, axis=0).astype("float32")


def embed_product_text(model, tokenizer, processor: DescriptionProcessor,
                        product: ProductInput, view: str = "description_canonical",
                        device: str = "cpu") -> np.ndarray:
    """Full chain for one product: raw fields -> the exact deterministic text
    view the adapter was trained on -> SigLIP2 text embedding (768,).

    `view` selects which of the four text fields `DescriptionProcessor`
    produces to embed: "description_clean", "description_canonical" (the
    default -- compact, taxonomy+attributes+top sentences), "taxonomy_text",
    or "description_chunks" (a tuple of up to 4 strings -- embed each
    separately and pool yourself if you want the multichunk view the
    promoted checkpoint actually used).
    """
    result = processor.transform(product)
    text = getattr(result, view)
    if view == "description_chunks":
        if not text:
            return np.zeros(model.config.text_config.hidden_size, dtype="float32")
        return encode_texts(model, tokenizer, list(text), device=device)
    if not text:
        return np.zeros(model.config.text_config.hidden_size, dtype="float32")
    return encode_texts(model, tokenizer, [text], device=device)[0]
