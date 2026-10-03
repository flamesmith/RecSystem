"""Deterministic description preprocessing — ported from the `SigLIP2`
branch's `SigLIP2_training/src/siglip2_training/{schema,config,text}.py`.

This is the exact text transformation the trained adapter
(`checkpoints/pilot_200k/taxonomy_w20_random_200k.pt`) was trained against —
faithfully ported (not reimplemented from scratch) so results here match
what that checkpoint expects. Consolidated from 3 small files into 1,
consistent with `adapter.py`'s "self-contained, not the whole training
package" scope. Logic is unchanged from the source; only the package-
relative imports were flattened.

Origin: `SigLIP2` branch, commit 43327cad712038579d06ac53996f3c8133b953e9.
Config: `description_v1.yaml` in this same folder, copied verbatim from
that branch's `SigLIP2_training/configs/description_v1.yaml`.

Usage:
    from SigLIP2.description_pipeline import DescriptionProcessor, ProductInput, load_yaml

    config = load_yaml("SigLIP2/description_v1.yaml")
    processor = DescriptionProcessor(config, tokenizer=siglip2_tokenizer)
    result = processor.transform(ProductInput(
        product_id="B000...", title="...", descriptions=["..."],
        features=["..."], details={"Brand": "...", ...}, categories=["...", "..."],
    ))
    # result.description_clean, .description_canonical, .taxonomy_text,
    # .description_chunks are the four text views the adapter was trained on.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "description_v1.yaml"


# ============================================================================
# config.py (ported)
# ============================================================================
def load_yaml(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


# ============================================================================
# schema.py (ported)
# ============================================================================
@dataclass(frozen=True)
class ProductInput:
    product_id: str
    title: str = ""
    descriptions: Sequence[str] = field(default_factory=tuple)
    features: Sequence[str] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)
    categories: Sequence[str] = field(default_factory=tuple)
    image_url: str | None = None


@dataclass(frozen=True)
class TransformResult:
    product_id: str
    title: str
    image_url: str | None
    category_path: str
    taxonomy_text: str
    description_clean: str
    description_canonical: str
    description_chunks: tuple[str, ...]
    pipeline_version: str
    config_hash: str
    source_hash: str
    cleaning_stats: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["description_chunks"] = list(self.description_chunks)
        value["cleaning_stats"] = dict(self.cleaning_stats)
        return value


# ============================================================================
# text.py (ported, logic unchanged)
# ============================================================================
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s*;\s*|\n+")
_FINGERPRINT_CHARACTERS = re.compile(r"[^a-z0-9]+")
_MEASUREMENT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:inches?|in\.?|feet|ft\.?|cm|mm|m|oz|ounces?|lb|lbs|pounds?|"
    r"g|kg|ml|l|liters?|gallons?|quarts?|watts?|volts?|count|pack)\b",
    flags=re.IGNORECASE,
)
_LEADING_MARKERS = re.compile(r"^[\s\-–—•●▪◦✓✔★☆☕]+")
_MISSING_PUNCTUATION_SPACE = re.compile(r"(?<=[.!?,;:])(?=[A-Za-z])")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def value(self) -> str:
        return " ".join(self.parts)


def _strip_html(value: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value)
        parser.close()
        return parser.value()
    except Exception:
        return re.sub(r"<[^>]+>", " ", value)


def _normalize_space(value: str) -> str:
    value = _MISSING_PUNCTUATION_SPACE.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def _with_sentence_ending(value: str) -> str:
    value = value.strip()
    return value if not value or value[-1] in ".!?" else value + "."


def _fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return _FINGERPRINT_CHARACTERS.sub(" ", normalized).strip()


def _near_duplicate(
    left_fingerprint: str,
    left_tokens: set[str],
    right_fingerprint: str,
    right_tokens: set[str],
    threshold: float,
) -> bool:
    if not left_tokens or not right_tokens:
        return False
    length_ratio = min(len(left_tokens), len(right_tokens)) / max(len(left_tokens), len(right_tokens))
    if length_ratio < 0.75:
        return False
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union)
    if jaccard >= threshold:
        return True
    if jaccard < max(0.65, threshold - 0.12):
        return False
    matcher = SequenceMatcher(None, left_fingerprint, right_fingerprint)
    return matcher.quick_ratio() >= threshold and matcher.ratio() >= threshold


def _split_sentences(value: str) -> list[str]:
    value = re.sub(r"[•●▪◦]+", "\n", value)
    sentences: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(value):
        cleaned = _normalize_space(_LEADING_MARKERS.sub("", part))
        if cleaned:
            sentences.append(cleaned)
    return sentences


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [], {})]
    if isinstance(value, Iterable):
        output: list[str] = []
        for item in value:
            output.extend(_flatten_strings(item))
        return output
    return [str(value)]


@dataclass(frozen=True)
class CleanedText:
    text: str
    sentences: tuple[str, ...]
    stats: Mapping[str, int]


class DescriptionProcessor:
    def __init__(self, config: dict[str, Any], tokenizer: Any | None = None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.hash = config_hash(config)
        self.version = str(config["pipeline"]["version"])
        self.boilerplate_patterns = tuple(
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in config.get("boilerplate", {}).get("drop_sentence_patterns", [])
        )

    def normalize(self, value: str) -> str:
        settings = self.config.get("normalization", {})
        value = html.unescape(str(value))
        if settings.get("strip_html", True):
            value = _strip_html(value)
        value = unicodedata.normalize(settings.get("unicode_form", "NFKC"), value)
        return _normalize_space(value)

    def clean_product_text(self, product: ProductInput) -> CleanedText:
        source_parts = [product.title, *product.descriptions, *product.features]
        normalized_parts = [self.normalize(part) for part in source_parts if str(part).strip()]
        source_text = " ".join(_with_sentence_ending(part) for part in normalized_parts if part)
        sentences = _split_sentences(source_text)
        settings = self.config.get("deduplication", {})
        threshold = float(settings.get("similarity_threshold", 0.92))
        minimum_tokens = int(settings.get("minimum_tokens_for_near_duplicate", 6))
        kept: list[str] = []
        kept_comparisons: list[tuple[str, set[str]]] = []
        fingerprints: set[str] = set()
        duplicate_count = 0
        boilerplate_count = 0
        for sentence in sentences:
            if any(pattern.search(sentence) for pattern in self.boilerplate_patterns):
                boilerplate_count += 1
                continue
            fingerprint = _fingerprint(sentence)
            tokens = set(fingerprint.split())
            if settings.get("exact_sentences", True) and fingerprint in fingerprints:
                duplicate_count += 1
                continue
            if settings.get("near_duplicate_sentences", True) and len(tokens) >= minimum_tokens:
                if any(
                    _near_duplicate(fingerprint, tokens, existing_fingerprint, existing_tokens, threshold)
                    for existing_fingerprint, existing_tokens in kept_comparisons
                ):
                    duplicate_count += 1
                    continue
            fingerprints.add(fingerprint)
            kept.append(sentence)
            kept_comparisons.append((fingerprint, tokens))
        cleaned = " ".join(_with_sentence_ending(sentence) for sentence in kept)
        return CleanedText(
            text=cleaned,
            sentences=tuple(kept),
            stats={
                "source_characters": len(source_text),
                "clean_characters": len(cleaned),
                "source_sentences": len(sentences),
                "kept_sentences": len(kept),
                "duplicate_sentences_removed": duplicate_count,
                "boilerplate_sentences_removed": boilerplate_count,
            },
        )

    def taxonomy_levels(self, categories: Sequence[str]) -> tuple[str, ...]:
        settings = self.config.get("taxonomy", {})
        excluded = {str(item).casefold() for item in settings.get("excluded_labels", [])}
        normalized: list[str] = []
        for category in _flatten_strings(categories):
            label = self.normalize(category)
            if label and label.casefold() not in excluded and label not in normalized:
                normalized.append(label)
        maximum = int(settings.get("maximum_informative_levels", 3))
        return tuple(normalized[-maximum:])

    def taxonomy_text(self, categories: Sequence[str]) -> str:
        separator = str(self.config.get("taxonomy", {}).get("separator", "; "))
        return separator.join(self.taxonomy_levels(categories))

    def selected_attributes(self, details: Mapping[str, Any]) -> tuple[str, ...]:
        settings = self.config.get("attributes", {})
        allowed = [str(key) for key in settings.get("include_keys", [])]
        maximum = int(settings.get("maximum_attributes", 8))
        values: list[str] = []
        for key in allowed:
            value = details.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (dict, list, tuple, set)):
                value = ", ".join(_flatten_strings(value))
            clean_value = self.normalize(str(value))
            if clean_value:
                values.append(f"{key}: {clean_value}")
            if len(values) >= maximum:
                break
        return tuple(values)

    def _sentence_score(self, sentence: str, index: int) -> float:
        settings = self.config.get("canonical", {})
        score = float(settings.get("first_sentence_bonus", 3.0)) if index == 0 else 0.0
        if _MEASUREMENT.search(sentence):
            score += float(settings.get("measurement_bonus", 2.0))
        lowered = sentence.casefold()
        bonus = float(settings.get("high_signal_term_bonus", 1.0))
        score += bonus * sum(term.casefold() in lowered for term in settings.get("high_signal_terms", []))
        return score

    def canonical_description(
        self,
        cleaned: CleanedText,
        categories: Sequence[str],
        details: Mapping[str, Any],
    ) -> str:
        settings = self.config.get("canonical", {})
        maximum_words = int(settings.get("maximum_words", 54))
        maximum_sentences = int(settings.get("maximum_sentences", 4))
        taxonomy = self.taxonomy_text(categories)
        attributes = self.selected_attributes(details)
        ranked = sorted(
            enumerate(cleaned.sentences),
            key=lambda item: (-self._sentence_score(item[1], item[0]), item[0]),
        )[:maximum_sentences]
        selected_sentences = [sentence for _, sentence in sorted(ranked, key=lambda item: item[0])]
        segments = []
        if taxonomy:
            segments.append(_with_sentence_ending(taxonomy))
        if attributes:
            segments.append(_with_sentence_ending("; ".join(attributes)))
        segments.extend(_with_sentence_ending(sentence) for sentence in selected_sentences)
        words: list[str] = []
        for segment in segments:
            candidate = segment.split()
            remaining = maximum_words - len(words)
            if remaining <= 0:
                break
            words.extend(candidate[:remaining])
        canonical = " ".join(words).strip()
        if canonical and canonical[-1] not in ".!?":
            canonical += "."
        return canonical

    def description_chunks(self, text: str, categories: Sequence[str]) -> tuple[str, ...]:
        if self.tokenizer is None or not text:
            return tuple()
        settings = self.config.get("chunking", {})
        maximum_tokens = int(settings.get("maximum_tokens", 64))
        overlap = int(settings.get("overlap_tokens", 12))
        maximum_chunks = int(settings.get("maximum_chunks", 4))
        reserved = int(settings.get("reserved_special_tokens", 2))
        levels = self.taxonomy_levels(categories)
        prefix = f"{levels[-1]}. " if levels and settings.get("prefix_with_leaf", True) else ""
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
        content_ids = self.tokenizer.encode(text, add_special_tokens=False)
        window = maximum_tokens - reserved - len(prefix_ids)
        if window <= 0:
            raise ValueError("Taxonomy prefix leaves no room for description tokens")
        if len(content_ids) <= window:
            starts = [0]
        else:
            step = max(1, window - overlap)
            all_starts = list(range(0, max(1, len(content_ids) - window + 1), step))
            final_start = max(0, len(content_ids) - window)
            if not all_starts or all_starts[-1] != final_start:
                all_starts.append(final_start)
            if len(all_starts) <= maximum_chunks:
                starts = all_starts
            else:
                positions = [round(index * (len(all_starts) - 1) / (maximum_chunks - 1)) for index in range(maximum_chunks)]
                starts = [all_starts[index] for index in positions]
        chunks: list[str] = []
        for start in starts[:maximum_chunks]:
            content = self.tokenizer.decode(content_ids[start : start + window], skip_special_tokens=True).strip()
            chunk = f"{prefix}{content}".strip()
            if chunk and chunk not in chunks:
                chunks.append(chunk)
        return tuple(chunks)

    def transform(self, product: ProductInput) -> TransformResult:
        cleaned = self.clean_product_text(product)
        taxonomy = self.taxonomy_text(product.categories)
        canonical = self.canonical_description(cleaned, product.categories, product.details)
        source_value = {
            "product_id": product.product_id,
            "title": product.title,
            "descriptions": list(product.descriptions),
            "features": list(product.features),
            "details": dict(product.details),
            "categories": list(product.categories),
            "image_url": product.image_url,
        }
        source_hash = hashlib.sha256(
            json.dumps(source_value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        return TransformResult(
            product_id=product.product_id,
            title=self.normalize(product.title),
            image_url=product.image_url,
            category_path=" > ".join(self.normalize(value) for value in product.categories if str(value).strip()),
            taxonomy_text=taxonomy,
            description_clean=cleaned.text,
            description_canonical=canonical,
            description_chunks=self.description_chunks(cleaned.text, product.categories),
            pipeline_version=self.version,
            config_hash=self.hash,
            source_hash=source_hash,
            cleaning_stats=cleaned.stats,
        )
