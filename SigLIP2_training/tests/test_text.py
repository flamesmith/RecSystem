from __future__ import annotations

import copy
import unittest
from pathlib import Path

from siglip2_training.config import config_hash, load_yaml
from siglip2_training.schema import ProductInput
from siglip2_training.text import DescriptionProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WhitespaceTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self.token_to_id:
                token_id = len(self.token_to_id) + 1
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
            ids.append(self.token_to_id[token])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(self.id_to_token[token_id] for token_id in ids)


class DescriptionProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml(PROJECT_ROOT / "configs" / "description_v1.yaml")

    def test_cleaning_removes_duplicates_and_boilerplate_but_preserves_attributes(self) -> None:
        processor = DescriptionProcessor(self.config)
        product = ProductInput(
            product_id="example",
            title="Modern walnut coffee table",
            descriptions=(
                "Solid walnut coffee table with a 42 inch top. "
                "Solid walnut coffee table with a 42 inch top. "
                "Actual color may vary due to your monitor.",
            ),
            features=("Black metal legs",),
            details={"Material": "Walnut", "Color": "Brown"},
            categories=("Home & Kitchen", "Furniture", "Living Room Furniture", "Coffee Tables"),
        )
        result = processor.transform(product)
        self.assertIn("42 inch", result.description_clean)
        self.assertIn("Black metal legs", result.description_clean)
        self.assertNotIn("color may vary", result.description_clean.lower())
        self.assertNotIn("!.", result.description_clean)
        self.assertEqual(result.cleaning_stats["duplicate_sentences_removed"], 1)
        self.assertEqual(result.cleaning_stats["boilerplate_sentences_removed"], 1)
        self.assertIn("Material: Walnut", result.description_canonical)
        self.assertIn("Coffee Tables", result.taxonomy_text)
        self.assertNotIn("Home & Kitchen", result.taxonomy_text)

    def test_config_hash_is_deterministic_and_sensitive_to_changes(self) -> None:
        first = config_hash(self.config)
        second = config_hash(copy.deepcopy(self.config))
        changed = copy.deepcopy(self.config)
        changed["pipeline"]["version"] = "1.0.1"
        self.assertEqual(first, second)
        self.assertNotEqual(first, config_hash(changed))

    def test_chunks_are_bounded_and_cover_the_tail(self) -> None:
        tokenizer = WhitespaceTokenizer()
        processor = DescriptionProcessor(self.config, tokenizer=tokenizer)
        words = [f"token{index}" for index in range(240)]
        product = ProductInput(
            product_id="long",
            descriptions=(" ".join(words),),
            categories=("Home & Kitchen", "Furniture", "Chairs"),
        )
        result = processor.transform(product)
        self.assertEqual(len(result.description_chunks), 4)
        for chunk in result.description_chunks:
            self.assertLessEqual(len(tokenizer.encode(chunk)), 62)
            self.assertTrue(chunk.startswith("Chairs."))
        self.assertIn("token239", result.description_chunks[-1])

    def test_transform_is_deterministic(self) -> None:
        processor = DescriptionProcessor(self.config)
        product = ProductInput(
            product_id="stable",
            title="Cotton bath towel",
            descriptions=("Soft and washable cotton towel.",),
            categories=("Home & Kitchen", "Bath", "Bath Towels"),
        )
        self.assertEqual(processor.transform(product), processor.transform(product))


if __name__ == "__main__":
    unittest.main()
