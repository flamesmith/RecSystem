from __future__ import annotations

import unittest

from siglip2_training.sampling import select_deterministic_sample, stable_priority, transform_and_split
from siglip2_training.text import DescriptionProcessor


DESCRIPTION_CONFIG = {
    "pipeline": {"version": "1.0.0"},
    "normalization": {"strip_html": True, "unicode_form": "NFKC"},
    "deduplication": {"exact_sentences": True, "near_duplicate_sentences": False},
    "taxonomy": {"maximum_informative_levels": 3, "excluded_labels": [], "separator": "; "},
    "attributes": {"include_keys": [], "maximum_attributes": 0},
    "canonical": {"maximum_words": 54, "maximum_sentences": 4},
    "image_url": {"required_variant": "MAIN", "preferred_sizes": ["large"]},
}


def raw_record(product_id: str, description: str, image_url: str) -> dict:
    return {
        "parent_asin": product_id,
        "title": f"Product {product_id}",
        "description": [description],
        "features": [],
        "details": {},
        "categories": ["Home & Kitchen", "Room", "Leaf"],
        "images": [{"variant": "MAIN", "large": image_url}],
    }


class SamplingTests(unittest.TestCase):
    def test_selection_is_independent_of_input_order(self) -> None:
        records = [raw_record(str(index), f"Description {index}", f"https://x/{index}.jpg") for index in range(20)]
        first = select_deterministic_sample(
            records, description_config=DESCRIPTION_CONFIG, sample_size=5, seed="seed"
        )
        second = select_deterministic_sample(
            reversed(records), description_config=DESCRIPTION_CONFIG, sample_size=5, seed="seed"
        )
        first_ids = [record["parent_asin"] for record in first.records]
        second_ids = [record["parent_asin"] for record in second.records]
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(
            first_ids,
            sorted(first_ids, key=lambda product_id: stable_priority(product_id, "seed")),
        )

    def test_exact_duplicate_content_stays_in_one_split(self) -> None:
        records = [
            raw_record("A", "The same description.", "https://x/a.jpg"),
            raw_record("B", "The same description.", "https://x/b.jpg"),
            raw_record("C", "Different description.", "https://x/a.jpg"),
        ]
        transformed, statistics = transform_and_split(
            records,
            description_config=DESCRIPTION_CONFIG,
            processor=DescriptionProcessor(DESCRIPTION_CONFIG),
            seed="seed",
            train_fraction=0.8,
            validation_fraction=0.1,
        )
        self.assertEqual(len({record["split"] for record in transformed}), 1)
        self.assertEqual(len({record["duplicate_group_id"] for record in transformed}), 1)
        self.assertEqual(statistics["duplicate_groups"], 1)


if __name__ == "__main__":
    unittest.main()
