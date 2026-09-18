from __future__ import annotations

import unittest

from siglip2_training.metadata import amazon_record_to_product, select_main_image_url


class MetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "image_url": {
                "required_variant": "MAIN",
                "preferred_sizes": ["large", "hi_res", "thumb"],
            }
        }

    def test_prefers_large_main_image(self) -> None:
        images = [
            {"variant": "PT01", "large": "https://example.test/secondary.jpg"},
            {
                "variant": "MAIN",
                "thumb": "https://example.test/thumb.jpg",
                "large": "https://example.test/large.jpg",
                "hi_res": "https://example.test/high.jpg",
            },
        ]
        self.assertEqual(
            select_main_image_url(images, self.config["image_url"]),
            "https://example.test/large.jpg",
        )

    def test_amazon_adapter_preserves_structured_fields(self) -> None:
        record = {
            "parent_asin": "B012345678",
            "title": "Example product",
            "description": ["First description"],
            "features": ["Feature one"],
            "details": {"Material": "Glass"},
            "categories": ["Home & Kitchen", "Kitchen & Dining", "Mug Sets"],
            "images": [{"variant": "MAIN", "large": "https://example.test/image.jpg"}],
        }
        product = amazon_record_to_product(record, self.config)
        self.assertIsNotNone(product)
        assert product is not None
        self.assertEqual(product.product_id, "B012345678")
        self.assertEqual(product.details["Material"], "Glass")
        self.assertEqual(product.categories[-1], "Mug Sets")
        self.assertEqual(product.image_url, "https://example.test/image.jpg")


if __name__ == "__main__":
    unittest.main()
