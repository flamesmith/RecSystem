from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from siglip2_training.cache import CacheLimits, ImageCache


def png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (96, 96)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color=color).save(output, format="PNG")
    return output.getvalue()


class ImageCacheTests(unittest.TestCase):
    def limits(self, maximum_bytes: int = 10_000, target_bytes: int = 8_000) -> CacheLimits:
        return CacheLimits(
            maximum_bytes=maximum_bytes,
            eviction_target_bytes=target_bytes,
            minimum_free_bytes=0,
            maximum_file_bytes=1_000_000,
            maximum_pixels=1_000_000,
            minimum_width=64,
            minimum_height=64,
            allowed_formats=("PNG",),
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            maximum_attempts=1,
            backoff_seconds=0,
            chunk_size_bytes=1024,
            user_agent="test",
        )

    def test_put_and_get_validated_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ImageCache(root, root / "cache.sqlite", self.limits())
            url = "https://example.test/image.png"
            stored = cache.put_bytes(url, png_bytes((255, 0, 0)))
            loaded = cache.get(url)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(stored.content_sha256, loaded.content_sha256)
            self.assertEqual((loaded.width, loaded.height), (96, 96))
            self.assertEqual(cache.status()["ready_entries"], 1)

    def test_rejects_image_below_minimum_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ImageCache(root, root / "cache.sqlite", self.limits())
            with self.assertRaisesRegex(ValueError, "too small"):
                cache.put_bytes("https://example.test/tiny.png", png_bytes((0, 0, 0), (32, 32)))

    def test_lru_eviction_removes_oldest_entry(self) -> None:
        first = png_bytes((255, 0, 0))
        second = png_bytes((0, 0, 255))
        maximum = len(first) + len(second) - 1
        target = max(len(first), len(second))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = ImageCache(root, root / "cache.sqlite", self.limits(maximum, target))
            first_url = "https://example.test/first.png"
            second_url = "https://example.test/second.png"
            cache.put_bytes(first_url, first)
            cache.put_bytes(second_url, second)
            self.assertIsNone(cache.get(first_url))
            self.assertIsNotNone(cache.get(second_url))
            self.assertEqual(cache.status()["ready_entries"], 1)


if __name__ == "__main__":
    unittest.main()
