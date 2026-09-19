from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from siglip2_training.features import (
    _completed_shards,
    _load_reuse_features,
    _reuse_feature_keys,
    _write_npz_atomic,
    choose_device,
)


class FeatureTests(unittest.TestCase):
    def test_atomic_npz_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "features-00000.npz"
            values = {
                "product_id": np.asarray(["A", "B"]),
                "image": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
            }
            _write_npz_atomic(path, values)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".npz.part").exists())
            with np.load(path, allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["product_id"], values["product_id"])
                np.testing.assert_array_equal(loaded["image"], values["image"])

    def test_device_choice_has_cpu_fallback(self) -> None:
        self.assertEqual(choose_device(["cpu"]), "cpu")

    def test_completed_shards_are_discovered_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _write_npz_atomic(
                directory / "features-00000.npz",
                {"product_id": np.asarray(["A", "B"]), "image": np.eye(2, dtype=np.float16)},
            )
            names, completed = _completed_shards(directory)
            self.assertEqual(names, ["features-00000.npz"])
            self.assertEqual(completed, {"A", "B"})

    def test_reuse_features_are_indexed_by_product_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _write_npz_atomic(
                directory / "features-00000.npz",
                {"product_id": np.asarray(["A", "B"]), "image": np.eye(2, dtype=np.float16)},
            )
            (directory / "index.json").write_text(
                '{"shards": ["features-00000.npz"]}', encoding="utf-8"
            )
            arrays, positions = _load_reuse_features(directory)
            self.assertEqual(positions, {"A": 0, "B": 1})
            np.testing.assert_array_equal(arrays["image"], np.eye(2, dtype=np.float16))

    def test_reuse_feature_keys_exclude_disabled_views(self) -> None:
        keys = _reuse_feature_keys(
            {
                "raw_first_64": True,
                "canonical_64": False,
                "taxonomy_64": True,
                "multichunk_64": True,
            }
        )
        self.assertEqual(
            keys,
            {"image", "text_raw", "text_taxonomy", "text_chunks", "text_chunk_mask"},
        )


if __name__ == "__main__":
    unittest.main()
