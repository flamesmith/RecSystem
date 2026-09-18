from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from siglip2_training.features import _write_npz_atomic, choose_device


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


if __name__ == "__main__":
    unittest.main()
