from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from siglip2_training.training import (
    group_safe_batches,
    paired_bootstrap_recall_deltas,
    paired_retrieval_metrics,
    text_view,
    train_adapter_pilot,
)
from siglip2_training.features import _write_npz_atomic


class TrainingTests(unittest.TestCase):
    def test_perfect_pairs_have_full_recall(self) -> None:
        values = np.eye(4, dtype=np.float32)
        metrics = paired_retrieval_metrics(values, values)
        self.assertEqual(metrics["bidirectional_recall@1"], 1.0)
        self.assertGreater(metrics["positive_pair_margin"], 0.0)

    def test_bootstrap_reports_zero_for_identical_systems(self) -> None:
        values = np.eye(4, dtype=np.float32)
        result = paired_bootstrap_recall_deltas(
            values,
            values,
            values,
            values,
            ks=(1,),
            resamples=100,
            confidence_level=0.95,
            seed=1,
        )["bidirectional_recall@1"]
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual(result["confidence_lower"], 0.0)
        self.assertEqual(result["confidence_upper"], 0.0)

    def test_taxonomy_is_only_added_when_present(self) -> None:
        arrays = {
            "text_chunks": np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32),
            "text_chunk_mask": np.asarray([[True], [True]]),
            "text_taxonomy": np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
            "taxonomy_mask": np.asarray([True, False]),
        }
        output = text_view(arrays, "multichunk_with_taxonomy", taxonomy_weight=0.5)
        self.assertGreater(output[0, 1], 0.0)
        np.testing.assert_allclose(output[1], np.asarray([1.0, 0.0]), atol=1e-6)

    def test_group_safe_batches_do_not_create_false_negatives(self) -> None:
        indices = np.arange(6)
        groups = np.asarray(["a", "a", "b", "c", "d", "e"])
        batches = list(group_safe_batches(indices, groups, 4, np.random.default_rng(1)))
        seen = []
        for batch in batches:
            batch_groups = groups[batch]
            self.assertEqual(len(set(batch_groups)), len(batch_groups))
            seen.extend(batch.tolist())
        self.assertEqual(sorted(seen), indices.tolist())

    def test_group_safe_batches_terminate_for_one_duplicate_group(self) -> None:
        indices = np.arange(3)
        groups = np.asarray(["same", "same", "same"])
        batches = list(group_safe_batches(indices, groups, 4, np.random.default_rng(1)))
        self.assertEqual(sorted(index for batch in batches for index in batch.tolist()), indices.tolist())
        self.assertTrue(all(len(batch) == 1 for batch in batches))

    def test_small_adapter_run_writes_guarded_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            feature_directory = root / "features"
            feature_directory.mkdir()
            identity = np.tile(np.eye(4, dtype=np.float16), (3, 1))
            splits = np.asarray(["train"] * 6 + ["validation"] * 3 + ["test"] * 3)
            arrays = {
                "product_id": np.asarray([f"P{index}" for index in range(12)]),
                "split": splits,
                "duplicate_group_id": np.asarray([f"G{index}" for index in range(12)]),
                "category_path": np.asarray(["Home & Kitchen > Decor"] * 12),
                "leaf_category": np.asarray(["Decor"] * 12),
                "image": identity,
                "text_raw": identity.copy(),
            }
            _write_npz_atomic(feature_directory / "features-00000.npz", arrays)
            index = {
                "source_records": 12,
                "written_records": 12,
                "failed_records": 0,
                "shards": ["features-00000.npz"],
                "embedding_dimension": 4,
                "device": "cpu",
                "pretrained_logit_scale": 10.0,
                "pretrained_logit_bias": -2.0,
            }
            (feature_directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
            training_config = {
                "seed": 1,
                "runtime": {
                    "device_preference": ["cpu"],
                    "batch_size": 4,
                    "epochs": 1,
                    "early_stopping_patience": 1,
                },
                "adapter": {
                    "bottleneck_dimension": 2,
                    "dropout": 0.0,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "maximum_logit_scale": 100.0,
                },
                "strategies": ["raw_first_64"],
            }
            guardrails = {
                "promotion_from_5k": {
                    "maximum_recall_at_10_regression_percentage_points": 2.0,
                    "maximum_recall_at_1_regression_percentage_points": 3.0,
                    "minimum_finite_embedding_fraction": 1.0,
                    "minimum_successful_feature_fraction": 0.97,
                    "require_positive_pair_margin": True,
                    "require_no_supported_slice_regression_over_percentage_points": 5.0,
                    "minimum_supported_slice_queries": 2,
                },
                "reconsider_at_50k": {
                    "maximum_recall_at_10_regression_vs_5k_percentage_points": 2.0,
                    "maximum_taxonomy_precision_at_10_regression_percentage_points": 3.0,
                },
            }
            results = train_adapter_pilot(
                feature_directory=feature_directory,
                output_directory=root / "output",
                training_config=training_config,
                guardrail_config=guardrails,
            )
            self.assertEqual(results["selection"]["strategy"], "raw_first_64")
            self.assertTrue((root / "output" / "results.json").exists())
            training_config["reference_5k"] = {
                "strategy": "raw_first_64",
                "checkpoint": str(root / "output" / "raw_first_64.pt"),
            }
            scaled_results = train_adapter_pilot(
                feature_directory=feature_directory,
                output_directory=root / "scaled-output",
                training_config=training_config,
                guardrail_config=guardrails,
            )
            self.assertIsNotNone(scaled_results["reference_5k_validation"])
            self.assertTrue(scaled_results["selection"]["scaling_guardrail_passed"])


if __name__ == "__main__":
    unittest.main()
