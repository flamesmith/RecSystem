from __future__ import annotations

import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from .features import choose_device


def load_feature_shards(directory: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feature_directory = Path(directory)
    index = json.loads((feature_directory / "index.json").read_text(encoding="utf-8"))
    parts: dict[str, list[np.ndarray]] = {}
    for shard_name in index["shards"]:
        with np.load(feature_directory / shard_name, allow_pickle=False) as shard:
            for key in shard.files:
                parts.setdefault(key, []).append(shard[key])
    arrays = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
    if len(arrays.get("product_id", [])) != int(index["written_records"]):
        raise ValueError("Feature index row count does not match shard contents")
    return arrays, index


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def text_view(arrays: dict[str, np.ndarray], strategy: str, taxonomy_weight: float = 0.20) -> np.ndarray:
    if strategy == "raw_first_64":
        return normalize_rows(arrays["text_raw"])
    if strategy == "canonical_64":
        return normalize_rows(arrays["text_canonical"])
    if strategy in {"multichunk_64", "multichunk_with_taxonomy"}:
        chunks = arrays["text_chunks"].astype(np.float32)
        mask = arrays["text_chunk_mask"].astype(np.float32)[..., None]
        pooled = normalize_rows((chunks * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1.0, None))
        if strategy == "multichunk_64":
            return pooled
        taxonomy = normalize_rows(arrays["text_taxonomy"])
        taxonomy_mask = arrays["taxonomy_mask"].astype(bool)[:, None]
        combined = normalize_rows((1.0 - taxonomy_weight) * pooled + taxonomy_weight * taxonomy)
        return np.where(taxonomy_mask, combined, pooled)
    raise ValueError(f"Unknown strategy: {strategy}")


def paired_retrieval_metrics(images: np.ndarray, texts: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    images = normalize_rows(images)
    texts = normalize_rows(texts)
    similarities = images @ texts.T
    targets = np.arange(len(images))
    image_ranked = np.argsort(-similarities, axis=1)
    text_ranked = np.argsort(-similarities.T, axis=1)
    metrics: dict[str, float] = {}
    for k in ks:
        effective = min(k, len(images))
        image_recall = np.any(image_ranked[:, :effective] == targets[:, None], axis=1)
        text_recall = np.any(text_ranked[:, :effective] == targets[:, None], axis=1)
        metrics[f"image_to_text_recall@{k}"] = float(image_recall.mean())
        metrics[f"text_to_image_recall@{k}"] = float(text_recall.mean())
        metrics[f"bidirectional_recall@{k}"] = float((image_recall.mean() + text_recall.mean()) / 2)
    positives = np.diag(similarities)
    off_diagonal = (similarities.sum() - positives.sum()) / max(similarities.size - len(images), 1)
    metrics["mean_positive_cosine"] = float(positives.mean())
    metrics["mean_off_diagonal_cosine"] = float(off_diagonal)
    metrics["positive_pair_margin"] = float(positives.mean() - off_diagonal)
    return metrics


def neighbor_taxonomy_metrics(
    images: np.ndarray, texts: np.ndarray, leaves: np.ndarray, ks: Sequence[int] = (1, 5, 10)
) -> dict[str, float]:
    similarities = normalize_rows(images) @ normalize_rows(texts).T
    np.fill_diagonal(similarities, -np.inf)
    valid_queries = np.flatnonzero(leaves != "")
    metrics: dict[str, float] = {"taxonomy_query_count": float(len(valid_queries))}
    if len(valid_queries) == 0:
        return metrics
    rankings = np.argsort(-similarities[valid_queries], axis=1)
    query_leaves = leaves[valid_queries]
    for k in ks:
        effective = min(k, max(len(leaves) - 1, 1))
        matches = leaves[rankings[:, :effective]] == query_leaves[:, None]
        metrics[f"cross_modal_same_leaf_precision@{k}"] = float(matches.mean())
        metrics[f"cross_modal_same_leaf_hit@{k}"] = float(matches.any(axis=1).mean())
    counts = Counter(str(value) for value in leaves if value)
    denominator = max(len(leaves) - 1, 1)
    metrics["random_same_leaf_expectation"] = float(
        np.mean([(counts[str(leaves[index])] - 1) / denominator for index in valid_queries])
    )
    return metrics


def broad_category(category_path: str) -> str:
    levels = [level.strip() for level in str(category_path).split(" > ") if level.strip()]
    if not levels:
        return "(missing taxonomy)"
    if levels[0].casefold() in {"home & kitchen", "amazon home"} and len(levels) > 1:
        return levels[1]
    return levels[0]


def slice_recall_at_10(
    images: np.ndarray,
    texts: np.ndarray,
    category_paths: np.ndarray,
    minimum_queries: int,
) -> dict[str, dict[str, float]]:
    similarities = normalize_rows(images) @ normalize_rows(texts).T
    targets = np.arange(len(images))
    image_hits = np.any(np.argsort(-similarities, axis=1)[:, : min(10, len(images))] == targets[:, None], axis=1)
    text_hits = np.any(np.argsort(-similarities.T, axis=1)[:, : min(10, len(images))] == targets[:, None], axis=1)
    labels = np.asarray([broad_category(value) for value in category_paths])
    output: dict[str, dict[str, float]] = {}
    for label, count in sorted(Counter(labels).items()):
        if count < minimum_queries:
            continue
        mask = labels == label
        output[label] = {
            "queries": int(count),
            "bidirectional_recall@10": float((image_hits[mask].mean() + text_hits[mask].mean()) / 2),
        }
    return output


def group_safe_batches(
    indices: np.ndarray,
    group_ids: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[np.ndarray]:
    shuffled = indices[rng.permutation(len(indices))].tolist()
    pending: list[int] = []
    while shuffled or pending:
        candidates = shuffled + pending
        shuffled = []
        pending = []
        batch: list[int] = []
        groups: set[str] = set()
        for index in candidates:
            group = str(group_ids[index])
            if len(batch) < batch_size and group not in groups:
                batch.append(index)
                groups.add(group)
            else:
                pending.append(index)
        if batch:
            yield np.asarray(batch, dtype=np.int64)
        shuffled, pending = pending, []


def _torch_modules():
    import torch
    from torch import nn

    class ResidualAdapter(nn.Module):
        def __init__(self, dimension: int, bottleneck: int, dropout: float) -> None:
            super().__init__()
            self.down = nn.Linear(dimension, bottleneck, bias=False)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.up = nn.Linear(bottleneck, dimension, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, values):
            import torch.nn.functional as functional

            residual = self.up(self.dropout(self.activation(self.down(values))))
            return functional.normalize(values + residual, dim=-1)

    class ContrastiveAdapters(nn.Module):
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

        def forward(self, images, texts):
            image_values, text_values = self.embeddings(images, texts)
            scale = self.logit_scale.exp().clamp(max=self.maximum_logit_scale)
            return scale * image_values @ text_values.T + self.logit_bias

    return torch, ContrastiveAdapters


def _sigmoid_contrastive_loss(logits):
    import torch
    import torch.nn.functional as functional

    labels = -torch.ones_like(logits)
    labels.fill_diagonal_(1.0)
    return -functional.logsigmoid(labels * logits).sum(dim=1).mean()


def _adapt_all(model: Any, images: np.ndarray, texts: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.inference_mode():
        image_values = torch.from_numpy(images).to(device)
        text_values = torch.from_numpy(texts).to(device)
        adapted_images, adapted_texts = model.embeddings(image_values, text_values)
    return adapted_images.cpu().numpy(), adapted_texts.cpu().numpy()


def evaluate_guardrails(
    baseline: dict[str, float],
    trained: dict[str, float],
    baseline_slices: dict[str, dict[str, float]],
    trained_slices: dict[str, dict[str, float]],
    guardrails: dict[str, Any],
    finite_fraction: float,
    feature_success_fraction: float,
) -> dict[str, Any]:
    settings = guardrails["promotion_from_5k"]
    checks = {
        "recall_at_10_regression": (
            trained["bidirectional_recall@10"] - baseline["bidirectional_recall@10"]
            >= -float(settings["maximum_recall_at_10_regression_percentage_points"]) / 100
        ),
        "recall_at_1_regression": (
            trained["bidirectional_recall@1"] - baseline["bidirectional_recall@1"]
            >= -float(settings["maximum_recall_at_1_regression_percentage_points"]) / 100
        ),
        "finite_embeddings": finite_fraction >= float(settings["minimum_finite_embedding_fraction"]),
        "feature_success": feature_success_fraction >= float(settings["minimum_successful_feature_fraction"]),
        "positive_pair_margin": (
            trained["positive_pair_margin"] > 0
            if settings.get("require_positive_pair_margin", True)
            else True
        ),
    }
    maximum_slice_regression = float(
        settings["require_no_supported_slice_regression_over_percentage_points"]
    ) / 100
    shared_slices = sorted(set(baseline_slices) & set(trained_slices))
    slice_deltas = {
        label: trained_slices[label]["bidirectional_recall@10"]
        - baseline_slices[label]["bidirectional_recall@10"]
        for label in shared_slices
    }
    checks["supported_slices"] = all(delta >= -maximum_slice_regression for delta in slice_deltas.values())
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "slice_recall_at_10_deltas": slice_deltas,
    }


def train_adapter_pilot(
    *,
    feature_directory: str | Path,
    output_directory: str | Path,
    training_config: dict[str, Any],
    guardrail_config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, feature_index = load_feature_shards(feature_directory)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    if (output_path / "results.json").exists():
        raise FileExistsError(f"Training output already exists: {output_path}")
    torch, ContrastiveAdapters = _torch_modules()
    seed = int(training_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runtime = training_config["runtime"]
    adapter_config = training_config["adapter"]
    device = choose_device(runtime.get("device_preference", ["mps", "cpu"]))
    images = normalize_rows(arrays["image"])
    splits = arrays["split"]
    train_indices = np.flatnonzero(splits == "train")
    validation_indices = np.flatnonzero(splits == "validation")
    test_indices = np.flatnonzero(splits == "test")
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise ValueError("Feature shards must contain non-empty train, validation, and test splits")
    feature_success_fraction = float(feature_index["written_records"] / max(feature_index["source_records"], 1))
    finite_fraction = float(
        np.mean([np.isfinite(value).all() for key, value in arrays.items() if value.dtype.kind == "f"])
    )
    strategies: list[dict[str, Any]] = []
    for strategy_number, strategy in enumerate(training_config["strategies"]):
        text_values = text_view(
            arrays,
            strategy,
            float(training_config.get("multichunk_with_taxonomy", {}).get("taxonomy_weight", 0.20)),
        )
        dimension = int(images.shape[1])
        model = ContrastiveAdapters(
            dimension=dimension,
            bottleneck=int(adapter_config["bottleneck_dimension"]),
            dropout=float(adapter_config["dropout"]),
            logit_scale=float(feature_index["pretrained_logit_scale"]),
            logit_bias=float(feature_index["pretrained_logit_bias"]),
            maximum_logit_scale=float(adapter_config["maximum_logit_scale"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(adapter_config["learning_rate"]),
            weight_decay=float(adapter_config["weight_decay"]),
        )
        validation_images = images[validation_indices]
        validation_texts = text_values[validation_indices]
        baseline_metrics = paired_retrieval_metrics(validation_images, validation_texts)
        baseline_taxonomy = neighbor_taxonomy_metrics(
            validation_images, validation_texts, arrays["leaf_category"][validation_indices]
        )
        minimum_slice_queries = int(
            guardrail_config["promotion_from_5k"]["minimum_supported_slice_queries"]
        )
        baseline_slices = slice_recall_at_10(
            validation_images,
            validation_texts,
            arrays["category_path"][validation_indices],
            minimum_slice_queries,
        )
        best_state = copy.deepcopy(model.state_dict())
        best_score = baseline_metrics["bidirectional_recall@10"]
        best_epoch = 0
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []
        strategy_rng = np.random.default_rng(seed + strategy_number)
        for epoch in range(1, int(runtime["epochs"]) + 1):
            model.train()
            losses: list[float] = []
            for batch_indices in group_safe_batches(
                train_indices,
                arrays["duplicate_group_id"],
                int(runtime["batch_size"]),
                strategy_rng,
            ):
                batch_images = torch.from_numpy(images[batch_indices]).to(device)
                batch_texts = torch.from_numpy(text_values[batch_indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = _sigmoid_contrastive_loss(model(batch_images, batch_texts))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss for {strategy} at epoch {epoch}")
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            adapted_images, adapted_texts = _adapt_all(
                model, validation_images, validation_texts, device
            )
            epoch_metrics = paired_retrieval_metrics(adapted_images, adapted_texts)
            score = epoch_metrics["bidirectional_recall@10"]
            history.append(
                {
                    "epoch": epoch,
                    "mean_loss": float(np.mean(losses)),
                    "validation_bidirectional_recall@1": epoch_metrics["bidirectional_recall@1"],
                    "validation_bidirectional_recall@10": score,
                }
            )
            if score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= int(runtime["early_stopping_patience"]):
                break
        model.load_state_dict(best_state)
        model.eval()
        adapted_validation_images, adapted_validation_texts = _adapt_all(
            model, validation_images, validation_texts, device
        )
        trained_metrics = paired_retrieval_metrics(adapted_validation_images, adapted_validation_texts)
        trained_taxonomy = neighbor_taxonomy_metrics(
            adapted_validation_images,
            adapted_validation_texts,
            arrays["leaf_category"][validation_indices],
        )
        trained_slices = slice_recall_at_10(
            adapted_validation_images,
            adapted_validation_texts,
            arrays["category_path"][validation_indices],
            minimum_slice_queries,
        )
        guardrails = evaluate_guardrails(
            baseline_metrics,
            trained_metrics,
            baseline_slices,
            trained_slices,
            guardrail_config,
            finite_fraction,
            feature_success_fraction,
        )
        checkpoint_path = output_path / f"{strategy}.pt"
        torch.save(
            {
                "strategy": strategy,
                "state_dict": model.state_dict(),
                "dimension": dimension,
                "training_config": training_config,
                "feature_index": feature_index,
            },
            checkpoint_path,
        )
        strategies.append(
            {
                "strategy": strategy,
                "best_epoch": best_epoch,
                "history": history,
                "baseline_validation": baseline_metrics,
                "trained_validation": trained_metrics,
                "baseline_taxonomy": baseline_taxonomy,
                "trained_taxonomy": trained_taxonomy,
                "baseline_slices": baseline_slices,
                "trained_slices": trained_slices,
                "guardrails": guardrails,
                "checkpoint": checkpoint_path.name,
            }
        )

    for result in strategies:
        use_adapter = (
            result["best_epoch"] > 0
            and result["guardrails"]["passed"]
            and result["trained_validation"]["bidirectional_recall@10"]
            >= result["baseline_validation"]["bidirectional_recall@10"]
        )
        result["selected_variant"] = "trained_adapter" if use_adapter else "pretrained"
        result["selected_validation"] = (
            result["trained_validation"] if use_adapter else result["baseline_validation"]
        )
    winner = max(
        strategies,
        key=lambda result: (
            result["selected_validation"]["bidirectional_recall@10"],
            result["selected_validation"]["bidirectional_recall@1"],
        ),
    )
    deploy_trained = winner["selected_variant"] == "trained_adapter"
    winning_texts = text_view(
        arrays,
        winner["strategy"],
        float(training_config.get("multichunk_with_taxonomy", {}).get("taxonomy_weight", 0.20)),
    )
    test_images = images[test_indices]
    test_texts = winning_texts[test_indices]
    baseline_test_metrics = paired_retrieval_metrics(test_images, test_texts)
    baseline_test_taxonomy = neighbor_taxonomy_metrics(
        test_images, test_texts, arrays["leaf_category"][test_indices]
    )
    if deploy_trained:
        winning_model = ContrastiveAdapters(
            dimension=int(images.shape[1]),
            bottleneck=int(adapter_config["bottleneck_dimension"]),
            dropout=float(adapter_config["dropout"]),
            logit_scale=float(feature_index["pretrained_logit_scale"]),
            logit_bias=float(feature_index["pretrained_logit_bias"]),
            maximum_logit_scale=float(adapter_config["maximum_logit_scale"]),
        ).to(device)
        checkpoint = torch.load(output_path / winner["checkpoint"], map_location=device)
        winning_model.load_state_dict(checkpoint["state_dict"])
        winning_model.eval()
        test_images, test_texts = _adapt_all(winning_model, test_images, test_texts, device)
    test_metrics = paired_retrieval_metrics(test_images, test_texts)
    test_taxonomy = neighbor_taxonomy_metrics(
        test_images, test_texts, arrays["leaf_category"][test_indices]
    )
    results = {
        "feature_directory": str(Path(feature_directory).resolve()),
        "device": device,
        "rows": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "finite_embedding_fraction": finite_fraction,
        "feature_success_fraction": feature_success_fraction,
        "strategies": strategies,
        "selection": {
            "strategy": winner["strategy"],
            "variant": "trained_adapter" if deploy_trained else "pretrained",
            "strategies_passing_guardrails": sum(
                result["guardrails"]["passed"] for result in strategies
            ),
            "winner_guardrails_passed": bool(winner["guardrails"]["passed"]),
            "validation": winner["selected_validation"],
        },
        "held_out_test": {
            "pretrained_retrieval": baseline_test_metrics,
            "selected_retrieval": test_metrics,
            "pretrained_taxonomy": baseline_test_taxonomy,
            "selected_taxonomy": test_taxonomy,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    temporary = output_path / "results.json.part"
    temporary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temporary.replace(output_path / "results.json")
    return results
