"""Condition-level MIL training with robust video-graph ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from .advanced_dataset import (
    AdvancedGraphStore,
    AdvancedStandardizer,
    ConditionGraphDataset,
    ConditionUnit,
    build_condition_units,
    resolve_cache_schema,
)
from .advanced_model import RepeatSetClassifier, build_encoder
from .constants import BINARY_NAMES, JAR3_NAMES, WATER_CODE
from .dataset import GraphRecord, load_graph_records
from .model import count_parameters
from .train import (
    _save_confusion_plot,
    _write_csv,
    choose_device,
    confusion_and_metrics,
    set_seed,
)

ADVANCED_TRAINER_VERSION = 4


def _labels(units: Sequence[ConditionUnit], task: str) -> np.ndarray:
    return np.asarray([unit.label_for(task) for unit in units], dtype=np.int64)


def _groups(units: Sequence[ConditionUnit]) -> np.ndarray:
    return np.asarray([unit.subject_id for unit in units])


def _run_signature(args: argparse.Namespace) -> str:
    payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    payload["advanced_trainer_version"] = ADVANCED_TRAINER_VERSION
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _prepare_fresh_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {path}. Choose a new "
            "--output-dir so folds/configurations cannot be mixed."
        )
    path.mkdir(parents=True, exist_ok=True)


def filter_neutral_records(
    records: Sequence[GraphRecord],
    *,
    min_detection_ratio: float,
    representation: str = "legacy",
) -> tuple[list[GraphRecord], dict[str, int]]:
    """Exclude repeats whose pre-trial neutral face signal is unusable."""
    if not 0.0 <= min_detection_ratio <= 1.0:
        raise ValueError("neutral baseline detection threshold must be in [0,1]")
    kept: list[GraphRecord] = []
    audit = {
        "records_checked": len(records),
        "excluded_missing_baseline": 0,
        "excluded_low_baseline_detection": 0,
        "included": 0,
    }
    for record in records:
        with np.load(record.graph_path, allow_pickle=False) as data:
            schema = resolve_cache_schema(
                data,
                representation=representation,
                path=record.graph_path,
            )
            if "baseline_seq" not in data:
                audit["excluded_missing_baseline"] += 1
                continue
            baseline = np.asarray(data["baseline_seq"])
            if baseline.ndim != 3 or baseline.shape[1:] != (
                schema.num_nodes,
                schema.num_features,
            ):
                raise ValueError(
                    f"{record.graph_path}: invalid baseline_seq shape "
                    f"{baseline.shape}"
                )
            ratio = (
                float(data["baseline_detection_ratio"])
                if "baseline_detection_ratio" in data
                else float(
                    (
                        baseline[
                            :, :, list(schema.observed_mask_indices)
                        ]
                        > 0.5
                    ).any(axis=(1, 2)).mean()
                )
            )
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"{record.graph_path}: invalid baseline detection ratio={ratio}"
            )
        if ratio < min_detection_ratio:
            audit["excluded_low_baseline_detection"] += 1
            continue
        kept.append(record)
    audit["included"] = len(kept)
    if not kept:
        raise ValueError(
            "No graph remains after neutral-baseline quality filtering; "
            f"audit={audit}"
        )
    return kept, audit


def make_unit_splits(
    units: Sequence[ConditionUnit],
    task: str,
    *,
    n_splits: int,
    seed: int,
    subset: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    all_indices = (
        np.arange(len(units), dtype=np.int64)
        if subset is None
        else np.asarray(subset, dtype=np.int64)
    )
    labels = _labels(units, task)[all_indices]
    groups = _groups(units)[all_indices]
    n_groups = len(np.unique(groups))
    if not 2 <= n_splits <= n_groups:
        raise ValueError(f"n_splits must be in [2,{n_groups}], got {n_splits}")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    result = []
    for train_local, test_local in splitter.split(
        np.zeros(len(labels)), labels, groups
    ):
        train = all_indices[np.asarray(train_local, dtype=np.int64)]
        test = all_indices[np.asarray(test_local, dtype=np.int64)]
        overlap = set(_groups(units)[train]).intersection(_groups(units)[test])
        if overlap:
            raise AssertionError(f"Subject leakage: {sorted(overlap)}")
        result.append((train, test))
    return result


def unit_class_weights(
    units: Sequence[ConditionUnit],
    unit_indices: Sequence[int],
    *,
    task: str,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    labels = _labels(units, task)[np.asarray(unit_indices, dtype=np.int64)]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training units missing a class: {counts.tolist()}")
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_loader(
    store: AdvancedGraphStore,
    units: Sequence[ConditionUnit],
    indices: Sequence[int],
    *,
    task: str,
    normalizer: AdvancedStandardizer,
    training: bool,
    batch_size: int,
    num_workers: int,
    temporal_crop_min: float,
    noise_std: float,
    repeat_dropout: float,
    device: torch.device,
) -> DataLoader:
    dataset = ConditionGraphDataset(
        store,
        units,
        indices,
        task=task,
        normalizer=normalizer,
        training=training,
        temporal_crop_min=temporal_crop_min if training else 1.0,
        noise_std=noise_std if training else 0.0,
        repeat_dropout=repeat_dropout if training else 0.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _new_model(
    store: AdvancedGraphStore,
    args: argparse.Namespace,
    *,
    num_classes: int,
    objective: str,
    device: torch.device,
) -> RepeatSetClassifier:
    encoder = build_encoder(
        args.model,
        num_features=store.num_features,
        num_nodes=store.num_nodes,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        temporal_pooling=args.temporal_pooling,
    )
    return RepeatSetClassifier(
        encoder,
        num_classes=num_classes,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        aggregation=args.aggregation,
        objective=objective,
    ).to(device)


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _forward(
    model: RepeatSetClassifier,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    return model(
        batch["graphs"],
        batch["adjacency"],
        batch["repeat_mask"],
    )


def _loss(
    model: RepeatSetClassifier,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    if model.objective == "ce":
        return functional.cross_entropy(outputs, labels, weight=class_weights)
    targets = torch.stack(
        ((labels > 0).float(), (labels > 1).float()), dim=1
    )
    per_sample = functional.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    ).mean(dim=1)
    weights = class_weights[labels]
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def train_epoch(
    model: RepeatSetClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward(model, batch)
        loss = _loss(model, outputs, batch["label"], class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = len(batch["label"])
        total_loss += float(loss.item()) * count
        correct += int(
            (model.predictions(outputs) == batch["label"]).sum().item()
        )
        total += count
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: RepeatSetClassifier,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    truth: list[int] = []
    prediction: list[int] = []
    probabilities: list[list[float]] = []
    unit_indices: list[int] = []
    total_loss = 0.0
    total = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        outputs = _forward(model, batch)
        loss = _loss(model, outputs, batch["label"], class_weights)
        probs = model.probabilities(outputs)
        predicted = model.predictions(outputs)
        count = len(batch["label"])
        total_loss += float(loss.item()) * count
        total += count
        truth.extend(batch["label"].cpu().tolist())
        prediction.extend(predicted.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        unit_indices.extend(batch["unit_index"].cpu().tolist())
    confusion, metrics = confusion_and_metrics(
        truth, prediction, num_classes=num_classes
    )
    metrics["loss"] = total_loss / max(total, 1)
    if truth:
        metrics["mae"] = float(
            np.abs(np.asarray(truth) - np.asarray(prediction)).mean()
        )
        metrics["quadratic_weighted_kappa"] = float(
            cohen_kappa_score(truth, prediction, weights="quadratic")
        )
    return {
        "truth": truth,
        "prediction": prediction,
        "probabilities": probabilities,
        "unit_indices": unit_indices,
        "confusion": confusion,
        "metrics": metrics,
    }


def select_epoch_count(
    store: AdvancedGraphStore,
    units: Sequence[ConditionUnit],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    *,
    task: str,
    num_classes: int,
    objective: str,
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    normalizer = AdvancedStandardizer.fit(store, units, train_indices)
    train_loader = _make_loader(
        store,
        units,
        train_indices,
        task=task,
        normalizer=normalizer,
        training=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_crop_min=args.temporal_crop_min,
        noise_std=args.noise_std,
        repeat_dropout=args.repeat_dropout,
        device=device,
    )
    validation_loader = _make_loader(
        store,
        units,
        validation_indices,
        task=task,
        normalizer=normalizer,
        training=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_crop_min=1.0,
        noise_std=0.0,
        repeat_dropout=0.0,
        device=device,
    )
    model = _new_model(
        store,
        args,
        num_classes=num_classes,
        objective=objective,
        device=device,
    )
    weights = unit_class_weights(
        units,
        train_indices,
        task=task,
        num_classes=num_classes,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = 1
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, weights, device
        )
        validation = evaluate(
            model,
            validation_loader,
            weights,
            device,
            num_classes=num_classes,
        )
        metrics = validation["metrics"]
        rank = (
            float(metrics["balanced_accuracy"]),
            float(metrics["macro_f1"]),
            -float(metrics["loss"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    return best_epoch, history


def pooled_inner_epoch(
    histories: Sequence[Sequence[dict[str, Any]]],
) -> tuple[int, list[dict[str, float | int]]]:
    """Choose one refit epoch from the mean inner-fold validation curve.

    Taking the median of independent argmax epochs is unstable when each
    validation fold contains only about 35--40 conditions.  We instead compare
    epochs on the portion of the learning curve observed by every inner fold.
    """
    if not histories or any(not history for history in histories):
        raise ValueError("Cannot select an epoch from empty histories")
    common_epochs = min(len(history) for history in histories)
    pooled: list[dict[str, float | int]] = []
    best_rank = (-math.inf, -math.inf, -math.inf)
    best_epoch = 1
    for offset in range(common_epochs):
        rows = [history[offset] for history in histories]
        epoch = offset + 1
        if any(int(row["epoch"]) != epoch for row in rows):
            raise ValueError("Inner histories must contain consecutive epochs")
        row = {
            "epoch": epoch,
            "n_inner_folds": len(rows),
            "mean_validation_balanced_accuracy": float(
                np.mean([row["validation_balanced_accuracy"] for row in rows])
            ),
            "mean_validation_macro_f1": float(
                np.mean([row["validation_macro_f1"] for row in rows])
            ),
            "mean_validation_loss": float(
                np.mean([row["validation_loss"] for row in rows])
            ),
        }
        pooled.append(row)
        rank = (
            float(row["mean_validation_balanced_accuracy"]),
            float(row["mean_validation_macro_f1"]),
            -float(row["mean_validation_loss"]),
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
    return best_epoch, pooled


def refit(
    store: AdvancedGraphStore,
    units: Sequence[ConditionUnit],
    train_indices: np.ndarray,
    args: argparse.Namespace,
    *,
    task: str,
    num_classes: int,
    objective: str,
    epochs: int,
    device: torch.device,
) -> tuple[
    RepeatSetClassifier,
    AdvancedStandardizer,
    list[dict[str, Any]],
]:
    normalizer = AdvancedStandardizer.fit(store, units, train_indices)
    loader = _make_loader(
        store,
        units,
        train_indices,
        task=task,
        normalizer=normalizer,
        training=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        temporal_crop_min=args.temporal_crop_min,
        noise_std=args.noise_std,
        repeat_dropout=args.repeat_dropout,
        device=device,
    )
    model = _new_model(
        store,
        args,
        num_classes=num_classes,
        objective=objective,
        device=device,
    )
    weights = unit_class_weights(
        units,
        train_indices,
        task=task,
        num_classes=num_classes,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = train_epoch(model, loader, optimizer, weights, device)
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    return model, normalizer, history


def prediction_rows(
    evaluation: dict[str, Any],
    units: Sequence[ConditionUnit],
    *,
    fold: int,
    majority_prediction: int,
    num_classes: int,
) -> list[dict[str, Any]]:
    rows = []
    for truth, predicted, probabilities, unit_index in zip(
        evaluation["truth"],
        evaluation["prediction"],
        evaluation["probabilities"],
        evaluation["unit_indices"],
    ):
        unit = units[int(unit_index)]
        row: dict[str, Any] = {
            "fold": fold,
            "unit_index": int(unit_index),
            "subject_id": unit.subject_id,
            "ma_mau": unit.ma_mau,
            "jar": unit.jar,
            "n_repeats": len(unit.record_indices),
            "repeats": "|".join(map(str, unit.repeats)),
            "true_label": int(truth),
            "predicted_label": int(predicted),
            "majority_prediction": int(majority_prediction),
        }
        for class_index in range(num_classes):
            row[f"prob_{class_index}"] = float(probabilities[class_index])
        rows.append(row)
    return rows


def baseline_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    field: str,
    num_classes: int,
) -> dict[str, Any]:
    _, metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in rows],
        [int(row[field]) for row in rows],
        num_classes=num_classes,
    )
    return metrics


def cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    num_classes: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    subjects = sorted(grouped)
    rng = np.random.default_rng(seed)
    model_scores = []
    majority_scores = []
    differences = []
    for _ in range(samples):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        sample_rows = [
            row
            for subject in chosen
            for row in grouped[str(subject)]
        ]
        _, model = confusion_and_metrics(
            [int(row["true_label"]) for row in sample_rows],
            [int(row["predicted_label"]) for row in sample_rows],
            num_classes=num_classes,
        )
        _, majority = confusion_and_metrics(
            [int(row["true_label"]) for row in sample_rows],
            [int(row["majority_prediction"]) for row in sample_rows],
            num_classes=num_classes,
        )
        model_score = float(model["balanced_accuracy"])
        majority_score = float(majority["balanced_accuracy"])
        model_scores.append(model_score)
        majority_scores.append(majority_score)
        differences.append(model_score - majority_score)

    def interval(values: Sequence[float]) -> dict[str, float]:
        low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
        return {
            "low_2_5": float(low),
            "median": float(median),
            "high_97_5": float(high),
        }

    return {
        "n_bootstrap": samples,
        "cluster": "subject_id",
        "model_balanced_accuracy": interval(model_scores),
        "majority_balanced_accuracy": interval(majority_scores),
        "model_minus_majority": interval(differences),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.representation not in {"legacy", "expression_v2"}:
        raise ValueError(f"Unknown representation {args.representation!r}")
    if args.manifest is None:
        args.manifest = Path(
            "output/video_jar_gnn/graph_manifest.csv"
            if args.representation == "legacy"
            else "output/video_jar_gnn/graph_manifest_expression_v2.csv"
        )
    canonical_rotation_requested = args.canonical_rotation
    if args.representation == "expression_v2":
        # Expression nodes are already pose-normalized by their extractor.
        # Legacy eye-coordinate rotation has no semantic meaning for them.
        args.canonical_rotation = False
        if args.relational_features:
            raise ValueError(
                "--relational-features is only valid for "
                "--representation legacy"
            )
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        raise ValueError("epoch settings must be positive")
    if args.min_epochs > args.epochs:
        raise ValueError("min-epochs cannot exceed epochs")
    if not 2 <= args.inner_folds:
        raise ValueError("inner-folds must be at least 2")
    if args.hidden_channels < 1 or args.batch_size < 1:
        raise ValueError("hidden-channels and batch-size must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if not 0.0 < args.temporal_crop_min <= 1.0:
        raise ValueError("temporal-crop-min must be in (0,1]")
    if not 0.0 <= args.repeat_dropout < 1.0:
        raise ValueError("repeat-dropout must be in [0,1)")
    if not 0.0 <= args.min_baseline_detection_ratio <= 1.0:
        raise ValueError("min-baseline-detection-ratio must be in [0,1]")
    if args.noise_std < 0 or args.num_workers < 0:
        raise ValueError("noise-std/num-workers invalid")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer settings invalid")
    if args.task == "binary" and args.objective == "ordinal":
        raise ValueError("ordinal objective is only valid for jar3")
    objective = (
        "ordinal"
        if args.objective == "auto" and args.task == "jar3"
        else "ce"
        if args.objective == "auto"
        else args.objective
    )
    preprocess_requested = args.preprocess
    if args.preprocess == "auto":
        args.preprocess = (
            "raw"
            if args.representation == "expression_v2"
            else "trial_delta"
            if args.task == "binary"
            else "water_delta"
        )
    requires_water = args.preprocess in {
        "water_delta",
        "absolute_water_delta",
    }
    set_seed(args.seed)
    device = choose_device(args.device)
    records, exclusions = load_graph_records(
        args.manifest,
        include_water=requires_water,
        min_detection_ratio=args.min_detection_ratio,
    )
    requires_neutral = args.preprocess in {
        "neutral_delta",
        "neutral_delta_motion",
    }
    neutral_quality_audit: dict[str, int] = {}
    if requires_neutral:
        records, neutral_quality_audit = filter_neutral_records(
            records,
            min_detection_ratio=args.min_baseline_detection_ratio,
            representation=args.representation,
        )
    store = AdvancedGraphStore(
        records,
        preprocess=args.preprocess,
        baseline_frames=args.baseline_frames,
        representation=args.representation,
        canonical_rotation=args.canonical_rotation,
        relational_features=args.relational_features,
    )
    units, condition_audit = build_condition_units(
        records,
        min_repeats=args.min_repeats,
        exclude_codes={WATER_CODE} if requires_water else None,
    )
    num_classes = 2 if args.task == "binary" else 3
    class_names = BINARY_NAMES if args.task == "binary" else JAR3_NAMES
    subject_count = len({unit.subject_id for unit in units})
    outer_splits = make_unit_splits(
        units,
        args.task,
        n_splits=args.cv_folds,
        seed=args.seed,
    )
    if args.fold_index is None:
        indexed_splits = list(enumerate(outer_splits))
    else:
        if not 0 <= args.fold_index < len(outer_splits):
            raise ValueError("fold-index outside available folds")
        indexed_splits = [(args.fold_index, outer_splits[args.fold_index])]

    if args.output_dir is None:
        geometry_name = (
            ("rot" if args.canonical_rotation else "no-rot")
            + "-"
            + ("relations" if args.relational_features else "no-relations")
        )
        configuration_dir = Path(
            "output/video_jar_gnn/runs_advanced"
        ) / args.task / (
            f"{args.representation}_{args.model}_{args.preprocess}_"
            f"{args.temporal_pooling}_"
            f"{geometry_name}_{args.aggregation}_video-only_{objective}"
        )
        fold_name = (
            "full"
            if args.fold_index is None
            else f"fold{args.fold_index + 1:02d}"
        )
        args.output_dir = configuration_dir / (
            f"{fold_name}_seed{args.seed}_{_run_signature(args)}"
        )
    _prepare_fresh_output(args.output_dir)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "objective_resolved": objective,
            "preprocess_requested": preprocess_requested,
            "preprocess_resolved": args.preprocess,
            "canonical_rotation_requested": canonical_rotation_requested,
            "canonical_rotation_resolved": args.canonical_rotation,
            "advanced_trainer_version": ADVANCED_TRAINER_VERSION,
            "device_resolved": str(device),
            "num_classes": num_classes,
            "class_names": class_names,
            "n_subjects": subject_count,
            "n_records_loaded": len(records),
            "n_supervised_conditions": len(units),
            "graph_shape_T_N_F": [
                store.sequence_length,
                store.num_nodes,
                store.num_features,
            ],
            "representation_schema": {
                **store.input_schema.to_dict(),
                "processed_feature_names": list(store.feature_names),
            },
            "input_contract": "video_graph_only",
            "uses_ma_mau_as_model_feature": False,
            "ma_mau_role": (
                "join/group key only; 605 also selects the unlabelled "
                "water reference when requested"
            ),
            "record_exclusions": exclusions,
            "condition_audit": condition_audit,
            "neutral_quality_audit": neutral_quality_audit,
            "water_used_as_unlabelled_reference": requires_water,
            "water_reference_scope": (
                "same subject, including outer-test subject, without water JAR"
                if requires_water
                else None
            ),
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (outer_train, outer_test) in indexed_splits:
        fold_number = fold_index + 1
        fold_seed = args.seed + fold_index * 1009
        inner_splits = make_unit_splits(
            units,
            args.task,
            n_splits=min(
                args.inner_folds,
                len(np.unique(_groups(units)[outer_train])),
            ),
            seed=fold_seed,
            subset=outer_train,
        )
        selected_epochs = []
        inner_histories: list[list[dict[str, Any]]] = []
        selection_rows = []
        for inner_index, (inner_train, inner_validation) in enumerate(
            inner_splits, start=1
        ):
            set_seed(fold_seed + inner_index * 101)
            best_epoch, history = select_epoch_count(
                store,
                units,
                inner_train,
                inner_validation,
                args,
                task=args.task,
                num_classes=num_classes,
                objective=objective,
                device=device,
            )
            selected_epochs.append(best_epoch)
            inner_histories.append(history)
            for row in history:
                selection_rows.append({"inner_fold": inner_index, **row})
        chosen_epochs, pooled_curve = pooled_inner_epoch(inner_histories)
        set_seed(fold_seed + 1)
        model, normalizer, refit_history = refit(
            store,
            units,
            outer_train,
            args,
            task=args.task,
            num_classes=num_classes,
            objective=objective,
            epochs=chosen_epochs,
            device=device,
        )
        weights = unit_class_weights(
            units,
            outer_train,
            task=args.task,
            num_classes=num_classes,
            device=device,
        )
        test_loader = _make_loader(
            store,
            units,
            outer_test,
            task=args.task,
            normalizer=normalizer,
            training=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            temporal_crop_min=1.0,
            noise_std=0.0,
            repeat_dropout=0.0,
            device=device,
        )
        evaluation = evaluate(
            model,
            test_loader,
            weights,
            device,
            num_classes=num_classes,
        )
        rows = prediction_rows(
            evaluation,
            units,
            fold=fold_number,
            majority_prediction=Counter(
                _labels(units, args.task)[outer_train].tolist()
            ).most_common(1)[0][0],
            num_classes=num_classes,
        )
        all_rows.extend(rows)
        train_subjects = sorted(
            {units[int(index)].subject_id for index in outer_train}
        )
        test_subjects = sorted(
            {units[int(index)].subject_id for index in outer_test}
        )
        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(fold_dir / "selection_history.csv", selection_rows)
        _write_csv(fold_dir / "selection_curve_pooled.csv", pooled_curve)
        _write_csv(fold_dir / "refit_history.csv", refit_history)
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "normalizer": normalizer.to_dict(),
            "input_contract": "video_graph_only",
            "uses_ma_mau_as_model_feature": False,
            "task": args.task,
            "class_names": class_names,
            "model": args.model,
            "representation": args.representation,
            "representation_schema": {
                **store.input_schema.to_dict(),
                "processed_feature_names": list(store.feature_names),
            },
            "preprocess": args.preprocess,
            "canonical_rotation": args.canonical_rotation,
            "relational_features": args.relational_features,
            "temporal_pooling": args.temporal_pooling,
            "aggregation": args.aggregation,
            "objective": objective,
            "advanced_trainer_version": ADVANCED_TRAINER_VERSION,
            "num_features": store.num_features,
            "num_nodes": store.num_nodes,
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "selected_epochs": chosen_epochs,
            "inner_selected_epochs": selected_epochs,
            "outer_train_subjects": train_subjects,
            "outer_test_subjects": test_subjects,
        }
        torch.save(checkpoint, fold_dir / "model.pt")
        fold_rows.append(
            {
                "fold": fold_number,
                "selected_epochs": chosen_epochs,
                "inner_selected_epochs": "|".join(map(str, selected_epochs)),
                "n_train_conditions": len(outer_train),
                "n_test_conditions": len(outer_test),
                "train_subjects": "|".join(train_subjects),
                "test_subjects": "|".join(test_subjects),
                **{
                    key: value
                    for key, value in evaluation["metrics"].items()
                    if isinstance(value, (int, float))
                },
            }
        )
        print(
            f"Fold {fold_number}/{len(outer_splits)}: "
            f"epochs={chosen_epochs} (inner={selected_epochs}), "
            f"test subjects={test_subjects}, "
            f"BAcc={evaluation['metrics']['balanced_accuracy']:.3f}, "
            f"macro-F1={evaluation['metrics']['macro_f1']:.3f}"
        )

    if args.fold_index is None:
        predicted = sorted(int(row["unit_index"]) for row in all_rows)
        if predicted != list(range(len(units))):
            raise AssertionError(
                "Full CV did not predict every condition exactly once"
            )
    confusion, metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in all_rows],
        [int(row["predicted_label"]) for row in all_rows],
        num_classes=num_classes,
    )
    metrics["mae"] = float(
        np.mean(
            [
                abs(int(row["true_label"]) - int(row["predicted_label"]))
                for row in all_rows
            ]
        )
    )
    metrics["quadratic_weighted_kappa"] = float(
        cohen_kappa_score(
            [int(row["true_label"]) for row in all_rows],
            [int(row["predicted_label"]) for row in all_rows],
            weights="quadratic",
        )
    )
    majority_metrics = baseline_metrics(
        all_rows, field="majority_prediction", num_classes=num_classes
    )
    bootstrap = (
        cluster_bootstrap(
            all_rows,
            num_classes=num_classes,
            samples=args.bootstrap_samples,
            seed=args.seed + 7001,
        )
        if args.fold_index is None
        else {}
    )
    _write_csv(args.output_dir / "predictions_condition.csv", all_rows)
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    np.save(args.output_dir / "confusion_condition.npy", confusion)
    plotted = _save_confusion_plot(
        confusion,
        class_names,
        args.output_dir / "confusion_condition.png",
    )
    summary = {
        "task": args.task,
        "class_names": class_names,
        "model": args.model,
        "representation": args.representation,
        "representation_schema": {
            **store.input_schema.to_dict(),
            "processed_feature_names": list(store.feature_names),
        },
        "preprocess": args.preprocess,
        "canonical_rotation": args.canonical_rotation,
        "relational_features": args.relational_features,
        "temporal_pooling": args.temporal_pooling,
        "aggregation": args.aggregation,
        "input_contract": "video_graph_only",
        "uses_ma_mau_as_model_feature": False,
        "objective": objective,
        "advanced_trainer_version": ADVANCED_TRAINER_VERSION,
        "partial_cv": args.fold_index is not None,
        "folds_run": [int(row["fold"]) for row in fold_rows],
        "n_parameters": count_parameters(model),
        "condition_level": {
            "metrics": metrics,
            "confusion": confusion.tolist(),
        },
        "baselines": {
            "majority": majority_metrics,
        },
        "subject_cluster_bootstrap": bootstrap,
        "condition_audit": condition_audit,
        "neutral_quality_audit": neutral_quality_audit,
        "record_exclusions": exclusions,
        "confusion_plot_written": plotted,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train condition-level repeat-set video models with subject-disjoint "
            "nested cross-validation."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Graph manifest. Defaults to graph_manifest.csv for legacy or "
            "graph_manifest_expression_v2.csv for expression_v2."
        ),
    )
    parser.add_argument(
        "--representation",
        choices=("legacy", "expression_v2"),
        default="legacy",
        help=(
            "Select the cache schema. Legacy caches remain the default; "
            "expression_v2 requires self-describing expression cache metadata."
        ),
    )
    parser.add_argument("--task", choices=("binary", "jar3"), default="jar3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model",
        choices=("stgcn", "tcn", "gru"),
        default="tcn",
        help="Default TCN is the compact video-only baseline.",
    )
    parser.add_argument(
        "--temporal-pooling",
        choices=("global", "segments"),
        default="segments",
        help="segments keeps separate 0-20%%, 20-50%% and 50-100%% responses.",
    )
    parser.add_argument(
        "--preprocess",
        choices=(
            "auto",
            "raw",
            "trial_delta",
            "trial_delta_motion",
            "neutral_delta",
            "neutral_delta_motion",
            "water_delta",
            "absolute_water_delta",
        ),
        default="auto",
        help=(
            "auto uses raw for expression_v2; for legacy it uses trial_delta "
            "for binary and water_delta for JAR3. "
            "absolute_water_delta keeps identity-heavy raw channels."
        ),
    )
    parser.add_argument(
        "--canonical-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove in-plane head roll from legacy cached coordinates. "
            "Always resolved off for pose-normalized expression_v2."
        ),
    )
    parser.add_argument(
        "--relational-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append legacy-only broadcast face-distance channels. Disabled "
            "by default because the current audit found subject confounding."
        ),
    )
    parser.add_argument(
        "--aggregation", choices=("mean", "mean_std"), default="mean"
    )
    parser.add_argument(
        "--objective", choices=("auto", "ce", "ordinal"), default="auto"
    )
    parser.add_argument("--baseline-frames", type=int, default=12)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
    parser.add_argument(
        "--min-baseline-detection-ratio",
        type=float,
        default=0.5,
        help="Minimum detected-face fraction for neutral baseline_seq repeats.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=12,
        help="Compact default for only 110--115 train conditions per outer fold.",
    )
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--temporal-crop-min", type=float, default=0.9)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--repeat-dropout", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    summary = run(args)
    metrics = summary["condition_level"]["metrics"]
    print(
        "Finished advanced condition CV: "
        f"BAcc={metrics['balanced_accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}"
    )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
