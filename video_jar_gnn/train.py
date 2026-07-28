"""Subject-disjoint cross-validation for binary or three-level JAR targets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from .constants import BINARY_NAMES, JAR3_NAMES
from .dataset import (
    FeatureStandardizer,
    GraphDataset,
    GraphRecord,
    GraphStore,
    load_graph_records,
)
from .model import FacialSTGCN, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms are not forced because a few CUDA convolution
    # kernels become prohibitively slow; seeds and split files are still saved.


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _labels(records: Sequence[GraphRecord], task: str) -> np.ndarray:
    return np.asarray([record.label_for(task) for record in records], dtype=np.int64)


def _groups(records: Sequence[GraphRecord]) -> np.ndarray:
    return np.asarray([record.subject_id for record in records])


def make_outer_splits(
    records: Sequence[GraphRecord],
    task: str,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = _labels(records, task)
    groups = _groups(records)
    n_groups = len(np.unique(groups))
    if not 2 <= n_splits <= n_groups:
        raise ValueError(f"n_splits must be in [2,{n_groups}], got {n_splits}")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    splits = [
        (np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64))
        for train, test in splitter.split(np.zeros(len(y)), y, groups)
    ]
    for train, test in splits:
        overlap = set(groups[train]).intersection(groups[test])
        if overlap:
            raise AssertionError(f"Subject leakage in outer split: {sorted(overlap)}")
    return splits


def make_inner_split(
    outer_train: np.ndarray,
    records: Sequence[GraphRecord],
    task: str,
    *,
    seed: int,
    requested_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    y_all = _labels(records, task)
    groups_all = _groups(records)
    y = y_all[outer_train]
    groups = groups_all[outer_train]
    n_groups = len(np.unique(groups))
    n_splits = min(requested_splits, n_groups)
    if n_splits < 2:
        raise ValueError("At least two training subjects are needed for inner validation")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    inner_train_local, validation_local = next(
        splitter.split(np.zeros(len(y)), y, groups)
    )
    inner_train = outer_train[inner_train_local]
    validation = outer_train[validation_local]
    overlap = set(groups_all[inner_train]).intersection(groups_all[validation])
    if overlap:
        raise AssertionError(f"Subject leakage in inner split: {sorted(overlap)}")
    return inner_train, validation


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"A training fold is missing a class: counts={counts.tolist()}")
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_and_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    num_classes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        confusion[int(actual), int(predicted)] += 1
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion).astype(np.float64)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    supported = support > 0
    # Match sklearn's default macro-F1 semantics: include classes occurring
    # in either y_true or y_pred. Thus predicting a class absent from a LOSO
    # test subject is still penalized, while a class absent on both sides is
    # not treated as an impossible false failure.
    f1_relevant = (support > 0) | (predicted_count > 0)
    metrics = {
        "n": int(len(truth)),
        "accuracy": float((truth == prediction).mean()) if len(truth) else 0.0,
        "balanced_accuracy": float(recall[supported].mean()) if supported.any() else 0.0,
        "macro_f1": float(f1[f1_relevant].mean()) if f1_relevant.any() else 0.0,
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.astype(int).tolist(),
    }
    return confusion, metrics


def _make_loader(
    store: GraphStore,
    indices: Sequence[int],
    *,
    task: str,
    normalizer: FeatureStandardizer,
    batch_size: int,
    training: bool,
    num_workers: int,
    temporal_crop_min: float,
    noise_std: float,
    device: torch.device,
) -> DataLoader:
    dataset = GraphDataset(
        store,
        indices,
        task=task,
        normalizer=normalizer,
        training=training,
        temporal_crop_min=temporal_crop_min if training else 1.0,
        noise_std=noise_std if training else 0.0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for graph, adjacency, labels, _ in loader:
        graph = graph.to(device, non_blocking=True)
        adjacency = adjacency.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph, adjacency)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    truth: list[int] = []
    prediction: list[int] = []
    probabilities: list[list[float]] = []
    record_indices: list[int] = []
    for graph, adjacency, labels, indices in loader:
        graph = graph.to(device, non_blocking=True)
        adjacency = adjacency.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(graph, adjacency)
        loss = criterion(logits, labels_device)
        probs = torch.softmax(logits, dim=1)
        predicted = probs.argmax(dim=1)
        total_loss += float(loss.item()) * labels.shape[0]
        total += labels.shape[0]
        truth.extend(labels.tolist())
        prediction.extend(predicted.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
        record_indices.extend(indices.tolist())
    confusion, metrics = confusion_and_metrics(
        truth, prediction, num_classes=num_classes
    )
    metrics["loss"] = total_loss / max(total, 1)
    return {
        "truth": truth,
        "prediction": prediction,
        "probabilities": probabilities,
        "record_indices": record_indices,
        "confusion": confusion,
        "metrics": metrics,
    }


def condition_metrics_from_evaluation(
    evaluation: dict[str, Any],
    records: Sequence[GraphRecord],
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Aggregate repeat probabilities before computing condition-level metrics."""
    grouped: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for truth, probabilities, index in zip(
        evaluation["truth"],
        evaluation["probabilities"],
        evaluation["record_indices"],
    ):
        record = records[int(index)]
        grouped[(record.subject_id, record.ma_mau)].append(
            (int(truth), np.asarray(probabilities, dtype=np.float64))
        )
    condition_truth: list[int] = []
    condition_prediction: list[int] = []
    for key, values in grouped.items():
        truths = {truth for truth, _ in values}
        if len(truths) != 1:
            raise AssertionError(f"Inconsistent labels within condition {key}: {truths}")
        mean_probability = np.stack(
            [probabilities for _, probabilities in values], axis=0
        ).mean(axis=0)
        condition_truth.append(next(iter(truths)))
        condition_prediction.append(int(mean_probability.argmax()))
    _, metrics = confusion_and_metrics(
        condition_truth,
        condition_prediction,
        num_classes=num_classes,
    )
    return metrics


def _new_model(
    store: GraphStore,
    *,
    num_classes: int,
    hidden_channels: int,
    dropout: float,
    device: torch.device,
) -> FacialSTGCN:
    return FacialSTGCN(
        num_features=store.num_features,
        num_nodes=store.num_nodes,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        dropout=dropout,
    ).to(device)


def select_epoch_count(
    store: GraphStore,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    task: str,
    num_classes: int,
    device: torch.device,
    hidden_channels: int,
    dropout: float,
    batch_size: int,
    num_workers: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    temporal_crop_min: float,
    noise_std: float,
) -> tuple[int, list[dict[str, Any]]]:
    normalizer = FeatureStandardizer.fit(store, train_indices)
    train_loader = _make_loader(
        store,
        train_indices,
        task=task,
        normalizer=normalizer,
        batch_size=batch_size,
        training=True,
        num_workers=num_workers,
        temporal_crop_min=temporal_crop_min,
        noise_std=noise_std,
        device=device,
    )
    validation_loader = _make_loader(
        store,
        validation_indices,
        task=task,
        normalizer=normalizer,
        batch_size=batch_size,
        training=False,
        num_workers=num_workers,
        temporal_crop_min=1.0,
        noise_std=0.0,
        device=device,
    )
    model = _new_model(
        store,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        dropout=dropout,
        device=device,
    )
    weights = class_weights(
        _labels(store.records, task)[train_indices], num_classes, device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = 1
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = _train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        validation = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            num_classes=num_classes,
        )
        validation_condition = condition_metrics_from_evaluation(
            validation,
            store.records,
            num_classes=num_classes,
        )
        score = float(validation_condition["balanced_accuracy"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation["metrics"]["loss"],
                "validation_accuracy": validation["metrics"]["accuracy"],
                "validation_trial_balanced_accuracy": validation["metrics"][
                    "balanced_accuracy"
                ],
                "validation_trial_macro_f1": validation["metrics"]["macro_f1"],
                "validation_condition_balanced_accuracy": score,
                "validation_condition_macro_f1": validation_condition["macro_f1"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    return best_epoch, history


def refit_outer_train(
    store: GraphStore,
    outer_train: np.ndarray,
    *,
    task: str,
    num_classes: int,
    device: torch.device,
    epochs: int,
    hidden_channels: int,
    dropout: float,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    temporal_crop_min: float,
    noise_std: float,
) -> tuple[FacialSTGCN, FeatureStandardizer, list[dict[str, Any]]]:
    normalizer = FeatureStandardizer.fit(store, outer_train)
    loader = _make_loader(
        store,
        outer_train,
        task=task,
        normalizer=normalizer,
        batch_size=batch_size,
        training=True,
        num_workers=num_workers,
        temporal_crop_min=temporal_crop_min,
        noise_std=noise_std,
        device=device,
    )
    model = _new_model(
        store,
        num_classes=num_classes,
        hidden_channels=hidden_channels,
        dropout=dropout,
        device=device,
    )
    weights = class_weights(
        _labels(store.records, task)[outer_train], num_classes, device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    for epoch in range(1, epochs + 1):
        loss, accuracy = _train_epoch(model, loader, optimizer, criterion, device)
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    return model, normalizer, history


def _condition_baseline_lookup(
    records: Sequence[GraphRecord],
    train_indices: Sequence[int],
    task: str,
) -> tuple[dict[int, int], int]:
    labels = [records[int(index)].label_for(task) for index in train_indices]
    fallback = Counter(labels).most_common(1)[0][0]
    per_code: dict[int, Counter[int]] = defaultdict(Counter)
    for index in train_indices:
        record = records[int(index)]
        per_code[record.ma_mau][record.label_for(task)] += 1
    lookup = {
        code: counts.most_common(1)[0][0] for code, counts in per_code.items()
    }
    return lookup, fallback


def _prediction_rows(
    result: dict[str, Any],
    records: Sequence[GraphRecord],
    *,
    fold: int,
    outer_train: Sequence[int],
    task: str,
    num_classes: int,
) -> list[dict[str, Any]]:
    lookup, fallback = _condition_baseline_lookup(records, outer_train, task)
    majority = fallback
    rows = []
    for true, predicted, probabilities, index in zip(
        result["truth"],
        result["prediction"],
        result["probabilities"],
        result["record_indices"],
    ):
        record = records[int(index)]
        row = {
            "fold": fold,
            "record_index": int(index),
            "sample_id": record.sample_id,
            "subject_id": record.subject_id,
            "ma_mau": record.ma_mau,
            "repeat": record.repeat,
            "jar": record.jar,
            "true_label": int(true),
            "predicted_label": int(predicted),
            "majority_baseline": int(majority),
            "condition_baseline": int(lookup.get(record.ma_mau, fallback)),
            "detection_ratio": record.detection_ratio,
        }
        for class_index in range(num_classes):
            row[f"prob_{class_index}"] = float(probabilities[class_index])
        rows.append(row)
    return rows


def aggregate_condition_predictions(
    trial_rows: Sequence[dict[str, Any]],
    *,
    num_classes: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["subject_id"]), int(row["ma_mau"]))].append(row)
    result = []
    for (subject_id, code), group in sorted(grouped.items()):
        truths = {int(row["true_label"]) for row in group}
        folds = {int(row["fold"]) for row in group}
        if len(truths) != 1 or len(folds) != 1:
            raise AssertionError(
                f"Inconsistent group ({subject_id},{code}): truths={truths}, folds={folds}"
            )
        probabilities = np.asarray(
            [
                [float(row[f"prob_{class_index}"]) for class_index in range(num_classes)]
                for row in group
            ],
            dtype=np.float64,
        ).mean(axis=0)
        row = {
            "fold": next(iter(folds)),
            "subject_id": subject_id,
            "ma_mau": code,
            "n_repeats": len(group),
            "true_label": next(iter(truths)),
            "predicted_label": int(probabilities.argmax()),
            "majority_baseline": Counter(
                int(item["majority_baseline"]) for item in group
            ).most_common(1)[0][0],
            "condition_baseline": Counter(
                int(item["condition_baseline"]) for item in group
            ).most_common(1)[0][0],
        }
        for class_index, probability in enumerate(probabilities):
            row[f"prob_{class_index}"] = float(probability)
        result.append(row)
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion_plot(
    confusion: np.ndarray,
    names: dict[int, str],
    path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_sums > 0,
    )
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = [names[index] for index in range(len(names))]
    axis.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Subject-disjoint cross-validation")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            axis.text(
                j,
                i,
                f"{normalized[i, j] * 100:.1f}%\n(n={confusion[i, j]})",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _baseline_metrics(
    rows: Sequence[dict[str, Any]],
    field: str,
    *,
    num_classes: int,
) -> dict[str, Any]:
    _, metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in rows],
        [int(row[field]) for row in rows],
        num_classes=num_classes,
    )
    return metrics


def run_cross_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir is None:
        args.output_dir = Path(f"output/video_jar_gnn/runs/{args.task}")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("epochs and patience must be positive")
    if args.batch_size < 1 or args.hidden_channels < 1:
        raise ValueError("batch-size and hidden-channels must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if not 0.0 < args.temporal_crop_min <= 1.0:
        raise ValueError("temporal-crop-min must be in (0,1]")
    if args.noise_std < 0:
        raise ValueError("noise-std must be non-negative")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if not 0.0 <= args.min_detection_ratio <= 1.0:
        raise ValueError("min-detection-ratio must be in [0,1]")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError(
            "learning-rate must be positive and weight-decay non-negative"
        )
    set_seed(args.seed)
    device = choose_device(args.device)
    records, exclusions = load_graph_records(
        args.manifest,
        include_water=args.include_water,
        min_detection_ratio=args.min_detection_ratio,
    )
    store = GraphStore(records)
    num_classes = 2 if args.task == "binary" else 3
    class_names = BINARY_NAMES if args.task == "binary" else JAR3_NAMES
    subject_count = len({record.subject_id for record in records})
    n_splits = subject_count if args.loso else args.cv_folds
    outer_splits = make_outer_splits(
        records,
        args.task,
        n_splits=n_splits,
        seed=args.seed,
    )
    if args.fold_index is not None:
        if not 0 <= args.fold_index < len(outer_splits):
            raise ValueError(
                f"fold-index must be in [0,{len(outer_splits) - 1}]"
            )
        indexed_splits = [(args.fold_index, outer_splits[args.fold_index])]
    else:
        indexed_splits = list(enumerate(outer_splits))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    config.update(
        {
            "device_resolved": str(device),
            "n_subjects": subject_count,
            "n_records": len(records),
            "num_classes": num_classes,
            "class_names": class_names,
            "exclusions": exclusions,
            "graph_shape_T_N_F": [
                store.sequence_length,
                store.num_nodes,
                store.num_features,
            ],
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_indices = np.arange(len(records), dtype=np.int64)
    labels_all = _labels(records, args.task)

    for fold_index, (outer_train, outer_test) in indexed_splits:
        fold_number = fold_index + 1
        fold_seed = args.seed + fold_index * 1009
        set_seed(fold_seed)
        inner_train, validation = make_inner_split(
            outer_train,
            records,
            args.task,
            seed=fold_seed,
        )
        train_subjects = sorted({records[int(i)].subject_id for i in outer_train})
        test_subjects = sorted({records[int(i)].subject_id for i in outer_test})
        print(
            f"Fold {fold_number}/{len(outer_splits)}: "
            f"train subjects={len(train_subjects)}, test={test_subjects}"
        )

        best_epoch, selection_history = select_epoch_count(
            store,
            inner_train,
            validation,
            task=args.task,
            num_classes=num_classes,
            device=device,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temporal_crop_min=args.temporal_crop_min,
            noise_std=args.noise_std,
        )
        # Refit from scratch on every non-test subject for the selected number
        # of epochs. The outer test subjects never influence normalization,
        # early stopping, class weights or model parameters.
        set_seed(fold_seed + 1)
        model, normalizer, refit_history = refit_outer_train(
            store,
            outer_train,
            task=args.task,
            num_classes=num_classes,
            device=device,
            epochs=best_epoch,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temporal_crop_min=args.temporal_crop_min,
            noise_std=args.noise_std,
        )
        weights = class_weights(labels_all[outer_train], num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        test_loader = _make_loader(
            store,
            outer_test,
            task=args.task,
            normalizer=normalizer,
            batch_size=args.batch_size,
            training=False,
            num_workers=args.num_workers,
            temporal_crop_min=1.0,
            noise_std=0.0,
            device=device,
        )
        test_result = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=num_classes,
        )
        prediction_rows = _prediction_rows(
            test_result,
            records,
            fold=fold_number,
            outer_train=outer_train,
            task=args.task,
            num_classes=num_classes,
        )
        all_trial_rows.extend(prediction_rows)
        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(fold_dir / "selection_history.csv", selection_history)
        _write_csv(fold_dir / "refit_history.csv", refit_history)
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "normalizer": normalizer.to_dict(),
            "task": args.task,
            "class_names": class_names,
            "num_features": store.num_features,
            "num_nodes": store.num_nodes,
            "hidden_channels": args.hidden_channels,
            "dropout": args.dropout,
            "selected_epochs": best_epoch,
            "outer_train_subjects": train_subjects,
            "outer_test_subjects": test_subjects,
        }
        torch.save(checkpoint, fold_dir / "model.pt")
        fold_row = {
            "fold": fold_number,
            "selected_epochs": best_epoch,
            "n_train": len(outer_train),
            "n_test": len(outer_test),
            "train_subjects": "|".join(train_subjects),
            "test_subjects": "|".join(test_subjects),
            **{
                f"trial_{key}": value
                for key, value in test_result["metrics"].items()
                if isinstance(value, (int, float))
            },
        }
        fold_rows.append(fold_row)
        print(
            f"  epochs={best_epoch}, "
            f"test BAcc={test_result['metrics']['balanced_accuracy']:.3f}, "
            f"macro-F1={test_result['metrics']['macro_f1']:.3f}"
        )

    if args.fold_index is None:
        predicted_indices = sorted(int(row["record_index"]) for row in all_trial_rows)
        if predicted_indices != all_indices.tolist():
            raise AssertionError("Full CV did not predict every included graph exactly once")

    condition_rows = aggregate_condition_predictions(
        all_trial_rows, num_classes=num_classes
    )
    trial_confusion, trial_metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in all_trial_rows],
        [int(row["predicted_label"]) for row in all_trial_rows],
        num_classes=num_classes,
    )
    condition_confusion, condition_metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in condition_rows],
        [int(row["predicted_label"]) for row in condition_rows],
        num_classes=num_classes,
    )
    for row in fold_rows:
        fold_conditions = [
            item for item in condition_rows if int(item["fold"]) == int(row["fold"])
        ]
        _, metrics = confusion_and_metrics(
            [int(item["true_label"]) for item in fold_conditions],
            [int(item["predicted_label"]) for item in fold_conditions],
            num_classes=num_classes,
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[f"condition_{key}"] = value

    _write_csv(args.output_dir / "predictions_trial.csv", all_trial_rows)
    _write_csv(args.output_dir / "predictions_condition.csv", condition_rows)
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    np.save(args.output_dir / "confusion_trial.npy", trial_confusion)
    np.save(args.output_dir / "confusion_condition.npy", condition_confusion)
    plotted_trial = _save_confusion_plot(
        trial_confusion, class_names, args.output_dir / "confusion_trial.png"
    )
    plotted_condition = _save_confusion_plot(
        condition_confusion,
        class_names,
        args.output_dir / "confusion_condition.png",
    )

    summary = {
        "task": args.task,
        "class_names": class_names,
        "include_water": args.include_water,
        "partial_cv": args.fold_index is not None,
        "folds_run": [int(row["fold"]) for row in fold_rows],
        "n_parameters": count_parameters(model),
        "trial_level": {
            "metrics": trial_metrics,
            "confusion": trial_confusion.tolist(),
        },
        "subject_condition_level": {
            "metrics": condition_metrics,
            "confusion": condition_confusion.tolist(),
        },
        "baselines": {
            "trial_majority": _baseline_metrics(
                all_trial_rows, "majority_baseline", num_classes=num_classes
            ),
            "trial_condition_only": _baseline_metrics(
                all_trial_rows, "condition_baseline", num_classes=num_classes
            ),
            "condition_majority": _baseline_metrics(
                condition_rows, "majority_baseline", num_classes=num_classes
            ),
            "condition_condition_only": _baseline_metrics(
                condition_rows, "condition_baseline", num_classes=num_classes
            ),
        },
        "confusion_plots_written": {
            "trial": plotted_trial,
            "condition": plotted_condition,
        },
        "exclusions": exclusions,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train facial ST-GCN with subject-disjoint nested CV."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/video_jar_gnn/graph_manifest.csv"),
    )
    parser.add_argument("--task", choices=("binary", "jar3"), default="jar3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--include-water",
        action="store_true",
        help="Include sample 605. Both tasks exclude water by default.",
    )
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Use leave-one-subject-out instead of 5-fold grouped CV.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        help=(
            "Run one zero-based outer fold for development. Use a distinct "
            "--output-dir for every concurrent process."
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--temporal-crop-min", type=float, default=0.9)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0...")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    summary = run_cross_validation(args)
    condition = summary["subject_condition_level"]["metrics"]
    print(
        "Finished: "
        f"condition-level balanced_accuracy={condition['balanced_accuracy']:.3f}, "
        f"macro_f1={condition['macro_f1']:.3f}"
    )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
