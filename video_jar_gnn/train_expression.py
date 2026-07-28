"""Leakage-safe classical baseline for ``expression_v2`` video graphs.

The independent sample is one ``subject_id × ma_mau`` condition.  Repeated
videos are summarized on their own real ``target_lsl`` timelines and averaged
only after summarization.  Response-window, feature-count and regularization
selection all happen inside subject-disjoint inner cross-validation.

``ma_mau`` is used only to group the five repeats and to identify rows in the
outputs.  It is never included in the feature matrix.  Water (605) is excluded
from supervised training because its JAR label is not a sweet-sample target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .advanced_dataset import ConditionUnit, build_condition_units
from .constants import BINARY_NAMES, JAR3_NAMES, WATER_CODE
from .dataset import GraphRecord, load_graph_records
from .expression import EXPRESSION_FEATURES, EXPRESSION_NODES
from .expression_audit import (
    DEFAULT_WINDOWS,
    ExpressionCache,
    ExpressionSchema,
    ResponseWindow,
    load_expression_cache,
    parse_response_window,
    summarize_expression_window,
    summary_feature_rows,
)
from .train import _save_confusion_plot, _write_csv, confusion_and_metrics
from .train_advanced import make_unit_splits


REPRESENTATION = "expression_v2"
DEFAULT_K_GRID = (8, 16, 32, 64)
DEFAULT_C_GRID = (0.01, 0.1, 1.0)


def _labels(units: Sequence[ConditionUnit], task: str) -> np.ndarray:
    return np.asarray(
        [unit.label_for(task) for unit in units], dtype=np.int64
    )


def _groups(units: Sequence[ConditionUnit]) -> np.ndarray:
    return np.asarray(
        [unit.subject_id for unit in units], dtype=object
    )


def _run_signature(args: argparse.Namespace) -> str:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "output_dir":
            continue
        if key == "windows":
            payload[key] = [window.slug for window in value]
        elif isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
        else:
            payload[key] = value
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _prepare_fresh_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {path}. Choose a new "
            "--output-dir so folds and configurations cannot be mixed."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_labelled_expression_caches(
    records: Sequence[GraphRecord],
) -> tuple[list[ExpressionCache], ExpressionSchema]:
    """Load expression caches in manifest-record order and enforce v2 schema."""
    caches: list[ExpressionCache] = []
    expected_schema: ExpressionSchema | None = None
    expected_tail = (len(EXPRESSION_NODES), len(EXPRESSION_FEATURES))
    for record in records:
        cache = load_expression_cache(
            record.graph_path,
            identity_override={
                "subject_id": record.subject_id,
                "condition_id": str(record.ma_mau),
                "repeat": record.repeat,
            },
        )
        if cache.graph.shape[1:] != expected_tail:
            raise ValueError(
                f"{record.graph_path}: expression_v2 graph must have shape "
                f"[T,{expected_tail[0]},{expected_tail[1]}], got "
                f"{cache.graph.shape}"
            )
        if expected_schema is None:
            expected_schema = cache.schema
        elif cache.schema != expected_schema:
            raise ValueError(
                f"{record.graph_path}: expression schema differs from "
                f"{caches[0].path}"
            )
        caches.append(cache)
    if not caches or expected_schema is None:
        raise ValueError("No expression_v2 cache was loaded")
    return caches, expected_schema


def _finite_repeat_mean(rows: np.ndarray) -> np.ndarray:
    """Average repeats column-wise while retaining all-missing columns."""
    values = np.asarray(rows, dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    total = np.where(finite, values, 0.0).sum(axis=0)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result.astype(np.float32)


def build_expression_feature_matrices(
    caches: Sequence[ExpressionCache],
    units: Sequence[ConditionUnit],
    windows: Sequence[ResponseWindow],
) -> dict[str, np.ndarray]:
    """Build one summary row per condition for every candidate window.

    Each repeat is cropped using its own real-time ``target_lsl`` offsets.
    This avoids assuming either 60 video rows or 100 EEG rows per second.
    """
    if not caches:
        raise ValueError("Cannot summarize an empty cache collection")
    unique_windows = tuple(dict.fromkeys(windows))
    if len(unique_windows) != len(windows):
        raise ValueError("Response windows must not contain duplicates")
    schema = caches[0].schema
    matrices: dict[str, list[np.ndarray]] = {
        window.slug: [] for window in windows
    }
    for unit in units:
        repeat_caches = [caches[index] for index in unit.record_indices]
        for window in windows:
            repeat_rows = np.stack(
                [
                    summarize_expression_window(
                        cache.graph,
                        cache.time_seconds,
                        cache.schema,
                        window,
                    )
                    for cache in repeat_caches
                ],
                axis=0,
            )
            matrices[window.slug].append(_finite_repeat_mean(repeat_rows))
    result = {
        slug: np.stack(rows, axis=0).astype(np.float32)
        for slug, rows in matrices.items()
    }
    expected_width = len(summary_feature_rows(schema))
    for slug, matrix in result.items():
        if matrix.shape != (len(units), expected_width):
            raise AssertionError(
                f"Window {slug}: summary shape {matrix.shape}, expected "
                f"({len(units)},{expected_width})"
            )
    return result


def make_expression_pipeline(
    *,
    k_features: int,
    c_value: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    """Create a low-capacity pipeline whose learned steps are fold-local."""
    if k_features < 0:
        raise ValueError("k_features must be non-negative (0 means all)")
    selector_k: int | str = "all" if k_features == 0 else int(k_features)
    return Pipeline(
        (
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("standardizer", StandardScaler()),
            (
                "selector",
                SelectKBest(score_func=f_classif, k=selector_k),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=int(max_iter),
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            ),
        )
    )


def fit_expression_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
    indices: Sequence[int],
    *,
    k_features: int,
    c_value: float,
    max_iter: int,
    seed: int,
    num_classes: int,
) -> Pipeline:
    """Fit imputation through classification on training rows only."""
    train_indices = np.asarray(indices, dtype=np.int64)
    observed = np.bincount(
        labels[train_indices], minlength=num_classes
    )
    if np.any(observed == 0):
        raise ValueError(
            "An expression training fold is missing a class: "
            f"{observed.tolist()}"
        )
    pipeline = make_expression_pipeline(
        k_features=k_features,
        c_value=c_value,
        max_iter=max_iter,
        seed=seed,
    )
    try:
        pipeline.fit(features[train_indices], labels[train_indices])
    except ValueError as error:
        message = str(error)
        if "k should be" in message or "k=" in message:
            raise ValueError(
                f"k={k_features} exceeds the usable feature count in this "
                "training fold; choose a smaller --k-grid"
            ) from error
        raise
    return pipeline


def _predict_probabilities(
    pipeline: Pipeline,
    features: np.ndarray,
    indices: Sequence[int],
    *,
    num_classes: int,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    raw = np.asarray(
        pipeline.predict_proba(features[selected]), dtype=np.float64
    )
    classes = np.asarray(
        pipeline.named_steps["classifier"].classes_, dtype=np.int64
    )
    probabilities = np.zeros(
        (len(selected), num_classes), dtype=np.float64
    )
    probabilities[:, classes] = raw
    sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(sums <= 0) or not np.isfinite(probabilities).all():
        raise ValueError("Classifier returned invalid probabilities")
    return probabilities / sums


def _probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    _, metrics = confusion_and_metrics(
        labels, predictions, num_classes=num_classes
    )
    epsilon = np.finfo(np.float64).tiny
    negative_log_likelihood = -np.log(
        np.clip(
            probabilities[np.arange(len(labels)), labels],
            epsilon,
            1.0,
        )
    ).mean()
    return {
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "negative_log_likelihood": float(negative_log_likelihood),
    }


def _valid_inner_splits(
    units: Sequence[ConditionUnit],
    task: str,
    outer_train: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
    num_classes: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use the largest fold count with every class in each training part."""
    subject_count = len(np.unique(_groups(units)[outer_train]))
    labels = _labels(units, task)
    for folds in range(min(requested_folds, subject_count), 1, -1):
        splits = make_unit_splits(
            units,
            task,
            n_splits=folds,
            seed=seed,
            subset=outer_train,
        )
        if all(
            np.all(
                np.bincount(labels[train], minlength=num_classes) > 0
            )
            for train, _ in splits
        ):
            validation = np.concatenate([valid for _, valid in splits])
            if sorted(validation.tolist()) != sorted(outer_train.tolist()):
                raise AssertionError(
                    "Inner validation rows must cover outer training once"
                )
            return splits
    raise ValueError(
        "Could not construct subject-disjoint inner folds whose training "
        "part contains every class"
    )


def assert_subject_disjoint_splits(
    units: Sequence[ConditionUnit],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Raise if any train/test split shares a participant."""
    groups = _groups(units)
    for split_index, (train, test) in enumerate(splits, start=1):
        overlap = set(groups[train]).intersection(groups[test])
        if overlap:
            raise AssertionError(
                f"Subject leakage in split {split_index}: {sorted(overlap)}"
            )


def expression_candidate_oof_probabilities(
    feature_matrices: Mapping[str, np.ndarray],
    labels: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    windows: Sequence[ResponseWindow],
    k_grid: Sequence[int],
    c_grid: Sequence[float],
    *,
    num_classes: int,
    max_iter: int,
    seed: int,
) -> dict[tuple[str, int, float], np.ndarray]:
    """Generate inner-CV probabilities for every window/K/C candidate."""
    candidates = [
        (window.slug, int(k_features), float(c_value))
        for window in windows
        for k_features in k_grid
        for c_value in c_grid
    ]
    result = {
        candidate: np.full(
            (len(labels), num_classes), np.nan, dtype=np.float64
        )
        for candidate in candidates
    }
    seen = np.zeros(len(labels), dtype=np.int64)
    for split_index, (train, validation) in enumerate(inner_splits):
        seen[validation] += 1
        for candidate_index, candidate in enumerate(candidates):
            window_slug, k_features, c_value = candidate
            pipeline = fit_expression_pipeline(
                feature_matrices[window_slug],
                labels,
                train,
                k_features=k_features,
                c_value=c_value,
                max_iter=max_iter,
                seed=seed + split_index * 1009 + candidate_index,
                num_classes=num_classes,
            )
            result[candidate][validation] = _predict_probabilities(
                pipeline,
                feature_matrices[window_slug],
                validation,
                num_classes=num_classes,
            )
    validation_union = np.concatenate(
        [validation for _, validation in inner_splits]
    )
    if (
        len(validation_union) != len(np.unique(validation_union))
        or np.any(seen[validation_union] != 1)
    ):
        raise AssertionError(
            "Inner CV must predict every outer-training condition once"
        )
    for candidate, probabilities in result.items():
        if not np.isfinite(probabilities[validation_union]).all():
            raise AssertionError(
                f"Missing inner OOF probability for candidate={candidate}"
            )
    return result


def select_expression_candidate(
    oof: Mapping[tuple[str, int, float], np.ndarray],
    labels: np.ndarray,
    outer_train: Sequence[int],
    windows: Sequence[ResponseWindow],
    *,
    num_classes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose window/K/C using outer-training OOF labels and nothing else."""
    train = np.asarray(outer_train, dtype=np.int64)
    window_lookup = {window.slug: window for window in windows}
    window_order = {
        window.slug: index for index, window in enumerate(windows)
    }
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    rows: list[dict[str, Any]] = []
    ordered_candidates = sorted(
        oof,
        key=lambda candidate: (
            window_order[candidate[0]],
            candidate[1],
            candidate[2],
        ),
    )
    for window_slug, k_features, c_value in ordered_candidates:
        probabilities = oof[(window_slug, k_features, c_value)][train]
        metrics = _probability_metrics(
            labels[train],
            probabilities,
            num_classes=num_classes,
        )
        window = window_lookup[window_slug]
        row = {
            "window": window_slug,
            "window_start_seconds": window.start_seconds,
            "window_end_seconds": window.end_seconds,
            "k_features": int(k_features),
            "selection_strategy": (
                "all_after_variance" if k_features == 0 else "kbest"
            ),
            "C": float(c_value),
            **metrics,
        }
        rows.append(row)
        # Exact ties favour stronger regularization, fewer selected features,
        # then a shorter/earlier response window.
        effective_k = 10**9 if k_features == 0 else int(k_features)
        key = (
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            -metrics["negative_log_likelihood"],
            -float(effective_k),
            -float(c_value),
            -window.duration_seconds,
            -window.start_seconds,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "window": window_slug,
                "window_start_seconds": window.start_seconds,
                "window_end_seconds": window.end_seconds,
                "k_features": int(k_features),
                "C": float(c_value),
            }
    if best is None:
        raise ValueError("No expression candidate was evaluated")
    return best, rows


def selected_original_feature_indices(pipeline: Pipeline) -> np.ndarray:
    """Map selector support back to expression summary columns."""
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["selector"]
    after_variance = np.flatnonzero(variance.get_support())
    return after_variance[
        np.asarray(selector.get_support(), dtype=bool)
    ]


def selected_feature_details(
    pipeline: Pipeline,
    schema: ExpressionSchema,
) -> list[dict[str, Any]]:
    """Describe selected columns using node, signal and statistic names."""
    descriptors = summary_feature_rows(schema)
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["selector"]
    after_variance = np.flatnonzero(variance.get_support())
    support = np.asarray(selector.get_support(), dtype=bool)
    original_indices = after_variance[support]
    scores = np.asarray(selector.scores_, dtype=np.float64)[support]
    sortable = np.nan_to_num(
        scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf
    )
    order = np.argsort(-sortable, kind="stable")
    result: list[dict[str, Any]] = []
    for rank, position in enumerate(order, start=1):
        feature_index = int(original_indices[position])
        score = float(scores[position])
        result.append(
            {
                "rank_by_training_f_score": rank,
                **descriptors[feature_index],
                "training_f_score": score if math.isfinite(score) else "",
            }
        )
    return result


def _complete_metrics(
    rows: Sequence[dict[str, Any]],
    *,
    num_classes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    truth = np.asarray(
        [int(row["true_label"]) for row in rows], dtype=np.int64
    )
    prediction = np.asarray(
        [int(row["predicted_label"]) for row in rows], dtype=np.int64
    )
    confusion, metrics = confusion_and_metrics(
        truth, prediction, num_classes=num_classes
    )
    metrics["mae"] = float(np.mean(np.abs(truth - prediction)))
    kappa = cohen_kappa_score(truth, prediction, weights="quadratic")
    metrics["quadratic_weighted_kappa"] = (
        float(kappa) if np.isfinite(kappa) else 0.0
    )
    return confusion, metrics


def _interval(values: Sequence[float]) -> dict[str, float]:
    low, median, high = np.percentile(values, [2.5, 50.0, 97.5])
    return {
        "low_2_5": float(low),
        "median": float(median),
        "high_97_5": float(high),
    }


def subject_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    num_classes: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap whole subjects for model and outer-train majority scores."""
    if samples <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    subjects = sorted(grouped)
    if not subjects:
        return {}
    rng = np.random.default_rng(seed)
    model_scores: list[float] = []
    majority_scores: list[float] = []
    differences: list[float] = []
    attempts = 0
    while (
        len(model_scores) < samples
        and attempts < max(samples * 20, 100)
    ):
        attempts += 1
        selected_subjects = rng.choice(
            subjects, size=len(subjects), replace=True
        )
        sampled = [
            row
            for subject in selected_subjects
            for row in grouped[str(subject)]
        ]
        support = np.bincount(
            [int(row["true_label"]) for row in sampled],
            minlength=num_classes,
        )
        if np.any(support == 0):
            continue
        _, model_metrics = _complete_metrics(
            sampled, num_classes=num_classes
        )
        _, majority_metrics = confusion_and_metrics(
            [int(row["true_label"]) for row in sampled],
            [int(row["majority_prediction"]) for row in sampled],
            num_classes=num_classes,
        )
        model_value = float(model_metrics["balanced_accuracy"])
        majority_value = float(majority_metrics["balanced_accuracy"])
        model_scores.append(model_value)
        majority_scores.append(majority_value)
        differences.append(model_value - majority_value)
    return {
        "requested_samples": int(samples),
        "accepted_samples": len(model_scores),
        "cluster": "subject_id",
        "balanced_accuracy": (
            _interval(model_scores) if model_scores else {}
        ),
        "majority_balanced_accuracy": (
            _interval(majority_scores) if majority_scores else {}
        ),
        "model_minus_majority": (
            _interval(differences) if differences else {}
        ),
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_detection_ratio <= 1.0:
        raise ValueError("min-detection-ratio must be in [0,1]")
    if not 1 <= args.min_repeats <= 5:
        raise ValueError("min-repeats must be in [1,5]")
    if args.cv_folds < 2 or args.inner_folds < 2:
        raise ValueError("cv-folds and inner-folds must be at least 2")
    if args.max_iter < 1:
        raise ValueError("max-iter must be positive")
    if not args.windows or len(args.windows) != len(set(args.windows)):
        raise ValueError("Response windows must be non-empty and unique")
    if not args.k_grid or any(value < 1 for value in args.k_grid):
        raise ValueError("Every k-grid value must be a positive integer")
    if not args.c_grid or any(value <= 0 for value in args.c_grid):
        raise ValueError("Every C-grid value must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be non-negative")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run nested subject-disjoint expression-window classification."""
    _validate_args(args)
    records, exclusions = load_graph_records(
        args.manifest,
        include_water=False,
        min_detection_ratio=args.min_detection_ratio,
    )
    if any(record.ma_mau == WATER_CODE for record in records):
        raise AssertionError("Water must not enter expression supervision")
    caches, schema = load_labelled_expression_caches(records)
    units, condition_audit = build_condition_units(
        records,
        min_repeats=args.min_repeats,
        max_repeats=5,
    )
    labels = _labels(units, args.task)
    groups = _groups(units)
    num_classes = 2 if args.task == "binary" else 3
    class_names = BINARY_NAMES if args.task == "binary" else JAR3_NAMES
    if np.any(np.bincount(labels, minlength=num_classes) == 0):
        raise ValueError("The supervised condition set is missing a class")
    feature_matrices = build_expression_feature_matrices(
        caches, units, args.windows
    )
    smallest_dimension = min(
        matrix.shape[1] for matrix in feature_matrices.values()
    )
    if max(args.k_grid) > smallest_dimension:
        raise ValueError(
            f"k-grid cannot exceed input feature dimension "
            f"{smallest_dimension}"
        )
    candidate_k_grid = list(args.k_grid)
    if args.include_all_features:
        candidate_k_grid.append(0)

    outer_splits = make_unit_splits(
        units,
        args.task,
        n_splits=args.cv_folds,
        seed=args.seed,
    )
    assert_subject_disjoint_splits(units, outer_splits)
    if args.fold_index is None:
        indexed_splits = list(enumerate(outer_splits))
    else:
        if not 0 <= args.fold_index < len(outer_splits):
            raise ValueError(
                f"fold-index must be in [0,{len(outer_splits) - 1}]"
            )
        indexed_splits = [
            (args.fold_index, outer_splits[args.fold_index])
        ]

    if args.output_dir is None:
        fold_name = (
            "full"
            if args.fold_index is None
            else f"fold{args.fold_index + 1:02d}"
        )
        args.output_dir = (
            Path("output/video_jar_gnn/runs_expression")
            / args.task
            / f"{fold_name}_seed{args.seed}_{_run_signature(args)}"
        )
    args.output_dir = Path(args.output_dir)
    _prepare_fresh_output(args.output_dir)
    config = {
        key: (
            [window.slug for window in value]
            if key == "windows"
            else str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in vars(args).items()
    }
    config.update(
        {
            "representation": REPRESENTATION,
            "input_graph_shape": [
                "T",
                len(EXPRESSION_NODES),
                len(EXPRESSION_FEATURES),
            ],
            "schema": {
                "node_names": list(schema.node_names),
                "feature_names": list(schema.feature_names),
                "signal_feature_indices": list(
                    schema.signal_feature_indices
                ),
                "observed_mask_indices": list(
                    schema.observed_mask_indices
                ),
                "excluded_feature_indices": list(
                    schema.excluded_feature_indices
                ),
            },
            "input_contract": "expression_summary_only",
            "uses_ma_mau_as_predictor": False,
            "ma_mau_roles": ["condition_grouping", "output_metadata"],
            "water_605_supervised": False,
            "timing_source": "target_lsl real seconds from trial onset",
            "window_selection_scope": (
                "subject-disjoint inner CV within each outer training fold"
            ),
            "condition_unit": "subject_id × ma_mau",
            "repeat_handling": (
                "summarize each real-time repeat window, then finite mean "
                "across up to five repeats"
            ),
            "pipeline_steps_fit_inside_each_training_fold": [
                "SimpleImputer(median)",
                "VarianceThreshold",
                "StandardScaler",
                "SelectKBest(f_classif)",
                "LogisticRegression(class_weight=balanced)",
            ],
            "n_records_loaded": len(records),
            "n_supervised_conditions": len(units),
            "n_subjects": len(set(groups)),
            "num_classes": num_classes,
            "class_names": class_names,
            "summary_dimensions": {
                slug: int(matrix.shape[1])
                for slug, matrix in feature_matrices.items()
            },
            "record_exclusions": exclusions,
            "condition_audit": condition_audit,
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for fold_index, (outer_train, outer_test) in indexed_splits:
        fold_number = fold_index + 1
        fold_seed = args.seed + fold_index * 1009
        outer_counts = np.bincount(
            labels[outer_train], minlength=num_classes
        )
        if np.any(outer_counts == 0):
            raise ValueError(
                f"Outer fold {fold_number} training set misses a class: "
                f"{outer_counts.tolist()}"
            )
        inner_splits = _valid_inner_splits(
            units,
            args.task,
            outer_train,
            requested_folds=args.inner_folds,
            seed=fold_seed,
            num_classes=num_classes,
        )
        assert_subject_disjoint_splits(units, inner_splits)
        oof = expression_candidate_oof_probabilities(
            feature_matrices,
            labels,
            inner_splits,
            args.windows,
            candidate_k_grid,
            args.c_grid,
            num_classes=num_classes,
            max_iter=args.max_iter,
            seed=fold_seed + 101,
        )
        selected, candidates = select_expression_candidate(
            oof,
            labels,
            outer_train,
            args.windows,
            num_classes=num_classes,
        )
        for candidate in candidates:
            selection_rows.append(
                {"fold": fold_number, **candidate}
            )
        selected_window = str(selected["window"])
        selected_matrix = feature_matrices[selected_window]
        pipeline = fit_expression_pipeline(
            selected_matrix,
            labels,
            outer_train,
            k_features=int(selected["k_features"]),
            c_value=float(selected["C"]),
            max_iter=args.max_iter,
            seed=fold_seed + 5001,
            num_classes=num_classes,
        )
        probabilities = _predict_probabilities(
            pipeline,
            selected_matrix,
            outer_test,
            num_classes=num_classes,
        )
        predictions = probabilities.argmax(axis=1)
        majority_prediction = int(
            np.bincount(
                labels[outer_train], minlength=num_classes
            ).argmax()
        )
        selected_indices = selected_original_feature_indices(pipeline)
        strategy = (
            "all_after_variance"
            if int(selected["k_features"]) == 0
            else "kbest"
        )
        train_subjects = sorted(set(groups[outer_train]))
        test_subjects = sorted(set(groups[outer_test]))
        for detail in selected_feature_details(pipeline, schema):
            feature_rows.append(
                {
                    "fold": fold_number,
                    "window": selected_window,
                    **detail,
                }
            )
        current_rows: list[dict[str, Any]] = []
        for local_index, unit_index in enumerate(outer_test):
            unit = units[int(unit_index)]
            row: dict[str, Any] = {
                "fold": fold_number,
                "unit_index": int(unit_index),
                "subject_id": unit.subject_id,
                # Identifier/grouping metadata only, never a predictor.
                "ma_mau": unit.ma_mau,
                "jar": unit.jar,
                "n_repeats": len(unit.record_indices),
                "repeats": "|".join(map(str, unit.repeats)),
                "true_label": int(labels[int(unit_index)]),
                "predicted_label": int(predictions[local_index]),
                "majority_prediction": majority_prediction,
                "selected_window": selected_window,
                "window_start_seconds": float(
                    selected["window_start_seconds"]
                ),
                "window_end_seconds": float(
                    selected["window_end_seconds"]
                ),
                "selected_strategy": strategy,
                "selected_k_candidate": int(selected["k_features"]),
                "selected_feature_count": len(selected_indices),
                "selected_C": float(selected["C"]),
            }
            for class_index in range(num_classes):
                row[f"prob_{class_index}"] = float(
                    probabilities[local_index, class_index]
                )
            current_rows.append(row)
        prediction_rows.extend(current_rows)
        _, metrics = _complete_metrics(
            current_rows, num_classes=num_classes
        )
        fold_rows.append(
            {
                "fold": fold_number,
                "n_inner_folds": len(inner_splits),
                "n_train_conditions": len(outer_train),
                "n_test_conditions": len(outer_test),
                "train_subjects": "|".join(map(str, train_subjects)),
                "test_subjects": "|".join(map(str, test_subjects)),
                "selected_window": selected_window,
                "window_start_seconds": float(
                    selected["window_start_seconds"]
                ),
                "window_end_seconds": float(
                    selected["window_end_seconds"]
                ),
                "selected_strategy": strategy,
                "selected_k_candidate": int(selected["k_features"]),
                "selected_feature_count": len(selected_indices),
                "selected_C": float(selected["C"]),
                **metrics,
            }
        )
        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "input_contract": "expression_summary_only",
                "uses_ma_mau_as_predictor": False,
                "representation": REPRESENTATION,
                "pipeline": pipeline,
                "task": args.task,
                "class_names": class_names,
                "response_window": {
                    "slug": selected_window,
                    "start_seconds": float(
                        selected["window_start_seconds"]
                    ),
                    "end_seconds": float(
                        selected["window_end_seconds"]
                    ),
                },
                "schema": config["schema"],
                "selected_original_feature_indices": (
                    selected_indices.astype(int).tolist()
                ),
                "repeat_aggregation": "finite_mean_of_repeat_summaries",
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
            },
            fold_dir / "expression_logistic.joblib",
        )
        print(
            f"Fold {fold_number}/{len(outer_splits)}: "
            f"window={selected_window}s, "
            f"k={'all' if int(selected['k_features']) == 0 else int(selected['k_features'])}, "
            f"C={float(selected['C']):g}, "
            f"BAcc={metrics['balanced_accuracy']:.3f}, "
            f"macro-F1={metrics['macro_f1']:.3f}"
        )

    _write_csv(
        args.output_dir / "predictions_condition.csv", prediction_rows
    )
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(
        args.output_dir / "selection_history.csv", selection_rows
    )
    _write_csv(
        args.output_dir / "selected_features.csv", feature_rows
    )
    if args.fold_index is None:
        predicted_indices = sorted(
            int(row["unit_index"]) for row in prediction_rows
        )
        if predicted_indices != list(range(len(units))):
            raise AssertionError(
                "Full CV did not predict every condition exactly once"
            )
    confusion, metrics = _complete_metrics(
        prediction_rows, num_classes=num_classes
    )
    np.save(args.output_dir / "confusion.npy", confusion)
    confusion_plot_written = _save_confusion_plot(
        confusion,
        class_names,
        args.output_dir / "confusion.png",
    )
    _, majority_metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in prediction_rows],
        [int(row["majority_prediction"]) for row in prediction_rows],
        num_classes=num_classes,
    )
    bootstrap = (
        subject_cluster_bootstrap(
            prediction_rows,
            num_classes=num_classes,
            samples=args.bootstrap_samples,
            seed=args.seed + 7001,
        )
        if args.fold_index is None
        else {}
    )
    selected_window_counts: dict[str, int] = defaultdict(int)
    for row in fold_rows:
        selected_window_counts[str(row["selected_window"])] += 1
    summary = {
        "task": args.task,
        "class_names": class_names,
        "representation": REPRESENTATION,
        "input_contract": "expression_summary_only",
        "uses_ma_mau_as_predictor": False,
        "water_605_supervised": False,
        "partial_cv": args.fold_index is not None,
        "folds_run": sorted({int(row["fold"]) for row in fold_rows}),
        "condition_unit": "subject_id × ma_mau",
        "window_selection_scope": (
            "subject-disjoint inner CV within each outer training fold"
        ),
        "selected_window_fold_counts": dict(
            sorted(selected_window_counts.items())
        ),
        "metrics": metrics,
        "confusion": confusion.tolist(),
        "confusion_plot_written": confusion_plot_written,
        "baseline": {"outer_train_majority": majority_metrics},
        "subject_cluster_bootstrap": bootstrap,
        "record_exclusions": exclusions,
        "condition_audit": condition_audit,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leakage-safe expression_v2 response-window classifier. "
            "The candidate window is selected only in inner subject CV; "
            "ma_mau is never a predictor and water 605 is excluded."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "output/video_jar_gnn/graph_manifest_expression_v2.csv"
        ),
    )
    parser.add_argument(
        "--task", choices=("binary", "jar3"), default="binary"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=parse_response_window,
        default=[
            parse_response_window(value) for value in DEFAULT_WINDOWS
        ],
        metavar="START:END",
        help=(
            "Candidate real-time response windows in seconds from trial "
            "onset. Selection is repeated inside every outer training fold."
        ),
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=5,
        help=(
            "Minimum usable repeats per subject-condition; the default "
            "requires all five experimental repeats."
        ),
    )
    parser.add_argument(
        "--min-detection-ratio", type=float, default=0.5
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--fold-index",
        type=int,
        help="Run one zero-based outer fold as a smoke test.",
    )
    parser.add_argument(
        "--k-grid",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_GRID),
        help="Candidate numbers of expression summary features.",
    )
    parser.add_argument(
        "--include-all-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also let inner CV choose all non-constant features with L2 "
            "regularization."
        ),
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_GRID),
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    summary = run(args)
    metrics = summary["metrics"]
    print(
        "Finished expression_v2 nested condition CV: "
        f"BAcc={metrics['balanced_accuracy']:.3f}, "
        f"macro-F1={metrics['macro_f1']:.3f}"
    )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
