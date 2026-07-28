"""Condition-level classical baselines for the video JAR problem.

This module intentionally treats ``subject × ma_mau`` as the independent
sample.  The five repeated video trials are averaged before feature
summarisation, so repeated measurements never inflate the effective sample
size.  All hyperparameters are selected with subject-disjoint inner folds.

The default comparison contains:

* a class-balanced sample-code logistic regression;
* face-only logistic regressions using raw, within-trial-delta and
  water-calibrated summaries;
* late probability fusion of each face representation with sample code.

Water (``ma_mau=605``) is never a supervised target.  It is used only as a
label-free, subject-specific reference by the ``water_delta`` variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler

from .advanced_dataset import (
    ConditionUnit,
    build_condition_units,
    preprocess_graph,
)
from .constants import (
    AU_NODES,
    BINARY_NAMES,
    FEATURE_NAMES,
    JAR3_NAMES,
    SAMPLE_CODES,
    WATER_CODE,
)
from .dataset import GraphRecord, GraphStore, load_graph_records
from .train import _save_confusion_plot, _write_csv, confusion_and_metrics
from .train_advanced import make_unit_splits


CONTINUOUS_FEATURE_INDICES = (0, 1, 2, 4, 5, 6, 7, 8, 9)
MOTION_FEATURE_INDICES = (6, 7, 8, 9)
SWEET_CODES = tuple(code for code in SAMPLE_CODES if code != WATER_CODE)
SWEET_CODE_TO_INDEX = {
    code: index for index, code in enumerate(SWEET_CODES)
}
FEATURE_MODES = ("raw", "trial_delta", "water_delta")
ALL_EXPERIMENTS = (
    "code_only",
    "face_raw",
    "face_trial_delta",
    "face_water_delta",
    "fusion_raw",
    "fusion_trial_delta",
    "fusion_water_delta",
)
VIDEO_ONLY_EXPERIMENTS = tuple(
    f"video_{mode}" for mode in FEATURE_MODES
)
TEMPORAL_STATISTIC_NAMES = (
    "mean",
    "std",
    "quantile_10",
    "quantile_90",
    "late_minus_early",
    "temporal_slope",
)
MOTION_STATISTIC_NAMES = (
    "absolute_mean",
    "absolute_std",
    "absolute_quantile_90",
    "absolute_max",
)


def _run_signature(args: argparse.Namespace) -> str:
    payload = {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in vars(args).items()
        if key != "output_dir"
    }
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


def summarize_condition_graph(graph: np.ndarray) -> np.ndarray:
    """Convert one condition graph ``[T,N,F]`` into robust temporal features.

    For every node and non-detection feature, the summary contains mean,
    standard deviation, 10/90th percentiles, late-minus-early change and
    temporal slope.  Absolute motion features additionally receive mean,
    standard deviation, 90th percentile and maximum motion energy.
    """
    values = np.asarray(graph, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError(
            f"Expected condition graph [T,N,F] with T>=2, got {values.shape}"
        )
    if values.shape[2] <= max(CONTINUOUS_FEATURE_INDICES):
        raise ValueError(f"Graph has too few features: {values.shape}")
    selected = values[:, :, CONTINUOUS_FEATURE_INDICES]
    window = max(1, int(round(values.shape[0] * 0.2)))
    timeline = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float64)
    timeline -= timeline.mean()
    denominator = float(np.square(timeline).sum())
    slope = (
        np.tensordot(timeline, selected, axes=(0, 0)) / denominator
        if denominator > 0
        else np.zeros_like(selected[0])
    )
    temporal_statistics = np.stack(
        (
            selected.mean(axis=0),
            selected.std(axis=0),
            np.quantile(selected, 0.10, axis=0),
            np.quantile(selected, 0.90, axis=0),
            selected[-window:].mean(axis=0)
            - selected[:window].mean(axis=0),
            slope,
        ),
        axis=-1,
    )
    absolute_motion = np.abs(values[:, :, MOTION_FEATURE_INDICES])
    motion_statistics = np.stack(
        (
            absolute_motion.mean(axis=0),
            absolute_motion.std(axis=0),
            np.quantile(absolute_motion, 0.90, axis=0),
            absolute_motion.max(axis=0),
        ),
        axis=-1,
    )
    result = np.concatenate(
        (temporal_statistics.reshape(-1), motion_statistics.reshape(-1))
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Condition summary contains NaN/Inf")
    return result


def code_feature_matrix(units: Sequence[ConditionUnit]) -> np.ndarray:
    """One-hot sweet-sample code matrix (water is not a target category)."""
    matrix = np.zeros((len(units), len(SWEET_CODES)), dtype=np.float32)
    for row, unit in enumerate(units):
        if unit.ma_mau not in SWEET_CODE_TO_INDEX:
            raise ValueError(
                f"Unexpected supervised ma_mau={unit.ma_mau}; "
                f"expected one of {SWEET_CODES}"
            )
        matrix[row, SWEET_CODE_TO_INDEX[unit.ma_mau]] = 1.0
    return matrix


def build_face_feature_matrices(
    store: GraphStore,
    units: Sequence[ConditionUnit],
    modes: Sequence[str],
    *,
    baseline_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Build deterministic condition summaries for requested representations."""
    requested = tuple(dict.fromkeys(modes))
    unknown = set(requested).difference(FEATURE_MODES)
    if unknown:
        raise ValueError(f"Unknown face feature modes: {sorted(unknown)}")

    water_indices: dict[str, list[int]] = defaultdict(list)
    if "water_delta" in requested:
        for index, record in enumerate(store.records):
            if record.ma_mau == WATER_CODE:
                water_indices[record.subject_id].append(index)
        missing = sorted(
            {
                unit.subject_id
                for unit in units
                if unit.subject_id not in water_indices
            }
        )
        if missing:
            raise ValueError(
                "water_delta requires at least one usable water trial for "
                f"every subject; missing={missing}"
            )
    water_references = {
        subject: np.mean(
            np.stack([store.graphs[index] for index in indices], axis=0),
            axis=0,
        ).astype(np.float32)
        for subject, indices in water_indices.items()
    }

    per_mode: dict[str, list[np.ndarray]] = {
        mode: [] for mode in requested
    }
    for unit in units:
        repeat_graphs = [
            store.graphs[index] for index in unit.record_indices
        ]
        raw_condition = np.mean(
            np.stack(repeat_graphs, axis=0), axis=0
        ).astype(np.float32)
        for mode in requested:
            if mode == "raw":
                condition = raw_condition
            elif mode == "trial_delta":
                transformed = [
                    preprocess_graph(
                        graph,
                        mode="trial_delta",
                        baseline_frames=baseline_frames,
                    )
                    for graph in repeat_graphs
                ]
                condition = np.mean(
                    np.stack(transformed, axis=0), axis=0
                ).astype(np.float32)
            else:
                condition = preprocess_graph(
                    raw_condition,
                    mode="water_delta",
                    water_reference=water_references[unit.subject_id],
                )
            per_mode[mode].append(summarize_condition_graph(condition))
    matrices = {
        mode: np.stack(rows, axis=0).astype(np.float32)
        for mode, rows in per_mode.items()
    }
    water_counts = {
        subject: len(indices)
        for subject, indices in sorted(water_indices.items())
    }
    return matrices, water_counts


def _labels(units: Sequence[ConditionUnit], task: str) -> np.ndarray:
    return np.asarray(
        [unit.label_for(task) for unit in units], dtype=np.int64
    )


def _groups(units: Sequence[ConditionUnit]) -> np.ndarray:
    return np.asarray([unit.subject_id for unit in units])


def _experiment_parts(experiment: str) -> tuple[str, str | None]:
    if experiment == "code_only":
        return "code", None
    family, mode = experiment.split("_", 1)
    return family, mode


def _make_logistic(
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    return Pipeline(
        (
            ("standardizer", StandardScaler()),
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


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
    num_classes: int,
) -> Pipeline:
    observed = np.bincount(labels[indices], minlength=num_classes)
    if np.any(observed == 0):
        raise ValueError(
            "A logistic-regression training fold is missing a class: "
            f"{observed.tolist()}"
        )
    model = _make_logistic(
        c_value=c_value, max_iter=max_iter, seed=seed
    )
    model.fit(features[indices], labels[indices])
    return model


def _predict_probabilities(
    model: Pipeline,
    features: np.ndarray,
    indices: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features[indices]), dtype=np.float64)
    classes = np.asarray(
        model.named_steps["classifier"].classes_, dtype=np.int64
    )
    probabilities = np.zeros((len(indices), num_classes), dtype=np.float64)
    probabilities[:, classes] = raw
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Logistic regression returned invalid probabilities")
    return probabilities / row_sums


def make_selected_video_pipeline(
    *,
    k_features: int,
    c_value: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    """Build the complete leakage-safe video feature pipeline.

    Every data-dependent step lives inside this object.  Callers must fit the
    pipeline only on the current inner- or outer-training subject indices.
    """
    if k_features < 0:
        raise ValueError("k_features must be non-negative (0 means all)")
    selector_k: int | str = "all" if k_features == 0 else int(k_features)
    return Pipeline(
        (
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


def fit_selected_video_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    k_features: int,
    c_value: float,
    max_iter: int,
    seed: int,
    num_classes: int,
) -> Pipeline:
    """Fit variance/scaling/selection/classification on training rows only."""
    train_indices = np.asarray(indices, dtype=np.int64)
    observed = np.bincount(labels[train_indices], minlength=num_classes)
    if np.any(observed == 0):
        raise ValueError(
            "A video-feature training fold is missing a class: "
            f"{observed.tolist()}"
        )
    pipeline = make_selected_video_pipeline(
        k_features=k_features,
        c_value=c_value,
        max_iter=max_iter,
        seed=seed,
    )
    try:
        pipeline.fit(features[train_indices], labels[train_indices])
    except ValueError as error:
        if "k should be" in str(error) or "k=" in str(error):
            raise ValueError(
                f"k={k_features} exceeds the usable feature count in this "
                "training fold; choose a smaller --k-grid"
            ) from error
        raise
    return pipeline


def video_feature_oof_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    k_grid: Sequence[int],
    c_grid: Sequence[float],
    *,
    num_classes: int,
    max_iter: int,
    seed: int,
) -> dict[tuple[int, float], np.ndarray]:
    """Generate inner-CV probabilities for every video-only candidate."""
    candidates = [
        (int(k_features), float(c_value))
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
        for candidate_index, (k_features, c_value) in enumerate(candidates):
            pipeline = fit_selected_video_pipeline(
                features,
                labels,
                train,
                k_features=k_features,
                c_value=c_value,
                max_iter=max_iter,
                seed=seed + split_index * 101 + candidate_index,
                num_classes=num_classes,
            )
            result[(k_features, c_value)][validation] = (
                _predict_probabilities(
                    pipeline,
                    features,
                    validation,
                    num_classes=num_classes,
                )
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
                f"Missing video OOF probabilities for candidate={candidate}"
            )
    return result


def select_video_feature_candidate(
    oof: dict[tuple[int, float], np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    *,
    num_classes: int,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Select ``k`` and ``C`` from inner OOF metrics only."""
    best: dict[str, float | int] | None = None
    best_key: tuple[float, ...] | None = None
    rows: list[dict[str, Any]] = []
    for k_features, c_value in sorted(oof):
        metrics = _probability_metrics(
            labels[outer_train],
            oof[(k_features, c_value)][outer_train],
            num_classes=num_classes,
        )
        rows.append(
            {
                "k_features": k_features,
                "selection_strategy": (
                    "all_after_variance" if k_features == 0 else "kbest"
                ),
                "C": c_value,
                **metrics,
            }
        )
        key = _selection_key(
            metrics,
            complexity_tiebreak=(
                -float(10**9 if k_features == 0 else k_features),
                -float(c_value),
            ),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "k_features": int(k_features),
                "C": float(c_value),
            }
    assert best is not None
    return best, rows


def selected_original_feature_indices(pipeline: Pipeline) -> np.ndarray:
    """Map SelectKBest support back to the 1,050 input-summary columns."""
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["selector"]
    after_variance = np.flatnonzero(variance.get_support())
    return after_variance[np.asarray(selector.get_support(), dtype=bool)]


def describe_summary_feature(index: int) -> dict[str, str]:
    """Decode a flattened condition-summary column into a readable feature."""
    temporal_width = (
        len(CONTINUOUS_FEATURE_INDICES) * len(TEMPORAL_STATISTIC_NAMES)
    )
    temporal_total = len(AU_NODES) * temporal_width
    motion_width = len(MOTION_FEATURE_INDICES) * len(MOTION_STATISTIC_NAMES)
    total = temporal_total + len(AU_NODES) * motion_width
    feature_index = int(index)
    if not 0 <= feature_index < total:
        raise ValueError(f"summary feature index must be in [0,{total - 1}]")
    if feature_index < temporal_total:
        node_index, within_node = divmod(feature_index, temporal_width)
        source_position, statistic_position = divmod(
            within_node, len(TEMPORAL_STATISTIC_NAMES)
        )
        source_index = CONTINUOUS_FEATURE_INDICES[source_position]
        statistic = TEMPORAL_STATISTIC_NAMES[statistic_position]
        family = "temporal"
    else:
        node_index, within_node = divmod(
            feature_index - temporal_total, motion_width
        )
        source_position, statistic_position = divmod(
            within_node, len(MOTION_STATISTIC_NAMES)
        )
        source_index = MOTION_FEATURE_INDICES[source_position]
        statistic = MOTION_STATISTIC_NAMES[statistic_position]
        family = "absolute_motion"
    return {
        "node": AU_NODES[node_index],
        "source_feature": FEATURE_NAMES[source_index],
        "statistic": statistic,
        "feature_family": family,
    }


def selected_feature_details(
    pipeline: Pipeline,
) -> list[dict[str, float | int | str]]:
    """Return selected columns ranked by train-fold-only univariate F score."""
    variance = pipeline.named_steps["variance"]
    selector = pipeline.named_steps["selector"]
    after_variance = np.flatnonzero(variance.get_support())
    support = np.asarray(selector.get_support(), dtype=bool)
    original_indices = after_variance[support]
    selected_scores = np.asarray(selector.scores_, dtype=np.float64)[support]
    sortable_scores = np.nan_to_num(
        selected_scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf
    )
    order = np.argsort(-sortable_scores, kind="stable")
    details: list[dict[str, float | int | str]] = []
    for rank, selected_position in enumerate(order, start=1):
        original_index = int(original_indices[selected_position])
        score = float(selected_scores[selected_position])
        details.append(
            {
                "rank_by_training_f_score": rank,
                "summary_feature_index": original_index,
                "training_f_score": score if np.isfinite(score) else "",
                **describe_summary_feature(original_index),
            }
        )
    return details


def summarize_feature_stability(
    rows: Sequence[dict[str, Any]],
    experiments: Sequence[str],
) -> dict[str, Any]:
    """Quantify whether nested CV repeatedly selects the same video features."""
    result: dict[str, Any] = {}
    for experiment in experiments:
        fold_features: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            if str(row["experiment"]) == experiment:
                fold_features[int(row["fold"])].add(
                    int(row["summary_feature_index"])
                )
        ordered = [fold_features[fold] for fold in sorted(fold_features)]
        pairwise_jaccard: list[float] = []
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                union = left | right
                pairwise_jaccard.append(
                    len(left & right) / len(union) if union else 1.0
                )
        frequency = Counter(
            feature_index
            for selected in ordered
            for feature_index in selected
        )
        top_features = []
        for feature_index, selected_fold_count in sorted(
            frequency.items(), key=lambda item: (-item[1], item[0])
        )[:10]:
            top_features.append(
                {
                    "summary_feature_index": int(feature_index),
                    "selected_fold_count": int(selected_fold_count),
                    **describe_summary_feature(feature_index),
                }
            )
        common = set.intersection(*ordered) if ordered else set()
        result[experiment] = {
            "n_folds": len(ordered),
            "selected_counts": [len(selected) for selected in ordered],
            "mean_pairwise_jaccard": (
                float(np.mean(pairwise_jaccard))
                if pairwise_jaccard
                else None
            ),
            "features_selected_in_every_fold": len(common),
            "top_features_by_fold_frequency": top_features,
        }
    return result


def _valid_inner_splits(
    units: Sequence[ConditionUnit],
    task: str,
    outer_train: np.ndarray,
    *,
    requested_folds: int,
    seed: int,
    num_classes: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Use the largest requested fold count whose train parts contain all classes."""
    subject_count = len(np.unique(_groups(units)[outer_train]))
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
                np.bincount(
                    _labels(units, task)[train], minlength=num_classes
                )
                > 0
            )
            for train, _ in splits
        ):
            return splits
    raise ValueError(
        "Could not construct subject-disjoint inner folds whose training "
        "part contains every class"
    )


def oof_probabilities(
    features: np.ndarray,
    labels: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    c_grid: Sequence[float],
    *,
    num_classes: int,
    max_iter: int,
    seed: int,
) -> dict[float, np.ndarray]:
    """Generate one inner-CV prediction per outer-training condition."""
    result = {
        float(c_value): np.full(
            (len(labels), num_classes), np.nan, dtype=np.float64
        )
        for c_value in c_grid
    }
    seen = np.zeros(len(labels), dtype=np.int64)
    for split_index, (train, validation) in enumerate(inner_splits):
        seen[validation] += 1
        for c_index, c_value in enumerate(c_grid):
            model = _fit_logistic(
                features,
                labels,
                train,
                c_value=float(c_value),
                max_iter=max_iter,
                seed=seed + split_index * 101 + c_index,
                num_classes=num_classes,
            )
            result[float(c_value)][validation] = _predict_probabilities(
                model,
                features,
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
    for c_value, probabilities in result.items():
        if not np.isfinite(probabilities[validation_union]).all():
            raise AssertionError(
                f"Missing OOF probabilities for C={c_value}"
            )
    return result


def fuse_probabilities(
    code_probabilities: np.ndarray,
    face_probabilities: np.ndarray,
    *,
    alpha_face: float,
) -> np.ndarray:
    """Geometric late fusion; 0=code only and 1=face only."""
    if not 0.0 <= alpha_face <= 1.0:
        raise ValueError("alpha_face must be in [0,1]")
    if code_probabilities.shape != face_probabilities.shape:
        raise ValueError("Code/face probability shapes do not match")
    epsilon = np.finfo(np.float64).tiny
    log_fused = (
        (1.0 - alpha_face)
        * np.log(np.clip(code_probabilities, epsilon, 1.0))
        + alpha_face
        * np.log(np.clip(face_probabilities, epsilon, 1.0))
    )
    log_fused -= log_fused.max(axis=1, keepdims=True)
    fused = np.exp(log_fused)
    return fused / fused.sum(axis=1, keepdims=True)


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


def _selection_key(
    metrics: dict[str, float],
    *,
    complexity_tiebreak: tuple[float, ...],
) -> tuple[float, ...]:
    return (
        metrics["balanced_accuracy"],
        metrics["macro_f1"],
        -metrics["negative_log_likelihood"],
        *complexity_tiebreak,
    )


def select_single_source(
    oof: dict[float, np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    *,
    num_classes: int,
) -> tuple[float, list[dict[str, Any]]]:
    best_c: float | None = None
    best_key: tuple[float, ...] | None = None
    rows: list[dict[str, Any]] = []
    for c_value in sorted(oof):
        metrics = _probability_metrics(
            labels[outer_train],
            oof[c_value][outer_train],
            num_classes=num_classes,
        )
        rows.append({"C": c_value, **metrics})
        key = _selection_key(
            metrics, complexity_tiebreak=(-float(c_value),)
        )
        if best_key is None or key > best_key:
            best_key = key
            best_c = float(c_value)
    assert best_c is not None
    return best_c, rows


def select_late_fusion(
    code_oof: dict[float, np.ndarray],
    face_oof: dict[float, np.ndarray],
    labels: np.ndarray,
    outer_train: np.ndarray,
    alpha_grid: Sequence[float],
    *,
    num_classes: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    best: dict[str, float] | None = None
    best_key: tuple[float, ...] | None = None
    rows: list[dict[str, Any]] = []
    for code_c in sorted(code_oof):
        for face_c in sorted(face_oof):
            for alpha in sorted(float(value) for value in alpha_grid):
                probabilities = fuse_probabilities(
                    code_oof[code_c][outer_train],
                    face_oof[face_c][outer_train],
                    alpha_face=alpha,
                )
                metrics = _probability_metrics(
                    labels[outer_train],
                    probabilities,
                    num_classes=num_classes,
                )
                row = {
                    "code_C": code_c,
                    "face_C": face_c,
                    "alpha_face": alpha,
                    **metrics,
                }
                rows.append(row)
                # On exact metric ties, prefer stronger regularisation and
                # less reliance on the extra video stream.
                key = _selection_key(
                    metrics,
                    complexity_tiebreak=(
                        -alpha,
                        -float(code_c),
                        -float(face_c),
                    ),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "code_C": float(code_c),
                        "face_C": float(face_c),
                        "alpha_face": float(alpha),
                    }
    assert best is not None
    return best, rows


def _complete_metrics(
    rows: Sequence[dict[str, Any]], *, num_classes: int
) -> tuple[np.ndarray, dict[str, Any]]:
    truth = [int(row["true_label"]) for row in rows]
    prediction = [int(row["predicted_label"]) for row in rows]
    confusion, metrics = confusion_and_metrics(
        truth, prediction, num_classes=num_classes
    )
    metrics["mae"] = float(
        np.mean(np.abs(np.asarray(truth) - np.asarray(prediction)))
    )
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
    experiments: Sequence[str],
    *,
    num_classes: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap whole subjects and compare each method with code-only."""
    if samples <= 0:
        return {}
    by_experiment_subject: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_experiment_subject[str(row["experiment"])][
            str(row["subject_id"])
        ].append(row)
    subject_sets = [
        set(by_experiment_subject[experiment])
        for experiment in experiments
    ]
    if not subject_sets:
        return {}
    subjects = sorted(set.intersection(*subject_sets))
    if not subjects:
        return {}
    rng = np.random.default_rng(seed)
    scores: dict[str, list[float]] = {
        experiment: [] for experiment in experiments
    }
    differences: dict[str, list[float]] = {
        experiment: [] for experiment in experiments
        if experiment != "code_only" and "code_only" in experiments
    }
    accepted = 0
    attempts = 0
    while accepted < samples and attempts < max(samples * 20, 100):
        attempts += 1
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        sampled: dict[str, list[dict[str, Any]]] = {}
        valid = True
        for experiment in experiments:
            experiment_rows = [
                row
                for subject in chosen
                for row in by_experiment_subject[experiment][str(subject)]
            ]
            support = np.bincount(
                [int(row["true_label"]) for row in experiment_rows],
                minlength=num_classes,
            )
            if np.any(support == 0):
                valid = False
                break
            sampled[experiment] = experiment_rows
        if not valid:
            continue
        iteration_scores = {}
        for experiment, experiment_rows in sampled.items():
            _, metrics = _complete_metrics(
                experiment_rows, num_classes=num_classes
            )
            value = float(metrics["balanced_accuracy"])
            scores[experiment].append(value)
            iteration_scores[experiment] = value
        if "code_only" in iteration_scores:
            for experiment in differences:
                differences[experiment].append(
                    iteration_scores[experiment]
                    - iteration_scores["code_only"]
                )
        accepted += 1
    result: dict[str, Any] = {
        "requested_samples": samples,
        "accepted_samples": accepted,
        "cluster": "subject_id",
        "balanced_accuracy": {
            experiment: _interval(values)
            for experiment, values in scores.items()
            if values
        },
    }
    if differences:
        result["minus_code_only"] = {
            experiment: _interval(values)
            for experiment, values in differences.items()
            if values
        }
    return result


def video_feature_cluster_bootstrap(
    rows: Sequence[dict[str, Any]],
    experiments: Sequence[str],
    *,
    num_classes: int,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap subjects for video models and their train-fold majority."""
    if samples <= 0:
        return {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["experiment"])][str(row["subject_id"])].append(row)
    subject_sets = [set(grouped[experiment]) for experiment in experiments]
    if not subject_sets:
        return {}
    subjects = sorted(set.intersection(*subject_sets))
    if not subjects:
        return {}
    rng = np.random.default_rng(seed)
    model_scores: dict[str, list[float]] = {
        experiment: [] for experiment in experiments
    }
    majority_scores: dict[str, list[float]] = {
        experiment: [] for experiment in experiments
    }
    differences: dict[str, list[float]] = {
        experiment: [] for experiment in experiments
    }
    accepted = 0
    attempts = 0
    while accepted < samples and attempts < max(samples * 20, 100):
        attempts += 1
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        sampled_by_experiment = {
            experiment: [
                row
                for subject in chosen
                for row in grouped[experiment][str(subject)]
            ]
            for experiment in experiments
        }
        reference_rows = sampled_by_experiment[experiments[0]]
        support = np.bincount(
            [int(row["true_label"]) for row in reference_rows],
            minlength=num_classes,
        )
        if np.any(support == 0):
            continue
        for experiment, experiment_rows in sampled_by_experiment.items():
            _, model_metrics = _complete_metrics(
                experiment_rows, num_classes=num_classes
            )
            _, majority_metrics = confusion_and_metrics(
                [int(row["true_label"]) for row in experiment_rows],
                [int(row["majority_prediction"]) for row in experiment_rows],
                num_classes=num_classes,
            )
            model_score = float(model_metrics["balanced_accuracy"])
            majority_score = float(majority_metrics["balanced_accuracy"])
            model_scores[experiment].append(model_score)
            majority_scores[experiment].append(majority_score)
            differences[experiment].append(model_score - majority_score)
        accepted += 1
    return {
        "requested_samples": samples,
        "accepted_samples": accepted,
        "cluster": "subject_id",
        "model_balanced_accuracy": {
            experiment: _interval(values)
            for experiment, values in model_scores.items()
            if values
        },
        "majority_balanced_accuracy": {
            experiment: _interval(values)
            for experiment, values in majority_scores.items()
            if values
        },
        "model_minus_majority": {
            experiment: _interval(values)
            for experiment, values in differences.items()
            if values
        },
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_detection_ratio <= 1.0:
        raise ValueError("min-detection-ratio must be in [0,1]")
    if args.min_repeats < 1 or args.baseline_frames < 2:
        raise ValueError("min-repeats must be positive and baseline-frames >= 2")
    if args.cv_folds < 2 or args.inner_folds < 2:
        raise ValueError("cv-folds and inner-folds must be at least 2")
    if args.max_iter < 1:
        raise ValueError("max-iter must be positive")
    if not args.c_grid or any(value <= 0 for value in args.c_grid):
        raise ValueError("Every C-grid value must be positive")
    if not args.alpha_grid or any(
        not 0.0 <= value <= 1.0 for value in args.alpha_grid
    ):
        raise ValueError("Every alpha-grid value must be in [0,1]")
    if len(args.variants) != len(set(args.variants)):
        raise ValueError("variants must not contain duplicates")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be non-negative")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the nested subject-disjoint classical comparison."""
    _validate_args(args)
    variants = tuple(args.variants)
    requested_modes = tuple(
        mode
        for mode in FEATURE_MODES
        if any(
            _experiment_parts(experiment)[1] == mode
            for experiment in variants
        )
    )
    requires_water = "water_delta" in requested_modes
    records, exclusions = load_graph_records(
        args.manifest,
        include_water=requires_water,
        min_detection_ratio=args.min_detection_ratio,
    )
    store = GraphStore(records)
    units, condition_audit = build_condition_units(
        records,
        min_repeats=args.min_repeats,
        exclude_codes={WATER_CODE} if requires_water else None,
    )
    labels = _labels(units, args.task)
    num_classes = 2 if args.task == "binary" else 3
    class_names = BINARY_NAMES if args.task == "binary" else JAR3_NAMES
    if np.any(np.bincount(labels, minlength=num_classes) == 0):
        raise ValueError("The supervised condition set is missing a class")
    code_features = code_feature_matrix(units)
    face_features, water_counts = build_face_feature_matrices(
        store,
        units,
        requested_modes,
        baseline_frames=args.baseline_frames,
    )
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
            raise ValueError(
                f"fold-index must be in [0,{len(outer_splits) - 1}]"
            )
        indexed_splits = [(args.fold_index, outer_splits[args.fold_index])]

    if args.output_dir is None:
        fold_name = (
            "full"
            if args.fold_index is None
            else f"fold{args.fold_index + 1:02d}"
        )
        args.output_dir = (
            Path("output/video_jar_gnn/runs_classical")
            / args.task
            / f"{fold_name}_seed{args.seed}_{_run_signature(args)}"
        )
    _prepare_fresh_output(args.output_dir)
    config = {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in vars(args).items()
    }
    config.update(
        {
            "n_records_loaded": len(records),
            "n_supervised_conditions": len(units),
            "n_subjects": len(set(_groups(units))),
            "num_classes": num_classes,
            "class_names": class_names,
            "sweet_code_order": SWEET_CODES,
            "condition_unit": "subject_id × ma_mau",
            "repeat_handling": "mean graph before temporal summary",
            "water_used_as_unlabelled_reference": requires_water,
            "water_reference_aggregation": (
                "mean across usable ma_mau=605 repeats"
                if requires_water
                else None
            ),
            "water_repeat_counts": water_counts,
            "face_feature_dimensions": {
                mode: int(matrix.shape[1])
                for mode, matrix in face_features.items()
            },
            "record_exclusions": exclusions,
            "condition_audit": condition_audit,
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    needs_code = any(
        _experiment_parts(experiment)[0] in {"code", "fusion"}
        for experiment in variants
    )
    modes_needing_face = {
        mode
        for experiment in variants
        for family, mode in [_experiment_parts(experiment)]
        if family in {"face", "fusion"} and mode is not None
    }

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
        code_oof = (
            oof_probabilities(
                code_features,
                labels,
                inner_splits,
                args.c_grid,
                num_classes=num_classes,
                max_iter=args.max_iter,
                seed=fold_seed + 11,
            )
            if needs_code
            else {}
        )
        face_oof = {
            mode: oof_probabilities(
                face_features[mode],
                labels,
                inner_splits,
                args.c_grid,
                num_classes=num_classes,
                max_iter=args.max_iter,
                seed=fold_seed + 101 + mode_index * 1000,
            )
            for mode_index, mode in enumerate(sorted(modes_needing_face))
        }
        fitted_cache: dict[
            tuple[str, str | None, float], tuple[Pipeline, np.ndarray]
        ] = {}

        def fit_outer_source(
            family: str, mode: str | None, c_value: float
        ) -> tuple[Pipeline, np.ndarray]:
            cache_key = (family, mode, float(c_value))
            if cache_key not in fitted_cache:
                matrix = (
                    code_features
                    if family == "code"
                    else face_features[str(mode)]
                )
                model = _fit_logistic(
                    matrix,
                    labels,
                    outer_train,
                    c_value=c_value,
                    max_iter=args.max_iter,
                    seed=fold_seed + len(fitted_cache) * 17,
                    num_classes=num_classes,
                )
                probabilities = _predict_probabilities(
                    model,
                    matrix,
                    outer_test,
                    num_classes=num_classes,
                )
                fitted_cache[cache_key] = (model, probabilities)
            return fitted_cache[cache_key]

        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_subjects = sorted(
            {units[int(index)].subject_id for index in outer_test}
        )
        train_subjects = sorted(
            {units[int(index)].subject_id for index in outer_train}
        )
        for experiment in variants:
            family, mode = _experiment_parts(experiment)
            code_model: Pipeline | None = None
            face_model: Pipeline | None = None
            selected_code_c: float | None = None
            selected_face_c: float | None = None
            selected_alpha: float | None = None
            candidate_rows: list[dict[str, Any]]
            if family == "code":
                selected_code_c, candidate_rows = select_single_source(
                    code_oof,
                    labels,
                    outer_train,
                    num_classes=num_classes,
                )
                code_model, probabilities = fit_outer_source(
                    "code", None, selected_code_c
                )
            elif family == "face":
                assert mode is not None
                selected_face_c, candidate_rows = select_single_source(
                    face_oof[mode],
                    labels,
                    outer_train,
                    num_classes=num_classes,
                )
                face_model, probabilities = fit_outer_source(
                    "face", mode, selected_face_c
                )
            else:
                assert mode is not None
                selected, candidate_rows = select_late_fusion(
                    code_oof,
                    face_oof[mode],
                    labels,
                    outer_train,
                    args.alpha_grid,
                    num_classes=num_classes,
                )
                selected_code_c = selected["code_C"]
                selected_face_c = selected["face_C"]
                selected_alpha = selected["alpha_face"]
                code_model, code_probabilities = fit_outer_source(
                    "code", None, selected_code_c
                )
                face_model, face_probabilities = fit_outer_source(
                    "face", mode, selected_face_c
                )
                probabilities = fuse_probabilities(
                    code_probabilities,
                    face_probabilities,
                    alpha_face=selected_alpha,
                )
            for candidate in candidate_rows:
                selection_rows.append(
                    {
                        "fold": fold_number,
                        "experiment": experiment,
                        **candidate,
                    }
                )
            predictions = probabilities.argmax(axis=1)
            current_rows = []
            for local_index, unit_index in enumerate(outer_test):
                unit = units[int(unit_index)]
                row: dict[str, Any] = {
                    "experiment": experiment,
                    "fold": fold_number,
                    "unit_index": int(unit_index),
                    "subject_id": unit.subject_id,
                    "ma_mau": unit.ma_mau,
                    "jar": unit.jar,
                    "n_repeats": len(unit.record_indices),
                    "repeats": "|".join(map(str, unit.repeats)),
                    "true_label": int(labels[int(unit_index)]),
                    "predicted_label": int(predictions[local_index]),
                    "feature_mode": mode,
                    "selected_code_C": selected_code_c,
                    "selected_face_C": selected_face_c,
                    "selected_alpha_face": selected_alpha,
                }
                for class_index in range(num_classes):
                    row[f"prob_{class_index}"] = float(
                        probabilities[local_index, class_index]
                    )
                current_rows.append(row)
            all_rows.extend(current_rows)
            _, fold_metrics = _complete_metrics(
                current_rows, num_classes=num_classes
            )
            fold_rows.append(
                {
                    "fold": fold_number,
                    "experiment": experiment,
                    "n_inner_folds": len(inner_splits),
                    "n_train_conditions": len(outer_train),
                    "n_test_conditions": len(outer_test),
                    "train_subjects": "|".join(train_subjects),
                    "test_subjects": "|".join(test_subjects),
                    "selected_code_C": selected_code_c,
                    "selected_face_C": selected_face_c,
                    "selected_alpha_face": selected_alpha,
                    **fold_metrics,
                }
            )
            joblib.dump(
                {
                    "experiment": experiment,
                    "task": args.task,
                    "class_names": class_names,
                    "sweet_code_order": SWEET_CODES,
                    "feature_mode": mode,
                    "summary_continuous_feature_indices": (
                        CONTINUOUS_FEATURE_INDICES
                    ),
                    "summary_motion_feature_indices": (
                        MOTION_FEATURE_INDICES
                    ),
                    "baseline_frames": args.baseline_frames,
                    "code_model": code_model,
                    "face_model": face_model,
                    "alpha_face": selected_alpha,
                    "train_subjects": train_subjects,
                    "test_subjects": test_subjects,
                },
                fold_dir / f"{experiment}.joblib",
            )
            print(
                f"Fold {fold_number}/{len(outer_splits)} "
                f"{experiment}: BAcc="
                f"{fold_metrics['balanced_accuracy']:.3f}, "
                f"macro-F1={fold_metrics['macro_f1']:.3f}"
            )

    _write_csv(args.output_dir / "predictions_condition.csv", all_rows)
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(args.output_dir / "selection_history.csv", selection_rows)
    experiment_summaries: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    for experiment in variants:
        experiment_rows = [
            row for row in all_rows if row["experiment"] == experiment
        ]
        if args.fold_index is None:
            predicted_indices = sorted(
                int(row["unit_index"]) for row in experiment_rows
            )
            if predicted_indices != list(range(len(units))):
                raise AssertionError(
                    f"{experiment} did not predict every condition exactly once"
                )
        confusion, metrics = _complete_metrics(
            experiment_rows, num_classes=num_classes
        )
        confusion_dir = args.output_dir / "confusions"
        confusion_dir.mkdir(parents=True, exist_ok=True)
        np.save(confusion_dir / f"{experiment}.npy", confusion)
        plotted = _save_confusion_plot(
            confusion,
            class_names,
            confusion_dir / f"{experiment}.png",
        )
        experiment_summaries[experiment] = {
            "metrics": metrics,
            "confusion": confusion.tolist(),
            "confusion_plot_written": plotted,
        }
        metric_rows.append({"experiment": experiment, **metrics})
    _write_csv(args.output_dir / "ablation_metrics.csv", metric_rows)
    bootstrap = (
        subject_cluster_bootstrap(
            all_rows,
            variants,
            num_classes=num_classes,
            samples=args.bootstrap_samples,
            seed=args.seed + 7001,
        )
        if args.fold_index is None
        else {}
    )
    best_experiment = max(
        variants,
        key=lambda name: (
            experiment_summaries[name]["metrics"]["balanced_accuracy"],
            experiment_summaries[name]["metrics"]["macro_f1"],
        ),
    )
    summary = {
        "task": args.task,
        "class_names": class_names,
        "partial_cv": args.fold_index is not None,
        "folds_run": sorted({int(row["fold"]) for row in fold_rows}),
        "condition_unit": "subject_id × ma_mau",
        "experiments": experiment_summaries,
        "best_by_balanced_accuracy": best_experiment,
        "subject_cluster_bootstrap": bootstrap,
        "water_used_as_unlabelled_reference": requires_water,
        "record_exclusions": exclusions,
        "condition_audit": condition_audit,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _validate_video_feature_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_detection_ratio <= 1.0:
        raise ValueError("min-detection-ratio must be in [0,1]")
    if args.min_repeats < 1 or args.baseline_frames < 2:
        raise ValueError("min-repeats must be positive and baseline-frames >= 2")
    if args.cv_folds < 2 or args.inner_folds < 2:
        raise ValueError("cv-folds and inner-folds must be at least 2")
    if args.max_iter < 1:
        raise ValueError("max-iter must be positive")
    if not args.c_grid or any(value <= 0 for value in args.c_grid):
        raise ValueError("Every C-grid value must be positive")
    if not args.k_grid or any(value < 1 for value in args.k_grid):
        raise ValueError("Every K-grid value must be a positive integer")
    if not args.modes:
        raise ValueError("At least one video mode is required")
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("modes must not contain duplicates")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be non-negative")


def run_video_features(args: argparse.Namespace) -> dict[str, Any]:
    """Run a strictly video-only selected-feature nested CV benchmark."""
    _validate_video_feature_args(args)
    modes = tuple(args.modes)
    experiments = tuple(f"video_{mode}" for mode in modes)
    requires_water = "water_delta" in modes
    records, exclusions = load_graph_records(
        args.manifest,
        include_water=requires_water,
        min_detection_ratio=args.min_detection_ratio,
    )
    store = GraphStore(records)
    units, condition_audit = build_condition_units(
        records,
        min_repeats=args.min_repeats,
        exclude_codes={WATER_CODE} if requires_water else None,
    )
    labels = _labels(units, args.task)
    num_classes = 2 if args.task == "binary" else 3
    class_names = BINARY_NAMES if args.task == "binary" else JAR3_NAMES
    if np.any(np.bincount(labels, minlength=num_classes) == 0):
        raise ValueError("The supervised condition set is missing a class")

    # This path deliberately never constructs code_feature_matrix.  ma_mau is
    # consumed only upstream by condition grouping and, for water_delta, by
    # the subject-specific WATER_CODE reference lookup.
    face_features, water_counts = build_face_feature_matrices(
        store,
        units,
        modes,
        baseline_frames=args.baseline_frames,
    )
    candidate_k_grid = list(args.k_grid)
    if args.include_all_features:
        candidate_k_grid.append(0)
    smallest_dimension = min(
        matrix.shape[1] for matrix in face_features.values()
    )
    if max(args.k_grid) > smallest_dimension:
        raise ValueError(
            f"k-grid cannot exceed input feature dimension "
            f"{smallest_dimension}"
        )
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
            raise ValueError(
                f"fold-index must be in [0,{len(outer_splits) - 1}]"
            )
        indexed_splits = [(args.fold_index, outer_splits[args.fold_index])]

    if args.output_dir is None:
        fold_name = (
            "full"
            if args.fold_index is None
            else f"fold{args.fold_index + 1:02d}"
        )
        args.output_dir = (
            Path("output/video_jar_gnn/runs_video_features")
            / args.task
            / f"{fold_name}_seed{args.seed}_{_run_signature(args)}"
        )
    _prepare_fresh_output(args.output_dir)
    config = {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in vars(args).items()
    }
    config.update(
        {
            "input_contract": "video_summary_only",
            "uses_ma_mau_as_predictor": False,
            "ma_mau_roles": [
                "condition_grouping",
                "output_metadata",
                *(
                    ["water_605_reference_selector"]
                    if requires_water
                    else []
                ),
            ],
            "condition_unit": "subject_id × ma_mau",
            "pipeline_steps_fit_inside_each_training_fold": [
                "VarianceThreshold",
                "StandardScaler",
                "SelectKBest(f_classif)",
                "LogisticRegression(class_weight=balanced)",
            ],
            "all_features_candidate": args.include_all_features,
            "repeat_handling": "mean graph before temporal summary",
            "n_records_loaded": len(records),
            "n_supervised_conditions": len(units),
            "n_subjects": len(set(_groups(units))),
            "num_classes": num_classes,
            "class_names": class_names,
            "face_feature_dimensions": {
                mode: int(matrix.shape[1])
                for mode, matrix in face_features.items()
            },
            "water_used_as_unlabelled_reference": requires_water,
            "water_reference_aggregation": (
                "mean across usable ma_mau=605 repeats"
                if requires_water
                else None
            ),
            "water_repeat_counts": water_counts,
            "record_exclusions": exclusions,
            "condition_audit": condition_audit,
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []
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
        train_subjects = sorted(
            {units[int(index)].subject_id for index in outer_train}
        )
        test_subjects = sorted(
            {units[int(index)].subject_id for index in outer_test}
        )
        majority_prediction = int(
            np.bincount(
                labels[outer_train], minlength=num_classes
            ).argmax()
        )
        fold_dir = args.output_dir / f"fold_{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for mode_index, mode in enumerate(modes):
            experiment = f"video_{mode}"
            features = face_features[mode]
            oof = video_feature_oof_probabilities(
                features,
                labels,
                inner_splits,
                candidate_k_grid,
                args.c_grid,
                num_classes=num_classes,
                max_iter=args.max_iter,
                seed=fold_seed + mode_index * 1000,
            )
            selected, candidates = select_video_feature_candidate(
                oof,
                labels,
                outer_train,
                num_classes=num_classes,
            )
            for candidate in candidates:
                selection_rows.append(
                    {
                        "fold": fold_number,
                        "experiment": experiment,
                        **candidate,
                    }
                )
            pipeline = fit_selected_video_pipeline(
                features,
                labels,
                outer_train,
                k_features=int(selected["k_features"]),
                c_value=float(selected["C"]),
                max_iter=args.max_iter,
                seed=fold_seed + 5000 + mode_index,
                num_classes=num_classes,
            )
            probabilities = _predict_probabilities(
                pipeline,
                features,
                outer_test,
                num_classes=num_classes,
            )
            predictions = probabilities.argmax(axis=1)
            original_indices = selected_original_feature_indices(pipeline)
            selected_strategy = (
                "all_after_variance"
                if int(selected["k_features"]) == 0
                else "kbest"
            )
            selected_feature_count = len(original_indices)
            for detail in selected_feature_details(pipeline):
                selected_feature_rows.append(
                    {
                        "fold": fold_number,
                        "experiment": experiment,
                        **detail,
                    }
                )
            current_rows: list[dict[str, Any]] = []
            for local_index, unit_index in enumerate(outer_test):
                unit = units[int(unit_index)]
                row: dict[str, Any] = {
                    "experiment": experiment,
                    "fold": fold_number,
                    "unit_index": int(unit_index),
                    "subject_id": unit.subject_id,
                    # Identifier/grouping metadata only; never a predictor.
                    "ma_mau": unit.ma_mau,
                    "jar": unit.jar,
                    "n_repeats": len(unit.record_indices),
                    "repeats": "|".join(map(str, unit.repeats)),
                    "true_label": int(labels[int(unit_index)]),
                    "predicted_label": int(predictions[local_index]),
                    "majority_prediction": majority_prediction,
                    "feature_mode": mode,
                    "selected_strategy": selected_strategy,
                    "selected_k_candidate": int(selected["k_features"]),
                    "selected_feature_count": selected_feature_count,
                    "selected_C": float(selected["C"]),
                }
                for class_index in range(num_classes):
                    row[f"prob_{class_index}"] = float(
                        probabilities[local_index, class_index]
                    )
                current_rows.append(row)
            all_rows.extend(current_rows)
            _, metrics = _complete_metrics(
                current_rows, num_classes=num_classes
            )
            fold_rows.append(
                {
                    "fold": fold_number,
                    "experiment": experiment,
                    "n_inner_folds": len(inner_splits),
                    "n_train_conditions": len(outer_train),
                    "n_test_conditions": len(outer_test),
                    "train_subjects": "|".join(train_subjects),
                    "test_subjects": "|".join(test_subjects),
                    "selected_strategy": selected_strategy,
                    "selected_k_candidate": int(selected["k_features"]),
                    "selected_feature_count": selected_feature_count,
                    "selected_C": float(selected["C"]),
                    **metrics,
                }
            )
            joblib.dump(
                {
                    "input_contract": "video_summary_only",
                    "uses_ma_mau_as_predictor": False,
                    "pipeline": pipeline,
                    "task": args.task,
                    "class_names": class_names,
                    "feature_mode": mode,
                    "summary_continuous_feature_indices": (
                        CONTINUOUS_FEATURE_INDICES
                    ),
                    "summary_motion_feature_indices": (
                        MOTION_FEATURE_INDICES
                    ),
                    "selected_original_feature_indices": (
                        original_indices.astype(int).tolist()
                    ),
                    "baseline_frames": args.baseline_frames,
                    "train_subjects": train_subjects,
                    "test_subjects": test_subjects,
                },
                fold_dir / f"{experiment}.joblib",
            )
            print(
                f"Fold {fold_number}/{len(outer_splits)} "
                f"{experiment}: k="
                f"{'all' if int(selected['k_features']) == 0 else int(selected['k_features'])}, "
                f"C={float(selected['C']):g}, "
                f"BAcc={metrics['balanced_accuracy']:.3f}, "
                f"macro-F1={metrics['macro_f1']:.3f}"
            )

    _write_csv(args.output_dir / "predictions_condition.csv", all_rows)
    _write_csv(args.output_dir / "fold_metrics.csv", fold_rows)
    _write_csv(args.output_dir / "selection_history.csv", selection_rows)
    _write_csv(
        args.output_dir / "selected_feature_indices.csv",
        selected_feature_rows,
    )
    feature_stability = summarize_feature_stability(
        selected_feature_rows, experiments
    )
    experiment_summaries: dict[str, Any] = {}
    metric_rows: list[dict[str, Any]] = []
    confusion_dir = args.output_dir / "confusions"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for experiment in experiments:
        experiment_rows = [
            row for row in all_rows if row["experiment"] == experiment
        ]
        if args.fold_index is None:
            predicted_indices = sorted(
                int(row["unit_index"]) for row in experiment_rows
            )
            if predicted_indices != list(range(len(units))):
                raise AssertionError(
                    f"{experiment} did not predict every condition exactly once"
                )
        confusion, metrics = _complete_metrics(
            experiment_rows, num_classes=num_classes
        )
        np.save(confusion_dir / f"{experiment}.npy", confusion)
        plotted = _save_confusion_plot(
            confusion,
            class_names,
            confusion_dir / f"{experiment}.png",
        )
        experiment_summaries[experiment] = {
            "metrics": metrics,
            "confusion": confusion.tolist(),
            "confusion_plot_written": plotted,
        }
        metric_rows.append({"experiment": experiment, **metrics})
    _write_csv(args.output_dir / "feature_metrics.csv", metric_rows)
    reference_rows = [
        row for row in all_rows if row["experiment"] == experiments[0]
    ]
    _, majority_metrics = confusion_and_metrics(
        [int(row["true_label"]) for row in reference_rows],
        [int(row["majority_prediction"]) for row in reference_rows],
        num_classes=num_classes,
    )
    bootstrap = (
        video_feature_cluster_bootstrap(
            all_rows,
            experiments,
            num_classes=num_classes,
            samples=args.bootstrap_samples,
            seed=args.seed + 7001,
        )
        if args.fold_index is None
        else {}
    )
    summary = {
        "task": args.task,
        "class_names": class_names,
        "input_contract": "video_summary_only",
        "uses_ma_mau_as_predictor": False,
        "partial_cv": args.fold_index is not None,
        "folds_run": sorted({int(row["fold"]) for row in fold_rows}),
        "condition_unit": "subject_id × ma_mau",
        "experiments": experiment_summaries,
        "baseline": {"outer_train_majority": majority_metrics},
        "subject_cluster_bootstrap": bootstrap,
        "feature_selection_stability": feature_stability,
        "water_used_as_unlabelled_reference": requires_water,
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
            "Compare condition-level code, face-summary and late-fusion "
            "classifiers with nested subject-disjoint cross-validation."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/video_jar_gnn/graph_manifest.csv"),
    )
    parser.add_argument("--task", choices=("binary", "jar3"), default="jar3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=ALL_EXPERIMENTS,
        default=list(ALL_EXPERIMENTS),
        help="Ablations to run; defaults to the complete seven-way comparison.",
    )
    parser.add_argument("--baseline-frames", type=int, default=12)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--fold-index",
        type=int,
        help="Run one zero-based outer fold as a smoke test.",
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
        help="Face weight for geometric late fusion.",
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_video_feature_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train selected-feature VIDEO-ONLY classifiers with nested "
            "subject-disjoint cross-validation. ma_mau is never a predictor."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/video_jar_gnn/graph_manifest.csv"),
    )
    parser.add_argument("--task", choices=("binary", "jar3"), default="jar3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=FEATURE_MODES,
        default=list(FEATURE_MODES),
        help=(
            "Video representations to compare. water_delta uses only the "
            "same subject's ma_mau=605 graph as an unlabelled reference."
        ),
    )
    parser.add_argument("--baseline-frames", type=int, default=12)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
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
        default=[16, 32, 64],
        help="Numbers of video summary features considered by SelectKBest.",
    )
    parser.add_argument(
        "--include-all-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also let inner CV choose all non-constant features with L2 "
            "regularization; useful when weak signal is distributed."
        ),
    )
    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 1.0],
    )
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    summary = run(args)
    print("Finished classical condition-level comparison:")
    for experiment, result in summary["experiments"].items():
        metrics = result["metrics"]
        print(
            f"  {experiment}: "
            f"BAcc={metrics['balanced_accuracy']:.3f}, "
            f"macro-F1={metrics['macro_f1']:.3f}"
        )
    print(f"Results: {args.output_dir}")
    return 0


def video_feature_main(argv: list[str] | None = None) -> int:
    args = make_video_feature_parser().parse_args(argv)
    summary = run_video_features(args)
    print("Finished VIDEO-ONLY selected-feature comparison:")
    for experiment, result in summary["experiments"].items():
        metrics = result["metrics"]
        print(
            f"  {experiment}: "
            f"BAcc={metrics['balanced_accuracy']:.3f}, "
            f"macro-F1={metrics['macro_f1']:.3f}"
        )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
