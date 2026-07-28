"""Label-free repeat-reliability audit for ``expression_v2`` graph caches.

The audit intentionally does not read JAR labels.  A condition identifier is
used only to match repeated measurements from the same participant.  Each
cache must contain ``graph_seq``, ``adj``, ``target_lsl`` and JSON ``meta``.
The expression schema is read from metadata, which makes this module
independent of a particular landmark or blendshape extractor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


REPRESENTATION = "expression_v2"
SUMMARY_STATISTICS = (
    "mean",
    "std",
    "quantile_10",
    "quantile_90",
    "late_minus_early",
    "slope_per_second",
)
DEFAULT_WINDOWS = ("0:2", "2:4", "4:6", "6:8", "8:10", "0:10")


@dataclass(frozen=True, order=True)
class ResponseWindow:
    """Half-open real-time response interval measured from trial onset."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.start_seconds < 0
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError(
                "A response window must satisfy "
                f"0 <= start < end, got {self.slug}"
            )

    @property
    def slug(self) -> str:
        return (
            f"{self.start_seconds:g}:{self.end_seconds:g}"
        )

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class ExpressionSchema:
    node_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    observed_mask_indices: tuple[int, ...]
    excluded_feature_indices: tuple[int, ...] = ()
    representation_version: int = 1

    @property
    def signal_feature_indices(self) -> tuple[int, ...]:
        masks = set(self.observed_mask_indices).union(
            self.excluded_feature_indices
        )
        return tuple(
            index for index in range(len(self.feature_names)) if index not in masks
        )


@dataclass(frozen=True)
class CacheIdentity:
    subject_id: str
    condition_id: str
    repeat: int


@dataclass(frozen=True)
class ExpressionCache:
    path: Path
    identity: CacheIdentity
    graph: np.ndarray
    adjacency: np.ndarray
    time_seconds: np.ndarray
    schema: ExpressionSchema


def parse_response_window(text: str) -> ResponseWindow:
    """Parse a ``START:END`` interval in seconds."""
    pieces = str(text).strip().split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            f"Window {text!r} must have START:END syntax"
        )
    try:
        return ResponseWindow(float(pieces[0]), float(pieces[1]))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _json_meta(value: np.ndarray, path: Path) -> dict[str, Any]:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{path}: meta must contain one JSON object")
    raw = array.reshape(()).item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"{path}: meta must be a JSON string")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON meta: {error}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{path}: meta JSON must be an object")
    return result


def _metadata_sources(meta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources: list[Mapping[str, Any]] = [meta]
    for key in ("schema", "expression_schema", "feature_schema"):
        nested = meta.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    return sources


def _metadata_value(
    meta: Mapping[str, Any],
    names: Sequence[str],
) -> Any:
    for source in _metadata_sources(meta):
        for name in names:
            if name in source:
                return source[name]
    return None


def _representation_name(meta: Mapping[str, Any]) -> str:
    value = _metadata_value(
        meta,
        ("representation", "representation_name", "representation_version"),
    )
    if isinstance(value, Mapping):
        value = value.get("name", value.get("version"))
    return str(value).strip() if value is not None else ""


def _string_tuple(value: Any, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{path}: metadata {field} must be a non-empty list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(
            f"{path}: metadata {field} contains empty or duplicate names"
        )
    return result


def _schema_from_meta(
    meta: Mapping[str, Any],
    path: Path,
    graph_shape: tuple[int, int, int],
) -> ExpressionSchema:
    representation = _representation_name(meta)
    if representation != REPRESENTATION:
        raise ValueError(
            f"{path}: expected representation={REPRESENTATION!r}, "
            f"got {representation!r}"
        )
    raw_version = _metadata_value(meta, ("representation_version",))
    try:
        representation_version = (
            1 if raw_version is None else int(raw_version)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: representation_version must be an integer"
        ) from error
    if representation_version != 1:
        raise ValueError(
            f"{path}: unsupported expression_v2 "
            f"representation_version={representation_version}; expected 1"
        )
    node_names = _string_tuple(
        _metadata_value(meta, ("node_names",)),
        "node_names",
        path,
    )
    feature_names = _string_tuple(
        _metadata_value(meta, ("feature_names",)),
        "feature_names",
        path,
    )
    raw_masks = _metadata_value(meta, ("observed_mask_indices",))
    if raw_masks is None:
        singular_mask = _metadata_value(
            meta, ("observed_feature_index", "observed_mask_index")
        )
        raw_masks = [] if singular_mask is None else [singular_mask]
    if not isinstance(raw_masks, (list, tuple)):
        raise ValueError(
            f"{path}: metadata observed_mask_indices must be a list"
        )
    try:
        masks = tuple(int(value) for value in raw_masks)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: observed_mask_indices must contain integers"
        ) from error
    if len(masks) != len(set(masks)):
        raise ValueError(f"{path}: observed_mask_indices contains duplicates")
    _, num_nodes, num_features = graph_shape
    if len(node_names) != num_nodes or len(feature_names) != num_features:
        raise ValueError(
            f"{path}: schema sizes nodes={len(node_names)}, "
            f"features={len(feature_names)} do not match graph {graph_shape}"
        )
    if any(index < 0 or index >= num_features for index in masks):
        raise ValueError(
            f"{path}: observed_mask_indices outside [0,{num_features - 1}]"
        )
    raw_excluded = _metadata_value(
        meta,
        ("excluded_feature_indices", "non_signal_feature_indices"),
    )
    if raw_excluded is None:
        singular_imputed = _metadata_value(meta, ("imputed_feature_index",))
        raw_excluded = (
            [] if singular_imputed is None else [singular_imputed]
        )
    if not isinstance(raw_excluded, (list, tuple)):
        raise ValueError(
            f"{path}: excluded_feature_indices must be a list"
        )
    try:
        excluded = tuple(int(value) for value in raw_excluded)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{path}: excluded_feature_indices must contain integers"
        ) from error
    # ``imputed`` is a quality indicator in the expression_v2 contract, not
    # an expression value.  Infer it for caches that only name the feature.
    excluded = tuple(
        dict.fromkeys(
            (
                *excluded,
                *(
                    [feature_names.index("imputed")]
                    if "imputed" in feature_names
                    else []
                ),
            )
        )
    )
    if any(index < 0 or index >= num_features for index in excluded):
        raise ValueError(
            f"{path}: excluded_feature_indices outside [0,{num_features - 1}]"
        )
    schema = ExpressionSchema(
        node_names,
        feature_names,
        masks,
        excluded,
        representation_version,
    )
    if not schema.signal_feature_indices:
        raise ValueError(f"{path}: schema contains no non-mask signal feature")
    return schema


def _identity_value(
    meta: Mapping[str, Any],
    names: Sequence[str],
) -> Any:
    value = _metadata_value(meta, names)
    if value is None:
        return None
    return value


def _identity_from_meta(
    meta: Mapping[str, Any],
    path: Path,
    override: Mapping[str, Any] | None = None,
) -> CacheIdentity:
    values: dict[str, Any] = dict(override or {})
    subject = values.get("subject_id")
    if subject is None:
        subject = _identity_value(meta, ("subject_id", "participant_id"))
    condition = values.get("condition_id")
    if condition is None:
        condition = _identity_value(
            meta,
            ("condition_id", "stimulus_id", "sample_code", "ma_mau"),
        )
    repeat = values.get("repeat")
    if repeat is None:
        repeat = _identity_value(meta, ("repeat", "repetition", "lan_lap"))
    subject_text = str(subject).strip() if subject is not None else ""
    condition_text = str(condition).strip() if condition is not None else ""
    try:
        repeat_number = int(float(repeat))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: missing/invalid repeat in metadata") from error
    if not subject_text or not condition_text or repeat_number < 1:
        raise ValueError(
            f"{path}: metadata must identify subject_id, condition_id and "
            "a positive repeat"
        )
    return CacheIdentity(subject_text, condition_text, repeat_number)


def load_expression_cache(
    path: Path,
    *,
    identity_override: Mapping[str, Any] | None = None,
) -> ExpressionCache:
    """Load and strictly validate one generic expression graph cache."""
    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as data:
        required = {"graph_seq", "adj", "target_lsl", "meta"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{cache_path}: missing arrays {sorted(missing)}")
        graph = np.asarray(data["graph_seq"], dtype=np.float32)
        adjacency = np.asarray(data["adj"], dtype=np.float32)
        target_lsl = np.asarray(data["target_lsl"], dtype=np.float64)
        meta = _json_meta(data["meta"], cache_path)
    if graph.ndim != 3 or graph.shape[0] < 2:
        raise ValueError(
            f"{cache_path}: graph_seq must be [T,N,F] with T>=2, "
            f"got {graph.shape}"
        )
    if adjacency.shape != (graph.shape[1], graph.shape[1]):
        raise ValueError(
            f"{cache_path}: adjacency {adjacency.shape} incompatible with "
            f"N={graph.shape[1]}"
        )
    if target_lsl.shape != (graph.shape[0],):
        raise ValueError(
            f"{cache_path}: target_lsl {target_lsl.shape} incompatible with "
            f"T={graph.shape[0]}"
        )
    if (
        not np.isfinite(graph).all()
        or not np.isfinite(adjacency).all()
        or not np.isfinite(target_lsl).all()
    ):
        raise ValueError(f"{cache_path}: graph/timing contains NaN or Inf")
    if np.any(np.diff(target_lsl) <= 0):
        raise ValueError(f"{cache_path}: target_lsl must be strictly increasing")
    schema = _schema_from_meta(meta, cache_path, tuple(graph.shape))
    for index in schema.observed_mask_indices:
        mask = graph[:, :, index]
        if np.any((mask < -1e-6) | (mask > 1.0 + 1e-6)):
            raise ValueError(
                f"{cache_path}: observed mask feature "
                f"{schema.feature_names[index]!r} is outside [0,1]"
            )
    identity = _identity_from_meta(meta, cache_path, identity_override)
    return ExpressionCache(
        path=cache_path,
        identity=identity,
        graph=graph,
        adjacency=adjacency,
        time_seconds=target_lsl - target_lsl[0],
        schema=schema,
    )


def _manifest_records(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if "graph_path" not in fields:
            raise ValueError(f"{path}: manifest must contain graph_path")
        condition_field = next(
            (
                field
                for field in (
                    "condition_id",
                    "stimulus_id",
                    "sample_code",
                    "ma_mau",
                )
                if field in fields
            ),
            None,
        )
        for line_number, row in enumerate(reader, start=2):
            graph_text = str(row.get("graph_path", "")).strip()
            if not graph_text:
                status = str(row.get("extract_status", "")).strip().lower()
                if "extract_status" in fields and status in {
                    "not_selected",
                    "error",
                }:
                    continue
                raise ValueError(f"{path}:{line_number}: empty graph_path")
            graph_path = Path(graph_text)
            if not graph_path.is_absolute() and not graph_path.is_file():
                graph_path = path.parent / graph_path
            override: dict[str, Any] = {}
            if str(row.get("subject_id", "")).strip():
                override["subject_id"] = row["subject_id"]
            if condition_field and str(row.get(condition_field, "")).strip():
                override["condition_id"] = row[condition_field]
            if str(row.get("repeat", "")).strip():
                override["repeat"] = row["repeat"]
            records.append((graph_path, override))
    if not records:
        raise ValueError(f"{path}: manifest is empty")
    return records


def discover_cache_records(
    *,
    cache_dir: Path | None,
    manifest: Path | None,
) -> list[tuple[Path, dict[str, Any]]]:
    if manifest is not None:
        if not manifest.is_file():
            raise ValueError(f"Manifest does not exist: {manifest}")
        return _manifest_records(manifest)
    if cache_dir is None:
        default_manifest = Path(
            "output/video_jar_gnn/graph_manifest_expression_v2.csv"
        )
        if default_manifest.is_file():
            return _manifest_records(default_manifest)
        default_directories = (
            Path("output/video_jar_gnn/expression_graphs"),
            Path("output/video_jar_gnn/graphs_expression_v2"),
        )
        cache_dir = next(
            (path for path in default_directories if path.is_dir()),
            None,
        )
    if cache_dir is None or not cache_dir.is_dir():
        raise ValueError(
            "No expression_v2 input found. Pass --manifest or --cache-dir; "
            "the automatic manifest path is "
            "output/video_jar_gnn/graph_manifest_expression_v2.csv"
        )
    paths = sorted(cache_dir.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz cache found under {cache_dir}")
    return [(path, {}) for path in paths]


def load_expression_caches(
    records: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    exclude_conditions: Iterable[str] = (),
) -> list[ExpressionCache]:
    excluded = {str(value).strip() for value in exclude_conditions}
    caches: list[ExpressionCache] = []
    expected_schema: ExpressionSchema | None = None
    seen: set[tuple[str, str, int]] = set()
    for path, override in records:
        cache = load_expression_cache(path, identity_override=override)
        if cache.identity.condition_id in excluded:
            continue
        if expected_schema is None:
            expected_schema = cache.schema
        elif cache.schema != expected_schema:
            raise ValueError(
                f"{path}: expression schema differs from "
                f"{caches[0].path}"
            )
        key = (
            cache.identity.subject_id,
            cache.identity.condition_id,
            cache.identity.repeat,
        )
        if key in seen:
            raise ValueError(f"Duplicate subject/condition/repeat: {key}")
        seen.add(key)
        caches.append(cache)
    if not caches:
        raise ValueError("No expression cache remains after filtering")
    return caches


def window_indices(
    time_seconds: np.ndarray,
    window: ResponseWindow,
) -> np.ndarray:
    """Return samples in ``[start,end)``; include the last trial sample."""
    times = np.asarray(time_seconds, dtype=np.float64)
    tolerance = max(1e-9, float(np.median(np.diff(times))) * 1e-6)
    include_end = window.end_seconds >= float(times[-1]) - tolerance
    if include_end:
        selected = (times >= window.start_seconds - tolerance) & (
            times <= window.end_seconds + tolerance
        )
    else:
        selected = (times >= window.start_seconds - tolerance) & (
            times < window.end_seconds - tolerance
        )
    indices = np.flatnonzero(selected)
    if len(indices) < 2:
        raise ValueError(
            f"Window {window.slug} contains only {len(indices)} time points "
            f"within cache duration 0:{times[-1]:g}"
        )
    return indices


def summarize_expression_window(
    graph: np.ndarray,
    time_seconds: np.ndarray,
    schema: ExpressionSchema,
    window: ResponseWindow,
) -> np.ndarray:
    """Create node/feature summaries with a true per-second OLS slope."""
    indices = window_indices(time_seconds, window)
    values = np.asarray(graph, dtype=np.float64)[indices]
    times = np.asarray(time_seconds, dtype=np.float64)[indices]
    signal = values[:, :, schema.signal_feature_indices]
    if schema.observed_mask_indices:
        masks = values[:, :, schema.observed_mask_indices]
        observed = np.any(masks >= 0.5, axis=2)
    else:
        observed = np.ones(signal.shape[:2], dtype=bool)
    num_nodes = signal.shape[1]
    num_features = signal.shape[2]
    result = np.full(
        (num_nodes, num_features, len(SUMMARY_STATISTICS)),
        np.nan,
        dtype=np.float64,
    )
    for node_index in range(num_nodes):
        valid = observed[:, node_index]
        if int(valid.sum()) < 2:
            continue
        node_values = signal[valid, node_index, :]
        node_times = times[valid]
        edge_count = max(1, int(round(len(node_times) * 0.2)))
        centered_time = node_times - node_times.mean()
        denominator = float(np.square(centered_time).sum())
        slope = (
            np.tensordot(centered_time, node_values, axes=(0, 0))
            / denominator
            if denominator > 0
            else np.full(num_features, np.nan)
        )
        result[node_index] = np.stack(
            (
                node_values.mean(axis=0),
                node_values.std(axis=0),
                np.quantile(node_values, 0.10, axis=0),
                np.quantile(node_values, 0.90, axis=0),
                node_values[-edge_count:].mean(axis=0)
                - node_values[:edge_count].mean(axis=0),
                slope,
            ),
            axis=1,
        )
    return result.reshape(-1).astype(np.float32)


def summary_feature_rows(schema: ExpressionSchema) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for node_name in schema.node_names:
        for feature_index in schema.signal_feature_indices:
            for statistic in SUMMARY_STATISTICS:
                rows.append(
                    {
                        "summary_feature_index": index,
                        "node": node_name,
                        "source_feature": schema.feature_names[feature_index],
                        "statistic": statistic,
                    }
                )
                index += 1
    return rows


def _unbalanced_icc(
    values: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float, float, int]:
    """One-way random-effects ICC(1,1)/(1,k) with unequal group sizes."""
    vector = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups)
    valid = np.isfinite(vector)
    grouped = [
        vector[valid & (group_values == group)]
        for group in np.unique(group_values[valid])
    ]
    grouped = [items for items in grouped if len(items) >= 2]
    n_groups = len(grouped)
    total_count = sum(len(items) for items in grouped)
    if n_groups < 2 or total_count <= n_groups:
        return math.nan, math.nan, math.nan, n_groups
    sizes = np.asarray([len(items) for items in grouped], dtype=np.float64)
    means = np.asarray([items.mean() for items in grouped])
    grand = float(
        sum(float(items.sum()) for items in grouped) / total_count
    )
    between_ss = float(np.sum(sizes * np.square(means - grand)))
    within_ss = float(
        sum(np.square(items - items.mean()).sum() for items in grouped)
    )
    ms_between = between_ss / (n_groups - 1)
    ms_within = within_ss / (total_count - n_groups)
    effective_k = float(
        (
            total_count
            - float(np.square(sizes).sum()) / float(total_count)
        )
        / (n_groups - 1)
    )
    denominator = ms_between + (effective_k - 1.0) * ms_within
    icc1 = (
        (ms_between - ms_within) / denominator
        if denominator > 0
        else math.nan
    )
    icck = (
        (ms_between - ms_within) / ms_between
        if ms_between > 0
        else math.nan
    )
    return float(icc1), float(icck), effective_k, n_groups


def _subject_center(
    matrix: np.ndarray,
    subjects: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(matrix, dtype=np.float64).copy()
    for subject in np.unique(subjects):
        rows = subjects == subject
        centered[rows] -= np.nanmean(centered[rows], axis=0)
    return centered


def feature_reliability(
    matrix: np.ndarray,
    subjects: np.ndarray,
    condition_groups: np.ndarray,
) -> list[dict[str, Any]]:
    centered = _subject_center(matrix, subjects)
    rows: list[dict[str, Any]] = []
    for index in range(matrix.shape[1]):
        raw_icc1, raw_icck, raw_k, raw_groups = _unbalanced_icc(
            matrix[:, index], condition_groups
        )
        centered_icc1, centered_icck, centered_k, centered_groups = (
            _unbalanced_icc(centered[:, index], condition_groups)
        )
        rows.append(
            {
                "summary_feature_index": index,
                "raw_icc1": raw_icc1,
                "raw_icck": raw_icck,
                "raw_effective_repeats": raw_k,
                "raw_n_conditions": raw_groups,
                "subject_centered_icc1": centered_icc1,
                "subject_centered_icck": centered_icck,
                "subject_centered_effective_repeats": centered_k,
                "subject_centered_n_conditions": centered_groups,
            }
        )
    return rows


def _standardize_for_distance(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    finite_counts = np.isfinite(values).sum(axis=0)
    means = np.nanmean(values, axis=0)
    scales = np.nanstd(values, axis=0)
    usable = (
        (finite_counts >= 2)
        & np.isfinite(means)
        & np.isfinite(scales)
        & (scales > 1e-12)
    )
    if not usable.any():
        raise ValueError("No finite non-constant summary feature for distances")
    return (values[:, usable] - means[usable]) / scales[usable], usable


def _rms_distance(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if not valid.any():
        return math.nan
    return float(np.sqrt(np.mean(np.square(left[valid] - right[valid]))))


def distance_reliability(
    matrix: np.ndarray,
    identities: Sequence[CacheIdentity],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subjects = np.asarray(
        [identity.subject_id for identity in identities], dtype=object
    )
    # Centre before estimating scale so stable morphology does not dominate
    # the denominator of a within-subject expression distance.
    standardized, usable = _standardize_for_distance(
        _subject_center(matrix, subjects)
    )
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities):
        by_subject[identity.subject_id].append(index)
    subject_rows: list[dict[str, Any]] = []
    all_within: list[float] = []
    all_between: list[float] = []
    for subject, indices in sorted(by_subject.items()):
        within: list[float] = []
        between: list[float] = []
        for left_index, right_index in combinations(indices, 2):
            left_identity = identities[left_index]
            right_identity = identities[right_index]
            distance = _rms_distance(
                standardized[left_index], standardized[right_index]
            )
            if not math.isfinite(distance):
                continue
            if left_identity.condition_id == right_identity.condition_id:
                if left_identity.repeat != right_identity.repeat:
                    within.append(distance)
            else:
                between.append(distance)
        auc = _distance_auc(within, between)
        within_median = _finite_median(within)
        between_median = _finite_median(between)
        ratio, separation = _distance_contrasts(
            within_median, between_median
        )
        subject_rows.append(
            {
                "subject_id": subject,
                "n_within_pairs": len(within),
                "n_between_pairs": len(between),
                "within_condition_distance_median": within_median,
                "different_condition_same_subject_distance_median": (
                    between_median
                ),
                "within_to_between_ratio": ratio,
                "distance_separation": separation,
                "pair_auc": auc,
            }
        )
        all_within.extend(within)
        all_between.extend(between)
    within_median = _finite_median(all_within)
    between_median = _finite_median(all_between)
    ratio, separation = _distance_contrasts(
        within_median, between_median
    )
    aggregate = {
        "n_usable_distance_features": int(usable.sum()),
        "n_within_pairs": len(all_within),
        "n_between_pairs": len(all_between),
        "within_condition_distance_median": within_median,
        "different_condition_same_subject_distance_median": between_median,
        "within_to_between_ratio": ratio,
        "distance_separation": separation,
        "pair_auc": _distance_auc(all_within, all_between),
        "median_subject_pair_auc": _finite_median(
            [row["pair_auc"] for row in subject_rows]
        ),
    }
    return aggregate, subject_rows


def _distance_auc(within: Sequence[float], between: Sequence[float]) -> float:
    if not within or not between:
        return math.nan
    labels = np.concatenate(
        (np.ones(len(within), dtype=np.int64), np.zeros(len(between), dtype=np.int64))
    )
    scores = -np.concatenate(
        (
            np.asarray(within, dtype=np.float64),
            np.asarray(between, dtype=np.float64),
        )
    )
    return float(roc_auc_score(labels, scores))


def _finite_median(values: Iterable[Any]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if len(finite) else math.nan


def _distance_contrasts(
    within: float,
    between: float,
) -> tuple[float, float]:
    if not math.isfinite(within) or not math.isfinite(between) or between <= 0:
        return math.nan, math.nan
    return within / between, (between - within) / between


def _distribution(prefix: str, values: Iterable[Any]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            f"{prefix}_n_features": 0,
            f"{prefix}_median": math.nan,
            f"{prefix}_q25": math.nan,
            f"{prefix}_q75": math.nan,
            f"{prefix}_fraction_gt_0": math.nan,
            f"{prefix}_fraction_ge_0_25": math.nan,
            f"{prefix}_fraction_ge_0_5": math.nan,
        }
    return {
        f"{prefix}_n_features": int(len(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"{prefix}_fraction_gt_0": float(np.mean(finite > 0)),
        f"{prefix}_fraction_ge_0_25": float(np.mean(finite >= 0.25)),
        f"{prefix}_fraction_ge_0_5": float(np.mean(finite >= 0.5)),
    }


def audit_window(
    caches: Sequence[ExpressionCache],
    window: ResponseWindow,
    *,
    min_repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[ExpressionCache]] = defaultdict(list)
    for cache in caches:
        grouped[
            (cache.identity.subject_id, cache.identity.condition_id)
        ].append(cache)
    included = {
        key: sorted(items, key=lambda cache: cache.identity.repeat)
        for key, items in grouped.items()
        if len(items) >= min_repeats
    }
    if len(included) < 2:
        raise ValueError(
            f"Window {window.slug}: fewer than two conditions have "
            f">={min_repeats} repeats"
        )
    ordered = [
        cache
        for key in sorted(included)
        for cache in included[key]
    ]
    observation_ratios: list[float] = []
    imputation_ratios: list[float] = []
    for cache in ordered:
        selected = window_indices(cache.time_seconds, window)
        if cache.schema.observed_mask_indices:
            observed = cache.graph[selected][
                :, :, list(cache.schema.observed_mask_indices)
            ]
            observation_ratios.append(
                float(np.mean(np.any(observed >= 0.5, axis=2)))
            )
        if "imputed" in cache.schema.feature_names:
            imputed_index = cache.schema.feature_names.index("imputed")
            imputation_ratios.append(
                float(np.mean(cache.graph[selected, :, imputed_index] >= 0.5))
            )
    summaries = np.stack(
        [
            summarize_expression_window(
                cache.graph,
                cache.time_seconds,
                cache.schema,
                window,
            )
            for cache in ordered
        ],
        axis=0,
    )
    identities = [cache.identity for cache in ordered]
    subjects = np.asarray(
        [identity.subject_id for identity in identities], dtype=object
    )
    condition_groups = np.asarray(
        [
            f"{identity.subject_id}\x1f{identity.condition_id}"
            for identity in identities
        ],
        dtype=object,
    )
    reliability = feature_reliability(
        summaries, subjects, condition_groups
    )
    descriptors = summary_feature_rows(ordered[0].schema)
    feature_rows = [
        {
            "window": window.slug,
            **descriptors[index],
            **row,
        }
        for index, row in enumerate(reliability)
    ]
    distance_metrics, subject_rows = distance_reliability(
        summaries, identities
    )
    for row in subject_rows:
        row["window"] = window.slug
    repeat_counts = [len(items) for items in included.values()]
    window_row: dict[str, Any] = {
        "window": window.slug,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
        "duration_seconds": window.duration_seconds,
        "n_subjects": len({key[0] for key in included}),
        "n_conditions": len(included),
        "n_repeat_records": len(ordered),
        "min_repeats": min(repeat_counts),
        "median_repeats": float(np.median(repeat_counts)),
        "max_repeats": max(repeat_counts),
        "n_summary_features": summaries.shape[1],
        "conditions_excluded_too_few_repeats": len(grouped) - len(included),
        "observed_ratio_min": (
            float(np.min(observation_ratios))
            if observation_ratios
            else math.nan
        ),
        "observed_ratio_median": _finite_median(observation_ratios),
        "observed_ratio_mean": (
            float(np.mean(observation_ratios))
            if observation_ratios
            else math.nan
        ),
        "imputed_ratio_median": _finite_median(imputation_ratios),
        **_distribution(
            "raw_icc1", [row["raw_icc1"] for row in reliability]
        ),
        **_distribution(
            "raw_icck", [row["raw_icck"] for row in reliability]
        ),
        **_distribution(
            "subject_centered_icc1",
            [row["subject_centered_icc1"] for row in reliability],
        ),
        **_distribution(
            "subject_centered_icck",
            [row["subject_centered_icck"] for row in reliability],
        ),
        **distance_metrics,
    }
    return window_row, feature_rows, subject_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        ""
                        if isinstance(value, float) and not math.isfinite(value)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _run_signature(args: argparse.Namespace) -> str:
    payload = {
        key: (
            [item.slug for item in value]
            if key == "windows"
            else str(value)
            if isinstance(value, Path)
            else value
        )
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:10]


def _prepare_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {path}; choose another path"
        )
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_repeats < 2:
        raise ValueError("min-repeats must be at least 2 for reliability")
    if len(args.windows) != len(set(args.windows)):
        raise ValueError("Response windows must not contain duplicates")
    records = discover_cache_records(
        cache_dir=args.cache_dir,
        manifest=args.manifest,
    )
    excluded_conditions = list(args.exclude_condition)
    if not args.include_water and "605" not in excluded_conditions:
        excluded_conditions.append("605")
    caches = load_expression_caches(
        records,
        exclude_conditions=excluded_conditions,
    )
    if args.output_dir is None:
        args.output_dir = (
            Path("output/video_jar_gnn/expression_audit")
            / f"run_{_run_signature(args)}"
        )
    _prepare_output(args.output_dir)
    window_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for window in args.windows:
        window_row, current_features, current_subjects = audit_window(
            caches,
            window,
            min_repeats=args.min_repeats,
        )
        window_rows.append(window_row)
        feature_rows.extend(current_features)
        subject_rows.extend(current_subjects)
    _write_csv(args.output_dir / "window_metrics.csv", window_rows)
    _write_csv(args.output_dir / "feature_reliability.csv", feature_rows)
    _write_csv(args.output_dir / "subject_distances.csv", subject_rows)
    schema = caches[0].schema
    condition_counts = Counter(
        cache.identity.condition_id for cache in caches
    )
    summary = {
        "audit_type": "label_free_repeat_reliability",
        "representation": REPRESENTATION,
        "uses_jar_labels": False,
        "uses_condition_id_as_predictor": False,
        "condition_id_role": "repeat grouping only",
        "distance_scaling_scope": (
            "descriptive subject-centred z-score across repeat summaries; "
            "no model fit"
        ),
        "input": {
            "manifest": str(args.manifest) if args.manifest else None,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "automatic_discovery": (
                args.manifest is None and args.cache_dir is None
            ),
        },
        "min_repeats": args.min_repeats,
        "cache_count": len(caches),
        "subject_count": len(
            {cache.identity.subject_id for cache in caches}
        ),
        "condition_record_counts": dict(sorted(condition_counts.items())),
        "water_included": bool(args.include_water),
        "excluded_conditions": excluded_conditions,
        "schema": {
            "representation_version": schema.representation_version,
            "node_names": list(schema.node_names),
            "feature_names": list(schema.feature_names),
            "observed_mask_indices": list(schema.observed_mask_indices),
            "excluded_feature_indices": list(
                schema.excluded_feature_indices
            ),
            "summary_statistics": list(SUMMARY_STATISTICS),
        },
        "windows": window_rows,
        "outputs": {
            "window_metrics": "window_metrics.csv",
            "feature_reliability": "feature_reliability.csv",
            "subject_distances": "subject_distances.csv",
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit label-free repeat reliability of generic expression_v2 "
            "graph caches over real-time response windows."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Directory searched recursively for expression_v2 .npz caches. "
            "When neither input option is given, the expression_v2 manifest "
            "used by train-advanced is discovered automatically."
        ),
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Optional CSV with graph_path and optional subject_id, "
            "condition_id/ma_mau, repeat overrides."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--windows",
        nargs="+",
        type=parse_response_window,
        default=[parse_response_window(value) for value in DEFAULT_WINDOWS],
        metavar="START:END",
        help="Real-time windows in seconds from trial onset.",
    )
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument(
        "--exclude-condition",
        nargs="*",
        default=[],
        help=(
            "Additional condition identifiers excluded from the audit."
        ),
    )
    parser.add_argument(
        "--include-water",
        action="store_true",
        help=(
            "Include condition 605. By default water is excluded so the "
            "audit matches the supervised sweet-condition population."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    summary = run(args)
    print("Finished label-free expression repeat audit:")
    for row in summary["windows"]:
        print(
            f"  {row['window']}: "
            f"centered ICC1={row['subject_centered_icc1_median']:.3f}, "
            f"within/between={row['within_to_between_ratio']:.3f}, "
            f"pair AUC={row['pair_auc']:.3f}"
        )
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
