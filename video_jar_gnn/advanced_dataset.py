"""Condition-level graph data processing for the advanced video classifiers.

The original trainer treats each repeat as a training example and aggregates
probabilities only at evaluation time.  This module instead makes one
``subject × ma_mau`` condition the statistical unit and presents all available
repeats to a shared encoder.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset

from .constants import AU_NODES, FEATURE_NAMES, WATER_CODE
from .dataset import GraphRecord
from .expression import (
    EXPRESSION_FEATURES as EXPRESSION_FEATURE_NAMES,
    EXPRESSION_NODES as EXPRESSION_NODE_NAMES,
    MASK_FEATURE_INDICES as EXPRESSION_OBSERVED_MASK_INDICES,
    REPRESENTATION_VERSION as EXPRESSION_REPRESENTATION_VERSION,
    STATIC_FEATURE_INDICES as EXPRESSION_STATIC_FEATURE_INDICES,
    VELOCITY_FEATURE_INDICES as EXPRESSION_VELOCITY_FEATURE_INDICES,
)


STATIC_FEATURE_INDICES = (0, 1, 2, 4, 5)
VELOCITY_FEATURE_INDICES = (6, 7, 8, 9)
DETECTION_FEATURE_INDEX = 3
NODE_TO_INDEX = {name: index for index, name in enumerate(AU_NODES)}
REPRESENTATIONS = ("legacy", "expression_v2")


@dataclass(frozen=True)
class GraphRepresentationSchema:
    """Self-describing node/feature contract for one graph representation."""

    representation: str
    representation_version: int
    node_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    observed_mask_indices: tuple[int, ...]
    static_feature_indices: tuple[int, ...]
    velocity_feature_indices: tuple[int, ...]

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "representation": self.representation,
            "representation_version": self.representation_version,
            "node_names": list(self.node_names),
            "feature_names": list(self.feature_names),
            "observed_mask_indices": list(self.observed_mask_indices),
            "static_feature_indices": list(self.static_feature_indices),
            "velocity_feature_indices": list(self.velocity_feature_indices),
        }


LEGACY_SCHEMA = GraphRepresentationSchema(
    representation="legacy",
    representation_version=1,
    node_names=tuple(AU_NODES),
    feature_names=tuple(FEATURE_NAMES),
    observed_mask_indices=(DETECTION_FEATURE_INDEX,),
    static_feature_indices=STATIC_FEATURE_INDICES,
    velocity_feature_indices=VELOCITY_FEATURE_INDICES,
)


def _metadata_dict(data: Any, path: Path) -> dict[str, Any]:
    if "meta" not in data:
        return {}
    raw = data["meta"]
    if getattr(raw, "shape", None) == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"{path}: cache meta must be a JSON string")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: cache meta is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: cache meta JSON must be an object")
    return metadata


def _index_tuple(
    metadata: dict[str, Any],
    key: str,
    *,
    num_features: int,
    path: Path,
    require_nonempty: bool,
) -> tuple[int, ...]:
    raw = metadata.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expression meta {key!r} must be a list")
    try:
        values = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: expression meta {key!r} contains a non-integer"
        ) from exc
    if require_nonempty and not values:
        raise ValueError(f"{path}: expression meta {key!r} cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{path}: expression meta {key!r} has duplicates")
    invalid = [index for index in values if not 0 <= index < num_features]
    if invalid:
        raise ValueError(
            f"{path}: expression meta {key!r} has out-of-range indices "
            f"{invalid} for F={num_features}"
        )
    return values


def resolve_cache_schema(
    data: Any,
    *,
    representation: str,
    path: Path,
) -> GraphRepresentationSchema:
    """Resolve and validate a cache schema without weakening legacy caches."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}")
    metadata = _metadata_dict(data, path)
    declared = str(metadata.get("representation", "")).strip()
    if representation == "legacy":
        if declared not in {"", "legacy"}:
            raise ValueError(
                f"{path}: cache representation is {declared!r}, expected 'legacy'"
            )
        return LEGACY_SCHEMA

    if declared != "expression_v2":
        inferred = declared or "legacy (missing representation metadata)"
        raise ValueError(
            f"{path}: cache representation is {inferred!r}, expected "
            "'expression_v2'; use the expression graph manifest/cache"
        )
    try:
        version = int(metadata["representation_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: expression meta requires integral representation_version"
        ) from exc
    if version != EXPRESSION_REPRESENTATION_VERSION:
        raise ValueError(
            f"{path}: unsupported expression representation_version={version}; "
            f"expected {EXPRESSION_REPRESENTATION_VERSION}"
        )
    node_names_raw = metadata.get("node_names")
    feature_names_raw = metadata.get("feature_names")
    if (
        not isinstance(node_names_raw, list)
        or not node_names_raw
        or not all(isinstance(value, str) and value for value in node_names_raw)
    ):
        raise ValueError(
            f"{path}: expression meta node_names must be a non-empty string list"
        )
    if (
        not isinstance(feature_names_raw, list)
        or not feature_names_raw
        or not all(
            isinstance(value, str) and value for value in feature_names_raw
        )
    ):
        raise ValueError(
            f"{path}: expression meta feature_names must be a non-empty string list"
        )
    node_names = tuple(node_names_raw)
    feature_names = tuple(feature_names_raw)
    if len(node_names) != len(set(node_names)):
        raise ValueError(f"{path}: expression meta node_names has duplicates")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError(f"{path}: expression meta feature_names has duplicates")
    num_features = len(feature_names)
    masks = _index_tuple(
        metadata,
        "observed_mask_indices",
        num_features=num_features,
        path=path,
        require_nonempty=True,
    )
    static = _index_tuple(
        metadata,
        "static_feature_indices",
        num_features=num_features,
        path=path,
        require_nonempty=False,
    )
    velocity = _index_tuple(
        metadata,
        "velocity_feature_indices",
        num_features=num_features,
        path=path,
        require_nonempty=False,
    )
    overlap = set(masks).intersection(static).union(
        set(masks).intersection(velocity)
    )
    if overlap:
        raise ValueError(
            f"{path}: observed-mask indices cannot also be static/velocity "
            f"features: {sorted(overlap)}"
        )
    expected_contract = {
        "node_names": EXPRESSION_NODE_NAMES,
        "feature_names": EXPRESSION_FEATURE_NAMES,
        "observed_mask_indices": EXPRESSION_OBSERVED_MASK_INDICES,
        "static_feature_indices": EXPRESSION_STATIC_FEATURE_INDICES,
        "velocity_feature_indices": EXPRESSION_VELOCITY_FEATURE_INDICES,
    }
    actual_contract = {
        "node_names": node_names,
        "feature_names": feature_names,
        "observed_mask_indices": masks,
        "static_feature_indices": static,
        "velocity_feature_indices": velocity,
    }
    incompatible = {
        key: {
            "cached": list(actual_contract[key]),
            "expected": list(expected),
        }
        for key, expected in expected_contract.items()
        if actual_contract[key] != expected
    }
    if incompatible:
        raise ValueError(
            f"{path}: incompatible expression_v2 schema: {incompatible}"
        )
    return GraphRepresentationSchema(
        representation=representation,
        representation_version=version,
        node_names=node_names,
        feature_names=feature_names,
        observed_mask_indices=masks,
        static_feature_indices=static,
        velocity_feature_indices=velocity,
    )


def canonicalize_eye_rotation(graph: np.ndarray) -> np.ndarray:
    """Remove in-plane head roll using the two eye-region centres.

    The extractor already centres/scales coordinates by the eyes, but does not
    rotate them.  This transform works on existing caches and rotates both
    ``(cx, cy)`` and ``(velocity_x, velocity_y)`` consistently.
    """
    if graph.ndim != 3 or graph.shape[1:] != (
        len(AU_NODES),
        len(FEATURE_NAMES),
    ):
        raise ValueError(f"Invalid graph shape: {graph.shape}")
    result = np.asarray(graph, dtype=np.float32).copy()
    left_nodes = [
        NODE_TO_INDEX["eye_left_upper"],
        NODE_TO_INDEX["eye_left_lower"],
    ]
    right_nodes = [
        NODE_TO_INDEX["eye_right_upper"],
        NODE_TO_INDEX["eye_right_lower"],
    ]
    left = result[:, left_nodes, :2].mean(axis=1)
    right = result[:, right_nodes, :2].mean(axis=1)
    direction = right - left
    # The MediaPipe left/right convention can make the eye vector point to
    # negative x.  Flip only the vector used to estimate roll so the face is
    # never accidentally rotated by 180 degrees.
    direction[direction[:, 0] < 0] *= -1.0
    norm = np.linalg.norm(direction, axis=1)
    valid = norm > 1e-6
    cosine = np.ones(len(result), dtype=np.float32)
    sine = np.zeros(len(result), dtype=np.float32)
    cosine[valid] = direction[valid, 0] / norm[valid]
    sine[valid] = direction[valid, 1] / norm[valid]

    for x_index, y_index in ((0, 1), (6, 7)):
        x = result[:, :, x_index].copy()
        y = result[:, :, y_index].copy()
        result[:, :, x_index] = cosine[:, None] * x + sine[:, None] * y
        result[:, :, y_index] = -sine[:, None] * x + cosine[:, None] * y
    return result


def _relation_channels(
    graph: np.ndarray,
    *,
    x_index: int,
    y_index: int,
) -> np.ndarray:
    """Return compact expression distances for every time step."""
    del x_index  # Reserved for future horizontal relations.
    y = graph[:, :, y_index]

    def node(name: str) -> np.ndarray:
        return y[:, NODE_TO_INDEX[name]]

    left_eye = 0.5 * (
        node("eye_left_upper") + node("eye_left_lower")
    )
    right_eye = 0.5 * (
        node("eye_right_upper") + node("eye_right_lower")
    )
    left_brow = 0.5 * (
        node("brow_left_inner") + node("brow_left_outer")
    )
    right_brow = 0.5 * (
        node("brow_right_inner") + node("brow_right_outer")
    )
    return np.stack(
        (
            node("lower_lip") - node("upper_lip"),
            node("eye_left_lower") - node("eye_left_upper"),
            node("eye_right_lower") - node("eye_right_upper"),
            left_eye - left_brow,
            right_eye - right_brow,
            node("chin_center") - node("lower_lip"),
            node("upper_lip") - node("nose_bridge"),
        ),
        axis=1,
    ).astype(np.float32)


def append_relational_features(graph: np.ndarray, *, mode: str) -> np.ndarray:
    """Append explicit mouth/eye/brow distances to each graph node.

    Broadcasting the same seven condition channels to each node keeps the
    existing ``[T,N,F]`` graph interface.  For ``absolute_water_delta`` both
    absolute and water-delta relations are appended.
    """
    relations = [_relation_channels(graph, x_index=0, y_index=1)]
    if mode == "absolute_water_delta":
        # The appended water-delta block contains all original features except
        # detection, so its cx/cy columns are 10 and 11.
        relations.append(_relation_channels(graph, x_index=10, y_index=11))
    broadcast = [
        np.repeat(values[:, None, :], graph.shape[1], axis=1)
        for values in relations
    ]
    return np.concatenate((graph, *broadcast), axis=2).astype(
        np.float32, copy=False
    )


@dataclass(frozen=True)
class ConditionUnit:
    """One independently rated subject-by-sample condition.

    ``ma_mau`` identifies the condition for grouping and water calibration; it
    is deliberately not exposed by :class:`ConditionGraphDataset` as a model
    input.
    """

    subject_id: str
    ma_mau: int
    jar: int
    jar3_label: int
    binary_label: int
    record_indices: tuple[int, ...]
    repeats: tuple[int, ...]

    def label_for(self, task: str) -> int:
        if task == "jar3":
            return self.jar3_label
        if task == "binary":
            return self.binary_label
        raise ValueError(f"Unknown task: {task!r}")


def build_condition_units(
    records: Sequence[GraphRecord],
    *,
    min_repeats: int = 1,
    max_repeats: int = 5,
    exclude_codes: set[int] | None = None,
) -> tuple[list[ConditionUnit], dict[str, object]]:
    """Group graph records without counting repeat labels as independent."""
    if not 1 <= min_repeats <= max_repeats:
        raise ValueError("min_repeats must be between 1 and max_repeats")
    grouped: dict[tuple[str, int], list[tuple[int, GraphRecord]]] = defaultdict(list)
    for index, record in enumerate(records):
        if exclude_codes and record.ma_mau in exclude_codes:
            continue
        grouped[(record.subject_id, record.ma_mau)].append((index, record))

    units: list[ConditionUnit] = []
    excluded = 0
    repeat_histogram: Counter[int] = Counter()
    for (subject_id, code), entries in sorted(grouped.items()):
        entries.sort(key=lambda item: item[1].repeat)
        repeats = [record.repeat for _, record in entries]
        if len(repeats) != len(set(repeats)):
            raise ValueError(
                f"Duplicate repeat in condition ({subject_id},{code}): {repeats}"
            )
        if len(entries) > max_repeats:
            raise ValueError(
                f"Condition ({subject_id},{code}) has {len(entries)} repeats; "
                f"maximum is {max_repeats}"
            )
        labels = {
            (record.jar, record.jar3_label, record.binary_label)
            for _, record in entries
        }
        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent labels in condition ({subject_id},{code}): {labels}"
            )
        repeat_histogram[len(entries)] += 1
        if len(entries) < min_repeats:
            excluded += 1
            continue
        jar, jar3_label, binary_label = next(iter(labels))
        units.append(
            ConditionUnit(
                subject_id=subject_id,
                ma_mau=code,
                jar=jar,
                jar3_label=jar3_label,
                binary_label=binary_label,
                record_indices=tuple(index for index, _ in entries),
                repeats=tuple(repeats),
            )
        )
    if not units:
        raise ValueError("No condition has enough usable repeats")
    audit: dict[str, object] = {
        "conditions_before_min_repeats": len(grouped),
        "conditions_included": len(units),
        "conditions_excluded_min_repeats": excluded,
        "repeat_count_distribution": {
            str(key): value for key, value in sorted(repeat_histogram.items())
        },
    }
    return units, audit


def _observed_frames(
    graph: np.ndarray,
    observed_mask_indices: tuple[int, ...],
) -> np.ndarray:
    masks = graph[:, :, list(observed_mask_indices)]
    return (masks > 0.5).any(axis=(1, 2))


def _baseline_static(
    graph: np.ndarray,
    baseline_frames: int,
    *,
    static_feature_indices: tuple[int, ...],
    observed_mask_indices: tuple[int, ...],
) -> np.ndarray:
    count = min(max(2, int(baseline_frames)), graph.shape[0])
    early = graph[:count]
    valid = _observed_frames(early, observed_mask_indices)
    source = early[valid] if valid.any() else early
    return np.median(
        source[:, :, list(static_feature_indices)], axis=0
    )


def _neutral_static(
    baseline: np.ndarray,
    *,
    num_nodes: int,
    num_features: int,
    static_feature_indices: tuple[int, ...],
    observed_mask_indices: tuple[int, ...],
) -> np.ndarray:
    if baseline.ndim != 3 or baseline.shape[1:] != (
        num_nodes,
        num_features,
    ):
        raise ValueError(f"Invalid neutral baseline shape: {baseline.shape}")
    valid = _observed_frames(baseline, observed_mask_indices)
    if not valid.any():
        raise ValueError("Neutral baseline contains no detected face frame")
    source = baseline[valid]
    return np.median(
        source[:, :, list(static_feature_indices)], axis=0
    )


def preprocess_graph(
    graph: np.ndarray,
    *,
    mode: str,
    baseline_frames: int = 12,
    neutral_baseline: np.ndarray | None = None,
    water_reference: np.ndarray | None = None,
    num_nodes: int = len(AU_NODES),
    num_features: int = len(FEATURE_NAMES),
    static_feature_indices: tuple[int, ...] = STATIC_FEATURE_INDICES,
    velocity_feature_indices: tuple[int, ...] = VELOCITY_FEATURE_INDICES,
    observed_mask_indices: tuple[int, ...] = (DETECTION_FEATURE_INDEX,),
) -> np.ndarray:
    """Apply a label-free, within-trial or pre-trial geometric transform.

    ``trial_delta`` uses the first ``baseline_frames`` of the active trial. It
    is a pose reference, not a claim that the participant is neutral.

    ``neutral_delta`` requires a separately extracted pre-trial
    ``baseline_seq``. ``*_motion`` additionally appends absolute velocities so
    a small model can represent motion energy without learning an absolute
    value operation.
    """
    if graph.ndim != 3 or graph.shape[1:] != (
        num_nodes,
        num_features,
    ):
        raise ValueError(f"Invalid graph shape: {graph.shape}")
    valid_modes = {
        "raw",
        "trial_delta",
        "trial_delta_motion",
        "neutral_delta",
        "neutral_delta_motion",
        "water_delta",
        "absolute_water_delta",
    }
    if mode not in valid_modes:
        raise ValueError(f"Unknown preprocessing mode {mode!r}")
    if mode != "raw" and mode not in {
        "water_delta",
        "absolute_water_delta",
    } and not static_feature_indices:
        raise ValueError(
            f"{mode} preprocessing requires static_feature_indices in the "
            "representation schema"
        )
    if mode.endswith("_motion") and not velocity_feature_indices:
        raise ValueError(
            f"{mode} preprocessing requires velocity_feature_indices in the "
            "representation schema"
        )
    result = np.asarray(graph, dtype=np.float32).copy()
    if mode in {"water_delta", "absolute_water_delta"}:
        if water_reference is None or water_reference.shape != result.shape:
            raise ValueError(
                "water preprocessing requires a same-shape water reference"
            )
        delta = result.copy()
        subtract_indices = tuple(
            index
            for index in range(result.shape[2])
            if index not in observed_mask_indices
        )
        delta[:, :, subtract_indices] -= water_reference[
            :, :, subtract_indices
        ]
        if mode == "water_delta":
            result = delta
        else:
            result = np.concatenate(
                (
                    result,
                    delta[
                        :,
                        :,
                        [
                            index
                            for index in range(delta.shape[2])
                            if index not in observed_mask_indices
                        ],
                    ],
                ),
                axis=2,
            )
        return result.astype(np.float32, copy=False)
    if mode != "raw":
        if mode.startswith("neutral_"):
            if neutral_baseline is None:
                raise ValueError(
                    "neutral_delta preprocessing requires baseline_seq in every graph"
                )
            baseline = _neutral_static(
                neutral_baseline,
                num_nodes=num_nodes,
                num_features=num_features,
                static_feature_indices=static_feature_indices,
                observed_mask_indices=observed_mask_indices,
            )
        else:
            baseline = _baseline_static(
                result,
                baseline_frames,
                static_feature_indices=static_feature_indices,
                observed_mask_indices=observed_mask_indices,
            )
        for column, feature_index in enumerate(static_feature_indices):
            result[:, :, feature_index] -= baseline[:, column][None, :]
    if mode.endswith("_motion"):
        motion = np.abs(result[:, :, list(velocity_feature_indices)])
        result = np.concatenate((result, motion), axis=2)
    return result.astype(np.float32, copy=False)


class AdvancedGraphStore:
    """Preload and transform every compact graph exactly once."""

    def __init__(
        self,
        records: Sequence[GraphRecord],
        *,
        preprocess: str,
        baseline_frames: int,
        representation: str = "legacy",
        canonical_rotation: bool = False,
        relational_features: bool = False,
    ):
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation {representation!r}")
        if representation != "legacy" and canonical_rotation:
            raise ValueError(
                "canonical rotation is defined only for the legacy face graph"
            )
        if representation != "legacy" and relational_features:
            raise ValueError(
                "relational features are defined only for the legacy face graph"
            )
        self.records = list(records)
        self.preprocess = preprocess
        self.baseline_frames = int(baseline_frames)
        self.representation = representation
        self.graphs: list[np.ndarray] = []
        self.adjacencies: list[np.ndarray] = []
        self.has_neutral_baseline: list[bool] = []
        raw_graphs: list[np.ndarray] = []
        raw_adjacencies: list[np.ndarray] = []
        neutral_baselines: list[np.ndarray | None] = []
        schema: GraphRepresentationSchema | None = None
        for record in self.records:
            with np.load(record.graph_path, allow_pickle=False) as data:
                if "graph_seq" not in data or "adj" not in data:
                    raise ValueError(f"{record.graph_path}: graph_seq/adj missing")
                current_schema = resolve_cache_schema(
                    data,
                    representation=representation,
                    path=record.graph_path,
                )
                if schema is None:
                    schema = current_schema
                elif current_schema != schema:
                    raise ValueError(
                        f"{record.graph_path}: representation schema differs "
                        "from earlier graph caches"
                    )
                raw = np.asarray(data["graph_seq"], dtype=np.float32)
                adjacency = np.asarray(data["adj"], dtype=np.float32)
                neutral = (
                    np.asarray(data["baseline_seq"], dtype=np.float32)
                    if "baseline_seq" in data
                    else None
                )
            if raw.ndim != 3 or raw.shape[1:] != (
                current_schema.num_nodes,
                current_schema.num_features,
            ):
                raise ValueError(
                    f"{record.graph_path}: graph_seq shape {raw.shape} does not "
                    f"match schema [T,{current_schema.num_nodes},"
                    f"{current_schema.num_features}]"
                )
            if neutral is not None and (
                neutral.ndim != 3 or neutral.shape[1:] != raw.shape[1:]
            ):
                raise ValueError(
                    f"{record.graph_path}: baseline_seq shape {neutral.shape} "
                    f"is incompatible with graph_seq {raw.shape}"
                )
            if adjacency.shape != (
                current_schema.num_nodes,
                current_schema.num_nodes,
            ):
                raise ValueError(
                    f"{record.graph_path}: adjacency {adjacency.shape} does not "
                    f"match schema N={current_schema.num_nodes}"
                )
            raw_graphs.append(raw)
            raw_adjacencies.append(adjacency)
            neutral_baselines.append(neutral)
        if schema is None:
            raise ValueError("Cannot build a graph store from no records")
        self.input_schema = schema
        self.node_names = schema.node_names
        self.input_feature_names = schema.feature_names
        self.observed_mask_indices = schema.observed_mask_indices
        if canonical_rotation:
            raw_graphs = [
                canonicalize_eye_rotation(graph) for graph in raw_graphs
            ]
            neutral_baselines = [
                canonicalize_eye_rotation(baseline)
                if baseline is not None
                else None
                for baseline in neutral_baselines
            ]
        water_by_subject: dict[str, np.ndarray] = {}
        if preprocess in {"water_delta", "absolute_water_delta"}:
            grouped_water: dict[str, list[np.ndarray]] = defaultdict(list)
            for record, raw in zip(self.records, raw_graphs):
                if record.ma_mau == WATER_CODE:
                    grouped_water[record.subject_id].append(raw)
            missing_subjects = sorted(
                {
                    record.subject_id
                    for record in self.records
                    if record.subject_id not in grouped_water
                }
            )
            if missing_subjects:
                raise ValueError(
                    f"Water reference missing for subjects: {missing_subjects}"
                )
            water_by_subject = {
                subject: np.median(np.stack(values, axis=0), axis=0).astype(
                    np.float32
                )
                for subject, values in grouped_water.items()
            }
        expected_shape: tuple[int, int, int] | None = None
        for record, raw, adjacency, neutral in zip(
            self.records,
            raw_graphs,
            raw_adjacencies,
            neutral_baselines,
        ):
            graph = preprocess_graph(
                raw,
                mode=preprocess,
                baseline_frames=baseline_frames,
                neutral_baseline=neutral,
                water_reference=water_by_subject.get(record.subject_id),
                num_nodes=schema.num_nodes,
                num_features=schema.num_features,
                static_feature_indices=schema.static_feature_indices,
                velocity_feature_indices=schema.velocity_feature_indices,
                observed_mask_indices=schema.observed_mask_indices,
            )
            if relational_features:
                graph = append_relational_features(graph, mode=preprocess)
            if not np.isfinite(graph).all() or not np.isfinite(adjacency).all():
                raise ValueError(f"{record.graph_path}: graph contains NaN/Inf")
            if expected_shape is None:
                expected_shape = tuple(graph.shape)
            elif tuple(graph.shape) != expected_shape:
                raise ValueError(
                    f"{record.graph_path}: graph shape {graph.shape}, "
                    f"expected {expected_shape}"
                )
            self.graphs.append(graph)
            self.adjacencies.append(adjacency)
            self.has_neutral_baseline.append(neutral is not None)
        assert expected_shape is not None
        self.sequence_length, self.num_nodes, self.num_features = expected_shape
        feature_names = list(schema.feature_names)
        if preprocess.endswith("_motion"):
            feature_names.extend(
                f"abs_{schema.feature_names[index]}"
                for index in schema.velocity_feature_indices
            )
        elif preprocess == "absolute_water_delta":
            feature_names.extend(
                f"water_delta_{name}"
                for index, name in enumerate(schema.feature_names)
                if index not in schema.observed_mask_indices
            )
        if relational_features:
            relation_names = [
                "mouth_open",
                "eye_left_open",
                "eye_right_open",
                "brow_left_raise",
                "brow_right_raise",
                "chin_to_lower_lip",
                "upper_lip_to_nose",
            ]
            feature_names.extend(relation_names)
            if preprocess == "absolute_water_delta":
                feature_names.extend(
                    f"water_delta_{name}" for name in relation_names
                )
        if len(feature_names) != self.num_features:
            raise AssertionError(
                "Processed feature-name contract does not match graph shape"
            )
        self.feature_names = tuple(feature_names)


@dataclass
class AdvancedStandardizer:
    """Per-feature statistics fitted only on outer/inner training subjects."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        store: AdvancedGraphStore,
        units: Sequence[ConditionUnit],
        unit_indices: Iterable[int],
    ) -> "AdvancedStandardizer":
        record_indices = sorted(
            {
                record_index
                for unit_index in unit_indices
                for record_index in units[int(unit_index)].record_indices
            }
        )
        total = np.zeros(store.num_features, dtype=np.float64)
        total_sq = np.zeros(store.num_features, dtype=np.float64)
        count = 0
        for record_index in record_indices:
            flattened = store.graphs[record_index].reshape(-1, store.num_features)
            values = flattened.astype(np.float64, copy=False)
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += len(values)
        if count == 0:
            raise ValueError("Cannot fit normalizer on an empty training fold")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
        scale = np.sqrt(variance)
        for index in store.observed_mask_indices:
            mean[index] = 0.0
            scale[index] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, graph: np.ndarray) -> np.ndarray:
        return ((graph - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }


class ConditionGraphDataset(Dataset):
    """Return video repeat sets and labels without a sample-code feature."""

    def __init__(
        self,
        store: AdvancedGraphStore,
        units: Sequence[ConditionUnit],
        unit_indices: Sequence[int],
        *,
        task: str,
        normalizer: AdvancedStandardizer,
        training: bool,
        max_repeats: int = 5,
        temporal_crop_min: float = 0.9,
        noise_std: float = 0.01,
        repeat_dropout: float = 0.0,
    ):
        if task not in {"binary", "jar3"}:
            raise ValueError("task must be binary or jar3")
        if not 0.0 < temporal_crop_min <= 1.0:
            raise ValueError("temporal_crop_min must be in (0,1]")
        if noise_std < 0 or not 0.0 <= repeat_dropout < 1.0:
            raise ValueError("Invalid augmentation parameters")
        self.store = store
        self.units = list(units)
        self.unit_indices = np.asarray(unit_indices, dtype=np.int64)
        self.task = task
        self.normalizer = normalizer
        self.training = bool(training)
        self.max_repeats = int(max_repeats)
        self.temporal_crop_min = float(temporal_crop_min)
        self.noise_std = float(noise_std)
        self.repeat_dropout = float(repeat_dropout)

    def __len__(self) -> int:
        return len(self.unit_indices)

    def _augment(self, graph: torch.Tensor) -> torch.Tensor:
        # [V,T,F], with delta preprocessing already completed before cropping.
        if self.temporal_crop_min < 1.0 and graph.shape[1] >= 8:
            fraction = float(
                torch.empty(1).uniform_(self.temporal_crop_min, 1.0).item()
            )
            crop_length = max(4, int(round(graph.shape[1] * fraction)))
            start_max = graph.shape[1] - crop_length
            start = (
                int(torch.randint(start_max + 1, (1,)).item())
                if start_max > 0
                else 0
            )
            cropped = graph[:, start : start + crop_length, :]
            masks = {
                index: cropped[:, :, index].unsqueeze(1)
                for index in self.store.observed_mask_indices
            }
            resized = torch_functional.interpolate(
                cropped.permute(0, 2, 1),
                size=graph.shape[1],
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            for index, mask in masks.items():
                nearest = torch_functional.interpolate(
                    mask,
                    size=graph.shape[1],
                    mode="nearest",
                )
                resized[:, :, index] = nearest.squeeze(1)
            graph = resized
        if self.noise_std > 0:
            noise = torch.randn_like(graph) * self.noise_std
            for index in self.store.observed_mask_indices:
                noise[:, :, index] = 0.0
            graph = graph + noise
        return graph

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        unit_index = int(self.unit_indices[item])
        unit = self.units[unit_index]
        graphs = torch.zeros(
            (
                self.max_repeats,
                self.store.num_nodes,
                self.store.sequence_length,
                self.store.num_features,
            ),
            dtype=torch.float32,
        )
        repeat_mask = torch.zeros(self.max_repeats, dtype=torch.bool)
        adjacency: torch.Tensor | None = None
        for position, record_index in enumerate(unit.record_indices):
            normalized = self.normalizer.transform(self.store.graphs[record_index])
            graph = torch.from_numpy(normalized).permute(1, 0, 2).contiguous()
            if self.training:
                graph = self._augment(graph)
            graphs[position] = graph
            repeat_mask[position] = True
            current_adjacency = torch.from_numpy(
                self.store.adjacencies[record_index]
            ).float()
            if adjacency is None:
                adjacency = current_adjacency
            elif not torch.equal(adjacency, current_adjacency):
                raise ValueError("Adjacency differs between repeats of one condition")
        if adjacency is None:
            raise ValueError("Condition has no graph")
        if self.training and self.repeat_dropout > 0 and repeat_mask.sum() > 1:
            keep = torch.rand(self.max_repeats) >= self.repeat_dropout
            keep &= repeat_mask
            if not keep.any():
                valid = torch.nonzero(repeat_mask, as_tuple=False).flatten()
                keep[valid[torch.randint(len(valid), (1,)).item()]] = True
            repeat_mask = keep
        return {
            "graphs": graphs,
            "adjacency": adjacency,
            "repeat_mask": repeat_mask,
            "label": torch.tensor(unit.label_for(self.task), dtype=torch.long),
            "unit_index": torch.tensor(unit_index, dtype=torch.long),
            "n_repeats": torch.tensor(len(unit.record_indices), dtype=torch.long),
        }
