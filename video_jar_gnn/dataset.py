"""Graph cache loading, fold-only normalization and PyTorch datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset

from .constants import AU_NODES, FEATURE_NAMES, WATER_CODE


@dataclass(frozen=True)
class GraphRecord:
    sample_id: str
    subject_id: str
    ma_mau: int
    repeat: int
    jar: int
    jar3_label: int
    binary_label: int
    graph_path: Path
    detection_ratio: float

    def label_for(self, task: str) -> int:
        if task == "jar3":
            return self.jar3_label
        if task == "binary":
            return self.binary_label
        raise ValueError(f"Unknown task: {task!r}")


def _int_field(row: dict[str, str], name: str) -> int:
    return int(float(row[name]))


def _float_field(row: dict[str, str], name: str, default: float) -> float:
    text = str(row.get(name, "")).strip()
    return float(text) if text else default


def load_graph_records(
    manifest: Path,
    *,
    include_water: bool = False,
    min_detection_ratio: float = 0.5,
) -> tuple[list[GraphRecord], dict[str, int]]:
    """Load usable graph rows and return explicit exclusion counts."""
    records: list[GraphRecord] = []
    counts = {
        "manifest_rows": 0,
        "excluded_water": 0,
        "excluded_status": 0,
        "excluded_quality": 0,
        "excluded_missing_graph": 0,
        "included": 0,
    }
    seen_sample_ids: set[str] = set()
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "subject_id",
            "ma_mau",
            "repeat",
            "jar",
            "jar3_label",
            "binary_label",
            "graph_path",
            "extract_status",
            "detection_ratio",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{manifest}: missing fields {sorted(missing)}")
        for row in reader:
            counts["manifest_rows"] += 1
            sample_id = str(row["sample_id"]).strip()
            if not sample_id:
                raise ValueError(f"{manifest}: empty sample_id")
            if sample_id in seen_sample_ids:
                raise ValueError(f"{manifest}: duplicate sample_id={sample_id}")
            seen_sample_ids.add(sample_id)
            code = _int_field(row, "ma_mau")
            if code == WATER_CODE and not include_water:
                counts["excluded_water"] += 1
                continue
            status = str(row["extract_status"]).strip()
            if status not in {"ok", "cached", "low_quality"}:
                counts["excluded_status"] += 1
                continue
            detection_ratio = _float_field(row, "detection_ratio", 1.0)
            if not np.isfinite(detection_ratio) or not 0.0 <= detection_ratio <= 1.0:
                raise ValueError(
                    f"{manifest}: invalid detection_ratio={detection_ratio!r} "
                    f"for {sample_id}"
                )
            if detection_ratio < min_detection_ratio:
                counts["excluded_quality"] += 1
                continue
            path_text = str(row.get("graph_path", "")).strip()
            graph_path = Path(path_text) if path_text else Path("__missing_graph__")
            if not path_text or not graph_path.is_file():
                counts["excluded_missing_graph"] += 1
                continue
            jar = _int_field(row, "jar")
            jar3_label = _int_field(row, "jar3_label")
            binary_label = _int_field(row, "binary_label")
            if jar not in (1, 2, 3, 4, 5):
                raise ValueError(f"{manifest}: invalid JAR={jar} for {sample_id}")
            if jar3_label not in (0, 1, 2) or binary_label not in (0, 1):
                raise ValueError(f"{manifest}: invalid target label for {sample_id}")
            records.append(
                GraphRecord(
                    sample_id=sample_id,
                    subject_id=row["subject_id"],
                    ma_mau=code,
                    repeat=_int_field(row, "repeat"),
                    jar=jar,
                    jar3_label=jar3_label,
                    binary_label=binary_label,
                    graph_path=graph_path,
                    detection_ratio=detection_ratio,
                )
            )
    counts["included"] = len(records)
    if not records:
        raise ValueError(
            f"No usable graphs in {manifest}; exclusion summary: {counts}"
        )
    return records, counts


class GraphStore:
    """Preload the compact graph cache once to avoid repeated NPZ decompression."""

    def __init__(self, records: Sequence[GraphRecord]):
        self.records = list(records)
        self.graphs: list[np.ndarray] = []
        self.adjacencies: list[np.ndarray] = []
        expected_shape: tuple[int, int, int] | None = None
        for record in self.records:
            with np.load(record.graph_path, allow_pickle=False) as data:
                graph = np.asarray(data["graph_seq"], dtype=np.float32)
                adjacency = np.asarray(data["adj"], dtype=np.float32)
            if graph.ndim != 3:
                raise ValueError(
                    f"{record.graph_path}: graph_seq must be [T,N,F], got {graph.shape}"
                )
            if graph.shape[1:] != (len(AU_NODES), len(FEATURE_NAMES)):
                raise ValueError(
                    f"{record.graph_path}: expected graph nodes/features "
                    f"[T,{len(AU_NODES)},{len(FEATURE_NAMES)}], got {graph.shape}"
                )
            if expected_shape is None:
                expected_shape = tuple(graph.shape)
            elif tuple(graph.shape) != expected_shape:
                raise ValueError(
                    f"Inconsistent graph shape: {record.graph_path} has {graph.shape}, "
                    f"expected {expected_shape}"
                )
            if adjacency.shape != (graph.shape[1], graph.shape[1]):
                raise ValueError(
                    f"{record.graph_path}: adjacency {adjacency.shape} is incompatible "
                    f"with N={graph.shape[1]}"
                )
            if not np.isfinite(graph).all() or not np.isfinite(adjacency).all():
                raise ValueError(f"{record.graph_path}: graph contains NaN/Inf")
            self.graphs.append(graph)
            self.adjacencies.append(adjacency)
        assert expected_shape is not None
        self.sequence_length, self.num_nodes, self.num_features = expected_shape


@dataclass
class FeatureStandardizer:
    """Per-feature statistics fitted on training graphs only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        store: GraphStore,
        indices: Iterable[int],
        detection_feature: int = 3,
    ) -> "FeatureStandardizer":
        total = None
        total_sq = None
        count = 0
        for index in indices:
            graph = store.graphs[int(index)].astype(np.float64, copy=False)
            flattened = graph.reshape(-1, graph.shape[-1])
            if total is None:
                total = np.zeros(flattened.shape[1], dtype=np.float64)
                total_sq = np.zeros(flattened.shape[1], dtype=np.float64)
            total += flattened.sum(axis=0)
            total_sq += np.square(flattened).sum(axis=0)
            count += flattened.shape[0]
        if count == 0 or total is None or total_sq is None:
            raise ValueError("Cannot fit normalization on an empty training fold")
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 1e-8)
        scale = np.sqrt(variance)
        # Detection is an interpretable 0/1 quality mask and should remain so.
        if 0 <= detection_feature < len(mean):
            mean[detection_feature] = 0.0
            scale[detection_feature] = 1.0
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, graph: np.ndarray) -> np.ndarray:
        return ((graph - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }


class GraphDataset(Dataset):
    """PyTorch view over selected indices in a shared :class:`GraphStore`."""

    def __init__(
        self,
        store: GraphStore,
        indices: Sequence[int],
        *,
        task: str,
        normalizer: FeatureStandardizer,
        training: bool = False,
        temporal_crop_min: float = 0.9,
        noise_std: float = 0.01,
    ):
        if task not in {"binary", "jar3"}:
            raise ValueError("task must be 'binary' or 'jar3'")
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.task = task
        self.normalizer = normalizer
        self.training = bool(training)
        self.temporal_crop_min = float(temporal_crop_min)
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, graph: torch.Tensor) -> torch.Tensor:
        # graph is [N,T,F]. Random cropping changes timing slightly without
        # ever mixing trials or subjects.
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
            detection = cropped[:, :, 3].unsqueeze(1)
            # interpolate expects [batch, channel, time]
            reshaped = cropped.permute(0, 2, 1)
            reshaped = torch_functional.interpolate(
                reshaped,
                size=graph.shape[1],
                mode="linear",
                align_corners=False,
            )
            graph = reshaped.permute(0, 2, 1)
            detection = torch_functional.interpolate(
                detection,
                size=graph.shape[1],
                mode="nearest",
            )
            graph[:, :, 3] = detection.squeeze(1)
        if self.noise_std > 0:
            noise = torch.randn_like(graph) * self.noise_std
            noise[:, :, 3] = 0.0
            graph = graph + noise
        return graph

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        raw = self.store.graphs[index]
        normalized = self.normalizer.transform(raw)
        graph = torch.from_numpy(normalized).permute(1, 0, 2).contiguous()
        if self.training:
            graph = self._augment(graph)
        adjacency = torch.from_numpy(self.store.adjacencies[index]).float()
        label = self.store.records[index].label_for(self.task)
        return graph, adjacency, torch.tensor(label, dtype=torch.long), index
