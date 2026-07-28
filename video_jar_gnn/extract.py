"""Extract normalized facial-region graph sequences directly from long videos.

Unlike the reference ``Video/b1_cut_exact_frames.py``, this extractor never
creates intermediate clips and never merges the five repeats of a sample. It
seeks to each manifest interval, samples by LSL time, and caches one compact
``.npz`` graph per trial.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .constants import AU_NODES, FEATURE_NAMES
from .expression import (
    EXPRESSION_FEATURES,
    EXPRESSION_NODES,
    IMPUTED_FEATURE_INDEX,
    OBSERVED_FEATURE_INDEX,
    REPRESENTATION as EXPRESSION_REPRESENTATION,
    build_expression_adjacency,
    build_expression_graph,
    expression_metadata,
)


EXTRA_MANIFEST_FIELDS = (
    "extract_status",
    "detection_ratio",
    "n_sampled_frames",
    "n_unique_frames",
    "sampled_duration_sec",
    "timing_source",
    "baseline_detection_ratio",
    "n_baseline_frames",
    "n_unique_baseline_frames",
    "baseline_duration_sec",
    "baseline_timing_source",
    "extract_error",
)


class CacheConfigurationMismatchError(RuntimeError):
    """A cache exists but was produced by a different extraction request."""


def _load_video_dependencies():
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Video extraction requires numpy, OpenCV and MediaPipe 0.10.21. "
            "Install video_jar_gnn/requirements.txt first."
        ) from exc
    return cv2, mp, np


def _load_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "This command requires numpy. Install "
            "video_jar_gnn/requirements.txt first."
        ) from exc
    return np


def _load_cv2_numpy():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video decoding requires OpenCV. Install "
            "video_jar_gnn/requirements.txt first."
        ) from exc
    return cv2, _load_numpy()


def _as_int(value: Any) -> int:
    return int(float(str(value).strip()))


def _as_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or ())
    required = {
        "sample_id",
        "subject_id",
        "ma_mau",
        "repeat",
        "jar",
        "start_frame",
        "end_frame",
        "video_path",
        "frame_label_path",
    }
    missing = required.difference(fields)
    if missing:
        raise RuntimeError(f"{path}: missing manifest columns {sorted(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted(
            sample_id
            for sample_id in set(sample_ids)
            if sample_ids.count(sample_id) > 1
        )
        raise RuntimeError(f"{path}: duplicate sample_id values: {duplicates}")
    return rows, fields


def write_graph_manifest(
    rows: Iterable[dict[str, Any]],
    original_fields: list[str],
    output: Path,
) -> None:
    fields = list(original_fields)
    for field in ("graph_path", *EXTRA_MANIFEST_FIELDS):
        if field not in fields:
            fields.append(field)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def preserve_unselected_manifest_rows(
    rows: list[dict[str, str]],
    *,
    selected_ids: set[str],
    output_manifest: Path,
) -> int:
    """Keep prior graph status when extraction is run in subject/batch mode."""
    if not output_manifest.is_file():
        return 0
    previous_rows, _ = read_manifest(output_manifest)
    previous_by_id = {row["sample_id"]: row for row in previous_rows}
    copied = 0
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in selected_ids or sample_id not in previous_by_id:
            continue
        previous = previous_by_id[sample_id]
        same_subject = str(row.get("subject_id", "")).strip() == str(
            previous.get("subject_id", "")
        ).strip()
        same_code = _as_int(row["ma_mau"]) == _as_int(previous["ma_mau"])
        same_repeat = _as_int(row["repeat"]) == _as_int(previous["repeat"])
        if not (same_subject and same_code and same_repeat):
            raise RuntimeError(
                f"{output_manifest}: prior row {sample_id} has a different "
                "subject_id/ma_mau/repeat key"
            )
        for field in ("graph_path", *EXTRA_MANIFEST_FIELDS):
            if field in previous:
                row[field] = previous[field]
        copied += 1
    return copied


def load_active_frame_times(
    path: Path,
) -> dict[tuple[int, int], list[tuple[int, float | None]]]:
    """Load only labelled frames, grouped by ``(ma_mau, repeat)``."""
    result: dict[tuple[int, int], list[tuple[int, float | None]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_idx", "t_lsl", "ma_mau", "lan_lap"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            code = _as_int(row["ma_mau"])
            if code == 0:
                continue
            repeat = _as_int(row["lan_lap"])
            result[(code, repeat)].append(
                (_as_int(row["frame_idx"]), _as_float(row["t_lsl"]))
            )
    return dict(result)


def load_active_and_preceding_frame_times(
    path: Path,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, float | None]]],
    dict[tuple[int, int], list[tuple[int, float | None]]],
]:
    """Load active trials and each trial's immediately preceding zero-code run.

    A neutral/background run is associated only with the active block that
    follows it. This prevents context from an earlier trial (or from another
    sample code) leaking into the requested pre-trial reference.
    """
    active: dict[
        tuple[int, int], list[tuple[int, float | None]]
    ] = defaultdict(list)
    preceding: dict[
        tuple[int, int], list[tuple[int, float | None]]
    ] = {}
    neutral_run: list[tuple[int, float | None]] = []
    active_key: tuple[int, int] | None = None

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_idx", "t_lsl", "ma_mau", "lan_lap"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            item = (_as_int(row["frame_idx"]), _as_float(row["t_lsl"]))
            code = _as_int(row["ma_mau"])
            if code == 0:
                if active_key is not None:
                    neutral_run = []
                neutral_run.append(item)
                active_key = None
                continue

            key = (code, _as_int(row["lan_lap"]))
            if key != active_key:
                if key in preceding:
                    raise RuntimeError(
                        f"{path}: active key={key} occurs in multiple blocks"
                    )
                preceding[key] = list(neutral_run)
                neutral_run = []
                active_key = key
            active[key].append(item)
    return dict(active), preceding


def _nearest_positions(times: list[float], targets) -> list[int]:
    positions: list[int] = []
    for target in targets:
        position = bisect.bisect_left(times, float(target))
        if position <= 0:
            positions.append(0)
        elif position >= len(times):
            positions.append(len(times) - 1)
        else:
            before, after = times[position - 1], times[position]
            positions.append(
                position - 1 if target - before <= after - target else position
            )
    return positions


def select_frames_by_time(
    frames: list[tuple[int, float | None]],
    *,
    num_frames: int,
    duration_mode: str,
    window_seconds: float,
):
    """Return fixed-length source frame indices and their sampling times."""
    np = _load_numpy()
    if not frames:
        raise RuntimeError("Empty labelled frame interval")
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")

    source_indices = [item[0] for item in frames]
    source_times = [item[1] for item in frames]
    have_times = all(value is not None for value in source_times)
    if have_times:
        times = [float(value) for value in source_times if value is not None]
        if any(right <= left for left, right in zip(times, times[1:])):
            have_times = False

    if have_times:
        start_time = times[0]
        labelled_end = times[-1]
        if duration_mode == "lsl":
            stop_time = min(labelled_end, start_time + window_seconds)
        elif duration_mode == "labelled":
            stop_time = labelled_end
        else:
            raise ValueError(f"Unknown duration_mode={duration_mode!r}")
        target_times = np.linspace(start_time, stop_time, num_frames, dtype=np.float64)
        chosen_positions = _nearest_positions(times, target_times)
        indices = [source_indices[pos] for pos in chosen_positions]
        actual_times = np.asarray(
            [times[pos] for pos in chosen_positions], dtype=np.float64
        )
        return indices, actual_times, target_times

    # Timing fallback retained for portability. The real study CSVs all have
    # valid, monotonic t_lsl, so this branch is not expected for production.
    stop_position = len(frames) - 1
    if duration_mode == "lsl":
        # The acquisition nominally runs at 60 fps.
        stop_position = min(stop_position, max(1, int(round(window_seconds * 60)) - 1))
    positions = np.rint(np.linspace(0, stop_position, num_frames)).astype(int)
    indices = [source_indices[int(pos)] for pos in positions]
    target_times = np.linspace(0.0, window_seconds, num_frames, dtype=np.float64)
    return indices, target_times.copy(), target_times


def select_preceding_frames_by_time(
    frames: list[tuple[int, float | None]],
    *,
    num_frames: int,
    window_seconds: float,
):
    """Resample the tail of a contiguous pre-trial background interval."""
    np = _load_numpy()
    if not frames:
        raise RuntimeError("No contiguous ma_mau=0 frames precede this trial")
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    source_indices = [item[0] for item in frames]
    source_times = [item[1] for item in frames]
    have_times = all(value is not None for value in source_times)
    if have_times:
        times = [float(value) for value in source_times if value is not None]
        if any(right <= left for left, right in zip(times, times[1:])):
            have_times = False

    if have_times:
        stop_time = times[-1]
        start_time = max(times[0], stop_time - window_seconds)
        target_times = np.linspace(start_time, stop_time, num_frames, dtype=np.float64)
        chosen_positions = _nearest_positions(times, target_times)
        indices = [source_indices[position] for position in chosen_positions]
        actual_times = np.asarray(
            [times[position] for position in chosen_positions], dtype=np.float64
        )
        return indices, actual_times, target_times

    nominal_count = max(2, int(round(window_seconds * 60.0)))
    start_position = max(0, len(frames) - nominal_count)
    positions = np.rint(
        np.linspace(start_position, len(frames) - 1, num_frames)
    ).astype(int)
    indices = [source_indices[int(position)] for position in positions]
    sampled_duration = min(
        window_seconds,
        max(1, len(frames) - 1 - start_position) / 60.0,
    )
    target_times = np.linspace(
        -sampled_duration, 0.0, num_frames, dtype=np.float64
    )
    return indices, target_times.copy(), target_times


def timing_source(frames: list[tuple[int, float | None]]) -> str:
    values = [value for _, value in frames]
    if all(value is not None for value in values):
        finite_values = [float(value) for value in values if value is not None]
        if all(
            right > left
            for left, right in zip(finite_values, finite_values[1:])
        ):
            return "lsl"
    return "nominal_60fps"


def sampling_fingerprint(indices, actual_times, target_times) -> str:
    """Digest the exact source frames/timing used to build a graph cache."""
    np = _load_numpy()
    digest = hashlib.sha256()
    digest.update(
        json.dumps([int(index) for index in indices], separators=(",", ":")).encode()
    )
    digest.update(np.asarray(actual_times, dtype="<f8").tobytes())
    digest.update(np.asarray(target_times, dtype="<f8").tobytes())
    return digest.hexdigest()


def build_adjacency():
    """Anatomical adjacency without self-loops (the model adds them once)."""
    np = _load_numpy()
    node_index = {name: index for index, name in enumerate(AU_NODES)}
    adjacency = np.zeros((len(AU_NODES), len(AU_NODES)), dtype=np.float32)
    edges = (
        ("brow_left_inner", "eye_left_upper"),
        ("brow_left_outer", "eye_left_upper"),
        ("brow_right_inner", "eye_right_upper"),
        ("brow_right_outer", "eye_right_upper"),
        ("brow_left_inner", "nose_bridge"),
        ("brow_right_inner", "nose_bridge"),
        ("eye_left_upper", "eye_left_lower"),
        ("eye_right_upper", "eye_right_lower"),
        ("nose_bridge", "eye_left_upper"),
        ("nose_bridge", "eye_right_upper"),
        ("nose_bridge", "upper_lip"),
        ("nose_alar_left", "upper_lip"),
        ("nose_alar_right", "upper_lip"),
        ("upper_lip", "lower_lip"),
        ("upper_lip", "lip_corners"),
        ("lower_lip", "lip_corners"),
        ("chin_center", "lower_lip"),
        ("brow_left_outer", "eye_left_lower"),
        ("brow_right_outer", "eye_right_lower"),
        ("nose_alar_left", "lip_corners"),
        ("nose_alar_right", "lip_corners"),
    )
    for left, right in edges:
        i, j = node_index[left], node_index[right]
        adjacency[i, j] = adjacency[j, i] = 1.0
    return adjacency


def _ids_from_edges(edges) -> list[int]:
    result: set[int] = set()
    for left, right in edges:
        result.add(int(left))
        result.add(int(right))
    return sorted(result)


def _split_near_far(ids: list[int], xy, anchor_index: int, np):
    if len(ids) < 2:
        return ids, ids
    distances = np.linalg.norm(xy[ids] - xy[anchor_index], axis=1)
    order = np.argsort(distances)
    midpoint = max(1, len(ids) // 2)
    near = [ids[int(index)] for index in order[:midpoint]]
    far = [ids[int(index)] for index in order[midpoint:]]
    return near, far or near


def _split_upper_lower(ids: list[int], xy, np):
    if len(ids) < 2:
        return ids, ids
    median_y = float(np.median(xy[ids, 1]))
    upper = [index for index in ids if xy[index, 1] <= median_y]
    lower = [index for index in ids if xy[index, 1] > median_y]
    return upper or ids, lower or ids


class FaceGraphExtractor:
    """MediaPipe face detector + FaceMesh to 15 normalized region nodes."""

    def __init__(
        self,
        *,
        resize: int = 256,
        detect_every: int = 5,
        min_detection_confidence: float = 0.5,
    ):
        cv2, mp, np = _load_video_dependencies()
        self.cv2, self.mp, self.np = cv2, mp, np
        self.resize = int(resize)
        self.detect_every = max(1, int(detect_every))
        self.min_detection_confidence = float(min_detection_confidence)
        self.detector = None
        self.mesh = None
        self._bbox = None
        self._last_base = None
        self._sample_index = 0

        face_mesh = mp.solutions.face_mesh
        self.left_brow = _ids_from_edges(face_mesh.FACEMESH_LEFT_EYEBROW)
        self.right_brow = _ids_from_edges(face_mesh.FACEMESH_RIGHT_EYEBROW)
        self.left_eye = _ids_from_edges(face_mesh.FACEMESH_LEFT_EYE)
        self.right_eye = _ids_from_edges(face_mesh.FACEMESH_RIGHT_EYE)
        self.lips = _ids_from_edges(face_mesh.FACEMESH_LIPS)
        self.nose = _ids_from_edges(face_mesh.FACEMESH_NOSE)

    def __enter__(self):
        self.detector = self.mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self.min_detection_confidence,
        )
        # FaceMesh is created per trial in reset_trial(). Its temporal tracker
        # must never carry state across a minutes-long seek or another person.
        self.mesh = None
        return self

    def _new_mesh(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
        self.mesh = self.mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        if self.mesh is not None:
            self.mesh.close()
        if self.detector is not None:
            self.detector.close()
        self.mesh = self.detector = None

    def reset_trial(self) -> None:
        self._new_mesh()
        self._bbox = None
        self._last_base = None
        self._sample_index = 0

    @staticmethod
    def _expand_square(x: float, y: float, w: float, h: float, width: int, height: int):
        center_x, center_y = x + w / 2.0, y + h / 2.0
        side = max(w, h) * 1.35
        x1 = max(0, int(round(center_x - side / 2.0)))
        y1 = max(0, int(round(center_y - side / 2.0)))
        x2 = min(width, int(round(center_x + side / 2.0)))
        y2 = min(height, int(round(center_y + side / 2.0)))
        return x1, y1, x2, y2

    def _detect_bbox(self, rgb):
        detection = self.detector.process(rgb)
        if not detection.detections:
            return None
        relative = detection.detections[0].location_data.relative_bounding_box
        height, width = rgb.shape[:2]
        return (
            relative.xmin * width,
            relative.ymin * height,
            relative.width * width,
            relative.height * height,
        )

    def _pool_region(self, landmarks, indices: list[int]):
        np, cv2 = self.np, self.cv2
        points = landmarks[indices]
        center = points[:, :3].mean(axis=0)
        xy = points[:, :2].astype(np.float32)
        if len(points) >= 3:
            hull = cv2.convexHull(xy.reshape(-1, 1, 2))
            area = float(cv2.contourArea(hull))
        else:
            area = 0.0
        width = float(xy[:, 0].max() - xy[:, 0].min())
        height = float(xy[:, 1].max() - xy[:, 1].min())
        # A log-ratio with a real floor fixes the 1e5 aspect outliers present
        # in the reference features while preserving opening/closing motion.
        aspect = float(np.clip(np.log((width + 1e-3) / (height + 1e-3)), -4.0, 4.0))
        return [center[0], center[1], center[2], 1.0, area, aspect]

    def _landmarks_to_nodes(self, mesh_landmarks):
        np = self.np
        if hasattr(mesh_landmarks, "landmark"):
            landmarks = np.asarray(
                [
                    (point.x, point.y, getattr(point, "z", 0.0))
                    for point in mesh_landmarks.landmark[:468]
                ],
                dtype=np.float32,
            )
        else:
            landmarks = np.asarray(mesh_landmarks, dtype=np.float32)
        if landmarks.shape != (468, 3):
            raise RuntimeError(
                f"FaceMesh returned invalid landmark shape {landmarks.shape}"
            )
        xy = landmarks[:, :2]

        # Normalize translation and scale using the two eye regions. This
        # removes most subject identity, face size and crop jitter.
        left_center = landmarks[self.left_eye].mean(axis=0)
        right_center = landmarks[self.right_eye].mean(axis=0)
        origin = (left_center + right_center) / 2.0
        eye_distance = float(np.linalg.norm(left_center[:2] - right_center[:2]))
        eye_distance = max(eye_distance, 1e-3)
        landmarks = (landmarks - origin) / eye_distance
        xy = landmarks[:, :2]

        left_brow_inner, left_brow_outer = _split_near_far(
            self.left_brow, xy, 1, np
        )
        right_brow_inner, right_brow_outer = _split_near_far(
            self.right_brow, xy, 1, np
        )
        left_eye_upper, left_eye_lower = _split_upper_lower(self.left_eye, xy, np)
        right_eye_upper, right_eye_lower = _split_upper_lower(self.right_eye, xy, np)
        upper_lip, lower_lip = _split_upper_lower(self.lips, xy, np)
        nose_x = float(xy[1, 0])
        nose_left = [index for index in self.nose if xy[index, 0] < nose_x]
        nose_right = [index for index in self.nose if xy[index, 0] >= nose_x]

        regions = {
            "brow_left_inner": left_brow_inner,
            "brow_left_outer": left_brow_outer,
            "brow_right_inner": right_brow_inner,
            "brow_right_outer": right_brow_outer,
            "eye_left_upper": left_eye_upper,
            "eye_left_lower": left_eye_lower,
            "eye_right_upper": right_eye_upper,
            "eye_right_lower": right_eye_lower,
            "nose_bridge": [6, 197, 195, 5, 4],
            "nose_alar_left": nose_left or self.nose,
            "nose_alar_right": nose_right or self.nose,
            "upper_lip": upper_lip,
            "lower_lip": lower_lip,
            "lip_corners": [61, 291],
            "chin_center": [152, 148, 176, 377, 400],
        }
        return np.asarray(
            [self._pool_region(landmarks, regions[name]) for name in AU_NODES],
            dtype=np.float32,
        )

    def _extract_landmark_array(self, frame):
        """Return raw ``[468,3]`` FaceMesh coordinates or ``None``."""
        cv2, np = self.cv2, self.np
        resized = cv2.resize(frame, (self.resize, self.resize))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self._bbox is None or self._sample_index % self.detect_every == 0:
            detected_bbox = self._detect_bbox(rgb)
            if detected_bbox is not None:
                if self._bbox is None:
                    self._bbox = detected_bbox
                else:
                    self._bbox = tuple(
                        0.7 * old + 0.3 * new
                        for old, new in zip(self._bbox, detected_bbox)
                    )

        mesh_input = rgb
        if self._bbox is not None:
            x, y, width, height = self._bbox
            x1, y1, x2, y2 = self._expand_square(
                x, y, width, height, self.resize, self.resize
            )
            crop = rgb[y1:y2, x1:x2]
            if crop.size:
                mesh_input = cv2.resize(crop, (self.resize, self.resize))

        result = self.mesh.process(mesh_input)
        self._sample_index += 1
        if result.multi_face_landmarks:
            return np.asarray(
                [
                    (point.x, point.y, getattr(point, "z", 0.0))
                    for point in result.multi_face_landmarks[0].landmark[:468]
                ],
                dtype=np.float32,
            )
        return None

    def extract_base(self, frame):
        """Return legacy base features ``[15,6]`` with missed-frame carry."""
        np = self.np
        landmarks = self._extract_landmark_array(frame)
        if landmarks is not None:
            base = self._landmarks_to_nodes(landmarks)
            self._last_base = base.copy()
            return base

        if self._last_base is not None:
            base = self._last_base.copy()
            base[:, 3] = 0.0
            return base
        return np.zeros((len(AU_NODES), 6), dtype=np.float32)

    def extract_landmarks(self, frame):
        """Return raw landmarks; keep misses as NaN for bounded imputation."""
        landmarks = self._extract_landmark_array(frame)
        if landmarks is not None:
            return landmarks
        return self.np.full((468, 3), self.np.nan, dtype=self.np.float32)


def _interpolate_missing(base, np):
    """Interpolate geometry across missing detections; retain detection mask."""
    result = base.copy()
    valid = base[:, 0, 3] > 0.5
    positions = np.arange(base.shape[0])
    valid_positions = positions[valid]
    if len(valid_positions) == 0:
        return result
    for node in range(base.shape[1]):
        for feature in (0, 1, 2, 4, 5):
            result[:, node, feature] = np.interp(
                positions,
                valid_positions,
                base[valid, node, feature],
            )
    return result


def add_temporal_features(base, target_times):
    """Convert ``[T,N,6]`` base geometry to the 10-feature graph contract."""
    np = _load_numpy()
    base = _interpolate_missing(base, np)
    temporal = np.zeros((base.shape[0], base.shape[1], 4), dtype=np.float32)
    if base.shape[0] > 1:
        times = np.asarray(target_times, dtype=np.float64)
        positive_steps = np.diff(times)
        valid_steps = positive_steps[positive_steps > 0]
        fallback = float(np.median(valid_steps)) if len(valid_steps) else 1.0
        dt = np.where(positive_steps > 1e-6, positive_steps, fallback)
        selected = base[:, :, [0, 1, 4, 5]]
        temporal[1:] = (selected[1:] - selected[:-1]) / dt[:, None, None]
        temporal = np.clip(temporal, -20.0, 20.0)
    return np.concatenate((base, temporal), axis=2).astype(np.float32)


def decode_selected_frames(
    cap,
    indices: list[int],
    extractor: FaceGraphExtractor,
    *,
    representation: str = "legacy",
):
    """Decode one interval sequentially and return bases in requested order."""
    cv2, np = _load_cv2_numpy()
    unique_indices = sorted(set(int(index) for index in indices))
    if not unique_indices:
        raise RuntimeError("No source frames selected")
    requested_start = unique_indices[0]
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, requested_start):
        raise RuntimeError(f"Video backend failed to seek to frame {requested_start}")
    reported_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    if not math.isfinite(reported_position):
        raise RuntimeError("Video backend did not report its position after seek")
    landed = int(round(reported_position))
    if landed > requested_start:
        raise RuntimeError(
            f"Video seek overshot: requested {requested_start}, landed at {landed}"
        )
    if requested_start - landed > 1000:
        raise RuntimeError(
            f"Video seek landed implausibly far away: requested {requested_start}, "
            f"landed at {landed}"
        )
    wanted = set(unique_indices)
    decoded: dict[int, Any] = {}
    current = landed
    last = unique_indices[-1]
    while current <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            if representation == EXPRESSION_REPRESENTATION:
                decoded[current] = extractor.extract_landmarks(frame)
            else:
                decoded[current] = extractor.extract_base(frame)
        current += 1
    missing = sorted(wanted.difference(decoded))
    if missing:
        raise RuntimeError(
            f"Could not decode {len(missing)} selected frames; "
            f"first missing frame={missing[0]}"
        )
    return np.stack([decoded[int(index)] for index in indices], axis=0)


def unique_source_observations(sequence, indices, actual_times):
    """Collapse repeated nearest-frame selections before motion estimation."""
    np = _load_numpy()
    frames = np.asarray(indices, dtype=np.int64)
    times = np.asarray(actual_times, dtype=np.float64)
    values = np.asarray(sequence)
    if (
        frames.ndim != 1
        or times.ndim != 1
        or len(frames) != len(times)
        or len(values) != len(frames)
    ):
        raise ValueError("source observations have inconsistent lengths")
    keep: list[int] = []
    last_frame: int | None = None
    last_time = -float("inf")
    for position, (frame, timestamp) in enumerate(zip(frames, times)):
        if int(frame) == last_frame:
            continue
        if not math.isfinite(float(timestamp)) or float(timestamp) <= last_time:
            continue
        keep.append(position)
        last_frame = int(frame)
        last_time = float(timestamp)
    if len(keep) < 2:
        raise RuntimeError(
            "Fewer than two unique, strictly timed source frames were selected"
        )
    return values[keep], times[keep]


def _load_cached_info(path: Path) -> dict[str, Any]:
    np = _load_numpy()
    with np.load(path, allow_pickle=False) as data:
        required = {
            "graph_seq",
            "adj",
            "sampled_frame_idx",
            "sampled_lsl",
            "target_lsl",
        }
        missing = required.difference(data.files)
        if missing:
            raise RuntimeError(f"{path}: cached graph is missing {sorted(missing)}")
        graph = np.asarray(data["graph_seq"])
        adjacency = np.asarray(data["adj"])
        sampled_indices = np.asarray(data["sampled_frame_idx"])
        sampled_lsl = np.asarray(data["sampled_lsl"])
        target_lsl = np.asarray(data["target_lsl"])
        metadata: dict[str, Any] = {}
        if "meta" in data:
            raw_meta = data["meta"]
            if getattr(raw_meta, "shape", None) == ():
                raw_meta = raw_meta.item()
            if isinstance(raw_meta, bytes):
                raw_meta = raw_meta.decode("utf-8")
            if isinstance(raw_meta, str):
                metadata = json.loads(raw_meta)
        representation = metadata.get("representation", "legacy")
        if representation == EXPRESSION_REPRESENTATION:
            node_names = tuple(metadata.get("node_names", ()))
            feature_names = tuple(metadata.get("feature_names", ()))
            if node_names != EXPRESSION_NODES:
                raise RuntimeError(
                    f"{path}: invalid expression_v2 node_names contract"
                )
            if feature_names != EXPRESSION_FEATURES:
                raise RuntimeError(
                    f"{path}: invalid expression_v2 feature_names contract"
                )
            expected_nodes = len(EXPRESSION_NODES)
            expected_features = len(EXPRESSION_FEATURES)
            detection_index = int(
                metadata.get("observed_feature_index", OBSERVED_FEATURE_INDEX)
            )
        elif representation in ("legacy", None):
            expected_nodes = len(AU_NODES)
            expected_features = len(FEATURE_NAMES)
            detection_index = 3
        else:
            raise RuntimeError(
                f"{path}: unsupported cached representation {representation!r}"
            )
        if graph.ndim != 3 or graph.shape[1:] != (
            expected_nodes,
            expected_features,
        ):
            raise RuntimeError(f"{path}: invalid graph_seq shape {graph.shape}")
        if adjacency.shape != (expected_nodes, expected_nodes):
            raise RuntimeError(f"{path}: invalid adjacency shape {adjacency.shape}")
        if not np.allclose(adjacency, adjacency.T):
            raise RuntimeError(f"{path}: cached adjacency is not symmetric")
        if any(
            array.ndim != 1
            for array in (sampled_indices, sampled_lsl, target_lsl)
        ):
            raise RuntimeError(f"{path}: cached sampling arrays must be one-dimensional")
        if any(
            len(array) != graph.shape[0]
            for array in (sampled_indices, sampled_lsl, target_lsl)
        ):
            raise RuntimeError(f"{path}: cached sampling arrays have wrong length")
        if not all(
            np.isfinite(array).all()
            for array in (graph, adjacency, sampled_lsl, target_lsl)
        ):
            raise RuntimeError(f"{path}: cached graph contains NaN/Inf")
        if np.any(np.diff(sampled_indices) < 0) or np.any(np.diff(sampled_lsl) < 0):
            raise RuntimeError(f"{path}: cached source sampling is not monotonic")
        if np.any(np.diff(target_lsl) <= 0):
            raise RuntimeError(f"{path}: cached target_lsl is not strictly increasing")
        if "detection_ratio" in data:
            detection_ratio = float(data["detection_ratio"])
        else:
            detection_ratio = float(
                (graph[:, 0, detection_index] > 0.5).mean()
            )
        if not math.isfinite(detection_ratio) or not 0.0 <= detection_ratio <= 1.0:
            raise RuntimeError(
                f"{path}: invalid cached detection_ratio={detection_ratio!r}"
            )
        cached_fingerprint = sampling_fingerprint(
            sampled_indices, sampled_lsl, target_lsl
        )
        metadata_fingerprint = metadata.get("sampling_fingerprint")
        if (
            metadata_fingerprint is not None
            and metadata_fingerprint != cached_fingerprint
        ):
            raise RuntimeError(
                f"{path}: cached sampling arrays do not match their fingerprint"
            )
        baseline_info: dict[str, Any] | None = None
        baseline_array_names = {
            "baseline_seq",
            "baseline_sampled_frame_idx",
            "baseline_sampled_lsl",
            "baseline_target_lsl",
        }
        expects_baseline = float(metadata.get("pre_context_seconds", 0.0) or 0.0) > 0
        has_any_baseline = bool(baseline_array_names.intersection(data.files))
        if expects_baseline or has_any_baseline:
            missing_baseline = baseline_array_names.difference(data.files)
            if missing_baseline:
                raise RuntimeError(
                    f"{path}: cached neutral baseline is missing "
                    f"{sorted(missing_baseline)}"
                )
            baseline = np.asarray(data["baseline_seq"])
            baseline_indices = np.asarray(data["baseline_sampled_frame_idx"])
            baseline_lsl = np.asarray(data["baseline_sampled_lsl"])
            baseline_target = np.asarray(data["baseline_target_lsl"])
            if baseline.ndim != 3 or baseline.shape[1:] != (
                expected_nodes,
                expected_features,
            ):
                raise RuntimeError(
                    f"{path}: invalid baseline_seq shape {baseline.shape}"
                )
            if any(
                array.ndim != 1
                for array in (
                    baseline_indices,
                    baseline_lsl,
                    baseline_target,
                )
            ):
                raise RuntimeError(
                    f"{path}: cached baseline sampling arrays must be "
                    "one-dimensional"
                )
            if any(
                len(array) != baseline.shape[0]
                for array in (
                    baseline_indices,
                    baseline_lsl,
                    baseline_target,
                )
            ):
                raise RuntimeError(
                    f"{path}: cached baseline sampling arrays have wrong length"
                )
            if not all(
                np.isfinite(array).all()
                for array in (
                    baseline,
                    baseline_lsl,
                    baseline_target,
                )
            ):
                raise RuntimeError(
                    f"{path}: cached neutral baseline contains NaN/Inf"
                )
            if (
                np.any(np.diff(baseline_indices) < 0)
                or np.any(np.diff(baseline_lsl) < 0)
                or np.any(np.diff(baseline_target) <= 0)
            ):
                raise RuntimeError(
                    f"{path}: cached neutral baseline sampling is not monotonic"
                )
            baseline_fingerprint = sampling_fingerprint(
                baseline_indices, baseline_lsl, baseline_target
            )
            metadata_baseline_fingerprint = metadata.get(
                "baseline_sampling_fingerprint"
            )
            if (
                metadata_baseline_fingerprint is not None
                and metadata_baseline_fingerprint != baseline_fingerprint
            ):
                raise RuntimeError(
                    f"{path}: cached baseline sampling arrays do not match "
                    "their fingerprint"
                )
            if "baseline_detection_ratio" in data:
                baseline_detection_ratio = float(
                    data["baseline_detection_ratio"]
                )
            else:
                baseline_detection_ratio = float(
                    (baseline[:, 0, detection_index] > 0.5).mean()
                )
            if (
                not math.isfinite(baseline_detection_ratio)
                or not 0.0 <= baseline_detection_ratio <= 1.0
            ):
                raise RuntimeError(
                    f"{path}: invalid cached baseline_detection_ratio="
                    f"{baseline_detection_ratio!r}"
                )
            baseline_info = {
                "detection_ratio": baseline_detection_ratio,
                "num_frames": int(baseline.shape[0]),
                "sampled_duration_sec": (
                    float(baseline_target[-1] - baseline_target[0])
                    if len(baseline_target) >= 2
                    else None
                ),
                "n_unique_frames": int(len(np.unique(baseline_indices))),
            }
        sampled_duration = (
            float(target_lsl[-1] - target_lsl[0])
            if len(target_lsl) >= 2
            else metadata.get("sampled_duration_sec")
        )
        return {
            "detection_ratio": detection_ratio,
            "num_frames": int(graph.shape[0]),
            "sampled_duration_sec": sampled_duration,
            "n_unique_frames": int(len(np.unique(sampled_indices))),
            "metadata": metadata,
            "baseline": baseline_info,
        }


def cached_configuration_mismatches(
    cached: dict[str, Any],
    *,
    expected_metadata: dict[str, Any],
    expected_num_frames: int,
    expected_baseline_frames: int | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Return every cache/command mismatch; empty means reuse is safe."""
    mismatches: dict[str, tuple[Any, Any]] = {}
    if int(cached["num_frames"]) != int(expected_num_frames):
        mismatches["num_frames"] = (
            int(cached["num_frames"]),
            int(expected_num_frames),
        )
    metadata = cached.get("metadata") or {}
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            mismatches[key] = (metadata.get(key), expected_value)
    if expected_baseline_frames is not None:
        baseline = cached.get("baseline")
        actual_baseline_frames = (
            int(baseline["num_frames"]) if baseline is not None else None
        )
        if actual_baseline_frames != int(expected_baseline_frames):
            mismatches["baseline_frames"] = (
                actual_baseline_frames,
                int(expected_baseline_frames),
            )
    return mismatches


def _atomic_save_npz(path: Path, np, **arrays: Any) -> None:
    """Write a cache completely before atomically replacing its final path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_manifest(
    manifest_path: Path,
    output_dir: Path,
    output_manifest: Path,
    *,
    representation: str = "legacy",
    num_frames: int = 96,
    duration_mode: str = "lsl",
    window_seconds: float = 10.0,
    resize: int = 256,
    detect_every: int = 5,
    min_detection_ratio: float = 0.5,
    pre_context_seconds: float = 0.0,
    baseline_frames: int = 60,
    trial_baseline_seconds: float = 1.25,
    max_impute_gap_sec: float = 0.5,
    subjects: set[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    if representation not in ("legacy", EXPRESSION_REPRESENTATION):
        raise ValueError(
            "representation must be 'legacy' or 'expression_v2'"
        )
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if resize < 32:
        raise ValueError("resize must be at least 32 pixels")
    if detect_every < 1:
        raise ValueError("detect_every must be positive")
    if not 0.0 <= min_detection_ratio <= 1.0:
        raise ValueError("min_detection_ratio must be in [0,1]")
    if pre_context_seconds < 0:
        raise ValueError("pre_context_seconds must be non-negative")
    if baseline_frames < 2:
        raise ValueError("baseline_frames must be at least 2")
    if trial_baseline_seconds <= 0:
        raise ValueError("trial_baseline_seconds must be positive")
    if max_impute_gap_sec < 0:
        raise ValueError("max_impute_gap_sec must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    cv2, _, np = _load_video_dependencies()
    rows, fields = read_manifest(manifest_path)
    selected = [
        row
        for row in rows
        if subjects is None or row["subject_id"] in subjects
    ]
    if limit is not None:
        selected = selected[: max(0, limit)]
    selected_ids = {row["sample_id"] for row in selected}
    preserved = preserve_unselected_manifest_rows(
        rows,
        selected_ids=selected_ids,
        output_manifest=output_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    adjacency = (
        build_expression_adjacency()
        if representation == EXPRESSION_REPRESENTATION
        else build_adjacency()
    )
    counts: defaultdict[str, int] = defaultdict(int)
    if preserved:
        counts["preserved_unselected"] = preserved

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[
            (row["subject_id"], row["video_path"], row["frame_label_path"])
        ].append(row)

    with FaceGraphExtractor(resize=resize, detect_every=detect_every) as extractor:
        for (subject_id, video_text, labels_text), subject_rows in sorted(grouped.items()):
            video_path = Path(video_text).resolve()
            labels_path = Path(labels_text).resolve()
            video_stat = video_path.stat()
            labels_stat = labels_path.stat()
            if pre_context_seconds > 0:
                frame_times, preceding_frame_times = (
                    load_active_and_preceding_frame_times(labels_path)
                )
            else:
                frame_times = load_active_frame_times(labels_path)
                preceding_frame_times = {}
            cap = cv2.VideoCapture(video_text)
            if not cap.isOpened():
                for row in subject_rows:
                    row["extract_status"] = "error"
                    row["extract_error"] = f"Cannot open video: {video_text}"
                    counts["error"] += 1
                continue
            subject_rows.sort(key=lambda row: _as_int(row["start_frame"]))

            for row in subject_rows:
                graph_path = (output_dir / subject_id / f"{row['sample_id']}.npz").resolve()
                row["graph_path"] = str(graph_path)
                row["extract_error"] = ""
                try:
                    key = (_as_int(row["ma_mau"]), _as_int(row["repeat"]))
                    if key not in frame_times:
                        raise RuntimeError(f"No frame interval for key={key}")
                    source_frames = frame_times[key]
                    timing = timing_source(source_frames)
                    indices, actual_times, target_times = select_frames_by_time(
                        source_frames,
                        num_frames=num_frames,
                        duration_mode=duration_mode,
                        window_seconds=window_seconds,
                    )
                    sampled_duration = float(target_times[-1] - target_times[0])
                    n_unique_frames = len(set(indices))
                    if n_unique_frames < num_frames:
                        counts["duplicate_sampling"] += 1
                        print(
                            f"[WARN] {row['sample_id']}: {n_unique_frames}/"
                            f"{num_frames} unique source frames after LSL resampling",
                            file=sys.stderr,
                        )
                    sample_digest = sampling_fingerprint(
                        indices, actual_times, target_times
                    )
                    baseline_indices = baseline_actual_times = None
                    baseline_target_times = baseline_source_frames = None
                    baseline_sampled_duration = None
                    baseline_n_unique_frames = None
                    baseline_timing = None
                    baseline_digest = None
                    if pre_context_seconds > 0:
                        baseline_source_frames = preceding_frame_times.get(key)
                        if not baseline_source_frames:
                            raise RuntimeError(
                                f"No contiguous ma_mau=0 context precedes key={key}"
                            )
                        (
                            baseline_indices,
                            baseline_actual_times,
                            baseline_target_times,
                        ) = select_preceding_frames_by_time(
                            baseline_source_frames,
                            num_frames=baseline_frames,
                            window_seconds=pre_context_seconds,
                        )
                        baseline_sampled_duration = float(
                            baseline_target_times[-1]
                            - baseline_target_times[0]
                        )
                        baseline_n_unique_frames = len(set(baseline_indices))
                        if baseline_n_unique_frames < baseline_frames:
                            counts["duplicate_baseline_sampling"] += 1
                            print(
                                f"[WARN] {row['sample_id']}: "
                                f"{baseline_n_unique_frames}/{baseline_frames} "
                                "unique neutral-context frames after LSL "
                                "resampling",
                                file=sys.stderr,
                            )
                        baseline_timing = timing_source(baseline_source_frames)
                        baseline_digest = sampling_fingerprint(
                            baseline_indices,
                            baseline_actual_times,
                            baseline_target_times,
                        )
                    expected_config = {
                        "sample_id": row["sample_id"],
                        "subject_id": row["subject_id"],
                        "ma_mau": key[0],
                        "repeat": key[1],
                        "duration_mode": duration_mode,
                        "sampled_duration_sec": sampled_duration,
                        "n_unique_frames": n_unique_frames,
                        "window_seconds": float(window_seconds),
                        "resize": int(resize),
                        "detect_every": int(detect_every),
                        "source_video": str(video_path),
                        "source_video_size": int(video_stat.st_size),
                        "source_video_mtime_ns": int(video_stat.st_mtime_ns),
                        "frame_label_path": str(labels_path),
                        "frame_label_size": int(labels_stat.st_size),
                        "frame_label_mtime_ns": int(labels_stat.st_mtime_ns),
                        "start_frame": _as_int(row["start_frame"]),
                        "end_frame": _as_int(row["end_frame"]),
                        "timing_source": timing,
                        "sampling_fingerprint": sample_digest,
                    }
                    if representation == EXPRESSION_REPRESENTATION:
                        expected_config.update(
                            {
                                "extractor_version": 4,
                                **expression_metadata(
                                    trial_baseline_seconds=(
                                        trial_baseline_seconds
                                    ),
                                    max_impute_gap_sec=max_impute_gap_sec,
                                ),
                            }
                        )
                    else:
                        expected_config.update(
                            {
                                "extractor_version": 1,
                                "feature_names": list(FEATURE_NAMES),
                                "au_nodes": list(AU_NODES),
                            }
                        )
                    if pre_context_seconds > 0:
                        expected_config.update(
                            {
                                "pre_context_seconds": float(
                                    pre_context_seconds
                                ),
                                "baseline_frames": int(baseline_frames),
                                "baseline_sampled_duration_sec": (
                                    baseline_sampled_duration
                                ),
                                "baseline_n_unique_frames": (
                                    baseline_n_unique_frames
                                ),
                                "baseline_timing_source": baseline_timing,
                                "baseline_sampling_fingerprint": baseline_digest,
                                "extractor_version": (
                                    4
                                    if representation
                                    == EXPRESSION_REPRESENTATION
                                    else 2
                                ),
                            }
                        )
                    if graph_path.exists() and not overwrite:
                        cached = _load_cached_info(graph_path)
                        mismatches = cached_configuration_mismatches(
                            cached,
                            expected_metadata=expected_config,
                            expected_num_frames=num_frames,
                            expected_baseline_frames=(
                                baseline_frames
                                if pre_context_seconds > 0
                                else None
                            ),
                        )
                        if mismatches:
                            raise CacheConfigurationMismatchError(
                                f"Cached graph configuration differs: {mismatches}. "
                                "Use --overwrite to extract it again."
                            )
                        detection_ratio = cached["detection_ratio"]
                        status = (
                            "cached"
                            if detection_ratio >= min_detection_ratio
                            else "low_quality"
                        )
                        row.update(
                            {
                                "extract_status": status,
                                "detection_ratio": f"{detection_ratio:.6f}",
                                "n_sampled_frames": cached["num_frames"],
                                "n_unique_frames": cached["n_unique_frames"],
                                "sampled_duration_sec": (
                                    f"{float(cached['sampled_duration_sec']):.6f}"
                                    if cached["sampled_duration_sec"] is not None
                                    else ""
                                ),
                                "timing_source": timing,
                            }
                        )
                        if pre_context_seconds > 0:
                            cached_baseline = cached["baseline"]
                            row.update(
                                {
                                    "baseline_detection_ratio": (
                                        f"{cached_baseline['detection_ratio']:.6f}"
                                    ),
                                    "n_baseline_frames": cached_baseline[
                                        "num_frames"
                                    ],
                                    "n_unique_baseline_frames": (
                                        cached_baseline["n_unique_frames"]
                                    ),
                                    "baseline_duration_sec": (
                                        f"{float(cached_baseline['sampled_duration_sec']):.6f}"
                                        if cached_baseline[
                                            "sampled_duration_sec"
                                        ]
                                        is not None
                                        else ""
                                    ),
                                    "baseline_timing_source": baseline_timing,
                                }
                            )
                        counts[status] += 1
                        continue

                    extractor.reset_trial()
                    baseline_graph = None
                    baseline_pose = None
                    baseline_detection_ratio = None
                    if pre_context_seconds > 0:
                        baseline_base = decode_selected_frames(
                            cap,
                            baseline_indices,
                            extractor,
                            representation=representation,
                        )
                        if representation == EXPRESSION_REPRESENTATION:
                            baseline_source_base, baseline_source_times = (
                                unique_source_observations(
                                    baseline_base,
                                    baseline_indices,
                                    baseline_actual_times,
                                )
                            )
                            baseline_graph, baseline_pose = (
                                build_expression_graph(
                                    baseline_source_base,
                                    baseline_source_times,
                                    output_times=baseline_target_times,
                                    trial_baseline_seconds=(
                                        trial_baseline_seconds
                                    ),
                                    max_impute_gap_sec=max_impute_gap_sec,
                                )
                            )
                            baseline_detection_ratio = float(
                                np.isfinite(baseline_base).all(axis=(1, 2)).mean()
                            )
                        else:
                            baseline_graph = add_temporal_features(
                                baseline_base, baseline_target_times
                            )
                            baseline_detection_ratio = float(
                                (baseline_base[:, 0, 3] > 0.5).mean()
                            )
                    base = decode_selected_frames(
                        cap,
                        indices,
                        extractor,
                        representation=representation,
                    )
                    pose = None
                    if representation == EXPRESSION_REPRESENTATION:
                        source_base, source_times = (
                            unique_source_observations(
                                base, indices, actual_times
                            )
                        )
                        graph, pose = build_expression_graph(
                            source_base,
                            source_times,
                            output_times=target_times,
                            trial_baseline_seconds=trial_baseline_seconds,
                            max_impute_gap_sec=max_impute_gap_sec,
                        )
                        detection_ratio = float(
                            np.isfinite(base).all(axis=(1, 2)).mean()
                        )
                    else:
                        graph = add_temporal_features(base, target_times)
                        detection_ratio = float(
                            (base[:, 0, 3] > 0.5).mean()
                        )
                    metadata = {
                        **expected_config,
                        "jar": _as_int(row["jar"]),
                    }
                    graph_path.parent.mkdir(parents=True, exist_ok=True)
                    arrays: dict[str, Any] = {
                        "graph_seq": graph,
                        "adj": adjacency,
                        "sampled_frame_idx": np.asarray(
                            indices, dtype=np.int64
                        ),
                        "sampled_lsl": np.asarray(
                            actual_times, dtype=np.float64
                        ),
                        "target_lsl": np.asarray(
                            target_times, dtype=np.float64
                        ),
                        "detection_ratio": np.asarray(
                            detection_ratio, dtype=np.float32
                        ),
                        "meta": np.asarray(
                            json.dumps(metadata, ensure_ascii=False)
                        ),
                    }
                    if pose is not None:
                        arrays["pose_seq"] = pose
                    if baseline_graph is not None:
                        arrays.update(
                            {
                                "baseline_seq": baseline_graph,
                                "baseline_sampled_frame_idx": np.asarray(
                                    baseline_indices, dtype=np.int64
                                ),
                                "baseline_sampled_lsl": np.asarray(
                                    baseline_actual_times, dtype=np.float64
                                ),
                                "baseline_target_lsl": np.asarray(
                                    baseline_target_times, dtype=np.float64
                                ),
                                "baseline_detection_ratio": np.asarray(
                                    baseline_detection_ratio,
                                    dtype=np.float32,
                                ),
                            }
                        )
                        if baseline_pose is not None:
                            arrays["baseline_pose_seq"] = baseline_pose
                    _atomic_save_npz(graph_path, np, **arrays)
                    status = (
                        "ok"
                        if detection_ratio >= min_detection_ratio
                        else "low_quality"
                    )
                    row.update(
                        {
                            "extract_status": status,
                            "detection_ratio": f"{detection_ratio:.6f}",
                            "n_sampled_frames": graph.shape[0],
                            "n_unique_frames": n_unique_frames,
                            "sampled_duration_sec": f"{sampled_duration:.6f}",
                            "timing_source": timing,
                        }
                    )
                    if baseline_graph is not None:
                        row.update(
                            {
                                "baseline_detection_ratio": (
                                    f"{baseline_detection_ratio:.6f}"
                                ),
                                "n_baseline_frames": baseline_graph.shape[0],
                                "n_unique_baseline_frames": (
                                    baseline_n_unique_frames
                                ),
                                "baseline_duration_sec": (
                                    f"{baseline_sampled_duration:.6f}"
                                ),
                                "baseline_timing_source": baseline_timing,
                            }
                        )
                    counts[status] += 1
                except CacheConfigurationMismatchError:
                    # Do not replace a usable graph manifest with rows marked
                    # error merely because the user requested a new cache
                    # configuration without --overwrite.
                    cap.release()
                    raise
                except Exception as exc:  # preserve all failures in the manifest
                    row["extract_status"] = "error"
                    row["extract_error"] = f"{type(exc).__name__}: {exc}"
                    counts["error"] += 1
                    print(
                        f"[ERROR] {row['sample_id']}: {row['extract_error']}",
                        file=sys.stderr,
                    )
            cap.release()

    for row in rows:
        if row["sample_id"] not in selected_ids and not row.get("extract_status"):
            row["extract_status"] = "not_selected"
            counts["not_selected"] += 1
    write_graph_manifest(rows, fields, output_manifest)
    return dict(counts)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe facial graphs for all manifest trials."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/video_jar_gnn/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Cache directory. Defaults to graphs for legacy and "
            "graphs_expression_v2 for expression_v2."
        ),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help=(
            "Output CSV. Defaults to graph_manifest.csv for legacy and "
            "graph_manifest_expression_v2.csv for expression_v2."
        ),
    )
    parser.add_argument(
        "--representation",
        choices=("legacy", EXPRESSION_REPRESENTATION),
        default="legacy",
        help=(
            "legacy keeps the old [T,15,10] cache. expression_v2 creates a "
            "separate head-pose robust [T,20,8] expression-proxy cache."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        help=(
            "Samples per trial. Default: 96 for legacy; 600 for "
            "expression_v2 (60 Hz over 10 seconds)."
        ),
    )
    parser.add_argument(
        "--duration-mode",
        choices=("lsl", "labelled"),
        default="lsl",
        help=(
            "'lsl': resample the first --window-seconds of real time; "
            "'labelled': resample across the entire 600-frame labelled interval."
        ),
    )
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--detect-every", type=int, default=5)
    parser.add_argument("--min-detection-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pre-context-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional duration of the contiguous ma_mau=0 interval immediately "
            "before each trial to cache as baseline_seq. Disabled by default."
        ),
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=60,
        help=(
            "Number of frames resampled from the pre-trial context when "
            "--pre-context-seconds is positive."
        ),
    )
    parser.add_argument(
        "--trial-baseline-seconds",
        type=float,
        default=1.25,
        help=(
            "Within-trial baseline duration used by expression_v2 delta and "
            "robust-z channels."
        ),
    )
    parser.add_argument(
        "--max-impute-gap-sec",
        type=float,
        default=0.5,
        help=(
            "Maximum internal missed-face gap interpolated by expression_v2."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="Optional Pxxx subset, e.g. --subjects P001 P002.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output_dir = args.output_dir
    output_manifest = args.output_manifest
    num_frames = args.num_frames
    if args.representation == EXPRESSION_REPRESENTATION:
        output_dir = output_dir or Path(
            "output/video_jar_gnn/graphs_expression_v2"
        )
        output_manifest = output_manifest or Path(
            "output/video_jar_gnn/graph_manifest_expression_v2.csv"
        )
        num_frames = 600 if num_frames is None else num_frames
    else:
        output_dir = output_dir or Path("output/video_jar_gnn/graphs")
        output_manifest = output_manifest or Path(
            "output/video_jar_gnn/graph_manifest.csv"
        )
        num_frames = 96 if num_frames is None else num_frames
    try:
        counts = extract_manifest(
            args.manifest,
            output_dir,
            output_manifest,
            representation=args.representation,
            num_frames=num_frames,
            duration_mode=args.duration_mode,
            window_seconds=args.window_seconds,
            resize=args.resize,
            detect_every=args.detect_every,
            min_detection_ratio=args.min_detection_ratio,
            pre_context_seconds=args.pre_context_seconds,
            baseline_frames=args.baseline_frames,
            trial_baseline_seconds=args.trial_baseline_seconds,
            max_impute_gap_sec=args.max_impute_gap_sec,
            subjects=set(args.subjects) if args.subjects else None,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"Extraction failed: {exc}") from exc
    print(f"Extraction summary: {counts}")
    print(f"Graph manifest: {output_manifest}")
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
