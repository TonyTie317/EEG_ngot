"""Head-pose robust facial-expression proxy graphs.

The values in this module are geometric expression proxies derived from
MediaPipe FaceMesh landmarks.  They are deliberately not described as
validated FACS Action Unit scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np


REPRESENTATION = "expression_v2"
REPRESENTATION_VERSION = 1

EXPRESSION_NODES = (
    "inner_brow_R",
    "inner_brow_L",
    "outer_brow_R",
    "outer_brow_L",
    "eye_open_R",
    "eye_open_L",
    "lid_cheek_gap_R",
    "lid_cheek_gap_L",
    "alar_extent_R",
    "alar_extent_L",
    "nose_upperlip_R",
    "nose_upperlip_L",
    "corner_horizontal_R",
    "corner_horizontal_L",
    "corner_vertical_R",
    "corner_vertical_L",
    "inner_lip_aperture",
    "outer_mouth_open",
    "lowerlip_chin_gap",
    "jaw_open",
)

EXPRESSION_FEATURES = (
    "value",
    "delta",
    "velocity",
    "acceleration",
    "abs_velocity",
    "baseline_robust_z",
    "observed",
    "imputed",
)

OBSERVED_FEATURE_INDEX = 6
IMPUTED_FEATURE_INDEX = 7
MASK_FEATURE_INDICES = (OBSERVED_FEATURE_INDEX, IMPUTED_FEATURE_INDEX)
STATIC_FEATURE_INDICES = (0, 1, 5)
VELOCITY_FEATURE_INDICES = (2, 3, 4)

POSE_ANCHORS = (33, 133, 362, 263, 168, 6)
MIN_REQUIRED_LANDMARKS = 468


def expression_metadata(
    *,
    trial_baseline_seconds: float = 1.25,
    max_impute_gap_sec: float = 0.5,
) -> dict[str, Any]:
    """Return the representation-specific portion of cache metadata."""
    return {
        "representation": REPRESENTATION,
        "representation_version": REPRESENTATION_VERSION,
        "node_names": list(EXPRESSION_NODES),
        "feature_names": list(EXPRESSION_FEATURES),
        "observed_mask_indices": list(MASK_FEATURE_INDICES),
        "static_feature_indices": list(STATIC_FEATURE_INDICES),
        "velocity_feature_indices": list(VELOCITY_FEATURE_INDICES),
        # Singular aliases make the semantics explicit to non-trainer tools.
        "observed_feature_index": OBSERVED_FEATURE_INDEX,
        "imputed_feature_index": IMPUTED_FEATURE_INDEX,
        "pose_normalization": "trial_3d_rigid_procrustes_v1",
        "temporal_resampling": (
            "unique_sampled_lsl_to_target_lsl_linear_v1"
        ),
        "temporal_filter": "local_quadratic_300ms_v1",
        "trial_baseline_seconds": float(trial_baseline_seconds),
        "max_impute_gap_sec": float(max_impute_gap_sec),
        "robust_z_scale_floor": 0.01,
        "robust_z_clip": 6.0,
    }


def build_expression_adjacency() -> np.ndarray:
    """Return a connected anatomical graph without self-loops."""
    adjacency = np.zeros(
        (len(EXPRESSION_NODES), len(EXPRESSION_NODES)), dtype=np.float32
    )
    edges = (
        # Bilateral counterparts.
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        # Right face.
        (0, 2),
        (0, 4),
        (2, 4),
        (4, 6),
        (6, 8),
        (6, 12),
        (6, 14),
        (8, 10),
        (10, 12),
        (10, 14),
        (12, 14),
        # Left face.
        (1, 3),
        (1, 5),
        (3, 5),
        (5, 7),
        (7, 9),
        (7, 13),
        (7, 15),
        (9, 11),
        (11, 13),
        (11, 15),
        (13, 15),
        # Central mouth and jaw.
        (10, 16),
        (11, 16),
        (12, 16),
        (13, 16),
        (14, 17),
        (15, 17),
        (16, 17),
        (17, 18),
        (17, 19),
        (18, 19),
    )
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = 1.0
    return adjacency


def _unit(vector: np.ndarray, *, floor: float = 1e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= floor:
        return None
    return vector / norm


def canonicalize_landmarks(
    landmarks: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Remove 3-D translation, eye scale, roll, and coarse yaw/pitch.

    Returns ``(canonical_landmarks, pose_diagnostics)``.  An invalid frame
    returns ``None`` while keeping finite zero diagnostics.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    pose = np.zeros(4, dtype=np.float32)
    if (
        points.ndim != 2
        or points.shape[0] < MIN_REQUIRED_LANDMARKS
        or points.shape[1] < 3
        or not np.isfinite(points[:MIN_REQUIRED_LANDMARKS, :3]).all()
    ):
        return None, pose
    points = points[:MIN_REQUIRED_LANDMARKS, :3]
    right_eye = (points[33] + points[133]) / 2.0
    left_eye = (points[362] + points[263]) / 2.0
    origin = (right_eye + left_eye) / 2.0
    eye_vector = left_eye - right_eye
    scale = float(np.linalg.norm(eye_vector))
    ex = _unit(eye_vector)
    if ex is None or scale <= 1e-8:
        return None, pose
    nose = points[[1, 2, 4]].mean(axis=0)
    ey = _unit((nose - origin) - ex * float(np.dot(nose - origin, ex)))
    if ey is None:
        return None, pose
    ez = _unit(np.cross(ex, ey))
    if ez is None:
        return None, pose
    # Re-orthogonalize to limit numerical drift.
    ey = _unit(np.cross(ez, ex))
    if ey is None:
        return None, pose
    basis = np.stack((ex, ey, ez), axis=1)
    canonical = ((points - origin) @ basis) / scale
    pose[:] = (
        np.arctan2(ex[1], ex[0]),
        np.arctan2(ex[2], np.linalg.norm(ex[:2])),
        np.arctan2(ey[2], np.linalg.norm(ey[:2])),
        scale,
    )
    if not np.isfinite(canonical).all():
        return None, np.zeros(4, dtype=np.float32)
    return canonical.astype(np.float32), pose


def _rigid_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve row-vector Kabsch alignment ``source @ R + t ~= target``."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center
    u, _, vt = np.linalg.svd(centered_source.T @ centered_target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def expression_proxy_values(canonical: np.ndarray) -> np.ndarray:
    """Calculate the 20 signed geometric expression proxies for one frame."""
    q = np.asarray(canonical, dtype=np.float64)
    if q.shape != (MIN_REQUIRED_LANDMARKS, 3) or not np.isfinite(q).all():
        raise ValueError(
            "canonical landmarks must have finite shape "
            f"({MIN_REQUIRED_LANDMARKS}, 3)"
        )

    def y(index: int) -> float:
        return float(q[index, 1])

    def x(index: int) -> float:
        return float(q[index, 0])

    def width(left: int, right: int) -> float:
        return max(float(np.linalg.norm(q[left, :2] - q[right, :2])), 1e-4)

    right_eye_width = width(33, 133)
    left_eye_width = width(362, 263)
    center = (q[13] + q[14]) / 2.0
    result = np.asarray(
        (
            (y(133) - y(107)) / right_eye_width,
            (y(362) - y(336)) / left_eye_width,
            (y(33) - y(70)) / right_eye_width,
            (y(263) - y(300)) / left_eye_width,
            np.mean((y(145) - y(159), y(153) - y(158), y(144) - y(160)))
            / right_eye_width,
            np.mean((y(374) - y(386), y(380) - y(385), y(373) - y(387)))
            / left_eye_width,
            y(205) - float(q[[144, 145, 153], 1].mean()),
            y(425) - float(q[[373, 374, 380], 1].mean()),
            x(2) - x(98),
            x(327) - x(2),
            y(37) - y(98),
            y(267) - y(327),
            float(center[0]) - x(61),
            x(291) - float(center[0]),
            y(61) - float(center[1]),
            y(291) - float(center[1]),
            y(14) - y(13),
            y(17) - y(0),
            y(152) - y(17),
            y(152) - y(168),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(result).all():
        raise ValueError("expression proxy calculation produced NaN/Inf")
    return result


def _contiguous_true_segments(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, splits) if len(part)]


def _interpolate_short_gaps(
    values: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = values.copy()
    available = observed.copy()
    imputed = np.zeros_like(observed)
    valid_positions = np.flatnonzero(observed)
    for left, right in zip(valid_positions, valid_positions[1:]):
        if right == left + 1:
            continue
        if float(times[right] - times[left]) > max_gap_sec + 1e-9:
            continue
        weights = (times[left + 1 : right] - times[left]) / (
            times[right] - times[left]
        )
        result[left + 1 : right] = (
            result[left][None, :] * (1.0 - weights[:, None])
            + result[right][None, :] * weights[:, None]
        )
        available[left + 1 : right] = True
        imputed[left + 1 : right] = True
    return result, available, imputed


def _resample_proxy_timeline(
    values: np.ndarray,
    observed: np.ndarray,
    available: np.ndarray,
    source_times: np.ndarray,
    target_times: np.ndarray,
    pose: np.ndarray,
    *,
    max_gap_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate unique source observations onto the uniform target grid."""
    result = np.zeros(
        (len(target_times), values.shape[1]), dtype=np.float64
    )
    target_available = np.zeros(len(target_times), dtype=bool)
    target_observed = np.zeros(len(target_times), dtype=bool)
    target_pose = np.zeros((len(target_times), pose.shape[1]), dtype=np.float32)
    valid_positions = np.flatnonzero(available)
    if len(valid_positions) == 0:
        return (
            result,
            target_observed,
            np.zeros_like(target_observed),
            target_pose,
        )
    valid_times = source_times[valid_positions]
    typical_step = (
        float(np.median(np.diff(source_times)))
        if len(source_times) >= 2
        else max_gap_sec
    )
    edge_tolerance = max(typical_step, 1e-6)
    for output_index, target in enumerate(target_times):
        insertion = int(np.searchsorted(valid_times, target, side="left"))
        if insertion == 0:
            source_index = int(valid_positions[0])
            if abs(float(target - source_times[source_index])) > edge_tolerance:
                continue
            result[output_index] = values[source_index]
            target_available[output_index] = True
        elif insertion == len(valid_positions):
            source_index = int(valid_positions[-1])
            if abs(float(target - source_times[source_index])) > edge_tolerance:
                continue
            result[output_index] = values[source_index]
            target_available[output_index] = True
        else:
            left = int(valid_positions[insertion - 1])
            right = int(valid_positions[insertion])
            span = float(source_times[right] - source_times[left])
            if span <= 0 or span > max_gap_sec + 1e-9:
                continue
            weight = float((target - source_times[left]) / span)
            result[output_index] = (
                values[left] * (1.0 - weight) + values[right] * weight
            )
            target_available[output_index] = True

        nearest = int(np.argmin(np.abs(source_times - target)))
        target_observed[output_index] = bool(observed[nearest])

    # Pose is diagnostic only; interpolate it on truly observed source frames.
    observed_positions = np.flatnonzero(observed)
    if len(observed_positions):
        for feature in range(pose.shape[1]):
            target_pose[:, feature] = np.interp(
                target_times,
                source_times[observed_positions],
                pose[observed_positions, feature],
            )
        target_pose[~target_available] = 0.0
    target_observed &= target_available
    target_imputed = target_available & ~target_observed
    return result, target_observed, target_imputed, target_pose


def _local_polynomial_segments(
    values: np.ndarray,
    available: np.ndarray,
    times: np.ndarray,
    *,
    half_window_seconds: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth and differentiate with local quadratic fits.

    Direct second differences at 60 Hz amplify sub-pixel landmark jitter by
    roughly ``60**2``.  A 300-ms local quadratic estimates value, velocity and
    acceleration together while respecting every long missing-data boundary.
    """
    smoothed = values.copy()
    velocity = np.zeros_like(values, dtype=np.float64)
    acceleration = np.zeros_like(values, dtype=np.float64)
    for segment in _contiguous_true_segments(available):
        if len(segment) == 1:
            continue
        if len(segment) == 2:
            step = max(float(times[segment[1]] - times[segment[0]]), 1e-6)
            slope = (values[segment[1]] - values[segment[0]]) / step
            velocity[segment] = slope
            continue
        segment_times = times[segment]
        minimum_points = min(5, len(segment))
        for position, frame_index in enumerate(segment):
            relative = segment_times - times[frame_index]
            local = np.flatnonzero(
                np.abs(relative) <= half_window_seconds + 1e-12
            )
            if len(local) < minimum_points:
                local = np.argsort(np.abs(relative))[:minimum_points]
                local.sort()
            local_time = relative[local]
            design = np.stack(
                (
                    np.ones(len(local_time), dtype=np.float64),
                    local_time,
                    np.square(local_time),
                ),
                axis=1,
            )
            coefficients, _, _, _ = np.linalg.lstsq(
                design, values[segment[local]], rcond=None
            )
            smoothed[frame_index] = coefficients[0]
            velocity[frame_index] = coefficients[1]
            acceleration[frame_index] = 2.0 * coefficients[2]
    return smoothed, velocity, acceleration


def build_expression_graph(
    landmark_sequence: np.ndarray,
    source_times: np.ndarray,
    *,
    output_times: np.ndarray | None = None,
    trial_baseline_seconds: float = 1.25,
    max_impute_gap_sec: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw FaceMesh frames to ``[T,20,8]`` and pose diagnostics.

    Missing detections must be represented by NaNs in ``landmark_sequence``.
    Only bounded internal gaps are interpolated.
    """
    landmarks = np.asarray(landmark_sequence, dtype=np.float64)
    times = np.asarray(source_times, dtype=np.float64)
    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] != (MIN_REQUIRED_LANDMARKS, 3)
    ):
        raise ValueError(
            "landmark_sequence must have shape "
            f"[T,{MIN_REQUIRED_LANDMARKS},3], got {landmarks.shape}"
        )
    if (
        times.ndim != 1
        or len(times) != len(landmarks)
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("source_times must be finite, strictly increasing, and length T")
    target_times = (
        times.copy()
        if output_times is None
        else np.asarray(output_times, dtype=np.float64)
    )
    if (
        target_times.ndim != 1
        or len(target_times) < 2
        or not np.isfinite(target_times).all()
        or np.any(np.diff(target_times) <= 0)
    ):
        raise ValueError("output_times must be finite and strictly increasing")
    if trial_baseline_seconds <= 0:
        raise ValueError("trial_baseline_seconds must be positive")
    if max_impute_gap_sec < 0:
        raise ValueError("max_impute_gap_sec must be non-negative")

    canonical: list[np.ndarray | None] = []
    pose = np.zeros((len(landmarks), 4), dtype=np.float32)
    observed = np.zeros(len(landmarks), dtype=bool)
    for index, frame in enumerate(landmarks):
        normalized, frame_pose = canonicalize_landmarks(frame)
        pose[index] = frame_pose
        canonical.append(normalized)
        observed[index] = normalized is not None

    values = np.full(
        (len(landmarks), len(EXPRESSION_NODES)), np.nan, dtype=np.float64
    )
    valid_positions = np.flatnonzero(observed)
    if len(valid_positions):
        baseline_limit = float(times[0] + trial_baseline_seconds)
        reference_positions = valid_positions[times[valid_positions] <= baseline_limit]
        if len(reference_positions) == 0:
            reference_positions = valid_positions[:1]
        reference = np.median(
            np.stack(
                [canonical[int(index)][list(POSE_ANCHORS)] for index in reference_positions],
                axis=0,
            ),
            axis=0,
        )
        for index in valid_positions:
            points = canonical[int(index)]
            rotation, translation = _rigid_align(
                points[list(POSE_ANCHORS)], reference
            )
            aligned = points @ rotation + translation
            values[index] = expression_proxy_values(aligned)

    values, available, imputed = _interpolate_short_gaps(
        values, observed, times, float(max_impute_gap_sec)
    )
    if output_times is not None:
        values, observed, imputed, pose = _resample_proxy_timeline(
            values,
            observed,
            available,
            times,
            target_times,
            pose,
            max_gap_sec=float(max_impute_gap_sec),
        )
        available = observed | imputed
        times = target_times
    values[~available] = 0.0
    smoothed, velocity, acceleration = _local_polynomial_segments(
        values, available, times
    )
    smoothed[~available] = 0.0

    baseline_mask = observed & (times <= times[0] + trial_baseline_seconds)
    if not baseline_mask.any():
        baseline_mask = observed
    if baseline_mask.any():
        baseline_median = np.median(smoothed[baseline_mask], axis=0)
        mad = np.median(
            np.abs(smoothed[baseline_mask] - baseline_median), axis=0
        )
    else:
        baseline_median = np.zeros(len(EXPRESSION_NODES), dtype=np.float64)
        mad = np.ones(len(EXPRESSION_NODES), dtype=np.float64)
    delta = smoothed - baseline_median
    # A real floor is essential because a nearly constant 1.25-s baseline can
    # otherwise turn harmless tracker noise into an enormous z score.
    robust_scale = np.maximum(1.4826 * mad, 1e-2)
    robust_z = np.clip(delta / robust_scale, -6.0, 6.0)
    delta[~available] = 0.0
    robust_z[~available] = 0.0

    # Robust clipping prevents a single landmark jump dominating a fold.
    velocity = np.clip(velocity, -10.0, 10.0)
    acceleration = np.clip(acceleration, -50.0, 50.0)
    graph = np.stack(
        (
            smoothed,
            delta,
            velocity,
            acceleration,
            np.abs(velocity),
            robust_z,
            np.broadcast_to(observed[:, None], smoothed.shape),
            np.broadcast_to(imputed[:, None], smoothed.shape),
        ),
        axis=2,
    ).astype(np.float32)
    if not np.isfinite(graph).all():
        raise RuntimeError("expression graph contains NaN/Inf")
    return graph, pose
