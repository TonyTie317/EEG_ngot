"""Domain constants shared by the video-JAR GNN pipeline."""

from __future__ import annotations

EXPECTED_SUBJECT_NUMBERS = tuple(i for i in range(1, 31) if i not in (12, 22))
EXPECTED_SUBJECTS = tuple(f"P{i:03d}" for i in EXPECTED_SUBJECT_NUMBERS)

SAMPLE_CODES = (189, 258, 453, 605, 762, 893)
WATER_CODE = 605
REPEATS = (1, 2, 3, 4, 5)

JAR3_NAMES = {
    0: "Khong_du",
    1: "Vua_phai",
    2: "Qua_nhieu",
}
BINARY_NAMES = {
    0: "Khac",
    1: "Vua_phai",
}

AU_NODES = (
    "brow_left_inner",
    "brow_left_outer",
    "brow_right_inner",
    "brow_right_outer",
    "eye_left_upper",
    "eye_left_lower",
    "eye_right_upper",
    "eye_right_lower",
    "nose_bridge",
    "nose_alar_left",
    "nose_alar_right",
    "upper_lip",
    "lower_lip",
    "lip_corners",
    "chin_center",
)

# Geometry is normalized to the inter-eye distance before these features are
# calculated. Derivatives are per second, not merely per captured frame.
FEATURE_NAMES = (
    "cx",
    "cy",
    "cz",
    "detected",
    "area",
    "aspect",
    "velocity_x",
    "velocity_y",
    "velocity_area",
    "velocity_aspect",
)


def jar_to_jar3(jar: int) -> int:
    """Map JAR 1..5 to 0=not enough, 1=just right, 2=too much."""
    if jar in (1, 2):
        return 0
    if jar == 3:
        return 1
    if jar in (4, 5):
        return 2
    raise ValueError(f"JAR must be an integer from 1 to 5, got {jar!r}")


def jar_to_binary(jar: int) -> int:
    """Map JAR to 1=just right (3), 0=other (1,2,4,5)."""
    if jar not in (1, 2, 3, 4, 5):
        raise ValueError(f"JAR must be an integer from 1 to 5, got {jar!r}")
    return int(jar == 3)
