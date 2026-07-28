"""Build and validate the trial manifest from the three read-only data sources.

Join key
--------
``(subject_id, ma_mau, repeat)``:

* ``N01`` in the video files maps to ``P001`` in the EEG/JAR files.
* Video frame labels provide the exact trial intervals.
* EEG CSV files provide the JAR value. No EEG samples enter the video model.

This module intentionally uses the Python standard library only. It can audit
the large CSV files even before the ML/video dependencies are installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    BINARY_NAMES,
    EXPECTED_SUBJECTS,
    JAR3_NAMES,
    REPEATS,
    SAMPLE_CODES,
    jar_to_binary,
    jar_to_jar3,
)


class ManifestError(RuntimeError):
    """Raised when data cannot be joined without ambiguity."""


MANIFEST_FIELDS = (
    "sample_id",
    "subject_id",
    "video_subject_id",
    "ma_mau",
    "repeat",
    "jar",
    "jar3_label",
    "jar3_name",
    "binary_label",
    "binary_name",
    "start_frame",
    "end_frame",
    "n_labelled_frames",
    "start_lsl",
    "end_lsl",
    "duration_lsl_sec",
    "video_path",
    "frame_label_path",
    "jar_path",
    "graph_path",
)


def _integral(value: Any, field: str) -> int:
    text = str(value).strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"Invalid {field} value: {value!r}") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ManifestError(f"{field} must be integral, got {value!r}")
    return int(numeric)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _subject_from_video_label(path: Path) -> tuple[str, str]:
    match = re.fullmatch(r"N(\d{2})_vid", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ManifestError(f"Unexpected frame-label filename: {path.name}")
    number = int(match.group(1))
    return f"P{number:03d}", f"N{number:02d}"


def _subject_from_video(path: Path) -> tuple[str, str] | None:
    match = re.match(r"^N(\d{2})_vid(?:[-_].*)?$", path.stem, flags=re.IGNORECASE)
    if not match:
        return None
    number = int(match.group(1))
    return f"P{number:03d}", f"N{number:02d}"


def _subject_from_jar(path: Path) -> str | None:
    match = re.match(r"^sub-P(\d{3})_.*_eeg$", path.stem, flags=re.IGNORECASE)
    return f"P{int(match.group(1)):03d}" if match else None


def discover_sources(
    video_dir: Path,
    frame_label_dir: Path,
    jar_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Discover one video, one frame-label CSV and one JAR CSV per subject."""
    sources: dict[str, dict[str, Path]] = defaultdict(dict)

    video_extensions = {".mp4", ".mov", ".mkv", ".avi"}
    for path in sorted(video_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in video_extensions:
            continue
        parsed = _subject_from_video(path)
        if parsed is None:
            continue
        subject_id, _ = parsed
        if "video" in sources[subject_id]:
            raise ManifestError(
                f"More than one video for {subject_id}: "
                f"{sources[subject_id]['video']} and {path}"
            )
        sources[subject_id]["video"] = path.resolve()

    for path in sorted(frame_label_dir.glob("N*_vid.csv")):
        subject_id, _ = _subject_from_video_label(path)
        if "frame_labels" in sources[subject_id]:
            raise ManifestError(f"More than one frame-label CSV for {subject_id}")
        sources[subject_id]["frame_labels"] = path.resolve()

    for path in sorted(jar_dir.glob("sub-P*_eeg.csv")):
        subject_id = _subject_from_jar(path)
        if subject_id is None:
            continue
        if "jar" in sources[subject_id]:
            raise ManifestError(f"More than one EEG/JAR CSV for {subject_id}")
        sources[subject_id]["jar"] = path.resolve()

    return dict(sources)


def read_frame_segments(path: Path) -> list[dict[str, Any]]:
    """Read contiguous non-zero ``(ma_mau, lan_lap)`` trial blocks."""
    required = {"frame_idx", "t_lsl", "ma_mau", "lan_lap"}
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_frame: int | None = None

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"{path}: missing columns {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            frame = _integral(row["frame_idx"], "frame_idx")
            if previous_frame is not None and frame <= previous_frame:
                raise ManifestError(
                    f"{path}:{line_number}: frame_idx is not strictly increasing"
                )
            if previous_frame is not None and frame != previous_frame + 1:
                raise ManifestError(
                    f"{path}:{line_number}: frame_idx gap "
                    f"{previous_frame} -> {frame}"
                )
            previous_frame = frame

            code = _integral(row["ma_mau"], "ma_mau")
            repeat = _integral(row["lan_lap"], "lan_lap")
            timestamp = _finite_float(row["t_lsl"])
            pair = (code, repeat) if code != 0 else None

            if pair is not None:
                if code not in SAMPLE_CODES:
                    raise ManifestError(f"{path}:{line_number}: unknown ma_mau={code}")
                if repeat not in REPEATS:
                    raise ManifestError(f"{path}:{line_number}: invalid lan_lap={repeat}")

            current_pair = (
                (current["ma_mau"], current["repeat"]) if current is not None else None
            )
            if pair != current_pair:
                if current is not None:
                    segments.append(current)
                    current = None
                if pair is not None:
                    current = {
                        "ma_mau": code,
                        "repeat": repeat,
                        "start_frame": frame,
                        "end_frame": frame,
                        "n_labelled_frames": 1,
                        "start_lsl": timestamp,
                        "end_lsl": timestamp,
                        "n_valid_lsl": int(timestamp is not None),
                        "lsl_monotonic": True,
                    }
            elif current is not None:
                current["end_frame"] = frame
                current["n_labelled_frames"] += 1
                if timestamp is not None:
                    if (
                        current["end_lsl"] is not None
                        and timestamp <= current["end_lsl"]
                    ):
                        current["lsl_monotonic"] = False
                    if current["start_lsl"] is None:
                        current["start_lsl"] = timestamp
                    current["end_lsl"] = timestamp
                    current["n_valid_lsl"] += 1

    if current is not None:
        segments.append(current)

    pair_counts = Counter((row["ma_mau"], row["repeat"]) for row in segments)
    repeated = [pair for pair, count in pair_counts.items() if count != 1]
    if repeated:
        raise ManifestError(f"{path}: trial pair occurs in multiple blocks: {repeated}")

    for row in segments:
        start_lsl, end_lsl = row["start_lsl"], row["end_lsl"]
        row["duration_lsl_sec"] = (
            end_lsl - start_lsl
            if start_lsl is not None and end_lsl is not None
            else None
        )
    return segments


def read_jar_trials(path: Path) -> dict[tuple[int, int], int]:
    """Read a consensus JAR value for each ``(ma_mau, repeat)``."""
    required = {"ma_mau", "repeat", "JAR"}
    values: dict[tuple[int, int], set[int]] = defaultdict(set)

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ManifestError(f"{path}: missing columns {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            raw = [str(row.get(key, "")).strip() for key in ("ma_mau", "repeat", "JAR")]
            if not any(raw):
                continue
            if not all(raw):
                raise ManifestError(
                    f"{path}:{line_number}: partially missing ma_mau/repeat/JAR"
                )
            code = _integral(raw[0], "ma_mau")
            repeat = _integral(raw[1], "repeat")
            jar = _integral(raw[2], "JAR")
            if code not in SAMPLE_CODES or repeat not in REPEATS:
                raise ManifestError(
                    f"{path}:{line_number}: invalid trial ({code}, {repeat})"
                )
            if jar not in (1, 2, 3, 4, 5):
                raise ManifestError(f"{path}:{line_number}: JAR outside 1..5: {jar}")
            values[(code, repeat)].add(jar)

    inconsistent = {key: sorted(value) for key, value in values.items() if len(value) != 1}
    if inconsistent:
        raise ManifestError(f"{path}: inconsistent JAR values: {inconsistent}")
    return {key: next(iter(value)) for key, value in values.items()}


def _validate_subject_set(
    sources: dict[str, dict[str, Path]],
    strict_complete: bool,
) -> list[str]:
    required_source_names = {"video", "frame_labels", "jar"}
    incomplete = {
        subject: sorted(required_source_names.difference(found))
        for subject, found in sources.items()
        if required_source_names.difference(found)
    }
    if incomplete:
        raise ManifestError(f"Subjects with missing sources: {incomplete}")

    subjects = sorted(sources)
    if strict_complete and tuple(subjects) != EXPECTED_SUBJECTS:
        missing = sorted(set(EXPECTED_SUBJECTS).difference(subjects))
        unexpected = sorted(set(subjects).difference(EXPECTED_SUBJECTS))
        raise ManifestError(
            f"Expected the 28 study subjects; missing={missing}, unexpected={unexpected}"
        )
    if not subjects:
        raise ManifestError("No complete subjects were discovered")
    return subjects


def build_manifest(
    video_dir: Path,
    frame_label_dir: Path,
    jar_dir: Path,
    *,
    strict_complete: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join all sources and return rows plus an audit summary."""
    for directory in (video_dir, frame_label_dir, jar_dir):
        if not directory.is_dir():
            raise ManifestError(f"Directory does not exist: {directory}")

    sources = discover_sources(video_dir, frame_label_dir, jar_dir)
    subjects = _validate_subject_set(sources, strict_complete)
    rows: list[dict[str, Any]] = []

    for subject_id in subjects:
        source = sources[subject_id]
        frame_segments = read_frame_segments(source["frame_labels"])
        jar_trials = read_jar_trials(source["jar"])
        jar_by_condition: dict[int, set[int]] = defaultdict(set)
        for (code, _repeat), jar in jar_trials.items():
            jar_by_condition[code].add(jar)
        inconsistent_conditions = {
            code: sorted(values)
            for code, values in jar_by_condition.items()
            if len(values) != 1
        }
        if inconsistent_conditions:
            raise ManifestError(
                f"{subject_id}: JAR differs between repeats of a condition: "
                f"{inconsistent_conditions}"
            )
        segment_keys = {(row["ma_mau"], row["repeat"]) for row in frame_segments}
        jar_keys = set(jar_trials)
        if segment_keys != jar_keys:
            raise ManifestError(
                f"{subject_id}: video/JAR trial keys differ; "
                f"missing JAR={sorted(segment_keys - jar_keys)}, "
                f"missing video={sorted(jar_keys - segment_keys)}"
            )

        expected_keys = {(code, repeat) for code in SAMPLE_CODES for repeat in REPEATS}
        if strict_complete and segment_keys != expected_keys:
            raise ManifestError(
                f"{subject_id}: expected 30 trials, got {len(segment_keys)}"
            )
        if strict_complete:
            bad_lengths = {
                (row["ma_mau"], row["repeat"]): row["n_labelled_frames"]
                for row in frame_segments
                if row["n_labelled_frames"] != 600
            }
            if bad_lengths:
                raise ManifestError(
                    f"{subject_id}: expected 600 labelled frames per trial: "
                    f"{bad_lengths}"
                )
            bad_timing = {
                (row["ma_mau"], row["repeat"]): {
                    "valid": row["n_valid_lsl"],
                    "frames": row["n_labelled_frames"],
                    "monotonic": row["lsl_monotonic"],
                }
                for row in frame_segments
                if row["n_valid_lsl"] != row["n_labelled_frames"]
                or not row["lsl_monotonic"]
            }
            if bad_timing:
                raise ManifestError(
                    f"{subject_id}: invalid active-frame t_lsl timing: {bad_timing}"
                )

        number = int(subject_id[1:])
        video_subject_id = f"N{number:02d}"
        for segment in frame_segments:
            code = int(segment["ma_mau"])
            repeat = int(segment["repeat"])
            jar = int(jar_trials[(code, repeat)])
            jar3 = jar_to_jar3(jar)
            binary = jar_to_binary(jar)
            rows.append(
                {
                    "sample_id": f"{subject_id}_{code}_R{repeat}",
                    "subject_id": subject_id,
                    "video_subject_id": video_subject_id,
                    "ma_mau": code,
                    "repeat": repeat,
                    "jar": jar,
                    "jar3_label": jar3,
                    "jar3_name": JAR3_NAMES[jar3],
                    "binary_label": binary,
                    "binary_name": BINARY_NAMES[binary],
                    "start_frame": segment["start_frame"],
                    "end_frame": segment["end_frame"],
                    "n_labelled_frames": segment["n_labelled_frames"],
                    "start_lsl": segment["start_lsl"],
                    "end_lsl": segment["end_lsl"],
                    "duration_lsl_sec": segment["duration_lsl_sec"],
                    "video_path": str(source["video"]),
                    "frame_label_path": str(source["frame_labels"]),
                    "jar_path": str(source["jar"]),
                    "graph_path": "",
                }
            )

    rows.sort(key=lambda row: (row["subject_id"], row["start_frame"]))
    trial_jar3 = Counter(row["jar3_name"] for row in rows)
    trial_binary = Counter(row["binary_name"] for row in rows)
    condition_rows = {
        (row["subject_id"], int(row["ma_mau"])): row for row in rows
    }
    non_water_rows = [row for row in rows if int(row["ma_mau"]) != 605]
    non_water_condition_rows = {
        (row["subject_id"], int(row["ma_mau"])): row for row in non_water_rows
    }
    condition_jar3 = Counter(row["jar3_name"] for row in condition_rows.values())
    condition_binary = Counter(row["binary_name"] for row in condition_rows.values())
    durations = [
        float(row["duration_lsl_sec"])
        for row in rows
        if row["duration_lsl_sec"] not in (None, "")
    ]

    audit = {
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_trials": len(rows),
        "trials_per_subject": dict(Counter(row["subject_id"] for row in rows)),
        "trial_jar3_counts": dict(sorted(trial_jar3.items())),
        "trial_binary_counts": dict(sorted(trial_binary.items())),
        "n_subject_condition_pairs": len(condition_rows),
        "condition_jar3_counts": dict(sorted(condition_jar3.items())),
        "condition_binary_counts": dict(sorted(condition_binary.items())),
        "non_water": {
            "n_trials": len(non_water_rows),
            "trial_jar3_counts": dict(
                sorted(Counter(row["jar3_name"] for row in non_water_rows).items())
            ),
            "trial_binary_counts": dict(
                sorted(Counter(row["binary_name"] for row in non_water_rows).items())
            ),
            "n_subject_condition_pairs": len(non_water_condition_rows),
            "condition_jar3_counts": dict(
                sorted(
                    Counter(
                        row["jar3_name"]
                        for row in non_water_condition_rows.values()
                    ).items()
                )
            ),
            "condition_binary_counts": dict(
                sorted(
                    Counter(
                        row["binary_name"]
                        for row in non_water_condition_rows.values()
                    ).items()
                )
            ),
        },
        "video_lsl_duration_sec": {
            "min": min(durations) if durations else None,
            "mean": sum(durations) / len(durations) if durations else None,
            "max": max(durations) if durations else None,
            "n_over_10_5_sec": sum(value > 10.5 for value in durations),
        },
    }
    return rows, audit


def write_manifest(
    rows: Iterable[dict[str, Any]],
    audit: dict[str, Any],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join video intervals and EEG-derived JAR labels."
    )
    parser.add_argument("--video-dir", type=Path, default=Path("data/data_video"))
    parser.add_argument(
        "--frame-label-dir",
        type=Path,
        default=Path("data/data_video (2)"),
    )
    parser.add_argument("--jar-dir", type=Path, default=Path("data/datadone"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/video_jar_gnn/manifest.csv"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a subset of subjects (useful for development fixtures).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        rows, audit = build_manifest(
            args.video_dir,
            args.frame_label_dir,
            args.jar_dir,
            strict_complete=not args.allow_incomplete,
        )
        write_manifest(rows, audit, args.output)
    except ManifestError as exc:
        raise SystemExit(f"Manifest validation failed: {exc}") from exc

    print(
        f"Manifest OK: {audit['n_subjects']} subjects, "
        f"{audit['n_trials']} trials -> {args.output}"
    )
    print(f"JAR3: {audit['trial_jar3_counts']}")
    print(f"Binary: {audit['trial_binary_counts']}")
    print(f"Audit: {args.output.with_suffix('.audit.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
