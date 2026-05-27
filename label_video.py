"""
Label video CSVs with ma_mau (condition code) and lan_lap (repeat number)
based on the realigned EEG onsets in output/epochs/realign_offsets.csv.

EEG is 100 Hz; video is 60 fps. EEG timestamps (`timestamp` column) and video
timestamps (`t_lsl` column) share the same LSL clock — we use that as the
common time reference to map each realigned EEG onset to a video frame.

For every trial we mark exactly 10 s of video (600 consecutive frames at 60 fps)
starting from the realigned onset, writing the condition code into `ma_mau`
and the repeat number (1..5) into `lan_lap`. Frames outside any trial window
get ma_mau=0 and lan_lap=0.

Subject mapping: P001 ↔ N01, P002 ↔ N02, … P030 ↔ N30 (P012 and P022 absent).

Files are modified in place (overwritten).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd


def parse_lsl_timestamp(s) -> float:
    """Parse an LSL timestamp string robustly.

    Some EEG CSVs were exported with `.` as a thousands-separator (e.g.
    ``6.762.247.025.680.930`` actually meaning ``6762247.025680930``),
    while others are normal floats (``7723578.08``).

    Strategy: try float() first. If the integer part would be ≥7 digits
    (LSL timestamps live in the 1e6–1e7 range, so a 7-digit integer is
    correct), return it as-is. Otherwise strip all dots and place the
    decimal point after the first 7 digits — that uniformly recovers the
    intended value across the observed corruptions.
    """
    s = str(s).strip()
    try:
        v = float(s)
        if 1_000_000 <= v < 10_000_000:
            return v
    except ValueError:
        pass
    digits = s.replace('.', '').replace(',', '')
    if len(digits) >= 7 and digits.isdigit():
        return float(digits[:7] + '.' + digits[7:])
    return float('nan')

ROOT = Path(__file__).resolve().parent
EEG_DIR = ROOT / "datadone"
VIDEO_DIR = ROOT / "data_video"
OFFSETS_CSV = ROOT / "output" / "epochs" / "realign_offsets.csv"

EEG_PATTERN = "sub-{subject}_ses-S001_task-Default_run-001_eeg.csv"
VIDEO_PATTERN = "N{num:02d}_vid.csv"

VIDEO_FPS = 60
TRIAL_DURATION_SEC = 10
FRAMES_PER_TRIAL = VIDEO_FPS * TRIAL_DURATION_SEC  # 600


def subject_to_video_num(subject_id: str) -> int:
    """P001 -> 1, P023 -> 23, etc."""
    return int(subject_id.lstrip("P"))


def label_subject(subject_id: str, sub_offsets: pd.DataFrame) -> None:
    vid_num = subject_to_video_num(subject_id)
    video_path = VIDEO_DIR / VIDEO_PATTERN.format(num=vid_num)
    eeg_path = EEG_DIR / EEG_PATTERN.format(subject=subject_id)

    if not video_path.exists():
        print(f"[{subject_id}] skip — no video file ({video_path.name})")
        return
    if not eeg_path.exists():
        print(f"[{subject_id}] skip — no EEG file")
        return

    # EEG timestamps — only the `timestamp` column is needed.
    eeg_raw = pd.read_csv(eeg_path, usecols=["timestamp"])
    eeg_ts = eeg_raw["timestamp"].map(parse_lsl_timestamp).to_numpy(dtype=float)
    if np.isnan(eeg_ts).any():
        n_bad = int(np.isnan(eeg_ts).sum())
        # Interpolate any remaining NaNs from neighbours (assumes 100 Hz).
        eeg_ts = pd.Series(eeg_ts).interpolate(method="linear", limit_direction="both").to_numpy()
        print(f"  [{subject_id}] {n_bad} EEG timestamp rows recovered via interpolation")

    video = pd.read_csv(video_path)
    if "t_lsl" not in video.columns:
        print(f"[{subject_id}] skip — video missing t_lsl column")
        return

    n_frames = len(video)
    vid_lsl = pd.to_numeric(video["t_lsl"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(vid_lsl).any():
        # Parse robustly if any non-numeric strings sneaked in.
        vid_lsl = video["t_lsl"].map(parse_lsl_timestamp).to_numpy(dtype=float)
        vid_lsl = pd.Series(vid_lsl).interpolate(method="linear", limit_direction="both").to_numpy()

    ma_mau = np.zeros(n_frames, dtype=np.int64)
    lan_lap = np.zeros(n_frames, dtype=np.int64)

    n_marked = 0
    n_clipped = 0
    for _, row in sub_offsets.iterrows():
        onset_idx = int(row["new_onset"])
        if onset_idx < 0 or onset_idx >= len(eeg_ts):
            print(f"  [{subject_id}] trial_ix={row['trial_ix']} new_onset={onset_idx} out of EEG range — skip")
            continue
        onset_lsl = eeg_ts[onset_idx]
        # Search the closest video frame to onset_lsl, then mark FRAMES_PER_TRIAL frames forward.
        start_frame = int(np.searchsorted(vid_lsl, onset_lsl, side="left"))
        # Refine: pick the closer of start_frame and start_frame-1.
        if 0 < start_frame < n_frames and start_frame > 0:
            if abs(vid_lsl[start_frame - 1] - onset_lsl) < abs(vid_lsl[start_frame] - onset_lsl):
                start_frame -= 1
        if start_frame >= n_frames:
            print(f"  [{subject_id}] trial_ix={row['trial_ix']} onset past end of video — skip")
            continue
        end_frame = start_frame + FRAMES_PER_TRIAL
        if end_frame > n_frames:
            n_clipped += 1
            end_frame = n_frames
        ma_mau[start_frame:end_frame] = int(row["condition"])
        lan_lap[start_frame:end_frame] = int(row["repeat"])
        n_marked += 1

    # Overwrite any pre-existing ma_mau/lan_lap columns; place them at the end.
    video = video.drop(columns=[c for c in ("ma_mau", "lan_lap") if c in video.columns])
    video["ma_mau"] = ma_mau
    video["lan_lap"] = lan_lap
    video.to_csv(video_path, index=False)

    marked = int((ma_mau != 0).sum())
    print(
        f"[{subject_id}→{video_path.name}] trials marked={n_marked} "
        f"(clipped at EOF={n_clipped}), labelled frames={marked}/{n_frames}"
    )


def main() -> None:
    offsets = pd.read_csv(OFFSETS_CSV)
    for sid, sub_df in offsets.groupby("subject_id", sort=True):
        label_subject(sid, sub_df)


if __name__ == "__main__":
    main()
