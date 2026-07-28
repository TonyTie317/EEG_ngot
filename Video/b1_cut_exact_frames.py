#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cut one segment per ma_mau by taking min/max frame of that code (sequential flavors).
- Input: video.mp4, labels.csv (needs: frame_idx, ma_mau; optional: timestamp)
- Output: one clip per ma_mau (saved directly into --outdir)
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path
import pandas as pd


def ffprobe_fps(video_path: str) -> float:
    cmd = [
        "ffprobe","-v","error",
        "-select_streams","v:0",
        "-show_entries","stream=avg_frame_rate",
        "-of","default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        num, den = out.split("/")
        return float(num) / float(den)
    return float(out)


def video_duration_sec(video_path: str) -> float:
    cmd = [
        "ffprobe","-v","error",
        "-show_entries","format=duration",
        "-of","default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def load_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # normalize columns
    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ["frame_idx", "frame_id"]: ren[c] = "frame_idx"
        elif lc == "ma_mau": ren[c] = "ma_mau"
        elif lc == "timestamp": ren[c] = "timestamp"
    df = df.rename(columns=ren)

    if "frame_idx" not in df.columns or "ma_mau" not in df.columns:
        raise ValueError("CSV must contain 'frame_idx' (or 'frame_id') and 'ma_mau'.")

    # clean ma_mau
    df["ma_mau"] = df["ma_mau"].astype(str).str.strip()
    df.loc[df["ma_mau"].isin(["", "nan", "NaN", "None"]), "ma_mau"] = pd.NA
    # drop ma_mau == 0
    df = df[~df["ma_mau"].isin(["0", 0])].copy()

    # normalize frame_idx
    df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["frame_idx"]).copy()
    df["frame_idx"] = df["frame_idx"].astype(int)
    return df


def group_simple_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    One block per ma_mau: take min/max frame; keep order by first appearance.
    Returns: ma_mau, start_frame, end_frame, n_frames, first_frame
    """
    lab = df.dropna(subset=["ma_mau"]).copy()
    if lab.empty:
        return pd.DataFrame(columns=["ma_mau","start_frame","end_frame","n_frames","first_frame"])

    agg = (lab.groupby("ma_mau", sort=False)
             .agg(start_frame=("frame_idx","min"),
                  end_frame=("frame_idx","max"),
                  first_frame=("frame_idx","min"))
             .reset_index())
    agg["n_frames"] = agg["end_frame"] - agg["start_frame"] + 1
    agg = agg.sort_values("first_frame").reset_index(drop=True)
    return agg[["ma_mau","start_frame","end_frame","n_frames","first_frame"]]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def ffmpeg_cut(input_video: str, start: float, duration: float, out_path: str, fast_copy: bool = False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fast_copy:
        cmd = ["ffmpeg","-y","-ss",f"{start:.6f}","-t",f"{duration:.6f}","-i",input_video,"-c","copy",out_path]
    else:
        cmd = ["ffmpeg","-y","-ss",f"{start:.6f}","-t",f"{duration:.6f}","-i",input_video,
               "-c:v","libx264","-preset","fast","-crf","18","-c:a","copy","-movflags","+faststart",out_path]
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Cut one segment per ma_mau by min/max frame.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fast-copy", action="store_true",
                    help="Use -c copy (fast, but boundary may align to keyframes)")
    ap.add_argument("--filename-format", type=str, default="{video_stem}_{code}.mp4",
                    help="Output file pattern in outdir. Variables: {video_stem}, {code}")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="Maximum duration in seconds for each segment (e.g., 5.0 for 5 seconds). If None, use full segment.")
    args = ap.parse_args()

    video = args.video
    labels = args.csv
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    fps = ffprobe_fps(video)
    dur = video_duration_sec(video)
    df = load_labels(labels)
    runs = group_simple_runs(df)  # one row per ma_mau

    manifest = outdir / "segments.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ma_mau","start_sec","end_sec","duration_sec",
            "start_frame","end_frame","n_frames","first_frame",
            "source_video","fps","output_file"
        ])
        writer.writeheader()

        base = Path(video).stem
        for _, r in runs.iterrows():
            code = str(r["ma_mau"])
            start_frame = int(r["start_frame"])
            end_frame = int(r["end_frame"])
            t0 = start_frame / fps
            t1 = (end_frame + 1) / fps  # include the last frame
            t0 = clamp(t0, 0.0, dur); t1 = clamp(t1, 0.0, dur)
            duration = max(0.0, t1 - t0)
            
            # Apply max_duration limit if specified
            if args.max_duration is not None and duration > args.max_duration:
                duration = args.max_duration
                t1 = t0 + duration
            
            if duration <= 0:
                continue

            # LƯU TRỰC TIẾP VÀO outdir (không tạo thư mục ma_mau)
            filename = args.filename_format.format(video_stem=base, code=code)
            outfile = outdir / filename

            ffmpeg_cut(video, t0, duration, str(outfile), fast_copy=args.fast_copy)

            writer.writerow({
                "ma_mau": code,
                "start_sec": round(t0,6),
                "end_sec": round(t1,6),
                "duration_sec": round(duration,6),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "n_frames": int(r["n_frames"]),
                "first_frame": int(r["first_frame"]),
                "source_video": os.path.basename(video),
                "fps": round(fps,6),
                "output_file": str(outfile)
            })

    print(f"Done. Segments: {len(runs)} | Manifest: {manifest}")
    print(f"FPS: {fps:.6f} | Duration: {dur:.3f}s")


if __name__ == "__main__":
    main()


# python3 cut_exact_frames.py \
#   --video "/home/tran.xuan.tien@sun-asterisk.com/SunAI /EEG/Labrecorder/Data_raw/P001/Kien1.mp4" \
#   --csv "/home/tran.xuan.tien@sun-asterisk.com/SunAI /EEG/Labrecorder/Data_raw/P001/Kien1_video_frames_with_times.csv" \
#   --outdir "/home/tran.xuan.tien@sun-asterisk.com/SunAI /EEG/Labrecorder/Data_chuan/P001"
#   --max-duration 5.0