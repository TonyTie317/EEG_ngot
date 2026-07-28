#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize AU-Graph .npz files exported from the FaceMesh→AU pipeline.

Inputs (.npz expected keys)
- graph_seq: float32 [T, N_AU, F] with feature_names like
    ['cx','cy','cz','vis','area','aspect','dcx','dcy','darea','daspect']
- adj      : float32 [N_AU, N_AU]
- meta     : dict or JSON string, may include:
    {'fps': 60.0, 'T':..., 'N_AU':..., 'feature_names': [...], 'au_nodes': [...], 'video_path': ..., ...}
- (optional) preview: uint8 [T, H, W, 3] — if present, we can overlay nodes/edges on frames.

Usage examples
-------------
# 1) Quick animation (auto-detect preview)
python visualize_npz.py --npz person_01_213.npz --mode animate

# 2) Only adjacency graph
python visualize_npz.py --npz person_01_213.npz --mode adj

# 3) Time-series for one node (by index or name)
python visualize_npz.py --npz person_01_213.npz --mode ts --node brow_left_inner
python visualize_npz.py --npz person_01_213.npz --mode ts --node 0

# 4) Animate a subrange and skip frames for speed
python visualize_npz.py --npz person_01_213.npz --mode animate --start 0 --end 600 --step 2

# 5) Save animation to MP4 (requires ffmpeg)
python visualize_npz.py --npz person_01_213.npz --mode animate --save out.mp4

Notes
-----
- Coordinates (cx, cy) are assumed to be normalized in [0,1]. We flip Y to match image origin.
- If preview exists, we scale points to image resolution and draw overlays on the video frame.
- Edge list is derived from adj > 0.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False


# ----------------------------
# Helpers
# ----------------------------

def load_npz(npz_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    data = np.load(str(npz_path), allow_pickle=True)
    X = data["graph_seq"]  # [T, N, F]
    A = data["adj"]        # [N, N]
    meta_raw = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
    if isinstance(meta_raw, (str, bytes)):
        meta = json.loads(meta_raw)
    elif isinstance(meta_raw, dict):
        meta = meta_raw
    else:
        meta = dict(meta_raw)

    preview = data["preview"] if "preview" in data.files else None
    return X, A, meta, preview


def index_of(lst: List[str], name: str, default: int = -1) -> int:
    try:
        return lst.index(name)
    except Exception:
        return default


def extract_features(X: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return (cx, cy, size, feature_names). size uses 'area' if present, else ones.
    X: [T, N, F]
    """
    feat_names = meta.get("feature_names") or ["cx","cy","cz","vis","area","aspect","dcx","dcy","darea","daspect"]
    def fidx(k: str) -> int:
        return index_of(feat_names, k)

    cx_i, cy_i = fidx("cx"), fidx("cy")
    area_i = fidx("area")
    if cx_i == -1 or cy_i == -1:
        raise ValueError("Missing 'cx' or 'cy' in feature_names")

    cx = X[..., cx_i]  # [T, N]
    cy = X[..., cy_i]
    if area_i != -1:
        size = X[..., area_i]
        # Normalize size to a reasonable point size range
        s = size.copy()
        s = s - np.nanmin(s)
        s = s / (np.nanmax(s) + 1e-6)
        size = 20.0 + 180.0 * s  # point size in scatter
    else:
        size = np.full_like(cx, 50.0)
    return cx, cy, size, feat_names


def make_edges(A: np.ndarray) -> np.ndarray:
    N = A.shape[0]
    edges = []
    thr = 1e-8
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] > thr or A[j, i] > thr:
                edges.append((i, j))
    return np.array(edges, dtype=int)


def get_labels(meta: Dict[str, Any], N: int) -> List[str]:
    labels = meta.get("au_nodes")
    if labels and len(labels) == N:
        return labels
    return [f"node_{i}" for i in range(N)]


# ----------------------------
# Plot modes
# ----------------------------

def plot_adjacency(A: np.ndarray, labels: List[str]):
    if _HAS_NX:
        G = nx.from_numpy_array(A)
        plt.figure(figsize=(7, 7))
        pos = nx.spring_layout(G, seed=0)
        nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.4)
        nx.draw_networkx_labels(G, pos, labels={i: lbl for i, lbl in enumerate(labels)}, font_size=8)
        plt.title("AU Adjacency Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    else:
        # Heatmap fallback
        plt.figure(figsize=(7, 6))
        plt.imshow(A, cmap="viridis")
        plt.colorbar(label="weight")
        plt.title("Adjacency (heatmap)")
        plt.xlabel("j")
        plt.ylabel("i")
        plt.tight_layout()
        plt.show()


def plot_timeseries(X: np.ndarray, meta: Dict[str, Any], node_sel: Union[int, str]):
    labels = get_labels(meta, X.shape[1])
    feat_names = meta.get("feature_names") or []
    if isinstance(node_sel, str) and not node_sel.isdigit():
        idx = index_of(labels, node_sel)
        if idx < 0:
            raise ValueError(f"Node name '{node_sel}' not found in au_nodes")
    else:
        idx = int(node_sel)
    if not (0 <= idx < X.shape[1]):
        raise ValueError(f"Node index {idx} out of range [0, {X.shape[1]-1}]")

    T = X.shape[0]
    t = np.arange(T)

    # Plot first 6 dims if available, else all
    dims = min(6, X.shape[2])
    plt.figure(figsize=(10, 6))
    for d in range(dims):
        plt.plot(t, X[:, idx, d], label=feat_names[d] if d < len(feat_names) else f"f{d}")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title(f"Time-series — {labels[idx]} (idx={idx})")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.show()


def animate_sequence(
    X: np.ndarray,
    A: np.ndarray,
    meta: Dict[str, Any],
    preview: Optional[np.ndarray] = None,
    start: int = 0,
    end: Optional[int] = None,
    step: int = 1,
    save_path: Optional[str] = None,
    overlay_alpha: float = 0.9,
):
    """Animate graph over time. If preview exists, overlay on video frames; otherwise 0-1 canvas.
    """
    cx, cy, size, _ = extract_features(X, meta)
    T, N = cx.shape[0], cx.shape[1]
    labels = get_labels(meta, N)
    edges = make_edges(A)

    if end is None or end > T:
        end = T
    frames = list(range(start, end, step))

    fps = float(meta.get("fps", 30.0))
    interval_ms = 1000.0 / max(1.0, fps / step)

    if preview is not None:
        H, W = preview.shape[1], preview.shape[2]

    fig = plt.figure(figsize=(10, 6))

    if preview is not None:
        ax = plt.gca()
        im = ax.imshow(preview[frames[0]])
        scat = ax.scatter(cx[frames[0]] * W, (1.0 - cy[frames[0]]) * H, s=size[frames[0]], alpha=overlay_alpha)
        # Pre-build edge segments
        segs = [
            [
                [cx[frames[0], i] * W, (1.0 - cy[frames[0], i]) * H],
                [cx[frames[0], j] * W, (1.0 - cy[frames[0], j]) * H],
            ]
            for i, j in edges
        ]
        lc = LineCollection(segs, linewidths=1.0, alpha=0.6)
        ax.add_collection(lc)
        ax.set_title(f"Overlay animation — {Path(meta.get('video_path','')).name}")
        ax.set_axis_off()

        def update(k):
            f = frames[k]
            im.set_data(preview[f])
            xy = np.column_stack([cx[f] * W, (1.0 - cy[f]) * H])
            scat.set_offsets(xy)
            scat.set_sizes(size[f])
            segs = [
                [
                    [cx[f, i] * W, (1.0 - cy[f, i]) * H],
                    [cx[f, j] * W, (1.0 - cy[f, j]) * H],
                ]
                for i, j in edges
            ]
            lc.set_segments(segs)
            return (im, scat, lc)

    else:
        # Normalized canvas [0,1]
        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.invert_yaxis()  # flip to screen coordinates
        ax.set_xlabel("cx")
        ax.set_ylabel("cy (flipped)")
        scat = ax.scatter(cx[frames[0]], cy[frames[0]], s=size[frames[0]], alpha=0.9)
        # Edges
        segs = [
            [
                [cx[frames[0], i], cy[frames[0], i]],
                [cx[frames[0], j], cy[frames[0], j]],
            ]
            for i, j in edges
        ]
        lc = LineCollection(segs, linewidths=1.0, alpha=0.5)
        ax.add_collection(lc)
        ax.set_title("AU Graph (normalized)")

        def update(k):
            f = frames[k]
            xy = np.column_stack([cx[f], cy[f]])
            scat.set_offsets(xy)
            scat.set_sizes(size[f])
            segs = [
                [
                    [cx[f, i], cy[f, i]],
                    [cx[f, j], cy[f, j]],
                ]
                for i, j in edges
            ]
            lc.set_segments(segs)
            return (scat, lc)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=False)

    if save_path:
        print(f"[INFO] Saving animation to {save_path} ...")
        try:
            ani.save(save_path, fps=max(1, int(fps/step)), dpi=150)
            print("[OK] Saved.")
        except Exception as e:
            print(f"[WARN] Could not save animation: {e}")

    plt.tight_layout()
    plt.show()


# ----------------------------
# Main CLI
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Visualize AU-Graph .npz")
    p.add_argument("--npz", required=True, help="Path to .npz file")
    p.add_argument("--mode", choices=["animate", "adj", "ts"], default="animate",
                   help="Visualization mode: animate | adj (graph) | ts (time-series)")
    p.add_argument("--node", default="0", help="Node index or name for --mode ts")
    p.add_argument("--start", type=int, default=0, help="Start frame (inclusive) for animation")
    p.add_argument("--end", type=int, default=None, help="End frame (exclusive) for animation")
    p.add_argument("--step", type=int, default=1, help="Frame stride for animation")
    p.add_argument("--save", default=None, help="Save animation to MP4/GIF path")
    args = p.parse_args()

    X, A, meta, preview = load_npz(args.npz)

    if args.mode == "adj":
        labels = get_labels(meta, A.shape[0])
        plot_adjacency(A, labels)
        return

    if args.mode == "ts":
        plot_timeseries(X, meta, args.node)
        return

    # animate
    animate_sequence(
        X=X,
        A=A,
        meta=meta,
        preview=preview,
        start=args.start,
        end=args.end,
        step=args.step,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
