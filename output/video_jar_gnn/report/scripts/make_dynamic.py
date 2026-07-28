#!/usr/bin/env python
"""Fig 6 (GNN graph animation + static small-multiples) and Fig 7 (real frame
with MediaPipe landmark overlay, eyes anonymised, mouth highlighted).

Run with: /home/gpu1/miniconda3/envs/video-jar-gnn/bin/python make_dynamic.py
"""
from __future__ import annotations
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch

ROOT = "/home/gpu1/tientx5/sens/EEG_ngot"
OUT = f"{ROOT}/output/video_jar_gnn/report/figures"
os.makedirs(OUT, exist_ok=True)

OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "blue": "#0072B2"}
REGION_COLORS = {"brow": OI["orange"], "eye": OI["sky"], "nose": OI["green"],
                 "mouth": OI["vermillion"], "jaw": OI["purple"]}
INK = "#1a1a1a"
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans",
                     "font.size": 11, "figure.facecolor": "white"})

LEGACY_NODES = [
    "brow_left_inner", "brow_left_outer", "brow_right_inner", "brow_right_outer",
    "eye_left_upper", "eye_left_lower", "eye_right_upper", "eye_right_lower",
    "nose_bridge", "nose_alar_left", "nose_alar_right",
    "upper_lip", "lower_lip", "lip_corners", "chin_center",
]


def region_of(node: str) -> str:
    n = node.lower()
    if "brow" in n:
        return "brow"
    if "eye" in n or "lid" in n:
        return "eye"
    if "nose" in n or "alar" in n:
        return "nose"
    if "jaw" in n or "chin" in n:
        return "jaw"
    return "mouth"


CACHE = f"{ROOT}/output/video_jar_gnn/graphs/P001/P001_189_R1.npz"


def _load_seq():
    d = np.load(CACHE, allow_pickle=True)
    g = np.asarray(d["graph_seq"])          # [T,15,10] cx,cy,cz,det,...
    adj = np.asarray(d["adj"])
    xy = g[:, :, :2].astype(float)          # [T,15,2]
    # per-node centre over time so the face stays framed; flip y for display
    xy[:, :, 1] *= -1.0
    return xy, adj


def fig6_animation():
    xy, adj = _load_seq()
    T = xy.shape[0]
    regs = [region_of(n) for n in LEGACY_NODES]
    colors = [REGION_COLORS[r] for r in regs]
    ii, jj = np.where(np.triu(adj) > 0)

    xmin, xmax = xy[:, :, 0].min(), xy[:, :, 0].max()
    ymin, ymax = xy[:, :, 1].min(), xy[:, :, 1].max()
    padx = (xmax - xmin) * 0.15 + 1e-3
    pady = (ymax - ymin) * 0.15 + 1e-3

    fig, ax = plt.subplots(figsize=(6.0, 6.6))
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.axis("off")
    title = ax.set_title("", fontsize=12, fontweight="bold")

    edge_lines = [ax.plot([], [], color="#c9c9c9", lw=1.6, zorder=1)[0]
                  for _ in ii]
    scat = ax.scatter(xy[0, :, 0], xy[0, :, 1], s=340, c=colors,
                      edgecolor="white", linewidth=1.4, zorder=3)
    present = [Patch(color=REGION_COLORS[r], label=r)
               for r in ("brow", "eye", "nose", "mouth", "jaw")]
    ax.legend(handles=present, loc="lower center", frameon=False, fontsize=8.5,
              ncol=5, bbox_to_anchor=(0.5, -0.04))

    step = 2
    frames = list(range(0, T, step))

    def update(fi):
        pts = xy[fi]
        scat.set_offsets(pts)
        for line, a, b in zip(edge_lines, ii, jj):
            line.set_data([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]])
        title.set_text(f"Hình 6 — Đồ thị khuôn mặt theo thời gian\n"
                       f"P001 · mẫu #189 · t = {fi/T*10:.1f}s / 10s")
        return [scat, *edge_lines, title]

    anim = FuncAnimation(fig, update, frames=frames, interval=90, blit=False)
    anim.save(f"{OUT}/fig6_gnn_animation.gif", writer=PillowWriter(fps=11))
    plt.close(fig)
    print("Fig6 GIF done.")

    # static small-multiples (for print) at 6 time points
    fig, axes = plt.subplots(1, 6, figsize=(15.5, 3.1))
    picks = np.linspace(0, T - 1, 6).astype(int)
    for ax, fi in zip(axes, picks):
        pts = xy[fi]
        for a, b in zip(ii, jj):
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color="#d0d0d0", lw=1.1, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=90, c=colors,
                   edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"t = {fi/T*10:.1f}s", fontsize=10)
    fig.suptitle("Hình 6b — Chuỗi đồ thị khuôn mặt trong 10 giây (P001, mẫu #189) — "
                 "bản tĩnh cho in ấn",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left", y=1.12)
    fig.subplots_adjust(top=0.80)
    fig.savefig(f"{OUT}/fig6_gnn_sequence_static.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig6 static done.")


def fig7_face_overlay():
    import cv2
    import mediapipe as mp

    video = f"{ROOT}/data/data_video/N01_vid-001.mp4"
    frame_idx = 113750  # inside P001_189_R1 labelled interval [113454, 114053]

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    tries = 0
    while (not ok) and tries < 5:
        ok, frame = cap.read()
        tries += 1
    cap.release()
    if not ok:
        print("Fig7 SKIP: could not decode frame")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5)
    res = mesh.process(rgb)
    mesh.close()
    if not res.multi_face_landmarks:
        print("Fig7 SKIP: no face detected in frame")
        return
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x * w, p.y * h] for p in lm[:468]])

    face_mesh = mp.solutions.face_mesh

    def edge_ids(conn):
        s = set()
        for a, b in conn:
            s.add((a, b))
        return s

    # crop tightly to face for a clean thesis figure
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    mx = (x1 - x0) * 0.35
    my = (y1 - y0) * 0.45
    cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    cx1, cy1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
    crop = rgb[cy0:cy1, cx0:cx1].copy()

    # anonymise the eyes with a bar (responsible default for a real participant)
    left_eye = pts[[33, 133, 159, 145]]
    right_eye = pts[[362, 263, 386, 374]]
    eye_top = int(min(left_eye[:, 1].min(), right_eye[:, 1].min()) - cy0 - 0.06 * (y1 - y0))
    eye_bot = int(max(left_eye[:, 1].max(), right_eye[:, 1].max()) - cy0 + 0.06 * (y1 - y0))
    ex0 = int(min(left_eye[:, 0].min(), right_eye[:, 0].min()) - cx0 - 0.05 * (x1 - x0))
    ex1 = int(max(left_eye[:, 0].max(), right_eye[:, 0].max()) - cx0 + 0.05 * (x1 - x0))
    crop[max(0, eye_top):max(0, eye_bot), max(0, ex0):max(0, ex1)] = 20

    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    ax.imshow(crop)

    def draw(conn, color, lw, alpha):
        for a, b in conn:
            xa, ya = pts[a, 0] - cx0, pts[a, 1] - cy0
            xb, yb = pts[b, 0] - cx0, pts[b, 1] - cy0
            ax.plot([xa, xb], [ya, yb], color=color, lw=lw, alpha=alpha, zorder=2)

    draw(face_mesh.FACEMESH_TESSELATION, OI["sky"], 0.35, 0.45)
    draw(face_mesh.FACEMESH_LIPS, OI["vermillion"], 2.4, 1.0)
    # mouth landmark points
    lip_ids = sorted({i for e in face_mesh.FACEMESH_LIPS for i in e})
    ax.scatter(pts[lip_ids, 0] - cx0, pts[lip_ids, 1] - cy0, s=10,
               color=OI["vermillion"], zorder=3)
    ax.axis("off")
    ax.set_title("Hình 7 — MediaPipe FaceMesh trên khung hình thật (P001, đang nếm)\n"
                 "Vùng MIỆNG (đỏ) = nơi tập trung tín hiệu yếu; mắt được ẩn danh",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.text(0.5, -0.03,
            "468 landmark → gộp thành 15 vùng cơ mặt (Hình 5). "
            "Ảnh minh họa phương pháp trích xuất; cần xác nhận đồng thuận khi dùng trong luận văn.",
            transform=ax.transAxes, ha="center", fontsize=8.2, color="#6b6b6b")
    fig.savefig(f"{OUT}/fig7_face_landmarks_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Fig7 done.")


if __name__ == "__main__":
    fig6_animation()
    fig7_face_overlay()
    print("\nDynamic figures written to", OUT)
