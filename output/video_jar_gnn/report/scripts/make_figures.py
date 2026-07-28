#!/usr/bin/env python
"""Generate thesis figures for the Video-JAR-GNN report (Fig 1-5).

Run with the video-jar-gnn conda python:
  /home/gpu1/miniconda3/envs/video-jar-gnn/bin/python make_figures.py

All figures are 300 dpi PNG, colourblind-safe (Okabe-Ito), light background.
"""
from __future__ import annotations
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = "/home/gpu1/tientx5/sens/EEG_ngot"
AUDIT = f"{ROOT}/output/video_jar_gnn/expression_audit/run_d33bb95400"
OUT = f"{ROOT}/output/video_jar_gnn/report/figures"
os.makedirs(OUT, exist_ok=True)

# ---- Okabe-Ito colourblind-safe palette ----
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
REGION_COLORS = {
    "brow": OI["orange"], "eye": OI["sky"], "nose": OI["green"],
    "mouth": OI["vermillion"], "jaw": OI["purple"],
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SWEET = [189, 258, 453, 762, 893]


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
    if any(k in n for k in ("lip", "mouth", "corner", "lowerlip")):
        return "mouth"
    return "nose"


# =====================================================================
# FIGURE 1 — Dose-response: JAR varies monotonically with concentration
# =====================================================================
def fig1_dose_response():
    subj_code_jar = defaultdict(dict)
    for path in sorted(glob.glob(f"{ROOT}/data/datadone/sub-P*_eeg.csv")):
        subj = os.path.basename(path).split("_")[0].replace("sub-", "")
        vals = defaultdict(set)
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    code = int(float(r["ma_mau"]))
                    jar = int(float(r["JAR"]))
                except (ValueError, TypeError, KeyError):
                    continue
                if code in SWEET and jar in (1, 2, 3, 4, 5):
                    vals[code].add(jar)
        for code, s in vals.items():
            if len(s) == 1:
                subj_code_jar[subj][code] = next(iter(s))

    # order codes by mean JAR = empirical perceived sweetness
    means = {c: np.mean([cj[c] for cj in subj_code_jar.values() if c in cj]) for c in SWEET}
    order = sorted(SWEET, key=lambda c: means[c])
    xpos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    # faint per-subject lines
    for subj, cj in subj_code_jar.items():
        ys = [cj.get(c, np.nan) for c in order]
        ax.plot(xpos, ys, color=MUTED, alpha=0.18, lw=0.9,
                marker="o", ms=2.5, zorder=1)
    # bold mean +/- 95% CI
    ms, los, his = [], [], []
    for c in order:
        vals = np.array([cj[c] for cj in subj_code_jar.values() if c in cj], float)
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        ms.append(m)
        los.append(m - 1.96 * se)
        his.append(m + 1.96 * se)
    ax.fill_between(xpos, los, his, color=OI["blue"], alpha=0.18, zorder=2)
    ax.plot(xpos, ms, color=OI["blue"], lw=2.6, marker="o", ms=7,
            zorder=3, label="Trung bình quần thể (±95% CI)")
    ax.axhline(3, color=OI["vermillion"], lw=1.4, ls="--", zorder=2)
    ax.text(len(order) - 1, 3.08, "JAR = 3  «Vừa phải»",
            color=OI["vermillion"], ha="right", va="bottom", fontsize=9.5)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"#{c}" for c in order])
    ax.set_xlabel("Mẫu sucrose (mã mẫu, xếp theo độ ngọt cảm nhận tăng dần →)")
    ax.set_ylabel("Rating JAR  (1 = chưa đủ … 5 = quá ngọt)")
    ax.set_ylim(0.6, 5.4)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("Hình 1 — Nhãn JAR biến thiên đơn điệu theo nồng độ\n"
                 "(28 người; 0/28 người có JAR hằng số → nhãn KHÔNG thoái hóa)",
                 fontsize=12, fontweight="bold", loc="left")
    leg = [Line2D([], [], color=OI["blue"], lw=2.6, marker="o", ms=7,
                  label="Trung bình quần thể (±95% CI)"),
           Line2D([], [], color=MUTED, lw=1.2, marker="o", ms=3, alpha=0.5,
                  label="Từng người (n=28)")]
    ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=9.5)
    ax.text(0.0, -0.17,
            "Nồng độ (code_prior) dự đoán JAR: BAcc jar3 = 0.64, binary = 0.63. "
            "Nút thắt KHÔNG ở nhãn — mà ở kênh khuôn mặt.",
            transform=ax.transAxes, fontsize=8.6, color=MUTED)
    fig.savefig(f"{OUT}/fig1_dose_response_jar.png")
    plt.close(fig)
    print("Fig1 done. Dose order by mean JAR:", order, {c: round(means[c], 2) for c in order})


# =====================================================================
# FIGURE 2 — The subject-identity confound (the key methodological point)
# =====================================================================
def fig2_identity_confound():
    summary = json.load(open(f"{AUDIT}/summary.json"))
    wins = [w for w in summary["windows"]]
    labels = [w["window"] for w in wins]
    raw1 = [w["raw_icc1_median"] for w in wins]
    rawk = [w["raw_icck_median"] for w in wins]
    cen1 = [w["subject_centered_icc1_median"] for w in wins]

    # per-feature distributions for window 0:10
    rows = list(csv.DictReader(open(f"{AUDIT}/feature_reliability.csv")))
    def col(rs, w, c):
        out = []
        for r in rs:
            if r["window"] == w:
                try:
                    out.append(float(r[c]))
                except ValueError:
                    pass
        return np.array(out)
    raw_dist = col(rows, "0:10", "raw_icc1")
    cen_dist = col(rows, "0:10", "subject_centered_icc1")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel A: per-window medians
    x = np.arange(len(labels))
    w = 0.26
    axA.bar(x - w, rawk, w, color=OI["vermillion"], label="ICC thô (trung bình k lần lặp)")
    axA.bar(x, raw1, w, color=OI["orange"], label="ICC thô (1 lần lặp)")
    axA.bar(x + w, cen1, w, color=OI["blue"], label="ICC sau khi khử danh tính")
    axA.axhline(0, color=INK, lw=0.9)
    axA.axhline(0.25, color=MUTED, lw=1.0, ls=":")
    axA.text(len(labels) - 0.5, 0.26, "ngưỡng 0.25", color=MUTED, fontsize=8.5, ha="right")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=9)
    axA.set_xlabel("Cửa sổ thời gian (giây)")
    axA.set_ylabel("ICC trung vị")
    axA.set_title("(A) Độ tin cậy lặp lại theo cửa sổ", fontsize=11, fontweight="bold", loc="left")
    axA.legend(loc="upper right", frameon=False, fontsize=8.6)

    # Panel B: distribution for window 0:10
    bins = np.linspace(-0.5, 1.0, 46)
    axB.hist(raw_dist, bins=bins, color=OI["vermillion"], alpha=0.55,
             label=f"ICC thô  (median {np.median(raw_dist):+.2f})")
    axB.hist(cen_dist, bins=bins, color=OI["blue"], alpha=0.65,
             label=f"Khử danh tính  (median {np.median(cen_dist):+.2f})")
    axB.axvline(0, color=INK, lw=1.0)
    axB.axvline(0.25, color=MUTED, lw=1.0, ls=":")
    axB.text(0.255, axB.get_ylim()[1] * 0.9, "0.25", color=MUTED, fontsize=8.5)
    axB.set_xlabel("ICC(1) của từng đặc trưng")
    axB.set_ylabel("Số đặc trưng (trong 720)")
    axB.set_title("(B) Phân bố đặc trưng, cửa sổ 0–10 s", fontsize=11, fontweight="bold", loc="left")
    axB.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle("Hình 2 — «Độ tin cậy» thô CHỈ là rò rỉ danh tính người; "
                 "khử danh tính → tín hiệu theo nồng độ ≈ 0",
                 fontsize=12.5, fontweight="bold", x=0.01, ha="left", y=1.02)
    frac = np.mean(cen_dist >= 0.25) * 100
    fig.text(0.01, -0.03,
             f"Sau khử danh tính: 0/4320 đặc trưng-cửa-sổ đạt ICC≥0.25 "
             f"(cửa sổ 0–10 s: {frac:.1f}%). Mọi trainer dùng LOSO/group CV → buộc "
             f"hoạt động trong đúng vùng ICC≈0 này ⇒ near-chance là tất yếu.",
             fontsize=8.8, color=MUTED)
    fig.savefig(f"{OUT}/fig2_identity_confound_icc.png")
    plt.close(fig)
    print("Fig2 done. raw0:10 med", round(float(np.median(raw_dist)), 3),
          "cen med", round(float(np.median(cen_dist)), 3))


# =====================================================================
# FIGURE 3 — Where the faint signal lives: mouth region, 2-4 s post-sip
# =====================================================================
def fig3_top_features():
    rows = list(csv.DictReader(open(f"{AUDIT}/feature_reliability.csv")))
    recs = []
    for r in rows:
        try:
            v = float(r["subject_centered_icc1"])
        except ValueError:
            continue
        recs.append((v, r["window"], r["node"], r["source_feature"], r["statistic"]))
    recs.sort(reverse=True)
    top = recs[:18][::-1]  # ascending for horizontal bars

    vals = [t[0] for t in top]
    regs = [region_of(t[2]) for t in top]
    colors = [REGION_COLORS[r] for r in regs]
    labels = [f"{t[2]} · {t[3]}/{t[4]}  [{t[1]}s]" for t in top]

    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    y = np.arange(len(top))
    ax.barh(y, vals, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.3)
    ax.axvline(0.25, color=MUTED, ls=":", lw=1.1)
    ax.text(0.25, len(top) - 0.2, "ngưỡng tín hiệu 0.25\n(KHÔNG feature nào đạt)",
            color=MUTED, fontsize=8.5, va="top", ha="center")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(-0.8, len(top) + 0.2)
    ax.set_xlabel("ICC(1) sau khi khử danh tính  (độ lặp lại của tín hiệu theo nồng độ)")
    ax.set_title("Hình 3 — 18 đặc trưng khuôn mặt «tin cậy» nhất đều ở VÙNG MIỆNG,\n"
                 "tập trung cửa sổ 2–4 s ngay sau khi nhấp (đỉnh chỉ 0.21)",
                 fontsize=12, fontweight="bold", loc="left")
    present = []
    for reg in ("mouth", "nose", "brow", "eye", "jaw"):
        if reg in regs:
            present.append(Patch(color=REGION_COLORS[reg], label=reg))
    ax.legend(handles=present, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=9.5, title="Vùng khuôn mặt")
    ax.text(0.0, -0.15,
            "Diễn giải: các đặc trưng nhỉnh trên nhiễu là mở miệng / kéo mép — "
            "cơ học NUỐT-NẾM, không phải phản ứng cảm xúc; và vẫn quá yếu để phân loại.",
            transform=ax.transAxes, fontsize=8.6, color=MUTED)
    fig.savefig(f"{OUT}/fig3_top_mouth_features.png")
    plt.close(fig)
    print("Fig3 done. Top feature:", top[-1])


# =====================================================================
# FIGURE 4 — Model performance vs baselines (all video-only near chance)
# =====================================================================
def fig4_models():
    # (label, task, model_bacc, ci_lo, ci_hi, chance, is_leak, note)
    data = [
        ("LogReg\n(expression)", "jar3", 0.3463, 0.262, 0.4276, 1/3, False, ""),
        ("ST-GCN\n(advanced)", "jar3", 0.3304, 0.2596, 0.4039, 1/3, False, ""),
        ("TCN expr_v2\n(video-only)", "binary", 0.518, 0.465, 0.5715, 0.5, False, ""),
        ("TCN trial_δ\n(video-only)", "binary", 0.4768, 0.3964, 0.5535, 0.5, False, ""),
        ("TCN prior-residual\n(RÒ RỈ mã mẫu)", "binary", 0.6307, 0.525, 0.7363, 0.5, True, "code_prior=0.63"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    x = np.arange(len(data))
    for i, (lab, task, b, lo, hi, ch, leak, note) in enumerate(data):
        color = OI["purple"] if leak else (OI["green"] if task == "jar3" else OI["blue"])
        ax.bar(i, b, 0.62, color=color, alpha=0.9,
               hatch="///" if leak else None, edgecolor="white")
        ax.errorbar(i, b, yerr=[[b - lo], [hi - b]], color=INK, lw=1.4,
                    capsize=4, capthick=1.4)
        ax.text(i, hi + 0.012, f"{b:.3f}", ha="center", fontsize=9, fontweight="bold")
    # chance lines
    ax.plot([-0.5, 1.5], [1/3, 1/3], color=OI["vermillion"], ls="--", lw=1.6)
    ax.text(-0.45, 1/3 + 0.008, "chance jar3 = 0.333", color=OI["vermillion"], fontsize=8.6)
    ax.plot([1.5, 4.5], [0.5, 0.5], color=OI["vermillion"], ls="--", lw=1.6)
    ax.text(2.6, 0.505, "chance binary = 0.50", color=OI["vermillion"], fontsize=8.6)
    ax.axvline(1.5, color=GRID, lw=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=8.6)
    ax.set_ylabel("Balanced accuracy (±95% bootstrap CI theo người)")
    ax.set_ylim(0, 0.82)
    ax.set_title("Hình 4 — Mọi mô hình CHỈ-VIDEO nằm ở mức ngẫu nhiên;\n"
                 "số duy nhất trên baseline là RÒ RỈ bảng mã mẫu, không phải khuôn mặt",
                 fontsize=12, fontweight="bold", loc="left")
    leg = [Patch(color=OI["green"], label="jar3 (3 lớp)"),
           Patch(color=OI["blue"], label="binary (2 lớp)"),
           Patch(facecolor=OI["purple"], hatch="///", edgecolor="white",
                 label="Rò rỉ code_prior (video +0.005, CI cắt 0)")]
    ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=8.8)
    fig.text(0.01, -0.02,
             "Mọi model_minus_baseline có CI 95% cắt ngang 0. ST-GCN jar3 (0.330) còn "
             "DƯỚI mức đa số. Video không thêm gì trên bảng nồng độ→JAR.",
             fontsize=8.7, color=MUTED)
    fig.savefig(f"{OUT}/fig4_model_vs_baseline.png")
    plt.close(fig)
    print("Fig4 done.")


# =====================================================================
# FIGURE 5 — The facial-region graph (what the GNN sees) coloured by region
# =====================================================================
LEGACY_NODES = [
    "brow_left_inner", "brow_left_outer", "brow_right_inner", "brow_right_outer",
    "eye_left_upper", "eye_left_lower", "eye_right_upper", "eye_right_lower",
    "nose_bridge", "nose_alar_left", "nose_alar_right",
    "upper_lip", "lower_lip", "lip_corners", "chin_center",
]


# Canonical (schematic) anatomical layout — readable positions for the 15
# nodes. The scientific content is the node set + the anatomical edges (real,
# from build_adjacency); positions are a clean schematic, not a measurement.
NODE_XY = {
    "brow_left_outer": (-1.35, 3.05), "brow_left_inner": (-0.55, 3.25),
    "brow_right_inner": (0.55, 3.25), "brow_right_outer": (1.35, 3.05),
    "eye_left_upper": (-1.0, 2.45), "eye_left_lower": (-1.0, 2.05),
    "eye_right_upper": (1.0, 2.45), "eye_right_lower": (1.0, 2.05),
    "nose_bridge": (0.0, 2.15), "nose_alar_left": (-0.5, 0.95),
    "nose_alar_right": (0.5, 0.95),
    "upper_lip": (0.0, 0.35), "lower_lip": (0.0, -0.2),
    "lip_corners": (0.95, 0.08), "chin_center": (0.0, -1.15),
}
NODE_SHORT = {
    "brow_left_outer": "brow_L_out", "brow_left_inner": "brow_L_in",
    "brow_right_inner": "brow_R_in", "brow_right_outer": "brow_R_out",
    "eye_left_upper": "eye_L_up", "eye_left_lower": "eye_L_lo",
    "eye_right_upper": "eye_R_up", "eye_right_lower": "eye_R_lo",
    "nose_bridge": "nose_bridge", "nose_alar_left": "alar_L",
    "nose_alar_right": "alar_R", "upper_lip": "upper_lip",
    "lower_lip": "lower_lip", "lip_corners": "lip_corners",
    "chin_center": "chin",
}


def fig5_graph():
    caches = sorted(glob.glob(f"{ROOT}/output/video_jar_gnn/graphs/**/*.npz", recursive=True))
    adj = np.asarray(np.load(caches[0], allow_pickle=True)["adj"])
    xs = np.array([NODE_XY[n][0] for n in LEGACY_NODES])
    ys = np.array([NODE_XY[n][1] for n in LEGACY_NODES])
    regs = [region_of(nm) for nm in LEGACY_NODES]

    fig, ax = plt.subplots(figsize=(7.4, 8.2))
    ii, jj = np.where(np.triu(adj) > 0)
    for a, b in zip(ii, jj):
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#c9c9c9", lw=1.8, zorder=1)
    for k, nm in enumerate(LEGACY_NODES):
        if regs[k] == "mouth":
            ax.scatter(xs[k], ys[k], s=1500, facecolor="none",
                       edgecolor=OI["vermillion"], linewidth=2.6, zorder=2)
    for k, nm in enumerate(LEGACY_NODES):
        ax.scatter(xs[k], ys[k], s=900, color=REGION_COLORS[regs[k]],
                   edgecolor="white", linewidth=2.0, zorder=3)
    for k, nm in enumerate(LEGACY_NODES):
        dy = -0.28 if regs[k] in ("mouth", "jaw", "nose") else 0.30
        va = "top" if dy < 0 else "bottom"
        ax.annotate(NODE_SHORT[nm], (xs[k], ys[k]), xytext=(0, dy * 40),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=7.6, color=INK, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.4)
    ax.set_ylim(-1.9, 3.9)
    ax.axis("off")
    ax.set_title("Hình 5 — Đồ thị 15 vùng khuôn mặt mà GNN xử lý\n"
                 "(node = vùng cơ mặt; cạnh = liên kết giải phẫu thực; sơ đồ minh họa)",
                 fontsize=12.5, fontweight="bold", loc="left")
    present = [Patch(color=REGION_COLORS[r], label=r)
               for r in ("brow", "eye", "nose", "mouth", "jaw")]
    present.append(Line2D([], [], color=OI["vermillion"], lw=2.6, marker="o",
                          markerfacecolor="none", ms=16,
                          label="Vùng mang tín hiệu (yếu, ICC≤0.21)"))
    ax.legend(handles=present, loc="upper center", frameon=False, fontsize=9.5,
              ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.savefig(f"{OUT}/fig5_face_graph_signal.png")
    plt.close(fig)
    print("Fig5 done (schematic layout).")


if __name__ == "__main__":
    fig1_dose_response()
    fig2_identity_confound()
    fig3_top_features()
    fig4_models()
    fig5_graph()
    print("\nAll static figures written to", OUT)
