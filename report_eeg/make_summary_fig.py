#!/usr/bin/env python3
"""Tạo hình tổng hợp so sánh ML vs DL (balanced accuracy) cho báo cáo EEG.
Số liệu lấy từ REPORT_ML_DL.md (mục 5.1) + kiểm chứng lại từ các CSV kết quả thô.
Task: Vua_phai vs Others (chance balanced_acc = 0.5).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

OUT = os.path.join(os.path.dirname(__file__), "figures")

# Okabe-Ito CVD-safe categorical pair
C_ML = "#0072B2"   # xanh dương  -> Machine Learning
C_DL = "#E69F00"   # cam         -> Deep Learning
INK  = "#222222"
MUTED = "#8a8a8a"

# (label, balanced_acc, family, flag)  flag: '' | 'fake' | 'fragile' | 'best'
# Nhãn mô tả PHƯƠNG PHÁP (không dùng v1/v2/v3 khó hiểu)
rows = [
    ("RandomForest cơ bản — không xử lý mất cân bằng",   0.500, "ML", "fake"),
    ("GradBoost + lọc nhiễu (Isolation Forest)",         0.558, "ML", ""),
    ("XGBoost + loại 3 người EEG nhiễu nhất",            0.604, "ML", ""),
    ("GradBoost + SMOTE (tăng mẫu lớp thiểu số)",        0.607, "ML", ""),
    ("DeepConvNet — CNN sâu, học từ tín hiệu thô",       0.624, "DL", ""),
    ("GradBoost + hạ ngưỡng quyết định (0.5→0.21)",      0.649, "ML", "best"),
    ("ShallowConvNet — CNN nông, học từ tín hiệu thô",   0.674, "DL", "best"),
    ("XGBoost + SMOTE, chỉ 2 feature",                   0.722, "ML", "fragile"),
]

rows_sorted = sorted(rows, key=lambda r: r[1])
labels = [r[0] for r in rows_sorted]
vals   = [r[1] for r in rows_sorted]
fams   = [r[2] for r in rows_sorted]
flags  = [r[3] for r in rows_sorted]
colors = [C_ML if f == "ML" else C_DL for f in fams]

fig, ax = plt.subplots(figsize=(13, 6.6))
y = range(len(rows_sorted))
bars = ax.barh(list(y), vals, color=colors, height=0.62, zorder=3,
               edgecolor="white", linewidth=1.2)

# Đánh dấu thứ cấp (secondary encoding): hatch cho bar 'giả' và 'kém ổn định'
for bar, flag in zip(bars, flags):
    if flag == "fake":
        bar.set_hatch("//"); bar.set_alpha(0.55)
    elif flag == "fragile":
        bar.set_hatch("xx"); bar.set_alpha(0.85)

# Nhãn giá trị ở cuối mỗi thanh
for yi, (v, flag) in enumerate(zip(vals, flags)):
    txt = f"{v:.3f}"
    if flag == "best":    txt += "  ★"
    if flag == "fragile": txt += "  ⚠"
    if flag == "fake":    txt += "  (giả)"
    ax.text(v + 0.004, yi, txt, va="center", ha="left",
            fontsize=10.5, color=INK,
            fontweight="bold" if flag in ("best", "fragile") else "normal")

# Đường chance (balanced_acc = 0.5)
ax.axvline(0.5, color=MUTED, lw=1.6, ls="--", zorder=2)
ax.text(0.5, len(rows_sorted) - 0.35, " chance = 0.50", color=MUTED,
        fontsize=9.5, ha="left", va="center", style="italic")

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0.45, 0.78)
ax.set_xlabel("Balanced accuracy  (Vua_phai vs Others — metric chính, chống mất cân bằng)",
              fontsize=11)
ax.set_title("So sánh ML vs DL — độ chính xác cân bằng theo tiến trình thực nghiệm",
             fontsize=13.5, fontweight="bold", pad=12)

# legend theo entity
leg = [Patch(facecolor=C_ML, label="Machine Learning (ML)"),
       Patch(facecolor=C_DL, label="Deep Learning (DL)"),
       Patch(facecolor="white", edgecolor=MUTED, hatch="//", label="Kết quả 'giả' (chỉ đoán lớp đa số)"),
       Patch(facecolor="white", edgecolor=MUTED, hatch="xx", label="Kém ổn định (K=2, dễ overfit fold)")]
ax.legend(handles=leg, loc="lower right", fontsize=9.5, framealpha=0.95,
          edgecolor="#dddddd")

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", color="#ececec", zorder=0)
ax.tick_params(length=0)
plt.tight_layout()
p = os.path.join(OUT, "08_ml_vs_dl_summary.png")
plt.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", p)

# ---- Hình 2: tradeoff accuracy vs balanced_acc (best mỗi họ) ----
fig2, ax2 = plt.subplots(figsize=(9.0, 5.6))
# (name, acc, bacc, color, ha, dx, dy)  — căn nhãn để không tràn/không chồng
pts = [
    ("XGBoost (tối ưu accuracy)",  0.778, 0.604, C_ML, "right", -12, 8),
    ("GradBoost + hạ ngưỡng",      0.681, 0.649, C_ML, "left",  10, 6),
    ("ShallowConvNet (CNN nông)",  0.639, 0.674, C_DL, "left",  10, 6),
    ("DeepConvNet (CNN sâu)",      0.520, 0.624, C_DL, "left",  10, 6),
]
for name, acc, bacc, c, ha, dx, dy in pts:
    ax2.scatter(acc, bacc, s=170, color=c, zorder=3, edgecolor="white", linewidth=1.5)
    ax2.annotate(name, (acc, bacc), textcoords="offset points",
                 xytext=(dx, dy), ha=ha, fontsize=9, color=INK)
ax2.axhline(0.5, color=MUTED, lw=1.3, ls="--")
ax2.axvline(0.756, color=MUTED, lw=1.3, ls=":")
ax2.text(0.757, 0.52, " majority acc = 0.756", color=MUTED, fontsize=8.5, rotation=90, va="bottom")
ax2.text(0.60, 0.505, "chance bacc = 0.50", color=MUTED, fontsize=8.5, style="italic")
ax2.set_xlabel("Accuracy thô  (dễ bị thổi phồng bởi lớp đa số)", fontsize=10.5)
ax2.set_ylabel("Balanced accuracy", fontsize=10.5)
ax2.set_title("Đánh đổi Accuracy ↔ Balanced accuracy", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_xlim(0.46, 0.86); ax2.set_ylim(0.48, 0.71)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(color="#ececec", zorder=0); ax2.tick_params(length=0)
leg2 = [Patch(facecolor=C_ML, label="ML"), Patch(facecolor=C_DL, label="DL")]
ax2.legend(handles=leg2, loc="lower left", fontsize=9.5, edgecolor="#dddddd")
plt.tight_layout()
p2 = os.path.join(OUT, "08_acc_vs_bacc_tradeoff.png")
plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", p2)
