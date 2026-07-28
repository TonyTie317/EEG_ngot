#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Huấn luyện baseline ST-GCN cho nhận diện 4 vị (ngọt, mặn, chua, đắng)
Input: split.csv + các file .npz
Tính năng:
 - Ép độ dài thời gian về T_fix (median của T train) để tránh lỗi stack
 - In train/val loss & accuracy mỗi epoch
 - Lưu model: stgcn_baseline.pt (best theo val_acc)
 - Vẽ train_curves.png (loss & acc theo epoch)
 - Vẽ conf_matrix.png (ma trận nhầm lẫn trên tập val)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_stgcn import AUSequenceDataset
from model_stgcn import SimpleSTGCN

# -----------------------------
# 0) Cấu hình chung
# -----------------------------
EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

LABEL2NAME = {0: "ngot", 1: "man", 2: "chua", 3: "dang"}

def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path="train_curves.png"):
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"📈 Saved training curve → {out_path}")

def compute_confusion_matrix(y_true, y_pred, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def plot_confusion_matrix(cm, labels, normalize=True, out_path="conf_matrix.png"):
    cm_disp = cm.astype(float)
    if normalize:
        row_sums = cm_disp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_disp = cm_disp / row_sums

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm_disp, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    thresh = cm_disp.max() / 2.0
    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            txt = f"{cm_disp[i, j]*100:.1f}%" if normalize else str(cm[i, j])
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if cm_disp[i, j] > thresh else "black", fontsize=9)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"📊 Saved confusion matrix → {out_path}")

# -----------------------------
# Adjacency normalization: symmetric Λ^{-1/2}(A+I)Λ^{-1/2}
# -----------------------------
def normalize_adj_symmetric(adj: torch.Tensor) -> torch.Tensor:
    """
    adj: [B, N, N] hoặc [N, N]
    return: [B, N, N] đã thêm self-loop và chuẩn hoá đối xứng
    """
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)  # [1, N, N]

    B, N, _ = adj.shape
    device = adj.device
    I = torch.eye(N, device=device).unsqueeze(0)  # [1, N, N]
    A = adj + I                                   # add self-loop

    # degree vector Λ_ii = sum_j A_ij
    D = A.sum(dim=-1)                              # [B, N]
    D_inv_sqrt = torch.pow(D.clamp_min(1.0), -0.5) # avoid div by 0

    # Λ^{-1/2} A Λ^{-1/2}
    A_norm = D_inv_sqrt.unsqueeze(-1) * A * D_inv_sqrt.unsqueeze(-2)  # [B, N, N]
    return A_norm

def main():
    # -----------------------------
    # 1) Đọc split.csv
    # -----------------------------
    if not os.path.exists("split.csv"):
        raise SystemExit("Không tìm thấy split.csv. Hãy chạy make_split.py trước.")

    df = pd.read_csv("split.csv")
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df   = df[df.split == "val"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise SystemExit("split.csv không có train/val hợp lệ.")

    # -----------------------------
    # 2) Chọn T_fix = median(T) của train
    # -----------------------------
    T_fix = int(train_df["T"].median())
    print(f"Using T_fix = {T_fix}")

    # -----------------------------
    # 3) DataLoader (Dataset đã ép T ở __getitem__)
    # -----------------------------
    train_loader = DataLoader(
        AUSequenceDataset(train_df, t_fix=T_fix, fix_mode="center"),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader   = DataLoader(
        AUSequenceDataset(val_df,   t_fix=T_fix, fix_mode="center"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # -----------------------------
    # 4) Suy ra N, F từ 1 file gốc (N & F không phụ thuộc T_fix)
    # -----------------------------
    sample_path = train_df.iloc[0]["path"]
    d0 = np.load(sample_path, allow_pickle=True)
    # graph_seq: [T, N, F]
    N = d0["graph_seq"].shape[1]
    F = d0["graph_seq"].shape[2]
    num_classes = 4
    print(f"N={N}, F={F}, Classes={num_classes}")

    # -----------------------------
    # 5) Model, Optim, Loss
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleSTGCN(in_feats=F, hid=128, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    # -----------------------------
    # 6) Train loop
    # -----------------------------
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, adj, y, _, _ in train_loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)

            # adj: [N, N] → [B, N, N] nếu cần
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)  # [B, N, N]

            # Chuẩn hoá đối xứng (GCN)
            A = normalize_adj_symmetric(adj)  # [B, N, N]

            logits = model(x, A)
            loss = ce(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        # ---- Validate ----
        model.eval()
        vtotal, vcorrect, vloss = 0, 0, 0.0
        all_true, all_pred = [], []
        with torch.no_grad():
            for x, adj, y, _, _ in val_loader:
                x, adj, y = x.to(device), adj.to(device), y.to(device)
                if adj.dim() == 2:
                    adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)

                A = normalize_adj_symmetric(adj)

                logits = model(x, A)
                loss = ce(logits, y)
                vloss += loss.item() * y.size(0)

                pred = logits.argmax(1)
                vcorrect += (pred == y).sum().item()
                vtotal += y.size(0)

                all_true.append(y.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

        val_loss = vloss / vtotal
        val_acc = vcorrect / vtotal

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "stgcn_baseline.pt")
            print(f"💾 Saved BEST model (val_acc={best_val_acc:.3f}) → stgcn_baseline.pt")

    # Lưu model sau cùng (tuỳ chọn)
    torch.save(model.state_dict(), "stgcn_last.pt")
    print("✅ Saved LAST model to stgcn_last.pt")

    # -----------------------------
    # 7) Vẽ curve
    # -----------------------------
    plot_curves(train_losses, val_losses, train_accs, val_accs, out_path="train_curves.png")

    # -----------------------------
    # 8) Confusion matrix (val)
    # -----------------------------
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, adj, y, _, _ in val_loader:
            x, adj = x.to(device), adj.to(device)
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
            A = normalize_adj_symmetric(adj)
            logits = model(x, A)
            pred = logits.argmax(1).cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(LABEL2NAME))
    plot_confusion_matrix(cm, labels=[LABEL2NAME[i] for i in range(len(LABEL2NAME))],
                          normalize=True, out_path="conf_matrix.png")

    print("🎉 Done.")

if __name__ == "__main__":
    main()
