#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script with Improved ST-GCN models
Supports 3 model variants: simple, hybrid, full
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset_stgcn import AUSequenceDataset
from model_stgcn_improved import create_stgcn

# -----------------------------
# Configuration
# -----------------------------
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Model selection: 'simple', 'hybrid', 'full'
MODEL_TYPE = 'hybrid'  # Change this to try different models

LABEL2NAME = {0: "ngot", 1: "man", 2: "chua", 3: "dang"}

def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path="train_curves_improved.png"):
    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"📈 Saved training curve → {out_path}")

def compute_confusion_matrix(y_true, y_pred, num_classes=4):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def plot_confusion_matrix(cm, labels, normalize=True, out_path="conf_matrix_improved.png"):
    cm_disp = cm.astype(float)
    if normalize:
        row_sums = cm_disp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_disp = cm_disp / row_sums

    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm_disp, interpolation="nearest", cmap='Blues')
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

def evaluate(model, loader, device, criterion):
    """Evaluate model on given dataloader"""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_true, all_pred = [], []
    
    with torch.no_grad():
        for x, adj, y, _, _ in loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            
            # For ImprovedSTGCN/HybridSTGCN, A should be [N, N]
            if adj.dim() == 3 and adj.size(0) == x.size(0):
                adj = adj[0]  # Use first sample's adj (they're all the same)
            
            logits = model(x, adj)
            loss = criterion(logits, y)
            
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    
    avg_loss = loss_sum / total
    acc = correct / total
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    
    return avg_loss, acc, y_true, y_pred

def main():
    print(f"{'='*60}")
    print(f"Training with MODEL_TYPE = '{MODEL_TYPE}'")
    print(f"{'='*60}\n")
    
    # -----------------------------
    # 1) Load data split
    # -----------------------------
    if not os.path.exists("split.csv"):
        raise SystemExit("❌ split.csv not found. Run make_split.py first.")

    df = pd.read_csv("split.csv")
    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise SystemExit("❌ No train/val data in split.csv")

    # -----------------------------
    # 2) Determine T_fix
    # -----------------------------
    T_fix = int(train_df["T"].median())
    print(f"Using T_fix = {T_fix}")

    # -----------------------------
    # 3) Create DataLoaders
    # -----------------------------
    train_loader = DataLoader(
        AUSequenceDataset(train_df, t_fix=T_fix, fix_mode="center"),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        AUSequenceDataset(val_df, t_fix=T_fix, fix_mode="center"),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # -----------------------------
    # 4) Get data dimensions
    # -----------------------------
    sample_path = train_df.iloc[0]["path"]
    d0 = np.load(sample_path, allow_pickle=True)
    N = d0["graph_seq"].shape[1]  # Number of nodes (15)
    F = d0["graph_seq"].shape[2]  # Number of features (10)
    num_classes = 4
    print(f"Data: N={N} nodes, F={F} features, Classes={num_classes}")

    # -----------------------------
    # 5) Create model
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if MODEL_TYPE == 'simple':
        # Use original SimpleSTGCN
        from model_stgcn import SimpleSTGCN
        model = SimpleSTGCN(in_feats=F, hid=128, num_classes=num_classes).to(device)
    else:
        # Use improved models
        model = create_stgcn(
            model_type=MODEL_TYPE,
            in_channels=F,
            num_classes=num_classes,
            num_nodes=N,
            hid=128,
            dropout=0.3
        ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {num_params:,}\n")

    # -----------------------------
    # 6) Optimizer & Loss
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # -----------------------------
    # 7) Training loop
    # -----------------------------
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15

    for epoch in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        
        for x, adj, y, _, _ in train_loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            
            # For improved models, use single adjacency [N, N]
            if MODEL_TYPE != 'simple':
                if adj.dim() == 3:
                    adj = adj[0]
            else:
                # For SimpleSTGCN, need [B, N, N]
                if adj.dim() == 2:
                    adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
                A = adj + torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
                A = A / (A.sum(dim=-1, keepdim=True).clamp_min(1.0))
                adj = A  # normalized adjacency
            
            # Forward
            if MODEL_TYPE == 'simple':
                logits = model(x, adj)
            else:
                logits = model(x, adj)
            
            loss = criterion(logits, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Stats
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        # ---- Validation ----
        val_loss, val_acc, y_true, y_pred = evaluate(
            model, val_loader, device, criterion
        )

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # LR scheduling
        scheduler.step(val_acc)

        # Print progress
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'model_type': MODEL_TYPE,
            }, f"stgcn_{MODEL_TYPE}_best.pt")
            print(f"  💾 Saved BEST model (val_acc={best_val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

    # Save last checkpoint
    torch.save(model.state_dict(), f"stgcn_{MODEL_TYPE}_last.pt")
    print(f"\n✅ Training completed. Best val_acc: {best_val_acc:.3f}")

    # -----------------------------
    # 8) Plot training curves
    # -----------------------------
    plot_curves(train_losses, val_losses, train_accs, val_accs, 
                out_path=f"train_curves_{MODEL_TYPE}.png")

    # -----------------------------
    # 9) Final confusion matrix on validation set
    # -----------------------------
    # Load best model
    checkpoint = torch.load(f"stgcn_{MODEL_TYPE}_best.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, y_true, y_pred = evaluate(model, val_loader, device, criterion)
    
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    plot_confusion_matrix(cm, 
                         labels=[LABEL2NAME[i] for i in range(num_classes)],
                         normalize=True, 
                         out_path=f"conf_matrix_{MODEL_TYPE}.png")

    print("\n🎉 All done!")
    print(f"Best model saved to: stgcn_{MODEL_TYPE}_best.pt")
    print(f"Training curves: train_curves_{MODEL_TYPE}.png")
    print(f"Confusion matrix: conf_matrix_{MODEL_TYPE}.png")


if __name__ == "__main__":
    main()
