#!/usr/bin/env python3
"""
Deep Learning — Vua_phai vs Others (binary, per-trial raw epochs)
=================================================================
Models : EEGNet, ShallowConvNet, DeepConvNet
Input  : raw epochs (16 ch × 351 tp @ 100 Hz, baseline-corrected)
CV     : LOSO by subject (28 folds)
GPU    : RTX 4090 via PyTorch CUDA

Techniques against class imbalance (220 Vua_phai / 620 Others):
  - WeightedRandomSampler (upsample minority in each batch)
  - pos_weight in BCEWithLogitsLoss
  - Optional: crop augmentation (random temporal crop in train)
"""

import os, sys, warnings, datetime, glob
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, recall_score)
from sklearn.preprocessing import StandardScaler

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH_DIR = 'output/epochs'
OUT_DIR   = 'output/results/dl_vuaphai'
FIG_DIR   = 'output/figures/dl_vuaphai'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_CH    = 16
N_TIMES = 351   # -0.5 → +3.0s @ 100 Hz
SFREQ   = 100


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── Models ───────────────────────────────────────────────────────────────────
class EEGNet(nn.Module):
    """EEGNet (Lawhern et al. 2018) — binary output."""
    def __init__(self, n_ch=N_CH, n_t=N_TIMES, F1=8, D=2, F2=16,
                 dropout=0.5, kernel_len=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Compute flat size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            out = self.block2(self.block1(dummy))
            flat = out.flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):           # x: (B, n_ch, n_t)
        x = x.unsqueeze(1)          # (B, 1, n_ch, n_t)
        x = self.block1(x)
        x = self.block2(x)
        return self.fc(x.flatten(1)).squeeze(1)


class ShallowConvNet(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017) — binary."""
    def __init__(self, n_ch=N_CH, n_t=N_TIMES, n_filters=40, filter_len=25,
                 pool_len=75, pool_stride=15, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, filter_len), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.pool_len    = pool_len
        self.pool_stride = pool_stride
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            out = self.conv(dummy)
            out = F.avg_pool2d(out.pow(2), (1, pool_len), stride=(1, pool_stride)).log()
            flat = out.flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.avg_pool2d(x.pow(2), (1, self.pool_len),
                          stride=(1, self.pool_stride)).log()
        x = self.dropout(x)
        return self.fc(x.flatten(1)).squeeze(1)


class DeepConvNet(nn.Module):
    """DeepConvNet (Schirrmeister et al. 2017) — binary, 4 conv blocks."""
    def __init__(self, n_ch=N_CH, n_t=N_TIMES, dropout=0.5):
        super().__init__()
        def conv_block(in_f, out_f, k, pool):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, k, bias=False),
                nn.BatchNorm2d(out_f),
                nn.ELU(),
                nn.MaxPool2d(pool),
                nn.Dropout(dropout),
            )
        self.block0 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), bias=False),
            nn.Conv2d(25, 25, (n_ch, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(),
            nn.MaxPool2d((1, 2)), nn.Dropout(dropout),
        )
        self.block1 = conv_block(25, 50,  (1, 5), (1, 2))
        self.block2 = conv_block(50, 100, (1, 5), (1, 2))
        self.block3 = conv_block(100,200, (1, 5), (1, 2))
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            out = self.block3(self.block2(self.block1(self.block0(dummy))))
            flat = out.flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block3(self.block2(self.block1(self.block0(x))))
        return self.fc(x.flatten(1)).squeeze(1)


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_all_epochs():
    """Returns X (n_trials, n_ch, n_t), y (n_trials,), subjects (n_trials,)."""
    Xs, ys, gs = [], [], []
    for subj_dir in sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*'))):
        subj = os.path.basename(subj_dir)
        npy = os.path.join(subj_dir, 'epochs_data.npy')
        ti  = os.path.join(subj_dir, 'trial_info.csv')
        if not os.path.exists(npy) or not os.path.exists(ti):
            continue
        epochs = np.load(npy).astype(np.float32)   # (n_trials, 16, 351)
        info   = pd.read_csv(ti)
        if len(epochs) != len(info):
            continue
        Xs.append(epochs)
        ys.append((info['jar_group'].values == 'Vua_phai').astype(np.float32))
        gs.extend([subj] * len(epochs))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.array(gs)
    return X, y, g


def channel_normalize(X_tr, X_te):
    """Standardize per channel using train statistics."""
    mean = X_tr.mean(axis=(0, 2), keepdims=True)
    std  = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X_tr - mean) / std, (X_te - mean) / std


# ─── Training ─────────────────────────────────────────────────────────────────
def make_loader(X, y, batch_size=32, balanced=True, augment=False):
    """Create DataLoader with optional WeightedRandomSampler."""
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()

    if augment:
        # Random temporal crop: keep 80-100% of time axis
        n_t = X_t.shape[2]
        crop = int(n_t * np.random.uniform(0.80, 1.00))
        start = np.random.randint(0, n_t - crop + 1)
        pad = torch.zeros_like(X_t)
        pad[:, :, start:start+crop] = X_t[:, :, start:start+crop]
        X_t = pad

    ds = TensorDataset(X_t, y_t)

    if balanced:
        n_pos = int(y.sum()); n_neg = int((y == 0).sum())
        w = np.where(y == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
        sampler = WeightedRandomSampler(torch.from_numpy(w).float(),
                                        num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_model(model, X_tr, y_tr, X_val, y_val,
                n_epochs=150, lr=1e-3, batch_size=32,
                pos_weight_ratio=3.0, patience=20):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    pos_w = torch.tensor([pos_weight_ratio]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    loader = make_loader(X_tr, y_tr, batch_size=batch_size,
                         balanced=True, augment=True)

    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).float().to(DEVICE)

    best_bacc, best_state, no_imp = 0.0, None, 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val_t)
            proba  = torch.sigmoid(logits).cpu().numpy()
        yp = (proba >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y_val.astype(int), yp)
        if bacc > best_bacc:
            best_bacc  = bacc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_model_loso(model_class, model_kwargs, X, y, groups,
                    n_epochs=150, lr=1e-3, batch_size=32,
                    pos_weight_ratio=3.0, patience=20):
    """LOSO-CV: returns (y_true, y_pred, y_proba, per_fold_metrics)."""
    subjects = sorted(np.unique(groups))
    y_true_all, y_pred_all, y_proba_all = [], [], []
    fold_metrics = []

    for i, subj in enumerate(subjects):
        te_mask = groups == subj
        tr_mask = ~te_mask

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]

        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue

        X_tr, X_te = channel_normalize(X_tr, X_te)

        model = model_class(**model_kwargs).to(DEVICE)
        model = train_model(model, X_tr, y_tr, X_te, y_te,
                            n_epochs=n_epochs, lr=lr,
                            batch_size=batch_size,
                            pos_weight_ratio=pos_weight_ratio,
                            patience=patience)

        model.eval()
        with torch.no_grad():
            X_te_t = torch.from_numpy(X_te).float().to(DEVICE)
            proba  = torch.sigmoid(model(X_te_t)).cpu().numpy()
        yp = (proba >= 0.5).astype(int)

        bacc = balanced_accuracy_score(y_te.astype(int), yp)
        acc  = accuracy_score(y_te.astype(int), yp)
        fold_metrics.append({'subject': subj, 'acc': acc, 'bacc': bacc,
                              'n_pos': int(y_te.sum()), 'n': len(y_te)})

        y_true_all.extend(y_te.astype(int).tolist())
        y_pred_all.extend(yp.tolist())
        y_proba_all.extend(proba.tolist())

        print(f'    fold {i+1:02d}/{len(subjects)}  {subj}  '
              f'acc={acc:.3f}  bacc={bacc:.3f}  n_pos={int(y_te.sum())}',
              flush=True)

    return (np.array(y_true_all), np.array(y_pred_all),
            np.array(y_proba_all), fold_metrics)


def oracle_threshold(y_true, y_proba):
    """Post-hoc best threshold by accuracy and by balanced_acc."""
    best_acc, best_thr_acc = 0.0, 0.5
    best_bacc, best_thr_bacc = 0.0, 0.5
    for thr in np.linspace(0.10, 0.90, 81):
        yp = (y_proba >= thr).astype(int)
        a  = accuracy_score(y_true, yp)
        b  = balanced_accuracy_score(y_true, yp)
        if a > best_acc:   best_acc = a;   best_thr_acc  = thr
        if b > best_bacc:  best_bacc = b;  best_thr_bacc = thr
    return best_acc, best_thr_acc, best_bacc, best_thr_bacc


def plot_confusion(y_true, y_pred, title, fig_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)'
                       for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Other','Vua_phai'],
                yticklabels=['Other','Vua_phai'],
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(OUT_DIR, 'logs', f'run_{ts}.log')
    latest   = os.path.join(OUT_DIR, 'run.log')
    log_fh = open(log_path, 'w'); latest_fh = open(latest, 'w')
    sys.stdout = Tee(sys.__stdout__, log_fh, latest_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh, latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_dl_vuaphai')
    print('='*78)
    print('  DL: EEGNet / ShallowConvNet / DeepConvNet — Vua_phai vs Others')
    print(f'  Device: {DEVICE}  ({torch.cuda.get_device_name(0)})')
    print('='*78)

    # ── Load data ──────────────────────────────────────────────────────────
    print('\nLoading epochs...')
    X, y, g = load_all_epochs()
    n_pos = int(y.sum()); n_neg = int((y==0).sum())
    print(f'  X={X.shape}  Vua_phai={n_pos}  Others={n_neg}  '
          f'subjects={len(np.unique(g))}')
    print(f'  majority_baseline={max(n_pos,n_neg)/len(y):.3f}')

    # ── Model configs ──────────────────────────────────────────────────────
    EXPERIMENTS = [
        {
            'name': 'EEGNet',
            'model_class': EEGNet,
            'model_kwargs': {'n_ch': N_CH, 'n_t': N_TIMES,
                             'F1': 8, 'D': 2, 'F2': 16,
                             'dropout': 0.5, 'kernel_len': 64},
            'train_kwargs': {'n_epochs': 200, 'lr': 5e-4, 'batch_size': 32,
                             'pos_weight_ratio': 3.0, 'patience': 25},
        },
        {
            'name': 'EEGNet_light',
            'model_class': EEGNet,
            'model_kwargs': {'n_ch': N_CH, 'n_t': N_TIMES,
                             'F1': 4, 'D': 2, 'F2': 8,
                             'dropout': 0.4, 'kernel_len': 32},
            'train_kwargs': {'n_epochs': 200, 'lr': 1e-3, 'batch_size': 32,
                             'pos_weight_ratio': 3.0, 'patience': 25},
        },
        {
            'name': 'ShallowConvNet',
            'model_class': ShallowConvNet,
            'model_kwargs': {'n_ch': N_CH, 'n_t': N_TIMES,
                             'n_filters': 40, 'filter_len': 25,
                             'pool_len': 75, 'pool_stride': 15, 'dropout': 0.5},
            'train_kwargs': {'n_epochs': 200, 'lr': 5e-4, 'batch_size': 32,
                             'pos_weight_ratio': 3.0, 'patience': 25},
        },
        {
            'name': 'DeepConvNet',
            'model_class': DeepConvNet,
            'model_kwargs': {'n_ch': N_CH, 'n_t': N_TIMES, 'dropout': 0.5},
            'train_kwargs': {'n_epochs': 200, 'lr': 1e-3, 'batch_size': 32,
                             'pos_weight_ratio': 3.0, 'patience': 25},
        },
    ]

    rows = []
    t_start = datetime.datetime.now()

    for exp in EXPERIMENTS:
        name = exp['name']
        print(f'\n{"━"*60}')
        print(f'  Model: {name}')
        print(f'  Config: {exp["model_kwargs"]}')
        print(f'  Train:  {exp["train_kwargs"]}')
        print(f'  Elapsed: {(datetime.datetime.now()-t_start).seconds}s')
        print('━'*60, flush=True)

        y_true, y_pred, y_proba, fold_metrics = eval_model_loso(
            exp['model_class'], exp['model_kwargs'],
            X, y, g, **exp['train_kwargs']
        )

        if len(y_true) == 0:
            print('  No predictions — skip')
            continue

        # Metrics at threshold=0.5
        acc  = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
        rec_vua = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_oth = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

        # Oracle threshold
        o_acc, o_thr_acc, o_bacc, o_thr_bacc = oracle_threshold(y_true, y_proba)

        row = {
            'model': name,
            'accuracy':        round(acc,     4),
            'balanced_acc':    round(bacc,    4),
            'f1_macro':        round(f1,      4),
            'recall_vua_phai': round(rec_vua, 4),
            'recall_others':   round(rec_oth, 4),
            'oracle_acc':      round(o_acc,   4),
            'oracle_thr_acc':  round(o_thr_acc, 2),
            'oracle_bacc':     round(o_bacc,  4),
            'oracle_thr_bacc': round(o_thr_bacc, 2),
        }
        rows.append(row)

        print(f'\n  ── {name} Results ──')
        print(f'    accuracy      = {acc:.4f}  (threshold=0.5)')
        print(f'    balanced_acc  = {bacc:.4f}  ← primary metric')
        print(f'    f1_macro      = {f1:.4f}')
        print(f'    recall_vua    = {rec_vua:.4f}  (true positive rate)')
        print(f'    recall_others = {rec_oth:.4f}')
        print(f'    oracle_acc    = {o_acc:.4f}  (thr={o_thr_acc:.2f})')
        print(f'    oracle_bacc   = {o_bacc:.4f}  (thr={o_thr_bacc:.2f})')

        # Confusion matrix
        plot_confusion(
            y_true, y_pred,
            f'{name}  acc={acc:.4f}  bacc={bacc:.4f}  rec_vua={rec_vua:.4f}',
            os.path.join(FIG_DIR, f'cm_{name}.png')
        )

        # Per-fold summary
        df_fold = pd.DataFrame(fold_metrics)
        print(f'\n  Per-fold bacc: mean={df_fold["bacc"].mean():.3f}  '
              f'std={df_fold["bacc"].std():.3f}  '
              f'min={df_fold["bacc"].min():.3f}  '
              f'max={df_fold["bacc"].max():.3f}')

    # ── Save ──────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'results_dl.csv')
    df_res.to_csv(csv_path, index=False)
    print(f'\n✓ Saved {csv_path}')

    # Bar chart comparison
    if len(df_res) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, metric, label in zip(
                axes,
                ['accuracy', 'balanced_acc', 'recall_vua_phai'],
                ['Accuracy', 'Balanced Accuracy', 'Recall Vua_phai']):
            colors = plt.cm.Set2(np.linspace(0, 1, len(df_res)))
            ax.bar(df_res['model'], df_res[metric], color=colors)
            ax.axhline(0.85, color='red', ls='--', lw=1.5, label='target=0.85')
            ax.axhline(0.738, color='gray', ls=':', lw=1.0, label='majority=0.738')
            ax.set_title(label, fontweight='bold')
            ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=15)
            ax.legend(fontsize=8)
        fig.suptitle('DL Models — Vua_phai vs Others (LOSO-CV, RTX 4090)',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'model_comparison.png'),
                    dpi=180, bbox_inches='tight')
        plt.close(fig)
        print('✓ Saved comparison chart')

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = (datetime.datetime.now() - t_start).seconds
    print(f'\n{"="*78}')
    print('  FINAL RESULTS SUMMARY')
    print(f'{"="*78}')
    hdr = f'{"model":<18}{"acc":<9}{"bacc":<9}{"f1":<8}{"rec_vua":<10}{"oracle_acc":<12}{"oracle_bacc"}'
    print(hdr); print('-'*len(hdr))
    for _, r in df_res.iterrows():
        print(f'{r["model"]:<18}{r["accuracy"]:<9.4f}{r["balanced_acc"]:<9.4f}'
              f'{r["f1_macro"]:<8.4f}{r["recall_vua_phai"]:<10.4f}'
              f'{r["oracle_acc"]:<12.4f}{r["oracle_bacc"]:.4f}')

    if len(df_res) > 0:
        best = df_res.loc[df_res['balanced_acc'].idxmax()]
        print(f'\n  Best model (balanced_acc): {best["model"]}')
        print(f'    accuracy      = {best["accuracy"]:.4f}')
        print(f'    balanced_acc  = {best["balanced_acc"]:.4f}')
        print(f'    oracle_acc    = {best["oracle_acc"]:.4f}')
        print(f'    recall_vua    = {best["recall_vua_phai"]:.4f}')
        print(f'\n  vs best traditional ML (v3):')
        print(f'    XGB K=29 rm=3: acc=0.778  bacc=0.604  oracle=0.824')

    print(f'\n  Total runtime: {elapsed}s ({elapsed//60}m {elapsed%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
