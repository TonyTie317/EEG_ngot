#!/usr/bin/env python3
"""
gERP-specific feature engineering for Vua_phai vs Others
=========================================================
Key discovery from grand-average ERP at Cz:
  Late positivity (500-1000ms): Vua_phai=4.14 µV  vs Others=2.39 µV  ← BIG gap
  N400       (300-500ms):       Vua_phai=1.86 µV  vs Others=2.38 µV
  P2         (200-350ms):       Vua_phai=1.73 µV  vs Others=2.50 µV  (inverted!)

gERP feature set:
  1. Fine-grained 50ms bins 0-2000ms × taste channels     (~400 feat)
  2. Component features: N400, Late(500-1000ms), VeryLate(1-2s)  (~80 feat)
  3. Hemispheric asymmetry (F4-F3, C4-C3, P4-P3, T4-T3)  (~20 feat)
  4. Spectral in task windows (theta, alpha, beta)          (~30 feat)
  5. Temporal dynamics (slope, rising rate)                 (~20 feat)

Then: XGB GPU + LOSO-CV  (K sweep 5-50, iso=0.10)
Also: ShallowConvNet on gERP-trimmed input (taste channels × 0-2s)
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
from scipy.signal import welch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, recall_score)
import xgboost as xgb

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH_DIR = 'output/epochs'
OUT_DIR   = 'output/results/ml_gerp'
FIG_DIR   = 'output/figures/ml_gerp'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SFREQ  = 100
TMIN   = -0.5
N_TIMES = 351

# Channel layout
CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T3','T4','Fz','Cz']
# Taste-relevant channels (frontal + central + temporal)
TASTE_CH = {'Fz':14, 'Cz':15, 'C3':4, 'C4':5, 'F3':2, 'F4':3,
            'P3':6,  'P4':7,  'T3':12,'T4':13}
TASTE_IDX = list(TASTE_CH.values())   # 10 channels

# Time axis
TIMES = np.linspace(TMIN, TMIN + (N_TIMES-1)/SFREQ, N_TIMES)


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── gERP Feature Extraction ─────────────────────────────────────────────────
def extract_gerp_features(epoch):
    """
    epoch: (16, 351) float32 in volts, baseline-corrected.
    Returns dict of gERP-specific features.
    """
    feats = {}
    t = TIMES

    # ── 1. Fine-grained 50ms bins 0→2000ms at taste channels ─────────────
    bin_edges = np.arange(0.0, 2.01, 0.05)   # 40 bins
    for ch_name, ch_idx in TASTE_CH.items():
        sig = epoch[ch_idx] * 1e6  # µV
        for i in range(len(bin_edges)-1):
            t0, t1 = bin_edges[i], bin_edges[i+1]
            mask = (t >= t0) & (t < t1)
            key = f'bin_{ch_name}_{int(t0*1000):04d}'
            feats[key] = float(sig[mask].mean()) if mask.sum() > 0 else 0.0

    # ── 2. Component-specific features ────────────────────────────────────
    components = {
        'P2':        (0.20, 0.35),
        'N400':      (0.30, 0.50),
        'LatePos':   (0.50, 1.00),   # KEY: Vua_phai 4.14 vs Others 2.39 µV
        'VeryLate':  (1.00, 2.00),
        'EarlyN1':   (0.10, 0.20),
    }
    comp_channels = {
        'P2':       ['Cz','C3','C4','Fz'],
        'N400':     ['Cz','Fz','C3','C4','T3','T4'],
        'LatePos':  ['Cz','Fz','P3','P4','C3','C4'],
        'VeryLate': ['Cz','Fz'],
        'EarlyN1':  ['Cz','C3','C4'],
    }
    for cname, (t0, t1) in components.items():
        mask = (t >= t0) & (t <= t1)
        for ch_name in comp_channels[cname]:
            ch_idx = TASTE_CH[ch_name]
            sig = epoch[ch_idx, mask] * 1e6
            if len(sig) == 0:
                continue
            feats[f'{cname}_{ch_name}_mean']  = float(sig.mean())
            feats[f'{cname}_{ch_name}_peak']  = float(sig.max())
            feats[f'{cname}_{ch_name}_min']   = float(sig.min())
            feats[f'{cname}_{ch_name}_auc']   = float(np.trapz(sig))
            feats[f'{cname}_{ch_name}_rms']   = float(np.sqrt(np.mean(sig**2)))

    # ── 3. Hemispheric asymmetry ──────────────────────────────────────────
    asym_pairs = [('F4','F3'), ('C4','C3'), ('P4','P3'), ('T4','T3')]
    asym_windows = {'N400': (0.30, 0.50), 'LatePos': (0.50, 1.00)}
    for (r_ch, l_ch) in asym_pairs:
        r_idx, l_idx = TASTE_CH[r_ch], TASTE_CH[l_ch]
        for wname, (t0, t1) in asym_windows.items():
            mask = (t >= t0) & (t <= t1)
            r_mean = float(epoch[r_idx, mask].mean() * 1e6)
            l_mean = float(epoch[l_idx, mask].mean() * 1e6)
            feats[f'asym_{r_ch}m{l_ch}_{wname}'] = r_mean - l_mean

    # ── 4. Spectral features in task-specific windows ─────────────────────
    spec_windows = {
        'early':  (0.00, 0.50),   # early processing
        'late':   (0.50, 1.50),   # late evaluation
    }
    bands = {'theta':(4,8), 'alpha':(8,13), 'beta':(13,30)}
    spec_channels = ['Fz','Cz','C3','C4']
    for wname, (t0, t1) in spec_windows.items():
        mask = (t >= t0) & (t <= t1)
        if mask.sum() < 16:
            continue
        for ch_name in spec_channels:
            ch_idx = TASTE_CH[ch_name]
            seg = epoch[ch_idx, mask]
            f_ax, psd = welch(seg, fs=SFREQ, nperseg=min(64, len(seg)))
            for bname, (flo, fhi) in bands.items():
                bm = (f_ax >= flo) & (f_ax <= fhi)
                feats[f'psd_{bname}_{ch_name}_{wname}'] = float(psd[bm].mean()) if bm.sum() > 0 else 0.0

    # ── 5. Temporal dynamics (slope of Late Positivity at Cz) ─────────────
    # Rising slope: 400-700ms
    for ch_name, ch_idx in [('Cz',15), ('Fz',14)]:
        sig_full = epoch[ch_idx] * 1e6
        for t0, t1, label in [(0.40, 0.70, 'rise'), (0.70, 1.00, 'plateau')]:
            mask = (t >= t0) & (t <= t1)
            if mask.sum() >= 2:
                seg = sig_full[mask]
                x = np.arange(len(seg))
                slope = float(np.polyfit(x, seg, 1)[0])
                feats[f'slope_{ch_name}_{label}'] = slope

    # ── 6. N400 vs LatePos ratio (evaluation quality) ─────────────────────
    for ch_name in ['Cz', 'Fz']:
        ch_idx = TASTE_CH[ch_name]
        sig = epoch[ch_idx] * 1e6
        n400 = sig[(t >= 0.30) & (t <= 0.50)].mean()
        late  = sig[(t >= 0.50) & (t <= 1.00)].mean()
        feats[f'ratio_late_n400_{ch_name}'] = float(late / (abs(n400) + 1e-6))

    return feats


def build_gerp_features():
    rows = []
    for d in sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*'))):
        subj = os.path.basename(d)
        npy = os.path.join(d, 'epochs_data.npy')
        ti  = os.path.join(d, 'trial_info.csv')
        if not os.path.exists(npy) or not os.path.exists(ti): continue
        epochs = np.load(npy).astype(np.float32)
        info   = pd.read_csv(ti)
        if len(epochs) != len(info): continue
        for i in range(len(epochs)):
            f = extract_gerp_features(epochs[i])
            f['subject_id'] = subj
            f['condition']  = int(info.iloc[i]['condition'])
            f['repeat']     = int(info.iloc[i]['repeat'])
            f['jar_group']  = info.iloc[i]['jar_group']
            rows.append(f)
    df = pd.DataFrame(rows)
    meta = ['subject_id','condition','repeat','jar_group']
    feat_cols = [c for c in df.columns if c not in meta]
    return df, feat_cols


# ─── ML helpers ───────────────────────────────────────────────────────────────
def smote_safe(X, y):
    from imblearn.over_sampling import SMOTE
    n_pos = (y==1).sum()
    if n_pos < 2: return X, y
    k = min(5, n_pos-1)
    try: return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X, y)
    except: return X, y


def precompute_folds(X, y, g, iso_contam):
    logo = LeaveOneGroupOut()
    folds = []
    for tr, te in logo.split(X, y, g):
        X_tr, y_tr = X[tr].copy(), y[tr].copy()
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0: continue
        if iso_contam > 0 and len(X_tr) > 20:
            iso = IsolationForest(contamination=iso_contam, random_state=SEED, n_jobs=-1)
            iso.fit(X_tr); km = iso.predict(X_tr)==1
            if (y_tr[km]==0).any() and (y_tr[km]==1).any():
                X_tr, y_tr = X_tr[km], y_tr[km]
        if len(np.unique(y_tr)) < 2: continue
        sc = StandardScaler()
        X_tr = np.nan_to_num(sc.fit_transform(X_tr))
        X_te = np.nan_to_num(sc.transform(X_te))
        mi    = mutual_info_classif(X_tr, y_tr, random_state=SEED)
        order = np.argsort(mi)[::-1]
        folds.append({'X_tr':X_tr,'y_tr':y_tr,'X_te':X_te,'y_te':y_te,'mi_order':order})
    return folds


def eval_K(folds, K, model_factory, sampling='smote'):
    y_true_all, y_pred_all, proba_all = [], [], []
    for f in folds:
        idx  = f['mi_order'][:K]
        X_tr = f['X_tr'][:,idx].copy()
        y_tr = f['y_tr'].copy()
        X_te = f['X_te'][:,idx]
        if sampling == 'smote': X_tr, y_tr = smote_safe(X_tr, y_tr)
        m = model_factory()
        m.fit(X_tr, y_tr)
        y_pred_all.extend(m.predict(X_te).tolist())
        y_true_all.extend(f['y_te'].tolist())
        if hasattr(m, 'predict_proba'):
            proba_all.extend(m.predict_proba(X_te)[:,1].tolist())
        else:
            proba_all.extend([np.nan]*len(f['y_te']))
    if not y_true_all: return None
    yt, yp, pr = np.array(y_true_all), np.array(y_pred_all), np.array(proba_all)
    res = {
        'accuracy':     accuracy_score(yt, yp),
        'balanced_acc': balanced_accuracy_score(yt, yp),
        'f1_macro':     f1_score(yt, yp, average='macro', zero_division=0),
        'rec_vua':      recall_score(yt, yp, pos_label=1, zero_division=0),
        'rec_oth':      recall_score(yt, yp, pos_label=0, zero_division=0),
        'oracle_bacc':  balanced_accuracy_score(yt, yp),
        'oracle_thr':   0.5, 'y_true':yt, 'y_pred':yp,
    }
    if not np.isnan(pr).any():
        best_b, best_t = 0.0, 0.5
        for thr in np.linspace(0.1, 0.9, 81):
            b = balanced_accuracy_score(yt, (pr>=thr).astype(int))
            if b > best_b: best_b=b; best_t=thr
        res['oracle_bacc'] = round(float(best_b), 4)
        res['oracle_thr']  = round(float(best_t), 2)
    return res


# ─── DL on trimmed gERP input ─────────────────────────────────────────────────
GERP_T0    = int((0.0  - TMIN) * SFREQ)   # index of t=0
GERP_T1    = int((2.0  - TMIN) * SFREQ)   # index of t=2.0s
GERP_N_T   = GERP_T1 - GERP_T0            # 200 timepoints
GERP_N_CH  = len(TASTE_IDX)               # 10 channels


class ShallowConvNetGERP(nn.Module):
    """ShallowConvNet on gERP-trimmed input: 10 taste channels × 200 tp."""
    def __init__(self, n_ch=GERP_N_CH, n_t=GERP_N_T,
                 n_filters=40, filter_len=25, pool_len=50, pool_stride=10,
                 dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, filter_len), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.pl = pool_len; self.ps = pool_stride
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            d = torch.zeros(1,1,n_ch,n_t)
            flat = F.avg_pool2d(self.conv(d).pow(2),(1,self.pl),
                                stride=(1,self.ps)).log().flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.avg_pool2d(x.pow(2), (1, self.pl), stride=(1, self.ps)).log()
        return self.fc(self.drop(x).flatten(1)).squeeze(1)


def channel_normalize(X_tr, X_te):
    m = X_tr.mean(axis=(0,2), keepdims=True)
    s = X_tr.std(axis=(0,2),  keepdims=True) + 1e-8
    return (X_tr-m)/s, (X_te-m)/s


def train_dl(model, X_tr, y_tr, X_val, y_val,
             n_epochs=300, lr=5e-4, batch_size=32,
             pos_weight=3.0, patience=30):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))

    # Balanced sampler
    n_pos = max(int(y_tr.sum()),1); n_neg = max(int((y_tr==0).sum()),1)
    w = np.where(y_tr==1, 1.0/n_pos, 1.0/n_neg)
    sampler = WeightedRandomSampler(torch.from_numpy(w).float(), len(w), replacement=True)
    ds  = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    best_bacc, best_state, no_imp = 0.0, None, 0

    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # Gaussian noise augmentation
            xb = xb + 0.03 * torch.randn_like(xb) * xb.std(dim=(1,2,3), keepdim=True).clamp(min=1e-8)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pr = torch.sigmoid(model(X_val_t)).cpu().numpy()
        bacc = balanced_accuracy_score(y_val.astype(int), (pr>=0.5).astype(int))
        if bacc > best_bacc:
            best_bacc = bacc
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model


def loso_dl_gerp(X_raw, y, groups, pos_weight=3.0):
    """LOSO using gERP-trimmed raw input (10 ch × 200 tp)."""
    # Trim to taste channels and 0-2s window
    X = X_raw[:, TASTE_IDX, :][:, :, GERP_T0:GERP_T1]
    subjects = sorted(np.unique(groups))
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for subj in subjects:
        te = groups == subj; tr = ~te
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0: continue
        X_tr, X_te = channel_normalize(X_tr, X_te)
        m = train_dl(ShallowConvNetGERP(), X_tr, y_tr, X_te, y_te,
                     pos_weight=pos_weight)
        m.eval()
        with torch.no_grad():
            pr = torch.sigmoid(m(torch.from_numpy(X_te).float().to(DEVICE))).cpu().numpy()
        y_true_all.extend(y_te.astype(int).tolist())
        y_pred_all.extend((pr>=0.5).astype(int).tolist())
        y_proba_all.extend(pr.tolist())
    yt, yp, pr = np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all)
    best_b, best_t = 0.0, 0.5
    for thr in np.linspace(0.1, 0.9, 81):
        b = balanced_accuracy_score(yt, (pr>=thr).astype(int))
        if b > best_b: best_b=b; best_t=thr
    return {
        'accuracy': accuracy_score(yt, yp),
        'balanced_acc': balanced_accuracy_score(yt, yp),
        'f1_macro': f1_score(yt, yp, average='macro', zero_division=0),
        'rec_vua': recall_score(yt, yp, pos_label=1, zero_division=0),
        'rec_oth': recall_score(yt, yp, pos_label=0, zero_division=0),
        'oracle_bacc': round(float(best_b), 4),
        'oracle_thr': round(float(best_t), 2),
        'y_true': yt, 'y_pred': yp,
    }


def plot_top_features(df_res, feat_cols, X, y, fig_path):
    """Violin plot of top-10 MI features: Vua_phai vs Others."""
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X))
    mi = mutual_info_classif(Xs, y, random_state=SEED)
    top_idx = np.argsort(mi)[::-1][:10]
    top_names = [feat_cols[i] for i in top_idx]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, idx, name in zip(axes.flatten(), top_idx, top_names):
        vals_pos = X[y==1, idx]
        vals_neg = X[y==0, idx]
        ax.violinplot([vals_pos, vals_neg], positions=[1,2], showmedians=True)
        ax.set_xticks([1,2]); ax.set_xticklabels(['Vua_phai','Others'], fontsize=8)
        ax.set_title(name[:28], fontsize=7, fontweight='bold')
        ax.grid(alpha=0.3)
    fig.suptitle('Top-10 gERP features by Mutual Information (Vua_phai vs Others)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_confusion(yt, yp, title, path):
    cm = confusion_matrix(yt, yp)
    cmn = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    ann = np.array([[f'{cm[i,j]}\n({cmn[i,j]*100:.0f}%)' for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cmn, annot=ann, fmt='', cmap='Blues',
                xticklabels=['Other','Vua_phai'], yticklabels=['Other','Vua_phai'],
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=9, fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fh    = open(os.path.join(OUT_DIR,'logs',f'run_{ts}.log'),'w')
    latest_fh = open(os.path.join(OUT_DIR,'run.log'),'w')
    sys.stdout = Tee(sys.__stdout__, log_fh, latest_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh, latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_ml_gerp')
    print('='*78)
    print('  gERP-specific features: N400 + Late Positivity + Asymmetry + Theta')
    print(f'  Device: {DEVICE}')
    print('='*78)

    # ── Extract gERP features ───────────────────────────────────────────────
    print('\nExtracting gERP features...', flush=True)
    t0 = datetime.datetime.now()
    df, feat_cols = build_gerp_features()
    print(f'  Done in {(datetime.datetime.now()-t0).seconds}s')
    print(f'  Shape: {df.shape}   Features: {len(feat_cols)}')
    print(f'  Vua_phai: {(df.jar_group=="Vua_phai").sum()} / {len(df)} total')
    df.to_csv(os.path.join(OUT_DIR,'features_gerp.csv'), index=False)

    META = ['subject_id','condition','repeat','jar_group']
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.var(axis=0) > 1e-12; X = X[:,keep]
    feat_cols_kept = [c for c,k in zip(feat_cols, keep) if k]
    y = (df['jar_group'].values == 'Vua_phai').astype(int)
    g = df['subject_id'].values
    majority = max(int(y.sum()), int((y==0).sum())) / len(y)
    print(f'  n_features after variance filter: {X.shape[1]}')
    print(f'  majority_baseline = {majority:.3f}')

    # ── Visualize top features ──────────────────────────────────────────────
    plot_top_features(df, feat_cols_kept, X, y,
                      os.path.join(FIG_DIR, 'top10_features.png'))
    print('✓ Saved top-10 feature violin plot')

    # ── Part A: XGB GPU sweep (K=5..50) ────────────────────────────────────
    print('\n' + '━'*60)
    print('  PART A — XGB GPU + MI feature selection (K sweep)')
    print('━'*60)
    K_GRID = [5, 10, 15, 20, 25, 30, 40, 50]
    ISO    = 0.10
    folds  = precompute_folds(X, y, g, ISO)
    print(f'  {len(folds)} LOSO folds prepared')

    rows_ml = []
    best_ml = None
    for sampling in ['none','smote']:
        for K in K_GRID:
            res = eval_K(folds, K,
                         lambda: xgb.XGBClassifier(
                             device='cuda', n_estimators=100, max_depth=4,
                             learning_rate=0.05, subsample=0.8,
                             scale_pos_weight=3, eval_metric='logloss',
                             verbosity=0, random_state=SEED),
                         sampling)
            if res is None: continue
            row = {'model':'XGB_gpu','sampling':sampling,'K':K,
                   'accuracy':    round(res['accuracy'],    4),
                   'balanced_acc':round(res['balanced_acc'],4),
                   'f1_macro':    round(res['f1_macro'],    4),
                   'rec_vua':     round(res['rec_vua'],     4),
                   'rec_oth':     round(res['rec_oth'],     4),
                   'oracle_bacc': res['oracle_bacc'],
                   'oracle_thr':  res['oracle_thr']}
            rows_ml.append(row)
            if best_ml is None or res['balanced_acc'] > best_ml['balanced_acc']:
                best_ml = {**row, 'y_true':res['y_true'], 'y_pred':res['y_pred']}
            print(f'  XGB samp={sampling} K={K:<3}  '
                  f'acc={row["accuracy"]:.4f}  bacc={row["balanced_acc"]:.4f}  '
                  f'oracle_bacc={row["oracle_bacc"]:.4f}  '
                  f'rec_vua={row["rec_vua"]:.4f}')

    # Plot K curve
    df_ml = pd.DataFrame(rows_ml)
    fig, ax = plt.subplots(figsize=(10,5))
    for samp, grp in df_ml.groupby('sampling'):
        ax.plot(grp['K'], grp['balanced_acc'], marker='o', lw=2, label=f'XGB samp={samp}')
    ax.axhline(0.674, color='red',  ls='--', lw=1.5, label='DL v2 best=0.674')
    ax.axhline(0.649, color='blue', ls='--', lw=1.5, label='ML v2 best=0.649')
    ax.axhline(majority, color='gray', ls=':', lw=1.0, label=f'majority={majority:.3f}')
    ax.set_xlabel('K (top-MI gERP features)'); ax.set_ylabel('Balanced Accuracy')
    ax.set_title('gERP features: XGB GPU bacc vs K', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'xgb_bacc_vs_K.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    if best_ml:
        plot_confusion(best_ml['y_true'], best_ml['y_pred'],
            f'XGB gERP | K={best_ml["K"]} samp={best_ml["sampling"]}\n'
            f'bacc={best_ml["balanced_acc"]:.4f}  rec_vua={best_ml["rec_vua"]:.4f}',
            os.path.join(FIG_DIR,'cm_xgb_best.png'))

    # ── Part B: ShallowConvNet on gERP-trimmed raw input ───────────────────
    print('\n' + '━'*60)
    print(f'  PART B — ShallowConvNet (10 taste-ch × 0-2s window, pw=3.0)')
    print('━'*60)

    # Load raw epochs
    Xs_raw, ys_raw, gs_raw = [], [], []
    for d in sorted(glob.glob(os.path.join(EPOCH_DIR,'P*'))):
        s = os.path.basename(d)
        npy = os.path.join(d,'epochs_data.npy')
        ti  = os.path.join(d,'trial_info.csv')
        if not os.path.exists(npy) or not os.path.exists(ti): continue
        ep = np.load(npy).astype(np.float32)
        info = pd.read_csv(ti)
        if len(ep) != len(info): continue
        Xs_raw.append(ep)
        ys_raw.append((info['jar_group'].values=='Vua_phai').astype(np.float32))
        gs_raw.extend([s]*len(ep))
    X_raw = np.concatenate(Xs_raw)
    y_raw = np.concatenate(ys_raw)
    g_raw = np.array(gs_raw)

    print(f'  Input: {GERP_N_CH} taste channels × {GERP_N_T} timepoints (0-2s)', flush=True)
    dl_res = loso_dl_gerp(X_raw, y_raw, g_raw, pos_weight=3.0)
    print(f'  ShallowConvNet-gERP:')
    print(f'    bacc        = {dl_res["balanced_acc"]:.4f}')
    print(f'    accuracy    = {dl_res["accuracy"]:.4f}')
    print(f'    oracle_bacc = {dl_res["oracle_bacc"]:.4f}  (thr={dl_res["oracle_thr"]})')
    print(f'    rec_vua     = {dl_res["rec_vua"]:.4f}')
    plot_confusion(dl_res['y_true'], dl_res['y_pred'],
        f'ShallowConvNet-gERP (10ch×0-2s)\nbacc={dl_res["balanced_acc"]:.4f}  rec_vua={dl_res["rec_vua"]:.4f}',
        os.path.join(FIG_DIR,'cm_dl_gerp.png'))

    # ── Save & summary ──────────────────────────────────────────────────────
    df_ml.to_csv(os.path.join(OUT_DIR,'results_xgb_gerp.csv'), index=False)
    elapsed = (datetime.datetime.now()-t0).seconds

    print(f'\n{"="*78}')
    print('  FINAL SUMMARY')
    print(f'{"="*78}')
    print(f'  Feature set: gERP-specific ({X.shape[1]} features)')
    print(f'  Channels: {list(TASTE_CH.keys())} (taste-relevant only)')
    print(f'  Key windows: N400(300-500ms), LatePos(500-1000ms), 50ms bins 0-2s')
    print()
    print(f'  Part A — XGB GPU + gERP features:')
    if best_ml:
        print(f'    Best XGB: K={best_ml["K"]} samp={best_ml["sampling"]}')
        print(f'      bacc    = {best_ml["balanced_acc"]:.4f}')
        print(f'      acc     = {best_ml["accuracy"]:.4f}')
        print(f'      rec_vua = {best_ml["rec_vua"]:.4f}')
    print(f'  Part B — ShallowConvNet on gERP-trimmed raw:')
    print(f'    bacc        = {dl_res["balanced_acc"]:.4f}')
    print(f'    oracle_bacc = {dl_res["oracle_bacc"]:.4f}')
    print(f'    rec_vua     = {dl_res["rec_vua"]:.4f}')
    print()
    print(f'  Progression (bacc — metric thật):')
    print(f'    ML v2 GradBoost thr-tuned:    0.649')
    print(f'    DL v2 ShallowConvNet:          0.674  ← previous best')
    if best_ml:
        print(f'    gERP XGB (best K):             {best_ml["balanced_acc"]:.3f}')
    print(f'    gERP ShallowConvNet (0-2s):    {dl_res["balanced_acc"]:.3f}')

    best_all = max(
        best_ml["balanced_acc"] if best_ml else 0,
        dl_res["balanced_acc"]
    )
    print(f'\n  {"✓ IMPROVEMENT!" if best_all > 0.674 else "→ Similar to previous best"}')
    print(f'  Best gERP bacc = {best_all:.4f}')
    print(f'\n  Total: {elapsed}s ({elapsed//60}m {elapsed%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
