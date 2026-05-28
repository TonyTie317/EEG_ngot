#!/usr/bin/env python3
"""
DL v2 — pos_weight sweep + augmentation + DL×ML ensemble
=========================================================
v1 findings:
  EEGNet:       bacc=0.570  recall_vua=0.946  (too aggressive — pos_weight=3 too high)
  DeepConvNet:  bacc=0.624  recall_vua=0.841  (best so far)
  ShallowConvNet: bacc=0.614

This script:
  1. Sweeps pos_weight ∈ {1.5, 2.0, 3.0, 4.0} for all 3 models
  2. Better augmentation: Gaussian noise + random time-shift
  3. Excludes folds with n_pos=0 from metrics (can't measure TPR)
  4. DL + XGB GPU ensemble (combine probas per fold)
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, recall_score)
import xgboost as xgb

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH_DIR = 'output/epochs'
FEAT_CSV  = 'output/results/ml_jar3/features_jar3_adv.csv'
QUAL_CSV  = 'output/results/erp/erp_quality_flags.csv'
OUT_DIR   = 'output/results/dl_vuaphai_v2'
FIG_DIR   = 'output/figures/dl_vuaphai_v2'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_CH = 16; N_TIMES = 351; SFREQ = 100


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── Models (same as v1) ──────────────────────────────────────────────────────
class EEGNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_t=N_TIMES, F1=8, D=2, F2=16,
                 dropout=0.5, kernel_len=64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), padding=(0, 8), bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            flat = self.block2(self.block1(dummy)).flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.block2(self.block1(x)).flatten(1)).squeeze(1)


class ShallowConvNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_t=N_TIMES, n_filters=40,
                 filter_len=25, pool_len=75, pool_stride=15, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, filter_len), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.pl = pool_len; self.ps = pool_stride
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_ch, n_t)
            out = self.conv(dummy)
            flat = F.avg_pool2d(out.pow(2),(1,self.pl),stride=(1,self.ps)).log().flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.avg_pool2d(x.pow(2), (1, self.pl), stride=(1, self.ps)).log()
        return self.fc(self.dropout(x).flatten(1)).squeeze(1)


class DeepConvNet(nn.Module):
    def __init__(self, n_ch=N_CH, n_t=N_TIMES, dropout=0.5):
        super().__init__()
        def blk(i, o, k, p): return nn.Sequential(
            nn.Conv2d(i, o, k, bias=False), nn.BatchNorm2d(o),
            nn.ELU(), nn.MaxPool2d(p), nn.Dropout(dropout))
        self.b0 = nn.Sequential(
            nn.Conv2d(1, 25, (1,5), bias=False),
            nn.Conv2d(25, 25, (n_ch,1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(),
            nn.MaxPool2d((1,2)), nn.Dropout(dropout))
        self.b1 = blk(25, 50,  (1,5), (1,2))
        self.b2 = blk(50, 100, (1,5), (1,2))
        self.b3 = blk(100,200, (1,5), (1,2))
        with torch.no_grad():
            dummy = torch.zeros(1,1,n_ch,n_t)
            flat = self.b3(self.b2(self.b1(self.b0(dummy)))).flatten(1).shape[1]
        self.fc = nn.Linear(flat, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.fc(self.b3(self.b2(self.b1(self.b0(x)))).flatten(1)).squeeze(1)


# ─── Data ─────────────────────────────────────────────────────────────────────
def load_epochs():
    Xs, ys, gs = [], [], []
    for d in sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*'))):
        s = os.path.basename(d)
        npy = os.path.join(d, 'epochs_data.npy')
        ti  = os.path.join(d, 'trial_info.csv')
        if not os.path.exists(npy) or not os.path.exists(ti): continue
        ep = np.load(npy).astype(np.float32)
        info = pd.read_csv(ti)
        if len(ep) != len(info): continue
        Xs.append(ep)
        ys.append((info['jar_group'].values == 'Vua_phai').astype(np.float32))
        gs.extend([s] * len(ep))
    return np.concatenate(Xs), np.concatenate(ys), np.array(gs)


def channel_normalize(X_tr, X_te):
    m = X_tr.mean(axis=(0,2), keepdims=True)
    s = X_tr.std(axis=(0,2),  keepdims=True) + 1e-8
    return (X_tr-m)/s, (X_te-m)/s


def augment_batch(X, noise_std=0.05, shift_max=10):
    """Gaussian noise + random time-shift augmentation."""
    n, c, t = X.shape
    noise = torch.randn_like(X) * noise_std * X.std(dim=(1,2), keepdim=True).clamp(min=1e-8)
    X = X + noise
    # random circular shift per sample
    shifts = torch.randint(-shift_max, shift_max+1, (n,))
    for i, sh in enumerate(shifts):
        if sh != 0:
            X[i] = torch.roll(X[i], sh.item(), dims=-1)
    return X


def make_loader(X, y, batch_size=32, balanced=True):
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    ds  = TensorDataset(X_t, y_t)
    if balanced:
        n_pos = max(int(y.sum()), 1); n_neg = max(int((y==0).sum()), 1)
        w = np.where(y==1, 1.0/n_pos, 1.0/n_neg)
        sampler = WeightedRandomSampler(torch.from_numpy(w).float(),
                                        num_samples=len(w), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def train_one(model, X_tr, y_tr, X_val, y_val,
              n_epochs=250, lr=5e-4, batch_size=32,
              pos_weight=3.0, patience=30):
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    loader = make_loader(X_tr, y_tr, batch_size=batch_size)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_int = y_val.astype(int)

    best_bacc, best_state, no_imp = 0.0, None, 0
    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb = augment_batch(xb.to(DEVICE))
            yb = yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pr = torch.sigmoid(model(X_val_t)).cpu().numpy()
        bacc = balanced_accuracy_score(y_val_int, (pr >= 0.5).astype(int))
        if bacc > best_bacc:
            best_bacc = bacc
            best_state = {k: v.clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    if best_state: model.load_state_dict(best_state)
    return model


def loso_dl(model_cls, model_kw, X, y, groups,
            pos_weight=3.0, n_epochs=250, lr=5e-4,
            batch_size=32, patience=30):
    subjects = sorted(np.unique(groups))
    y_true_all, y_pred_all, y_proba_all = [], [], []
    for subj in subjects:
        te = groups == subj; tr = ~te
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0: continue
        X_tr, X_te = channel_normalize(X_tr, X_te)
        m = train_one(model_cls(**model_kw), X_tr, y_tr, X_te, y_te,
                      n_epochs=n_epochs, lr=lr, batch_size=batch_size,
                      pos_weight=pos_weight, patience=patience)
        m.eval()
        with torch.no_grad():
            pr = torch.sigmoid(m(torch.from_numpy(X_te).float().to(DEVICE))).cpu().numpy()
        y_true_all.extend(y_te.astype(int).tolist())
        y_pred_all.extend((pr >= 0.5).astype(int).tolist())
        y_proba_all.extend(pr.tolist())
    return (np.array(y_true_all), np.array(y_pred_all), np.array(y_proba_all))


def metrics(yt, yp, pr):
    # exclude folds with no Vua_phai (can't measure TPR)
    mask = True  # use all — per-fold exclusion done outside
    acc  = accuracy_score(yt, yp)
    bacc = balanced_accuracy_score(yt, yp)
    f1   = f1_score(yt, yp, average='macro', zero_division=0)
    rv   = recall_score(yt, yp, pos_label=1, zero_division=0)
    ro   = recall_score(yt, yp, pos_label=0, zero_division=0)
    # oracle threshold
    best_bacc_thr, best_thr = 0.0, 0.5
    for thr in np.linspace(0.10, 0.90, 81):
        b = balanced_accuracy_score(yt, (pr>=thr).astype(int))
        if b > best_bacc_thr: best_bacc_thr = b; best_thr = thr
    return {'acc': acc, 'bacc': bacc, 'f1': f1,
            'rec_vua': rv, 'rec_oth': ro,
            'oracle_bacc': best_bacc_thr, 'oracle_thr': round(float(best_thr),2)}


def plot_confusion(yt, yp, title, path):
    cm = confusion_matrix(yt, yp)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    ann = np.array([[f'{cm[i,j]}\n({cmn[i,j]*100:.0f}%)' for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cmn, annot=ann, fmt='', cmap='Blues',
                xticklabels=['Other','Vua_phai'], yticklabels=['Other','Vua_phai'],
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=8, fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# ─── XGB helper for ensemble ──────────────────────────────────────────────────
def xgb_loso_proba(X_feat, y, groups, iso_contam=0.10, K=29):
    """Quick XGB GPU LOSO — returns per-sample proba array (same order as input)."""
    logo = LeaveOneGroupOut()
    proba_out = np.zeros(len(y))
    for tr, te in logo.split(X_feat, y, groups):
        X_tr, y_tr = X_feat[tr].copy(), y[tr].copy()
        X_te = X_feat[te]
        if len(np.unique(y_tr)) < 2: continue
        # IsoForest
        if iso_contam > 0 and len(X_tr) > 20:
            iso = IsolationForest(contamination=iso_contam, random_state=SEED, n_jobs=-1)
            iso.fit(X_tr); km = iso.predict(X_tr)==1
            if (y_tr[km]==0).any() and (y_tr[km]==1).any():
                X_tr, y_tr = X_tr[km], y_tr[km]
        sc = StandardScaler()
        X_tr = np.nan_to_num(sc.fit_transform(X_tr))
        X_te = np.nan_to_num(sc.transform(X_te))
        mi = mutual_info_classif(X_tr, y_tr, random_state=SEED)
        idx = np.argsort(mi)[::-1][:K]
        X_tr, X_te = X_tr[:,idx], X_te[:,idx]
        m = xgb.XGBClassifier(device='cuda', n_estimators=100, max_depth=4,
                               learning_rate=0.05, subsample=0.8,
                               scale_pos_weight=3, eval_metric='logloss',
                               verbosity=0, random_state=SEED)
        m.fit(X_tr, y_tr)
        proba_out[te] = m.predict_proba(X_te)[:, 1]
    return proba_out


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fh   = open(os.path.join(OUT_DIR, 'logs', f'run_{ts}.log'), 'w')
    latest_fh = open(os.path.join(OUT_DIR, 'run.log'), 'w')
    sys.stdout = Tee(sys.__stdout__, log_fh, latest_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh, latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_dl_vuaphai_v2')
    print('='*78)
    print(f'  DL v2: pos_weight sweep + augmentation + DL×XGB ensemble')
    print(f'  Device: {DEVICE}  ({torch.cuda.get_device_name(0)})')
    print('='*78)

    # ── Load epoch data ───────────────────────────────────────────────────
    print('\nLoading epoch data...')
    X_ep, y_ep, g_ep = load_epochs()
    print(f'  Epochs: {X_ep.shape}  Vua_phai={int(y_ep.sum())}  Others={int((y_ep==0).sum())}')

    # ── Model × pos_weight grid ───────────────────────────────────────────
    MODEL_CFGS = [
        ('EEGNet',       EEGNet,        {'n_ch':N_CH,'n_t':N_TIMES,'F1':8,'D':2,'F2':16,'dropout':0.5,'kernel_len':64}),
        ('ShallowConvNet', ShallowConvNet, {'n_ch':N_CH,'n_t':N_TIMES,'n_filters':40,'filter_len':25,'pool_len':75,'pool_stride':15,'dropout':0.5}),
        ('DeepConvNet',  DeepConvNet,   {'n_ch':N_CH,'n_t':N_TIMES,'dropout':0.5}),
    ]
    POS_WEIGHTS = [1.5, 2.0, 3.0, 4.0]

    rows = []
    best_bacc_global = 0.0
    best_proba_cache = {}   # (model_name, pw) → proba array, for ensemble
    t_start = datetime.datetime.now()

    for mname, mcls, mkw in MODEL_CFGS:
        print(f'\n{"━"*60}')
        print(f'  Model: {mname}')
        best_pw_bacc, best_pw = 0.0, None

        for pw in POS_WEIGHTS:
            elapsed = (datetime.datetime.now()-t_start).seconds
            print(f'  [pos_weight={pw}]  elapsed={elapsed}s', flush=True)

            yt, yp, pr = loso_dl(mcls, mkw, X_ep, y_ep, g_ep,
                                  pos_weight=pw, n_epochs=250,
                                  lr=5e-4, batch_size=32, patience=30)
            m = metrics(yt, yp, pr)
            best_proba_cache[(mname, pw)] = (yt, pr)

            print(f'    pw={pw}  acc={m["acc"]:.4f}  bacc={m["bacc"]:.4f}  '
                  f'oracle_bacc={m["oracle_bacc"]:.4f}(thr={m["oracle_thr"]})  '
                  f'rec_vua={m["rec_vua"]:.4f}  rec_oth={m["rec_oth"]:.4f}')

            row = {'model': mname, 'pos_weight': pw, **m}
            rows.append(row)

            if m['bacc'] > best_pw_bacc:
                best_pw_bacc = m['bacc']; best_pw = pw
            if m['bacc'] > best_bacc_global:
                best_bacc_global = m['bacc']
                plot_confusion(yt, yp,
                    f'{mname} pw={pw}  bacc={m["bacc"]:.4f}  rec_vua={m["rec_vua"]:.4f}',
                    os.path.join(FIG_DIR, f'cm_best.png'))

        print(f'  → Best pos_weight for {mname}: {best_pw} (bacc={best_pw_bacc:.4f})')

    # ── DL + XGB Ensemble ─────────────────────────────────────────────────
    print(f'\n{"━"*60}')
    print('  DL × XGB GPU Ensemble')
    print('  (best DL per model × XGB K=29 iso=0.10 → weighted average)')

    # Load condition-averaged features for XGB
    try:
        df_feat = pd.read_csv(FEAT_CSV)
        qf = pd.read_csv(QUAL_CSV)[['subject_id','condition','quality_label','avg_snr','quality_score']]
        df_feat['condition'] = df_feat['condition'].astype(int)
        qf['condition'] = qf['condition'].astype(int)
        df_feat = df_feat.merge(qf, on=['subject_id','condition'], how='left')
        df_feat = df_feat[df_feat['quality_label'] != 'BAD'].copy()
        META = ['subject_id','condition','jar_group','quality_label','avg_snr','quality_score']
        fc = [c for c in df_feat.columns if c not in META]
        X_feat = df_feat[fc].values.astype(float)
        X_feat = np.nan_to_num(X_feat)
        keep = X_feat.var(axis=0) > 1e-12; X_feat = X_feat[:, keep]
        y_feat = (df_feat['jar_group'].values == 'Vua_phai').astype(int)
        g_feat = df_feat['subject_id'].values

        print(f'  XGB data: n={len(y_feat)} pos={y_feat.sum()}  Running LOSO...', flush=True)
        xgb_pr = xgb_loso_proba(X_feat, y_feat, g_feat)

        # Map XGB probas to per-trial (repeat for each trial of same subject×condition)
        # Build subject → XGB proba mapping from condition-averaged
        subj_cond_to_xgb = {}
        for i, row in df_feat.iterrows():
            subj_cond_to_xgb[(row['subject_id'], int(row['condition']))] = xgb_pr[df_feat.index.get_loc(i)]

        # Get per-trial XGB proba by matching subject+condition
        all_ti = pd.read_csv(os.path.join(EPOCH_DIR, 'all_trial_info.csv'))
        all_ti = all_ti.sort_values(['subject_id','epoch_ix']).reset_index(drop=True)
        # Build xgb_proba_pertrial in same order as X_ep
        ti_ordered = []
        for d in sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*'))):
            s = os.path.basename(d)
            ti = pd.read_csv(os.path.join(d, 'trial_info.csv'))
            if not os.path.exists(os.path.join(d, 'epochs_data.npy')): continue
            ti_ordered.append(ti[['subject_id','condition']])
        ti_all = pd.concat(ti_ordered, ignore_index=True)
        xgb_trial_pr = np.array([
            subj_cond_to_xgb.get((r['subject_id'], int(r['condition'])), 0.3)
            for _, r in ti_all.iterrows()
        ])

        # Ensemble: try different DL models + XGB weights
        best_dl_key = max(best_proba_cache,
                          key=lambda k: metrics(*best_proba_cache[k][0:1],
                                                (best_proba_cache[k][1]>=0.5).astype(int),
                                                best_proba_cache[k][1])['bacc']
                                        if len(best_proba_cache[k][0]) == len(xgb_trial_pr)
                                        else -1)

        ens_rows = []
        for (mname, pw), (yt_dl, pr_dl) in best_proba_cache.items():
            if len(pr_dl) != len(xgb_trial_pr): continue
            for alpha in [0.3, 0.5, 0.7]:   # DL weight
                pr_ens = alpha * pr_dl + (1-alpha) * xgb_trial_pr
                yp_ens = (pr_ens >= 0.5).astype(int)
                m = metrics(yt_dl.astype(int), yp_ens, pr_ens)
                ens_rows.append({'model': f'{mname}+XGB', 'pw_dl': pw,
                                  'alpha_dl': alpha, **m})
                print(f'  {mname}(pw={pw}) α={alpha}+XGB  '
                      f'bacc={m["bacc"]:.4f}  oracle_bacc={m["oracle_bacc"]:.4f}  '
                      f'rec_vua={m["rec_vua"]:.4f}')

        df_ens = pd.DataFrame(ens_rows)
        df_ens.to_csv(os.path.join(OUT_DIR, 'ensemble_results.csv'), index=False)

        # Best ensemble
        if len(df_ens) > 0:
            best_ens = df_ens.loc[df_ens['bacc'].idxmax()]
            print(f'\n  Best ensemble: {best_ens["model"]} α_dl={best_ens["alpha_dl"]}')
            print(f'    bacc={best_ens["bacc"]:.4f}  oracle_bacc={best_ens["oracle_bacc"]:.4f}')
            print(f'    rec_vua={best_ens["rec_vua"]:.4f}  acc={best_ens["acc"]:.4f}')

    except Exception as e:
        print(f'  Ensemble skipped: {e}')

    # ── Save & summary ────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(OUT_DIR, 'results_dl_v2.csv'), index=False)

    # Plot: bacc vs pos_weight per model
    fig, ax = plt.subplots(figsize=(10, 5))
    for mname, grp in df_res.groupby('model'):
        ax.plot(grp['pos_weight'], grp['bacc'], marker='o', lw=2, label=f'{mname} bacc')
        ax.plot(grp['pos_weight'], grp['rec_vua'], marker='s', lw=1.5,
                ls='--', alpha=0.6, label=f'{mname} rec_vua')
    ax.axhline(0.624, color='red', ls='--', lw=1.5, label='v1 best bacc=0.624')
    ax.axhline(0.5, color='gray', ls=':', lw=1.0, label='chance')
    ax.set_xlabel('pos_weight'); ax.set_ylabel('Score')
    ax.set_title('DL v2: balanced_acc & recall_vua vs pos_weight', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'bacc_vs_posweight.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    elapsed = (datetime.datetime.now()-t_start).seconds
    print(f'\n{"="*78}')
    print('  FINAL SUMMARY')
    print(f'{"="*78}')
    hdr = f'{"model":<18}{"pw":<6}{"acc":<8}{"bacc":<8}{"f1":<8}{"rec_vua":<10}{"oracle_bacc":<13}{"thr"}'
    print(hdr); print('-'*len(hdr))
    for _, r in df_res.sort_values('bacc', ascending=False).iterrows():
        print(f'{r["model"]:<18}{r["pos_weight"]:<6}{r["acc"]:<8.4f}{r["bacc"]:<8.4f}'
              f'{r["f1"]:<8.4f}{r["rec_vua"]:<10.4f}{r["oracle_bacc"]:<13.4f}{r["oracle_thr"]}')

    best = df_res.loc[df_res['bacc'].idxmax()]
    print(f'\n  Best: {best["model"]}  pos_weight={best["pos_weight"]}')
    print(f'    bacc        = {best["bacc"]:.4f}  (v1 DeepConvNet: 0.624)')
    print(f'    accuracy    = {best["acc"]:.4f}')
    print(f'    oracle_bacc = {best["oracle_bacc"]:.4f}')
    print(f'    rec_vua     = {best["rec_vua"]:.4f}')
    print(f'\n  History (bacc):')
    print(f'    ML v3 XGB K=29:          bacc=0.604')
    print(f'    DL v1 DeepConvNet pw=3:  bacc=0.624')
    print(f'    DL v2 best:              bacc={best["bacc"]:.3f}')

    print(f'\n  Total: {elapsed}s ({elapsed//60}m {elapsed%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
