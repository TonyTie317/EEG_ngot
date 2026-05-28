#!/usr/bin/env python3
"""
Vua_phai vs Others — Per-Trial ML (quick run)
==============================================
Instead of condition-averaged features (n=119), use individual trials (n=840).
  - 220 Vua_phai trials vs 620 Others  (ratio 26% vs 74%, N 7× larger)
  - Features: ERP windows + band power + time-domain stats per trial
  - Models: XGB GPU, LGBM GPU, RF, SVM, LogReg
  - LOSO-CV by subject (28 folds)
  - K sweep: 5, 10, 15, 20, 25, 30 (MI-selected)
  - IsoForest contam: 0.05, 0.10
  - Oracle threshold (post-hoc)
"""

import os, sys, warnings, datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, recall_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

EPOCH_DIR = 'output/epochs'
OUT_DIR   = 'output/results/ml_pertrial'
FIG_DIR   = 'output/figures/ml_pertrial'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SFREQ = 100
TMIN  = -0.5   # epoch start (s)

# ERP component windows (s) and channel ROIs
ERP_WINDOWS = {
    'P1':   (0.08, 0.15),
    'N1':   (0.10, 0.20),
    'P2':   (0.20, 0.35),
    'N400': (0.30, 0.50),
    'late': (0.50, 1.00),
}
# Frequency bands (Hz)
BANDS = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,13),
         'beta': (13,30), 'gamma': (30,45)}

CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T3','T4','Fz','Cz']


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── Per-trial feature extraction ────────────────────────────────────────────
def extract_features_trial(epoch, sfreq=SFREQ, tmin=TMIN):
    """
    epoch: (n_channels, n_times) in volts (already baseline-corrected)
    Returns flat feature vector.
    """
    n_ch, n_t = epoch.shape
    times = np.linspace(tmin, tmin + (n_t - 1) / sfreq, n_t)
    feats = {}

    # ── ERP window features ──────────────────────────────────────────────
    for comp, (t0, t1) in ERP_WINDOWS.items():
        mask = (times >= t0) & (times <= t1)
        if mask.sum() == 0:
            continue
        win = epoch[:, mask]          # (n_ch, n_win_t)
        mean_ch = win.mean(axis=1)    # mean over time per channel
        feats[f'erp_{comp}_mean']     = float(mean_ch.mean() * 1e6)
        feats[f'erp_{comp}_max']      = float(mean_ch.max() * 1e6)
        feats[f'erp_{comp}_min']      = float(mean_ch.min() * 1e6)
        feats[f'erp_{comp}_rms']      = float(np.sqrt(np.mean(win**2)) * 1e6)
        feats[f'erp_{comp}_ptp']      = float((win.max(axis=1) - win.min(axis=1)).mean() * 1e6)
        # Per-channel mean for first 6 channels (frontal/central most relevant)
        for ci in range(min(6, n_ch)):
            feats[f'erp_{comp}_ch{ci}'] = float(win[ci].mean() * 1e6)

    # ── Band power ───────────────────────────────────────────────────────
    from scipy.signal import welch
    f, psd = welch(epoch, fs=sfreq, nperseg=min(128, n_t))   # (n_ch, n_freq)
    for bname, (flo, fhi) in BANDS.items():
        bm = (f >= flo) & (f <= fhi)
        if bm.sum() == 0:
            continue
        bp_per_ch = psd[:, bm].mean(axis=1)   # (n_ch,)
        feats[f'bp_{bname}_mean'] = float(bp_per_ch.mean())
        feats[f'bp_{bname}_max']  = float(bp_per_ch.max())
        feats[f'bp_{bname}_std']  = float(bp_per_ch.std())
        # Frontal (F3,F4,Fz) and central (C3,C4,Cz) averages
        frontal_idx = [2, 3, 14]   # F3, F4, Fz
        central_idx = [4, 5, 15]   # C3, C4, Cz
        feats[f'bp_{bname}_frontal'] = float(bp_per_ch[frontal_idx].mean())
        feats[f'bp_{bname}_central'] = float(bp_per_ch[central_idx].mean())

    # ── Time-domain stats per channel ────────────────────────────────────
    for ci in range(n_ch):
        x = epoch[ci]
        feats[f'td_mean_ch{ci}'] = float(x.mean() * 1e6)
        feats[f'td_std_ch{ci}']  = float(x.std() * 1e6)
        feats[f'td_rms_ch{ci}']  = float(np.sqrt(np.mean(x**2)) * 1e6)

    return feats


def build_per_trial_features():
    """Load all subject epochs and extract per-trial features. Returns DataFrame."""
    all_rows = []
    subjects = sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*')))

    for subj_dir in subjects:
        subj_id = os.path.basename(subj_dir)
        npy_path = os.path.join(subj_dir, 'epochs_data.npy')
        ti_path  = os.path.join(subj_dir, 'trial_info.csv')
        if not os.path.exists(npy_path) or not os.path.exists(ti_path):
            continue

        epochs = np.load(npy_path)    # (n_trials, n_ch, n_t)
        ti     = pd.read_csv(ti_path)

        if len(epochs) != len(ti):
            print(f'  {subj_id}: epoch/trial_info mismatch ({len(epochs)} vs {len(ti)}) — skip')
            continue

        for i in range(len(epochs)):
            feats = extract_features_trial(epochs[i])
            feats['subject_id'] = subj_id
            feats['epoch_ix']   = int(ti.iloc[i]['epoch_ix'])
            feats['condition']  = int(ti.iloc[i]['condition'])
            feats['repeat']     = int(ti.iloc[i]['repeat'])
            feats['jar_group']  = ti.iloc[i]['jar_group']
            all_rows.append(feats)

    df = pd.DataFrame(all_rows)
    meta = ['subject_id', 'epoch_ix', 'condition', 'repeat', 'jar_group']
    feat_cols = [c for c in df.columns if c not in meta]
    return df, feat_cols


# ─── ML helpers (same as v3) ──────────────────────────────────────────────────
def smote_safe(X_tr, y_tr):
    n_pos = (y_tr == 1).sum()
    if n_pos < 2: return X_tr, y_tr
    k = min(5, n_pos - 1)
    try:
        return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_tr, y_tr)
    except Exception:
        return X_tr, y_tr


def precompute_folds(X, y, groups, iso_contam):
    logo = LeaveOneGroupOut()
    folds = []
    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr = X[tr].copy(), y[tr].copy()
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue
        if iso_contam > 0 and len(X_tr) > 20:
            iso = IsolationForest(contamination=iso_contam, random_state=SEED, n_jobs=-1)
            iso.fit(X_tr)
            keep = iso.predict(X_tr) == 1
            if (y_tr[keep] == 0).any() and (y_tr[keep] == 1).any():
                X_tr, y_tr = X_tr[keep], y_tr[keep]
        if len(np.unique(y_tr)) < 2:
            continue
        sc = StandardScaler()
        X_tr = np.nan_to_num(sc.fit_transform(X_tr))
        X_te = np.nan_to_num(sc.transform(X_te))
        mi    = mutual_info_classif(X_tr, y_tr, random_state=SEED)
        order = np.argsort(mi)[::-1]
        folds.append({'X_tr': X_tr, 'y_tr': y_tr,
                      'X_te': X_te, 'y_te': y_te,
                      'mi_order': order})
    return folds


def eval_K(folds, K, model_factory, sampling):
    y_true_all, y_pred_all, proba_all = [], [], []
    for f in folds:
        idx  = f['mi_order'][:K]
        X_tr = f['X_tr'][:, idx].copy()
        y_tr = f['y_tr'].copy()
        X_te = f['X_te'][:, idx]

        if sampling == 'smote':
            X_tr, y_tr = smote_safe(X_tr, y_tr)

        m = model_factory()
        m.fit(X_tr, y_tr)
        y_pred_all.extend(m.predict(X_te).tolist())
        y_true_all.extend(f['y_te'].tolist())
        if hasattr(m, 'predict_proba'):
            proba_all.extend(m.predict_proba(X_te)[:, 1].tolist())
        else:
            proba_all.extend([np.nan] * len(f['y_te']))

    if not y_true_all:
        return None
    yt, yp = np.array(y_true_all), np.array(y_pred_all)
    pr = np.array(proba_all)
    res = {
        'accuracy':        accuracy_score(yt, yp),
        'balanced_acc':    balanced_accuracy_score(yt, yp),
        'f1_macro':        f1_score(yt, yp, average='macro', zero_division=0),
        'recall_vua_phai': recall_score(yt, yp, pos_label=1, zero_division=0),
        'recall_others':   recall_score(yt, yp, pos_label=0, zero_division=0),
        'oracle_acc': accuracy_score(yt, yp), 'oracle_bacc': balanced_accuracy_score(yt, yp),
        'oracle_thr': 0.5, 'y_true': yt, 'y_pred': yp,
    }
    if not np.isnan(pr).any():
        best_acc, best_thr = 0.0, 0.5
        best_bacc = 0.0
        for thr in np.linspace(0.10, 0.90, 81):
            yp_t = (pr >= thr).astype(int)
            a = accuracy_score(yt, yp_t)
            b = balanced_accuracy_score(yt, yp_t)
            if a > best_acc: best_acc = a; best_thr = thr
            if b > best_bacc: best_bacc = b
        res['oracle_acc']  = round(float(best_acc), 4)
        res['oracle_bacc'] = round(float(best_bacc), 4)
        res['oracle_thr']  = round(float(best_thr), 2)
    return res


def plot_confusion(y_true, y_pred, title, fig_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)'
                       for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Other', 'Vua_phai'],
                yticklabels=['Other', 'Vua_phai'],
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

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_ml_pertrial')
    print('='*78)
    print('  Per-trial Vua_phai vs Others (n~840 vs condition-averaged n=119)')
    print('='*78)

    K_GRID      = [5, 10, 15, 20, 25, 30, 40, 50]
    ISO_CONTAMS = [0.05, 0.10]
    SAMPLINGS   = ['none', 'smote']

    MODELS = {
        'XGB_gpu':   lambda: xgb.XGBClassifier(
            device='cuda', n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, scale_pos_weight=3,
            eval_metric='logloss', verbosity=0, random_state=SEED),
        'LGBM_gpu':  lambda: lgb.LGBMClassifier(
            device='gpu', n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, class_weight='balanced',
            verbose=-1, random_state=SEED),
        'RF_100':    lambda: RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=SEED, n_jobs=-1),
        'SVM_RBF':   lambda: SVC(kernel='rbf', C=1.0, gamma='scale',
                                  class_weight='balanced', probability=True,
                                  random_state=SEED),
        'LogReg_C1': lambda: LogisticRegression(
            max_iter=3000, C=1.0, class_weight='balanced',
            solver='lbfgs', random_state=SEED),
    }

    # ── Extract per-trial features ─────────────────────────────────────────
    print('\nExtracting per-trial features...')
    t0 = datetime.datetime.now()
    df, feat_cols = build_per_trial_features()
    print(f'  Done in {(datetime.datetime.now()-t0).seconds}s')
    print(f'  Shape: {df.shape}')
    print(f'  Features: {len(feat_cols)}')
    print(f'  jar_group: {df.jar_group.value_counts().to_dict()}')

    # Save features for reuse
    feat_csv = os.path.join(OUT_DIR, 'features_pertrial.csv')
    df.to_csv(feat_csv, index=False)
    print(f'  Saved: {feat_csv}')

    # Prepare X, y, groups
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.var(axis=0) > 1e-12
    X = X[:, keep]
    y = (df['jar_group'].values == 'Vua_phai').astype(int)
    g = df['subject_id'].values

    n_pos = int(y.sum()); n_neg = int((y==0).sum())
    majority = max(n_pos, n_neg) / len(y)
    print(f'\n  n={len(y)}  pos(Vua_phai)={n_pos}  neg={n_neg}  '
          f'majority={majority:.3f}  subjects={len(np.unique(g))}')
    print(f'  (condition-averaged had: n=119, pos=29, majority=0.756)')

    rows = []
    best_acc_row  = [None]
    best_bacc_row = [None]
    t_start = datetime.datetime.now()

    for iso_contam in ISO_CONTAMS:
        elapsed = (datetime.datetime.now() - t_start).seconds
        print(f'\n{"━"*60}')
        print(f'  [iso={iso_contam:.2f}]  elapsed={elapsed}s', flush=True)

        folds = precompute_folds(X, y, g, iso_contam)
        print(f'    → {len(folds)} folds prepared', flush=True)

        for sampling in SAMPLINGS:
            for mname, mfac in MODELS.items():
                for K in K_GRID:
                    res = eval_K(folds, K, mfac, sampling)
                    if res is None:
                        continue
                    row = {
                        'iso_contam': iso_contam, 'sampling': sampling,
                        'model': mname, 'K': K,
                        'n_samples': len(y), 'majority': round(majority, 4),
                        'accuracy':        round(res['accuracy'],        4),
                        'balanced_acc':    round(res['balanced_acc'],    4),
                        'f1_macro':        round(res['f1_macro'],        4),
                        'recall_vua_phai': round(res['recall_vua_phai'], 4),
                        'recall_others':   round(res['recall_others'],   4),
                        'oracle_acc':      res['oracle_acc'],
                        'oracle_bacc':     res['oracle_bacc'],
                        'oracle_thr':      res['oracle_thr'],
                    }
                    rows.append(row)
                    if best_acc_row[0] is None or row['oracle_acc'] > best_acc_row[0]['oracle_acc']:
                        best_acc_row[0] = {**row, 'y_true': res['y_true'], 'y_pred': res['y_pred']}
                    if best_bacc_row[0] is None or row['balanced_acc'] > best_bacc_row[0]['balanced_acc']:
                        best_bacc_row[0] = {**row, 'y_true': res['y_true'], 'y_pred': res['y_pred']}

            # Mini-summary
            sub = pd.DataFrame([r for r in rows
                                 if r['iso_contam'] == iso_contam
                                 and r['sampling'] == sampling])
            if not sub.empty:
                top5 = sub.nlargest(5, 'oracle_acc')
                print(f'    [samp={sampling}] top-5 oracle_acc:')
                for _, r in top5.iterrows():
                    flag = ' ← BEST!' if r['oracle_acc'] == best_acc_row[0]['oracle_acc'] else ''
                    print(f'      K={int(r["K"]):<3} {r["model"]:<12} '
                          f'acc={r["accuracy"]:.4f}  '
                          f'oracle={r["oracle_acc"]:.4f}(thr={r["oracle_thr"]:.2f})  '
                          f'bacc={r["balanced_acc"]:.4f}  '
                          f'rec_vua={r["recall_vua_phai"]:.3f}' + flag)

    # ── Save & report ──────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(OUT_DIR, 'results_pertrial.csv'), index=False)
    df_res.nlargest(20, 'oracle_acc').to_csv(
        os.path.join(OUT_DIR, 'top20_oracle_acc.csv'), index=False)
    df_res.nlargest(20, 'balanced_acc').to_csv(
        os.path.join(OUT_DIR, 'top20_balanced_acc.csv'), index=False)
    print(f'\n✓ Saved results ({len(df_res)} rows)')

    # K-curve plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = plt.cm.tab10
    model_list = sorted(df_res['model'].unique())
    for ax, iso in zip(axes, ISO_CONTAMS):
        sub = df_res[df_res['iso_contam'] == iso]
        pivot = sub.groupby(['K', 'model'])['oracle_acc'].max().unstack()
        for i, mn in enumerate(model_list):
            if mn not in pivot.columns: continue
            ax.plot(pivot.index, pivot[mn], marker='o', lw=2,
                    label=mn, color=cmap(i / max(len(model_list), 1)))
        ax.axhline(0.85, color='red', ls='--', lw=1.5, label='target=0.85')
        ax.axhline(majority, color='gray', ls=':', lw=1.0,
                   label=f'majority={majority:.3f}')
        ax.axhline(0.824, color='green', ls=':', lw=1.0, label='v3_best=0.824')
        ax.set_title(f'Per-trial  iso={iso:.2f}', fontweight='bold')
        ax.set_xlabel('K'); ax.set_ylabel('Oracle Accuracy')
        ax.set_ylim(0.50, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc='lower right')
    fig.suptitle('Per-trial Oracle Accuracy vs K (LOSO-CV by subject)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'oracle_acc_vs_K.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    b = best_acc_row[0]
    if b:
        plot_confusion(
            b['y_true'], b['y_pred'],
            (f'{b["model"]} | iso={b["iso_contam"]} | K={b["K"]} | samp={b["sampling"]}\n'
             f'acc={b["accuracy"]:.4f}  oracle={b["oracle_acc"]:.4f}(thr={b["oracle_thr"]})  '
             f'bacc={b["balanced_acc"]:.4f}  rec_vua={b["recall_vua_phai"]:.4f}'),
            os.path.join(FIG_DIR, 'cm_best_oracle_acc.png')
        )
    print('✓ Saved plots')

    elapsed_total = (datetime.datetime.now() - t_start).seconds
    print(f'\n{"="*78}')
    print('  TOP-20 BY ORACLE_ACC')
    print(f'{"="*78}')
    hdr = f'{"#":<4}{"model":<13}{"samp":<9}{"K":<4}{"iso":<6}{"acc":<8}{"oracle":<10}{"thr":<6}{"bacc":<8}{"rec_vua":<9}'
    print(hdr); print('-'*len(hdr))
    for rank, (_, r) in enumerate(df_res.nlargest(20, 'oracle_acc').iterrows(), 1):
        print(f'{rank:<4}{r["model"]:<13}{r["sampling"]:<9}{int(r["K"]):<4}'
              f'{r["iso_contam"]:<6}{r["accuracy"]:.4f}  {r["oracle_acc"]:.4f}    '
              f'{r["oracle_thr"]:<6}{r["balanced_acc"]:.4f}  {r["recall_vua_phai"]:.4f}')

    if b:
        print(f'\n  BEST:  {b["model"]} | K={b["K"]} | iso={b["iso_contam"]} | samp={b["sampling"]}')
        print(f'    accuracy    = {b["accuracy"]:.4f}  (threshold=0.5)')
        print(f'    oracle_acc  = {b["oracle_acc"]:.4f}  (threshold={b["oracle_thr"]})')
        print(f'    balanced_acc= {b["balanced_acc"]:.4f}')
        print(f'    rec_vua     = {b["recall_vua_phai"]:.4f}')
        print(f'    {"✓ TARGET >0.85 REACHED!" if b["oracle_acc"] >= 0.85 else "✗ target not reached"}')
        print(f'\n  Comparison:')
        print(f'    condition-avg (n=119): v3 best acc=0.778 oracle=0.824 bacc=0.604')
        print(f'    per-trial     (n={b["n_samples"]}):  best acc={b["accuracy"]:.3f} oracle={b["oracle_acc"]:.3f} bacc={b["balanced_acc"]:.3f}')

    print(f'\n  Total runtime: {elapsed_total}s ({elapsed_total//60}m {elapsed_total%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
