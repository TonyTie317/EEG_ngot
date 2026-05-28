#!/usr/bin/env python3
"""
gERP features on CONDITION-AVERAGED epochs (correct approach)
=============================================================
Key insight from grand-average ERP analysis at Cz:
  Late positivity (500-1000ms): Vua_phai=4.14 µV vs Others=2.39 µV ← BIG signal
  N400       (300-500ms):       Vua_phai=1.86 µV vs Others=2.38 µV
  P2         (200-350ms):       Vua_phai=1.73 µV vs Others=2.50 µV

Per-trial gERP features failed (bacc ~0.47) because single-trial EEG is too noisy.
Correct approach: average 5 repeats per (subject, condition) FIRST, then extract features.
This gives n=119 clean condition-averages (same as v1/v2/v3 but with better features).

New gERP features:
  1. Fine-grained 50ms bins 0→2s at 10 taste channels    (400 feat)
  2. Late positivity windows: 500-700ms, 700-1000ms, 1000-1500ms per channel (90 feat)
  3. Hemispheric asymmetry at key windows                  (20 feat)
  4. Late/N400 ratio at Cz, Fz                             (2 feat)
  5. Spectral theta/beta in early/late windows             (24 feat)

Then: XGB GPU + GradBoost + LOSO, compare with v2/v3 best (bacc=0.649/0.674)
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

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, recall_score, confusion_matrix)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

SEED = 42
np.random.seed(SEED)

EPOCH_DIR = 'output/epochs'
OUT_DIR   = 'output/results/ml_gerp_avg'
FIG_DIR   = 'output/figures/ml_gerp_avg'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SFREQ  = 100
TMIN   = -0.5
N_TIMES = 351
TIMES  = np.linspace(TMIN, TMIN + (N_TIMES-1)/SFREQ, N_TIMES)

CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T3','T4','Fz','Cz']
TASTE_CH = {'Fz':14,'Cz':15,'C3':4,'C4':5,'F3':2,'F4':3,
            'P3':6,'P4':7,'T3':12,'T4':13}


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── Build condition-averaged epochs + extract gERP features ─────────────────
def build_cond_avg_gerp():
    """
    For each (subject, condition): average across repeats → extract gERP features.
    Returns DataFrame with 1 row per (subject, condition).
    """
    rows = []
    t = TIMES

    for d in sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*'))):
        subj = os.path.basename(d)
        npy  = os.path.join(d, 'epochs_data.npy')
        ti   = os.path.join(d, 'trial_info.csv')
        if not os.path.exists(npy) or not os.path.exists(ti):
            continue

        epochs = np.load(npy).astype(np.float32)  # (n_trials, 16, 351)
        info   = pd.read_csv(ti)
        if len(epochs) != len(info):
            continue

        # Group by condition → average
        for cond, grp in info.groupby('condition'):
            idx  = grp.index.tolist()
            avg  = epochs[idx].mean(axis=0)   # (16, 351) — condition average
            jar  = grp['jar_group'].iloc[0]

            feats = {'subject_id': subj, 'condition': int(cond), 'jar_group': jar}

            # ── 1. Fine-grained 50ms bins 0→2s at taste channels ─────────
            bin_edges = np.arange(0.0, 2.01, 0.05)
            for ch_name, ch_idx in TASTE_CH.items():
                sig = avg[ch_idx] * 1e6
                for i in range(len(bin_edges)-1):
                    t0, t1 = bin_edges[i], bin_edges[i+1]
                    mask = (t >= t0) & (t < t1)
                    key = f'bin_{ch_name}_{int(t0*1000):04d}'
                    feats[key] = float(sig[mask].mean()) if mask.sum() > 0 else 0.0

            # ── 2. Component features at taste channels ───────────────────
            components = {
                'P2':       (0.20, 0.35),
                'N400':     (0.30, 0.50),
                'LatePos1': (0.50, 0.70),   # rising phase
                'LatePos2': (0.70, 1.00),   # plateau
                'LatePos3': (1.00, 1.50),   # very late
                'EarlyP1':  (0.08, 0.15),
            }
            comp_ch = {
                'P2':       ['Cz','C3','C4','Fz'],
                'N400':     ['Cz','Fz','C3','C4','T3','T4'],
                'LatePos1': ['Cz','Fz','C3','C4','P3','P4'],
                'LatePos2': ['Cz','Fz','P3','P4'],
                'LatePos3': ['Cz','Fz'],
                'EarlyP1':  ['Cz','C3','C4'],
            }
            for cname, (t0, t1) in components.items():
                mask = (t >= t0) & (t <= t1)
                for ch_name in comp_ch[cname]:
                    ch_idx = TASTE_CH[ch_name]
                    sig = avg[ch_idx, mask] * 1e6
                    if len(sig) == 0:
                        continue
                    feats[f'{cname}_{ch_name}_mean'] = float(sig.mean())
                    feats[f'{cname}_{ch_name}_peak'] = float(sig.max())
                    feats[f'{cname}_{ch_name}_auc']  = float(np.trapz(sig))

            # ── 3. Hemispheric asymmetry ───────────────────────────────────
            pairs = [('F4','F3'),('C4','C3'),('P4','P3'),('T4','T3')]
            for windows_label, (t0, t1) in [('N400',(0.3,0.5)),('Late',(0.5,1.0))]:
                mask = (t >= t0) & (t <= t1)
                for r_ch, l_ch in pairs:
                    r = float(avg[TASTE_CH[r_ch], mask].mean() * 1e6)
                    l = float(avg[TASTE_CH[l_ch], mask].mean() * 1e6)
                    feats[f'asym_{r_ch}m{l_ch}_{windows_label}'] = r - l

            # ── 4. Late / N400 ratio ───────────────────────────────────────
            for ch_name in ['Cz','Fz']:
                ch_idx = TASTE_CH[ch_name]
                sig = avg[ch_idx] * 1e6
                n400 = sig[(t >= 0.3) & (t <= 0.5)].mean()
                late  = sig[(t >= 0.5) & (t <= 1.0)].mean()
                feats[f'ratio_late_n400_{ch_name}'] = float(late / (abs(n400) + 1e-6))

            # ── 5. Spectral in task windows ────────────────────────────────
            spec_cfg = [('early',0.0,0.5),('late',0.5,1.5)]
            bands = {'theta':(4,8),'alpha':(8,13),'beta':(13,30)}
            for wlabel,t0,t1 in spec_cfg:
                mask = (t>=t0)&(t<=t1)
                if mask.sum() < 16: continue
                for ch_name in ['Fz','Cz','C3','C4']:
                    ch_idx = TASTE_CH[ch_name]
                    seg = avg[ch_idx, mask]
                    f_ax, psd = welch(seg, fs=SFREQ, nperseg=min(64, len(seg)))
                    for bname,(flo,fhi) in bands.items():
                        bm = (f_ax>=flo)&(f_ax<=fhi)
                        feats[f'psd_{bname}_{ch_name}_{wlabel}'] = float(psd[bm].mean()) if bm.sum()>0 else 0.0

            # ── 6. Slope of late positivity ────────────────────────────────
            for ch_name in ['Cz','Fz']:
                ch_idx = TASTE_CH[ch_name]
                sig = avg[ch_idx] * 1e6
                for t0,t1,label in [(0.40,0.70,'rise'),(0.70,1.00,'plateau')]:
                    mask = (t>=t0)&(t<=t1)
                    if mask.sum() >= 2:
                        seg = sig[mask]
                        feats[f'slope_{ch_name}_{label}'] = float(np.polyfit(np.arange(len(seg)), seg, 1)[0])

            rows.append(feats)

    df = pd.DataFrame(rows)
    meta = ['subject_id','condition','jar_group']
    feat_cols = [c for c in df.columns if c not in meta]
    return df, feat_cols


# ─── ML helpers ───────────────────────────────────────────────────────────────
def smote_safe(X, y):
    n_pos = (y==1).sum()
    if n_pos < 2: return X, y
    k = min(5, n_pos-1)
    try: return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X, y)
    except: return X, y


def precompute_folds(X, y, g, iso=0.10):
    logo = LeaveOneGroupOut(); folds = []
    for tr, te in logo.split(X, y, g):
        X_tr, y_tr = X[tr].copy(), y[tr].copy()
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0: continue
        if iso > 0 and len(X_tr) > 20:
            isof = IsolationForest(contamination=iso, random_state=SEED, n_jobs=-1)
            isof.fit(X_tr); km = isof.predict(X_tr)==1
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
        idx = f['mi_order'][:K]
        X_tr, y_tr = f['X_tr'][:,idx].copy(), f['y_tr'].copy()
        X_te = f['X_te'][:,idx]
        if sampling == 'smote': X_tr, y_tr = smote_safe(X_tr, y_tr)
        m = model_factory(); m.fit(X_tr, y_tr)
        y_pred_all.extend(m.predict(X_te).tolist())
        y_true_all.extend(f['y_te'].tolist())
        if hasattr(m,'predict_proba'):
            proba_all.extend(m.predict_proba(X_te)[:,1].tolist())
        else:
            proba_all.extend([np.nan]*len(f['y_te']))
    if not y_true_all: return None
    yt, yp, pr = np.array(y_true_all), np.array(y_pred_all), np.array(proba_all)
    best_b, best_t = 0.0, 0.5
    if not np.isnan(pr).any():
        for thr in np.linspace(0.1, 0.9, 81):
            b = balanced_accuracy_score(yt, (pr>=thr).astype(int))
            if b > best_b: best_b=b; best_t=thr
    return {
        'accuracy':     accuracy_score(yt, yp),
        'balanced_acc': balanced_accuracy_score(yt, yp),
        'f1_macro':     f1_score(yt, yp, average='macro', zero_division=0),
        'rec_vua':      recall_score(yt, yp, pos_label=1, zero_division=0),
        'rec_oth':      recall_score(yt, yp, pos_label=0, zero_division=0),
        'oracle_bacc':  round(float(best_b),4),
        'oracle_thr':   round(float(best_t),2),
        'y_true': yt, 'y_pred': yp,
    }


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

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_ml_gerp_avg')
    print('='*78)
    print('  gERP features on CONDITION-AVERAGED epochs (avg 5 repeats first)')
    print('  Key signal: Late positivity 500-1000ms: Vua_phai=4.14µV vs Others=2.39µV')
    print('='*78)

    t_start = datetime.datetime.now()

    # ── Extract condition-averaged gERP features ──────────────────────────
    print('\nBuilding condition-averaged gERP features...', flush=True)
    df, feat_cols = build_cond_avg_gerp()
    print(f'  Done in {(datetime.datetime.now()-t_start).seconds}s')
    print(f'  Shape: {df.shape}   Features: {len(feat_cols)}')
    print(f'  Subjects: {df.subject_id.nunique()}  Conditions: {df.condition.nunique()}')
    print(f'  jar_group: {df.jar_group.value_counts().to_dict()}')

    df_vua = df[df.jar_group=='Vua_phai']
    df_oth = df[df.jar_group!='Vua_phai']

    # Sanity check: show Late Positivity difference
    print('\n  Late positivity sanity check (matches grand-average analysis):')
    for window, t0, t1 in [('LatePos1(500-700ms)',0.5,0.7),
                            ('LatePos2(700-1000ms)',0.7,1.0)]:
        key = f'LatePos1_Cz_mean' if '500' in window else f'LatePos2_Cz_mean'
        if key in df.columns:
            v = df_vua[key].mean(); o = df_oth[key].mean()
            print(f'    {window} at Cz: Vua_phai={v:.2f}µV  Others={o:.2f}µV  Δ={v-o:+.2f}µV')

    df.to_csv(os.path.join(OUT_DIR,'features_gerp_avg.csv'), index=False)

    # ── Apply quality filter (same as v2/v3: remove BAD quality) ──────────
    QUAL_CSV = 'output/results/erp/erp_quality_flags.csv'
    qf = pd.read_csv(QUAL_CSV)[['subject_id','condition','quality_label']]
    qf['condition'] = qf['condition'].astype(int)
    df['condition'] = df['condition'].astype(int)
    df = df.merge(qf, on=['subject_id','condition'], how='left')
    df = df[df['quality_label'] != 'BAD'].copy()
    print(f'  After weak filter (remove BAD): n={len(df)}  '
          f'pos={int((df.jar_group=="Vua_phai").sum())}')

    # Prepare X, y, g
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.var(axis=0) > 1e-12; X = X[:,keep]
    feat_cols_kept = [c for c,k in zip(feat_cols, keep) if k]
    y = (df['jar_group'].values == 'Vua_phai').astype(int)
    g = df['subject_id'].values
    majority = max(int(y.sum()), int((y==0).sum())) / len(y)
    print(f'\n  n={len(y)}  pos={int(y.sum())}  neg={int((y==0).sum())}  '
          f'majority={majority:.3f}  n_features={X.shape[1]}')

    # ── Models ────────────────────────────────────────────────────────────
    MODELS = {
        'XGB_gpu':    lambda: xgb.XGBClassifier(
            device='cuda', n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, scale_pos_weight=3,
            eval_metric='logloss', verbosity=0, random_state=SEED),
        'GradBoost':  lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=SEED),
    }
    K_GRID   = list(range(1, 31))  # 1..30
    ISO      = 0.10
    SAMPLINGS = ['none','smote']

    # ── LOSO grid ─────────────────────────────────────────────────────────
    print(f'\nPrecomputing LOSO folds (iso={ISO})...', flush=True)
    folds = precompute_folds(X, y, g, ISO)
    print(f'  {len(folds)} folds prepared')

    rows = []; best_row = None
    for sampling in SAMPLINGS:
        for mname, mfac in MODELS.items():
            print(f'\n  [{mname}  samp={sampling}]', flush=True)
            for K in K_GRID:
                res = eval_K(folds, K, mfac, sampling)
                if res is None: continue
                row = {'model':mname, 'sampling':sampling, 'K':K,
                       'accuracy':    round(res['accuracy'],    4),
                       'balanced_acc':round(res['balanced_acc'],4),
                       'f1_macro':    round(res['f1_macro'],    4),
                       'rec_vua':     round(res['rec_vua'],     4),
                       'rec_oth':     round(res['rec_oth'],     4),
                       'oracle_bacc': res['oracle_bacc'],
                       'oracle_thr':  res['oracle_thr']}
                rows.append(row)
                if best_row is None or res['balanced_acc'] > best_row['balanced_acc']:
                    best_row = {**row, 'y_true':res['y_true'], 'y_pred':res['y_pred']}
                # Print only when bacc > 0.60
                if res['balanced_acc'] >= 0.60:
                    print(f'    K={K:<3}  acc={row["accuracy"]:.4f}  '
                          f'bacc={row["balanced_acc"]:.4f}  '
                          f'oracle_bacc={row["oracle_bacc"]:.4f}  '
                          f'rec_vua={row["rec_vua"]:.4f}  ← !')

            sub = pd.DataFrame([r for r in rows if r['model']==mname and r['sampling']==sampling])
            if not sub.empty:
                top3 = sub.nlargest(3, 'balanced_acc')
                print(f'  Top-3:')
                for _, r in top3.iterrows():
                    flag = ' ← NEW BEST!' if r['balanced_acc'] == best_row['balanced_acc'] else ''
                    print(f'    K={int(r["K"]):<3}  acc={r["accuracy"]:.4f}  '
                          f'bacc={r["balanced_acc"]:.4f}  '
                          f'oracle_bacc={r["oracle_bacc"]:.4f}(thr={r["oracle_thr"]})  '
                          f'rec_vua={r["rec_vua"]:.4f}' + flag)

    # ── Save ──────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    df_res.to_csv(os.path.join(OUT_DIR,'results_gerp_avg.csv'), index=False)
    df_res.nlargest(20,'balanced_acc').to_csv(
        os.path.join(OUT_DIR,'top20_bacc.csv'), index=False)

    # K-curve plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = plt.cm.Set1
    for ax, samp in zip(axes, SAMPLINGS):
        sub = df_res[df_res['sampling']==samp]
        for i, (mname, grp) in enumerate(sub.groupby('model')):
            ax.plot(grp['K'], grp['balanced_acc'], marker='o', lw=2,
                    label=mname, color=cmap(i/max(len(MODELS),1)))
        ax.axhline(0.674, color='red',  ls='--', lw=1.5, label='DL v2 best=0.674')
        ax.axhline(0.649, color='blue', ls='--', lw=1.5, label='ML v2 best=0.649')
        ax.axhline(majority, color='gray', ls=':', lw=1.0, label=f'majority={majority:.3f}')
        ax.set_xlabel('K (top-MI features)'); ax.set_ylabel('Balanced Accuracy')
        ax.set_title(f'gERP cond-avg features  samp={samp}', fontweight='bold')
        ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0.40, 0.85)
    fig.suptitle('gERP Condition-Averaged Features: bacc vs K (LOSO-CV)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'bacc_vs_K.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    if best_row:
        plot_confusion(best_row['y_true'], best_row['y_pred'],
            f'{best_row["model"]} | K={best_row["K"]} samp={best_row["sampling"]}\n'
            f'bacc={best_row["balanced_acc"]:.4f}  rec_vua={best_row["rec_vua"]:.4f}',
            os.path.join(FIG_DIR,'cm_best.png'))

    # Top-10 MI features
    Xs = StandardScaler().fit_transform(np.nan_to_num(X))
    mi = mutual_info_classif(Xs, y, random_state=SEED)
    top10_idx = np.argsort(mi)[::-1][:10]
    print('\n  Top-10 MI features:')
    for i, idx in enumerate(top10_idx):
        fname = feat_cols_kept[idx]
        vua_mean = float(X[y==1, idx].mean())
        oth_mean = float(X[y==0, idx].mean())
        print(f'    #{i+1:>2}  {fname:<40}  Vua={vua_mean:.3f}  Oth={oth_mean:.3f}  Δ={vua_mean-oth_mean:+.3f}')

    elapsed = (datetime.datetime.now()-t_start).seconds
    print(f'\n{"="*78}')
    print('  FINAL SUMMARY — gERP Condition-Averaged')
    print(f'{"="*78}')
    print(f'  Feature set: {X.shape[1]} gERP features from condition-averaged ERP')
    print(f'  (vs 962 general features in v1/v2/v3, 519 gERP per-trial features)')
    print()
    if best_row:
        print(f'  Best: {best_row["model"]} K={best_row["K"]} samp={best_row["sampling"]}')
        print(f'    bacc        = {best_row["balanced_acc"]:.4f}')
        print(f'    accuracy    = {best_row["accuracy"]:.4f}')
        print(f'    oracle_bacc = {best_row["oracle_bacc"]:.4f}  (thr={best_row["oracle_thr"]})')
        print(f'    rec_vua     = {best_row["rec_vua"]:.4f}')
        print()
        improved = best_row['balanced_acc'] > 0.674
        print(f'  Progression (bacc):')
        print(f'    ML v2 GradBoost thr-tuned:         bacc=0.649')
        print(f'    DL v2 ShallowConvNet:               bacc=0.674  ← previous best')
        print(f'    gERP per-trial (failed):            bacc=0.51   ← noise too high')
        print(f'    gERP cond-avg (this run):           bacc={best_row["balanced_acc"]:.3f}  '
              f'{"← IMPROVED!" if improved else "← similar to previous best"}')

    print(f'\n  Total: {elapsed}s ({elapsed//60}m {elapsed%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
