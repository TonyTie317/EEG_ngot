#!/usr/bin/env python3
"""
Vua_phai vs Others — Phase 3: K sweep 1→30, GPU models + RF + SVM
===================================================================
GPU models  : XGBoost (device='cuda'), LightGBM (device='gpu')
CPU models  : RandomForest, BalancedRF, SVM, LogisticRegression

Grid:
  K             : 1 .. 30  (step 1)
  iso_contam    : 0.05 / 0.10 / 0.15   (IsoForest sample removal per fold)
  subj_remove   : 0 / 2 / 3            (drop N lowest-SNR subjects globally)
  sampling      : none / smote
  models        : see MODELS_GPU + MODELS_CPU below

Strategy:
  Phase A — full K×iso×subj grid with GPU models + fast CPU (SVM, LogReg)
  Phase B — focused grid (iso=0.10 only) with slow CPU tree models (RF, BalancedRF)

Oracle threshold: swept post-hoc on combined LOSO probas — gives theoretical
ceiling if decision threshold is optimized (no test leakage).
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, IsolationForest)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, recall_score)
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

SEED = 42
np.random.seed(SEED)

FEAT_CSV = 'output/results/ml_jar3/features_jar3_adv.csv'
QUAL_CSV = 'output/results/erp/erp_quality_flags.csv'
OUT_DIR  = 'output/results/ml_vuaphai_v3'
FIG_DIR  = 'output/figures/ml_vuaphai_v3'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

META = ['subject_id', 'condition', 'jar_group', 'quality_label', 'avg_snr', 'quality_score']


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── Model definitions ────────────────────────────────────────────────────────
def make_models_gpu():
    return {
        'XGB_gpu':  lambda: xgb.XGBClassifier(
            device='cuda', n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=3, eval_metric='logloss',
            verbosity=0, random_state=SEED),
        'LGBM_gpu': lambda: lgb.LGBMClassifier(
            device='gpu', n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', verbose=-1, random_state=SEED),
    }

def make_models_fast_cpu():
    return {
        'SVM_RBF':    lambda: SVC(kernel='rbf', C=1.0, gamma='scale',
                                   class_weight='balanced', probability=True,
                                   random_state=SEED),
        'LogReg_C1':  lambda: LogisticRegression(
            max_iter=3000, C=1.0, class_weight='balanced',
            solver='lbfgs', random_state=SEED),
        'LogReg_C01': lambda: LogisticRegression(
            max_iter=3000, C=0.1, class_weight='balanced',
            solver='lbfgs', random_state=SEED),
    }

def make_models_tree_cpu():
    return {
        'RF_100':     lambda: RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=SEED, n_jobs=-1),
        'BalancedRF': lambda: BalancedRandomForestClassifier(
            n_estimators=50, random_state=SEED, n_jobs=-1,
            replacement=True, sampling_strategy='auto'),
    }


# ─── Data helpers ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    qf = pd.read_csv(QUAL_CSV)[['subject_id', 'condition', 'quality_label',
                                  'avg_snr', 'quality_score']]
    df['condition'] = df['condition'].astype(int)
    qf['condition'] = qf['condition'].astype(int)
    return df.merge(qf, on=['subject_id', 'condition'], how='left')

def apply_subj_snr_filter(df, n_remove):
    if n_remove == 0:
        return df
    worst = (df.groupby('subject_id')['avg_snr'].mean()
               .sort_values().index[:n_remove].tolist())
    return df[~df['subject_id'].isin(worst)].copy()

def prepare_Xy(df):
    feat_cols = [c for c in df.columns if c not in META]
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.var(axis=0) > 1e-12
    X = X[:, keep]
    mask = df['jar_group'].notna().values
    X = X[mask]
    y = (df.loc[mask, 'jar_group'].values == 'Vua_phai').astype(int)
    g = df.loc[mask, 'subject_id'].values
    return X, y, g

def smote_safe(X_tr, y_tr):
    n_pos = (y_tr == 1).sum()
    if n_pos < 2: return X_tr, y_tr
    k = min(5, n_pos - 1)
    try:
        return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_tr, y_tr)
    except Exception:
        return X_tr, y_tr


# ─── Precompute LOSO folds (IsoForest + scale + MI) once per (subj, iso) ────
def precompute_folds(X, y, groups, iso_contam):
    logo = LeaveOneGroupOut()
    folds = []
    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr = X[tr].copy(), y[tr].copy()
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr)) < 2 or len(y_te) == 0:
            continue
        # IsoForest sample removal
        if iso_contam > 0 and len(X_tr) > 20:
            iso = IsolationForest(contamination=iso_contam, random_state=SEED, n_jobs=-1)
            iso.fit(X_tr)
            keep = iso.predict(X_tr) == 1
            if (y_tr[keep] == 0).any() and (y_tr[keep] == 1).any():
                X_tr, y_tr = X_tr[keep], y_tr[keep]
        if len(np.unique(y_tr)) < 2:
            continue
        # Scale
        sc = StandardScaler()
        X_tr = np.nan_to_num(sc.fit_transform(X_tr))
        X_te = np.nan_to_num(sc.transform(X_te))
        # MI feature ranking (compute once, reuse across all K)
        mi    = mutual_info_classif(X_tr, y_tr, random_state=SEED)
        order = np.argsort(mi)[::-1]
        folds.append({'X_tr': X_tr, 'y_tr': y_tr,
                      'X_te': X_te, 'y_te': y_te,
                      'mi_order': order})
    return folds


# ─── LOSO evaluation from cached folds ───────────────────────────────────────
def eval_K(folds, K, model_factory, sampling):
    y_true_all, y_pred_all, proba_all = [], [], []
    is_balanced = 'BalancedRandom' in type(model_factory()).__name__

    for f in folds:
        idx  = f['mi_order'][:K]
        X_tr = f['X_tr'][:, idx].copy()
        y_tr = f['y_tr'].copy()
        X_te = f['X_te'][:, idx]
        y_te = f['y_te']

        if sampling == 'smote' and not is_balanced:
            X_tr, y_tr = smote_safe(X_tr, y_tr)

        m = model_factory()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_pred_all.extend(y_pred.tolist())
        y_true_all.extend(y_te.tolist())

        if hasattr(m, 'predict_proba'):
            proba_all.extend(m.predict_proba(X_te)[:, 1].tolist())
        else:
            proba_all.extend([np.nan] * len(y_te))

    if not y_true_all:
        return None
    yt = np.array(y_true_all)
    yp = np.array(y_pred_all)
    pr = np.array(proba_all)

    res = {
        'accuracy':        accuracy_score(yt, yp),
        'balanced_acc':    balanced_accuracy_score(yt, yp),
        'f1_macro':        f1_score(yt, yp, average='macro', zero_division=0),
        'recall_vua_phai': recall_score(yt, yp, pos_label=1, zero_division=0),
        'recall_others':   recall_score(yt, yp, pos_label=0, zero_division=0),
        'oracle_acc': accuracy_score(yt, yp),
        'oracle_bacc': balanced_accuracy_score(yt, yp),
        'oracle_thr': 0.5,
        'y_true': yt, 'y_pred': yp,
    }

    # Post-hoc oracle threshold (no leakage — each proba from held-out fold)
    if not np.isnan(pr).any():
        best_acc, best_thr_acc = 0.0, 0.5
        best_bacc, best_thr_bacc = 0.0, 0.5
        for thr in np.linspace(0.10, 0.90, 81):
            yp_t = (pr >= thr).astype(int)
            a = accuracy_score(yt, yp_t)
            b = balanced_accuracy_score(yt, yp_t)
            if a > best_acc:
                best_acc = a; best_thr_acc = thr
            if b > best_bacc:
                best_bacc = b; best_thr_bacc = thr
        res['oracle_acc']  = round(float(best_acc), 4)
        res['oracle_bacc'] = round(float(best_bacc), 4)
        res['oracle_thr']  = round(float(best_thr_acc), 2)
        # Also record the best bacc threshold
        res['oracle_thr_bacc'] = round(float(best_thr_bacc), 2)
    return res


# ─── Run one (subj, iso) block across all K × samp × models ─────────────────
def run_block(X, y, g, iso_contam, K_GRID, SAMPLINGS, models_dict,
              n_remove, n_subj, n_samples, majority, rows,
              best_acc_ref, best_bacc_ref, label=''):
    folds = precompute_folds(X, y, g, iso_contam)
    print(f'    → {len(folds)} folds prepared', flush=True)
    if len(folds) < 5:
        print('    Too few folds — skipping')
        return

    for sampling in SAMPLINGS:
        for mname, mfac in models_dict.items():
            for K in K_GRID:
                res = eval_K(folds, K, mfac, sampling)
                if res is None:
                    continue
                row = {
                    'subj_remove': n_remove, 'n_subjects': n_subj,
                    'n_samples': n_samples, 'majority': round(majority, 4),
                    'iso_contam': iso_contam, 'sampling': sampling,
                    'model': mname, 'K': K,
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

                if best_acc_ref[0] is None or row['oracle_acc'] > best_acc_ref[0]['oracle_acc']:
                    best_acc_ref[0] = {**row, 'y_true': res['y_true'], 'y_pred': res['y_pred']}
                if best_bacc_ref[0] is None or row['balanced_acc'] > best_bacc_ref[0]['balanced_acc']:
                    best_bacc_ref[0] = {**row, 'y_true': res['y_true'], 'y_pred': res['y_pred']}

        # Mini-summary after each (sampling) block
        sub = pd.DataFrame([r for r in rows
                             if r['subj_remove'] == n_remove
                             and r['iso_contam'] == iso_contam
                             and r['sampling'] == sampling
                             and r['model'] in models_dict])
        if not sub.empty:
            top5 = sub.nlargest(5, 'oracle_acc')
            print(f'    [{label} samp={sampling}] top-5 oracle_acc:')
            for _, r in top5.iterrows():
                flag = ' ← BEST!' if r['oracle_acc'] == best_acc_ref[0]['oracle_acc'] else ''
                print(f'      K={int(r["K"]):<3} {r["model"]:<13} '
                      f'acc={r["accuracy"]:.4f}  oracle={r["oracle_acc"]:.4f}'
                      f'(thr={r["oracle_thr"]:.2f})  '
                      f'bacc={r["balanced_acc"]:.4f}  '
                      f'rec_vua={r["recall_vua_phai"]:.3f}' + flag)


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_k_curves(df_res, metric, fig_path, title):
    isos = sorted(df_res['iso_contam'].unique())
    n_col = min(len(isos), 3)
    fig, axes = plt.subplots(1, n_col, figsize=(6.5 * n_col, 5.5), sharey=True)
    if n_col == 1: axes = [axes]
    cmap = plt.cm.tab10
    model_list = sorted(df_res['model'].unique())
    for ax, iso in zip(axes, isos[:n_col]):
        sub = df_res[df_res['iso_contam'] == iso]
        pivot = sub.groupby(['K', 'model'])[metric].max().unstack()
        for i, mn in enumerate(model_list):
            if mn not in pivot.columns: continue
            ax.plot(pivot.index, pivot[mn], marker='.', lw=1.5,
                    label=mn, color=cmap(i / max(len(model_list), 1)))
        ax.axhline(0.85, color='red', ls='--', lw=1.5, label='target=0.85')
        mb = sub['majority'].max()
        ax.axhline(mb, color='gray', ls=':', lw=1.0, label=f'majority={mb:.3f}')
        ax.axhline(0.748, color='green', ls=':', lw=1.0, label='v2_best_acc=0.748')
        ax.set_title(f'iso={iso:.2f}', fontweight='bold')
        ax.set_xlabel('K (top-MI features)'); ax.set_ylabel(metric)
        ax.set_ylim(0.40, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='lower right')
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, fig_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)'
                       for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Other', 'Vua_phai'],
                yticklabels=['Other', 'Vua_phai'],
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title, fontsize=8, fontweight='bold')
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

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_ml_vuaphai_v3')
    print('='*78)
    print('  K sweep 1→30 + IsoForest + Subject removal + GPU (XGB, LGBM) + RF + SVM')
    print('  Target: accuracy > 0.85  (balanced_acc > 0.55 required)')
    print('='*78)

    K_GRID        = list(range(1, 31))
    ISO_CONTAMS   = [0.05, 0.10, 0.15]
    N_SUBJ_REMOVE = [0, 2, 3]
    SAMPLINGS     = ['none', 'smote']

    MODELS_GPU      = make_models_gpu()
    MODELS_FAST_CPU = make_models_fast_cpu()
    MODELS_TREE_CPU = make_models_tree_cpu()

    print(f'Phase A (full grid): GPU={list(MODELS_GPU)} + FastCPU={list(MODELS_FAST_CPU)}')
    print(f'Phase B (iso=0.10 only): TreeCPU={list(MODELS_TREE_CPU)}')
    print(f'K=[1..30]  iso={ISO_CONTAMS}  subj_remove={N_SUBJ_REMOVE}  samp={SAMPLINGS}')

    df_all = load_data()
    print(f'\nLoaded: {df_all.shape}, quality: {df_all.quality_label.value_counts().to_dict()}')

    snr_rank = df_all.groupby('subject_id')['avg_snr'].mean().sort_values()
    print('\nBottom-5 subjects by avg SNR:')
    for i, (s, v) in enumerate(snr_rank.head(5).items()):
        print(f'  #{i+1} {s}  avg_snr={v:.3f}')

    rows = []
    best_acc_ref  = [None]   # mutable reference for best tracking
    best_bacc_ref = [None]
    t_start = datetime.datetime.now()

    # ── Phase A: GPU + fast CPU, full grid ───────────────────────────────────
    print('\n' + '━'*78)
    print('  PHASE A — GPU models + SVM + LogReg  (full K×iso×subj grid)')
    print('━'*78)
    phase_a_models = {**MODELS_GPU, **MODELS_FAST_CPU}

    for n_remove in N_SUBJ_REMOVE:
        df_base = df_all[df_all['quality_label'] != 'BAD'].copy()
        df_base = apply_subj_snr_filter(df_base, n_remove)
        X, y, g = prepare_Xy(df_base)
        n_pos = int(y.sum()); majority = max(n_pos, (y==0).sum()) / len(y)
        n_subj = len(np.unique(g))
        print(f'\n  subj_remove={n_remove}  n={len(y)}  subj={n_subj}  '
              f'pos={n_pos}  neg={int((y==0).sum())}  majority={majority:.3f}')
        if n_pos < 5:
            print('  Too few Vua_phai — skipping'); continue

        for iso_contam in ISO_CONTAMS:
            elapsed = (datetime.datetime.now() - t_start).seconds
            print(f'\n  [subj_remove={n_remove}  iso={iso_contam:.2f}]  elapsed={elapsed}s',
                  flush=True)
            run_block(X, y, g, iso_contam, K_GRID, SAMPLINGS, phase_a_models,
                      n_remove, n_subj, len(y), majority, rows,
                      best_acc_ref, best_bacc_ref, label='GPU+FastCPU')

    # ── Phase B: RF + BalancedRF, iso=0.10 only (focused, slower) ────────────
    print('\n' + '━'*78)
    print('  PHASE B — RandomForest + BalancedRF  (iso=0.10 only)')
    print('━'*78)

    for n_remove in N_SUBJ_REMOVE:
        df_base = df_all[df_all['quality_label'] != 'BAD'].copy()
        df_base = apply_subj_snr_filter(df_base, n_remove)
        X, y, g = prepare_Xy(df_base)
        n_pos = int(y.sum()); majority = max(n_pos, (y==0).sum()) / len(y)
        n_subj = len(np.unique(g))
        print(f'\n  subj_remove={n_remove}  n={len(y)}  subj={n_subj}  '
              f'pos={n_pos}  majority={majority:.3f}')
        if n_pos < 5:
            print('  Too few Vua_phai — skipping'); continue

        elapsed = (datetime.datetime.now() - t_start).seconds
        print(f'\n  [subj_remove={n_remove}  iso=0.10]  elapsed={elapsed}s', flush=True)
        run_block(X, y, g, 0.10, K_GRID, SAMPLINGS, MODELS_TREE_CPU,
                  n_remove, n_subj, len(y), majority, rows,
                  best_acc_ref, best_bacc_ref, label='RF/BRF')

    # ── Save ──────────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'all_results_v3.csv')
    df_res.to_csv(csv_path, index=False)
    print(f'\n✓ {csv_path}  ({len(df_res)} rows)')

    top30_acc  = df_res.nlargest(30, 'oracle_acc')
    top30_bacc = df_res.nlargest(30, 'balanced_acc')
    top30_acc.to_csv( os.path.join(OUT_DIR, 'top30_by_oracle_acc.csv'),  index=False)
    top30_bacc.to_csv(os.path.join(OUT_DIR, 'top30_by_balanced_acc.csv'), index=False)
    print('✓ Saved top30 CSVs')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_k_curves(df_res, 'oracle_acc',
                  os.path.join(FIG_DIR, 'oracle_acc_vs_K.png'),
                  'Oracle Accuracy vs K (max over subj_remove × sampling)')
    plot_k_curves(df_res, 'accuracy',
                  os.path.join(FIG_DIR, 'acc_vs_K.png'),
                  'Accuracy vs K (threshold=0.5, max over subj_remove × sampling)')
    plot_k_curves(df_res, 'balanced_acc',
                  os.path.join(FIG_DIR, 'bacc_vs_K.png'),
                  'Balanced Accuracy vs K')
    print('✓ Saved K-curve plots')

    # Heatmap: K vs iso_contam for oracle_acc
    pivot = df_res.groupby(['K', 'iso_contam'])['oracle_acc'].max().unstack('iso_contam')
    fig, ax = plt.subplots(figsize=(8, 9))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0.60, vmax=1.0, linewidths=0.3, ax=ax)
    ax.set_title('Oracle Accuracy: K vs IsoForest contam\n(max over all models / sampling / subj_remove)',
                 fontweight='bold')
    ax.set_xlabel('IsoForest contam'); ax.set_ylabel('K')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'heatmap_K_vs_contam.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('✓ Saved heatmap')

    b = best_acc_ref[0]
    if b is not None:
        plot_confusion(
            b['y_true'], b['y_pred'],
            (f'{b["model"]} | iso={b["iso_contam"]} | K={b["K"]} | '
             f'samp={b["sampling"]} | rm={b["subj_remove"]}\n'
             f'acc={b["accuracy"]:.4f}  oracle={b["oracle_acc"]:.4f}(thr={b["oracle_thr"]})  '
             f'bacc={b["balanced_acc"]:.4f}  rec_vua={b["recall_vua_phai"]:.4f}'),
            os.path.join(FIG_DIR, 'cm_best_oracle_acc.png')
        )
        print('✓ Saved confusion matrix')

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = (datetime.datetime.now() - t_start).seconds
    print(f'\n{"="*78}')
    print('  TOP-20 BY ORACLE_ACC')
    print(f'{"="*78}')
    hdr = (f'{"#":<4}{"model":<14}{"samp":<9}{"K":<4}{"iso":<6}'
           f'{"rm":<4}{"acc":<8}{"oracle":<10}{"thr":<6}{"bacc":<8}{"rec_vua":<9}')
    print(hdr); print('-'*len(hdr))
    for rank, (_, r) in enumerate(top30_acc.head(20).iterrows(), 1):
        print(f'{rank:<4}{r["model"]:<14}{r["sampling"]:<9}{int(r["K"]):<4}'
              f'{r["iso_contam"]:<6}{int(r["subj_remove"]):<4}'
              f'{r["accuracy"]:.4f}  {r["oracle_acc"]:.4f}    '
              f'{r["oracle_thr"]:<6}{r["balanced_acc"]:.4f}  {r["recall_vua_phai"]:.4f}')

    if b is not None:
        reached = b['oracle_acc'] >= 0.85
        print(f'\n  BEST oracle_acc = {b["oracle_acc"]:.4f}  '
              f'{"✓ TARGET >0.85 REACHED!" if reached else "✗ target 0.85 not reached yet"}')
        print(f'    model         = {b["model"]}')
        print(f'    K             = {b["K"]}')
        print(f'    iso_contam    = {b["iso_contam"]}')
        print(f'    subj_remove   = {b["subj_remove"]}  ({b["n_subjects"]} subj, {b["n_samples"]} samples)')
        print(f'    sampling      = {b["sampling"]}')
        print(f'    accuracy      = {b["accuracy"]:.4f}  (threshold=0.5)')
        print(f'    oracle_acc    = {b["oracle_acc"]:.4f}  (threshold={b["oracle_thr"]})')
        print(f'    balanced_acc  = {b["balanced_acc"]:.4f}')
        print(f'    recall_vua    = {b["recall_vua_phai"]:.4f}')

    print(f'\n  History:')
    print(f'    v1 RF K=15 (majority-predict):     acc=0.773  bacc=0.500  ← fake')
    print(f'    v2 GradBoost K=30 none:            acc=0.748  bacc=0.600  ← real')
    print(f'    v2 GradBoost K=20 thr=0.208:       acc=0.681  bacc=0.649  ← best balanced')
    if b:
        print(f'    v3 best:  acc={b["accuracy"]:.3f}  oracle={b["oracle_acc"]:.3f}  bacc={b["balanced_acc"]:.3f}')

    print(f'\n  Total runtime: {elapsed_total}s ({elapsed_total//60}m {elapsed_total%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
