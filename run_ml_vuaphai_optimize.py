#!/usr/bin/env python3
"""
Optimize ML — Vua_phai vs Others (binary classification)
=========================================================
Goal: push accuracy beyond strict-filter baseline (acc 0.778, LogReg K=5).

Strategies:
  1. Outlier handling
     a. standard       : StandardScaler
     b. std+wins       : StandardScaler + winsorize features at z=±3
     c. std+iforest    : StandardScaler + IsolationForest sample removal (cont=0.10)
     d. robust         : RobustScaler (median/IQR)
  2. Feature selection: MI top-K, K ∈ {3, 5, 8, 10, 15, 20, 30}
     (MI fitted INSIDE each LOSO fold — no leakage)
  3. Models: LogReg (C=0.1/1/10), SVM-RBF, RandomForest, GradBoost
  4. Filters: weak (drop BAD), strict (GOOD only), snr_q1

Optimization: cache MI scores per (filter, strategy, fold) so the inner
K × model loop reuses them — avoids recomputing MI for every K.
"""

import os, sys, warnings, datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix)

SEED = 42
np.random.seed(SEED)

FEAT_CSV = 'output/results/ml_jar3/features_jar3_adv.csv'
QUAL_CSV = 'output/results/erp/erp_quality_flags.csv'
OUT_DIR  = 'output/results/ml_vuaphai_optimize'
FIG_DIR  = 'output/figures/ml_vuaphai_optimize'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

META = ['subject_id', 'condition', 'jar_group']


# ──────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    qf = pd.read_csv(QUAL_CSV)[['subject_id', 'condition', 'quality_label',
                                 'avg_snr', 'quality_score']]
    df['condition'] = df['condition'].astype(int)
    qf['condition'] = qf['condition'].astype(int)
    df = df.merge(qf, on=['subject_id', 'condition'], how='left')
    return df


def apply_filter(df, name):
    if name == 'weak':
        return df[df['quality_label'] != 'BAD'].copy()
    if name == 'strict':
        return df[df['quality_label'] == 'GOOD'].copy()
    if name == 'snr_q1':
        thr = df['avg_snr'].quantile(0.25)
        return df[df['avg_snr'] >= thr].copy()
    raise ValueError(name)


def prepare_Xy(df):
    feat_cols = [c for c in df.columns
                 if c not in META + ['quality_label', 'avg_snr', 'quality_score']]
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    var = X.var(axis=0)
    keep = var > 1e-12
    feat_cols = [c for c, k in zip(feat_cols, keep) if k]
    X = X[:, keep]
    mask = df['jar_group'].notna().values
    X = X[mask]
    y = (df.loc[mask, 'jar_group'].values == 'Vua_phai').astype(int)
    g = df.loc[mask, 'subject_id'].values
    return X, y, g, feat_cols


# ──────────────────────────────────────────────────────────────────────────────
def preprocess_fold(X_tr, y_tr, X_te, scaler_kind, winsorize, iforest_contam):
    """Apply outlier handling + scaling to a single fold."""
    if iforest_contam > 0 and len(X_tr) > 30:
        iso = IsolationForest(contamination=iforest_contam,
                               random_state=SEED, n_jobs=-1)
        iso.fit(X_tr)
        keep = iso.predict(X_tr) == 1
        if (y_tr[keep] == 0).any() and (y_tr[keep] == 1).any():
            X_tr = X_tr[keep]; y_tr = y_tr[keep]
    sc = RobustScaler() if scaler_kind == 'robust' else StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    if winsorize:
        X_tr = np.clip(X_tr, -3, 3)
        X_te = np.clip(X_te, -3, 3)
    X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
    X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)
    return X_tr, y_tr, X_te


def precompute_folds(X, y, groups, strategy):
    """For each LOSO fold, preprocess + compute MI once. Returns list of dicts."""
    logo = LeaveOneGroupOut()
    folds = []
    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr_orig = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr_orig)) < 2 or len(y_te) == 0:
            continue
        X_tr_pp, y_tr_pp, X_te_pp = preprocess_fold(
            X_tr.copy(), y_tr_orig.copy(), X_te.copy(),
            strategy['scaler'], strategy['wins'], strategy['iso']
        )
        if len(np.unique(y_tr_pp)) < 2:
            continue
        mi = mutual_info_classif(X_tr_pp, y_tr_pp, random_state=SEED)
        order = np.argsort(mi)[::-1]
        folds.append({
            'X_tr': X_tr_pp, 'y_tr': y_tr_pp,
            'X_te': X_te_pp, 'y_te': y_te,
            'mi_order': order,
        })
    return folds


def eval_K_model(folds, K, model_factory):
    y_true_all, y_pred_all = [], []
    for f in folds:
        idx = f['mi_order'][:K]
        X_tr = f['X_tr'][:, idx]; X_te = f['X_te'][:, idx]
        m = model_factory()
        m.fit(X_tr, f['y_tr'])
        y_pred = m.predict(X_te)
        y_true_all.extend(f['y_te'].tolist())
        y_pred_all.extend(y_pred.tolist())
    if not y_true_all:
        return None
    y_true = np.array(y_true_all); y_pred = np.array(y_pred_all)
    return {
        'accuracy':     accuracy_score(y_true, y_pred),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        'f1':           f1_score(y_true, y_pred, average='macro',
                                  zero_division=0),
        'y_true':       y_true, 'y_pred': y_pred,
    }


# ──────────────────────────────────────────────────────────────────────────────
def get_models():
    return {
        'LogReg_C0.1': lambda: LogisticRegression(max_iter=3000, C=0.1,
                                                    class_weight='balanced',
                                                    solver='lbfgs',
                                                    random_state=SEED),
        'LogReg_C1':   lambda: LogisticRegression(max_iter=3000, C=1.0,
                                                    class_weight='balanced',
                                                    solver='lbfgs',
                                                    random_state=SEED),
        'LogReg_C10':  lambda: LogisticRegression(max_iter=3000, C=10.0,
                                                    class_weight='balanced',
                                                    solver='lbfgs',
                                                    random_state=SEED),
        'SVM_RBF':     lambda: SVC(kernel='rbf', C=1.0, gamma='scale',
                                    class_weight='balanced',
                                    random_state=SEED),
        'RForest':     lambda: RandomForestClassifier(n_estimators=400,
                                                        max_depth=None,
                                                        class_weight='balanced',
                                                        random_state=SEED,
                                                        n_jobs=-1),
        'GradBoost':   lambda: GradientBoostingClassifier(n_estimators=200,
                                                            learning_rate=0.05,
                                                            max_depth=3,
                                                            random_state=SEED),
    }


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(OUT_DIR, 'logs', f'run_{ts}.log')
    latest = os.path.join(OUT_DIR, 'run.log')
    log_fh = open(log_path, 'w'); latest_fh = open(latest, 'w')
    sys.stdout = Tee(sys.__stdout__, log_fh, latest_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh, latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] '
          f'logging to {log_path}')
    print('='*78)
    print('  VUA_PHAI vs OTHERS — outlier-aware optimization')
    print('='*78)

    df_all = load_data()
    print(f'Loaded {df_all.shape}, quality: '
          f'{df_all.quality_label.value_counts().to_dict()}')

    filters    = ['weak', 'strict', 'snr_q1']
    K_grid     = [3, 5, 8, 10, 15, 20, 30]
    models     = get_models()
    strategies = [
        {'name': 'standard',    'scaler': 'standard', 'wins': False, 'iso': 0.0},
        {'name': 'std+wins',    'scaler': 'standard', 'wins': True,  'iso': 0.0},
        {'name': 'std+iforest', 'scaler': 'standard', 'wins': False, 'iso': 0.10},
        {'name': 'robust',      'scaler': 'robust',   'wins': False, 'iso': 0.0},
    ]

    rows = []; best_overall = None
    total = len(filters) * len(strategies)
    step = 0

    for filt in filters:
        df = apply_filter(df_all, filt)
        X, y, g, feat_cols = prepare_Xy(df)
        n_pos = int(y.sum()); n_neg = int((y == 0).sum())
        majority = max(n_pos, n_neg) / len(y)
        n_subj = len(np.unique(g))
        print(f'\n── Filter [{filt}]  n={len(y)}  subj={n_subj}  '
              f'pos={n_pos}  neg={n_neg}  majority_baseline={majority:.3f} ──')

        for strat in strategies:
            step += 1
            print(f'  [{step}/{total}] precomputing folds for '
                  f'strategy={strat["name"]} ...', flush=True)
            folds = precompute_folds(X, y, g, strat)
            print(f'    → {len(folds)} folds prepared')

            for K in K_grid:
                for mname, mfac in models.items():
                    res = eval_K_model(folds, K, mfac)
                    if res is None: continue
                    row = {
                        'filter': filt, 'n_samples': len(y),
                        'n_subjects': n_subj, 'majority_baseline': majority,
                        'strategy': strat['name'], 'K': K, 'model': mname,
                        'accuracy': res['accuracy'],
                        'balanced_acc': res['balanced_acc'],
                        'f1': res['f1'],
                    }
                    rows.append(row)
                    if (best_overall is None
                        or res['accuracy'] > best_overall['accuracy']):
                        best_overall = {**row, 'y_true': res['y_true'],
                                        'y_pred': res['y_pred']}

            # Per-strategy top-3
            sub = pd.DataFrame([r for r in rows if r['filter'] == filt
                                and r['strategy'] == strat['name']])
            top3 = sub.sort_values('accuracy', ascending=False).head(3)
            for _, r in top3.iterrows():
                print(f'      K={r["K"]:<3} {r["model"]:<14} '
                      f'acc={r["accuracy"]:.3f} bacc={r["balanced_acc"]:.3f} '
                      f'f1={r["f1"]:.3f}')

    # Save full results
    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'all_results.csv')
    df_res.to_csv(csv_path, index=False)
    print(f'\n✓ Saved: {csv_path}  ({len(df_res)} rows)')

    best_per_filter = (df_res.sort_values('accuracy', ascending=False)
                              .groupby('filter').head(1)
                              .sort_values('accuracy', ascending=False))
    bp_path = os.path.join(OUT_DIR, 'best_per_filter.csv')
    best_per_filter.to_csv(bp_path, index=False)
    print(f'✓ Saved: {bp_path}')

    # ── Plot: accuracy vs K (max over strategies), per filter ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    for ax, filt in zip(axes, filters):
        sub = df_res[df_res['filter'] == filt]
        pivot = sub.groupby(['K', 'model'])['accuracy'].max().unstack()
        cmap = plt.cm.tab10
        for i, mname in enumerate(pivot.columns):
            ax.plot(pivot.index, pivot[mname], marker='o', linewidth=2,
                    label=mname, color=cmap(i / max(len(pivot.columns), 1)))
        ax.axhline(0.5, color='red', linestyle='--', linewidth=1.0,
                   label='chance=0.50')
        mb = sub['majority_baseline'].iloc[0]
        ax.axhline(mb, color='gray', linestyle=':', linewidth=1.0,
                   label=f'majority={mb:.2f}')
        ax.set_xlabel('Top-K features (MI)')
        ax.set_ylabel('Accuracy' if filt == filters[0] else '')
        ax.set_title(f'{filt}  (n={int(sub["n_samples"].iloc[0])})',
                     fontweight='bold')
        ax.set_ylim(0.30, 0.95)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='lower right')
    fig.suptitle('Vua_phai vs Others — accuracy sweep (best strategy per K×model, LOSO-CV)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'accuracy_sweep.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved: {fig_path}')

    # Confusion matrix for overall best
    cm = confusion_matrix(best_overall['y_true'], best_overall['y_pred'])
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)'
                       for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Other', 'Vua_phai'],
                yticklabels=['Other', 'Vua_phai'],
                vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'BEST: {best_overall["filter"]} | '
                  f'{best_overall["strategy"]} | K={best_overall["K"]} | '
                  f'{best_overall["model"]}\n'
                  f'acc={best_overall["accuracy"]:.3f} | '
                  f'bacc={best_overall["balanced_acc"]:.3f} | '
                  f'f1={best_overall["f1"]:.3f}',
                  fontsize=10, fontweight='bold')
    fig.tight_layout()
    cm_path = os.path.join(FIG_DIR, 'best_confusion_matrix.png')
    fig.savefig(cm_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved: {cm_path}')

    # Summary
    print('\n' + '='*78)
    print('  BEST PER FILTER (by accuracy)')
    print('='*78)
    print(f'{"filter":<10}{"n":<6}{"strategy":<14}{"K":<5}{"model":<14}'
          f'{"acc":<8}{"bacc":<8}{"f1":<8}')
    for _, r in best_per_filter.iterrows():
        print(f'{r["filter"]:<10}{int(r["n_samples"]):<6}'
              f'{r["strategy"]:<14}{int(r["K"]):<5}{r["model"]:<14}'
              f'{r["accuracy"]:.3f}   {r["balanced_acc"]:.3f}   {r["f1"]:.3f}')

    print('\n  🏆 OVERALL BEST:')
    print(f'    filter       = {best_overall["filter"]}')
    print(f'    strategy     = {best_overall["strategy"]}')
    print(f'    K            = {best_overall["K"]}')
    print(f'    model        = {best_overall["model"]}')
    print(f'    accuracy     = {best_overall["accuracy"]:.4f}')
    print(f'    balanced_acc = {best_overall["balanced_acc"]:.4f}')
    print(f'    f1_macro     = {best_overall["f1"]:.4f}')
    print(f'    n_samples    = {int(best_overall["n_samples"])}')

    print('\n  📊 Class balance per filter:')
    for filt in filters:
        d = apply_filter(df_all, filt)
        y_filt = (d['jar_group'].values == 'Vua_phai').astype(int)
        mb = max((y_filt == 1).mean(), (y_filt == 0).mean())
        print(f'    {filt:<10}: pos={int((y_filt==1).sum()):<3} '
              f'neg={int((y_filt==0).sum()):<3} majority_baseline={mb:.3f}')

    print('='*78 + '\n')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
