#!/usr/bin/env python3
"""
Vua_phai vs Others — Phase 2 optimization
==========================================
Builds on v1 best: weak filter + std+iforest + K=15 + RandomForest
  acc=0.773, balanced_acc=0.558, f1_macro=0.548

New strategies:
  1. SMOTE / ADASYN / BorderlineSMOTE  — oversample minority in each train fold
  2. BalancedRandomForest / EasyEnsemble (imblearn) — imbalance-aware ensembles
  3. Threshold tuning  — CV-estimated decision threshold per fold (no test leakage)
  4. Stacking          — RF + SVM + LogReg OOF proba → meta-LogReg

Primary metric: balanced_accuracy (not raw accuracy, which is dominated by majority class).
"""

import os, sys, warnings, datetime, logging
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, recall_score)
from sklearn.base import clone

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

SEED = 42
np.random.seed(SEED)

FEAT_CSV = 'output/results/ml_jar3/features_jar3_adv.csv'
QUAL_CSV = 'output/results/erp/erp_quality_flags.csv'
OUT_DIR  = 'output/results/ml_vuaphai_v2'
FIG_DIR  = 'output/figures/ml_vuaphai_v2'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

META = ['subject_id', 'condition', 'jar_group', 'quality_label', 'avg_snr', 'quality_score']


# ─── Logging ─────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── Data loading ─────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    qf = pd.read_csv(QUAL_CSV)[['subject_id', 'condition', 'quality_label',
                                 'avg_snr', 'quality_score']]
    df['condition'] = df['condition'].astype(int)
    qf['condition'] = qf['condition'].astype(int)
    return df.merge(qf, on=['subject_id', 'condition'], how='left')


def prepare_Xy(df):
    feat_cols = [c for c in df.columns if c not in META]
    X = df[feat_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = X.var(axis=0) > 1e-12
    X = X[:, keep]
    feat_cols = [c for c, k in zip(feat_cols, keep) if k]
    mask = df['jar_group'].notna().values
    X = X[mask]
    y = (df.loc[mask, 'jar_group'].values == 'Vua_phai').astype(int)
    g = df.loc[mask, 'subject_id'].values
    return X, y, g, feat_cols


# ─── Sampling helpers ─────────────────────────────────────────────────────────
def apply_sampling(X_tr, y_tr, method, seed=SEED):
    """Oversample minority class. Returns augmented (X, y)."""
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    if n_pos < 2 or method == 'none':
        return X_tr, y_tr
    k = min(5, n_pos - 1)
    if k < 1:
        return X_tr, y_tr
    try:
        if method == 'smote':
            sampler = SMOTE(random_state=seed, k_neighbors=k)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=seed, n_neighbors=k)
        elif method == 'borderline':
            sampler = BorderlineSMOTE(random_state=seed, k_neighbors=k)
        else:
            return X_tr, y_tr
        return sampler.fit_resample(X_tr, y_tr)
    except Exception:
        return X_tr, y_tr


# ─── Threshold tuning (inner CV, no test leakage) ────────────────────────────
def tune_threshold_cv(model, X_tr, y_tr, sampling_method, n_folds=3, seed=SEED):
    """
    Estimate optimal decision threshold via inner StratifiedKFold (3-fold).
    Uses OOF proba on original (non-augmented) train to keep threshold generalizable.
    Returns best threshold (float in [0.1, 0.9]).
    """
    if not hasattr(model, 'predict_proba'):
        return 0.5
    n_pos = (y_tr == 1).sum()
    if n_pos < 3:
        return 0.5
    n_folds = min(n_folds, n_pos)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y_tr))
    for tr_i, va_i in skf.split(X_tr, y_tr):
        X_sub, y_sub = X_tr[tr_i], y_tr[tr_i]
        X_val = X_tr[va_i]
        # Augment sub-train for inner fit; evaluate on ORIGINAL val (no augmentation)
        X_aug, y_aug = apply_sampling(X_sub, y_sub, sampling_method, seed)
        m = clone(model)
        m.fit(X_aug, y_aug)
        oof_proba[va_i] = m.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        score = balanced_accuracy_score(y_tr, (oof_proba >= thr).astype(int))
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


# ─── Preprocessing per fold (from v1 best) ───────────────────────────────────
def preprocess_fold(X_tr, y_tr, X_te, iso_contam=0.10):
    if iso_contam > 0 and len(X_tr) > 30:
        iso = IsolationForest(contamination=iso_contam, random_state=SEED, n_jobs=-1)
        iso.fit(X_tr)
        keep = iso.predict(X_tr) == 1
        if (y_tr[keep] == 0).any() and (y_tr[keep] == 1).any():
            X_tr = X_tr[keep]; y_tr = y_tr[keep]
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    return (np.nan_to_num(X_tr), y_tr, np.nan_to_num(X_te))


# ─── Precompute LOSO folds with MI ──────────────────────────────────────────
def precompute_folds(X, y, groups, iso_contam=0.10):
    logo = LeaveOneGroupOut()
    folds = []
    for tr, te in logo.split(X, y, groups):
        X_tr, y_tr_orig = X[tr], y[tr]
        X_te, y_te = X[te], y[te]
        if len(np.unique(y_tr_orig)) < 2 or len(y_te) == 0:
            continue
        X_tr_pp, y_tr_pp, X_te_pp = preprocess_fold(
            X_tr.copy(), y_tr_orig.copy(), X_te.copy(), iso_contam)
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


# ─── Single (sampling, K, model, threshold_tuned) evaluation ─────────────────
def eval_combo(folds, K, model_factory, sampling, tune_thr):
    y_true_all, y_pred_all, thresholds_used = [], [], []
    for f in folds:
        idx = f['mi_order'][:K]
        X_tr_sel = f['X_tr'][:, idx]
        X_te_sel = f['X_te'][:, idx]
        y_tr = f['y_tr']
        y_te = f['y_te']

        model = model_factory()
        supports_proba = hasattr(model, 'predict_proba')

        # Threshold tuning uses original (non-augmented) distribution
        thr = 0.5
        if tune_thr and supports_proba:
            thr = tune_threshold_cv(model, X_tr_sel, y_tr, sampling)
        thresholds_used.append(thr)

        # Augment for training (BalancedRF/EasyEnsemble handle imbalance internally)
        model_name = type(model).__name__
        needs_sampling = model_name not in ('BalancedRandomForestClassifier',)
        if needs_sampling and sampling != 'none':
            X_fit, y_fit = apply_sampling(X_tr_sel, y_tr, sampling)
        else:
            X_fit, y_fit = X_tr_sel, y_tr

        model.fit(X_fit, y_fit)

        if supports_proba and thr != 0.5:
            proba = model.predict_proba(X_te_sel)[:, 1]
            y_pred = (proba >= thr).astype(int)
        else:
            y_pred = model.predict(X_te_sel)

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    if not y_true_all:
        return None
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    return {
        'accuracy':         accuracy_score(y_true, y_pred),
        'balanced_acc':     balanced_accuracy_score(y_true, y_pred),
        'f1_macro':         f1_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_vua_phai':  recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_others':    recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'avg_threshold':    np.mean(thresholds_used),
        'y_true':           y_true,
        'y_pred':           y_pred,
    }


# ─── Stacking (RF + SVM + LogReg → meta-LogReg) ──────────────────────────────
def eval_stacking(folds, K, sampling):
    """
    Level-0: RF, SVM, LogReg (each with sampling).
    Level-1: LogReg meta trained on OOF proba from level-0 (inner LOSO on train).
    No test data involved in meta-training.
    """
    base_factories = {
        'GBT':    lambda: GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                                      max_depth=3, random_state=SEED),
        'SVM':    lambda: SVC(kernel='rbf', C=1.0, gamma='scale',
                               class_weight='balanced', probability=True, random_state=SEED),
        'LogReg': lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced',
                                              solver='lbfgs', random_state=SEED),
    }
    meta = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced',
                               solver='lbfgs', random_state=SEED)

    y_true_all, y_pred_all = [], []
    for f in folds:
        idx = f['mi_order'][:K]
        X_tr_sel = f['X_tr'][:, idx]
        X_te_sel = f['X_te'][:, idx]
        y_tr = f['y_tr']
        y_te = f['y_te']
        n_pos = (y_tr == 1).sum()
        if n_pos < 3:
            continue

        # ── inner LOSO within train to get OOF proba for meta-learner ──────
        inner_logo = LeaveOneGroupOut()
        inner_groups = np.arange(len(y_tr))  # each sample its own "subject" (pseudo-LOSO)
        # Use StratifiedKFold instead for speed with small N
        n_inner = min(5, n_pos)
        skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)

        oof_proba = np.zeros((len(y_tr), len(base_factories)))
        for fi, (tr_i, va_i) in enumerate(skf.split(X_tr_sel, y_tr)):
            X_sub, y_sub = X_tr_sel[tr_i], y_tr[tr_i]
            X_val = X_tr_sel[va_i]
            X_aug, y_aug = apply_sampling(X_sub, y_sub, sampling)
            for bi, (bname, bfac) in enumerate(base_factories.items()):
                bm = bfac()
                bm.fit(X_aug, y_aug)
                oof_proba[va_i, bi] = bm.predict_proba(X_val)[:, 1]

        # ── train meta on OOF proba ─────────────────────────────────────────
        meta_m = clone(meta)
        meta_m.fit(oof_proba, y_tr)

        # ── level-0 predictions on test ─────────────────────────────────────
        X_aug_full, y_aug_full = apply_sampling(X_tr_sel, y_tr, sampling)
        te_proba = np.zeros((len(y_te), len(base_factories)))
        for bi, (bname, bfac) in enumerate(base_factories.items()):
            bm = bfac()
            bm.fit(X_aug_full, y_aug_full)
            te_proba[:, bi] = bm.predict_proba(X_te_sel)[:, 1]

        y_pred = meta_m.predict(te_proba)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    if not y_true_all:
        return None
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    return {
        'accuracy':        accuracy_score(y_true, y_pred),
        'balanced_acc':    balanced_accuracy_score(y_true, y_pred),
        'f1_macro':        f1_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_vua_phai': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_others':   recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'avg_threshold':   0.5,
        'y_true':          y_true,
        'y_pred':          y_pred,
    }


# ─── Model grid ──────────────────────────────────────────────────────────────
def get_models():
    return {
        # RF variants with different class weights
        'RF_balanced':  lambda: RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight='balanced',
            random_state=SEED, n_jobs=-1),
        'RF_w2':        lambda: RandomForestClassifier(
            n_estimators=200, class_weight={0: 1, 1: 2},
            random_state=SEED, n_jobs=-1),
        'RF_w3':        lambda: RandomForestClassifier(
            n_estimators=200, class_weight={0: 1, 1: 3},
            random_state=SEED, n_jobs=-1),
        # BalancedRF: bootstraps equal class sizes in each tree (no SMOTE needed)
        'BalancedRF':   lambda: BalancedRandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1,
            replacement=True, sampling_strategy='auto'),
        # Linear + kernel
        'LogReg_C0.1':  lambda: LogisticRegression(
            max_iter=3000, C=0.1, class_weight='balanced',
            solver='lbfgs', random_state=SEED),
        'LogReg_C1':    lambda: LogisticRegression(
            max_iter=3000, C=1.0, class_weight='balanced',
            solver='lbfgs', random_state=SEED),
        'SVM_RBF':      lambda: SVC(
            kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
            probability=True, random_state=SEED),
        'GradBoost':    lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            random_state=SEED),
    }


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_balanced_acc_sweep(df_res, fig_path):
    """balanced_acc vs K, colored by model, faceted by sampling."""
    samplings = df_res['sampling'].unique()
    n_col = len(samplings)
    fig, axes = plt.subplots(1, n_col, figsize=(6 * n_col, 5.5), sharey=True)
    if n_col == 1:
        axes = [axes]
    cmap = plt.cm.tab10
    models = sorted(df_res['model'].unique())
    for ax, samp in zip(axes, samplings):
        sub = df_res[df_res['sampling'] == samp]
        pivot = sub.groupby(['K', 'model'])['balanced_acc'].max().unstack()
        for i, mname in enumerate(models):
            if mname not in pivot.columns:
                continue
            ax.plot(pivot.index, pivot[mname], marker='o', linewidth=2,
                    label=mname, color=cmap(i / max(len(models), 1)))
        ax.axhline(0.558, color='red', linestyle='--', linewidth=1.5,
                   label='v1 best bacc=0.558')
        ax.axhline(0.500, color='gray', linestyle=':', linewidth=1.0,
                   label='chance=0.500')
        ax.set_xlabel('Top-K features (MI)')
        ax.set_ylabel('Balanced Accuracy' if samp == samplings[0] else '')
        ax.set_title(f'sampling={samp}', fontweight='bold')
        ax.set_ylim(0.35, 0.85)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='lower right')
    fig.suptitle('Vua_phai vs Others — balanced_acc sweep (LOSO-CV, iso=0.10, weak filter)',
                 fontsize=12, fontweight='bold')
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
    ax.set_title(title, fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(OUT_DIR, 'logs', f'run_{ts}.log')
    latest   = os.path.join(OUT_DIR, 'run.log')
    log_fh = open(log_path, 'w'); latest_fh = open(latest, 'w')
    sys.stdout = Tee(sys.__stdout__, log_fh, latest_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh, latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_ml_vuaphai_v2')
    print('='*78)
    print('  Vua_phai vs Others — Phase 2: SMOTE + BalancedRF + ThreshTuning + Stacking')
    print('='*78)

    FILTER     = 'weak'
    ISO_CONTAM = 0.10
    K_GRID     = [10, 15, 20, 30]
    SAMPLINGS  = ['none', 'smote', 'adasyn', 'borderline']
    # Threshold tuning is expensive for tree models — applied only to fast models in Phase 2
    FAST_MODELS = {'LogReg_C0.1', 'LogReg_C1', 'SVM_RBF', 'GradBoost'}

    df_all = load_data()
    print(f'Loaded: {df_all.shape}, quality: {df_all.quality_label.value_counts().to_dict()}')

    df = df_all[df_all['quality_label'] != 'BAD'].copy()
    X, y, g, feat_cols = prepare_Xy(df)
    n_pos = int(y.sum()); n_neg = int((y == 0).sum())
    majority = max(n_pos, n_neg) / len(y)
    print(f'\nFilter [{FILTER}]  n={len(y)}  pos(Vua_phai)={n_pos}  neg={n_neg}  '
          f'majority_baseline={majority:.3f}')

    print(f'\nPrecomputing LOSO folds (IsoForest contam={ISO_CONTAM})...')
    folds = precompute_folds(X, y, g, ISO_CONTAM)
    print(f'  {len(folds)} folds prepared')

    models = get_models()
    rows   = []
    best_by_bacc = None

    # ── Phase 1: Full grid, no threshold tuning ───────────────────────────────
    p1_total = len(SAMPLINGS) * len(K_GRID) * len(models)
    print(f'\n── Phase 1: Grid sweep (no threshold tuning) — {p1_total} combos ──')
    t0 = datetime.datetime.now()

    for sampling in SAMPLINGS:
        print(f'\n  [sampling={sampling}]', flush=True)
        for K in K_GRID:
            for mname, mfac in models.items():
                res = eval_combo(folds, K, mfac, sampling, tune_thr=False)
                if res is None:
                    continue
                row = {
                    'phase': 1, 'model': mname, 'sampling': sampling, 'K': K,
                    'threshold_tuned': False,
                    'accuracy':        round(res['accuracy'],        4),
                    'balanced_acc':    round(res['balanced_acc'],    4),
                    'f1_macro':        round(res['f1_macro'],        4),
                    'recall_vua_phai': round(res['recall_vua_phai'], 4),
                    'recall_others':   round(res['recall_others'],   4),
                    'avg_threshold':   round(res['avg_threshold'],   3),
                }
                rows.append(row)
                if (best_by_bacc is None or
                        res['balanced_acc'] > best_by_bacc['balanced_acc']):
                    best_by_bacc = {**row, 'y_true': res['y_true'],
                                    'y_pred': res['y_pred']}

        # Top-5 per sampling block
        sub = pd.DataFrame([r for r in rows if r['sampling'] == sampling])
        for _, r in sub.nlargest(5, 'balanced_acc').iterrows():
            print(f'    K={int(r["K"]):<3} {r["model"]:<16} '
                  f'acc={r["accuracy"]:.3f}  bacc={r["balanced_acc"]:.3f}  '
                  f'f1={r["f1_macro"]:.3f}  rec_vua={r["recall_vua_phai"]:.3f}')

    elapsed = (datetime.datetime.now() - t0).seconds
    print(f'\n  Phase 1 done in {elapsed}s')

    # ── Phase 2: Threshold tuning for fast models (LogReg, SVM, GradBoost) ───
    # Use K and sampling combos from Phase 1 top-20 by balanced_acc
    df_p1  = pd.DataFrame(rows)
    top20  = df_p1.nlargest(20, 'balanced_acc')
    fast_combos = top20[top20['model'].isin(FAST_MODELS)][['sampling', 'K', 'model']].drop_duplicates()
    # Always include K=15 smote/none combos for all fast models
    extra = pd.DataFrame([
        {'sampling': s, 'K': k, 'model': m}
        for s in ['none', 'smote'] for k in [15, 20] for m in FAST_MODELS
    ])
    fast_combos = pd.concat([fast_combos, extra]).drop_duplicates()

    p2_total = len(fast_combos)
    print(f'\n── Phase 2: Threshold tuning (fast models only) — {p2_total} combos ──')
    t0 = datetime.datetime.now()

    for _, combo in fast_combos.iterrows():
        sampling = combo['sampling']; K = int(combo['K']); mname = combo['model']
        mfac = models[mname]
        res = eval_combo(folds, K, mfac, sampling, tune_thr=True)
        if res is None:
            continue
        row = {
            'phase': 2, 'model': mname, 'sampling': sampling, 'K': K,
            'threshold_tuned': True,
            'accuracy':        round(res['accuracy'],        4),
            'balanced_acc':    round(res['balanced_acc'],    4),
            'f1_macro':        round(res['f1_macro'],        4),
            'recall_vua_phai': round(res['recall_vua_phai'], 4),
            'recall_others':   round(res['recall_others'],   4),
            'avg_threshold':   round(res['avg_threshold'],   3),
        }
        rows.append(row)
        if res['balanced_acc'] > best_by_bacc['balanced_acc']:
            best_by_bacc = {**row, 'y_true': res['y_true'],
                            'y_pred': res['y_pred']}
        print(f'  {mname:<16} samp={sampling:<10} K={K:<3}  '
              f'bacc={row["balanced_acc"]:.3f}  acc={row["accuracy"]:.3f}  '
              f'rec_vua={row["recall_vua_phai"]:.3f}  thr={row["avg_threshold"]:.2f}')

    elapsed = (datetime.datetime.now() - t0).seconds
    print(f'\n  Phase 2 done in {elapsed}s')

    # ── Phase 3: Stacking (GBT + SVM + LogReg → meta-LogReg) ─────────────────
    # Limited to most promising samplings to keep runtime manageable
    print('\n── Phase 3: Stacking (GBT + SVM + LogReg → meta-LogReg) ──')
    t0 = datetime.datetime.now()
    for sampling in ['none', 'smote', 'adasyn']:
        for K in [15, 20]:
            res = eval_stacking(folds, K, sampling)
            if res is None:
                continue
            row = {
                'phase': 3, 'model': 'Stack_RF+SVM+LR', 'sampling': sampling, 'K': K,
                'threshold_tuned': False,
                'accuracy':        round(res['accuracy'],        4),
                'balanced_acc':    round(res['balanced_acc'],    4),
                'f1_macro':        round(res['f1_macro'],        4),
                'recall_vua_phai': round(res['recall_vua_phai'], 4),
                'recall_others':   round(res['recall_others'],   4),
                'avg_threshold':   0.5,
            }
            rows.append(row)
            if res['balanced_acc'] > best_by_bacc['balanced_acc']:
                best_by_bacc = {**row, 'y_true': res['y_true'],
                                'y_pred': res['y_pred']}
            print(f'  samp={sampling:<10} K={K:<3}  '
                  f'bacc={row["balanced_acc"]:.3f}  acc={row["accuracy"]:.3f}  '
                  f'f1={row["f1_macro"]:.3f}  rec_vua={row["recall_vua_phai"]:.3f}')

    elapsed = (datetime.datetime.now() - t0).seconds
    print(f'\n  Phase 3 done in {elapsed}s')

    # ── Save results ─────────────────────────────────────────────────────────
    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'all_results_v2.csv')
    df_res.to_csv(csv_path, index=False)
    print(f'\n✓ Saved {csv_path}  ({len(df_res)} rows)')

    # Best per sampling (by balanced_acc)
    best_per_samp = (df_res.sort_values('balanced_acc', ascending=False)
                           .groupby('sampling').head(1)
                           .sort_values('balanced_acc', ascending=False))
    bp_path = os.path.join(OUT_DIR, 'best_per_sampling.csv')
    best_per_samp.to_csv(bp_path, index=False)
    print(f'✓ Saved {bp_path}')

    # Top-10 overall by balanced_acc
    top10 = df_res.nlargest(10, 'balanced_acc')
    t10_path = os.path.join(OUT_DIR, 'top10_by_bacc.csv')
    top10.to_csv(t10_path, index=False)
    print(f'✓ Saved {t10_path}')

    # ── Plots ────────────────────────────────────────────────────────────────
    # Main sweep: no threshold tuning (cleaner view)
    sub_plot = df_res[~df_res['threshold_tuned']]
    fig_path = os.path.join(FIG_DIR, 'balanced_acc_sweep.png')
    plot_balanced_acc_sweep(sub_plot, fig_path)
    print(f'✓ Saved {fig_path}')

    # Confusion matrix for overall best
    cm_path = os.path.join(FIG_DIR, 'best_confusion_matrix.png')
    b = best_by_bacc
    title = (f'{b["model"]} | sampling={b["sampling"]} | K={b["K"]} | '
             f'thr_tuned={b["threshold_tuned"]}\n'
             f'acc={b["accuracy"]:.3f}  bacc={b["balanced_acc"]:.3f}  '
             f'f1={b["f1_macro"]:.3f}  '
             f'rec_vua={b["recall_vua_phai"]:.3f}  rec_other={b["recall_others"]:.3f}')
    plot_confusion(b['y_true'], b['y_pred'], title, cm_path)
    print(f'✓ Saved {cm_path}')

    # Comparison: balanced_acc distribution by sampling (box/violin)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df_res.groupby('sampling')['balanced_acc'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_res, x='sampling', y='balanced_acc', order=order,
                hue='sampling', palette='Set2', legend=False, ax=ax)
    ax.axhline(0.558, color='red', linestyle='--', linewidth=1.5, label='v1 best=0.558')
    ax.axhline(0.500, color='gray', linestyle=':', linewidth=1.0, label='chance=0.500')
    ax.set_title('Balanced Accuracy distribution by sampling strategy (all K × models)',
                 fontweight='bold')
    ax.set_ylabel('Balanced Accuracy'); ax.set_xlabel('Sampling')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'sampling_comparison.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved sampling_comparison.png')

    # ── Summary ──────────────────────────────────────────────────────────────
    print('\n' + '='*78)
    print('  BEST PER SAMPLING (primary metric: balanced_accuracy)')
    print('='*78)
    hdr = f'{"sampling":<14}{"model":<18}{"K":<5}{"thr_tuned":<12}{"acc":<8}{"bacc":<8}{"f1":<8}{"rec_vua":<10}{"rec_oth":<8}'
    print(hdr)
    print('-'*len(hdr))
    for _, r in best_per_samp.iterrows():
        print(f'{r["sampling"]:<14}{r["model"]:<18}{int(r["K"]):<5}'
              f'{str(r["threshold_tuned"]):<12}'
              f'{r["accuracy"]:.3f}   {r["balanced_acc"]:.3f}   {r["f1_macro"]:.3f}   '
              f'{r["recall_vua_phai"]:.3f}      {r["recall_others"]:.3f}')

    print('\n  TOP-10 by balanced_acc:')
    hdr2 = f'  {"#":<4}{"model":<18}{"sampling":<14}{"K":<5}{"thr":<8}{"bacc":<8}{"acc":<8}{"f1":<8}{"rec_vua":<10}'
    print(hdr2)
    for rank, (_, r) in enumerate(top10.iterrows(), 1):
        print(f'  {rank:<4}{r["model"]:<18}{r["sampling"]:<14}{int(r["K"]):<5}'
              f'{str(r["threshold_tuned"]):<8}'
              f'{r["balanced_acc"]:.3f}   {r["accuracy"]:.3f}   {r["f1_macro"]:.3f}   '
              f'{r["recall_vua_phai"]:.3f}')

    print(f'\n  OVERALL BEST (balanced_acc={b["balanced_acc"]:.4f}):')
    print(f'    model           = {b["model"]}')
    print(f'    sampling        = {b["sampling"]}')
    print(f'    K               = {b["K"]}')
    print(f'    threshold_tuned = {b["threshold_tuned"]}')
    print(f'    avg_threshold   = {b["avg_threshold"]:.3f}')
    print(f'    accuracy        = {b["accuracy"]:.4f}')
    print(f'    balanced_acc    = {b["balanced_acc"]:.4f}  (v1 best: 0.558)')
    print(f'    f1_macro        = {b["f1_macro"]:.4f}')
    print(f'    recall_vua_phai = {b["recall_vua_phai"]:.4f}  (true positive rate)')
    print(f'    recall_others   = {b["recall_others"]:.4f}  (true negative rate)')
    print(f'    majority_baseline acc = {majority:.4f}')

    print('\n  Improvement vs v1 best:')
    print(f'    Δ balanced_acc = {b["balanced_acc"] - 0.558:+.4f}')
    print(f'    Δ accuracy     = {b["accuracy"] - 0.773:+.4f}')

    print('='*78)
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
