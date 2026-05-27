#!/usr/bin/env python3
"""
ML JAR 3-class — LEAK-FREE feature selection + model sweep.

Khác biệt với `run_ml_top_features.py` / `run_ml_jar3.py`:
  * StandardScaler được fit CHỈ trên fold train (không xem fold test).
  * Mutual Information / ANOVA F-score được tính CHỈ trên fold train.
  * Tránh leakage: scaler + feature selection nằm BÊN TRONG vòng LOSO.

Feature pool (chọn bằng --features):
  basic        — ERP component mean amplitude per channel + log-bandpower per channel.
                 (Trích nhanh, per-TRIAL → ~570 mẫu × 144 features)
  adv          — Đầy đủ ERP + bandpower + Hjorth + time-domain + SEF/centroid +
                 band ratios + alpha asymmetry per (subject × condition)
                 (Tải từ cache output/results/erp/ml_features.csv,
                  ~168 mẫu × ~334 features)
  full_adv     — Reuse pipeline từ run_ml_jar3.py: thêm DWT, coherence, STFT, CWT,
                 TFR, PLV, correlation, 1/f slope... per (subject × condition).
                 (Tự build & cache vào output/results/ml_jar3/features_jar3_adv.csv)

Pipeline:
  1. Load / cache feature matrix.
  2. LOSO theo subject_id.
     Trong mỗi fold: scaler.fit(X_train) → MI/ANOVA on X_train → top-K → train.
  3. Sweep K × models, chọn best.

Usage:
    .venv/bin/python run_ml_jar3_leakfree.py --features adv
    .venv/bin/python run_ml_jar3_leakfree.py --features full_adv --k 10 30 80 200
    .venv/bin/python run_ml_jar3_leakfree.py --features basic --selector anova
"""

import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from scipy import signal as sp_signal
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, classification_report)

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.constants import ALL_SUBJECTS, FREQ_BANDS, CONCENTRATIONS
from pipeline.erp_analysis import apply_woody_realign

# ─── Paths ─────────────────────────────────────────────────────────────
EPOCHS_BASE = 'output/epochs'
TRIAL_INFO_CSV = 'output/epochs/all_trial_info.csv'
QUALITY_FLAGS = 'output/results/erp/erp_quality_flags.csv'
ADV_CACHE   = 'output/results/erp/ml_features.csv'              # 168 rows × 334 cols
FULL_CACHE  = 'output/results/ml_jar3/features_jar3_adv.csv'    # built by run_ml_jar3.py
OUT_DIR = 'output/results/ml_jar3_leakfree'
FIG_DIR = 'output/figures/ml_jar3_leakfree'

SEED = 42
np.random.seed(SEED)
N_JOBS = 4

JAR_ORDER = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
COMP_WINDOWS = {
    'P1':   (0.090, 0.150),
    'N1':   (0.140, 0.240),
    'P2':   (0.230, 0.350),
    'N400': (0.350, 0.550),
}


# ═══════════════════════════════════════════════════════════════════════
# Feature extraction — BASIC (per-trial)
# ═══════════════════════════════════════════════════════════════════════

def extract_basic_features(all_epochs, trial_info, channels=CHANNELS):
    """Trích ERP + bandpower per channel cho mỗi TRIAL.

    Returns
    -------
    X : np.ndarray (n_trials, n_features)
    feature_names : list of str
    meta : pd.DataFrame [subject_id, jar_group, condition]
    """
    X_cat = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
    ch_names = all_epochs[0].ch_names
    times = all_epochs[0].times
    sfreq = all_epochs[0].info['sfreq']
    ch_to_idx = {ch: i for i, ch in enumerate(ch_names)}

    n_trials = X_cat.shape[0]
    nperseg = min(128, X_cat.shape[2])
    freqs, psd = sp_signal.welch(X_cat, fs=sfreq, nperseg=nperseg, axis=2)

    feat_names = []
    for comp in COMP_WINDOWS:
        for ch in channels:
            if ch in ch_to_idx:
                feat_names.append(f'ERP_{comp}_{ch}')
    for ch in channels:
        if ch in ch_to_idx:
            for bname in FREQ_BANDS:
                feat_names.append(f'BP_{bname}_{ch}')

    X = np.zeros((n_trials, len(feat_names)), dtype=np.float64)
    col = 0
    for comp, (tmin, tmax) in COMP_WINDOWS.items():
        tm = (times >= tmin) & (times <= tmax)
        for ch in channels:
            if ch in ch_to_idx:
                ci = ch_to_idx[ch]
                X[:, col] = X_cat[:, ci, tm].mean(axis=1) * 1e6
                col += 1
    for ch in channels:
        if ch in ch_to_idx:
            ci = ch_to_idx[ch]
            for _, (bf, bt) in FREQ_BANDS.items():
                fm = (freqs >= bf) & (freqs <= bt)
                bp = psd[:, ci, fm].mean(axis=1)
                X[:, col] = np.log10(bp + 1e-20)
                col += 1

    meta = pd.DataFrame({
        'subject_id': trial_info['subject_id'].values,
        'jar_group':  trial_info['jar_group'].values,
        'condition':  trial_info['condition'].values,
    })
    return X, feat_names, meta


# ═══════════════════════════════════════════════════════════════════════
# Feature loading — ADV / FULL_ADV (per subject × condition, cached)
# ═══════════════════════════════════════════════════════════════════════

def load_cached_features(path, logger=None):
    """Load cached feature matrix per (subject × condition)."""
    if not os.path.exists(path):
        return None, None, None
    df = pd.read_csv(path)
    meta_cols = {'subject_id', 'condition', 'condition_label',
                 'jar_group', 'epoch_idx', 'sample_type',
                 'event_code', 'jar_numeric'}
    feat_cols = [c for c in df.columns
                 if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Drop rows with NaN JAR
    df = df[df['jar_group'].notna()].reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop zero-variance columns
    var = X.var(axis=0)
    keep = var > 1e-12
    X = X[:, keep]
    feat_cols = [f for f, k in zip(feat_cols, keep) if k]

    meta = df[['subject_id', 'condition', 'jar_group']].copy()
    if logger:
        logger.info(f"Loaded {path}: {X.shape[0]} rows × {X.shape[1]} features")
    return X, feat_cols, meta


def build_full_adv_features(logger):
    """Build & cache full advanced feature matrix (per subject × condition).

    Uses run_ml_jar3.py's build_feature_matrix which incorporates DWT,
    coherence, STFT, CWT, TFR, PLV, correlation, 1/f slope etc.
    """
    from run_ml_jar3 import build_feature_matrix
    print('  Building full-adv features (this may take several minutes)...')
    all_epochs = []
    for sid in ALL_SUBJECTS:
        fif = os.path.join(EPOCHS_BASE, sid, 'epochs_epo.fif')
        if os.path.exists(fif):
            try:
                all_epochs.append(mne.read_epochs(fif, preload=True, verbose=False))
            except Exception as e:
                if logger:
                    logger.warning(f"[{sid}] {e}")
    trial_info = pd.read_csv(TRIAL_INFO_CSV)
    all_epochs, trial_info = apply_woody_realign(all_epochs, trial_info, logger)
    df_feat = build_feature_matrix(all_epochs, trial_info, logger)
    ensure_dir(os.path.dirname(FULL_CACHE))
    df_feat.to_csv(FULL_CACHE, index=False)
    print(f'  ✓ Cached to {FULL_CACHE}')
    return df_feat


# ═══════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════

def make_models():
    """Tạo fresh model instances mỗi lần (tránh state leak giữa fold)."""
    return {
        'LogReg':           lambda: LogisticRegression(
            max_iter=3000, C=1.0, random_state=SEED, n_jobs=N_JOBS,
            class_weight='balanced'),
        'LR_L1':            lambda: LogisticRegression(
            max_iter=3000, C=0.5, penalty='l1', solver='saga',
            random_state=SEED, n_jobs=N_JOBS, class_weight='balanced'),
        'SVM_RBF':          lambda: SVC(kernel='rbf', gamma='scale', C=1.0,
            random_state=SEED, class_weight='balanced'),
        'SVM_Linear':       lambda: SVC(kernel='linear', C=1.0,
            random_state=SEED, class_weight='balanced'),
        'RandomForest':     lambda: RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=SEED,
            class_weight='balanced', n_jobs=N_JOBS),
        'GradBoost':        lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=SEED),
        'LDA':              lambda: LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto'),
        'KNN':              lambda: KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=N_JOBS),
        'MLP':              lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=SEED,
            early_stopping=True),
    }


# ═══════════════════════════════════════════════════════════════════════
# LOSO with leak-free feature selection
# ═══════════════════════════════════════════════════════════════════════

def loso_eval(X, y, groups, model_factory, k, selector='mi'):
    """LOSO CV với scaler + feature selection BÊN TRONG fold.

    Mọi thao tác "học" (scaler.fit, MI/ANOVA score, model.fit)
    chỉ thấy dữ liệu fold train. Test fold KHÔNG bị peek.
    """
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all, fold_accs = [], [], []

    for tr, te in logo.split(X, y, groups):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # 1. Scaler chỉ fit trên train ─────────────────────────────────
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        # 2. Feature selection chỉ trên train ─────────────────────────
        if k is not None and k < X_tr_s.shape[1]:
            if selector == 'mi':
                scores = mutual_info_classif(X_tr_s, y_tr,
                                             random_state=SEED, n_neighbors=5)
            else:
                scores, _ = f_classif(X_tr_s, y_tr)
                scores = np.nan_to_num(scores, nan=0.0)
            top_idx = np.argsort(scores)[-k:]
            X_tr_s = X_tr_s[:, top_idx]
            X_te_s = X_te_s[:, top_idx]

        # 3. Fit model mới mỗi fold ────────────────────────────────────
        mdl = model_factory()
        mdl.fit(X_tr_s, y_tr)
        y_hat = mdl.predict(X_te_s)

        y_true_all.extend(y_te)
        y_pred_all.extend(y_hat)
        fold_accs.append(accuracy_score(y_te, y_hat))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    return {
        'accuracy':          accuracy_score(y_true_all, y_pred_all),
        'balanced_accuracy': balanced_accuracy_score(y_true_all, y_pred_all),
        'f1_macro':          f1_score(y_true_all, y_pred_all,
                                       average='macro', zero_division=0),
        'f1_weighted':       f1_score(y_true_all, y_pred_all,
                                       average='weighted', zero_division=0),
        'cm':                confusion_matrix(y_true_all, y_pred_all),
        'y_true':            y_true_all,
        'y_pred':            y_pred_all,
        'fold_acc_mean':     float(np.mean(fold_accs)),
        'fold_acc_std':      float(np.std(fold_accs)),
    }


# ═══════════════════════════════════════════════════════════════════════
# Data builders for each --features mode
# ═══════════════════════════════════════════════════════════════════════

def get_basic_data(logger, no_realign=False, no_quality=False):
    """Per-trial basic features (~570 trials × 144 features)."""
    print('\n[basic] Loading epochs + extracting per-trial features...')
    all_epochs = []
    for sid in ALL_SUBJECTS:
        fif = os.path.join(EPOCHS_BASE, sid, 'epochs_epo.fif')
        if os.path.exists(fif):
            all_epochs.append(mne.read_epochs(fif, preload=True, verbose=False))
    trial_info = pd.read_csv(TRIAL_INFO_CSV)

    if not no_realign:
        all_epochs, trial_info = apply_woody_realign(all_epochs, trial_info, logger)

    # Quality filter
    if not no_quality and os.path.exists(QUALITY_FLAGS):
        qf = pd.read_csv(QUALITY_FLAGS)
        keep = set()
        for _, r in qf[qf['has_real_pattern'] == True].iterrows():
            keep.add((r['subject_id'], int(r['condition'])))
        mask = trial_info.apply(
            lambda r: (r['subject_id'], int(r['condition'])) in keep, axis=1
        ).values
        info = all_epochs[0].info
        tmin0 = all_epochs[0].tmin
        X_cat = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
        X_cat = X_cat[mask]
        trial_info = trial_info[mask].reset_index(drop=True)
        all_epochs = []
        for sid in ALL_SUBJECTS:
            sm = (trial_info['subject_id'] == sid).values
            if sm.any():
                all_epochs.append(mne.EpochsArray(X_cat[sm], info,
                                                   tmin=tmin0, verbose=False))
        print(f'  Quality-filtered: {len(trial_info)} trials')

    X, feat_names, meta = extract_basic_features(all_epochs, trial_info)
    valid = ~np.isnan(X).any(axis=1) & meta['jar_group'].notna().values
    return X[valid], feat_names, meta[valid].reset_index(drop=True)


def get_adv_data(logger):
    """Per (subject × condition) advanced features từ cache ml_features.csv."""
    print(f'\n[adv] Loading cached features from {ADV_CACHE}...')
    if not os.path.exists(ADV_CACHE):
        print('  ERROR: ml_features.csv not found. Run ERP analysis first.')
        return None, None, None
    X, feat_names, meta = load_cached_features(ADV_CACHE, logger=logger)
    print(f'  Loaded: {X.shape[0]} rows (subject×condition) × {X.shape[1]} features')
    return X, feat_names, meta


def get_full_adv_data(logger):
    """Per (subject × condition) full advanced features (incl. TFR/CWT/PLV)."""
    if not os.path.exists(FULL_CACHE):
        print(f'\n[full_adv] Cache {FULL_CACHE} not found; building...')
        build_full_adv_features(logger)
    print(f'\n[full_adv] Loading cached features from {FULL_CACHE}...')
    X, feat_names, meta = load_cached_features(FULL_CACHE, logger=logger)
    print(f'  Loaded: {X.shape[0]} rows (subject×condition) × {X.shape[1]} features')
    return X, feat_names, meta


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', choices=['basic', 'adv', 'full_adv'],
                        default='adv',
                        help='Feature pool (basic = per-trial 144 feats; '
                             'adv = cached ml_features.csv ~334 feats; '
                             'full_adv = build/use TFR+CWT+PLV+... per sub×cond)')
    parser.add_argument('--k', nargs='+', type=int,
                        default=[5, 10, 15, 20, 30, 50, 80, 120, 200],
                        help='Top-K features cần thử (lọc inside fold)')
    parser.add_argument('--selector', choices=['mi', 'anova'], default='mi',
                        help='Feature ranking method (inside fold)')
    parser.add_argument('--no-quality-filter', action='store_true')
    parser.add_argument('--no-realign', action='store_true')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset models to run (default: all)')
    args = parser.parse_args()

    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    print('=' * 72)
    print('  ML JAR 3-class — LEAK-FREE feature selection + model sweep')
    print('=' * 72)
    print(f'  Feature mode : {args.features}')
    print(f'  Selector     : {args.selector}')
    print(f'  K sweep      : {args.k}')

    # ── Load data ─────────────────────────────────────────────────────
    if args.features == 'basic':
        X, feat_names, meta = get_basic_data(
            logger, no_realign=args.no_realign,
            no_quality=args.no_quality_filter)
    elif args.features == 'adv':
        X, feat_names, meta = get_adv_data(logger)
    else:
        X, feat_names, meta = get_full_adv_data(logger)

    if X is None:
        return

    # ── Encode labels ──────────────────────────────────────────────────
    le = {l: i for i, l in enumerate(JAR_ORDER)}
    y_str = meta['jar_group'].values
    y = np.array([le[v] for v in y_str])
    groups = meta['subject_id'].values

    print(f'\n  Final dataset: X = {X.shape}, n_subjects = {len(np.unique(groups))}')
    print(f'  Class distribution:')
    for cls in JAR_ORDER:
        n = int((y_str == cls).sum())
        print(f'    {cls:12s}: {n:4d} ({n/len(y_str)*100:.1f}%)')
    print(f'  Chance        : {1/3:.3f}')
    print(f'  Majority class: '
          f'{max((y_str == c).sum() for c in JAR_ORDER) / len(y_str):.3f}')

    # ── Run sweep ──────────────────────────────────────────────────────
    print(f'\n[run] LOSO sweep — {len(args.k)} K values × models...')
    all_models = make_models()
    if args.models is not None:
        all_models = {k: v for k, v in all_models.items() if k in args.models}
    print(f'  Models: {list(all_models)}')

    rows = []
    n_max = X.shape[1]
    k_list = sorted(set([k for k in args.k if k <= n_max] + [n_max]))

    for k in k_list:
        for mname, factory in all_models.items():
            res = loso_eval(X, y, groups, factory, k=k, selector=args.selector)
            rows.append({
                'k': k,
                'model': mname,
                'accuracy': res['accuracy'],
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'f1_weighted': res['f1_weighted'],
                'fold_acc_mean': res['fold_acc_mean'],
                'fold_acc_std': res['fold_acc_std'],
            })
            print(f'  K={k:4d}  {mname:14s}  acc={res["accuracy"]:.4f}  '
                  f'bal={res["balanced_accuracy"]:.4f}  '
                  f'f1={res["f1_macro"]:.4f}  '
                  f'(fold {res["fold_acc_mean"]:.3f}±{res["fold_acc_std"]:.3f})')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, f'sweep_{args.features}_{args.selector}.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n  → Saved: {out_csv}')

    # ── Best results ───────────────────────────────────────────────────
    print(f'\n{"="*72}')
    print(f'  TOP 10 (sort by balanced_accuracy)')
    print(f'{"="*72}')
    df_sorted = df.sort_values('balanced_accuracy', ascending=False).head(10)
    print(df_sorted.to_string(index=False))

    best = df_sorted.iloc[0]
    print(f'\n  🏆 BEST: model={best["model"]}, K={int(best["k"])}, '
          f'acc={best["accuracy"]:.4f}, '
          f'bal_acc={best["balanced_accuracy"]:.4f}, '
          f'f1_macro={best["f1_macro"]:.4f}')

    # ── Rerun BEST for confusion matrix / classification report ───────
    best_factory = all_models[best['model']]
    best_res = loso_eval(X, y, groups, best_factory, k=int(best['k']),
                         selector=args.selector)
    print('\n  Classification report (BEST):')
    print(classification_report(best_res['y_true'], best_res['y_pred'],
                                 target_names=JAR_ORDER, zero_division=0))
    print('\n  Confusion matrix (rows=true, cols=pred):')
    print(pd.DataFrame(best_res['cm'], index=JAR_ORDER, columns=JAR_ORDER))

    # ── Plot: accuracy & balanced_accuracy vs K ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, metric, title in zip(axes, ['accuracy', 'balanced_accuracy'],
                                   ['Accuracy', 'Balanced accuracy']):
        for mname in all_models:
            md = df[df['model'] == mname].sort_values('k')
            ax.plot(md['k'], md[metric], '-o', label=mname, markersize=4)
        ax.axhline(1/3, color='gray', ls='--', alpha=0.5, label='chance')
        ax.set_xlabel('Top-K features (selected inside fold)')
        ax.set_ylabel(title)
        ax.set_title(f'JAR 3-class — {title} vs K  '
                     f'(feat={args.features}, {args.selector.upper()}, leak-free)')
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR,
                             f'sweep_{args.features}_{args.selector}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  Figure: {fig_path}')

    # ── Confusion matrix plot for BEST ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cm = best_res['cm']
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_n, cmap='Blues', vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)',
                    ha='center', va='center',
                    color='white' if cm_n[i,j] > 0.5 else 'black', fontsize=10)
    ax.set_xticks(range(len(JAR_ORDER))); ax.set_xticklabels(JAR_ORDER, rotation=20)
    ax.set_yticks(range(len(JAR_ORDER))); ax.set_yticklabels(JAR_ORDER)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'BEST [{args.features}]: {best["model"]} K={int(best["k"])}  '
                 f'acc={best["accuracy"]:.3f}  bal={best["balanced_accuracy"]:.3f}')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_path2 = os.path.join(FIG_DIR,
                              f'confusion_best_{args.features}_{args.selector}.png')
    fig.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figure: {fig_path2}')

    print(f'\n✅ Done. Results in {OUT_DIR}/, figures in {FIG_DIR}/')


if __name__ == '__main__':
    main()
