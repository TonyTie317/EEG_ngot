#!/usr/bin/env python3
"""
ML Pipeline — JAR 3-class Classification (Không đủ / Vừa phải / Quá nhiều)
===========================================================================
Features:
  - ERP components (P1, N1, P2, N400): mean_amp, peak_amp, peak_lat
  - Bandpower per channel per band (delta, theta, alpha, beta, gamma)
  - Hjorth parameters (activity, mobility, complexity)
  - Time-domain stats (mean, std, skew, kurtosis, ptp, rms)
  - Spectral features (SEF50, SEF90, centroid, band ratios)
  - DWT wavelet features (db4, energy + entropy per level)
  - Connectivity: coherence between ROI pairs per band
  - Alpha frontal asymmetry (F3/F4, F7/F8)
  - ERP micro-windows (10ms bins trong từng component window)

Feature selection:
  - Mutual Information (MI) — top-K
  - ANOVA F-score — top-K
  - PCA — variance threshold

Models (LOSO-CV):
  - Logistic Regression (L2)
  - SVM (RBF)
  - Random Forest
  - XGBoost (nếu có)

Usage:
    .venv/bin/python run_ml_jar3.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis, f_oneway
from scipy.signal import coherence as sp_coherence, welch as sp_welch

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix, classification_report)
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import joblib

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.constants import CONCENTRATIONS, CONCENTRATION_LABELS, ALL_SUBJECTS, FREQ_BANDS
from pipeline.erp_analysis import apply_woody_realign

# Advanced feature modules from src/
from src.features_advanced.spectral_advanced import compute_advanced_spectral_features
from src.features_advanced.stft_features import compute_stft_features
from src.features_advanced.wavelet_features import compute_dwt_features, compute_cwt_features
from src.features_advanced.timefreq_features import compute_tfr_features, compute_plv_features
from src.features_advanced.alpha_analysis import compute_alpha_features
from src.features_advanced.connectivity import (
    compute_coherence_features, compute_plv_connectivity,
    compute_correlation_features
)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS_BASE  = 'output/epochs'
OUT_DIR      = 'output/results/ml_jar3'
FIG_DIR      = 'output/figures/ml_jar3'
SEED         = 42

COMP_WINDOWS = {
    'P1':   (0.090, 0.150),
    'N1':   (0.140, 0.240),
    'P2':   (0.230, 0.350),
    'N400': (0.350, 0.550),
}
COMP_ROI = {
    'P1':   ['F3', 'F4', 'C3', 'C4'],
    'N1':   ['F7', 'F8', 'T7', 'T8'],
    'P2':   ['C3', 'C4', 'P3', 'P4'],
    'N400': ['C3', 'C4', 'P3', 'P4', 'F3', 'F4'],
}
COMP_MODE = {'P1': 'pos', 'N1': 'neg', 'P2': 'pos', 'N400': 'neg'}

# ROI pairs for connectivity
ROI_PAIRS = [
    ('F3', 'F4'), ('F7', 'F8'), ('C3', 'C4'), ('P3', 'P4'),
    ('F3', 'C3'), ('F4', 'C4'), ('C3', 'P3'), ('C4', 'P4'),
    ('F7', 'T7'), ('F8', 'T8'), ('T7', 'P7'), ('T8', 'P8'),
    ('Fp1', 'F3'), ('Fp2', 'F4'),
]

JAR_LABELS = {
    'Khong_du':  'Not enough\n(JAR 1-2)',
    'Vua_phai':  'Just right\n(JAR 3)',
    'Qua_nhieu': 'Too much\n(JAR 4-5)',
}
JAR_ORDER = ['Khong_du', 'Vua_phai', 'Qua_nhieu']

np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction — per epoch (averaged per subject × condition)
# ──────────────────────────────────────────────────────────────────────────────

def _pick(ch_names_all, roi):
    return [ch for ch in roi if ch in ch_names_all]


def extract_erp_features(avg, times, ch_names, sfreq):
    """ERP component features from condition-averaged signal."""
    feats = {}
    for comp, (tw0, tw1) in COMP_WINDOWS.items():
        roi = _pick(ch_names, COMP_ROI[comp])
        if not roi:
            roi = ch_names[:4]
        roi_idx = [ch_names.index(ch) for ch in roi]
        tm = (times >= tw0) & (times <= tw1)
        win = avg[roi_idx][:, tm]          # (n_roi, n_t_win)
        roi_avg = win.mean(axis=0)         # (n_t_win,)

        mean_amp = win.mean()
        if COMP_MODE[comp] == 'pos':
            pk_idx = np.argmax(roi_avg)
        else:
            pk_idx = np.argmin(roi_avg)
        peak_amp = roi_avg[pk_idx]
        peak_lat = times[tm][pk_idx]

        feats[f'erp_{comp}_mean_amp'] = float(mean_amp * 1e6)
        feats[f'erp_{comp}_peak_amp'] = float(peak_amp * 1e6)
        feats[f'erp_{comp}_peak_lat'] = float(peak_lat * 1000)

        # Area under curve (signed)
        feats[f'erp_{comp}_auc'] = float(np.trapz(roi_avg) * 1e6)

        # RMS in window
        feats[f'erp_{comp}_rms'] = float(np.sqrt(np.mean(roi_avg**2)) * 1e6)

        # Micro-windows: 50ms bins
        bin_size = max(1, int(0.05 * sfreq))
        win_times = times[tm]
        for bi in range(0, len(win_times), bin_size):
            sl = slice(bi, bi + bin_size)
            t_label = f'{win_times[bi]*1000:.0f}ms'
            feats[f'erp_{comp}_bin{t_label}'] = float(roi_avg[sl].mean() * 1e6)

    # Difference features
    p2_roi_idx = [ch_names.index(ch) for ch in _pick(ch_names, COMP_ROI['P2']) or ch_names[:2]]
    n400_roi_idx = [ch_names.index(ch) for ch in _pick(ch_names, COMP_ROI['N400']) or ch_names[:2]]
    tm_p2   = (times >= 0.23) & (times <= 0.35)
    tm_n400 = (times >= 0.35) & (times <= 0.55)
    feats['erp_P2_N400_ratio'] = (
        float(avg[p2_roi_idx][:, tm_p2].mean() / (abs(avg[n400_roi_idx][:, tm_n400].mean()) + 1e-12))
    )
    return feats


def extract_bandpower_features(avg, sfreq, ch_names):
    """Per-channel per-band power + ratios."""
    feats = {}
    n_fft = min(128, avg.shape[1])

    band_power = {}
    for ci, ch in enumerate(ch_names):
        x = avg[ci]
        fw, ps = sp_welch(x, fs=sfreq, nperseg=n_fft, nfft=n_fft)
        valid = (fw > 0) & (fw <= sfreq / 2)
        fw, ps = fw[valid], ps[valid]
        bp = {}
        for bname, (bf, bt) in FREQ_BANDS.items():
            bm = (fw >= bf) & (fw <= bt)
            bp[bname] = float(ps[bm].mean()) if bm.sum() > 0 else 0.0
            feats[f'bp_{bname}_{ch}'] = bp[bname]
        band_power[(ci, ch)] = bp

        # Relative power
        total = sum(bp.values()) + 1e-20
        for bname in FREQ_BANDS:
            feats[f'bp_rel_{bname}_{ch}'] = bp[bname] / total

        # Band ratios
        for num, den in [('theta','alpha'),('alpha','beta'),
                          ('theta','beta'),('delta','alpha'),('gamma','beta')]:
            d = bp.get(den, 0) + 1e-20
            feats[f'ratio_{num}_{den}_{ch}'] = bp.get(num, 0) / d

        # Spectral edge freq (SEF50, SEF90) + centroid
        cum_ps = np.cumsum(ps)
        for pct, pname in [(0.50,'sef50'),(0.90,'sef90')]:
            idx = int(np.searchsorted(cum_ps, pct * cum_ps[-1]))
            feats[f'{pname}_{ch}'] = float(fw[min(idx, len(fw)-1)])
        feats[f'spec_cent_{ch}'] = float(np.sum(fw * ps) / (np.sum(ps) + 1e-20))

    # Alpha frontal asymmetry
    for left, right in [('F3','F4'),('F7','F8')]:
        if left in ch_names and right in ch_names:
            bp_l = band_power[(ch_names.index(left), left)].get('alpha', 1e-10)
            bp_r = band_power[(ch_names.index(right), right)].get('alpha', 1e-10)
            feats[f'alpha_asym_{left}_{right}'] = np.log(bp_r + 1e-10) - np.log(bp_l + 1e-10)

    return feats


def extract_hjorth_timedomain(avg, ch_names):
    """Hjorth parameters + extended time-domain stats per channel."""
    feats = {}
    for ci, ch in enumerate(ch_names):
        x = avg[ci]
        dx = np.diff(x)
        ddx = np.diff(dx)
        act = float(np.var(x))
        mob = float(np.sqrt(np.var(dx) / act)) if act > 1e-12 else 0.0
        cmp = float(np.sqrt(np.var(ddx) / np.var(dx)) / mob) \
              if (np.var(dx) > 1e-12 and mob > 1e-12) else 0.0
        feats[f'hjorth_act_{ch}']  = act
        feats[f'hjorth_mob_{ch}']  = mob
        feats[f'hjorth_cmp_{ch}']  = cmp

        feats[f'td_mean_{ch}']     = float(np.mean(x))
        feats[f'td_std_{ch}']      = float(np.std(x))
        feats[f'td_skew_{ch}']     = float(skew(x))
        feats[f'td_kurt_{ch}']     = float(kurtosis(x))
        feats[f'td_ptp_{ch}']      = float(np.ptp(x))
        feats[f'td_rms_{ch}']      = float(np.sqrt(np.mean(x**2)))
        feats[f'td_energy_{ch}']   = float(np.sum(x**2))
        feats[f'td_zcr_{ch}']      = float(np.sum(np.diff(np.sign(x)) != 0) / len(x))
        # Line length
        feats[f'td_linelen_{ch}']  = float(np.sum(np.abs(dx)))
    return feats


def extract_wavelet_features(avg, ch_names, wavelet='db4', max_level=5):
    """DWT per channel: energy + entropy per level."""
    feats = {}
    try:
        import pywt
    except ImportError:
        return feats
    for ci, ch in enumerate(ch_names):
        x = avg[ci]
        try:
            coeffs = pywt.wavedec(x, wavelet, level=max_level)
        except Exception:
            continue
        for li, c in enumerate(coeffs):
            level_name = f'cA{max_level}' if li == 0 else f'cD{max_level - li + 1}'
            energy = float(np.sum(c**2))
            feats[f'dwt_{level_name}_energy_{ch}'] = energy
            # Shannon entropy
            p = c**2
            pt = p.sum() + 1e-20
            p = p / pt
            p = p[p > 0]
            feats[f'dwt_{level_name}_entropy_{ch}'] = float(-np.sum(p * np.log2(p + 1e-20)))
    return feats


def extract_connectivity_features(avg, times, ch_names, sfreq):
    """Coherence between ROI pairs per frequency band (post-stimulus window)."""
    feats = {}
    # Use 0–600ms post-stimulus
    tm = (times >= 0.0) & (times <= 0.6)
    seg = avg[:, tm]
    nperseg = min(32, seg.shape[1])
    if nperseg < 4:
        return feats

    ch_idx = {ch: i for i, ch in enumerate(ch_names)}

    for bname, (bf, bt) in FREQ_BANDS.items():
        for ca, cb in ROI_PAIRS:
            if ca not in ch_idx or cb not in ch_idx:
                continue
            try:
                fw, coh = sp_coherence(
                    seg[ch_idx[ca]], seg[ch_idx[cb]],
                    fs=sfreq, nperseg=nperseg
                )
                bm = (fw >= bf) & (fw <= bt)
                if bm.sum() > 0:
                    feats[f'coh_{bname}_{ca}_{cb}'] = float(coh[bm].mean())
            except Exception:
                pass
    return feats


def _long_df_to_flat(df: pd.DataFrame) -> dict:
    """Convert long-format feature DataFrame (1 epoch) to flat dict."""
    if df.empty:
        return {}
    return dict(zip(df['feature_name'], df['value']))


def _make_single_epoch(avg, info, tmin):
    """Wrap (n_ch, n_t) array into a single-epoch mne.EpochsArray."""
    return mne.EpochsArray(
        avg[np.newaxis, :, :],   # (1, n_ch, n_t)
        info=info,
        tmin=tmin,
        verbose=False,
    )


def extract_advanced_features(avg, info, tmin):
    """Run all src/features_advanced extractors on a single averaged epoch."""
    feats = {}
    ep = _make_single_epoch(avg, info, tmin)

    # ── Advanced spectral (SEF50/90/95, centroid, band ratios, 1/f slope) ──
    try:
        df = compute_advanced_spectral_features(
            ep, FREQ_BANDS,
            sef_percentiles=[50, 90, 95],
            band_ratios=[['theta','alpha'],['alpha','beta'],
                         ['theta','beta'],['delta','alpha'],['gamma','beta']],
            compute_aperiodic=True,
        )
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── STFT features (mean/std/max power per band, entropy, variability) ──
    try:
        n_t = avg.shape[1]
        win = min(32, n_t)
        df = compute_stft_features(ep, FREQ_BANDS,
                                   window_size=win, hop_length=win//2,
                                   n_fft=min(64, n_t))
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── DWT wavelet (db4: energy + entropy per level) ──
    try:
        df = compute_dwt_features(ep, wavelet='db4', max_level=5)
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── CWT (complex Morlet: mean/max power per band + wavelet entropy) ──
    try:
        df = compute_cwt_features(ep, FREQ_BANDS)
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── TFR Morlet (mean/peak power, rise/fall rate per band) ──
    try:
        df = compute_tfr_features(ep, FREQ_BANDS, n_cycles=5)
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── PLV (Phase-Locking Value) per band per ROI pair ──
    try:
        df = compute_plv_features(ep, FREQ_BANDS, pair_strategy='roi')
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── Alpha analysis (peak freq, relative power, asymmetry, coherence) ──
    try:
        df = compute_alpha_features(ep, compute_coherence=True)
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── Coherence per band per ROI pair ──
    try:
        df = compute_coherence_features(ep, FREQ_BANDS, pair_strategy='roi')
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── PLV connectivity per band per ROI pair ──
    try:
        df = compute_plv_connectivity(ep, FREQ_BANDS, pair_strategy='roi')
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    # ── Correlation matrix stats ──
    try:
        df = compute_correlation_features(ep)
        feats.update(_long_df_to_flat(df))
    except Exception:
        pass

    return feats


def extract_all_features_for_epoch(avg, times, ch_names, sfreq, info=None, tmin=-0.2):
    """Tổng hợp tất cả features từ một epoch đã averaged."""
    feats = {}
    feats.update(extract_erp_features(avg, times, ch_names, sfreq))
    feats.update(extract_bandpower_features(avg, sfreq, ch_names))
    feats.update(extract_hjorth_timedomain(avg, ch_names))
    feats.update(extract_wavelet_features(avg, ch_names))
    feats.update(extract_connectivity_features(avg, times, ch_names, sfreq))
    # Advanced features (requires mne.Info)
    if info is not None:
        feats.update(extract_advanced_features(avg, info, tmin))
    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Build feature matrix from epochs
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(all_epochs, all_trial_info, logger):
    """
    Mỗi row = 1 trial (subject × condition).
    Average 5 repeats → extract features → label = jar_group.
    """
    rows = []
    offset = 0
    for ep_idx, epochs in enumerate(all_epochs):
        n_ep = len(epochs)
        ti = all_trial_info.iloc[offset:offset + n_ep].reset_index(drop=True)
        offset += n_ep

        subj = ti['subject_id'].iloc[0]
        data = epochs.get_data()        # (n_ep, n_ch, n_t)
        ch_names = [ch for ch in epochs.ch_names if not ch.upper().startswith('ECG')]
        ch_idx = [epochs.ch_names.index(ch) for ch in ch_names]
        times = epochs.times
        sfreq = epochs.info['sfreq']

        info_ep  = epochs.info
        tmin_ep  = epochs.tmin

        for cond in CONCENTRATIONS:
            mask = (ti['condition'] == cond).values
            if mask.sum() == 0:
                continue

            # Average across repeats
            avg = data[mask][:, ch_idx, :].mean(axis=0)  # (n_ch, n_t)

            # JAR group
            jar = ti.loc[ti['condition'] == cond, 'jar_group'].values
            jar = jar[pd.notna(jar)]
            if len(jar) == 0:
                continue
            jar_group = jar[0]

            feats = extract_all_features_for_epoch(
                avg, times, ch_names, sfreq,
                info=info_ep, tmin=tmin_ep
            )
            feats['subject_id'] = subj
            feats['condition']  = cond
            feats['jar_group']  = jar_group
            rows.append(feats)

        if (ep_idx + 1) % 5 == 0:
            print(f'  [{ep_idx+1}/{len(all_epochs)}] subjects processed...')

    df = pd.DataFrame(rows)
    logger.info(f'Feature matrix: {df.shape[0]} rows × {df.shape[1]} cols')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Feature selection
# ──────────────────────────────────────────────────────────────────────────────

def select_features(X, y, feature_names, method='mi', n_top=50):
    """
    method: 'mi' | 'anova' | 'pca'
    Returns (X_sel, selected_names, scores_df)
    """
    if method == 'pca':
        pca = PCA(n_components=min(n_top, X.shape[1], X.shape[0]-1),
                  random_state=SEED)
        X_sel = pca.fit_transform(X)
        names = [f'PC{i+1}' for i in range(X_sel.shape[1])]
        scores_df = pd.DataFrame({
            'feature': names,
            'score': pca.explained_variance_ratio_,
        })
        return X_sel, names, scores_df, pca

    if method == 'mi':
        scores = mutual_info_classif(X, y, random_state=SEED)
        method_label = 'MI'
    else:  # anova
        F_vals, _ = f_classif(X, y)
        scores = np.nan_to_num(F_vals, nan=0.0, posinf=0.0)
        method_label = 'F'

    scores_df = pd.DataFrame({'feature': feature_names, 'score': scores})
    scores_df = scores_df.sort_values('score', ascending=False).reset_index(drop=True)

    top_names = scores_df.head(n_top)['feature'].tolist()
    top_idx   = [feature_names.index(n) for n in top_names if n in feature_names]
    X_sel     = X[:, top_idx]

    return X_sel, top_names, scores_df, None


# ──────────────────────────────────────────────────────────────────────────────
# LOSO training
# ──────────────────────────────────────────────────────────────────────────────

def get_models():
    models = {
        'LogisticReg': LogisticRegression(max_iter=2000, C=1.0,
                                          class_weight='balanced',
                                          random_state=SEED, solver='saga',
                                          multi_class='multinomial'),
        'SVM_RBF':     SVC(kernel='rbf', C=10, gamma='scale',
                           class_weight='balanced', probability=True,
                           random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None,
                                               class_weight='balanced',
                                               random_state=SEED, n_jobs=-1),
        'GradBoost':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                   max_depth=3, random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models['XGBoost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=-1
        )
    except ImportError:
        pass
    return models


def loso_cv(X, y, groups, model, label_names):
    """Leave-One-Subject-Out cross-validation."""
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    fold_accs = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        fold_accs.append(accuracy_score(y_te, y_pred))

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    return {
        'accuracy':      accuracy_score(y_true, y_pred),
        'balanced_acc':  balanced_accuracy_score(y_true, y_pred),
        'f1_macro':      f1_score(y_true, y_pred, average='macro'),
        'f1_weighted':   f1_score(y_true, y_pred, average='weighted'),
        'cm':            confusion_matrix(y_true, y_pred),
        'fold_accs':     fold_accs,
        'y_true':        y_true,
        'y_pred':        y_pred,
        'report':        classification_report(y_true, y_pred,
                                               target_names=label_names,
                                               zero_division=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(scores_df, method, n_top, fig_dir):
    top = scores_df.head(n_top)
    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.28)))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    ax.barh(range(len(top)), top['score'].values[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'].values[::-1], fontsize=7)
    ax.set_xlabel('Score', fontsize=11)
    ax.set_title(f'Top-{n_top} Features — {method.upper()}', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, f'feat_importance_{method}_top{n_top}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_confusion_matrix_fig(cm, label_names, title, fig_dir, fname):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f'{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)'
                       for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(fig_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_model_comparison(summary_df, fig_dir):
    """Bar chart comparing all models × feature-selection methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['balanced_acc', 'f1_macro', 'accuracy']
    titles  = ['Balanced Accuracy', 'F1-macro', 'Accuracy']
    chance  = 1/3

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = summary_df.pivot_table(
            index='model', columns='feat_method', values=metric
        )
        pivot.plot(kind='bar', ax=ax, rot=30, width=0.7)
        ax.axhline(chance, color='red', linestyle='--', linewidth=1.5,
                   label=f'Chance ({chance:.2f})')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('JAR 3-class Classification — Model × Feature Selection\n'
                 'LOSO-CV (Leave-One-Subject-Out)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(fig_dir, 'model_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


def plot_per_fold_bars(fold_accs_dict, fig_dir, n_subjects):
    """Per-subject accuracy for each model."""
    n_models = len(fold_accs_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    chance = 1/3
    for ax, (name, fold_accs) in zip(axes, fold_accs_dict.items()):
        colors = ['#2ecc71' if a > chance else '#e74c3c' for a in fold_accs]
        ax.bar(range(1, len(fold_accs)+1), fold_accs, color=colors, edgecolor='white')
        ax.axhline(chance, color='gray', linestyle='--', label=f'Chance ({chance:.2f})')
        ax.axhline(np.mean(fold_accs), color='navy', linewidth=2,
                   label=f'Mean ({np.mean(fold_accs):.3f})')
        ax.set_xlabel('Subject (fold)', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
    fig.suptitle('Per-Subject LOSO Accuracy — JAR 3-class', fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(fig_dir, 'per_fold_accuracy.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


def plot_feature_type_breakdown(scores_df, fig_dir, method):
    """Pie chart: which feature types dominate top-50."""
    top50 = scores_df.head(50)['feature'].tolist()
    type_map = {
        'erp_':         'ERP components',
        'bp_':          'Bandpower',
        'ratio_':       'Band ratios',
        'sef':          'Spectral edge',
        'spec_':        'Spectral centroid',
        'one_over_f':   '1/f slope',
        'alpha_':       'Alpha analysis',
        'hjorth_':      'Hjorth',
        'td_':          'Time-domain',
        'dwt_':         'DWT wavelet',
        'cwt_':         'CWT wavelet',
        'stf_':         'STFT',
        'tfr_':         'TFR/Morlet',
        'coh_':         'Coherence',
        'plv_':         'PLV',
        'corr_':        'Correlation',
    }
    counts = {v: 0 for v in type_map.values()}
    counts['Other'] = 0
    for f in top50:
        matched = False
        for prefix, label in type_map.items():
            if f.startswith(prefix):
                counts[label] += 1
                matched = True
                break
        if not matched:
            counts['Other'] += 1
    counts = {k: v for k, v in counts.items() if v > 0}
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts.values(), labels=counts.keys(), autopct='%1.0f%%',
        startangle=140, pctdistance=0.82,
        colors=plt.cm.tab10(np.linspace(0, 1, len(counts)))
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(f'Feature Type Distribution — Top-50 ({method.upper()})',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(fig_dir, f'feature_type_breakdown_{method}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


def plot_pca_variance(pca, fig_dir):
    """PCA explained variance curve."""
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(cumvar)+1), cumvar * 100, 'o-', color='steelblue', linewidth=2)
    ax.axhline(80, color='gray', linestyle='--', label='80% variance')
    ax.axhline(90, color='orange', linestyle='--', label='90% variance')
    n80 = int(np.searchsorted(cumvar, 0.80)) + 1
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    ax.axvline(n80, color='gray', linestyle=':', alpha=0.7, label=f'n={n80} → 80%')
    ax.axvline(n90, color='orange', linestyle=':', alpha=0.7, label=f'n={n90} → 90%')
    ax.set_xlabel('Number of PCs', fontsize=11)
    ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax.set_title('PCA — Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, 'pca_variance.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


def plot_feature_sweep(sweep_df, fig_dir, method):
    """Balanced accuracy vs n_top_features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = sweep_df['model'].unique()
    cmap = plt.cm.tab10
    for ax, metric, ylabel in [
        (axes[0], 'balanced_acc', 'Balanced Accuracy'),
        (axes[1], 'f1_macro', 'F1-macro'),
    ]:
        for mi, mname in enumerate(models):
            sub = sweep_df[sweep_df['model'] == mname].sort_values('n_top')
            ax.plot(sub['n_top'], sub[metric], marker='o', linewidth=2,
                    label=mname, color=cmap(mi / len(models)))
        ax.axhline(1/3, color='red', linestyle='--', linewidth=1.5,
                   label='Chance (0.333)')
        ax.set_xlabel(f'Top-N Features ({method.upper()})', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
    fig.suptitle(f'Feature Count Sweep — JAR 3-class | {method.upper()} selection',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(fig_dir, f'feature_sweep_{method}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)
    ensure_dir(OUT_DIR)
    ensure_dir(FIG_DIR)

    print('\n' + '='*70)
    print('  ML JAR 3-CLASS — EEG Feature Engineering + Selection')
    print('='*70)

    # ── 1. Load epochs ──────────────────────────────────────────────────────
    print('\n[1/5] Loading epochs...')
    all_epochs = []
    for sid in ALL_SUBJECTS:
        fif = os.path.join(EPOCHS_BASE, sid, 'epochs_epo.fif')
        if not os.path.exists(fif):
            continue
        try:
            ep = mne.read_epochs(fif, preload=True, verbose=False)
            all_epochs.append(ep)
        except Exception as e:
            logger.warning(f'[{sid}] {e}')

    ti_path = os.path.join(EPOCHS_BASE, 'all_trial_info.csv')
    all_trial_info = pd.read_csv(ti_path)

    # Apply Woody realignment
    all_epochs, all_trial_info = apply_woody_realign(all_epochs, all_trial_info, logger)
    print(f'  Loaded: {len(all_epochs)} subjects, {len(all_trial_info)} trials after realign')

    # ── 2. Build feature matrix ─────────────────────────────────────────────
    print('\n[2/5] Extracting features (ERP + bandpower + Hjorth + wavelet + connectivity)...')
    feat_csv = os.path.join(OUT_DIR, 'features_jar3.csv')
    feat_csv_adv = os.path.join(OUT_DIR, 'features_jar3_adv.csv')

    if os.path.exists(feat_csv_adv):
        print(f'  (loading cached advanced features from {feat_csv_adv})')
        df_feat = pd.read_csv(feat_csv_adv)
    else:
        df_feat = build_feature_matrix(all_epochs, all_trial_info, logger)
        df_feat.to_csv(feat_csv_adv, index=False)
        print(f'  ✓ Saved: {feat_csv_adv}')

    print(f'  Feature matrix: {df_feat.shape[0]} samples × {df_feat.shape[1]} columns')

    # ── 3. Prepare X, y ─────────────────────────────────────────────────────
    meta_cols = ['subject_id', 'condition', 'jar_group']
    feat_cols = [c for c in df_feat.columns if c not in meta_cols]

    # Drop NaN/Inf columns
    X_raw = df_feat[feat_cols].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Drop zero-variance columns
    variances = X_raw.var(axis=0)
    keep_mask = variances > 1e-12
    feat_cols_clean = [f for f, k in zip(feat_cols, keep_mask) if k]
    X_raw = X_raw[:, keep_mask]
    print(f'  Features after removing zero-variance: {X_raw.shape[1]}')

    # Encode JAR labels
    le = LabelEncoder()
    le.fit(JAR_ORDER)
    df_feat_valid = df_feat.dropna(subset=['jar_group'])
    valid_idx = df_feat_valid.index.tolist()

    X_all   = X_raw[valid_idx]
    y_all   = le.transform(df_feat_valid['jar_group'].values)
    grp_all = df_feat_valid['subject_id'].values
    label_names = [JAR_LABELS[j].split('\n')[0] for j in JAR_ORDER]

    print(f'  X: {X_all.shape}, JAR distribution: '
          + str({JAR_ORDER[i]: int((y_all==i).sum()) for i in range(3)}))

    # ── 4. Feature selection ─────────────────────────────────────────────────
    print('\n[3/5] Feature selection (MI, ANOVA, PCA)...')

    # Scale once for MI/ANOVA
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(X_all)
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

    sel_results = {}
    pca_obj = None

    for method in ['mi', 'anova', 'pca']:
        print(f'  → {method.upper()}...')
        n_top = 50 if method != 'pca' else 40
        X_sel, names, scores_df, pca_obj_tmp = select_features(
            X_scaled, y_all, feat_cols_clean, method=method, n_top=n_top
        )
        sel_results[method] = {'X': X_sel, 'names': names, 'scores': scores_df}
        scores_df.to_csv(os.path.join(OUT_DIR, f'feature_scores_{method}.csv'), index=False)

        if method != 'pca':
            plot_feature_importance(scores_df, method, n_top, FIG_DIR)
            plot_feature_type_breakdown(scores_df, FIG_DIR, method)
        else:
            pca_obj = pca_obj_tmp
            plot_pca_variance(pca_obj, FIG_DIR)
        print(f'    Selected: {X_sel.shape[1]} features/PCs')

    # ── 5. Train models ──────────────────────────────────────────────────────
    print('\n[4/5] Training models (LOSO-CV)...')
    models = get_models()
    summary_rows = []
    best_fold_accs = {}  # for per-fold plot of best method
    all_sweep_rows = []

    # Sweep: n_top = [20, 30, 50, 75, 100] for MI and ANOVA
    n_top_sweep = [15, 25, 40, 60, 80, 100]

    for method in ['mi', 'anova', 'pca']:
        print(f'\n  ── {method.upper()} features ──')

        if method == 'pca':
            X_method = sel_results['pca']['X']
        else:
            X_method = sel_results[method]['X']

        for mname, model in models.items():
            res = loso_cv(X_method, y_all, grp_all, model, label_names)
            acc   = res['accuracy']
            bacc  = res['balanced_acc']
            f1m   = res['f1_macro']
            print(f'    {mname:<15} acc={acc:.3f}  bal_acc={bacc:.3f}  f1={f1m:.3f}  '
                  f'fold_std={np.std(res["fold_accs"]):.3f}')

            summary_rows.append({
                'feat_method': method, 'model': mname,
                'accuracy': acc, 'balanced_acc': bacc,
                'f1_macro': f1m, 'f1_weighted': res['f1_weighted'],
                'fold_mean': np.mean(res['fold_accs']),
                'fold_std':  np.std(res['fold_accs']),
            })

            # Save confusion matrix plot
            cm_title = f'{mname} ({method.upper()}, n={X_method.shape[1]})\nbal_acc={bacc:.3f} f1={f1m:.3f}'
            plot_confusion_matrix_fig(
                res['cm'], label_names, cm_title, FIG_DIR,
                f'cm_{method}_{mname}.png'
            )

            # Track best fold accs for top model
            key = f'{mname}_{method}'
            best_fold_accs[key] = res['fold_accs']

        # Feature sweep (only for MI and ANOVA)
        if method in ('mi', 'anova'):
            print(f'  Sweep n_top={n_top_sweep} for {method.upper()}...')
            for n_top in n_top_sweep:
                if n_top > X_scaled.shape[1]:
                    continue
                X_sw, _, _, _ = select_features(
                    X_scaled, y_all, feat_cols_clean, method=method, n_top=n_top
                )
                for mname, model in models.items():
                    r = loso_cv(X_sw, y_all, grp_all, model, label_names)
                    all_sweep_rows.append({
                        'method': method, 'model': mname, 'n_top': n_top,
                        'balanced_acc': r['balanced_acc'], 'f1_macro': r['f1_macro'],
                    })

    # ── 6. Save & plot results ───────────────────────────────────────────────
    print('\n[5/5] Saving results & plots...')

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)

    # Best result
    best = summary_df.sort_values('balanced_acc', ascending=False).iloc[0]
    print(f'\n  🏆 BEST: {best["model"]} + {best["feat_method"].upper()} | '
          f'bal_acc={best["balanced_acc"]:.3f} | f1={best["f1_macro"]:.3f}')

    # Plots
    plot_model_comparison(summary_df, FIG_DIR)

    # Per-fold for top 4 models (best method)
    best_method = best['feat_method']
    top4 = summary_df[summary_df['feat_method']==best_method].nlargest(4,'balanced_acc')
    fold_dict = {r['model']: best_fold_accs[f'{r["model"]}_{r["feat_method"]}']
                 for _, r in top4.iterrows()
                 if f'{r["model"]}_{r["feat_method"]}' in best_fold_accs}
    if fold_dict:
        plot_per_fold_bars(fold_dict, FIG_DIR, len(np.unique(grp_all)))

    # Feature sweep plots
    if all_sweep_rows:
        sweep_df = pd.DataFrame(all_sweep_rows)
        for method in ['mi', 'anova']:
            sub = sweep_df[sweep_df['method']==method]
            if not sub.empty:
                plot_feature_sweep(sub, FIG_DIR, method)

    # ── Print report ─────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('  SUMMARY — JAR 3-class LOSO-CV Results')
    print('='*70)
    print(f'\n  Chance level: {1/3:.3f} (33.3%)\n')
    for method in ['mi', 'anova', 'pca']:
        sub = summary_df[summary_df['feat_method']==method]
        print(f'  [{method.upper()}]')
        for _, r in sub.sort_values('balanced_acc', ascending=False).iterrows():
            marker = '✅' if r['balanced_acc'] > 1/3 + 0.05 else '⚠️ '
            print(f'    {marker} {r["model"]:<15} bal_acc={r["balanced_acc"]:.3f}  '
                  f'f1={r["f1_macro"]:.3f}  ±{r["fold_std"]:.3f}')
        print()

    print(f'  Results: {OUT_DIR}')
    print(f'  Figures: {FIG_DIR}')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
