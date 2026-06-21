"""
Machine-learning classification with Leave-One-Subject-Out (LOSO) CV.

Features = per-trial band power (5 bands × 6 ROIs, absolute + relative) plus
per-channel relative band power. Tasks: sucrose-vs-sucralose (binary) and
sweetness intensity (3-class). Models: LogReg, SVM, RandomForest, XGBoost.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import BAND_ORDER, EEG_CHANNELS, FREQ_BANDS, ROIS, SFREQ
from .spectral import band_power, compute_psd, total_power

try:
    import xgboost  # noqa: F401
    XGB_AVAILABLE = True
except Exception:                                       # noqa: BLE001
    XGB_AVAILABLE = False


def build_trial_features(all_epochs: list, cfg: Dict[str, Any],
                         logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """One feature row per trial: ROI band power + per-channel relative power.

    Returns a DataFrame with meta columns (subject, ma_mau, substance, intensity,
    repeat) and many ``feat_*`` columns.
    """
    rows = []
    roi_idx = {r: [EEG_CHANNELS.index(c) for c in chs if c in EEG_CHANNELS]
               for r, chs in ROIS.items()}
    for ep in all_epochs:
        data = ep.get_data(copy=False)
        meta = ep.metadata.reset_index(drop=True)
        freqs, psd = compute_psd(data, SFREQ, cfg)        # (n_ep, n_ch, n_f)
        tot = total_power(freqs, psd, cfg['spectral'].get('fmin', 1.0),
                          cfg['spectral'].get('fmax', 45.0))   # (n_ep, n_ch)
        bp = {b: band_power(freqs, psd, FREQ_BANDS[b]) for b in BAND_ORDER}  # (n_ep,n_ch)
        for i in range(data.shape[0]):
            m = meta.iloc[i]
            feat = {'subject': m['subject'], 'ma_mau': m['ma_mau'],
                    'substance': m['substance'], 'intensity': int(m['intensity']),
                    'repeat': int(m['repeat'])}
            # per-channel relative band power
            for b in BAND_ORDER:
                rel = bp[b][i] / np.where(tot[i] > 0, tot[i], np.nan)
                for ci, ch in enumerate(EEG_CHANNELS):
                    feat[f'feat_rel_{b}_{ch}'] = rel[ci]
                # ROI absolute + relative
                for r, idx in roi_idx.items():
                    feat[f'feat_abs_{b}_{r}'] = bp[b][i, idx].mean()
                    feat[f'feat_rel_{b}_{r}'] = np.nanmean(rel[idx])
            rows.append(feat)
    df = pd.DataFrame(rows)
    if logger:
        nfeat = len([c for c in df.columns if c.startswith('feat_')])
        logger.info(f"ML features: {len(df)} trials × {nfeat} features")
    return df


def _models(random_state: int, n_classes: int) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    def pipe(clf):
        return Pipeline([('impute', SimpleImputer(strategy='mean')),
                         ('scale', StandardScaler()), ('clf', clf)])

    models = {
        'logreg': pipe(LogisticRegression(max_iter=2000, C=1.0,
                                          class_weight='balanced',
                                          random_state=random_state)),
        'svm': pipe(SVC(kernel='rbf', C=1.0, class_weight='balanced',
                        random_state=random_state)),
        'rf': pipe(RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                          random_state=random_state, n_jobs=-1)),
    }
    if XGB_AVAILABLE:
        from xgboost import XGBClassifier
        models['xgboost'] = pipe(XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, eval_metric='mlogloss', random_state=random_state,
            n_jobs=-1, num_class=n_classes if n_classes > 2 else None))
    return models


def run_loso(features: pd.DataFrame, task: str, cfg: Dict[str, Any],
             models: Optional[List[str]] = None,
             logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """LOSO CV for one task ('substance' or 'intensity').

    Returns {model: {accuracy, f1_macro, confusion_matrix, per_fold, ...}, meta}.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    rs = cfg['ml'].get('random_state', 42)
    df = features.copy()
    if task == 'substance':
        df = df[df['substance'].isin(['Sucrose', 'Sucralose'])]
        ycol = 'substance'
    elif task == 'intensity':
        df = df[df['substance'].isin(['Sucrose', 'Sucralose'])]
        ycol = 'intensity'
    else:
        raise ValueError(task)

    feat_cols = [c for c in df.columns if c.startswith('feat_')]
    X = df[feat_cols].to_numpy(float)
    le = LabelEncoder()
    y = le.fit_transform(df[ycol].astype(str).values)
    groups = df['subject'].values
    class_names = list(le.classes_)
    n_classes = len(class_names)

    wanted = models or cfg['ml'].get('models', ['logreg', 'svm', 'rf'])
    if XGB_AVAILABLE and 'xgboost' not in wanted:
        wanted = wanted + ['xgboost']
    model_objs = _models(rs, n_classes)

    logo = LeaveOneGroupOut()
    results = {}
    for name in wanted:
        if name not in model_objs:
            continue
        y_true, y_pred, fold_acc, fold_sids = [], [], [], []
        for tr, te in logo.split(X, y, groups):
            clf = model_objs[name]
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            y_true.extend(y[te]); y_pred.extend(p)
            fold_acc.append(accuracy_score(y[te], p))
            fold_sids.append(groups[te][0])
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        results[name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'per_fold_accuracy': fold_acc, 'fold_subjects': fold_sids,
            'mean_fold_accuracy': float(np.mean(fold_acc)),
            'std_fold_accuracy': float(np.std(fold_acc)),
        }
        if logger:
            logger.info(f"  [{task}] {name}: acc={results[name]['accuracy']:.3f} "
                        f"f1={results[name]['f1_macro']:.3f}")
    results['_meta'] = {'task': task, 'class_names': class_names,
                        'chance': 1.0 / n_classes, 'n_trials': len(df),
                        'n_features': len(feat_cols)}
    return results
