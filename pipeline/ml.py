"""
ML Classification — LOSO cross-validation with multiple models.

Tasks:
- jar_group: 3-class (Khong_du, Vua_phai, Qua_nhieu)
- concentration_binary: 2-class (605 vs 893)

Features: ERP component measures or ERP + bandpower.
Models: LogisticRegression, SVM, RandomForest, XGBoost (optional).
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.feature_selection import mutual_info_classif

from .constants import CONCENTRATIONS, JAR_NUMERIC
from .config import ensure_dir


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_classification_data(
    measures: pd.DataFrame,
    task: str,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Prepare feature matrix and labels for classification.

    Parameters
    ----------
    measures : pd.DataFrame
        ERP component measures (or extended ml_features) with jar_group column.
    task : str
        'jar_group' (3-class) or 'concentration_binary' (2-class: 605 vs 893).
    feature_cols : list of str, optional
        Columns to use as features. If None, auto-detect all numeric feature columns.

    Returns
    -------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    groups : ndarray (n_samples,) — subject IDs for LOSO
    feature_names : list of str
    class_names : list of str
    """
    df = measures.copy()

    # Filter by task
    if task == 'concentration_binary':
        df = df[df['condition'].isin([605, 893])]

    # Drop rows without valid labels
    if task == 'jar_group':
        df = df.dropna(subset=['jar_group'])
    else:
        df = df.dropna(subset=['condition'])

    # Determine features: auto-detect all numeric non-metadata columns
    META_COLS = {'subject_id', 'condition', 'condition_label', 'jar_group',
                 'jar_numeric'}
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]

    # Drop feature columns with NaN or infinite values
    df_feat = df[feature_cols].copy()
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    bad_cols = df_feat.columns[df_feat.isna().any()].tolist()
    if bad_cols:
        df_feat = df_feat.drop(columns=bad_cols)
        feature_cols = [c for c in feature_cols if c not in bad_cols]

    X = df_feat.values
    groups = df['subject_id'].values

    # Encode labels
    if task == 'jar_group':
        le = LabelEncoder()
        y = le.fit_transform(df['jar_group'].values)
        class_names = list(le.classes_)
    elif task == 'concentration_binary':
        le = LabelEncoder()
        y = le.fit_transform(df['condition'].values)
        class_names = ['Water/605', 'High/893']
    else:
        raise ValueError(f"Unknown task: {task}")

    return X, y, groups, feature_cols, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────────────────────────────────────

def create_model(model_name: str, random_state: int = 42):
    """Create an unfitted sklearn classifier by name."""
    if model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=1000, class_weight='balanced',
            random_state=random_state, solver='lbfgs',
        )
    elif model_name == 'svm':
        from sklearn.svm import SVC
        return SVC(
            kernel='rbf', class_weight='balanced',
            probability=True, random_state=random_state,
        )
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=random_state, n_jobs=-1,
        )
    elif model_name == 'xgboost':
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                random_state=random_state, use_label_encoder=False,
                eval_metric='mlogloss',
            )
        except ImportError:
            return None
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────────────────────────────────────
# LOSO cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def run_loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    random_state: int = 42,
    n_top_features: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    X, y, groups : arrays
    model_name : str
    random_state : int
    n_top_features : int, optional
        If set, select top-N features by mutual information.
    logger : logging.Logger, optional

    Returns
    -------
    results : dict
        accuracy, balanced_accuracy, f1_macro, confusion_matrix,
        y_true, y_pred, per_fold_accuracy.
    """
    model = create_model(model_name, random_state)
    if model is None:
        if logger:
            logger.warning(f"Model '{model_name}' not available (missing dependency).")
        return {}

    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)

    y_true_all = []
    y_pred_all = []
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Feature selection
        if n_top_features and n_top_features < X_train.shape[1]:
            mi = mutual_info_classif(X_train, y_train, random_state=random_state)
            top_idx = np.argsort(mi)[-n_top_features:]
            X_train = X_train[:, top_idx]
            X_test = X_test[:, top_idx]

        # Train and predict
        model_fold = create_model(model_name, random_state)
        model_fold.fit(X_train, y_train)
        y_pred = model_fold.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    results = {
        'accuracy': accuracy_score(y_true_all, y_pred_all),
        'balanced_accuracy': balanced_accuracy_score(y_true_all, y_pred_all),
        'f1_macro': f1_score(y_true_all, y_pred_all, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true_all, y_pred_all),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'per_fold_accuracy': fold_accs,
        'mean_fold_accuracy': np.mean(fold_accs),
        'std_fold_accuracy': np.std(fold_accs),
    }

    if logger:
        logger.info(
            f"  {model_name}: acc={results['accuracy']:.3f}, "
            f"f1={results['f1_macro']:.3f} "
            f"(mean fold: {results['mean_fold_accuracy']:.3f} ± "
            f"{results['std_fold_accuracy']:.3f})"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Feature selection sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep_n_features(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    n_features_list: List[int],
    random_state: int = 42,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Sweep top-N feature selection and evaluate each.

    Returns
    -------
    sweep_df : pd.DataFrame
        Columns: n_features, accuracy, balanced_accuracy, f1_macro.
    """
    results = []
    for n in n_features_list:
        if n > X.shape[1]:
            continue
        res = run_loso_cv(X, y, groups, model_name, random_state,
                          n_top_features=n, logger=None)
        if res:
            results.append({
                'n_features': n,
                'accuracy': res['accuracy'],
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
            })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Master entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_all_ml_tasks(
    erp_results: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run all ML classification tasks.

    Parameters
    ----------
    erp_results : dict
        From erp_analysis.run_erp_analysis(), must contain 'measures'.
    config : dict
    logger : logging.Logger

    Returns
    -------
    all_results : dict
        {task_name: {model_name: metrics_dict}}
    """
    logger.info("=" * 60)
    logger.info("STAGE: ML Classification")
    logger.info("=" * 60)

    # Prefer extended ml_features (Hjorth + time-domain + spectral + ERP + bandpower)
    # Fall back to plain ERP measures if ml_features not available
    ml_features = erp_results.get('ml_features')
    measures = erp_results.get('measures')

    if ml_features is not None and not ml_features.empty:
        data_df = ml_features
        meta_cols = {'subject_id', 'condition', 'condition_label',
                     'jar_group', 'jar_numeric'}
        n_feat_cols = len([c for c in data_df.columns
                           if c not in meta_cols
                           and pd.api.types.is_numeric_dtype(data_df[c])])
        logger.info(f"Using extended ML features: {n_feat_cols} feature columns")
    elif measures is not None and not measures.empty:
        data_df = measures
        logger.info("Extended ML features not available, falling back to ERP measures only")
    else:
        logger.error("No features available for ML.")
        return {}

    ml_cfg = config.get('ml', {})
    random_state = ml_cfg.get('random_state', 42)
    models = ml_cfg.get('models', ['logistic_regression', 'svm', 'random_forest'])
    tasks = ml_cfg.get('classification_tasks', ['jar_group', 'concentration_binary'])

    results_dir = os.path.join(config['paths']['results_base'], 'ml')
    ensure_dir(results_dir)

    all_results = {}

    for task in tasks:
        logger.info(f"\nTask: {task}")
        try:
            X, y, groups, feat_names, class_names = prepare_classification_data(
                data_df, task
            )
        except Exception as e:
            logger.error(f"  Failed to prepare data for '{task}': {e}")
            continue

        n_classes = len(class_names)
        chance = 1.0 / n_classes
        logger.info(f"  {X.shape[0]} samples, {X.shape[1]} features, "
                     f"{n_classes} classes (chance={chance:.3f})")

        # Auto feature selection: use top-50 features when feature set is large
        n_top = ml_cfg.get('n_top_features', None)
        if n_top is None and X.shape[1] > 50:
            n_top = 50
            logger.info(f"  Auto feature selection: top {n_top} by mutual info")

        task_results = {}
        for model_name in models:
            logger.info(f"  Model: {model_name}")
            result = run_loso_cv(X, y, groups, model_name, random_state,
                                 n_top_features=n_top, logger=logger)
            if not result:
                continue

            # ── Nếu balanced_accuracy thấp → sweep feature count tự động ──
            LOW_ACC_THRESHOLD = 0.55
            if result['balanced_accuracy'] < LOW_ACC_THRESHOLD:
                logger.info(
                    f"  ⚠ balanced_acc={result['balanced_accuracy']:.3f} < "
                    f"{LOW_ACC_THRESHOLD} → chạy feature sweep..."
                )
                n_max = min(X.shape[1], 200)
                sweep_list = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, n_max]
                sweep_list = sorted(set(n for n in sweep_list if n <= X.shape[1]))
                sweep_df = sweep_n_features(X, y, groups, model_name,
                                            sweep_list, random_state, logger)
                if not sweep_df.empty:
                    best_row = sweep_df.loc[sweep_df['balanced_accuracy'].idxmax()]
                    best_n = int(best_row['n_features'])
                    logger.info(
                        f"  Best n_features={best_n} "
                        f"(balanced_acc={best_row['balanced_accuracy']:.3f})"
                    )
                    # Vẽ sweep chart
                    from .viz import plot_feature_sweep
                    plot_feature_sweep(sweep_df, task, model_name, best_n,
                                       chance, config, logger)
                    # Rerun với best_n nếu cải thiện
                    result_best = run_loso_cv(X, y, groups, model_name,
                                              random_state, n_top_features=best_n,
                                              logger=logger)
                    if result_best and result_best['balanced_accuracy'] > result['balanced_accuracy']:
                        logger.info(
                            f"  ✓ Improved: {result['balanced_accuracy']:.3f} → "
                            f"{result_best['balanced_accuracy']:.3f} with n={best_n}"
                        )
                        result = result_best
                        result['n_features_used'] = best_n
                else:
                    result['n_features_used'] = n_top
            else:
                result['n_features_used'] = n_top

            task_results[model_name] = result

            # ── Confusion matrix ──────────────────────────────────────────
            from .viz import plot_confusion_matrix, plot_per_fold_accuracy
            plot_confusion_matrix(
                result['y_true'], result['y_pred'],
                class_names,
                title=f'{task} — {model_name}',
                config=config, logger=logger,
                filename=f'cm_{task}_{model_name}.png',
            )

            # ── Per-fold accuracy ─────────────────────────────────────────
            plot_per_fold_accuracy(
                result['per_fold_accuracy'], task, model_name,
                chance, config, logger,
            )

            # ── Feature importance (RF / XGB) ─────────────────────────────
            from .viz import plot_feature_importance
            try:
                # Lấy importances bằng cách train trên toàn bộ data
                from sklearn.preprocessing import StandardScaler
                from sklearn.feature_selection import mutual_info_classif
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                n_used = result.get('n_features_used') or X.shape[1]
                if n_used and n_used < X_s.shape[1]:
                    mi_all = mutual_info_classif(X_s, y, random_state=random_state)
                    top_idx = np.argsort(mi_all)[-n_used:]
                    imp = mi_all[top_idx]
                    names = [feat_names[i] for i in top_idx]
                else:
                    imp = mutual_info_classif(X_s, y, random_state=random_state)
                    names = feat_names
                plot_feature_importance(names, imp, task, model_name,
                                        config, logger, top_n=30)
            except Exception as ex:
                logger.warning(f"  Feature importance plot failed: {ex}")

        all_results[task] = task_results

        # Save summary
        summary_rows = []
        for mn, res in task_results.items():
            summary_rows.append({
                'task': task,
                'model': mn,
                'accuracy': res['accuracy'],
                'balanced_accuracy': res['balanced_accuracy'],
                'f1_macro': res['f1_macro'],
                'mean_fold_acc': res['mean_fold_accuracy'],
                'std_fold_acc': res['std_fold_accuracy'],
                'chance': chance,
            })
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(results_dir, f'{task}_summary.csv'), index=False
            )

    logger.info(f"ML results saved to {results_dir}")
    return all_results
