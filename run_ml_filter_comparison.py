#!/usr/bin/env python3
"""
ML Filter Comparison
====================
Compare quality filters for ML on the same feature matrix:
  - none   : keep all 168 subj×cond
  - weak   : drop BAD only (keep GOOD+WEAK)
  - strict : drop BAD+WEAK (keep GOOD only)
  - snr_q1 : continuous filter — drop bottom-25% by avg_SNR

Tasks (LOSO-CV):
  - JAR 3-class           (Khong_du vs Vua_phai vs Qua_nhieu)
  - Vua_phai vs Others    (binary, positive = "just right")
  - High vs Water         (binary, conditions 893 vs 605)

Reuses cached features from output/results/ml_jar3/features_jar3_adv.csv
(168 rows × 1362 features, one row per subject×condition).

Outputs:
  output/results/ml_filter_comparison/all_results.csv
  output/figures/ml_filter_comparison/accuracy_by_filter.png
"""

import os, sys, warnings, datetime
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Tee logger — write stdout to both terminal and log file
# ──────────────────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

SEED = 42
np.random.seed(SEED)

FEAT_CSV = 'output/results/ml_jar3/features_jar3_adv.csv'
QUAL_CSV = 'output/results/erp/erp_quality_flags.csv'
OUT_DIR  = 'output/results/ml_filter_comparison'
FIG_DIR  = 'output/figures/ml_filter_comparison'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TOP_K   = 5            # match report finding K=5 optimal
META    = ['subject_id', 'condition', 'jar_group']


# ──────────────────────────────────────────────────────────────────────────────
# Load & merge
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    qf = pd.read_csv(QUAL_CSV)[['subject_id', 'condition', 'quality_label', 'avg_snr']]
    df['condition'] = df['condition'].astype(int)
    qf['condition'] = qf['condition'].astype(int)
    df = df.merge(qf, on=['subject_id', 'condition'], how='left')
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────────
def apply_filter(df, name):
    if name == 'none':
        return df.copy()
    if name == 'weak':           # drop BAD
        return df[df['quality_label'] != 'BAD'].copy()
    if name == 'strict':         # GOOD only
        return df[df['quality_label'] == 'GOOD'].copy()
    if name == 'snr_q1':         # drop bottom quartile of avg_SNR
        thr = df['avg_snr'].quantile(0.25)
        return df[df['avg_snr'] >= thr].copy()
    raise ValueError(name)


# ──────────────────────────────────────────────────────────────────────────────
# Task setup
# ──────────────────────────────────────────────────────────────────────────────
def make_task(df, task):
    """Return (X, y, groups, label_names, chance)."""
    feat_cols = [c for c in df.columns
                 if c not in META + ['quality_label', 'avg_snr']]
    X_raw = df[feat_cols].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    var = X_raw.var(axis=0)
    feat_cols = [c for c, v in zip(feat_cols, var) if v > 1e-12]
    X_raw = X_raw[:, var > 1e-12]
    groups = df['subject_id'].values

    if task == 'JAR3':
        mask = df['jar_group'].notna().values
        order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
        y_str = df.loc[mask, 'jar_group'].values
        y = np.array([order.index(j) for j in y_str])
        X = X_raw[mask]
        g = groups[mask]
        return X, y, g, order, 1/3, feat_cols
    if task == 'VuaphaiVsOthers':
        mask = df['jar_group'].notna().values
        y = (df.loc[mask, 'jar_group'].values == 'Vua_phai').astype(int)
        X = X_raw[mask]
        g = groups[mask]
        return X, y, g, ['Other', 'Vua_phai'], 0.5, feat_cols
    if task == 'HighVsWater':
        mask = df['condition'].isin([605, 893]).values
        y = (df.loc[mask, 'condition'].values == 893).astype(int)
        X = X_raw[mask]
        g = groups[mask]
        return X, y, g, ['Water', 'High'], 0.5, feat_cols
    raise ValueError(task)


# ──────────────────────────────────────────────────────────────────────────────
# LOSO with MI top-K (selection inside fold to avoid leakage)
# ──────────────────────────────────────────────────────────────────────────────
def loso_with_mi(X, y, groups, model_factory, k=TOP_K):
    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []

    # If too few classes per fold (binary task with one class missing), skip fold
    for tr, te in logo.split(X, y, groups):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_tr)) < 2:
            continue
        if len(np.unique(y_te)) < 1 or len(y_te) == 0:
            continue
        # scale
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
        X_te = np.nan_to_num(X_te, nan=0, posinf=0, neginf=0)
        # MI selection on training fold
        mi = mutual_info_classif(X_tr, y_tr, random_state=SEED)
        top_idx = np.argsort(mi)[::-1][:k]
        X_tr = X_tr[:, top_idx]
        X_te = X_te[:, top_idx]
        # fit
        m = model_factory()
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    if not y_true_all:
        return {'accuracy': np.nan, 'balanced_acc': np.nan, 'f1': np.nan,
                'n_pred': 0}
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    return {
        'accuracy':     accuracy_score(y_true, y_pred),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        'f1':           f1_score(y_true, y_pred, average='macro',
                                  zero_division=0),
        'n_pred':       len(y_true),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ── set up log file ───────────────────────────────────────────────────────
    log_dir = os.path.join(OUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'run_{ts}.log')
    latest_path = os.path.join(OUT_DIR, 'run.log')  # also stable "latest" log
    log_fh = open(log_path, 'w')
    latest_fh = open(latest_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_fh, latest_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh, latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] '
          f'logging to {log_path}')
    print('\n' + '='*72)
    print('  ML FILTER COMPARISON — none / weak / strict / snr_q1')
    print('='*72)
    df_all = load_data()
    print(f'Loaded features: {df_all.shape}, '
          f'quality labels merged ({df_all.quality_label.value_counts().to_dict()})')

    filters = ['none', 'weak', 'strict', 'snr_q1']
    tasks   = ['JAR3', 'VuaphaiVsOthers', 'HighVsWater']
    models  = {
        'LogReg': lambda: LogisticRegression(max_iter=2000, C=1.0,
                                              class_weight='balanced',
                                              solver='lbfgs',
                                              random_state=SEED),
        'RForest': lambda: RandomForestClassifier(n_estimators=300,
                                                   class_weight='balanced',
                                                   random_state=SEED, n_jobs=-1),
    }

    rows = []
    for filt in filters:
        df = apply_filter(df_all, filt)
        n_rows = len(df)
        n_subj = df['subject_id'].nunique()
        for task in tasks:
            try:
                X, y, g, lbls, chance, feat_cols = make_task(df, task)
            except Exception as e:
                print(f'  ! {filt}/{task} → {e}')
                continue
            cls_dist = {lbls[i]: int((y == i).sum()) for i in range(len(lbls))}
            for mname, mfac in models.items():
                res = loso_with_mi(X, y, g, mfac, k=TOP_K)
                rows.append({
                    'filter': filt, 'task': task, 'model': mname,
                    'n_samples': len(y),
                    'n_subjects': len(np.unique(g)),
                    'n_features': len(feat_cols),
                    'chance': chance,
                    'accuracy': res['accuracy'],
                    'balanced_acc': res['balanced_acc'],
                    'f1': res['f1'],
                    'class_dist': str(cls_dist),
                })
                acc = res['accuracy']
                acc_str = f'{acc:.3f}' if not np.isnan(acc) else 'NA'
                bacc_str = (f'{res["balanced_acc"]:.3f}'
                            if not np.isnan(res['balanced_acc']) else 'NA')
                print(f'  [{filt:>7}] {task:<18} {mname:<8} '
                      f'n={len(y):>3} subj={len(np.unique(g)):>2} '
                      f'acc={acc_str} bacc={bacc_str} chance={chance:.2f}')

    df_res = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, 'all_results.csv')
    df_res.to_csv(csv_path, index=False)
    print(f'\n✓ Saved: {csv_path}')

    # ── Figure: grouped bar chart ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    for ax, task in zip(axes, tasks):
        sub = df_res[df_res['task'] == task]
        pivot_acc = sub.pivot_table(index='filter', columns='model',
                                     values='accuracy').reindex(filters)
        pivot_bacc = sub.pivot_table(index='filter', columns='model',
                                      values='balanced_acc').reindex(filters)
        chance = sub['chance'].iloc[0]

        x = np.arange(len(filters))
        w = 0.18
        colors = {'LogReg': '#2E86AB', 'RForest': '#A23B72'}
        for i, m in enumerate(['LogReg', 'RForest']):
            ax.bar(x + (i - 0.5) * w * 2, pivot_acc[m].values, w * 1.6,
                   label=f'{m} acc', color=colors[m], alpha=0.95)
            ax.bar(x + (i - 0.5) * w * 2 + w * 1.6 * 0.0,
                   pivot_bacc[m].values, w * 1.6 * 0.0)  # placeholder

        # overlay balanced acc as hatched bar to the right
        for i, m in enumerate(['LogReg', 'RForest']):
            ax.bar(x + (i + 1.0) * w * 1.0, pivot_bacc[m].values, w,
                   color=colors[m], alpha=0.45, hatch='//',
                   label=f'{m} bal_acc')

        ax.axhline(chance, color='red', linestyle='--', linewidth=1.2,
                   label=f'chance={chance:.2f}')
        ax.set_xticks(x)
        ax.set_xticklabels(filters, rotation=0)
        ax.set_ylabel('Score')
        ax.set_title(task, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(1.0, pivot_acc.values.max() * 1.15))
        ax.grid(axis='y', alpha=0.3)
        # n_samples annotation
        n_by_filt = sub.drop_duplicates(['filter'])[['filter', 'n_samples',
                                                      'n_subjects']]
        for xi, filt in enumerate(filters):
            r = n_by_filt[n_by_filt['filter'] == filt]
            if not r.empty:
                ax.annotate(f'n={int(r["n_samples"].iloc[0])}\n'
                            f's={int(r["n_subjects"].iloc[0])}',
                            xy=(xi, 0.02), ha='center', fontsize=8,
                            color='gray')
        ax.legend(fontsize=7, ncol=2, loc='upper right')

    fig.suptitle('ML Filter Comparison — LOSO-CV, MI top-5 features (selected inside fold)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'accuracy_by_filter.png')
    fig.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ Saved: {fig_path}')

    # ── Print summary table ────────────────────────────────────────────────────
    print('\n' + '='*72)
    print('  SUMMARY (best model per filter × task)')
    print('='*72)
    print(f'{"task":<20}{"filter":<10}{"n":<6}{"subj":<6}'
          f'{"model":<10}{"acc":<8}{"bal_acc":<10}{"f1":<8}')
    for task in tasks:
        sub = df_res[df_res['task'] == task]
        for filt in filters:
            cell = sub[sub['filter'] == filt]
            if cell.empty:
                continue
            best = cell.sort_values('balanced_acc', ascending=False).iloc[0]
            print(f'{task:<20}{filt:<10}{int(best["n_samples"]):<6}'
                  f'{int(best["n_subjects"]):<6}{best["model"]:<10}'
                  f'{best["accuracy"]:.3f}   {best["balanced_acc"]:.3f}     '
                  f'{best["f1"]:.3f}')
    print('='*72 + '\n')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')

    log_fh.close()
    latest_fh.close()


if __name__ == '__main__':
    main()
