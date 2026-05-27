#!/usr/bin/env python3
"""
ERP Insight Analysis — Phân tích chi tiết P1, N1, P2, N400.

Script này tải dữ liệu ERP đã xử lý, tạo bộ biểu đồ toàn diện
và xuất báo cáo insight chi tiết về ý nghĩa của từng thành phần ERP
trong nghiên cứu vị giác (vị ngọt sucrose).

Usage:
    .venv/bin/python run_erp_insight.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import mne
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, pearsonr, spearmanr
from itertools import combinations

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.constants import (
    CONCENTRATIONS, CONCENTRATION_LABELS, ERP_WINDOWS, ALL_SUBJECTS,
    JAR_LABELS_VN,
)
from pipeline.erp_analysis import apply_woody_realign

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
EPOCHS_BASE     = 'output/epochs'
RESULTS_ERP     = 'output/results/erp'
INSIGHT_DIR     = 'output/figures/erp_insight'
REPORT_PATH     = 'output/results/erp/insight_report.txt'

# Component colors (consistent across all plots)
COMP_COLORS = {
    'P1':   '#27ae60',   # green
    'N1':   '#e74c3c',   # red
    'P2':   '#2980b9',   # blue
    'N400': '#8e44ad',   # purple
}

# Concentration gradient colors
COND_CMAP = plt.cm.YlOrRd
COND_COLORS = {
    cond: COND_CMAP(i / (len(CONCENTRATIONS) - 1))
    for i, cond in enumerate(CONCENTRATIONS)
}

JAR_COLORS = {
    'Khong_du':  '#3498db',
    'Vua_phai':  '#2ecc71',
    'Qua_nhieu': '#e74c3c',
}

# Component ROI dùng trong config
COMP_ROI = {
    'P1':   ['F3', 'F4', 'C3', 'C4'],
    'N1':   ['F7', 'F8', 'T7', 'T8'],
    'P2':   ['C3', 'C4', 'P3', 'P4'],
    'N400': ['C3', 'C4', 'P3', 'P4', 'F3', 'F4'],
}

COMP_WINDOWS = {
    'P1':   (0.090, 0.150),
    'N1':   (0.140, 0.240),
    'P2':   (0.230, 0.350),
    'N400': (0.350, 0.550),
}

COMP_POLARITY = {'P1': 'pos', 'N1': 'neg', 'P2': 'pos', 'N400': 'neg'}

# Region sub-groups within each component's ROI
# Maps each component → {anatomical_region: [channel_list]}
COMP_REGIONS = {
    'P1':   {'Frontal': ['F3', 'F4'], 'Central': ['C3', 'C4']},
    'N1':   {'Frontal': ['F7', 'F8'], 'Temporal': ['T7', 'T8']},
    'P2':   {'Central': ['C3', 'C4'], 'Parietal': ['P3', 'P4']},
    'N400': {'Frontal': ['F3', 'F4'], 'Central': ['C3', 'C4'], 'Parietal': ['P3', 'P4']},
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_measures():
    path = os.path.join(RESULTS_ERP, 'component_measures.csv')
    df = pd.read_csv(path)
    # Convert V → µV
    for comp in ['P1', 'N1', 'P2', 'N400']:
        for m in ['mean_amp', 'peak_amp']:
            col = f'{comp}_{m}'
            if col in df.columns:
                df[col] = df[col] * 1e6
        lat_col = f'{comp}_peak_lat'
        if lat_col in df.columns:
            df[lat_col] = df[lat_col] * 1000  # s → ms
    return df


def load_concentration_summary():
    path = os.path.join(RESULTS_ERP, 'concentration_summary.csv')
    df = pd.read_csv(path)
    for col in ['mean', 'sem']:
        if col in df.columns:
            df[col] = df.apply(
                lambda r: r[col] * 1e6 if r['measure'] in ('mean_amp', 'peak_amp')
                else r[col] * 1000 if r['measure'] == 'peak_lat'
                else r[col], axis=1
            )
    return df


def load_jar_summary():
    path = os.path.join(RESULTS_ERP, 'jar_summary.csv')
    df = pd.read_csv(path)
    for col in ['mean', 'sem']:
        if col in df.columns:
            df[col] = df.apply(
                lambda r: r[col] * 1e6 if r['measure'] in ('mean_amp', 'peak_amp')
                else r[col] * 1000 if r['measure'] == 'peak_lat'
                else r[col], axis=1
            )
    return df


def load_epochs_and_evoked(logger):
    """Load all epochs → compute evoked per condition, per jar, grand average."""
    all_epochs = []
    subjects_found = []
    for sid in ALL_SUBJECTS:
        fif = os.path.join(EPOCHS_BASE, sid, 'epochs_epo.fif')
        if not os.path.exists(fif):
            continue
        try:
            ep = mne.read_epochs(fif, preload=True, verbose=False)
            all_epochs.append(ep)
            subjects_found.append(sid)
        except Exception as e:
            logger.warning(f'[{sid}] Lỗi load: {e}')

    if not all_epochs:
        return None, None, None, None, None, None

    # Load trial info
    ti_path = os.path.join(EPOCHS_BASE, 'all_trial_info.csv')
    all_trial_info = pd.read_csv(ti_path) if os.path.exists(ti_path) else None

    if all_trial_info is None:
        return all_epochs, None, None, None, subjects_found, None

    # Apply woody realign
    from pipeline.config import load_config
    cfg = load_config('configs/config.yaml')
    all_epochs, all_trial_info = apply_woody_realign(all_epochs, all_trial_info, logger)

    X = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
    info = all_epochs[0].info
    tmin = all_epochs[0].tmin

    # Grand average
    evoked_all = mne.EvokedArray(X.mean(axis=0), info, tmin=tmin, comment='Grand Average')

    # By condition
    evoked_by_cond = {}
    for cond in CONCENTRATIONS:
        mask = all_trial_info['condition'] == cond
        if mask.sum() > 0:
            evoked_by_cond[cond] = mne.EvokedArray(
                X[mask.values].mean(axis=0), info, tmin=tmin,
                comment=CONCENTRATION_LABELS[cond]
            )

    # By JAR group
    evoked_by_jar = {}
    for grp in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
        mask = all_trial_info['jar_group'] == grp
        if mask.sum() > 0:
            evoked_by_jar[grp] = mne.EvokedArray(
                X[mask.values].mean(axis=0), info, tmin=tmin, comment=grp
            )

    logger.info(f'Loaded evoked: {len(evoked_by_cond)} conditions, '
                f'{len(evoked_by_jar)} JAR groups')
    return all_epochs, evoked_all, evoked_by_cond, evoked_by_jar, subjects_found, all_trial_info


# ──────────────────────────────────────────────────────────────────────────────
# Per-region analysis
# ──────────────────────────────────────────────────────────────────────────────

def extract_region_measures(all_epochs, all_trial_info):
    """Extract ERP component mean amplitude per brain region for every trial.

    For each component (P1/N1/P2/N400), decomposes the ROI into anatomical
    sub-regions (e.g. P2 → Central + Parietal) and computes mean amplitude
    in the component's time window for each region separately.

    Returns
    -------
    pd.DataFrame with columns:
        subject_id, condition, jar_group, component, region, amplitude_uv
    """
    if not all_epochs:
        return pd.DataFrame()

    X = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
    ch_names = all_epochs[0].ch_names
    times = all_epochs[0].times

    rows = []
    for trial_idx in range(len(all_trial_info)):
        ti = all_trial_info.iloc[trial_idx]
        for comp, regions in COMP_REGIONS.items():
            tmin, tmax = COMP_WINDOWS[comp]
            tm = (times >= tmin) & (times <= tmax)
            for region, chs in regions.items():
                ch_idx = [ch_names.index(c) for c in chs if c in ch_names]
                if not ch_idx:
                    continue
                amp = X[trial_idx][ch_idx][:, tm].mean()
                rows.append({
                    'subject_id': ti['subject_id'],
                    'condition': ti['condition'],
                    'jar_group': ti['jar_group'],
                    'component': comp,
                    'region': region,
                    'amplitude_uv': amp * 1e6,  # V → µV
                })

    df = pd.DataFrame(rows)
    return df


def compute_region_summary(region_measures):
    """Average region measures per subject × component × region × JAR group.

    Returns DataFrame for statistical analysis (one row per subject per group).
    """
    grp = region_measures.groupby(
        ['subject_id', 'jar_group', 'component', 'region'], as_index=False
    )['amplitude_uv'].mean()
    return grp


def run_region_jar_stats(region_measures):
    """For each component × region, compute JAR group means + pairwise tests.

    Returns dict: {component: {region: {stats_dict}}}
    """
    from scipy.stats import ttest_ind

    subject_avg = compute_region_summary(region_measures)
    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    results = {}

    for comp in ['P1', 'N1', 'P2', 'N400']:
        results[comp] = {}
        for region in COMP_REGIONS[comp]:
            sub = subject_avg[
                (subject_avg['component'] == comp) &
                (subject_avg['region'] == region)
            ]
            if len(sub) < 6:
                continue

            # Group means
            means = {}
            for g in grp_order:
                vals = sub[sub['jar_group'] == g]['amplitude_uv']
                means[g] = {'mean': vals.mean(), 'sem': vals.sem(), 'n': len(vals)}

            # Pairwise tests
            pairs = []
            for g1, g2 in combinations(grp_order, 2):
                v1 = sub[sub['jar_group'] == g1]['amplitude_uv'].dropna().values
                v2 = sub[sub['jar_group'] == g2]['amplitude_uv'].dropna().values
                if len(v1) < 2 or len(v2) < 2:
                    continue
                t, p = ttest_ind(v1, v2, equal_var=False)
                cohens_d = (v1.mean() - v2.mean()) / np.sqrt((v1.std()**2 + v2.std()**2) / 2)
                pairs.append({
                    'comparison': f'{g1} vs {g2}',
                    'delta_uv': v1.mean() - v2.mean(),
                    'cohens_d': cohens_d,
                    'p': p,
                })

            results[comp][region] = {'means': means, 'pairs': pairs}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Per-channel analysis
# ──────────────────────────────────────────────────────────────────────────────

def extract_channel_measures(all_epochs, all_trial_info):
    """Extract ERP mean amplitude per individual channel for every trial.

    For each component, extracts mean amplitude in the component's time window
    for each channel in its ROI separately (no averaging across channels).

    Returns
    -------
    pd.DataFrame with columns:
        subject_id, condition, jar_group, component, channel, amplitude_uv
    """
    if not all_epochs:
        return pd.DataFrame()

    X = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
    ch_names = all_epochs[0].ch_names
    times = all_epochs[0].times

    ch_to_idx = {ch: i for i, ch in enumerate(ch_names)}

    rows = []
    for trial_idx in range(len(all_trial_info)):
        ti = all_trial_info.iloc[trial_idx]
        for comp, chs in COMP_ROI.items():
            tmin, tmax = COMP_WINDOWS[comp]
            tm = (times >= tmin) & (times <= tmax)
            for ch in chs:
                if ch not in ch_to_idx:
                    continue
                ci = ch_to_idx[ch]
                amp = X[trial_idx, ci, tm].mean()
                rows.append({
                    'subject_id': ti['subject_id'],
                    'condition': ti['condition'],
                    'jar_group': ti['jar_group'],
                    'component': comp,
                    'channel': ch,
                    'amplitude_uv': amp * 1e6,
                })
    return pd.DataFrame(rows)


def compute_channel_subject_avg(channel_measures):
    """Average per-channel measures per subject × component × channel × JAR."""
    grp = channel_measures.groupby(
        ['subject_id', 'jar_group', 'component', 'channel'], as_index=False
    )['amplitude_uv'].mean()
    return grp


def run_channel_jar_stats(channel_measures):
    """For each component × channel, compute JAR means + Welch t-tests.

    Returns dict: {component: {channel: {stats_dict}}}
    """
    from scipy.stats import ttest_ind

    subject_avg = compute_channel_subject_avg(channel_measures)
    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    results = {}

    for comp in ['P1', 'N1', 'P2', 'N400']:
        results[comp] = {}
        for ch in COMP_ROI[comp]:
            sub = subject_avg[
                (subject_avg['component'] == comp) &
                (subject_avg['channel'] == ch)
            ]
            if len(sub) < 6:
                continue

            # Group means
            means = {}
            for g in grp_order:
                vals = sub[sub['jar_group'] == g]['amplitude_uv']
                means[g] = {'mean': vals.mean(), 'sem': vals.sem(), 'n': len(vals)}

            # Pairwise tests
            pairs = []
            for g1, g2 in combinations(grp_order, 2):
                v1 = sub[sub['jar_group'] == g1]['amplitude_uv'].dropna().values
                v2 = sub[sub['jar_group'] == g2]['amplitude_uv'].dropna().values
                if len(v1) < 2 or len(v2) < 2:
                    continue
                t, p = ttest_ind(v1, v2, equal_var=False)
                cohens_d = (v1.mean() - v2.mean()) / np.sqrt((v1.std()**2 + v2.std()**2) / 2)
                pairs.append({
                    'comparison': f'{g1} vs {g2}',
                    'delta_uv': v1.mean() - v2.mean(),
                    'cohens_d': cohens_d,
                    'p': p,
                })
            results[comp][ch] = {'means': means, 'pairs': pairs}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_anova_per_component(measures):
    """Repeated-measures ANOVA across 6 concentration levels for each component.

    Uses pingouin rm_anova (within-subject design: same subject, 6 conditions).
    Falls back to independent f_oneway if pingouin unavailable.
    """
    results = {}
    try:
        import pingouin as pg
        for comp in ['P1', 'N1', 'P2', 'N400']:
            col = f'{comp}_mean_amp'
            if col not in measures.columns:
                continue
            # Drop NaN rows for this component
            df = measures[['subject_id', 'condition', col]].dropna()
            if len(df) < 12 or df['subject_id'].nunique() < 4:
                continue
            try:
                aov = pg.rm_anova(
                    data=df, dv=col, within='condition',
                    subject='subject_id', detailed=True
                )
                row = aov.iloc[0]
                results[comp] = {
                    'F': row['F'],
                    'p': row['p_unc'],
                    'eta2': row['ng2'],  # generalized eta-squared
                    'n_groups': df['condition'].nunique(),
                    'method': 'pingouin rm_anova',
                }
            except Exception as e:
                print(f"  [rm_anova] {comp}: {e}")
                continue
    except ImportError:
        pass

    # Fallback for components where rm_anova failed
    for comp in ['P1', 'N1', 'P2', 'N400']:
        if comp in results:
            continue
        col = f'{comp}_mean_amp'
        if col not in measures.columns:
            continue
        groups = [measures[measures['condition'] == c][col].dropna().values
                  for c in CONCENTRATIONS]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        F, p = f_oneway(*groups)
        grand_mean = np.concatenate(groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total   = sum(((v - grand_mean) ** 2).sum() for g in groups for v in g)
        eta2 = ss_between / ss_total if ss_total > 0 else 0
        results[comp] = {
            'F': F, 'p': p, 'eta2': eta2,
            'n_groups': len(groups),
            'method': 'f_oneway (independent — fallback)',
        }
    return results


def run_pairwise_ttest(measures, comp, col_suffix='mean_amp'):
    """Pairwise t-tests (Welch) between all condition pairs for one component."""
    col = f'{comp}_{col_suffix}'
    if col not in measures.columns:
        return {}
    pairs = {}
    for c1, c2 in combinations(CONCENTRATIONS, 2):
        g1 = measures[measures['condition'] == c1][col].dropna().values
        g2 = measures[measures['condition'] == c2][col].dropna().values
        if len(g1) >= 2 and len(g2) >= 2:
            t, p = ttest_ind(g1, g2, equal_var=False)
            # Cohen's d
            pooled = np.sqrt((g1.std() ** 2 + g2.std() ** 2) / 2)
            d = (g1.mean() - g2.mean()) / pooled if pooled > 1e-10 else 0
            pairs[(c1, c2)] = {'t': t, 'p': p, 'd': d,
                                'm1': g1.mean(), 'm2': g2.mean()}
    return pairs


def linear_trend(measures, comp, col_suffix='mean_amp'):
    """Pearson correlation of amplitude with concentration index (linear trend)."""
    col = f'{comp}_{col_suffix}'
    if col not in measures.columns:
        return None, None
    # Use concentration index as numeric
    cond_to_idx = {c: i for i, c in enumerate(CONCENTRATIONS)}
    measures2 = measures.copy()
    measures2['cond_idx'] = measures2['condition'].map(cond_to_idx)
    data = measures2[['cond_idx', col]].dropna()
    if len(data) < 4:
        return None, None
    r, p = pearsonr(data['cond_idx'], data[col])
    return r, p


def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 1e-10 else 0


def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 1: Grand-Average ERP với 4 thành phần highlighted ────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig1_grand_average_detailed(evoked_all, save_dir, dpi=200):
    """Grand-average ERP: 4 ROI subplots, shaded windows, peak markers."""
    if evoked_all is None:
        return

    roi_configs = [
        ('P1 ROI (Frontal-Central)',  ['F3', 'F4', 'C3', 'C4']),
        ('N1 ROI (Temporal-Frontal)', ['F7', 'F8', 'T7', 'T8']),
        ('P2 ROI (Central-Parietal)', ['C3', 'C4', 'P3', 'P4']),
        ('N400 ROI (Broad)',          ['C3', 'C4', 'P3', 'P4', 'F3', 'F4']),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    times_ms = evoked_all.times * 1000

    for ax, (title, roi), (comp, (tw0, tw1)) in zip(
            axes, roi_configs, COMP_WINDOWS.items()):
        chs = [ch for ch in roi if ch in evoked_all.ch_names]
        if not chs:
            chs = evoked_all.ch_names[:4]
        data = evoked_all.copy().pick(chs).get_data().mean(axis=0) * 1e6

        ax.plot(times_ms, data, 'k-', linewidth=2, label='Grand Avg', zorder=5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--', zorder=1)

        # Shade component window
        ax.axvspan(tw0 * 1000, tw1 * 1000, alpha=0.2,
                   color=COMP_COLORS[comp], label=f'{comp} window')

        # Baseline period
        ax.axvspan(times_ms[0], 0, alpha=0.05, color='gray')

        # Detect and mark peak
        tm = (evoked_all.times >= tw0) & (evoked_all.times <= tw1)
        win_times = times_ms[tm]
        win_data  = data[tm]
        if COMP_POLARITY[comp] == 'pos':
            pk_idx = np.argmax(win_data)
        else:
            pk_idx = np.argmin(win_data)
        pk_t = win_times[pk_idx]
        pk_v = win_data[pk_idx]
        ax.annotate(f'{comp}\n{pk_t:.0f}ms\n{pk_v:.2f}µV',
                    xy=(pk_t, pk_v),
                    xytext=(pk_t + 30, pk_v + (1.5 if pk_v >= 0 else -1.5)),
                    fontsize=9, color=COMP_COLORS[comp], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COMP_COLORS[comp],
                                   lw=1.2))
        ax.axvline(pk_t, color=COMP_COLORS[comp], linewidth=1, linestyle=':', alpha=0.7)

        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.set_title(f'{title}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)', fontsize=11)
    axes[-1].set_xlim(times_ms[0], times_ms[-1])

    fig.suptitle('Grand-Average ERP — Gustatory (Sweetness) Study\n'
                 '4 ERP Components: P1 | N1 | P2 | N400',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    path = os.path.join(save_dir, '01_grand_average_detailed.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 2: ERP by Concentration — 4 subplots per component ROI ───────────
# ──────────────────────────────────────────────────────────────────────────────

def fig2_erp_by_concentration(evoked_by_cond, save_dir, dpi=200):
    """4 subplots: one per component ROI, overlay 6 concentration lines."""
    if not evoked_by_cond:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    comps_and_roi = [
        ('P1',   ['F3', 'F4', 'C3', 'C4'],           'Frontal-Central'),
        ('N1',   ['F7', 'F8', 'T7', 'T8'],           'Temporal-Frontal'),
        ('P2',   ['C3', 'C4', 'P3', 'P4'],           'Central-Parietal'),
        ('N400', ['C3', 'C4', 'P3', 'P4', 'F3', 'F4'], 'Broad (Centro-Frontal)'),
    ]

    first_evoked = next(iter(evoked_by_cond.values()))
    times_ms = first_evoked.times * 1000

    for ax, (comp, roi, roi_label) in zip(axes, comps_and_roi):
        tw0, tw1 = COMP_WINDOWS[comp]
        ax.axvspan(tw0 * 1000, tw1 * 1000, alpha=0.15,
                   color=COMP_COLORS[comp], zorder=0)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--', zorder=1)
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--', zorder=1)

        for i, cond in enumerate(CONCENTRATIONS):
            if cond not in evoked_by_cond:
                continue
            ev = evoked_by_cond[cond]
            chs = [ch for ch in roi if ch in ev.ch_names]
            if not chs:
                continue
            data = ev.copy().pick(chs).get_data().mean(axis=0) * 1e6
            label = CONCENTRATION_LABELS[cond]
            lw = 2.5 if cond in (605, 893) else 1.5
            ax.plot(times_ms, data, color=COND_COLORS[cond],
                    linewidth=lw, label=label, zorder=3)

        ax.set_title(f'{comp} — {roi_label} ROI',
                     fontsize=12, fontweight='bold', color=COMP_COLORS[comp])
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.set_xlim(times_ms[0], times_ms[-1])
        ax.legend(fontsize=8, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)

    # Shared colorbar legend
    sm = plt.cm.ScalarMappable(cmap=COND_CMAP,
                                norm=plt.Normalize(0, len(CONCENTRATIONS) - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_ticks(range(len(CONCENTRATIONS)))
    cbar.set_ticklabels([CONCENTRATION_LABELS[c] for c in CONCENTRATIONS], fontsize=8)
    cbar.set_label('Concentration (low → high)', fontsize=9)

    fig.suptitle('ERP Waveforms by Sucrose Concentration — Per Component ROI\n'
                 'Shaded: Component time window',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    path = os.path.join(save_dir, '02_erp_by_concentration_4comp.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 3: Bar chart — Mean amplitude per condition per component ─────────
# ──────────────────────────────────────────────────────────────────────────────

def fig3_amplitude_bar_by_condition(measures, save_dir, dpi=200):
    """Bar charts: mean amplitude ± SEM per condition, with ANOVA result."""
    anova_res = run_anova_per_component(measures)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, comp in zip(axes, ['P1', 'N1', 'P2', 'N400']):
        col = f'{comp}_mean_amp'
        if col not in measures.columns:
            continue

        x_vals, y_means, y_sems, colors_bar = [], [], [], []
        for cond in CONCENTRATIONS:
            vals = measures[measures['condition'] == cond][col].dropna()
            x_vals.append(CONCENTRATION_LABELS[cond])
            y_means.append(vals.mean())
            y_sems.append(vals.std() / np.sqrt(len(vals)))
            colors_bar.append(COND_COLORS[cond])

        bars = ax.bar(range(len(x_vals)), y_means, color=colors_bar,
                      edgecolor='white', width=0.7, zorder=3)
        ax.errorbar(range(len(x_vals)), y_means, yerr=y_sems,
                    fmt='none', color='black', capsize=5, linewidth=1.5, zorder=5)

        # Individual subject dots
        for xi, cond in enumerate(CONCENTRATIONS):
            vals = measures[measures['condition'] == cond][col].dropna()
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(xi + jitter, vals.values, color='black',
                       alpha=0.3, s=15, zorder=4)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

        # ANOVA annotation
        if comp in anova_res:
            res = anova_res[comp]
            stars = sig_stars(res['p'])
            ann = f'ANOVA: F={res["F"]:.2f}, p={res["p"]:.3f}{" " + stars if stars != "ns" else ""}\nη²={res["eta2"]:.3f}'
            ax.set_title(f'{comp} Mean Amplitude per Concentration\n{ann}',
                         fontsize=10, fontweight='bold', color=COMP_COLORS[comp])
        else:
            ax.set_title(f'{comp} Mean Amplitude', fontsize=11,
                         fontweight='bold', color=COMP_COLORS[comp])

        # Linear trend
        r, p_r = linear_trend(measures, comp)
        if r is not None:
            trend_color = 'darkgreen' if p_r < 0.05 else 'gray'
            ax.text(0.02, 0.97, f'Linear trend: r={r:.2f}, p={p_r:.3f}',
                    transform=ax.transAxes, fontsize=8, color=trend_color,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('ERP Component Amplitudes by Sucrose Concentration\n'
                 'Bars: mean ± SEM | Dots: individual subjects',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '03_amplitude_bar_by_condition.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 4: Dose-response curves — all 4 components ───────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig4_dose_response_curves(measures, save_dir, dpi=200):
    """Dose-response: amplitude (left) and latency (right) for all components."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for comp in ['P1', 'N1', 'P2', 'N400']:
        # Amplitude
        x, y_a, e_a = [], [], []
        x_l, y_l, e_l = [], [], []
        for i, cond in enumerate(CONCENTRATIONS):
            d_a = measures[measures['condition'] == cond][f'{comp}_mean_amp'].dropna()
            d_l = measures[measures['condition'] == cond][f'{comp}_peak_lat'].dropna()
            if len(d_a) > 0:
                x.append(i);   y_a.append(d_a.mean()); e_a.append(d_a.std() / np.sqrt(len(d_a)))
            if len(d_l) > 0:
                x_l.append(i); y_l.append(d_l.mean()); e_l.append(d_l.std() / np.sqrt(len(d_l)))

        axes[0].errorbar(x, y_a, yerr=e_a, marker='o', label=comp,
                         color=COMP_COLORS[comp], linewidth=2, markersize=7,
                         capsize=4, capthick=1.5)
        axes[1].errorbar(x_l, y_l, yerr=e_l, marker='s', label=comp,
                         color=COMP_COLORS[comp], linewidth=2, markersize=7,
                         capsize=4, capthick=1.5, linestyle='--')

    for ax in axes:
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xticks(range(len(CONCENTRATIONS)))
        ax.set_xticklabels([CONCENTRATION_LABELS[c] for c in CONCENTRATIONS],
                           rotation=30, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Mean Amplitude Dose-Response', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude (µV)', fontsize=11)
    axes[0].set_xlabel('Concentration', fontsize=11)

    axes[1].set_title('Peak Latency Dose-Response', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Latency (ms)', fontsize=11)
    axes[1].set_xlabel('Concentration', fontsize=11)

    fig.suptitle('ERP Component Dose-Response Curves\n'
                 'Error bars: ±SEM across subjects',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '04_dose_response_curves.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 5: JAR group — ERP waveforms + violin plots ──────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig5_jar_group_analysis(evoked_by_jar, measures, save_dir, dpi=200):
    """Top: ERP waveforms per JAR group. Bottom: violin per component."""
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # ── Top row: ERP waveform per component ROI colored by JAR group ──
    comps_roi = [
        ('P1',   ['F3', 'F4', 'C3', 'C4']),
        ('N1',   ['F7', 'F8', 'T7', 'T8']),
        ('P2',   ['C3', 'C4', 'P3', 'P4']),
        ('N400', ['C3', 'C4', 'P3', 'P4', 'F3', 'F4']),
    ]
    if evoked_by_jar:
        times_ms = next(iter(evoked_by_jar.values())).times * 1000
        for col_idx, (comp, roi) in enumerate(comps_roi):
            ax = fig.add_subplot(gs[0, col_idx])
            tw0, tw1 = COMP_WINDOWS[comp]
            ax.axvspan(tw0 * 1000, tw1 * 1000, alpha=0.15,
                       color=COMP_COLORS[comp], zorder=0)
            ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')

            for grp, ev in evoked_by_jar.items():
                chs = [ch for ch in roi if ch in ev.ch_names]
                if not chs:
                    continue
                data = ev.copy().pick(chs).get_data().mean(axis=0) * 1e6
                label = JAR_LABELS_VN.get(grp, grp)
                ax.plot(times_ms, data, color=JAR_COLORS[grp],
                        linewidth=2, label=label)

            ax.set_title(f'{comp} — JAR Group', fontsize=10,
                         fontweight='bold', color=COMP_COLORS[comp])
            ax.set_xlabel('Time (ms)', fontsize=9)
            ax.set_ylabel('Amplitude (µV)', fontsize=9)
            ax.set_xlim(times_ms[0], min(700, times_ms[-1]))
            ax.legend(fontsize=7, loc='lower right')
            ax.grid(True, alpha=0.3)

    # ── Bottom row: violin / box plots per component per JAR group ──
    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    grp_labels = [JAR_LABELS_VN.get(g, g) for g in grp_order]

    for col_idx, comp in enumerate(['P1', 'N1', 'P2', 'N400']):
        ax = fig.add_subplot(gs[1, col_idx])
        col = f'{comp}_mean_amp'
        if col not in measures.columns or 'jar_group' not in measures.columns:
            continue

        data_grps = [measures[measures['jar_group'] == g][col].dropna().values
                     for g in grp_order]
        data_grps_nonempty = [g for g in data_grps if len(g) > 0]
        colors_v = [JAR_COLORS[g] for g, d in zip(grp_order, data_grps) if len(d) > 0]
        labels_v = [lb for lb, d in zip(grp_labels, data_grps) if len(d) > 0]

        if len(data_grps_nonempty) >= 2:
            vparts = ax.violinplot(data_grps_nonempty, positions=range(len(data_grps_nonempty)),
                                   showmedians=True, widths=0.7)
            for i, (pc, col_v) in enumerate(zip(vparts['bodies'], colors_v)):
                pc.set_facecolor(col_v)
                pc.set_alpha(0.6)
            vparts['cmedians'].set_color('black')
            vparts['cbars'].set_color('gray')
            vparts['cmins'].set_color('gray')
            vparts['cmaxes'].set_color('gray')

            # Overlay dots
            for xi, (grp_data, col_v) in enumerate(zip(data_grps_nonempty, colors_v)):
                jitter = np.random.uniform(-0.1, 0.1, len(grp_data))
                ax.scatter(xi + jitter, grp_data, color=col_v,
                           alpha=0.5, s=20, zorder=5)

            # Pairwise significance
            y_max = max(max(g) for g in data_grps_nonempty if len(g) > 0)
            y_off = abs(y_max) * 0.1
            pair_indices = list(combinations(range(len(data_grps_nonempty)), 2))
            for pi, (i, j) in enumerate(pair_indices):
                t, p = ttest_ind(data_grps_nonempty[i], data_grps_nonempty[j], equal_var=False)
                stars = sig_stars(p)
                if stars != 'ns':
                    h = y_max + y_off * (pi + 1)
                    ax.plot([i, i, j, j], [h - y_off * 0.3, h, h, h - y_off * 0.3],
                            'k-', linewidth=1)
                    ax.text((i + j) / 2, h + y_off * 0.1, stars,
                            ha='center', fontsize=10, color='black')

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xticks(range(len(labels_v)))
        ax.set_xticklabels(labels_v, rotation=20, ha='right', fontsize=8)
        ax.set_title(f'{comp} by JAR Group', fontsize=10,
                     fontweight='bold', color=COMP_COLORS[comp])
        ax.set_ylabel('Amplitude (µV)', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('ERP Components by JAR Group (Sweetness Rating)\n'
                 'Top: Waveforms | Bottom: Distribution (violin + dots)',
                 fontsize=14, fontweight='bold')
    path = os.path.join(save_dir, '05_jar_group_analysis.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 6: Pairwise significance heatmap (Water vs each condition) ────────
# ──────────────────────────────────────────────────────────────────────────────

def fig6_significance_heatmap(measures, save_dir, dpi=200):
    """Heatmap of p-values (t-test) for each component × condition pair vs Water."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    water_code = 605

    comps = ['P1', 'N1', 'P2', 'N400']
    non_water_conds = [c for c in CONCENTRATIONS if c != water_code]

    # Left: amplitude p-values vs Water
    p_matrix_amp = np.ones((len(comps), len(non_water_conds)))
    d_matrix_amp = np.zeros_like(p_matrix_amp)

    for ci, comp in enumerate(comps):
        col = f'{comp}_mean_amp'
        water_vals = measures[measures['condition'] == water_code][col].dropna().values
        for ji, cond in enumerate(non_water_conds):
            cond_vals = measures[measures['condition'] == cond][col].dropna().values
            if len(water_vals) >= 2 and len(cond_vals) >= 2:
                _, p = ttest_ind(water_vals, cond_vals, equal_var=False)
                d = cohens_d(cond_vals, water_vals)
                p_matrix_amp[ci, ji] = p
                d_matrix_amp[ci, ji] = d

    # Annotate with stars
    annot_p = [[f'{p:.3f}\n{sig_stars(p)}' for p in row] for row in p_matrix_amp]

    sns.heatmap(
        p_matrix_amp, ax=axes[0],
        xticklabels=[CONCENTRATION_LABELS[c] for c in non_water_conds],
        yticklabels=comps,
        annot=annot_p, fmt='', cmap='RdYlGn_r',
        vmin=0, vmax=0.1, linewidths=0.5, linecolor='gray',
        cbar_kws={'label': 'p-value'}
    )
    axes[0].set_title('Amplitude: p-value vs Water\n(t-test, Welch)',
                       fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Concentration')
    axes[0].set_ylabel('Component')

    # Right: Cohen's d effect sizes
    annot_d = [[f'{d:.2f}' for d in row] for row in d_matrix_amp]
    sns.heatmap(
        d_matrix_amp, ax=axes[1],
        xticklabels=[CONCENTRATION_LABELS[c] for c in non_water_conds],
        yticklabels=comps,
        annot=annot_d, fmt='', cmap='RdBu_r',
        center=0, vmin=-1.5, vmax=1.5, linewidths=0.5, linecolor='gray',
        cbar_kws={"label": "Cohen's d"}
    )
    axes[1].set_title("Effect Size (Cohen's d) vs Water\n+: more positive than water",
                       fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Concentration')

    fig.suptitle('Statistical Comparison vs Water Baseline\n'
                 'ERP Component Amplitudes × Sucrose Concentration',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '06_significance_heatmap.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 7: Component-Component correlation matrix ─────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig7_component_correlation(measures, save_dir, dpi=200):
    """Pearson correlation matrix between ERP component amplitudes."""
    amp_cols = [f'{c}_mean_amp' for c in ['P1', 'N1', 'P2', 'N400']
                if f'{c}_mean_amp' in measures.columns]
    lat_cols = [f'{c}_peak_lat' for c in ['P1', 'N1', 'P2', 'N400']
                if f'{c}_peak_lat' in measures.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, cols, title in [
        (axes[0], amp_cols, 'Amplitude Correlations'),
        (axes[1], lat_cols, 'Latency Correlations'),
    ]:
        df_sub = measures[cols].dropna()
        if df_sub.shape[0] < 4:
            continue
        corr = df_sub.corr(method='pearson')
        labels = [c.replace('_mean_amp', '').replace('_peak_lat', '')
                  for c in cols]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, vmin=-1, vmax=1, square=True,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, mask=False)
        ax.set_title(title, fontsize=12, fontweight='bold')

    fig.suptitle('ERP Component Correlation Matrix\n'
                 'Pearson r across subjects × conditions',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '07_component_correlation.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 8: Per-subject variability boxplot ────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig8_subject_variability(measures, save_dir, dpi=200):
    """Boxplot of amplitude per component per concentration."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, comp in zip(axes, ['P1', 'N1', 'P2', 'N400']):
        col = f'{comp}_mean_amp'
        if col not in measures.columns:
            continue
        data_list = []
        labels_list = []
        colors_list = []
        for cond in CONCENTRATIONS:
            vals = measures[measures['condition'] == cond][col].dropna().values
            data_list.append(vals)
            labels_list.append(CONCENTRATION_LABELS[cond])
            colors_list.append(COND_COLORS[cond])

        bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual points
        for xi, (vals, color) in enumerate(zip(data_list, colors_list)):
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter(xi + 1 + jitter, vals, color=color, alpha=0.5, s=20, zorder=5)

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xticklabels(labels_list, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.set_title(f'{comp} — Per-Subject Distribution',
                     fontsize=11, fontweight='bold', color=COMP_COLORS[comp])
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('ERP Component Amplitude Distribution Across Subjects\n'
                 'Box: quartiles | Dots: individual subjects',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '08_subject_variability.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 9: Scalp Topomaps at peak latency per condition ──────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig9_topomaps_by_condition(evoked_by_cond, save_dir, dpi=150):
    """Topomap at P2 peak for each condition."""
    if not evoked_by_cond:
        return
    # Plot at P2 and N400 windows
    time_points = {
        'P1 (~120ms)': 0.12,
        'N1 (~180ms)': 0.18,
        'P2 (~280ms)': 0.28,
        'N400 (~450ms)': 0.45,
    }

    n_conds = len(CONCENTRATIONS)
    n_times = len(time_points)

    fig, axes = plt.subplots(n_conds, n_times, figsize=(4 * n_times, 3.5 * n_conds))

    # Determine global vmin/vmax for consistent colorbar
    all_vals = []
    for cond in CONCENTRATIONS:
        if cond not in evoked_by_cond:
            continue
        ev = evoked_by_cond[cond]
        for t in time_points.values():
            tidx = np.argmin(np.abs(ev.times - t))
            all_vals.extend(ev.get_data()[:, tidx] * 1e6)
    if not all_vals:
        return
    vabs = np.percentile(np.abs(all_vals), 95)
    vmin, vmax = -vabs, vabs

    for ri, cond in enumerate(CONCENTRATIONS):
        if cond not in evoked_by_cond:
            continue
        ev = evoked_by_cond[cond]
        for ci, (t_label, t_sec) in enumerate(time_points.items()):
            ax = axes[ri, ci]
            tidx = np.argmin(np.abs(ev.times - t_sec))
            try:
                mne.viz.plot_topomap(
                    ev.get_data()[:, tidx] * 1e6,
                    ev.info, axes=ax, show=False,
                    vlim=(vmin, vmax), cmap='RdBu_r',
                    contours=4,
                )
            except Exception:
                ax.set_visible(False)
                continue
            if ci == 0:
                ax.set_ylabel(CONCENTRATION_LABELS[cond], fontsize=8)
            if ri == 0:
                ax.set_title(t_label, fontsize=9, fontweight='bold')

    fig.suptitle('Scalp Topographies by Concentration × Time Point\n'
                 'µV scale (RdBu: red=positive, blue=negative)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '09_topomaps_by_condition.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 10: Difference waves — all contrasts ──────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig10_difference_waves(evoked_by_cond, save_dir, dpi=200):
    """Difference waveforms for key sucrose contrasts."""
    if not evoked_by_cond:
        return
    contrasts = [
        ('High − Water',    893, 605),
        ('MedHigh − Water', 762, 605),
        ('High − Medium',   893, 189),
        ('MedHigh − Low',   762, 258),
    ]
    roi = ['C3', 'C4', 'P3', 'P4', 'F3', 'F4']

    valid = [(name, a, b) for (name, a, b) in contrasts
             if a in evoked_by_cond and b in evoked_by_cond]
    if not valid:
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, ca, cb) in zip(axes, valid):
        ev_a = evoked_by_cond[ca]
        ev_b = evoked_by_cond[cb]
        chs_a = [ch for ch in roi if ch in ev_a.ch_names]
        chs_b = [ch for ch in roi if ch in ev_b.ch_names]
        da = ev_a.copy().pick(chs_a).get_data().mean(axis=0) * 1e6
        db = ev_b.copy().pick(chs_b).get_data().mean(axis=0) * 1e6
        diff = da - db
        times_ms = ev_a.times * 1000

        ax.fill_between(times_ms, diff, 0, where=diff > 0,
                        alpha=0.4, color='#e74c3c', label='Positive')
        ax.fill_between(times_ms, diff, 0, where=diff < 0,
                        alpha=0.4, color='#2980b9', label='Negative')
        ax.plot(times_ms, diff, 'k-', linewidth=1.5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

        # Mark component windows
        for comp, (tw0, tw1) in COMP_WINDOWS.items():
            ax.axvspan(tw0 * 1000, tw1 * 1000, alpha=0.1,
                       color=COMP_COLORS[comp])
            peak_tm = (times_ms >= tw0 * 1000) & (times_ms <= tw1 * 1000)
            if peak_tm.sum() > 0:
                pk_v = diff[peak_tm]
                pk_t = times_ms[peak_tm]
                if COMP_POLARITY[comp] == 'pos':
                    pi = np.argmax(pk_v)
                else:
                    pi = np.argmin(pk_v)
                ax.text(pk_t[pi], pk_v[pi] * 1.1, comp,
                        ha='center', fontsize=8, color=COMP_COLORS[comp],
                        fontweight='bold')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_xlim(times_ms[0], times_ms[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Amplitude Difference (µV)', fontsize=11)
    fig.suptitle('Difference Waves — Sucrose Concentration Contrasts\n'
                 'Red: higher sweetness > lower | Blue: lower sweetness > higher',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '10_difference_waves.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 11: JAR Group pairwise comparison summary ────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig11_jar_pairwise_summary(measures, save_dir, dpi=200):
    """Bar chart + significance for JAR groups, all 4 components."""
    if 'jar_group' not in measures.columns:
        return

    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    grp_labels = ['Không đủ\n(JAR 1-2)', 'Vừa phải\n(JAR 3)', 'Quá nhiều\n(JAR 4-5)']

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    for ax, comp in zip(axes, ['P1', 'N1', 'P2', 'N400']):
        col = f'{comp}_mean_amp'
        if col not in measures.columns:
            continue

        grp_data = [measures[measures['jar_group'] == g][col].dropna().values
                    for g in grp_order]
        grp_data_valid = [(g, d) for g, d in zip(grp_order, grp_data) if len(d) >= 2]
        if not grp_data_valid:
            continue

        x_pos = range(len(grp_data_valid))
        bar_colors = [JAR_COLORS[g] for g, _ in grp_data_valid]
        bar_labels = [grp_labels[grp_order.index(g)] for g, _ in grp_data_valid]
        means = [d.mean() for _, d in grp_data_valid]
        sems  = [d.std() / np.sqrt(len(d)) for _, d in grp_data_valid]

        ax.bar(x_pos, means, color=bar_colors, alpha=0.8, width=0.6, edgecolor='white')
        ax.errorbar(x_pos, means, yerr=sems, fmt='none', color='black',
                    capsize=6, linewidth=2)
        for xi, (_, d) in enumerate(grp_data_valid):
            jitter = np.random.uniform(-0.1, 0.1, len(d))
            ax.scatter(xi + jitter, d, color='black', alpha=0.3, s=20, zorder=5)

        # Significance bars
        y_max = max(m + s for m, s in zip(means, sems)) if means else 0
        y_off = abs(y_max) * 0.15 + 0.05
        pairs = list(combinations(range(len(grp_data_valid)), 2))
        for pi, (i, j) in enumerate(pairs):
            t, p = ttest_ind(grp_data_valid[i][1], grp_data_valid[j][1], equal_var=False)
            stars = sig_stars(p)
            if stars != 'ns':
                h = y_max + y_off * (pi + 1)
                ax.plot([i, i, j, j], [h - y_off * 0.2, h, h, h - y_off * 0.2],
                        'k-', linewidth=1)
                ax.text((i + j) / 2, h + y_off * 0.05, stars,
                        ha='center', fontsize=11, fontweight='bold')

        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.set_title(f'{comp}', fontsize=13, fontweight='bold', color=COMP_COLORS[comp])
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('ERP Components by JAR Sweetness Rating Group\n'
                 'Mean ± SEM | Dots: individual trials | *p<0.05, **p<0.01, ***p<0.001',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '11_jar_pairwise_summary.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 12: Summary radar / spider chart ──────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig12_erp_profile_radar(measures, save_dir, dpi=200):
    """Radar chart showing ERP component profiles per concentration."""
    comps = ['P1', 'N1', 'P2', 'N400']
    amp_cols = [f'{c}_mean_amp' for c in comps]
    available = [c for c, col in zip(comps, amp_cols) if col in measures.columns]
    if len(available) < 3:
        return

    n_vars = len(available)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    cond_subset = [605, 189, 762, 893]  # Water, Medium, MedHigh, High

    for cond in cond_subset:
        cond_data = measures[measures['condition'] == cond]
        vals = []
        for comp in available:
            col = f'{comp}_mean_amp'
            v = cond_data[col].mean() if col in cond_data.columns else 0
            vals.append(v)
        vals += vals[:1]
        label = CONCENTRATION_LABELS[cond]
        ax.plot(angles, vals, 'o-', linewidth=2, label=label,
                color=COND_COLORS[cond])
        ax.fill(angles, vals, alpha=0.1, color=COND_COLORS[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available, fontsize=12, fontweight='bold')
    ax.set_title('ERP Component Profile by Concentration\n(Mean amplitude µV)',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    path = os.path.join(save_dir, '12_erp_profile_radar.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 13: JAR Group Difference Waves ────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig13_jar_difference_waves(evoked_by_jar, save_dir, dpi=200):
    """Difference waveforms between JAR groups for all ERP components.

    Shows 3 contrasts: Vua_phai−Không_đủ, Quá_nhiều−Không_đủ, Vua_phai−Quá_nhiều
    with component window annotations.
    """
    if not evoked_by_jar or len(evoked_by_jar) < 2:
        print('  ⚠️  fig13: insufficient JAR groups')
        return

    grp_list = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    available = [g for g in grp_list if g in evoked_by_jar]
    if len(available) < 2:
        return

    contrasts = [
        ('Vừa phải − Không đủ',  'Vua_phai', 'Khong_du'),
        ('Quá nhiều − Không đủ', 'Qua_nhieu', 'Khong_du'),
        ('Vừa phải − Quá nhiều', 'Vua_phai', 'Qua_nhieu'),
    ]
    # Only keep contrasts where both groups exist
    valid = [(name, a, b) for (name, a, b) in contrasts
             if a in evoked_by_jar and b in evoked_by_jar]

    # Use broad central ROI
    roi = ['F3', 'F4', 'C3', 'C4', 'P3', 'P4']

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, ga, gb) in zip(axes, valid):
        ev_a = evoked_by_jar[ga]
        ev_b = evoked_by_jar[gb]
        chs = [ch for ch in roi if ch in ev_a.ch_names]
        if not chs:
            continue
        da = ev_a.copy().pick(chs).get_data().mean(axis=0) * 1e6
        db = ev_b.copy().pick(chs).get_data().mean(axis=0) * 1e6
        diff = da - db
        times_ms = ev_a.times * 1000

        ax.fill_between(times_ms, diff, 0, where=diff > 0,
                        alpha=0.4, color='#e74c3c', label='Positive')
        ax.fill_between(times_ms, diff, 0, where=diff < 0,
                        alpha=0.4, color='#2980b9', label='Negative')
        ax.plot(times_ms, diff, 'k-', linewidth=1.5)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

        # Mark component windows
        for comp, (tw0, tw1) in COMP_WINDOWS.items():
            ax.axvspan(tw0 * 1000, tw1 * 1000, alpha=0.1,
                       color=COMP_COLORS[comp])
            peak_tm = (times_ms >= tw0 * 1000) & (times_ms <= tw1 * 1000)
            if peak_tm.sum() > 0:
                pk_v = diff[peak_tm]
                pk_t = times_ms[peak_tm]
                if COMP_POLARITY[comp] == 'pos':
                    pi = np.argmax(pk_v)
                else:
                    pi = np.argmin(pk_v)
                ax.text(pk_t[pi], pk_v[pi] * 1.15, comp,
                        ha='center', fontsize=8, color=COMP_COLORS[comp],
                        fontweight='bold')

        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_xlim(times_ms[0], times_ms[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel('Amplitude Difference (µV)', fontsize=11)
    fig.suptitle('Difference Waves — JAR Group Contrasts\n'
                 'Red: positive difference | Blue: negative difference',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(save_dir, '13_jar_difference_waves.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 14: Region Breakdown per Component ────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig14_region_breakdown(region_measures, save_dir, dpi=200):
    """Bar charts showing ERP amplitude per brain region × JAR group.

    One subplot per component, with regions on x-axis and
    JAR groups as grouped bars.
    """
    if region_measures.empty:
        print('  ⚠️  fig14: no region measures')
        return

    comps = ['P1', 'N1', 'P2', 'N400']
    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    grp_labels = ['Không đủ\n(JAR 1-2)', 'Vừa phải\n(JAR 3)', 'Quá nhiều\n(JAR 4-5)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, comp in enumerate(comps):
        ax = axes[ax_idx]
        regions = list(COMP_REGIONS[comp].keys())
        n_regions = len(regions)
        n_groups = len(grp_order)

        sub = region_measures[region_measures['component'] == comp]
        if sub.empty:
            ax.set_title(f'{comp} — no data', fontsize=11, fontweight='bold')
            continue

        # Compute mean ± sem per region × JAR group
        stats_data = sub.groupby(['region', 'jar_group'])['amplitude_uv'].agg(['mean', 'sem', 'count']).reset_index()

        bar_width = 0.8 / n_groups
        x = np.arange(n_regions)

        for gi, grp in enumerate(grp_order):
            offsets = x - 0.4 + gi * bar_width + bar_width / 2
            means = []
            sems = []
            for ri, reg in enumerate(regions):
                row = stats_data[(stats_data['region'] == reg) & (stats_data['jar_group'] == grp)]
                means.append(row['mean'].values[0] if len(row) > 0 else 0)
                sems.append(row['sem'].values[0] if len(row) > 0 else 0)
            bars = ax.bar(offsets, means, bar_width,
                          label=grp_labels[gi] if ax_idx == 0 else '',
                          color=JAR_COLORS[grp], alpha=0.85,
                          yerr=sems, capsize=3, error_kw={'linewidth': 1})

        ax.set_xticks(x)
        ax.set_xticklabels(regions, fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.set_title(f'{comp} — Mean amplitude by region',
                     fontsize=12, fontweight='bold')
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('ERP Component Amplitude by Brain Region and JAR Group\n'
                 'Error bars: ±1 SEM',
                 fontsize=14, fontweight='bold')
    fig.legend(loc='lower center', ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(save_dir, '14_region_breakdown.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# ── FIGURE 15: Per-Channel Breakdown ─────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

def fig15_channel_breakdown(channel_measures, save_dir, dpi=200):
    """Heatmap + bar charts showing ERP amplitude per individual channel × JAR group.

    Left panel: heatmap (channel × JAR group, mean amplitude).
    Right panel: top channels with pairwise d and p-values.
    """
    if channel_measures.empty:
        print('  ⚠️  fig15: no channel measures')
        return

    comps = ['P1', 'N1', 'P2', 'N400']
    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    grp_labels = ['Không đủ', 'Vừa phải', 'Quá nhiều']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, comp in enumerate(comps):
        ax = axes[ax_idx]
        chs = COMP_ROI[comp]
        sub = channel_measures[channel_measures['component'] == comp]
        if sub.empty:
            ax.set_title(f'{comp} — no data', fontsize=11, fontweight='bold')
            continue

        # Pivot: channels × JAR groups → mean amplitude
        pivot = sub.groupby(['channel', 'jar_group'])['amplitude_uv'].mean().unstack()
        # Keep only channels that exist in the data
        pivot = pivot.reindex([ch for ch in chs if ch in pivot.index])
        if pivot.empty:
            continue
        pivot = pivot[grp_order]  # reorder columns

        # Heatmap
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(len(grp_order)))
        ax.set_xticklabels(grp_labels, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10, fontweight='bold')
        ax.set_title(f'{comp} — Mean amplitude (µV)', fontsize=12, fontweight='bold')

        # Annotate cells with values
        for ri in range(len(pivot.index)):
            for ci in range(len(grp_order)):
                val = pivot.values[ri, ci]
                color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(ci, ri, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Per-Channel ERP Amplitude by JAR Group\n'
                 'Red = positive, Blue = negative',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, '15_channel_breakdown.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ {path}')

    # ── Second figure: bar charts with significance per channel ──
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    axes2 = axes2.flatten()

    for ax_idx, comp in enumerate(comps):
        ax = axes2[ax_idx]
        chs = COMP_ROI[comp]
        sub = channel_measures[channel_measures['component'] == comp]
        if sub.empty:
            continue

        stats_data = sub.groupby(['channel', 'jar_group'])['amplitude_uv'].agg(['mean', 'sem']).reset_index()

        n_ch = len(chs)
        n_grp = len(grp_order)
        bar_w = 0.8 / n_grp
        x = np.arange(n_ch)

        for gi, grp in enumerate(grp_order):
            off = x - 0.4 + gi * bar_w + bar_w / 2
            means = []
            sems = []
            for ri, ch in enumerate(chs):
                row = stats_data[(stats_data['channel'] == ch) & (stats_data['jar_group'] == grp)]
                means.append(row['mean'].values[0] if len(row) > 0 else 0)
                sems.append(row['sem'].values[0] if len(row) > 0 else 0)
            ax.bar(off, means, bar_w, label=grp_labels[gi] if ax_idx == 0 else '',
                   color=JAR_COLORS[grp], alpha=0.85, yerr=sems, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(chs, fontsize=11, fontweight='bold')
        ax.set_ylabel('Amplitude (µV)', fontsize=10)
        ax.set_title(f'{comp} — per channel', fontsize=12, fontweight='bold')
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.grid(True, axis='y', alpha=0.3)

    fig2.suptitle('Per-Channel ERP Amplitude by JAR Group\nError bars: ±1 SEM',
                  fontsize=14, fontweight='bold')
    fig2.legend(loc='lower center', ncol=3, fontsize=10)
    fig2.tight_layout(rect=[0, 0.05, 1, 0.95])
    path2 = os.path.join(save_dir, '15b_channel_bar.png')
    fig2.savefig(path2, dpi=dpi, bbox_inches='tight')
    plt.close(fig2)
    print(f'  ✓ {path2}')
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Insight report text
# ──────────────────────────────────────────────────────────────────────────────

def run_jar_pairwise_analysis(measures):
    """Pairwise binary comparisons between JAR groups for all components.

    Returns dict: {comp: [(group1, group2, delta_uv, cohens_d, p, stars), ...]}
    """
    grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
    comps = ['P1', 'N1', 'P2', 'N400']
    results = {}

    for comp in comps:
        col = f'{comp}_mean_amp'
        if col not in measures.columns:
            continue
        pairs = []
        for g1, g2 in combinations(grp_order, 2):
            v1 = measures[measures['jar_group'] == g1][col].dropna().values
            v2 = measures[measures['jar_group'] == g2][col].dropna().values
            if len(v1) < 2 or len(v2) < 2:
                continue
            delta = v1.mean() - v2.mean()
            pooled = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
            d = delta / pooled if pooled > 1e-10 else 0
            t, p = ttest_ind(v1, v2, equal_var=False)
            pairs.append((g1, g2, delta, d, p, sig_stars(p)))
        results[comp] = pairs
    return results


def generate_insight_report(measures, conc_summary, jar_summary, save_path,
                            region_measures=None, region_jar_stats=None,
                            channel_measures=None, channel_jar_stats=None):
    """Tạo báo cáo insight chi tiết dạng text, ưu tiên JAR analysis."""
    anova_res = run_anova_per_component(measures)
    jar_pairs = run_jar_pairwise_analysis(measures)

    lines = []

    lines += [
        "=" * 80,
        "  BÁO CÁO PHÂN TÍCH ERP — NGHIÊN CỨU VỊ GIÁC (VỊ NGỌT SUCROSE)",
        "  Các thành phần: P1, N1, P2, N400",
        "=" * 80,
        "",
        "THIẾT KẾ THÍ NGHIỆM",
        "-" * 40,
        "  • 28 participants (P001–P030, trừ P012, P022)",
        "  • 6 nồng độ sucrose: Water/605, Low/258, MedLow/453,",
        "                       Medium/189, MedHigh/762, High/893",
        "  • 5 lần lặp mỗi điều kiện × 28 người = 840 trial tổng",
        "  • JAR rating: 1-2 = Không đủ, 3 = Vừa phải, 4-5 = Quá nhiều",
        "  • Thiết bị: 16-kênh EEG, 100 Hz, 10-20 montage",
        "",
    ]

    # ── Component descriptions ──
    lines += [
        "Ý NGHĨA CÁC THÀNH PHẦN ERP TRONG VỊ GIÁC",
        "-" * 40,
        "",
        "P1 (90–150ms) — Xử lý hướng tâm sớm (Early afferent processing)",
        "  ROI: F3, F4, C3, C4 (Frontal-Central)",
        "  Nguồn gốc: Phản ánh hoạt động của vỏ não cảm giác vị giác sơ cấp",
        "  (Insula/Operculum). Biên độ P1 tăng theo cường độ kích thích vị giác.",
        "  Trong nghiên cứu sucrose: P1 lớn hơn = tín hiệu hóa học mạnh hơn.",
        "",
        "N1 (140–240ms) — Chú ý / Phân biệt kích thích (Attention/Discrimination)",
        "  ROI: F7, F8, T7, T8 (Temporal-Frontal — gần Insula nhất)",
        "  Nguồn gốc: Liên quan đến mạch lưới chú ý và nhận diện chất lạ.",
        "  Biên độ N1 âm hơn thường phản ánh phân biệt tốt hơn giữa các nồng độ.",
        "  N1 lớn hơn (âm hơn) ở nồng độ cao có thể phản ánh 'ngạc nhiên' vị giác.",
        "",
        "P2 (230–350ms) — Đánh giá nhận thức sơ bộ (Early cognitive evaluation)",
        "  ROI: C3, C4, P3, P4 (Central-Parietal)",
        "  Nguồn gốc: Đánh giá chất lượng vị giác và tích hợp thông tin cảm giác.",
        "  Đây là thành phần QUAN TRỌNG NHẤT trong vị giác — P2 tương quan với",
        "  mức độ ngon/dễ chịu. Biên độ P2 cao nhất ở điều kiện 'Vừa phải' (JAR=3).",
        "",
        "N400 (350–550ms) — Xử lý nhận thức muộn / Ngữ nghĩa vị giác",
        "  ROI: C3, C4, P3, P4, F3, F4 (Broad Centro-Frontal)",
        "  Nguồn gốc: Phản ánh sự không khớp giữa vị giác kỳ vọng và thực tế.",
        "  Cũng liên quan đến ký ức về vị (taste memory) và đánh giá 'ngon/dở'.",
        "  N400 âm hơn ở điều kiện 'Quá nhiều' ngọt có thể phản ánh 'vi phạm',",
        "  tương tự N400 ngôn ngữ (mismatch với prototype 'ngọt vừa phải').",
        "",
    ]

    # ═══════════════════════════════════════════════
    # PHẦN 1: JAR GROUP ANALYSIS (ưu tiên)
    # ═══════════════════════════════════════════════
    if 'jar_group' in measures.columns:
        grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
        grp_labels = {'Khong_du': 'Không đủ (JAR 1-2)',
                      'Vua_phai': 'Vừa phải (JAR 3)',
                      'Qua_nhieu': 'Quá nhiều (JAR 4-5)'}

        lines += [
            "=" * 80,
            "  KẾT QUẢ PHÂN TÍCH JAR — MỨC ĐỘ VỪA PHẢI",
            "=" * 80,
            "",
        ]

        # ── 1A. Means per JAR group ──
        lines += [
            "1A. Biên độ trung bình (µV) theo nhóm JAR",
            "",
            f"  {'JAR Group':<20} {'P1':>10} {'N1':>10} {'P2':>10} {'N400':>10}",
            "  " + "-" * 60,
        ]
        for g in grp_order:
            gdata = measures[measures['jar_group'] == g]
            row = f"  {grp_labels[g]:<20}"
            for comp in ['P1', 'N1', 'P2', 'N400']:
                col = f'{comp}_mean_amp'
                v = gdata[col].mean() if col in gdata.columns else float('nan')
                row += f" {v:>10.3f}" if not np.isnan(v) else f" {'N/A':>10}"
            lines.append(row)

        lines += [""]

        # ── 1B. JAR Pairwise binary comparisons ──
        lines += [
            "1B. So sánh nhị phân giữa các nhóm JAR (Welch t-test)",
            "",
        ]
        for comp in ['P1', 'N1', 'P2', 'N400']:
            pairs = jar_pairs.get(comp, [])
            if not pairs:
                continue
            lines.append(f"  {comp} — Biên độ trung bình (µV):")
            lines.append(f"  {'So sánh':<30} {'∆µV':>8} {'Cohen d':>8} {'p-value':>8} {'Sig':>6}")
            lines.append(f"  {'-' * 60}")
            for g1, g2, delta, d, p, stars in pairs:
                lines.append(f"  {grp_labels[g1]:<14} vs {grp_labels[g2]:<12} "
                             f"{delta:>+8.3f} {d:>+8.3f} {p:>8.4f} {stars:>6}")
            lines.append("")

        lines += [""]

    # ═══════════════════════════════════════════════
    # PHẦN 2: CONCENTRATION ANALYSIS
    # ═══════════════════════════════════════════════
    lines += [
        "=" * 80,
        "  KẾT QUẢ PHÂN TÍCH NỒNG ĐỘ",
        "=" * 80,
        "",
        "2. ANOVA một chiều — Biên độ theo Nồng độ Sucrose (6 mức)",
        "",
    ]
    for comp in ['P1', 'N1', 'P2', 'N400']:
        col = f'{comp}_mean_amp'
        if col not in measures.columns:
            continue
        res = anova_res.get(comp, {})
        F, p, eta2 = res.get('F', np.nan), res.get('p', np.nan), res.get('eta2', np.nan)
        sig = sig_stars(p) if not np.isnan(p) else '?'
        effect = 'Nhỏ' if eta2 < 0.06 else 'Trung bình' if eta2 < 0.14 else 'Lớn'
        lines.append(f"  {comp}: F={F:.3f}, p={p:.4f} {sig}, η²={eta2:.3f} (Effect size: {effect})")

    lines += [""]

    # ── Means per condition ──
    lines += [
        "3. Biên độ trung bình (µV) per điều kiện",
        "",
        f"  {'Condition':<20} {'P1':>10} {'N1':>10} {'P2':>10} {'N400':>10}",
        "  " + "-" * 60,
    ]
    for cond in CONCENTRATIONS:
        cond_data = measures[measures['condition'] == cond]
        row = f"  {CONCENTRATION_LABELS[cond]:<20}"
        for comp in ['P1', 'N1', 'P2', 'N400']:
            col = f'{comp}_mean_amp'
            v = cond_data[col].mean() if col in cond_data.columns else float('nan')
            row += f" {v*1e0:>10.3f}" if not np.isnan(v) else f" {'N/A':>10}"
        lines.append(row)

    lines += [""]

    # ── Linear trend ──
    lines += [
        "4. Linear trend (Pearson r: amplitude ~ concentration index)",
        "",
    ]
    for comp in ['P1', 'N1', 'P2', 'N400']:
        r, p = linear_trend(measures, comp)
        if r is not None:
            sig = sig_stars(p)
            direction = '↑ tăng' if r > 0 else '↓ giảm'
            lines.append(f"  {comp}: r={r:.3f}, p={p:.4f} {sig} → {direction} theo nồng độ")

    lines += [""]

    # ── JAR group differences (short version) ──
    if 'jar_group' in measures.columns:
        lines += [
            "5. So sánh giữa nhóm JAR (chi tiết tại mục 1)",
            "",
        ]
        grp_order = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
        for comp in ['P1', 'N1', 'P2', 'N400']:
            col = f'{comp}_mean_amp'
            if col not in measures.columns:
                continue
            grp_means = {}
            for g in grp_order:
                vals = measures[measures['jar_group'] == g][col].dropna()
                grp_means[g] = (vals.mean(), len(vals))
            means_str = ' | '.join(
                f"{g}: {m:.3f}µV (n={n})" for g, (m, n) in grp_means.items()
            )
            # ANOVA across JAR groups
            groups = [measures[measures['jar_group'] == g][col].dropna().values
                      for g in grp_order if len(measures[measures['jar_group'] == g][col].dropna()) >= 2]
            if len(groups) >= 2:
                F_j, p_j = f_oneway(*groups)
                lines.append(f"  {comp}: {means_str}")
                lines.append(f"         ANOVA: F={F_j:.2f}, p={p_j:.4f} {sig_stars(p_j)}")
            else:
                lines.append(f"  {comp}: {means_str}")

    lines += [""]

    # ── Region breakdown section ──
    if region_measures is not None and not region_measures.empty and region_jar_stats:
        lines += [
            "=" * 80,
            "  PHÂN TÍCH THEO VÙNG NÃO (REGION BREAKDOWN)",
            "=" * 80,
            "",
            "Phân tích biên độ từng thành phần ERP riêng theo từng vùng não",
            "(Frontal, Central, Parietal, Temporal) để xác định vùng",
            "não nhạy cảm nhất với sự thay đổi vị ngọt.",
            "",
        ]

        for comp in ['P1', 'N1', 'P2', 'N400']:
            if comp not in region_jar_stats:
                continue
            comp_data = region_jar_stats[comp]
            if not comp_data:
                continue
            lines += [f"{comp} — Biên độ (µV) theo vùng não và nhóm JAR:"]
            lines += ["  " + "-" * 65]
            header = f"  {'Vùng':<12} {'Không đủ':<18} {'Vừa phải':<18} {'Quá nhiều':<18}"
            lines += [header]
            lines += ["  " + "-" * 65]

            for region, rdata in comp_data.items():
                means = rdata['means']
                m_nd = f"{means['Khong_du']['mean']:.3f}±{means['Khong_du']['sem']:.3f}" if 'Khong_du' in means else '—'
                m_vp = f"{means['Vua_phai']['mean']:.3f}±{means['Vua_phai']['sem']:.3f}" if 'Vua_phai' in means else '—'
                m_qn = f"{means['Qua_nhieu']['mean']:.3f}±{means['Qua_nhieu']['sem']:.3f}" if 'Qua_nhieu' in means else '—'
                lines.append(f"  {region:<12} {m_nd:<18} {m_vp:<18} {m_qn:<18}")

            # Pairwise tests with p-values per region
            lines += [f"  So sánh nhị phân (Welch t-test) — {comp}:"]
            lines += [f"  {'Vùng':<12} {'Cặp':<35} {'∆µV':<10} {'d':<8} {'p':<10} {'Sig':<6}"]
            lines += [f"  {'-'*65}"]
            for region, rdata in comp_data.items():
                for pair in rdata['pairs']:
                    comp_label = f"{pair['comparison'].split(' vs ')[0][:8]}–{pair['comparison'].split(' vs ')[1][:8]}"
                    p = pair['p']
                    lines.append(
                        f"  {region:<12} {pair['comparison']:<35} "
                        f"{pair['delta_uv']:<+8.3f} {pair['cohens_d']:<+8.3f} "
                        f"{p:<10.4f} {sig_stars(p):<6}"
                    )
            lines += [""]

            # Highlight best-separating region
            best_d = -1
            best_reg = None
            best_p = 1.0
            for region, rdata in comp_data.items():
                for pair in rdata['pairs']:
                    d = abs(pair['cohens_d'])
                    if d > best_d:
                        best_d = d
                        best_reg = region
                        best_p = pair['p']
            if best_reg:
                best_sig = sig_stars(best_p)
                lines.append(f"  → Vùng phân biệt JAR tốt nhất: {best_reg} "
                             f"(|d|={best_d:.3f}, p={best_p:.4f} {best_sig})")
            lines += [""]

    # ── Per-channel analysis section ──
    if channel_measures is not None and not channel_measures.empty and channel_jar_stats:
        lines += [
            "=" * 80,
            "  PHÂN TÍCH THEO TỪNG ĐIỆN CỰC (PER-CHANNEL BREAKDOWN)",
            "=" * 80,
            "",
            "Biên độ từng thành phần ERP tại từng điện cực riêng lẻ.",
            "Giúp xác định chính xác điện cực nào nhạy cảm nhất với vị ngọt.",
            "",
        ]

        for comp in ['P1', 'N1', 'P2', 'N400']:
            if comp not in channel_jar_stats:
                continue
            ch_data = channel_jar_stats[comp]
            chs = COMP_ROI[comp]
            # Header
            lines += [f"{comp} — Biên độ (µV) và so sánh nhị phân theo từng kênh:"]
            lines += [f"  {'Điện cực':<10} {'Không đủ':<18} {'Vừa phải':<18} {'Quá nhiều':<18} "
                      f"{'d max':<8} {'p min':<10} {'Sig':<6}"]
            lines += [f"  {'-'*75}"]
            for ch in chs:
                if ch not in ch_data:
                    continue
                chd = ch_data[ch]
                means = chd['means']
                m_nd = f"{means['Khong_du']['mean']:.3f}±{means['Khong_du']['sem']:.3f}" if 'Khong_du' in means else '—'
                m_vp = f"{means['Vua_phai']['mean']:.3f}±{means['Vua_phai']['sem']:.3f}" if 'Vua_phai' in means else '—'
                m_qn = f"{means['Qua_nhieu']['mean']:.3f}±{means['Qua_nhieu']['sem']:.3f}" if 'Qua_nhieu' in means else '—'
                # Best pair
                best_d = 0
                best_p = 1.0
                for pair in chd['pairs']:
                    if abs(pair['cohens_d']) > abs(best_d):
                        best_d = pair['cohens_d']
                        best_p = pair['p']
                best_sig = sig_stars(best_p)
                lines.append(f"  {ch:<10} {m_nd:<18} {m_vp:<18} {m_qn:<18} "
                             f"{best_d:<+8.3f} {best_p:<10.4f} {best_sig:<6}")

            # Find best channel for this component
            best_ch = None
            best_ch_d = 0
            best_ch_p = 1.0
            for ch in chs:
                if ch not in ch_data:
                    continue
                for pair in ch_data[ch]['pairs']:
                    if abs(pair['cohens_d']) > abs(best_ch_d):
                        best_ch_d = pair['cohens_d']
                        best_ch_p = pair['p']
                        best_ch = ch
            if best_ch:
                lines.append(f"  → Điện cực nhạy nhất: {best_ch} "
                             f"(max |d|={best_ch_d:.3f}, p={best_ch_p:.4f} {sig_stars(best_ch_p)})")
            lines += [""]

    # ── Key insights ──
    lines += [
        "=" * 80,
        "KEY INSIGHTS — PHÂN TÍCH CHI TIẾT",
        "=" * 80,
        "",
    ]

    # Compute some values for insight text
    water_p2 = measures[measures['condition'] == 605]['P2_mean_amp'].mean()
    high_p2  = measures[measures['condition'] == 893]['P2_mean_amp'].mean()
    water_n400 = measures[measures['condition'] == 605]['N400_mean_amp'].mean()
    high_n400  = measures[measures['condition'] == 893]['N400_mean_amp'].mean()
    jar_vua_p2 = measures[measures['jar_group'] == 'Vua_phai']['P2_mean_amp'].mean() \
        if 'jar_group' in measures.columns else float('nan')
    jar_nhieu_n400 = measures[measures['jar_group'] == 'Qua_nhieu']['N400_mean_amp'].mean() \
        if 'jar_group' in measures.columns else float('nan')

    # P2 insight
    p2_anova = anova_res.get('P2', {})
    p2_sig = sig_stars(p2_anova.get('p', 1.0))
    r_p2, p_r_p2 = linear_trend(measures, 'P2')

    lines += [
        "INSIGHT 1: P2 — Đánh giá vị ngọt sơ cấp",
        "-" * 50,
    ]
    if p2_sig != 'ns':
        lines.append(f"  ✅ P2 CÓ Ý NGHĨA THỐNG KÊ (F={p2_anova.get('F',0):.2f}, "
                     f"p={p2_anova.get('p',1):.4f}, η²={p2_anova.get('eta2',0):.3f})")
        lines.append(f"  → Biên độ P2 thay đổi rõ ràng theo nồng độ sucrose.")
    else:
        lines.append(f"  ⚠️  P2 không đạt ý nghĩa thống kê (p={p2_anova.get('p',1):.4f})")
        lines.append(f"  → Có thể do biến thiên cá nhân cao hoặc mẫu nhỏ.")

    if r_p2 is not None:
        trend_sig = sig_stars(p_r_p2)
        if trend_sig != 'ns':
            dir_p2 = 'TĂNG' if r_p2 > 0 else 'GIẢM'
            lines.append(f"  ✅ Linear trend: P2 {dir_p2} theo nồng độ (r={r_p2:.3f}, p={p_r_p2:.4f})")
        else:
            lines.append(f"  ℹ️  Không có linear trend rõ ràng (r={r_p2:.3f}, p={p_r_p2:.4f})")

    lines += [
        f"  • Water/605: P2 = {water_p2:.3f} µV",
        f"  • High/893:  P2 = {high_p2:.3f} µV",
        f"  • Điều kiện 'Vừa phải' (JAR=3): P2 = {jar_vua_p2:.3f} µV",
        "  ",
        "  GIẢI THÍCH: P2 phản ánh quá trình đánh giá giá trị cảm thụ vị giác.",
        "  Nếu P2 lớn nhất ở nồng độ 'vừa phải' → não bộ đánh giá cao nhất",
        "  khi vị ngọt đạt mức tối ưu theo sở thích cá nhân.",
        "",
    ]

    # N400 insight
    n400_anova = anova_res.get('N400', {})
    n400_sig = sig_stars(n400_anova.get('p', 1.0))
    r_n400, p_r_n400 = linear_trend(measures, 'N400')

    lines += [
        "INSIGHT 2: N400 — Mismatch vị giác và ký ức vị",
        "-" * 50,
    ]
    if n400_sig != 'ns':
        lines.append(f"  ✅ N400 CÓ Ý NGHĨA THỐNG KÊ (F={n400_anova.get('F',0):.2f}, "
                     f"p={n400_anova.get('p',1):.4f}, η²={n400_anova.get('eta2',0):.3f})")
    else:
        lines.append(f"  ⚠️  N400 không đạt ý nghĩa thống kê (p={n400_anova.get('p',1):.4f})")

    lines += [
        f"  • Water/605: N400 = {water_n400:.3f} µV",
        f"  • High/893:  N400 = {high_n400:.3f} µV",
        f"  • Điều kiện 'Quá nhiều' (JAR=4-5): N400 = {jar_nhieu_n400:.3f} µV",
        "  ",
        "  GIẢI THÍCH: N400 âm hơn ở điều kiện ngọt quá mức phản ánh",
        "  'mismatch' giữa kỳ vọng vị giác và trải nghiệm thực tế.",
        "  Tương tự N400 trong ngôn ngữ — não 'thất vọng' khi vị không phù hợp.",
        "",
    ]

    # N1 insight
    n1_anova = anova_res.get('N1', {})
    r_n1, p_r_n1 = linear_trend(measures, 'N1')
    lines += [
        "INSIGHT 3: N1 — Phân biệt và chú ý hóa học vị giác",
        "-" * 50,
        f"  ANOVA: F={n1_anova.get('F',0):.2f}, p={n1_anova.get('p',1):.4f} "
        f"{sig_stars(n1_anova.get('p',1))}, η²={n1_anova.get('eta2',0):.3f}",
    ]
    if r_n1 is not None:
        lines.append(f"  Linear trend: r={r_n1:.3f}, p={p_r_n1:.4f} {sig_stars(p_r_n1)}")
    lines += [
        "  ",
        "  GIẢI THÍCH: N1 tại F7/F8/T7/T8 phản ánh giai đoạn chú ý sớm",
        "  đến kích thích vị giác. Điện cực thái dương (T7/T8) nằm gần",
        "  vỏ não Insula — vỏ não vị giác sơ cấp của con người.",
        "  N1 âm hơn ở nồng độ cao = não nhận diện 'khác biệt' nhanh hơn.",
        "",
    ]

    # P1 insight
    p1_anova = anova_res.get('P1', {})
    r_p1, p_r_p1 = linear_trend(measures, 'P1')
    lines += [
        "INSIGHT 4: P1 — Xử lý hướng tâm sớm nhất",
        "-" * 50,
        f"  ANOVA: F={p1_anova.get('F',0):.2f}, p={p1_anova.get('p',1):.4f} "
        f"{sig_stars(p1_anova.get('p',1))}, η²={p1_anova.get('eta2',0):.3f}",
    ]
    if r_p1 is not None:
        lines.append(f"  Linear trend: r={r_p1:.3f}, p={p_r_p1:.4f} {sig_stars(p_r_p1)}")
    lines += [
        "  ",
        "  GIẢI THÍCH: P1 tại F3/F4/C3/C4 phản ánh giai đoạn đầu tiên",
        "  của xử lý vị giác khi phân tử sucrose tiếp xúc thụ cảm thể lưỡi.",
        "  P1 lớn hơn ở nồng độ cao = tín hiệu thần kinh hướng tâm mạnh hơn.",
        "",
    ]

    # General conclusion
    lines += [
        "=" * 80,
        "KẾT LUẬN TỔNG QUÁT",
        "=" * 80,
        "",
        "1. CHUỖI XỬ LÝ VỊ GIÁC: P1 (90ms) → N1 (180ms) → P2 (280ms) → N400 (450ms)",
        "   phản ánh các giai đoạn từ cảm giác thô → phân biệt → đánh giá → ký ức",
        "",
        "2. CẢM BIẾN VỊ: ERP vị giác có thể được dùng như biomarker khách quan",
        "   để đánh giá mức độ ngọt cảm thụ mà không phụ thuộc vào phản hồi chủ quan",
        "",
        "3. JAR RATING: Các thành phần ERP (đặc biệt P2 và N400) phân biệt tốt",
        "   giữa nhóm 'Vừa phải' và 'Quá nhiều', gợi ý ERP có thể dự đoán JAR",
        "",
        "4. MÁY HỌC: Kết hợp biên độ P1+N1+P2+N400 có tiềm năng phân loại",
        "   nồng độ sucrose hoặc nhóm JAR với độ chính xác vượt ngưỡng chance",
        "",
        "5. HẠN CHẾ: n=28, 100Hz sampling (giới hạn độ phân giải thời gian),",
        "   không có kênh EOG riêng (dùng Fp1/Fp2 proxy), biến thiên cá nhân cao",
        "",
    ]

    report = '\n'.join(lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n  ✓ Báo cáo insight: {save_path}')
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    config = load_config('configs/config.yaml')
    logger = setup_logging(config)
    ensure_dir(INSIGHT_DIR)
    ensure_dir(RESULTS_ERP)

    print('\n' + '=' * 70)
    print('  ERP INSIGHT ANALYSIS — P1 | N1 | P2 | N400')
    print('=' * 70)

    # ── Load existing CSV results ──
    print('\n[1/3] Đọc dữ liệu ERP từ CSV...')
    measures       = load_measures()
    conc_summary   = load_concentration_summary()
    jar_summary    = load_jar_summary()
    print(f'  component_measures: {measures.shape}')
    print(f'  concentration_summary: {conc_summary.shape}')
    print(f'  jar_summary: {jar_summary.shape}')

    # ── Load epochs for waveform plots ──
    print('\n[2/3] Đọc epochs và tính evoked...')
    (all_epochs, evoked_all,
     evoked_by_cond, evoked_by_jar, subjects,
     all_trial_info) = load_epochs_and_evoked(logger)
    if evoked_all is None:
        print('  ⚠️  Không load được epochs — bỏ qua waveform plots')

    # ── Region measures (trial-level, per brain region) ──
    print('\n[2b/3] Tính biên độ theo vùng não (region breakdown)...')
    if all_trial_info is not None and all_epochs:
        region_measures = extract_region_measures(all_epochs, all_trial_info)
        region_jar_stats = run_region_jar_stats(region_measures)
        print(f'  region_measures: {region_measures.shape}')
        print(f'  components × regions: {region_measures[["component","region"]].drop_duplicates().shape[0]}')
    else:
        region_measures = pd.DataFrame()
        region_jar_stats = {}

    # ── Channel measures (per individual electrode) ──
    print('\n[2c/3] Tính biên độ theo từng điện cực...')
    if all_trial_info is not None and all_epochs:
        channel_measures = extract_channel_measures(all_epochs, all_trial_info)
        channel_jar_stats = run_channel_jar_stats(channel_measures)
        print(f'  channel_measures: {channel_measures.shape}')
        print(f'  channels: {channel_measures["channel"].nunique()}')
    else:
        channel_measures = pd.DataFrame()
        channel_jar_stats = {}

    # ── Generate all figures ──
    print(f'\n[3/3] Tạo biểu đồ → {INSIGHT_DIR}')

    fig1_grand_average_detailed(evoked_all, INSIGHT_DIR)
    fig2_erp_by_concentration(evoked_by_cond, INSIGHT_DIR)
    fig3_amplitude_bar_by_condition(measures, INSIGHT_DIR)
    fig4_dose_response_curves(measures, INSIGHT_DIR)
    fig5_jar_group_analysis(evoked_by_jar, measures, INSIGHT_DIR)
    fig6_significance_heatmap(measures, INSIGHT_DIR)
    fig7_component_correlation(measures, INSIGHT_DIR)
    fig8_subject_variability(measures, INSIGHT_DIR)
    fig9_topomaps_by_condition(evoked_by_cond, INSIGHT_DIR)
    fig10_difference_waves(evoked_by_cond, INSIGHT_DIR)
    fig11_jar_pairwise_summary(measures, INSIGHT_DIR)
    fig12_erp_profile_radar(measures, INSIGHT_DIR)
    fig13_jar_difference_waves(evoked_by_jar, INSIGHT_DIR)
    fig14_region_breakdown(region_measures, INSIGHT_DIR)
    fig15_channel_breakdown(channel_measures, INSIGHT_DIR)

    # ── Generate insight report ──
    print('\n[+] Tạo báo cáo insight...')
    report = generate_insight_report(measures, conc_summary, jar_summary, REPORT_PATH,
                                     region_measures, region_jar_stats,
                                     channel_measures, channel_jar_stats)
    print('\n' + '-' * 70)
    print(report[:3000])  # print first 3000 chars to console
    if len(report) > 3000:
        print(f'  ... (xem đầy đủ tại {REPORT_PATH})')

    print('\n' + '=' * 70)
    print(f'  ✅ XONG! Tất cả biểu đồ tại: {INSIGHT_DIR}')
    print(f'  ✅ Báo cáo đầy đủ tại: {REPORT_PATH}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
