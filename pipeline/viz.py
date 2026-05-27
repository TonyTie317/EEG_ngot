"""
Visualization — ERP waveforms, topomaps, dose-response curves, model results.

All plots saved to config.paths.figures_base.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from .constants import (
    CONCENTRATIONS, CONCENTRATION_LABELS, ERP_WINDOWS, ROI,
    JAR_LABELS_VN, EEG_CHANNELS,
)
from .config import ensure_dir


def _get_fig_dir(config: Dict[str, Any]) -> str:
    """Get and create figures directory."""
    fig_dir = config['paths'].get('figures_base', 'output/figures')
    ensure_dir(fig_dir)
    return fig_dir


def _get_dpi(config: Dict[str, Any]) -> int:
    return config.get('visualization', {}).get('dpi', 300)


def _setup_style(config: Dict[str, Any]):
    """Apply matplotlib style from config."""
    style = config.get('visualization', {}).get('style', 'seaborn-v0_8-whitegrid')
    try:
        plt.style.use(style)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# ERP Waveform Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_grand_average_erp(
    evoked: mne.Evoked,
    config: Dict[str, Any],
    logger: logging.Logger,
    components: Optional[list] = None,
) -> str:
    """Plot grand-average ERP with highlighted component windows.

    One subplot per ROI (Frontal, Central, Parietal).

    Returns
    -------
    save_path : str
    """
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    if components is None:
        components = ['P1', 'N1', 'P2', 'N400']

    roi_groups = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
        'Central': ['C3', 'C4'],
        'Parietal': ['P3', 'P4'],
    }

    # Filter to channels present in data
    roi_groups = {
        name: [ch for ch in chs if ch in evoked.ch_names]
        for name, chs in roi_groups.items()
    }
    roi_groups = {k: v for k, v in roi_groups.items() if v}

    fig, axes = plt.subplots(len(roi_groups), 1, figsize=(12, 4 * len(roi_groups)),
                             sharex=True)
    if len(roi_groups) == 1:
        axes = [axes]

    times = evoked.times * 1000  # convert to ms
    data = evoked.get_data()  # (n_ch, n_times)

    component_colors = {
        'P1': '#2ecc71', 'N1': '#e74c3c',
        'P2': '#3498db', 'N400': '#9b59b6',
    }

    for ax, (roi_name, chs) in zip(axes, roi_groups.items()):
        # Average across ROI channels
        ch_indices = [evoked.ch_names.index(ch) for ch in chs]
        roi_data = data[ch_indices].mean(axis=0) * 1e6  # V → µV

        ax.plot(times, roi_data, 'k-', linewidth=1.5, label='Grand Average')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        # Shade component windows
        for comp in components:
            tmin_w, tmax_w = ERP_WINDOWS[comp]
            ax.axvspan(tmin_w * 1000, tmax_w * 1000, alpha=0.15,
                       color=component_colors.get(comp, 'gray'),
                       label=f'{comp} ({int(tmin_w*1000)}-{int(tmax_w*1000)}ms)')

        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'{roi_name} ROI')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (ms)')
    axes[-1].set_xlim(times[0], times[-1])
    fig.suptitle('Grand-Average ERP', fontsize=14, fontweight='bold')
    fig.tight_layout()

    save_path = os.path.join(fig_dir, 'grand_average_erp.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_erp_by_condition(
    evoked_by_condition: Dict[int, mne.Evoked],
    config: Dict[str, Any],
    logger: logging.Logger,
    roi_channels: Optional[list] = None,
) -> str:
    """Overlay ERP waveforms for each concentration level.

    Returns
    -------
    save_path : str
    """
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    if roi_channels is None:
        roi_channels = ['C3', 'C4', 'P3', 'P4']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: Centro-parietal ROI
    ax = axes[0]
    cmap = plt.cm.YlOrRd
    colors = [cmap(i / (len(CONCENTRATIONS) - 1)) for i in range(len(CONCENTRATIONS))]

    for i, cond in enumerate(CONCENTRATIONS):
        if cond not in evoked_by_condition:
            continue
        evoked_cond = evoked_by_condition[cond]
        chs = [ch for ch in roi_channels if ch in evoked_cond.ch_names]
        if not chs:
            continue
        data = evoked_cond.copy().pick(chs).get_data().mean(axis=0) * 1e6
        label = CONCENTRATION_LABELS.get(cond, str(cond))
        ax.plot(evoked_cond.times * 1000, data, color=colors[i],
                linewidth=1.5, label=label)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    # Shade P2 window
    ax.axvspan(350, 450, alpha=0.1, color='blue', label='P2 window')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Centro-Parietal ROI')
    ax.legend(fontsize=7, loc='upper right')

    # Right: Frontal ROI
    ax = axes[1]
    frontal_chs = ['Fp1', 'Fp2', 'F3', 'F4']
    for i, cond in enumerate(CONCENTRATIONS):
        if cond not in evoked_by_condition:
            continue
        evoked_cond = evoked_by_condition[cond]
        chs = [ch for ch in frontal_chs if ch in evoked_cond.ch_names]
        if not chs:
            continue
        data = evoked_cond.copy().pick(chs).get_data().mean(axis=0) * 1e6
        label = CONCENTRATION_LABELS.get(cond, str(cond))
        ax.plot(evoked_cond.times * 1000, data, color=colors[i],
                linewidth=1.5, label=label)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvspan(350, 500, alpha=0.1, color='purple', label='N400 window')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Frontal ROI')
    ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('ERP by Concentration Level', fontsize=14, fontweight='bold')
    fig.tight_layout()

    save_path = os.path.join(fig_dir, 'erp_by_concentration.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_erp_by_jar_group(
    evoked_by_jar: Dict[str, mne.Evoked],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Overlay ERP waveforms for each JAR group with SEM bands.

    Returns
    -------
    save_path : str
    """
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    group_colors = {
        'Khong_du': '#3498db',
        'Vua_phai': '#2ecc71',
        'Qua_nhieu': '#e74c3c',
    }
    roi_channels = ['C3', 'C4', 'P3', 'P4', 'F3', 'F4']

    fig, ax = plt.subplots(figsize=(12, 5))

    for group_name, evoked in evoked_by_jar.items():
        chs = [ch for ch in roi_channels if ch in evoked.ch_names]
        data = evoked.copy().pick(chs).get_data().mean(axis=0) * 1e6
        color = group_colors.get(group_name, 'gray')
        label = JAR_LABELS_VN.get(group_name, group_name)
        ax.plot(evoked.times * 1000, data, color=color,
                linewidth=2, label=label)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvspan(80, 120, alpha=0.1, color='green')
    ax.axvspan(350, 450, alpha=0.1, color='blue')
    ax.axvspan(350, 500, alpha=0.1, color='purple')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('ERP by JAR Group (Centro-Parietal + Frontal ROI)')
    ax.legend(fontsize=10)

    fig.tight_layout()
    save_path = os.path.join(fig_dir, 'erp_by_jar_group.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_topomap_components(
    evoked: mne.Evoked,
    config: Dict[str, Any],
    logger: logging.Logger,
    time_points: Optional[list] = None,
) -> str:
    """Plot scalp topographies at key time points.

    Parameters
    ----------
    time_points : list of float
        Time points in seconds. Default: [0.1, 0.15, 0.4, 0.45].
    """
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    if time_points is None:
        time_points = [0.100, 0.150, 0.400, 0.450]

    fig = evoked.plot_topomap(
        times=time_points,
        colorbar=True,
        ch_type='eeg',
        units='µV',
        scalings=dict(eeg=1e6),
        time_format='%0.3f s',
        show=False,
    )

    fig.suptitle('Scalp Topography at Component Latencies', fontsize=14)
    save_path = os.path.join(fig_dir, 'topomap_components.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close('all')
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_peak_dose_response(
    measures: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
    component: str = 'P2',
    measure: str = 'mean_amp',
) -> str:
    """Plot ERP peak amplitude/latency as function of concentration.

    X-axis: concentration level, Y-axis: amplitude (µV) or latency (ms).
    Error bars: SEM across subjects.
    """
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    col = f'{component}_{measure}'
    if col not in measures.columns:
        logger.warning(f"Column '{col}' not in measures")
        return ''

    fig, ax = plt.subplots(figsize=(8, 5))

    x_vals = []
    y_vals = []
    y_errs = []

    for cond in CONCENTRATIONS:
        cond_data = measures[measures['condition'] == cond][col].dropna()
        if len(cond_data) > 0:
            x_vals.append(cond)
            y_vals.append(cond_data.mean() * 1e6 if 'amp' in measure else cond_data.mean() * 1000)
            y_errs.append(
                (cond_data.std() / np.sqrt(len(cond_data))) * (1e6 if 'amp' in measure else 1000)
            )

    ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o-', capsize=5,
                linewidth=2, markersize=8)
    ax.set_xlabel('Concentration')
    ax.set_ylabel('Amplitude (µV)' if 'amp' in measure else 'Latency (ms)')
    ax.set_title(f'{component} {measure} vs Concentration')
    ax.set_xticks(CONCENTRATIONS)
    ax.set_xticklabels([CONCENTRATION_LABELS[c] for c in CONCENTRATIONS],
                       rotation=45, ha='right')

    fig.tight_layout()
    save_path = os.path.join(fig_dir, f'{component}_{measure}_dose_response.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_difference_waves(
    diff_waves: Dict[str, mne.Evoked],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Plot difference waves (e.g., high - water, JAR contrasts)."""
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    roi_channels = ['C3', 'C4', 'P3', 'P4', 'F3', 'F4']

    n = len(diff_waves)
    if n == 0:
        return ''

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, evoked) in zip(axes, diff_waves.items()):
        chs = [ch for ch in roi_channels if ch in evoked.ch_names]
        data = evoked.copy().pick(chs).get_data().mean(axis=0) * 1e6
        ax.plot(evoked.times * 1000, data, 'k-', linewidth=1.5)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.fill_between(evoked.times * 1000, data, 0,
                        where=data > 0, alpha=0.3, color='red')
        ax.fill_between(evoked.times * 1000, data, 0,
                        where=data < 0, alpha=0.3, color='blue')
        ax.set_xlabel('Time (ms)')
        ax.set_title(name.replace('_', ' '))

    axes[0].set_ylabel('Amplitude (µV)')
    fig.suptitle('Difference Waves', fontsize=14, fontweight='bold')
    fig.tight_layout()

    save_path = os.path.join(fig_dir, 'difference_waves.png')
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# ML/DL Result Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list,
    title: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    filename: str = 'confusion_matrix.png',
) -> str:
    """Plot a labeled confusion matrix with accuracy annotation."""
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title}\nAccuracy: {acc:.3f}')

    fig.tight_layout()
    save_path = os.path.join(fig_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_model_comparison(
    results: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger,
    filename: str = 'model_comparison.png',
) -> str:
    """Bar chart comparing accuracy/F1 across models and tasks."""
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    if not results:
        return ''

    # Flatten results into a plottable DataFrame
    rows = []
    for task, task_results in results.items():
        for model_name, metrics in task_results.items():
            if isinstance(metrics, dict):
                rows.append({
                    'task': task,
                    'model': model_name,
                    'accuracy': metrics.get('accuracy', 0),
                    'f1_macro': metrics.get('f1_macro', 0),
                })

    if not rows:
        return ''

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ['accuracy', 'f1_macro']):
        pivot = df.pivot(index='model', columns='task', values=metric)
        pivot.plot(kind='bar', ax=ax, rot=45)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend(title='Task')

    fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()

    save_path = os.path.join(fig_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


# ──────────────────────────────────────────────────────────────────────────────
# Master entry point
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    task: str,
    model_name: str,
    config: Dict[str, Any],
    logger: logging.Logger,
    top_n: int = 30,
) -> str:
    """Bar chart of top-N feature importances (RF/XGB) or MI scores."""
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    ax.barh(range(len(names)), vals, color='steelblue')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top-{top_n} Feature Importance\n{task} — {model_name}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    filename = f'feat_importance_{task}_{model_name}.png'
    save_path = os.path.join(fig_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_feature_sweep(
    sweep_df: pd.DataFrame,
    task: str,
    model_name: str,
    best_n: int,
    chance: float,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Line plot of accuracy vs n_features sweep."""
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df['n_features'], sweep_df['balanced_accuracy'],
            marker='o', color='steelblue', label='Balanced Acc')
    ax.plot(sweep_df['n_features'], sweep_df['f1_macro'],
            marker='s', color='tomato', linestyle='--', label='F1-macro')
    ax.axhline(chance, color='grey', linestyle=':', label=f'Chance ({chance:.2f})')
    ax.axvline(best_n, color='green', linestyle='--', alpha=0.7,
               label=f'Best n={best_n}')
    ax.set_xlabel('Number of Features (top-N by MI)')
    ax.set_ylabel('Score')
    ax.set_title(f'Feature Count Sweep\n{task} — {model_name}',
                 fontsize=12, fontweight='bold')
    ax.legend()
    fig.tight_layout()

    filename = f'feat_sweep_{task}_{model_name}.png'
    save_path = os.path.join(fig_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def plot_per_fold_accuracy(
    fold_accs: List[float],
    task: str,
    model_name: str,
    chance: float,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    """Bar chart of per-fold (per-subject) LOSO accuracy."""
    _setup_style(config)
    fig_dir = _get_fig_dir(config)
    dpi = _get_dpi(config)

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['#4CAF50' if a > chance else '#EF5350' for a in fold_accs]
    ax.bar(range(1, len(fold_accs) + 1), fold_accs, color=colors, edgecolor='white')
    ax.axhline(chance, color='grey', linestyle='--', label=f'Chance ({chance:.2f})')
    ax.axhline(np.mean(fold_accs), color='navy', linestyle='-',
               linewidth=1.5, label=f'Mean ({np.mean(fold_accs):.3f})')
    ax.set_xlabel('Subject (fold)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Per-Subject LOSO Accuracy\n{task} — {model_name}',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    filename = f'fold_acc_{task}_{model_name}.png'
    save_path = os.path.join(fig_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return save_path


def generate_all_figures(
    erp_results: Dict[str, Any],
    ml_results: Optional[Dict[str, Any]] = None,
    dl_results: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Generate all figures from ERP, ML, and DL results."""
    if config is None or logger is None:
        return

    logger.info("=" * 60)
    logger.info("STAGE: Visualization")
    logger.info("=" * 60)

    if not erp_results:
        logger.warning("No ERP results to plot.")
        return

    # ERP figures
    if 'evoked_all' in erp_results:
        plot_grand_average_erp(erp_results['evoked_all'], config, logger)

    if 'evoked_by_condition' in erp_results:
        plot_erp_by_condition(erp_results['evoked_by_condition'], config, logger)

    if 'evoked_by_jar_group' in erp_results:
        plot_erp_by_jar_group(erp_results['evoked_by_jar_group'], config, logger)

    if 'evoked_all' in erp_results:
        plot_topomap_components(erp_results['evoked_all'], config, logger)

    if 'measures' in erp_results:
        for comp in ['P1', 'P2', 'N400']:
            for mtype in ['mean_amp', 'peak_amp']:
                plot_peak_dose_response(
                    erp_results['measures'], config, logger,
                    component=comp, measure=mtype,
                )

    if 'diff_waves' in erp_results:
        plot_difference_waves(erp_results['diff_waves'], config, logger)

    # ML/DL figures
    if ml_results:
        plot_model_comparison(ml_results, config, logger,
                              filename='ml_model_comparison.png')

    if dl_results:
        plot_model_comparison(dl_results, config, logger,
                              filename='dl_model_comparison.png')

    logger.info("All figures generated.")
