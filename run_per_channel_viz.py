#!/usr/bin/env python3
"""
Visualize per-channel ERP effects — tập trung vào channels significant nhất.

1. C4 / P2 waveform theo JAR group và concentration
2. F7 / N400 waveform theo JAR group
3. Dose-response curves (P2 amplitude × concentration) tại C4, F4, P3
4. Topomap P2 và N400 theo JAR group

Usage:
    .venv/bin/python run_per_channel_viz.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.constants import (
    ALL_SUBJECTS, CONCENTRATIONS, CONCENTRATION_LABELS,
    ERP_WINDOWS, ERP_ROI, ERP_PEAK_MODE, EEG_CHANNELS,
)

EPOCHS_BASE = 'output/epochs'
TRIAL_INFO_CSV = 'output/epochs/all_trial_info.csv'
FIGS_DIR = 'output/figures/per_channel'
REALIGN_CSV = 'output/epochs/realign_offsets.csv'
_REALIGN_TMIN = -0.2
_REALIGN_TMAX = 1.0

JAR_COLORS = {'Khong_du': '#e41a1c', 'Vua_phai': '#4daf4a', 'Qua_nhieu': '#377eb8'}
JAR_LABELS = {'Khong_du': 'Không đủ', 'Vua_phai': 'Vừa phải', 'Qua_nhieu': 'Quá nhiều'}
COND_COLORS = {605: '#4daf4a', 258: '#377eb8', 453: '#ff7f00', 189: '#984ea3', 762: '#e41a1c', 893: '#a65628'}
WINDOW_COLORS = {'P1': '#4CAF50', 'N1': '#2196F3', 'P2': '#FF9800', 'N400': '#E91E63'}


def load_all_data():
    """Load epochs + trial_info cho tất cả subjects."""
    all_data = []
    all_ti = []
    ch_names = None
    times = None
    sfreq = None

    for sid in ALL_SUBJECTS:
        fif_path = os.path.join(EPOCHS_BASE, sid, 'epochs_epo.fif')
        if not os.path.exists(fif_path):
            continue
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        ti = pd.read_csv(os.path.join(EPOCHS_BASE, sid, 'trial_info.csv'))
        data = epochs.get_data()

        if ch_names is None:
            ch_names = epochs.ch_names
            times = epochs.times
            sfreq = epochs.info['sfreq']

        # Woody realignment
        if os.path.exists(REALIGN_CSV):
            offsets = pd.read_csv(REALIGN_CSV)
            subj_off = offsets[offsets['subject_id'] == sid]
            if len(subj_off) > 0:
                pre = int(abs(_REALIGN_TMIN) * sfreq)
                post = int(_REALIGN_TMAX * sfreq)
                win_len = pre + post + 1
                offset_orig = int(abs(epochs.tmin) * sfreq)
                new_data = []
                kept_idx = []
                for ep_i in range(len(epochs)):
                    final_off = int(subj_off.iloc[ep_i]['offset_final']) if ep_i < len(subj_off) else 0
                    true_onset = offset_orig + final_off
                    start = true_onset - pre
                    end = start + win_len
                    if start >= 0 and end <= data.shape[2]:
                        new_data.append(data[ep_i, :, start:end])
                        kept_idx.append(ep_i)
                if new_data:
                    data = np.stack(new_data, axis=0)
                    ti = ti.iloc[kept_idx].reset_index(drop=True)
                    times = np.arange(-pre, post + 1) / sfreq

        all_data.append(data)
        all_ti.append(ti)

    X = np.concatenate(all_data, axis=0)
    trial_info = pd.concat(all_ti, ignore_index=True)
    return X, times, ch_names, trial_info


def plot_channel_waveform_jar(X, times, ch_names, trial_info, ch_name, component):
    """ERP waveform tại 1 channel, split by JAR group."""
    ch_idx = ch_names.index(ch_name) if ch_name in ch_names else -1
    if ch_idx < 0:
        return None

    ch_data = X[:, ch_idx, :]  # (n_epochs, n_times)
    window = ERP_WINDOWS.get(component, (0.09, 0.15))

    fig, ax = plt.subplots(figsize=(8, 5))

    for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
        mask = trial_info['jar_group'] == jg
        if mask.sum() == 0:
            continue
        avg = ch_data[mask.values].mean(axis=0) * 1e6  # µV
        sem = ch_data[mask.values].std(axis=0) / np.sqrt(mask.sum()) * 1e6
        ax.plot(times, avg, color=JAR_COLORS[jg], linewidth=2,
                label=f'{JAR_LABELS[jg]} (n={mask.sum()})')
        ax.fill_between(times, avg - sem, avg + sem, color=JAR_COLORS[jg], alpha=0.1)

    # Component window
    ax.axvspan(window[0], window[1], alpha=0.12, color=WINDOW_COLORS.get(component, 'gray'))
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-0.2, 0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title(f'{ch_name} — {component} Waveform by JAR Group', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, f'{ch_name}_{component}_by_JAR.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_dose_response(X, times, ch_names, trial_info, channels, component, measure='mean_amp'):
    """Dose-response curves: ERP amplitude × concentration, split by JAR."""
    n_ch = len(channels)
    fig, axes = plt.subplots(1, n_ch, figsize=(5 * n_ch, 4), sharey=True)
    if n_ch == 1:
        axes = [axes]

    window = ERP_WINDOWS.get(component, (0.09, 0.15))
    t_mask = (times >= window[0]) & (times <= window[1])

    for ax_idx, ch_name in enumerate(channels):
        ax = axes[ax_idx]
        ch_idx = ch_names.index(ch_name) if ch_name in ch_names else -1
        if ch_idx < 0:
            continue

        ch_data = X[:, ch_idx, :]  # (n_epochs, n_times)

        for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
            x_vals = []
            y_vals = []
            err_vals = []
            n_vals = []

            for cond in CONCENTRATIONS:
                mask = (trial_info['jar_group'] == jg) & (trial_info['condition'] == cond)
                if mask.sum() < 2:
                    continue
                win_data = ch_data[mask.values][:, t_mask]
                if measure == 'mean_amp':
                    vals = win_data.mean(axis=1)
                else:
                    vals = win_data.max(axis=1) if ERP_PEAK_MODE.get(component, 'pos') == 'pos' else win_data.min(axis=1)

                x_vals.append(cond)
                y_vals.append(vals.mean() * 1e6)
                err_vals.append(vals.std() / np.sqrt(len(vals)) * 1e6)
                n_vals.append(len(vals))

            if x_vals:
                ax.errorbar(range(len(x_vals)), y_vals, yerr=err_vals,
                           color=JAR_COLORS[jg], linewidth=2, marker='o',
                           label=f'{JAR_LABELS[jg]}', capsize=3)
                for i, n in enumerate(n_vals):
                    ax.annotate(f'n={n}', (i, y_vals[i]), fontsize=6,
                               ha='center', va='bottom' if y_vals[i] > 0 else 'top')

        ax.set_xticks(range(len(CONCENTRATIONS)))
        ax.set_xticklabels([CONCENTRATION_LABELS[c] for c in CONCENTRATIONS], rotation=30, ha='right')
        ax.set_ylabel(f'{component} {measure} (µV)')
        ax.set_title(f'{ch_name}', fontsize=12)
        ax.legend(fontsize=8)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.grid(alpha=0.2)

    fig.suptitle(f'{component} {measure} — Dose-Response by JAR Group', fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, f'dose_response_{component}_{"_".join(channels)}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_topomap_jarp2(X, times, ch_names, trial_info, component='P2'):
    """Topomap: ERP component topography theo JAR group."""
    window = ERP_WINDOWS.get(component, (0.23, 0.35))
    t_mask = (times >= window[0]) & (times <= window[1])
    if t_mask.sum() == 0:
        return

    import warnings
    # Tạo montage từ channel names
    info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)
    except Exception:
        pass  # montage không critical cho topomap

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    vmin, vmax = None, None
    topo_data = []

    for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
        mask = trial_info['jar_group'] == jg
        if mask.sum() == 0:
            continue
        win_data = X[mask.values][:, :, t_mask]
        topo = win_data.mean(axis=(0, 2)) * 1e6
        topo_data.append(topo)

    all_topo = np.concatenate(topo_data) if topo_data else np.zeros(len(ch_names))
    vmax = max(abs(all_topo.min()), abs(all_topo.max()))
    vmin, vmax = -vmax, vmax

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for idx, (jg, label) in enumerate(zip(
                ['Khong_du', 'Vua_phai', 'Qua_nhieu'],
                ['Không đủ', 'Vừa phải', 'Quá nhiều'])):
            mask = trial_info['jar_group'] == jg
            if mask.sum() == 0:
                continue
            win_data = X[mask.values][:, :, t_mask]
            topo = win_data.mean(axis=(0, 2)) * 1e6
            mne.viz.plot_topomap(topo, info, axes=axes[idx], show=False,
                                vlim=(vmin, vmax), cmap='RdBu_r',
                                sensors=True)
            axes[idx].set_title(f'{label}\n(n={mask.sum()})', fontsize=11)

    fig.suptitle(f'{component} Topography by JAR Group ({window[0]*1000:.0f}-{window[1]*1000:.0f}ms)',
                 fontsize=13, y=1.05)
    plt.tight_layout()
    path = os.path.join(FIGS_DIR, f'topomap_{component}_by_JAR.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_component_comparison_bars(X, times, ch_names, trial_info):
    """Bar chart: C4 P2 amp và F7 N400 amp theo JAR group."""
    pairs = [('C4', 'P2', 'peak_amp'), ('F7', 'N400', 'peak_amp'),
             ('C4', 'P2', 'mean_amp'), ('P4', 'N400', 'mean_amp')]

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    if len(pairs) == 1:
        axes = [axes]

    for ax_idx, (ch_name, comp, measure) in enumerate(pairs):
        ax = axes[ax_idx]
        ch_idx = ch_names.index(ch_name) if ch_name in ch_names else -1
        if ch_idx < 0:
            continue

        window = ERP_WINDOWS.get(comp, (0.09, 0.15))
        t_mask = (times >= window[0]) & (times <= window[1])
        mode = ERP_PEAK_MODE.get(comp, 'pos')

        ch_data = X[:, ch_idx, :]
        means = []
        errs = []
        ns = []
        groups_plot = []

        for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
            mask = trial_info['jar_group'] == jg
            if mask.sum() < 2:
                continue
            win_data = ch_data[mask.values][:, t_mask]
            if measure == 'mean_amp':
                vals = win_data.mean(axis=1)
            elif measure == 'peak_amp':
                if mode == 'pos':
                    vals = win_data.max(axis=1)
                else:
                    vals = win_data.min(axis=1)
            else:
                continue
            means.append(vals.mean() * 1e6)
            errs.append(vals.std() / np.sqrt(len(vals)) * 1e6)
            ns.append(len(vals))
            groups_plot.append(JAR_LABELS[jg])

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=errs, capsize=5,
                     color=[JAR_COLORS[jg] for jg in ['Khong_du', 'Vua_phai', 'Qua_nhieu']],
                     alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(groups_plot)
        ax.set_ylabel(f'{comp} {measure} (µV)')
        ax.set_title(f'{ch_name} — {comp} {measure}', fontsize=11)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        for i, n in enumerate(ns):
            ax.annotate(f'n={n}', (x[i], means[i]), ha='center', va='bottom' if means[i] > 0 else 'top',
                       fontsize=8)
        ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, 'key_channels_JAR_bars.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    config = load_config('configs/config.yaml')
    logger = setup_logging(config)
    os.makedirs(FIGS_DIR, exist_ok=True)

    logger.info('Loading data...')
    X, times, ch_names, trial_info = load_all_data()
    logger.info(f'Data: {X.shape[0]} epochs, {X.shape[1]} channels')

    # 1. C4 P2 waveform by JAR
    logger.info('Plotting C4 P2 waveform by JAR...')
    plot_channel_waveform_jar(X, times, ch_names, trial_info, 'C4', 'P2')

    # 2. F7 N400 waveform by JAR
    logger.info('Plotting F7 N400 waveform by JAR...')
    plot_channel_waveform_jar(X, times, ch_names, trial_info, 'F7', 'N400')

    # 3. F4 P2 waveform (gần significant) by JAR
    logger.info('Plotting F4 P2 waveform by JAR...')
    plot_channel_waveform_jar(X, times, ch_names, trial_info, 'F4', 'P2')

    # 4. Dose-response curves
    logger.info('Plotting dose-response curves...')
    plot_dose_response(X, times, ch_names, trial_info, ['C4', 'F4', 'P3'], 'P2', 'peak_amp')

    # 5. Topomap
    logger.info('Plotting topomap...')
    plot_topomap_jarp2(X, times, ch_names, trial_info, 'P2')
    plot_topomap_jarp2(X, times, ch_names, trial_info, 'N400')

    # 6. Bar charts
    logger.info('Plotting bar charts...')
    plot_component_comparison_bars(X, times, ch_names, trial_info)

    logger.info(f'All figures saved to {FIGS_DIR}')
    print(f'\nFigures in {FIGS_DIR}:')
    for f in sorted(os.listdir(FIGS_DIR)):
        print(f'  {f}')


if __name__ == '__main__':
    main()
