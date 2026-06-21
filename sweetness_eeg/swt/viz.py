"""Plotting helpers (matplotlib Agg backend). All functions save a PNG and close."""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .constants import (
    BAND_COLORS, COND_COLORS, COND_DISPLAY, CONDITIONS, EEG_CHANNELS,
    FREQ_BANDS, SFREQ, SUBSTANCE_COLORS,
)

sns.set_theme(style='whitegrid', context='notebook')


def _save(fig, path: str, dpi: int = 150) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path


def mne_info():
    """An MNE Info (16 EEG, 10-20 montage) for topomaps."""
    import mne
    info = mne.create_info(list(EEG_CHANNELS), SFREQ, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                     match_case=False, on_missing='ignore', verbose=False)
    return info


# ── PSD spectra ───────────────────────────────────────────────────────────────
def plot_psd_spectra(freqs, grand: Dict[str, np.ndarray], path: str,
                     title: str = 'Grand-average PSD by condition',
                     conditions: Optional[List[str]] = None) -> str:
    conditions = conditions or [c for c in CONDITIONS if c in grand]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for b, (lo, hi) in FREQ_BANDS.items():
        ax.axvspan(lo, hi, color=BAND_COLORS[b], alpha=0.06)
    for c in conditions:
        ax.semilogy(freqs, grand[c], color=COND_COLORS.get(c, 'k'),
                    label=COND_DISPLAY.get(c, c), lw=1.8)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V²/Hz, log)')
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlim(freqs.min(), freqs.max())
    return _save(fig, path)


# ── Topomaps ──────────────────────────────────────────────────────────────────
def plot_topomap_grid(value_fn, bands: List[str], conditions: List[str], path: str,
                      title: str = '', cmap: str = 'RdBu_r',
                      symmetric: bool = False, unit: str = '') -> str:
    """Grid of topomaps: rows = bands, cols = conditions.

    value_fn(condition, band) -> channel vector (len EEG_CHANNELS).
    """
    import mne
    info = mne_info()
    nrow, ncol = len(bands), len(conditions)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol + 1.2, 2.1 * nrow + 0.8))
    axes = np.atleast_2d(axes)
    for r, band in enumerate(bands):
        mats = [value_fn(c, band) for c in conditions]
        allv = np.concatenate([m[np.isfinite(m)] for m in mats]) if mats else np.array([0.])
        if symmetric:
            vmax = np.nanmax(np.abs(allv)) if allv.size else 1.0
            vlim = (-vmax, vmax)
        else:
            vlim = (np.nanmin(allv), np.nanmax(allv)) if allv.size else (0, 1)
        for c, ax in enumerate(axes[r]):
            mne.viz.plot_topomap(mats[c], info, axes=ax, show=False, cmap=cmap,
                                 vlim=vlim, contours=4)
            if r == 0:
                ax.set_title(COND_DISPLAY.get(conditions[c], conditions[c]), fontsize=9)
            if c == 0:
                ax.text(-0.35, 0.5, band, transform=ax.transAxes, fontsize=11,
                        fontweight='bold', va='center', ha='right',
                        color=BAND_COLORS.get(band, 'k'))
        # one colorbar per row
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vlim[0], vmax=vlim[1]))
        cb = fig.colorbar(sm, ax=list(axes[r]), fraction=0.012, pad=0.01)
        cb.ax.tick_params(labelsize=6)
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.subplots_adjust(top=0.92)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_topomap_row(values: List[np.ndarray], titles: List[str], path: str,
                     suptitle: str = '', cmap: str = 'RdBu_r',
                     symmetric: bool = True) -> str:
    """A single row of topomaps (e.g. sucrose−sucralose difference per band)."""
    import mne
    info = mne_info()
    fig, axes = plt.subplots(1, len(values), figsize=(2.4 * len(values) + 1, 2.8))
    axes = np.atleast_1d(axes)
    for ax, v, t in zip(axes, values, titles):
        finite = v[np.isfinite(v)]
        vmax = np.nanmax(np.abs(finite)) if finite.size else 1.0
        vlim = (-vmax, vmax) if symmetric else (np.nanmin(finite), np.nanmax(finite))
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap,
                                     vlim=vlim, contours=4)
        ax.set_title(t, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12, fontweight='bold')
    return _save(fig, path)


# ── Bars / boxes / dose-response ─────────────────────────────────────────────
def plot_condition_box(df, value: str, path: str, title: str = '', ylabel: str = '',
                       conditions: Optional[List[str]] = None) -> str:
    conditions = conditions or [c for c in CONDITIONS if c in df['ma_mau'].unique()]
    fig, ax = plt.subplots(figsize=(9, 5))
    order = conditions
    pal = [COND_COLORS.get(c, '#888') for c in order]
    sns.boxplot(data=df[df['ma_mau'].isin(order)], x='ma_mau', y=value, order=order,
                palette=pal, ax=ax, showfliers=False)
    sns.stripplot(data=df[df['ma_mau'].isin(order)], x='ma_mau', y=value, order=order,
                  color='0.25', size=2.5, alpha=0.4, ax=ax)
    ax.set_xticklabels([COND_DISPLAY.get(c, c) for c in order], rotation=30, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel or value)
    ax.set_title(title)
    return _save(fig, path)


def plot_dose_response(summary, value_mean: str, value_sem: str, path: str,
                       title: str = '', ylabel: str = '') -> str:
    """Dose-response: intensity (x) × substance (lines), with SEM error bars."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for sub in ['Sucrose', 'Sucralose']:
        s = summary[summary['substance'] == sub].sort_values('intensity')
        if s.empty:
            continue
        ax.errorbar(s['intensity'], s[value_mean], yerr=s[value_sem], marker='o',
                    capsize=4, lw=2, color=SUBSTANCE_COLORS[sub], label=sub)
    # water reference line
    w = summary[summary['substance'] == 'Water']
    if not w.empty:
        ax.axhline(w[value_mean].iloc[0], color='gray', ls='--', alpha=0.7,
                   label='Water')
    ax.set_xticks([5, 7, 12])
    ax.set_xticklabels(['~5%', '~7.5%', '~12%'])
    ax.set_xlabel('Perceived sweetness intensity')
    ax.set_ylabel(ylabel or value_mean)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


def plot_grouped_bars(summary, value_mean: str, value_sem: str, path: str,
                      title: str = '', ylabel: str = '') -> str:
    """Grouped bars: x = intensity, hue = substance."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    intensities = [5, 7, 12]
    width = 0.38
    x = np.arange(len(intensities))
    for k, sub in enumerate(['Sucrose', 'Sucralose']):
        s = summary[summary['substance'] == sub].set_index('intensity')
        means = [s.loc[i, value_mean] if i in s.index else np.nan for i in intensities]
        sems = [s.loc[i, value_sem] if i in s.index else 0 for i in intensities]
        ax.bar(x + (k - 0.5) * width, means, width, yerr=sems, capsize=4,
               color=SUBSTANCE_COLORS[sub], label=sub)
    ax.set_xticks(x)
    ax.set_xticklabels(['~5%', '~7.5%', '~12%'])
    ax.set_ylabel(ylabel or value_mean)
    ax.set_title(title)
    ax.legend()
    return _save(fig, path)


# ── Time-resolved & TFR ──────────────────────────────────────────────────────
def plot_time_resolved(tr_df, band: str, path: str, title: str = '',
                       by: str = 'substance') -> str:
    """Mean ± SEM band power over time, grouped by substance (or condition)."""
    sub = tr_df[tr_df['band'] == band]
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = ['Sucrose', 'Sucralose', 'Water'] if by == 'substance' \
        else [c for c in CONDITIONS if c in sub['ma_mau'].unique()]
    for g in groups:
        col = 'substance' if by == 'substance' else 'ma_mau'
        gd = sub[sub[col] == g]
        if gd.empty:
            continue
        # subject mean first, then group SEM across subjects
        per = gd.groupby(['subject', 't_center'])['power'].mean().reset_index()
        stat = per.groupby('t_center')['power'].agg(['mean', 'sem']).reset_index()
        color = SUBSTANCE_COLORS.get(g, COND_COLORS.get(g, 'k'))
        ax.plot(stat['t_center'], stat['mean'], lw=2, color=color, label=g)
        ax.fill_between(stat['t_center'], stat['mean'] - stat['sem'],
                        stat['mean'] + stat['sem'], color=color, alpha=0.18)
    ax.set_xlabel('Time within 10 s tasting window (s)')
    ax.set_ylabel(f'{band} power (V²)')
    ax.set_title(title or f'Time-resolved {band} power')
    ax.legend()
    return _save(fig, path)


def plot_tfr(freqs, times, tfr: Dict[str, np.ndarray], path: str,
             title: str = '') -> str:
    """Side-by-side TFR spectrograms per substance (relative to each map's mean)."""
    subs = [s for s in ['Sucrose', 'Sucralose', 'Water'] if s in tfr]
    fig, axes = plt.subplots(1, len(subs), figsize=(4.4 * len(subs), 4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subs):
        z = 10 * np.log10(tfr[s] / tfr[s].mean(axis=1, keepdims=True))
        im = ax.pcolormesh(times, freqs, z, cmap='RdBu_r', shading='auto',
                           vmin=-3, vmax=3)
        ax.set_title(s)
        ax.set_xlabel('Time (s)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('dB vs mean')
    axes[0].set_ylabel('Frequency (Hz)')
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    return _save(fig, path)


# ── Generic ──────────────────────────────────────────────────────────────────
def plot_confusion(cm: np.ndarray, labels: List[str], path: str, title: str = '',
                   subtitle: str = '') -> str:
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 1.0 * len(labels) + 2.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title((title + ('\n' + subtitle if subtitle else '')))
    return _save(fig, path)


def plot_bar_simple(labels: List[str], values: List[float], path: str,
                    title: str = '', ylabel: str = '', chance: Optional[float] = None,
                    errors: Optional[List[float]] = None) -> str:
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 4.5))
    bars = ax.bar(range(len(labels)), values, yerr=errors, capsize=4,
                  color='steelblue', edgecolor='white')
    if chance is not None:
        ax.axhline(chance, color='gray', ls='--', label=f'Chance ({chance:.2f})')
        ax.legend()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center',
                va='bottom', fontsize=8)
    return _save(fig, path)


def plot_scatter_corr(x, y, path: str, xlabel: str = '', ylabel: str = '',
                      title: str = '', hue=None) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    if hue is not None:
        for g in np.unique(hue):
            m = hue == g
            ax.scatter(np.asarray(x)[m], np.asarray(y)[m], s=22, alpha=0.7,
                       label=str(g), color=SUBSTANCE_COLORS.get(str(g)))
        ax.legend(fontsize=8)
    else:
        ax.scatter(x, y, s=22, alpha=0.7, color='steelblue')
    # regression line
    xv, yv = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    if ok.sum() > 2:
        b, a = np.polyfit(xv[ok], yv[ok], 1)
        xs = np.linspace(xv[ok].min(), xv[ok].max(), 50)
        ax.plot(xs, b * xs + a, color='k', lw=1.5, ls='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return _save(fig, path)
