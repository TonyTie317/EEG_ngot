"""
Spectral analysis — the core of the study (mirrors the reference paper).

Power spectral density (Welch) over the 10 s tasting window, band power for the
five canonical bands (δ, θ, α, β, γ), ROI aggregation (frontal & gustatory),
time-resolved band power (early vs late "aftertaste" windows) and Morlet TFR.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

from .constants import (
    BAND_ORDER, EEG_CHANNELS, FREQ_BANDS, ROIS, SFREQ,
)


# ── PSD ───────────────────────────────────────────────────────────────────────
def compute_psd(data: np.ndarray, sfreq: float, cfg: Dict[str, Any]
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD of an array (..., n_times). Returns (freqs, psd[..., n_freqs])."""
    sp = cfg.get('spectral', {})
    nperseg = int(sp.get('welch_seconds', 2.0) * sfreq)
    nperseg = min(nperseg, data.shape[-1])
    noverlap = int(nperseg * sp.get('welch_overlap', 0.5))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
    fmin, fmax = sp.get('fmin', 1.0), sp.get('fmax', 45.0)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[..., mask]


def band_power(freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]
               ) -> np.ndarray:
    """Integrate PSD over a band (trapezoid) → power array (psd shape minus freq)."""
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    return np.trapezoid(psd[..., m], freqs[m], axis=-1)


def total_power(freqs: np.ndarray, psd: np.ndarray,
                fmin: float = 1.0, fmax: float = 45.0) -> np.ndarray:
    m = (freqs >= fmin) & (freqs <= fmax)
    return np.trapezoid(psd[..., m], freqs[m], axis=-1)


# ── Per-trial band-power table ────────────────────────────────────────────────
def bandpower_table(all_epochs: list, cfg: Dict[str, Any],
                    logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Per-trial absolute & relative band power for every channel.

    Long format: subject, ma_mau, substance, intensity, repeat, channel,
    band, abs_power, rel_power.
    """
    rows = []
    for ep in all_epochs:
        data = ep.get_data(copy=False)               # (n_ep, n_ch, n_times)
        meta = ep.metadata.reset_index(drop=True)
        freqs, psd = compute_psd(data, SFREQ, cfg)   # (n_ep, n_ch, n_freqs)
        tot = total_power(freqs, psd, cfg['spectral'].get('fmin', 1.0),
                          cfg['spectral'].get('fmax', 45.0))     # (n_ep, n_ch)
        bp = {b: band_power(freqs, psd, FREQ_BANDS[b]) for b in BAND_ORDER}
        for i in range(data.shape[0]):
            m = meta.iloc[i]
            for ci, ch in enumerate(EEG_CHANNELS):
                for b in BAND_ORDER:
                    abs_p = bp[b][i, ci]
                    rows.append({
                        'subject': m['subject'], 'ma_mau': m['ma_mau'],
                        'substance': m['substance'], 'intensity': int(m['intensity']),
                        'repeat': int(m['repeat']), 'channel': ch, 'band': b,
                        'abs_power': abs_p,
                        'rel_power': abs_p / tot[i, ci] if tot[i, ci] > 0 else np.nan,
                    })
    df = pd.DataFrame(rows)
    if logger:
        logger.info(f"Band-power table: {len(df)} rows "
                    f"({df['subject'].nunique()} subjects)")
    return df


def roi_bandpower(bp_long: pd.DataFrame, rois: Dict[str, List[str]] = ROIS
                  ) -> pd.DataFrame:
    """Average channel band power within each ROI.

    Returns long format: subject, ma_mau, substance, intensity, roi, band,
    abs_power, rel_power.
    """
    ch2roi = []
    for roi, chans in rois.items():
        for ch in chans:
            ch2roi.append({'channel': ch, 'roi': roi})
    map_df = pd.DataFrame(ch2roi)
    merged = bp_long.merge(map_df, on='channel')
    agg = (merged
           .groupby(['subject', 'ma_mau', 'substance', 'intensity', 'roi', 'band'],
                    dropna=False)[['abs_power', 'rel_power']]
           .mean().reset_index())
    return agg


def grand_psd_by_condition(all_epochs: list, cfg: Dict[str, Any]
                           ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Subject-averaged PSD per condition, channel-averaged.

    Returns (freqs, {condition: psd_mean_over_subjects_and_channels[n_freqs]}).
    Two-step average (within subject, then across subjects) to avoid trial-count bias.
    """
    per_subj = {}            # condition -> list of (n_freqs) subject means
    freqs_ref = None
    for ep in all_epochs:
        data = ep.get_data(copy=False)
        meta = ep.metadata.reset_index(drop=True)
        freqs, psd = compute_psd(data, SFREQ, cfg)
        freqs_ref = freqs
        ch_mean = psd.mean(axis=1)                    # (n_ep, n_freqs)
        for cond in meta['ma_mau'].unique():
            idx = np.where(meta['ma_mau'].values == cond)[0]
            per_subj.setdefault(cond, []).append(ch_mean[idx].mean(axis=0))
    grand = {c: np.mean(np.stack(v), axis=0) for c, v in per_subj.items()}
    return freqs_ref, grand


# ── Topomap helper data ───────────────────────────────────────────────────────
def topomap_band_values(bp_long: pd.DataFrame, condition: str, band: str,
                        value: str = 'rel_power') -> np.ndarray:
    """Channel vector (in EEG_CHANNELS order) of mean band power for a condition."""
    sub = bp_long[(bp_long['ma_mau'] == condition) & (bp_long['band'] == band)]
    per_ch = sub.groupby('channel')[value].mean()
    return np.array([per_ch.get(ch, np.nan) for ch in EEG_CHANNELS])


# ── Time-resolved band power ─────────────────────────────────────────────────
def time_resolved_bandpower(all_epochs: list, cfg: Dict[str, Any],
                            roi: str = 'Frontal') -> pd.DataFrame:
    """Sliding-window band power over the 10 s trial for one ROI.

    Returns long: subject, ma_mau, substance, intensity, band, t_center, power.
    """
    sp = cfg.get('spectral', {})
    win = sp.get('window_seconds', 1.0)
    step = sp.get('window_step', 0.5)
    wn = int(win * SFREQ)
    sn = int(step * SFREQ)
    roi_ch = [EEG_CHANNELS.index(c) for c in ROIS[roi] if c in EEG_CHANNELS]

    rows = []
    for ep in all_epochs:
        data = ep.get_data(copy=False)[:, roi_ch, :]   # (n_ep, n_roi_ch, n_times)
        meta = ep.metadata.reset_index(drop=True)
        n_times = data.shape[-1]
        starts = list(range(0, n_times - wn + 1, sn))
        for st in starts:
            seg = data[:, :, st:st + wn]
            freqs, psd = compute_psd(seg, SFREQ, cfg)
            t_center = (st + wn / 2) / SFREQ
            for b in BAND_ORDER:
                bp = band_power(freqs, psd, FREQ_BANDS[b]).mean(axis=1)  # (n_ep,)
                for i in range(data.shape[0]):
                    m = meta.iloc[i]
                    rows.append({
                        'subject': m['subject'], 'ma_mau': m['ma_mau'],
                        'substance': m['substance'], 'intensity': int(m['intensity']),
                        'band': b, 't_center': t_center, 'power': bp[i],
                    })
    return pd.DataFrame(rows)


def window_bandpower(all_epochs: list, cfg: Dict[str, Any], window: Tuple[float, float],
                     roi: str = 'Frontal') -> pd.DataFrame:
    """Mean band power within a time window (e.g. early vs late) for one ROI.

    Returns long: subject, ma_mau, substance, intensity, band, power.
    """
    roi_ch = [EEG_CHANNELS.index(c) for c in ROIS[roi] if c in EEG_CHANNELS]
    s0, s1 = int(window[0] * SFREQ), int(window[1] * SFREQ)
    rows = []
    for ep in all_epochs:
        data = ep.get_data(copy=False)[:, roi_ch, s0:s1]
        meta = ep.metadata.reset_index(drop=True)
        freqs, psd = compute_psd(data, SFREQ, cfg)
        for b in BAND_ORDER:
            bp = band_power(freqs, psd, FREQ_BANDS[b]).mean(axis=1)
            for i in range(data.shape[0]):
                m = meta.iloc[i]
                rows.append({
                    'subject': m['subject'], 'ma_mau': m['ma_mau'],
                    'substance': m['substance'], 'intensity': int(m['intensity']),
                    'band': b, 'power': bp[i],
                })
    return pd.DataFrame(rows)


# ── TFR ──────────────────────────────────────────────────────────────────────
def compute_tfr_by_substance(all_epochs: list, cfg: Dict[str, Any], roi: str = 'Frontal'
                             ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Morlet TFR averaged within ROI, per substance (Sucrose/Sucralose/Water).

    Returns (freqs, times, {substance: tfr[n_freqs, n_times]}) — power, baseline-free.
    """
    from mne.time_frequency import tfr_array_morlet

    tcfg = cfg.get('tfr', {})
    freqs = np.linspace(tcfg.get('fmin', 2.0), tcfg.get('fmax', 45.0),
                        tcfg.get('n_freqs', 40))
    n_cycles = freqs * tcfg.get('n_cycles_factor', 0.5)
    roi_ch = [EEG_CHANNELS.index(c) for c in ROIS[roi] if c in EEG_CHANNELS]

    acc: Dict[str, list] = {}
    times = None
    for ep in all_epochs:
        data = ep.get_data(copy=False)[:, roi_ch, :]
        meta = ep.metadata.reset_index(drop=True)
        power = tfr_array_morlet(data, sfreq=SFREQ, freqs=freqs, n_cycles=n_cycles,
                                 output='power', verbose=False)  # (n_ep, n_ch, n_f, n_t)
        power = power.mean(axis=1)                                # avg ROI channels
        times = np.arange(power.shape[-1]) / SFREQ
        for sub in meta['substance'].unique():
            idx = np.where(meta['substance'].values == sub)[0]
            acc.setdefault(sub, []).append(power[idx].mean(axis=0))
    grand = {s: np.mean(np.stack(v), axis=0) for s, v in acc.items()}
    return freqs, times, grand
