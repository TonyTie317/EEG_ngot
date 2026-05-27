"""
Connectivity features: Coherence, PLV, Correlation matrix statistics.
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, List, Tuple
from scipy.signal import coherence as scipy_coherence, butter, filtfilt, hilbert

from ._helpers import (
    get_eeg_channel_names, get_channel_pairs, get_roi_pairs,
    make_long_row
)


def compute_coherence_features(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    channel_pairs: List[Tuple[str, str]] = None,
    pair_strategy: str = 'roi',
) -> pd.DataFrame:
    """
    Compute coherence between channel pairs per frequency band.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    channel_pairs : list of (ch_a, ch_b)
        Specific channel pairs
    pair_strategy : str
        'roi', 'full', or 'inter_roi'

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    ch_names = get_eeg_channel_names(epochs)
    ch_indices = {ch: epochs.ch_names.index(ch) for ch in ch_names}
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    if channel_pairs is None:
        if pair_strategy == 'roi':
            channel_pairs = get_roi_pairs(ch_names)
        else:
            channel_pairs = get_channel_pairs(ch_names)

    features_list = []

    for epoch_idx in range(n_epochs):
        for band_name, (fmin, fmax) in freq_bands.items():
            for ch_a, ch_b in channel_pairs:
                if ch_a not in ch_indices or ch_b not in ch_indices:
                    continue

                idx_a = ch_indices[ch_a]
                idx_b = ch_indices[ch_b]

                try:
                    nperseg = min(64, data.shape[2])
                    freqs_coh, coh = scipy_coherence(
                        data[epoch_idx, idx_a, :],
                        data[epoch_idx, idx_b, :],
                        fs=sfreq, nperseg=nperseg
                    )
                    band_mask = (freqs_coh >= fmin) & (freqs_coh <= fmax)
                    if np.any(band_mask):
                        mean_coh = np.mean(coh[band_mask])
                        features_list.append(make_long_row(
                            epoch_idx, f'{ch_a}-{ch_b}', 'connectivity',
                            f'coherence_{band_name}_{ch_a}_vs_{ch_b}', mean_coh
                        ))
                except Exception:
                    continue

    return pd.DataFrame(features_list)


def compute_plv_connectivity(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    channel_pairs: List[Tuple[str, str]] = None,
    pair_strategy: str = 'roi',
) -> pd.DataFrame:
    """
    Compute PLV (Phase-Locking Value) connectivity between channel pairs.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    channel_pairs : list of (ch_a, ch_b)
        Channel pairs
    pair_strategy : str
        'roi' or 'full'

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    ch_names = get_eeg_channel_names(epochs)
    ch_indices = {ch: epochs.ch_names.index(ch) for ch in ch_names}
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    if channel_pairs is None:
        if pair_strategy == 'roi':
            channel_pairs = get_roi_pairs(ch_names)
        else:
            channel_pairs = get_channel_pairs(ch_names)

    features_list = []

    for epoch_idx in range(n_epochs):
        for band_name, (fmin, fmax) in freq_bands.items():
            nyq = sfreq / 2.0
            low = max(fmin / nyq, 0.001)
            high = min(fmax / nyq, 0.999)
            if low >= high:
                continue
            try:
                b, a = butter(4, [low, high], btype='band')
            except Exception:
                continue

            for ch_a, ch_b in channel_pairs:
                if ch_a not in ch_indices or ch_b not in ch_indices:
                    continue

                try:
                    sig_a = filtfilt(b, a, data[epoch_idx, ch_indices[ch_a], :])
                    sig_b = filtfilt(b, a, data[epoch_idx, ch_indices[ch_b], :])
                    phase_a = np.angle(hilbert(sig_a))
                    phase_b = np.angle(hilbert(sig_b))
                    plv = np.abs(np.mean(np.exp(1j * (phase_a - phase_b))))
                except Exception:
                    continue

                features_list.append(make_long_row(
                    epoch_idx, f'{ch_a}-{ch_b}', 'connectivity',
                    f'plv_{band_name}_{ch_a}_vs_{ch_b}', plv
                ))

    return pd.DataFrame(features_list)


def compute_correlation_features(
    epochs: mne.Epochs,
    rois: Dict[str, List[str]] = None,
) -> pd.DataFrame:
    """
    Compute correlation matrix statistics across channels.

    Features:
    - Mean, std, min, max of off-diagonal correlation
    - Mean correlation per ROI

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    rois : dict
        ROI definitions {name: [channels]}

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    if rois is None:
        rois = {
            'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
            'central': ['C3', 'C4'],
            'parietal': ['P3', 'P4'],
            'occipital': ['O1', 'O2'],
        }

    ch_names = get_eeg_channel_names(epochs)
    ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]
    n_epochs = len(epochs)
    data = epochs.get_data()

    features_list = []

    for epoch_idx in range(n_epochs):
        # Correlation matrix across channels
        epoch_data = data[epoch_idx, ch_indices, :]
        n_ch = len(ch_indices)

        if n_ch < 2:
            continue

        corr_matrix = np.corrcoef(epoch_data)  # (n_ch, n_ch)

        # Upper triangle (off-diagonal)
        upper_indices = np.triu_indices(n_ch, k=1)
        off_diag = corr_matrix[upper_indices]

        features_list.append(make_long_row(
            epoch_idx, 'global', 'connectivity',
            'corr_mean', np.mean(off_diag)
        ))
        features_list.append(make_long_row(
            epoch_idx, 'global', 'connectivity',
            'corr_std', np.std(off_diag)
        ))
        features_list.append(make_long_row(
            epoch_idx, 'global', 'connectivity',
            'corr_min', np.min(off_diag)
        ))
        features_list.append(make_long_row(
            epoch_idx, 'global', 'connectivity',
            'corr_max', np.max(off_diag)
        ))

        # Per-ROI mean correlation
        for roi_name, roi_channels in rois.items():
            roi_chs = [ch for ch in roi_channels if ch in ch_names]
            if len(roi_chs) < 2:
                continue
            roi_idx = [ch_names.index(ch) for ch in roi_chs]
            roi_corr = corr_matrix[np.ix_(roi_idx, roi_idx)]
            roi_upper = roi_corr[np.triu_indices(len(roi_idx), k=1)]
            features_list.append(make_long_row(
                epoch_idx, roi_name, 'connectivity',
                f'corr_{roi_name}', np.mean(roi_upper)
            ))

    return pd.DataFrame(features_list)
