"""
Wavelet transform features: DWT (Discrete Wavelet Transform) and
CWT (Continuous Wavelet Transform).
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, List, Optional

from ._helpers import get_eeg_channel_names, get_band_mask, make_long_row


def _shannon_entropy(coeffs: np.ndarray) -> float:
    """Compute Shannon entropy of wavelet coefficients."""
    power = coeffs ** 2
    total = np.sum(power)
    if total == 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_dwt_features(
    epochs: mne.Epochs,
    wavelet: str = 'db4',
    max_level: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute DWT-based features per epoch per channel.

    For each decomposition level (detail coefficients cD_i):
    - Mean, Std, Energy, Shannon Entropy
    For final approximation (cA_n):
    - Energy, Entropy

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    wavelet : str
        Wavelet name (e.g. 'db4')
    max_level : int or None
        Max decomposition level (auto if None)

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    try:
        import pywt
    except ImportError:
        import warnings
        warnings.warn("PyWavelets not installed, skipping DWT features. Install with: pip install PyWavelets")
        return pd.DataFrame()

    ch_names = get_eeg_channel_names(epochs)
    ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]
    n_epochs = len(epochs)
    data = epochs.get_data()
    n_times = data.shape[2]

    # Auto-compute max level
    if max_level is None:
        max_level = pywt.dwt_max_level(n_times, pywt.Wavelet(wavelet).dec_len)
    max_level = min(max_level, 7)  # Cap at 7 for stability

    features_list = []

    for epoch_idx in range(n_epochs):
        for ch_idx, ch_name in zip(ch_indices, ch_names):
            x = data[epoch_idx, ch_idx, :]

            try:
                coeffs = pywt.wavedec(x, wavelet, level=max_level)
            except Exception:
                continue

            # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
            # Detail coefficients: coeffs[1:] = cD_n ... cD_1
            for level_idx, cD in enumerate(coeffs[1:], start=1):
                level_name = f'level{level_idx}'
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'wavelet',
                    f'dwt_mean_{level_name}_{ch_name}', np.mean(cD)
                ))
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'wavelet',
                    f'dwt_std_{level_name}_{ch_name}', np.std(cD)
                ))
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'wavelet',
                    f'dwt_energy_{level_name}_{ch_name}', np.sum(cD ** 2)
                ))
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'wavelet',
                    f'dwt_entropy_{level_name}_{ch_name}', _shannon_entropy(cD)
                ))

            # Approximation coefficients
            cA = coeffs[0]
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'wavelet',
                f'dwt_approx_energy_{ch_name}', np.sum(cA ** 2)
            ))
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'wavelet',
                f'dwt_approx_entropy_{ch_name}', _shannon_entropy(cA)
            ))

    return pd.DataFrame(features_list)


def compute_cwt_features(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    wavelet: str = 'cmor1.5-1.0',
) -> pd.DataFrame:
    """
    Compute CWT-based features per epoch per channel.

    Features:
    - Mean/max scalogram power per frequency band
    - Wavelet entropy (across all scales)
    - Total wavelet energy

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    wavelet : str
        CWT wavelet name (default: complex Morlet)

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    try:
        import pywt
    except ImportError:
        return pd.DataFrame()

    ch_names = get_eeg_channel_names(epochs)
    ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    # Generate scales from frequency range 0.5-40 Hz
    # For cmor1.5-1.0: central_freq = 1.5, scale = central_freq * sfreq / freq
    central_freq = 1.5  # default for cmor1.5-1.0
    freqs_of_interest = np.linspace(0.5, 40.0, 50)
    scales = central_freq * sfreq / freqs_of_interest

    features_list = []

    for epoch_idx in range(n_epochs):
        for ch_idx, ch_name in zip(ch_indices, ch_names):
            x = data[epoch_idx, ch_idx, :]

            try:
                coeffs, freqs = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / sfreq)
            except Exception:
                continue

            # Scalogram power: (n_scales, n_times)
            power = np.abs(coeffs) ** 2

            # Band-specific features
            for band_name, (fmin, fmax) in freq_bands.items():
                band_mask = get_band_mask(freqs, fmin, fmax)
                if not np.any(band_mask):
                    continue
                band_power = power[band_mask, :]
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'wavelet',
                    f'cwt_meanpower_{band_name}_{ch_name}',
                    np.mean(band_power)
                ))
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'wavelet',
                    f'cwt_maxpower_{band_name}_{ch_name}',
                    np.max(band_power)
                ))

            # Wavelet entropy
            total_energy = np.sum(power)
            if total_energy > 0:
                p = power.flatten() / total_energy
                p = p[p > 0]
                wavelet_entropy = -np.sum(p * np.log2(p))
            else:
                wavelet_entropy = 0.0

            features_list.append(make_long_row(
                epoch_idx, ch_name, 'wavelet',
                f'cwt_entropy_{ch_name}', wavelet_entropy
            ))
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'wavelet',
                f'cwt_total_energy_{ch_name}', total_energy
            ))

    return pd.DataFrame(features_list)
