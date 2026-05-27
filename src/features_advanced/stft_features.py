"""
STFT (Short-Time Fourier Transform) features.
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, List
from scipy.signal import stft

from ._helpers import get_eeg_channel_names, get_band_mask, make_long_row


def compute_stft_features(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    window_size: int = 64,
    hop_length: int = 32,
    n_fft: int = 128,
) -> pd.DataFrame:
    """
    Compute STFT-based features per epoch per channel.

    Features per frequency band:
    - Mean power
    - Std power
    - Max power
    Additional:
    - Spectral entropy (mean across time windows)
    - Power variability across time windows
    - Total power mean and std across time

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    window_size : int
        STFT window length in samples
    hop_length : int
        STFT hop length in samples
    n_fft : int
        FFT size

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    ch_names = get_eeg_channel_names(epochs)
    ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    # Ensure window_size doesn't exceed epoch length
    n_times = data.shape[2]
    window_size = min(window_size, n_times)
    hop_length = min(hop_length, window_size)

    features_list = []

    for epoch_idx in range(n_epochs):
        for ch_idx, ch_name in zip(ch_indices, ch_names):
            x = data[epoch_idx, ch_idx, :]

            freqs, times_stft, Zxx = stft(
                x, fs=sfreq, window='hann',
                nperseg=window_size, noverlap=window_size - hop_length,
                nfft=n_fft
            )
            # Power spectrogram: (n_freqs, n_time_windows)
            S = np.abs(Zxx) ** 2

            # Band-specific features
            for band_name, (fmin, fmax) in freq_bands.items():
                band_mask = get_band_mask(freqs, fmin, fmax)
                if not np.any(band_mask):
                    continue

                band_power = S[band_mask, :]  # (n_band_freqs, n_time_windows)
                mean_over_time = np.mean(band_power, axis=1)  # (n_band_freqs,)

                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'stft',
                    f'stf_meanpower_{band_name}_{ch_name}',
                    np.mean(mean_over_time)
                ))
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'stft',
                    f'stf_stdpower_{band_name}_{ch_name}',
                    np.std(mean_over_time)
                ))
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'stft',
                    f'stf_maxpower_{band_name}_{ch_name}',
                    np.max(mean_over_time)
                ))

            # Spectral entropy per time window (averaged)
            entropies = []
            for t in range(S.shape[1]):
                power_t = S[:, t]
                total = np.sum(power_t)
                if total > 0:
                    p = power_t / total
                    p = p[p > 0]
                    entropies.append(-np.sum(p * np.log2(p)))
            mean_entropy = np.mean(entropies) if entropies else 0.0
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'stft',
                f'stf_spec_entropy_{ch_name}', mean_entropy
            ))

            # Power variability across time
            total_power_per_time = np.sum(S, axis=0)  # (n_time_windows,)
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'stft',
                f'stf_power_variability_{ch_name}',
                np.std(total_power_per_time)
            ))
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'stft',
                f'stf_total_power_mean_{ch_name}',
                np.mean(total_power_per_time)
            ))

    return pd.DataFrame(features_list)
