"""
Alpha band analysis features: alpha peak frequency, relative power,
frontal asymmetry, alpha coherence.
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, List, Tuple
from scipy.signal import welch, coherence as scipy_coherence

from ._helpers import (
    get_eeg_channel_names, get_band_mask, make_long_row,
    get_hemisphere_pairs, safe_divide
)


def compute_alpha_features(
    epochs: mne.Epochs,
    alpha_band: List[float] = None,
    frontal_pairs: List[Tuple[str, str]] = None,
    compute_coherence: bool = True,
) -> pd.DataFrame:
    """
    Compute alpha-band-specific features.

    Features:
    - Alpha peak frequency per channel
    - Alpha absolute power per channel
    - Alpha relative power (alpha/total) per channel
    - Alpha suppression index per channel
    - Frontal alpha asymmetry (ln(R) - ln(L))
    - Alpha coherence for frontal pairs

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    alpha_band : list of [fmin, fmax]
        Alpha frequency range
    frontal_pairs : list of (left_ch, right_ch)
        Channel pairs for asymmetry
    compute_coherence : bool
        Whether to compute alpha coherence

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    if alpha_band is None:
        alpha_band = [8, 12]
    if frontal_pairs is None:
        frontal_pairs = [('F3', 'F4'), ('F7', 'F8')]

    ch_names = get_eeg_channel_names(epochs)
    ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    features_list = []

    for epoch_idx in range(n_epochs):
        alpha_powers = {}

        for ch_idx, ch_name in zip(ch_indices, ch_names):
            x = data[epoch_idx, ch_idx, :]
            n_fft = min(128, len(x))
            freqs, psd = welch(x, fs=sfreq, nperseg=n_fft, nfft=n_fft)

            # Alpha band
            alpha_mask = get_band_mask(freqs, alpha_band[0], alpha_band[1])
            alpha_power = np.mean(psd[alpha_mask]) if np.any(alpha_mask) else 0.0
            alpha_powers[ch_name] = alpha_power

            # Alpha peak frequency
            if np.any(alpha_mask):
                alpha_freqs = freqs[alpha_mask]
                alpha_psd = psd[alpha_mask]
                peak_freq = alpha_freqs[np.argmax(alpha_psd)]
            else:
                peak_freq = 0.0

            features_list.append(make_long_row(
                epoch_idx, ch_name, 'alpha',
                f'alpha_peak_freq_{ch_name}', peak_freq
            ))
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'alpha',
                f'alpha_power_{ch_name}', alpha_power
            ))

            # Total power (0.5-40 Hz)
            total_mask = get_band_mask(freqs, 0.5, 40.0)
            total_power = np.mean(psd[total_mask]) if np.any(total_mask) else 1e-20

            # Relative power
            relative_power = safe_divide(alpha_power, total_power)
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'alpha',
                f'alpha_relative_{ch_name}', relative_power
            ))

            # Alpha suppression index
            suppression = 1.0 - relative_power
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'alpha',
                f'alpha_suppression_{ch_name}', suppression
            ))

        # Frontal alpha asymmetry: ln(R) - ln(L)
        for left_ch, right_ch in frontal_pairs:
            if left_ch in alpha_powers and right_ch in alpha_powers:
                left_power = max(alpha_powers[left_ch], 1e-20)
                right_power = max(alpha_powers[right_ch], 1e-20)
                asymmetry = np.log(right_power) - np.log(left_power)
                features_list.append(make_long_row(
                    epoch_idx, f'{left_ch}-{right_ch}', 'alpha',
                    f'alpha_asymmetry_{left_ch}_{right_ch}', asymmetry
                ))

        # Parietal and occipital asymmetry
        for left_ch, right_ch in [('P3', 'P4'), ('O1', 'O2'), ('P7', 'P8')]:
            if left_ch in alpha_powers and right_ch in alpha_powers:
                left_power = max(alpha_powers[left_ch], 1e-20)
                right_power = max(alpha_powers[right_ch], 1e-20)
                asymmetry = np.log(right_power) - np.log(left_power)
                features_list.append(make_long_row(
                    epoch_idx, f'{left_ch}-{right_ch}', 'alpha',
                    f'alpha_asymmetry_{left_ch}_{right_ch}', asymmetry
                ))

        # Alpha coherence for frontal channel pairs
        if compute_coherence:
            frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8']
            frontal_avail = [ch for ch in frontal_channels if ch in ch_names]
            coh_pairs = []
            for i in range(len(frontal_avail)):
                for j in range(i + 1, len(frontal_avail)):
                    coh_pairs.append((frontal_avail[i], frontal_avail[j]))

            for ch_a, ch_b in coh_pairs:
                idx_a = epochs.ch_names.index(ch_a)
                idx_b = epochs.ch_names.index(ch_b)
                freqs_coh, coh = scipy_coherence(
                    data[epoch_idx, idx_a, :],
                    data[epoch_idx, idx_b, :],
                    fs=sfreq, nperseg=min(64, len(x))
                )
                alpha_coh_mask = get_band_mask(freqs_coh, alpha_band[0], alpha_band[1])
                if np.any(alpha_coh_mask):
                    mean_coh = np.mean(coh[alpha_coh_mask])
                    features_list.append(make_long_row(
                        epoch_idx, f'{ch_a}-{ch_b}', 'alpha',
                        f'alpha_coherence_{ch_a}_{ch_b}', mean_coh
                    ))

    return pd.DataFrame(features_list)
