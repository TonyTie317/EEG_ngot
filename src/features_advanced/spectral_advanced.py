"""
Advanced spectral features: Spectral Edge Frequency, Spectral Centroid,
Band Ratios, 1/f slope (aperiodic fit).
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, List

from ._helpers import get_eeg_channel_names, get_band_mask, make_long_row, safe_divide


def compute_advanced_spectral_features(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    sef_percentiles: List[float] = None,
    band_ratios: List[List[str]] = None,
    compute_aperiodic: bool = True,
) -> pd.DataFrame:
    """
    Compute advanced spectral features per epoch per channel.

    Features:
    - Spectral Edge Frequency (SEF) at given percentiles
    - Spectral Centroid
    - Frequency band power ratios
    - 1/f slope (aperiodic signal fit)

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    sef_percentiles : list of float
        Percentiles for SEF (e.g. [50, 90, 95])
    band_ratios : list of [numerator_band, denominator_band]
        Band pairs for ratio computation
    compute_aperiodic : bool
        Whether to compute 1/f slope

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    if sef_percentiles is None:
        sef_percentiles = [50, 90, 95]
    if band_ratios is None:
        band_ratios = [
            ['theta', 'alpha'], ['alpha', 'beta'],
            ['theta', 'beta'], ['delta', 'alpha'],
            ['gamma', 'beta'],
        ]

    ch_names = get_eeg_channel_names(epochs)
    ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    features_list = []

    for epoch_idx in range(n_epochs):
        for ch_idx, ch_name in zip(ch_indices, ch_names):
            x = data[epoch_idx, ch_idx, :]

            # Compute PSD via Welch
            from scipy.signal import welch
            n_fft = min(128, len(x))
            freqs, psd = welch(x, fs=sfreq, nperseg=n_fft, nfft=n_fft)

            # Only use positive frequencies in range
            valid_mask = (freqs > 0) & (freqs <= sfreq / 2)
            freqs = freqs[valid_mask]
            psd = psd[valid_mask]

            if len(freqs) == 0 or np.sum(psd) == 0:
                continue

            # Spectral Edge Frequency
            cumsum_psd = np.cumsum(psd)
            total_power = cumsum_psd[-1]
            for pct in sef_percentiles:
                threshold = total_power * (pct / 100.0)
                edge_idx = np.searchsorted(cumsum_psd, threshold)
                edge_idx = min(edge_idx, len(freqs) - 1)
                sef = freqs[edge_idx]
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'spectral_adv',
                    f'sef_{int(pct)}_{ch_name}', sef
                ))

            # Spectral Centroid
            centroid = safe_divide(np.sum(freqs * psd), np.sum(psd))
            features_list.append(make_long_row(
                epoch_idx, ch_name, 'spectral_adv',
                f'spectral_centroid_{ch_name}', centroid
            ))

            # Band power ratios
            band_powers = {}
            for band_name, (fmin, fmax) in freq_bands.items():
                band_mask = get_band_mask(freqs, fmin, fmax)
                band_powers[band_name] = np.mean(psd[band_mask]) if np.any(band_mask) else 0.0

            for num_band, den_band in band_ratios:
                if num_band in band_powers and den_band in band_powers:
                    ratio = safe_divide(band_powers[num_band], band_powers[den_band])
                    features_list.append(make_long_row(
                        epoch_idx, ch_name, 'spectral_adv',
                        f'ratio_{num_band}_{den_band}_{ch_name}', ratio
                    ))

            # 1/f slope (aperiodic fit)
            if compute_aperiodic:
                fit_mask = (freqs >= 1.0) & (freqs <= 40.0)
                if np.sum(fit_mask) > 2:
                    fit_freqs = freqs[fit_mask]
                    fit_psd = psd[fit_mask]
                    # Avoid log(0)
                    fit_psd = np.maximum(fit_psd, 1e-20)
                    try:
                        slope, intercept = np.polyfit(np.log(fit_freqs), np.log(fit_psd), 1)
                        features_list.append(make_long_row(
                            epoch_idx, ch_name, 'spectral_adv',
                            f'one_over_f_slope_{ch_name}', slope
                        ))
                        features_list.append(make_long_row(
                            epoch_idx, ch_name, 'spectral_adv',
                            f'one_over_f_offset_{ch_name}', intercept
                        ))
                    except Exception:
                        pass

    return pd.DataFrame(features_list)
