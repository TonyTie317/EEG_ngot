"""
Time-frequency features using Morlet wavelets (MNE TFR)
and Phase-Locking Value (PLV).
"""

import numpy as np
import pandas as pd
import mne
from typing import Dict, Any, List, Tuple, Union

from ._helpers import get_eeg_channel_names, get_roi_pairs, make_long_row


def compute_tfr_features(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    n_cycles: Union[int, List[float]] = 7,
    use_multitaper: bool = False,
) -> pd.DataFrame:
    """
    Compute time-frequency features using Morlet wavelets.

    Features per band per channel:
    - Mean TFR power
    - Peak power (max across time)
    - Power rise rate
    - Power fall rate

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    n_cycles : int or list of float
        Number of cycles for wavelet
    use_multitaper : bool
        Use multitaper instead of Morlet

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    ch_names = get_eeg_channel_names(epochs)
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']

    # Build frequency array for TFR
    all_freqs = np.linspace(1.0, 40.0, 40)

    features_list = []

    for epoch_idx in range(n_epochs):
        single_epoch = epochs[epoch_idx]

        try:
            # Pick only EEG channels
            epoch_eeg = single_epoch.copy().pick_types(eeg=True, exclude=[])
            tfr = epoch_eeg.compute_tfr(
                method='morlet' if not use_multitaper else 'multitaper',
                freqs=all_freqs,
                n_cycles=n_cycles,
                average=False,
                return_itc=False,
            )
        except Exception:
            continue

        # tfr data: (1, n_channels, n_freqs, n_times)
        tfr_data = tfr.data[0]  # (n_channels, n_freqs, n_times)
        tfr_freqs = tfr.freqs
        tfr_times = tfr.times

        epoch_ch_names = epoch_eeg.ch_names

        for ch_i, ch_name in enumerate(epoch_ch_names):
            if ch_name not in ch_names:
                continue

            for band_name, (fmin, fmax) in freq_bands.items():
                freq_mask = (tfr_freqs >= fmin) & (tfr_freqs <= fmax)
                if not np.any(freq_mask):
                    continue

                # Mean power across frequencies -> (n_times,)
                band_power = np.mean(tfr_data[ch_i, freq_mask, :], axis=0)
                mean_power = np.mean(band_power)

                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'timefreq',
                    f'tfr_meanpower_{band_name}_{ch_name}', mean_power
                ))

                # Peak power
                peak_idx = np.argmax(band_power)
                peak_power = band_power[peak_idx]
                features_list.append(make_long_row(
                    epoch_idx, ch_name, 'timefreq',
                    f'tfr_peakpower_{band_name}_{ch_name}', peak_power
                ))

                # Rise rate: (peak - onset) / time_to_peak
                if peak_idx > 0 and len(tfr_times) > 1:
                    dt = tfr_times[1] - tfr_times[0]
                    rise_rate = (peak_power - band_power[0]) / (peak_idx * dt)
                    features_list.append(make_long_row(
                        epoch_idx, ch_name, 'timefreq',
                        f'tfr_riserate_{band_name}_{ch_name}', rise_rate
                    ))

                    # Fall rate: (peak - offset) / time_from_peak
                    offset_power = band_power[-1]
                    fall_samples = len(band_power) - 1 - peak_idx
                    if fall_samples > 0:
                        fall_rate = (peak_power - offset_power) / (fall_samples * dt)
                        features_list.append(make_long_row(
                            epoch_idx, ch_name, 'timefreq',
                            f'tfr_fallrate_{band_name}_{ch_name}', fall_rate
                        ))

    return pd.DataFrame(features_list)


def compute_plv_features(
    epochs: mne.Epochs,
    freq_bands: Dict[str, List[float]],
    channel_pairs: List[Tuple[str, str]] = None,
    pair_strategy: str = 'roi',
) -> pd.DataFrame:
    """
    Compute Phase-Locking Value (PLV) between channel pairs.

    PLV = |mean(exp(1j * (phase_i - phase_j)))| across time.

    Parameters
    ----------
    epochs : mne.Epochs
        EEG epochs
    freq_bands : dict
        Frequency bands {name: [fmin, fmax]}
    channel_pairs : list of (ch_a, ch_b)
        Specific pairs to compute
    pair_strategy : str
        'roi' for ROI-based pairs, 'full' for all pairs

    Returns
    -------
    pd.DataFrame with long-format feature rows
    """
    from scipy.signal import butter, filtfilt, hilbert

    ch_names = get_eeg_channel_names(epochs)
    ch_indices = {ch: epochs.ch_names.index(ch) for ch in ch_names}
    n_epochs = len(epochs)
    sfreq = epochs.info['sfreq']
    data = epochs.get_data()

    if channel_pairs is None:
        if pair_strategy == 'roi':
            channel_pairs = get_roi_pairs(ch_names)
        else:
            from ._helpers import get_channel_pairs
            channel_pairs = get_channel_pairs(ch_names)

    features_list = []

    for epoch_idx in range(n_epochs):
        for band_name, (fmin, fmax) in freq_bands.items():
            # Design bandpass filter
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

                sig_a = data[epoch_idx, ch_indices[ch_a], :]
                sig_b = data[epoch_idx, ch_indices[ch_b], :]

                try:
                    filt_a = filtfilt(b, a, sig_a)
                    filt_b = filtfilt(b, a, sig_b)
                    phase_a = np.angle(hilbert(filt_a))
                    phase_b = np.angle(hilbert(filt_b))
                    plv = np.abs(np.mean(np.exp(1j * (phase_a - phase_b))))
                except Exception:
                    continue

                features_list.append(make_long_row(
                    epoch_idx, f'{ch_a}-{ch_b}', 'timefreq',
                    f'tfr_plv_{band_name}_{ch_a}_vs_{ch_b}', plv
                ))

    return pd.DataFrame(features_list)
