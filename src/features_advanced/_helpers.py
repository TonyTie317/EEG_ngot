"""
Shared utilities for advanced feature extraction.
"""

import numpy as np
from typing import List, Tuple, Dict
import mne


def get_eeg_channel_names(epochs: mne.Epochs) -> List[str]:
    """Return channel names excluding ECG."""
    return [ch for ch in epochs.ch_names if not ch.upper().startswith('ECG')]


def get_channel_pairs(ch_names: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique unordered channel pairs."""
    pairs = []
    for i in range(len(ch_names)):
        for j in range(i + 1, len(ch_names)):
            pairs.append((ch_names[i], ch_names[j]))
    return pairs


def get_band_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Return boolean mask for frequency array within [fmin, fmax]."""
    return (freqs >= fmin) & (freqs <= fmax)


def make_long_row(epoch_idx: int, channel: str, feature_type: str,
                  feature_name: str, value: float) -> dict:
    """Create one row dict for long-format DataFrame."""
    return {
        'epoch_idx': epoch_idx,
        'channel': channel,
        'feature_type': feature_type,
        'feature_name': feature_name,
        'value': float(value)
    }


def get_hemisphere_pairs() -> Dict[str, Tuple[List[str], List[str]]]:
    """Return left/right hemisphere channel groupings per ROI."""
    return {
        'frontal': (['F3', 'F7'], ['F4', 'F8']),
        'central': (['C3'], ['C4']),
        'parietal': (['P3', 'P7'], ['P4', 'P8']),
        'occipital': (['O1'], ['O2']),
        'prefrontal': (['Fp1'], ['Fp2']),
        'temporal': (['T7'], ['T8']),
    }


def get_roi_pairs(ch_names: List[str]) -> List[Tuple[str, str]]:
    """Generate channel pairs based on ROI proximity for connectivity features."""
    rois = {
        'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
        'central': ['C3', 'C4'],
        'parietal': ['P3', 'P4'],
        'occipital': ['O1', 'O2'],
        'temporal': ['T7', 'T8'],
    }

    pairs = []

    # Intra-ROI pairs
    for roi_channels in rois.values():
        for i in range(len(roi_channels)):
            for j in range(i + 1, len(roi_channels)):
                if roi_channels[i] in ch_names and roi_channels[j] in ch_names:
                    pairs.append((roi_channels[i], roi_channels[j]))

    # Homologous inter-hemisphere pairs
    homologous = [
        ('Fp1', 'Fp2'), ('F3', 'F4'), ('F7', 'F8'),
        ('C3', 'C4'), ('T7', 'T8'),
        ('P3', 'P4'), ('P7', 'P8'),
        ('O1', 'O2'),
    ]
    for left, right in homologous:
        if left in ch_names and right in ch_names:
            if (left, right) not in pairs and (right, left) not in pairs:
                pairs.append((left, right))

    return pairs


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns default on zero denominator."""
    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if np.isfinite(result) else default
