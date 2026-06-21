"""
Functional connectivity between the Frontal and Gustatory ROIs.

Magnitude-squared coherence (Welch) between ROI-averaged signals, integrated per
frequency band, compared between sucrose and sucralose. Supports the reward /
fronto-insular communication hypothesis from the reference paper.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import coherence

from .constants import BAND_ORDER, EEG_CHANNELS, FREQ_BANDS, ROIS, SFREQ


def roi_pair_coherence(all_epochs: list, cfg: Dict[str, Any],
                       roi_a: str = 'Frontal', roi_b: str = 'Gustatory',
                       logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Per-trial band-wise coherence between two ROIs (ROI-averaged signals).

    Returns long: subject, ma_mau, substance, intensity, band, coherence.
    """
    ia = [EEG_CHANNELS.index(c) for c in ROIS[roi_a] if c in EEG_CHANNELS]
    ib = [EEG_CHANNELS.index(c) for c in ROIS[roi_b] if c in EEG_CHANNELS]
    sp = cfg.get('spectral', {})
    nperseg = min(int(sp.get('welch_seconds', 2.0) * SFREQ), 256)

    rows = []
    for ep in all_epochs:
        data = ep.get_data(copy=False)
        meta = ep.metadata.reset_index(drop=True)
        sig_a = data[:, ia, :].mean(axis=1)              # (n_ep, n_times)
        sig_b = data[:, ib, :].mean(axis=1)
        for i in range(data.shape[0]):
            f, cxy = coherence(sig_a[i], sig_b[i], fs=SFREQ, nperseg=nperseg)
            m = meta.iloc[i]
            for b in BAND_ORDER:
                lo, hi = FREQ_BANDS[b]
                mask = (f >= lo) & (f <= hi)
                rows.append({
                    'subject': m['subject'], 'ma_mau': m['ma_mau'],
                    'substance': m['substance'], 'intensity': int(m['intensity']),
                    'band': b, 'coherence': float(np.nanmean(cxy[mask])),
                })
    df = pd.DataFrame(rows)
    if logger:
        logger.info(f"Connectivity ({roi_a}-{roi_b}): {len(df)} rows")
    return df
