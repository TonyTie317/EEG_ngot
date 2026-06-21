"""
Preprocessing for the 10 s gustatory tasting windows.

Order: pick EEG → average reference → notch (mains) → band-pass (0.5–45 Hz) →
optional ICA (Fp1/Fp2 EOG proxy). Filtering happens on the continuous Raw before
epoching. Adapted from ``pipeline/preprocess.py`` (kept independent).
"""

import logging
from typing import Any, Dict

import mne


def preprocess_raw(raw, config: Dict[str, Any], logger: logging.Logger):
    """Apply the full preprocessing chain to a continuous RawArray (in place copy)."""
    prep = config['preprocessing']
    raw = raw.copy()

    raw.set_eeg_reference(prep.get('reference', 'average'),
                          projection=False, verbose=False)

    notch = prep.get('notch_freq')
    nyq = raw.info['sfreq'] / 2.0
    if notch and notch < nyq:
        raw.notch_filter([notch], verbose=False)
    elif notch:
        logger.warning(f"  notch {notch} Hz >= Nyquist {nyq} Hz — skipped")

    l_freq = prep.get('l_freq', 0.5)
    h_freq = min(prep.get('h_freq', 45), nyq - 0.5)
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)

    ica_cfg = prep.get('ica', {})
    if ica_cfg.get('enabled', False):
        raw = _apply_ica(raw, ica_cfg, logger)

    return raw


def _apply_ica(raw, ica_config: Dict[str, Any], logger: logging.Logger):
    """ICA artifact removal using Fp1/Fp2 as an EOG proxy."""
    try:
        ica = mne.preprocessing.ICA(
            n_components=ica_config.get('n_components', 15),
            method=ica_config.get('method', 'picard'),
            max_iter=ica_config.get('max_iter', 512),
            random_state=ica_config.get('random_state', 42),
            verbose=False,
        )
        ica.fit(raw, verbose=False)
        exclude = []
        if ica_config.get('auto_exclude_eog', True):
            eog_ch = [c for c in raw.ch_names if c.upper() in ('FP1', 'FP2')]
            if eog_ch:
                idx, _ = ica.find_bads_eog(
                    raw, ch_name=eog_ch[0],
                    threshold=ica_config.get('eog_threshold', 2.5), verbose=False)
                exclude.extend(idx)
        exclude = sorted(set(exclude))
        if exclude:
            raw = ica.apply(raw, exclude=exclude, verbose=False)
            logger.info(f"  ICA removed components {exclude}")
        else:
            logger.info("  ICA: no artifact components detected")
    except Exception as e:                              # noqa: BLE001
        logger.warning(f"  ICA failed ({e}); continuing without ICA")
    return raw
