"""
Preprocessing — referencing, filtering, optional ICA artifact removal.

Pipeline order:
1. Pick EEG channels only (drop ECG)
2. Set average reference
3. Notch filter (50 Hz)
4. Bandpass filter (0.5–45 Hz)
5. Optional ICA for EOG/ECG artifact removal
"""

import logging
from typing import Any, Dict, List

import mne


def preprocess_raw(raw: mne.io.RawArray, config: Dict[str, Any],
                   logger: logging.Logger) -> mne.io.RawArray:
    """Apply full preprocessing pipeline to a Raw object.

    Parameters
    ----------
    raw : mne.io.RawArray
        Raw EEG data (with EEG + ECG channels).
    config : dict
        Configuration with 'preprocessing' section.
    logger : logging.Logger

    Returns
    -------
    raw_clean : mne.io.RawArray
        Preprocessed data (EEG channels only, ECG dropped after ICA).
    """
    prep = config['preprocessing']
    raw = raw.copy()

    # 1. Pick EEG channels
    eeg_picks = mne.pick_types(raw.info, eeg=True, ecg=False, exclude=[])
    raw.pick([raw.ch_names[i] for i in eeg_picks])
    logger.info(f"  Selected {len(raw.ch_names)} EEG channels")

    # 2. Set average reference
    ref = prep.get('reference', 'average')
    raw.set_eeg_reference(ref, projection=False, verbose=False)
    logger.info(f"  Set reference: {ref}")

    # 3. Notch filter
    notch_freq = prep.get('notch_freq')
    nyquist = raw.info['sfreq'] / 2.0
    if notch_freq:
        if notch_freq >= nyquist:
            logger.warning(
                f"  Notch freq {notch_freq} Hz >= Nyquist {nyquist} Hz, skipping notch filter"
            )
        else:
            raw.notch_filter([notch_freq], verbose=False)
            logger.info(f"  Notch filter: {notch_freq} Hz")

    # 4. Bandpass filter
    l_freq = prep.get('l_freq', 0.5)
    h_freq = prep.get('h_freq', 45)
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    logger.info(f"  Bandpass filter: {l_freq}-{h_freq} Hz")

    # 5. ICA
    ica_cfg = prep.get('ica', {})
    if ica_cfg.get('enabled', False):
        raw = apply_ica(raw, ica_cfg, logger)

    return raw


def apply_ica(raw: mne.io.RawArray, ica_config: Dict[str, Any],
              logger: logging.Logger) -> mne.io.RawArray:
    """Apply ICA for EOG/ECG artifact removal.

    Uses Fp1/Fp2 as EOG proxy channels if auto_exclude_eog is True.
    Fits ICA with configurable method (default: picard).

    Parameters
    ----------
    raw : mne.io.RawArray
        Pre-filtered EEG data.
    ica_config : dict
        ICA configuration section.
    logger : logging.Logger

    Returns
    -------
    raw_clean : mne.io.RawArray
        Data with artifact components removed.
    """
    n_components = ica_config.get('n_components', 15)
    method = ica_config.get('method', 'picard')
    max_iter = ica_config.get('max_iter', 512)
    random_state = ica_config.get('random_state', 42)

    try:
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=method,
            max_iter=max_iter,
            random_state=random_state,
            verbose=False,
        )
        ica.fit(raw, verbose=False)
        logger.info(f"  ICA fitted: {n_components} components ({method})")

        # Auto-detect EOG artifacts using Fp1/Fp2
        exclude_idx = []
        if ica_config.get('auto_exclude_eog', True):
            eog_ch = [ch for ch in raw.ch_names if ch.upper() in ('FP1', 'FP2')]
            if eog_ch:
                eog_indices, _ = ica.find_bads_eog(
                    raw, ch_name=eog_ch[0],
                    threshold=ica_config.get('eog_threshold', 3.0),
                    verbose=False,
                )
                exclude_idx.extend(eog_indices)
                if eog_indices:
                    logger.info(f"  ICA excluded EOG components: {eog_indices}")

        # Auto-detect ECG artifacts
        # (Note: ECG channels were already dropped, so this relies on
        #  cross-correlation with heartbeat pattern in EEG)
        if ica_config.get('auto_exclude_ecg', False):
            try:
                ecg_indices, _ = ica.find_bads_ecg(
                    raw,
                    threshold=ica_config.get('ecg_threshold', 3.0),
                    verbose=False,
                )
                exclude_idx.extend(ecg_indices)
                if ecg_indices:
                    logger.info(f"  ICA excluded ECG components: {ecg_indices}")
            except Exception as ecg_err:
                logger.debug(f"  ICA ECG detection skipped: {ecg_err}")

        # Deduplicate
        exclude_idx = list(set(exclude_idx))
        if exclude_idx:
            raw = ica.apply(raw, exclude=exclude_idx, verbose=False)
            logger.info(f"  ICA removed {len(exclude_idx)} artifact components")
        else:
            logger.info("  ICA: no artifact components found, data unchanged")

    except Exception as e:
        logger.warning(f"  ICA failed: {e}. Skipping ICA.")

    return raw


def preprocess_all_subjects(subjects_data: List[Dict[str, Any]],
                            config: Dict[str, Any],
                            logger: logging.Logger) -> List[Dict[str, Any]]:
    """Preprocess all subjects in-place.

    Parameters
    ----------
    subjects_data : list of dict
        Each dict must have 'raw' (mne.io.RawArray) and 'subject_id'.
    config : dict
    logger : logging.Logger

    Returns
    -------
    subjects_data : list of dict
        Same list with 'raw' replaced by preprocessed versions.
    """
    for sdata in subjects_data:
        sid = sdata['subject_id']
        logger.info(f"[{sid}] Preprocessing...")
        try:
            sdata['raw'] = preprocess_raw(sdata['raw'], config, logger)
        except Exception as e:
            logger.error(f"[{sid}] Preprocessing failed: {e}")

    return subjects_data
