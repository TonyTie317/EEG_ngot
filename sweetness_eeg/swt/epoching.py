"""
Epoching: slice the 10 s tasting windows from the preprocessed continuous Raw.

Each trial is a contiguous 1000-sample (10 s) run of one blind code. We build an
``mne.EpochsArray`` per subject with rich metadata (substance, intensity, repeat,
code, subject) so every downstream analysis can group by condition. Caching to
``results/epochs/`` keeps re-runs fast.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ensure_dir, result_path
from .constants import EEG_CHANNELS, SFREQ, TRIAL_SAMPLES
from . import loader, preprocess


def make_epochs(subject_data: Dict[str, Any], config: Dict[str, Any],
                logger: logging.Logger):
    """Build an EpochsArray for one preprocessed subject.

    Returns (epochs, metadata_df). Trials shorter than TRIAL_SAMPLES are skipped.
    Peak-to-peak rejection (config epoching.reject_uv) drops noisy trials.
    """
    import mne

    raw = subject_data['raw']
    trials = subject_data['trials']
    sid = subject_data['subject_id']
    data = raw.get_data()                      # (n_ch, n_times), volts
    n_samples = TRIAL_SAMPLES

    segs, meta_rows = [], []
    for _, t in trials.iterrows():
        s0 = int(t['start'])
        s1 = s0 + n_samples
        if s1 > data.shape[1]:
            continue
        segs.append(data[:, s0:s1])
        meta_rows.append({
            'subject': sid, 'code': int(t['code']), 'ma_mau': t['ma_mau'],
            'substance': t['substance'], 'intensity': int(t['intensity']),
            'repeat': int(t['repeat']),
        })

    if not segs:
        return None, pd.DataFrame()

    X = np.stack(segs, axis=0)                  # (n_trials, n_ch, n_samples)
    meta = pd.DataFrame(meta_rows)

    info = mne.create_info(list(EEG_CHANNELS), SFREQ, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                     match_case=False, on_missing='ignore', verbose=False)
    epochs = mne.EpochsArray(X, info, tmin=config['epoching'].get('tmin', 0.0),
                             verbose=False)
    epochs.metadata = meta

    reject_uv = config['epoching'].get('reject_uv')
    if reject_uv:
        ptp = (X.max(axis=2) - X.min(axis=2)).max(axis=1)   # peak-to-peak (V) per trial
        keep = ptp < reject_uv * 1e-6
        n_drop = int((~keep).sum())
        if n_drop:
            epochs = epochs[np.where(keep)[0]]
            meta = meta.iloc[np.where(keep)[0]].reset_index(drop=True)
            epochs.metadata = meta
            logger.info(f"[{sid}] dropped {n_drop}/{len(X)} trials (>{reject_uv}µV p-p)")
    logger.info(f"[{sid}] {len(epochs)} epochs kept")
    return epochs, meta


def build_all_epochs(config: Dict[str, Any], logger: logging.Logger,
                     subjects: Optional[List[str]] = None,
                     cache: bool = True) -> Tuple[list, pd.DataFrame]:
    """Load → preprocess → epoch every subject. Returns (list_of_epochs, all_metadata)."""
    from .constants import ALL_SUBJECTS
    subjects = subjects or ALL_SUBJECTS
    cache_dir = ensure_dir(result_path(config, 'epochs'))

    all_epochs, metas = [], []
    for sid in subjects:
        fif = os.path.join(cache_dir, f'{sid}-epo.fif')
        if cache and os.path.exists(fif):
            import mne
            ep = mne.read_epochs(fif, verbose=False)
            all_epochs.append(ep)
            metas.append(ep.metadata)
            logger.info(f"[{sid}] loaded cached epochs ({len(ep)})")
            continue
        try:
            sd = loader.load_subject(sid, config['paths']['raw_data'], logger)
            sd['raw'] = preprocess.preprocess_raw(sd['raw'], config, logger)
            ep, meta = make_epochs(sd, config, logger)
            if ep is None:
                continue
            if cache:
                ep.save(fif, overwrite=True, verbose=False)
            all_epochs.append(ep)
            metas.append(meta)
        except Exception as e:                          # noqa: BLE001
            logger.error(f"[{sid}] epoching failed: {e}")

    all_meta = pd.concat(metas, ignore_index=True) if metas else pd.DataFrame()
    logger.info(f"TOTAL: {len(all_epochs)} subjects, {len(all_meta)} epochs")
    return all_epochs, all_meta


def load_all_epochs(config: Dict[str, Any], logger: logging.Logger):
    """Load cached epochs only (used by analysis stages)."""
    return build_all_epochs(config, logger, cache=True)
