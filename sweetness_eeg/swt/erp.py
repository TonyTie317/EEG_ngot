"""
Onset-locked ERP analysis (supplementary).

The 10 s trial begins at cup delivery / start of tasting. We re-epoch the
preprocessed continuous signal around that onset with a short pre-onset baseline,
then compute condition grand-averages and simple component peaks. This is an
exploratory addition — the continuous tasting paradigm has no sharp sensory
trigger, so ERP results are interpreted cautiously.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import EEG_CHANNELS, SFREQ
from . import loader, preprocess


def build_onset_epochs(cfg: Dict[str, Any], logger: logging.Logger,
                       subjects: Optional[List[str]] = None,
                       tmin: float = -0.5, tmax: float = 2.0,
                       baseline=(-0.5, -0.2)) -> list:
    """Epoch the preprocessed signal around each trial onset (with baseline)."""
    import mne
    from .constants import ALL_SUBJECTS
    subjects = subjects or ALL_SUBJECTS
    pre = int(abs(tmin) * SFREQ)
    post = int(tmax * SFREQ)

    info = mne.create_info(list(EEG_CHANNELS), SFREQ, ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1020'),
                     match_case=False, on_missing='ignore', verbose=False)

    all_ep = []
    for sid in subjects:
        try:
            sd = loader.load_subject(sid, cfg['paths']['raw_data'], logger)
            sd['raw'] = preprocess.preprocess_raw(sd['raw'], cfg, logger)
            data = sd['raw'].get_data()
            segs, meta = [], []
            for _, t in sd['trials'].iterrows():
                s0 = int(t['start'])
                a, b = s0 - pre, s0 + post
                if a < 0 or b > data.shape[1]:
                    continue
                segs.append(data[:, a:b])
                meta.append({'subject': sid, 'ma_mau': t['ma_mau'],
                             'substance': t['substance'],
                             'intensity': int(t['intensity'])})
            if not segs:
                continue
            ep = mne.EpochsArray(np.stack(segs), info, tmin=tmin,
                                 baseline=baseline, verbose=False)
            ep.metadata = pd.DataFrame(meta)
            all_ep.append(ep)
        except Exception as e:                          # noqa: BLE001
            logger.error(f"[{sid}] onset epoching failed: {e}")
    return all_ep


def grand_average_by(all_ep: list, by: str = 'substance'
                     ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Subject-then-group averaged ERP. Returns (times, {group: data[n_ch,n_t]})."""
    times = all_ep[0].times if all_ep else np.array([])
    acc: Dict[str, list] = {}
    for ep in all_ep:
        data = ep.get_data(copy=False)
        meta = ep.metadata.reset_index(drop=True)
        for g in meta[by].unique():
            idx = np.where(meta[by].values == g)[0]
            acc.setdefault(g, []).append(data[idx].mean(axis=0))
    grand = {g: np.mean(np.stack(v), axis=0) for g, v in acc.items()}
    return times, grand
