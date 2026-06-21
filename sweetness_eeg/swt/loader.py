"""
Loader for the ``data/datamoi/`` recordings.

The datamoi CSVs are heterogeneous:
  * most files use ``;`` as separator and ``,`` as the decimal mark;
  * a few (P008, P010-P012, P014-P015, P020) use ``,`` separator + ``.`` decimal;
  * P023 carries an extra empty column between ``code`` and ``ma_mau``.

To be robust we sniff the delimiter, parse with the matching decimal, and select
columns **by name** (never by position). Trials are the contiguous runs of a
single blind ``code`` (1000 samples = 10 s each, 14 per subject).
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    CHANNEL_ALIASES, CODE_TO_LABEL, EEG_CHANNELS, LABEL_INFO, RAW_EEG_CHANNELS,
    SFREQ, UV_TO_V, label_to_intensity, label_to_substance,
)


def get_csv_path(subject_id: str, raw_dir: str) -> str:
    """Return the datamoi CSV path for a subject id (e.g. ``P001``)."""
    return os.path.join(raw_dir, f"{subject_id}_KT88_with_times_10s.csv")


def _sniff_format(path: str) -> Dict[str, str]:
    """Detect (sep, decimal) by inspecting the header line."""
    with open(path, 'r', encoding='utf-8-sig') as fh:
        header = fh.readline()
    if header.count(';') >= header.count(','):
        return {'sep': ';', 'decimal': ','}
    return {'sep': ',', 'decimal': '.'}


def read_subject_dataframe(subject_id: str, raw_dir: str,
                           logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read one subject's CSV into a tidy DataFrame.

    Returns a frame with the 16 EEG channels (standard 10-20 names, in µV),
    plus ``code``, ``ma_mau`` and ``lan_lap`` columns. Channel values are kept
    in microvolts here; conversion to volts happens in :func:`build_raw`.
    """
    path = get_csv_path(subject_id, raw_dir)
    fmt = _sniff_format(path)
    df = pd.read_csv(path, sep=fmt['sep'], decimal=fmt['decimal'],
                     encoding='utf-8-sig', engine='python', dtype=str)
    # Normalise column names (strip BOM/whitespace).
    df.columns = [str(c).replace('﻿', '').strip() for c in df.columns]

    # EEG channels by name → numeric.
    missing = [c for c in RAW_EEG_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"{subject_id}: missing channels {missing}")
    eeg = df[RAW_EEG_CHANNELS].apply(
        lambda s: pd.to_numeric(s.str.replace(',', '.', regex=False)
                                if fmt['decimal'] == ',' else s, errors='coerce')
    )
    eeg = eeg.rename(columns=CHANNEL_ALIASES)

    out = eeg.copy()
    # code → int (blind sample code); blanks become 0.
    code = pd.to_numeric(df.get('code'), errors='coerce').fillna(0).astype(int)
    out['code'] = code
    out['ma_mau'] = df.get('ma_mau').astype(str).str.strip() if 'ma_mau' in df else ''
    out['lan_lap'] = df.get('lan_lap').astype(str).str.strip() if 'lan_lap' in df else ''
    # Clean sentinel strings.
    out['ma_mau'] = out['ma_mau'].replace({'nan': '', 'None': ''})

    if logger:
        logger.info(f"[{subject_id}] read {len(out)} rows ({fmt['sep']!r} sep)")
    return out


def segment_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Identify the 14 trials as contiguous runs of a single blind code.

    Returns a trial table with one row per trial:
    columns = code, ma_mau, substance, intensity, repeat, start, n.
    """
    code = df['code'].values
    rows: List[Dict[str, Any]] = []
    n = len(code)
    i = 0
    repeat_counter: Dict[str, int] = {}
    while i < n:
        c = code[i]
        if c == 0 or c not in CODE_TO_LABEL:
            i += 1
            continue
        j = i
        while j < n and code[j] == c:
            j += 1
        label = CODE_TO_LABEL[c]
        repeat_counter[label] = repeat_counter.get(label, 0) + 1
        rows.append({
            'code': int(c),
            'ma_mau': label,
            'substance': label_to_substance(label),
            'intensity': label_to_intensity(label),
            'repeat': repeat_counter[label],
            'start': int(i),
            'n': int(j - i),
        })
        i = j
    return pd.DataFrame(rows)


def build_raw(df: pd.DataFrame, subject_id: str,
              logger: Optional[logging.Logger] = None):
    """Build an MNE RawArray (volts) with a 10-20 montage from a subject frame."""
    import mne

    data = df[EEG_CHANNELS].to_numpy(dtype=float).T * UV_TO_V  # (n_ch, n_times) V
    info = mne.create_info(ch_names=list(EEG_CHANNELS),
                           sfreq=SFREQ, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='warn', verbose=False)
    return raw


def load_subject(subject_id: str, raw_dir: str,
                 logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Full load for one subject: dataframe, raw, trial table.

    Returns
    -------
    dict with keys: subject_id, df, raw, trials
    """
    df = read_subject_dataframe(subject_id, raw_dir, logger)
    trials = segment_trials(df)
    raw = build_raw(df, subject_id, logger)
    if logger:
        logger.info(f"[{subject_id}] {len(trials)} trials, "
                    f"conditions={sorted(trials['ma_mau'].unique())}")
    return {'subject_id': subject_id, 'df': df, 'raw': raw, 'trials': trials}
