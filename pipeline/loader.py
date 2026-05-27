"""
Data loading — CSV → MNE RawArray + structured trial metadata.

Reads flat CSV files from datadone/, detects trial boundaries from the
ma_mau column, extracts JAR ratings, and creates MNE RawArray objects
with proper channel types and montage.
"""

import os
import math
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import mne

from .constants import (
    EEG_CHANNELS, ECG_CHANNELS, NON_SIGNAL_COLUMNS, CHANNEL_ALIASES,
    ALL_SUBJECTS, CONCENTRATIONS, CONCENTRATION_LABELS,
    TRIALS_PER_SUBJECT, N_REPEATS, SFREQ, TRIAL_DURATION_SAMPLES,
    map_jar_to_group, JAR_NUMERIC,
)
from .config import ensure_dir


def get_csv_path(subject_id: str, config: Dict[str, Any]) -> str:
    """Construct the flat CSV path in datadone/.

    Parameters
    ----------
    subject_id : str
        Subject ID, e.g. 'P001'.
    config : dict
        Configuration with paths.raw_data and paths.csv_pattern.

    Returns
    -------
    path : str
        Absolute path to CSV file.
    """
    raw_dir = config['paths']['raw_data']
    pattern = config['paths']['csv_pattern']
    filename = pattern.format(subject=subject_id)
    return os.path.join(raw_dir, filename)


def load_subject(subject_id: str, config: Dict[str, Any],
                 logger: logging.Logger) -> Dict[str, Any]:
    """Load one subject's CSV and return structured data.

    Parameters
    ----------
    subject_id : str
        Subject ID, e.g. 'P001'.
    config : dict
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    data : dict
        Keys:
        - 'raw': mne.io.RawArray (16 EEG + 2 ECG channels, converted µV→V)
        - 'trials': list of 30 trial dicts
        - 'subject_id': str
        - 'trial_table': pd.DataFrame with trial metadata
    """
    csv_path = get_csv_path(subject_id, config)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info(f"[{subject_id}] Loading {csv_path}")
    df = pd.read_csv(csv_path)

    # ── Create MNE RawArray ───────────────────────────────────────────────
    prep_cfg = config.get('preprocessing', {})
    # Non-signal columns: prefer config list, fall back to constants
    non_sig = set(prep_cfg.get('non_signal_cols', NON_SIGNAL_COLUMNS))
    signal_cols = [c for c in df.columns if c not in non_sig]
    # Drop columns that are all NaN
    signal_cols = [c for c in signal_cols if not df[c].isna().all()]

    # ── Rename channels: config map first, then constants aliases ─────────
    cfg_rename = prep_cfg.get('rename_channels', {})
    # Merge config rename with built-in aliases (config takes precedence)
    full_rename = {**CHANNEL_ALIASES, **cfg_rename}
    rename_map = {c: full_rename[c] for c in signal_cols if c in full_rename}
    if rename_map:
        df = df.rename(columns=rename_map)
        signal_cols = [rename_map.get(c, c) for c in signal_cols]
        logger.info(f"[{subject_id}] Renamed channels: {rename_map}")

    # ── Normalize remaining channel names to canonical 10-20 (case-insensitive)
    canonical_all = EEG_CHANNELS + ECG_CHANNELS
    lower_to_canonical = {ch.lower(): ch for ch in canonical_all}
    case_rename = {}
    for c in signal_cols:
        canonical = lower_to_canonical.get(c.lower())
        if canonical and canonical != c:
            case_rename[c] = canonical
    if case_rename:
        df = df.rename(columns=case_rename)
        signal_cols = [case_rename.get(c, c) for c in signal_cols]
        logger.info(f"[{subject_id}] Case-normalized channels: {case_rename}")

    # Keep only known EEG/ECG channels (drop unexpected signal columns)
    known_set = set(canonical_all)
    unknown = [c for c in signal_cols if c not in known_set]
    if unknown:
        logger.warning(f"[{subject_id}] Dropping unknown signal columns: {unknown}")
        signal_cols = [c for c in signal_cols if c in known_set]

    # Classify channel types
    ch_types = []
    for c in signal_cols:
        if c.upper().startswith('ECG'):
            ch_types.append('ecg')
        else:
            ch_types.append('eeg')

    # Convert µV → V
    data_mat = df[signal_cols].to_numpy(dtype=np.float64)
    # Replace NaN with 0 for MNE compatibility (NaN only in metadata-like cols)
    data_mat = np.nan_to_num(data_mat, nan=0.0)
    data_mat = data_mat.T * 1e-6  # (n_channels, n_times) in Volts

    info = mne.create_info(ch_names=signal_cols, sfreq=SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(data_mat, info, verbose=False)

    # Set montage from config
    montage_name = prep_cfg.get('montage', 'standard_1020')
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=False, on_missing='ignore')
    except Exception as e:
        logger.warning(f"[{subject_id}] Could not set montage '{montage_name}': {e}")

    # ── Detect trials ─────────────────────────────────────────────────────
    trials = detect_trials(df, logger, subject_id)
    if len(trials) != TRIALS_PER_SUBJECT:
        logger.warning(
            f"[{subject_id}] Expected {TRIALS_PER_SUBJECT} trials, "
            f"got {len(trials)}"
        )

    # ── Build trial table ─────────────────────────────────────────────────
    trial_table = pd.DataFrame(trials)

    logger.info(
        f"[{subject_id}] Loaded: {raw.info['nchan']} channels, "
        f"{len(trials)} trials, {raw.n_times} samples"
    )

    return {
        'raw': raw,
        'trials': trials,
        'subject_id': subject_id,
        'trial_table': trial_table,
    }


def detect_trials(df: pd.DataFrame, logger: logging.Logger,
                  subject_id: str = '') -> List[Dict[str, Any]]:
    """Detect trial boundaries from the ma_mau column.

    A trial is a contiguous block where ma_mau has a non-NaN, non-zero
    value. Each block should be exactly 1100 samples (11s at 100 Hz).

    Parameters
    ----------
    df : pd.DataFrame
        Raw CSV data with 'ma_mau', 'repeat', 'JAR' columns.
    logger : logging.Logger
    subject_id : str
        For log messages.

    Returns
    -------
    trials : list of dict
        Each dict has: trial_ix, condition, repeat, jar, jar_group,
        jar_numeric, condition_label, start_sample, end_sample,
        onset_sec, n_samples.
    """
    vals = df['ma_mau'].to_numpy()
    is_valid = ~pd.isna(vals)

    # Group contiguous non-zero segments
    raw_segments = []
    i = 0
    N = len(vals)
    while i < N:
        if is_valid[i] and vals[i] != 0:
            event_code = int(vals[i])
            j = i
            while j + 1 < N and is_valid[j + 1] and vals[j + 1] == event_code:
                j += 1
            raw_segments.append({
                'condition': event_code,
                'start': i,
                'end': j,
            })
            i = j + 1
        else:
            i += 1

    # Further split segments by 'repeat' value changes
    # (handles case where same condition appears consecutively with different repeats)
    trials = []
    trial_ix = 0
    for seg in raw_segments:
        seg_df = df.iloc[seg['start']:seg['end'] + 1].copy()
        condition = seg['condition']

        # Check if repeat column varies within this segment
        repeats = seg_df['repeat'].dropna().values
        if len(repeats) > 0 and len(np.unique(repeats)) > 1:
            # Split by repeat changes
            rep_vals = seg_df['repeat'].fillna(method='ffill').values
            boundaries = [0]
            for k in range(1, len(rep_vals)):
                if not math.isnan(rep_vals[k]) and not math.isnan(rep_vals[k - 1]):
                    if rep_vals[k] != rep_vals[k - 1]:
                        boundaries.append(k)
            boundaries.append(len(rep_vals))

            for b in range(len(boundaries) - 1):
                s = seg['start'] + boundaries[b]
                e = seg['start'] + boundaries[b + 1] - 1
                sub_df = df.iloc[s:e + 1]
                repeat_val = int(sub_df['repeat'].dropna().iloc[0])
                jar_val = float(sub_df['JAR'].dropna().iloc[0])
                jar_group = map_jar_to_group(jar_val)

                trials.append(_make_trial_dict(
                    trial_ix, condition, repeat_val, jar_val, jar_group,
                    s, e, len(sub_df)
                ))
                trial_ix += 1
        else:
            # Single repeat within this segment
            repeat_val = int(seg_df['repeat'].dropna().iloc[0])
            jar_val = float(seg_df['JAR'].dropna().iloc[0])
            jar_group = map_jar_to_group(jar_val)

            trials.append(_make_trial_dict(
                trial_ix, condition, repeat_val, jar_val, jar_group,
                seg['start'], seg['end'], seg['end'] - seg['start'] + 1
            ))
            trial_ix += 1

    return trials


def _make_trial_dict(trial_ix, condition, repeat, jar, jar_group,
                     start_sample, end_sample, n_samples):
    """Build a single trial dict."""
    return {
        'trial_ix': trial_ix,
        'condition': condition,
        'condition_label': CONCENTRATION_LABELS.get(condition, str(condition)),
        'repeat': repeat,
        'jar': jar,
        'jar_group': jar_group,
        'jar_numeric': JAR_NUMERIC.get(jar_group, -1) if jar_group else -1,
        'start_sample': start_sample,
        'end_sample': end_sample,
        'onset_sec': start_sample / SFREQ,
        'n_samples': n_samples,
    }


def load_all_subjects(config: Dict[str, Any],
                      logger: logging.Logger) -> List[Dict[str, Any]]:
    """Load all 28 subjects.

    Skips subjects whose CSV does not exist, logs warnings for failures.

    Parameters
    ----------
    config : dict
    logger : logging.Logger

    Returns
    -------
    subjects_data : list of dict
        One dict per successfully loaded subject.
    """
    subjects_data = []
    for sid in ALL_SUBJECTS:
        try:
            data = load_subject(sid, config, logger)
            subjects_data.append(data)
        except FileNotFoundError:
            logger.warning(f"[{sid}] CSV not found, skipping.")
        except Exception as e:
            logger.error(f"[{sid}] Failed to load: {e}")

    logger.info(f"Loaded {len(subjects_data)}/{len(ALL_SUBJECTS)} subjects")
    return subjects_data
