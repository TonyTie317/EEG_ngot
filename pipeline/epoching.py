"""
Epoching — extract fixed-length EEG segments around each trial onset.

Creates MNE Epochs from trial onsets, applies baseline correction,
rejects noisy epochs, and saves to disk (.fif + .csv + .npy).
"""

import os
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import mne

from .constants import ALL_SUBJECTS, SFREQ
from .config import ensure_dir

REALIGN_CSV = 'output/epochs/realign_offsets.csv'


def load_realign_offsets(logger=None):
    """Đọc realign_offsets.csv, trả về dict {(subject_id, trial_ix): new_onset}.
    Trả về None nếu file không tồn tại.
    """
    if not os.path.exists(REALIGN_CSV):
        if logger:
            logger.info(f'  [epoching] Không tìm thấy {REALIGN_CSV} → dùng trigger gốc')
        return None
    df = pd.read_csv(REALIGN_CSV)
    mapping = {}
    for _, row in df.iterrows():
        key = (row['subject_id'], int(row['trial_ix']))
        mapping[key] = int(row['new_onset'])
    if logger:
        logger.info(f'  [epoching] Loaded realign offsets: {len(mapping)} trials từ {REALIGN_CSV}')
    return mapping


def create_epochs(subject_data: Dict[str, Any], config: Dict[str, Any],
                  logger: logging.Logger) -> Tuple[mne.Epochs, pd.DataFrame]:
    """Create epochs for one subject.

    Parameters
    ----------
    subject_data : dict
        Must have 'raw' (preprocessed), 'trials' (list of trial dicts),
        and 'subject_id'.
    config : dict
        With 'epoching' section.
    logger : logging.Logger

    Returns
    -------
    epochs : mne.Epochs
        Epochs with shape (n_trials, n_channels, n_times).
    trial_info : pd.DataFrame
        Columns: subject_id, epoch_ix, condition, condition_label,
        repeat, jar, jar_group, jar_numeric.
    """
    raw = subject_data['raw']
    trials = subject_data['trials']
    sid = subject_data['subject_id']
    ep_cfg = config['epoching']

    tmin = ep_cfg['tmin']
    tmax = ep_cfg['tmax']
    baseline = ep_cfg.get('baseline', [tmin, 0.0])
    _reject_raw = ep_cfg.get('reject', {'eeg': 200e-6})
    reject = {k: float(v) for k, v in _reject_raw.items()} if _reject_raw else None

    # Build events array for MNE: (n_events, 3) with [sample, 0, event_id]
    events = []
    trial_info_rows = []
    event_id_map = {}
    # Use condition_repeat as unique event_id
    for i, trial in enumerate(trials):
        onset_sample = int(trial['start_sample'])
        cond = trial['condition']
        rep = trial['repeat']
        event_code = cond * 10 + rep  # unique per trial
        event_id_map[f"{cond}_{rep}"] = event_code
        events.append([onset_sample, 0, event_code])
        trial_info_rows.append({
            'subject_id': sid,
            'epoch_ix': i,
            'condition': cond,
            'condition_label': trial['condition_label'],
            'repeat': rep,
            'jar': trial['jar'],
            'jar_group': trial['jar_group'],
            'jar_numeric': trial['jar_numeric'],
        })

    events = np.array(events, dtype=int)

    # Create epochs
    epochs = mne.Epochs(
        raw, events, event_id=event_id_map,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        reject=reject,
        reject_by_annotation=ep_cfg.get('reject_by_annotation', True),
        preload=True,
        verbose=False,
    )

    # Drop metadata for rejected epochs
    trial_info = pd.DataFrame(trial_info_rows)
    kept_indices = epochs.selection  # indices that survived rejection
    if len(kept_indices) < len(trial_info):
        n_rejected = len(trial_info) - len(kept_indices)
        logger.info(
            f"  [{sid}] Rejected {n_rejected}/{len(trial_info)} epochs"
        )
        trial_info = trial_info.iloc[kept_indices].reset_index(drop=True)

    logger.info(
        f"  [{sid}] Created {len(epochs)} epochs: "
        f"{epochs.get_data().shape}"
    )

    return epochs, trial_info


def create_epochs_all_subjects(
    subjects_data: List[Dict[str, Any]],
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[List[mne.Epochs], pd.DataFrame]:
    """Create epochs for all subjects.

    Returns
    -------
    all_epochs : list of mne.Epochs
    all_trial_info : pd.DataFrame
        Concatenated trial info from all subjects.
    """
    all_epochs = []
    all_trial_info = []

    for sdata in subjects_data:
        sid = sdata['subject_id']
        logger.info(f"[{sid}] Creating epochs...")
        try:
            epochs, trial_info = create_epochs(sdata, config, logger)
            all_epochs.append(epochs)
            all_trial_info.append(trial_info)
        except Exception as e:
            logger.error(f"[{sid}] Epoching failed: {e}")

    if not all_trial_info:
        logger.error("No epochs created for any subject.")
        return all_epochs, pd.DataFrame()

    all_trial_info = pd.concat(all_trial_info, ignore_index=True)
    logger.info(
        f"Total: {len(all_epochs)} subjects, "
        f"{len(all_trial_info)} epochs"
    )

    return all_epochs, all_trial_info


def save_epochs(subject_id: str, epochs: mne.Epochs,
                trial_info: pd.DataFrame, config: Dict[str, Any],
                logger: logging.Logger) -> None:
    """Save epochs to disk.

    Saves:
    - {output_base}/epochs/{subject_id}/epochs.fif
    - {output_base}/epochs/{subject_id}/epochs_data.npy
    - {output_base}/epochs/{subject_id}/trial_info.csv
    """
    out_dir = os.path.join(config['paths']['output_base'], 'epochs', subject_id)
    ensure_dir(out_dir)

    epochs.save(os.path.join(out_dir, 'epochs_epo.fif'), overwrite=True, verbose=False)
    np.save(os.path.join(out_dir, 'epochs_data.npy'), epochs.get_data())
    trial_info.to_csv(os.path.join(out_dir, 'trial_info.csv'), index=False)

    logger.info(f"  [{subject_id}] Saved epochs to {out_dir}")


def load_epochs(subject_id: str, config: Dict[str, Any],
                logger: logging.Logger) -> Tuple[mne.Epochs, pd.DataFrame]:
    """Load previously saved epochs and trial info.

    Returns
    -------
    epochs : mne.Epochs
    trial_info : pd.DataFrame
    """
    out_dir = os.path.join(config['paths']['output_base'], 'epochs', subject_id)
    # Prefer MNE-compliant name; fall back to legacy name for existing files
    fif_path = os.path.join(out_dir, 'epochs_epo.fif')
    if not os.path.exists(fif_path):
        fif_path = os.path.join(out_dir, 'epochs.fif')
    csv_path = os.path.join(out_dir, 'trial_info.csv')

    if not os.path.exists(fif_path):
        raise FileNotFoundError(f"Epochs not found: {fif_path}")

    epochs = mne.read_epochs(fif_path, verbose=False)
    trial_info = pd.read_csv(csv_path)

    logger.info(f"  [{subject_id}] Loaded {len(epochs)} epochs from disk")
    return epochs, trial_info


def save_all_epochs(all_epochs: List[mne.Epochs],
                    all_trial_info: pd.DataFrame,
                    subjects: List[str], config: Dict[str, Any],
                    logger: logging.Logger) -> None:
    """Save epochs for all subjects and a combined trial_info CSV."""
    for sid, epochs in zip(subjects, all_epochs):
        mask = all_trial_info['subject_id'] == sid
        ti = all_trial_info[mask].reset_index(drop=True)
        save_epochs(sid, epochs, ti, config, logger)

    # Save combined trial_info
    combined_dir = os.path.join(config['paths']['output_base'], 'epochs')
    all_trial_info.to_csv(
        os.path.join(combined_dir, 'all_trial_info.csv'), index=False
    )
    logger.info(f"Saved combined trial_info ({len(all_trial_info)} epochs)")


def load_all_epochs(config: Dict[str, Any],
                    logger: logging.Logger) -> Tuple[List[mne.Epochs], pd.DataFrame]:
    """Load all subjects' epochs from disk.

    Returns
    -------
    all_epochs : list of mne.Epochs
    all_trial_info : pd.DataFrame
    """
    from .constants import ALL_SUBJECTS

    all_epochs = []

    for sid in ALL_SUBJECTS:
        try:
            epochs, _ = load_epochs(sid, config, logger)
            all_epochs.append(epochs)
        except FileNotFoundError:
            logger.warning(f"[{sid}] No saved epochs, skipping.")
        except Exception as e:
            logger.warning(f"[{sid}] Failed to load epochs: {e}")

    # Dùng all_trial_info.csv tổng hợp (đầy đủ 840 rows)
    all_ti_path = os.path.join(
        config['paths'].get('output_base', 'output'), 'epochs', 'all_trial_info.csv'
    )
    if os.path.exists(all_ti_path):
        all_trial_info = pd.read_csv(all_ti_path)
        logger.info(f"Loaded trial_info: {len(all_trial_info)} rows from {all_ti_path}")
    else:
        logger.warning(f"all_trial_info.csv không tìm thấy tại {all_ti_path}")
        all_trial_info = pd.DataFrame()

    logger.info(f"Loaded {len(all_epochs)} subjects from disk")
    return all_epochs, all_trial_info
