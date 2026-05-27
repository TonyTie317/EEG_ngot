"""
ERP Analysis — grand-average, peak detection, component measures.

Core module for gERP research:
- Grand-average ERP across all subjects and conditions
- Peak detection for P1, N1, P2, N400 components
- Per-trial component measure extraction
- Comparison by concentration level and JAR group
- Difference waves for key contrasts
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
from scipy.signal import welch as sp_welch
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

from .constants import (
    CONCENTRATIONS, CONCENTRATION_LABELS, ERP_WINDOWS, ERP_PEAK_MODE,
    ERP_ROI, ROI, JAR_NUMERIC, FREQ_BANDS,
)
from .config import ensure_dir

REALIGN_CSV = 'output/epochs/realign_offsets.csv'
# Cửa sổ ERP sau true onset
_REALIGN_TMIN = -0.2   # 200ms baseline trước true onset
_REALIGN_TMAX =  1.0   # 1s ERP window sau true onset


def apply_woody_realign(all_epochs: List[mne.Epochs],
                        all_trial_info: pd.DataFrame,
                        logger: logging.Logger):
    """Áp dụng Woody realignment: cắt epoch [-0.2s, +1.0s] quanh true onset.

    Returns
    -------
    realigned_epochs : list of mne.Epochs  (tmin=-0.2, tmax=1.0)
    filtered_trial_info : pd.DataFrame     (đã lọc theo kept mask)
    """
    if not os.path.exists(REALIGN_CSV):
        logger.warning(f'[ERP] Không tìm thấy {REALIGN_CSV} — dùng trigger gốc')
        return all_epochs, all_trial_info

    offsets_df = pd.read_csv(REALIGN_CSV)
    SFREQ = 100
    pre  = int(abs(_REALIGN_TMIN) * SFREQ)   # 20 mẫu
    post = int(_REALIGN_TMAX * SFREQ)         # 100 mẫu
    win_len = pre + post + 1                  # 121 mẫu

    # Lấy danh sách subject theo thứ tự đúng từ trial_info (giữ order)
    subjects = all_trial_info['subject_id'].unique()  # ordered unique
    # Đảm bảo thứ tự khớp all_epochs bằng cách group trial_info theo subject
    subj_to_rows = {sid: all_trial_info[all_trial_info['subject_id'] == sid].index.tolist()
                    for sid in subjects}

    realigned = []
    global_keep = []   # list of original row indices to keep

    for epochs, sid in zip(all_epochs, subjects):
        subj_offsets = offsets_df[offsets_df['subject_id'] == sid]
        row_indices  = subj_to_rows[sid]   # original indices in all_trial_info
        raw_data = epochs.get_data()            # (n_ep, n_ch, n_t)
        info = epochs.info
        offset_orig = int(abs(epochs.tmin) * SFREQ)  # samples trước trigger

        new_data = []
        kept_mask = []
        for ep_i in range(len(epochs)):
            final_off = int(subj_offsets.iloc[ep_i]['offset_final']) \
                if ep_i < len(subj_offsets) else 0
            true_onset_idx = offset_orig + final_off
            start = true_onset_idx - pre
            end   = start + win_len
            if start < 0 or end > raw_data.shape[2]:
                kept_mask.append(False)
                continue
            new_data.append(raw_data[ep_i, :, start:end])
            kept_mask.append(True)

        # Ghi nhớ row indices được giữ lại
        for k, keep in enumerate(kept_mask):
            if keep and k < len(row_indices):
                global_keep.append(row_indices[k])

        if new_data:
            arr = np.stack(new_data, axis=0)
            realigned.append(mne.EpochsArray(arr, info, tmin=_REALIGN_TMIN,
                                             verbose=False))
        else:
            logger.warning(f'[ERP] [{sid}] Không có epoch nào sau re-align')
            realigned.append(epochs)

    filtered_info = all_trial_info.loc[global_keep].reset_index(drop=True)
    n_kept = len(global_keep)
    logger.info(f'[ERP] Realign áp dụng: {n_kept}/{len(all_trial_info)} epochs giữ lại, '
                f'window [{_REALIGN_TMIN*1000:.0f}ms, {_REALIGN_TMAX*1000:.0f}ms]')
    return realigned, filtered_info


# ──────────────────────────────────────────────────────────────────────────────
# Grand-average ERP computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_grand_average(
    all_epochs: List[mne.Epochs],
    all_trial_info: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Compute grand-average ERP across all subjects.

    Parameters
    ----------
    all_epochs : list of mne.Epochs
        One per subject.
    all_trial_info : pd.DataFrame
        Combined trial info with subject_id, condition, jar_group columns.
    config : dict
    logger : logging.Logger

    Returns
    -------
    results : dict
        Keys:
        - 'evoked_all': mne.Evoked (average of all epochs)
        - 'evoked_by_condition': {condition: mne.Evoked}
        - 'evoked_by_jar_group': {group: mne.Evoked}
        - 'evoked_by_subject_condition': {(subject, condition): mne.Evoked}
        - 'measures': pd.DataFrame (per-subject, per-condition component measures)
    """
    # Concatenate all epochs into one
    all_epochs_data = []
    for epochs in all_epochs:
        all_epochs_data.append(epochs.get_data())
    X = np.concatenate(all_epochs_data, axis=0)  # (n_total_epochs, n_ch, n_times)

    # Use first subject's info as template
    info = all_epochs[0].info
    tmin = all_epochs[0].tmin

    logger.info(f"Grand average: {X.shape[0]} total epochs, {X.shape[1]} channels")

    # ── Evoked: all epochs ────────────────────────────────────────────────
    evoked_all = mne.EvokedArray(
        X.mean(axis=0), info, tmin=tmin, comment='All'
    )

    # ── Evoked: by condition ───────────────────────────────────────────────
    evoked_by_condition = {}
    for cond in CONCENTRATIONS:
        mask = all_trial_info['condition'] == cond
        if mask.sum() > 0:
            cond_data = X[mask.values].mean(axis=0)
            label = CONCENTRATION_LABELS.get(cond, str(cond))
            evoked_by_condition[cond] = mne.EvokedArray(
                cond_data, info, tmin=tmin, comment=label
            )

    # ── Evoked: by JAR group ──────────────────────────────────────────────
    evoked_by_jar = {}
    for group_name in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
        mask = all_trial_info['jar_group'] == group_name
        if mask.sum() > 0:
            group_data = X[mask.values].mean(axis=0)
            evoked_by_jar[group_name] = mne.EvokedArray(
                group_data, info, tmin=tmin, comment=group_name
            )

    # ── Evoked: per-subject per-condition ──────────────────────────────────
    evoked_by_subj_cond = {}
    subjects = all_trial_info['subject_id'].unique()
    for subj in subjects:
        subj_mask = all_trial_info['subject_id'] == subj
        for cond in CONCENTRATIONS:
            mask = subj_mask & (all_trial_info['condition'] == cond)
            if mask.sum() >= 1:
                sc_data = X[mask.values].mean(axis=0)
                evoked_by_subj_cond[(subj, cond)] = mne.EvokedArray(
                    sc_data, info, tmin=tmin,
                    comment=f"{subj}_{CONCENTRATION_LABELS.get(cond, str(cond))}"
                )

    # ── Extract per-subject per-condition component measures ───────────────
    measures = extract_component_measures(
        all_epochs, all_trial_info, config, logger
    )

    logger.info(
        f"Grand average computed: {len(evoked_by_condition)} conditions, "
        f"{len(evoked_by_jar)} JAR groups"
    )

    return {
        'evoked_all': evoked_all,
        'evoked_by_condition': evoked_by_condition,
        'evoked_by_jar_group': evoked_by_jar,
        'evoked_by_subject_condition': evoked_by_subj_cond,
        'measures': measures,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Peak detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_peaks(evoked: mne.Evoked, component: str,
                 config: Dict[str, Any]) -> Dict[str, Any]:
    """Detect peak amplitude and latency for an ERP component.

    Parameters
    ----------
    evoked : mne.Evoked
    component : str
        One of 'P1', 'N1', 'P2', 'N400'.
    config : dict
        With erp_analysis section containing windows and ROI.

    Returns
    -------
    result : dict
        Keys: peak_amplitude (V), peak_latency (s), peak_channel,
        mean_amplitude (V), window_start, window_end.
    """
    erp_cfg = config.get('erp_analysis', {})

    # Get time window for this component
    window_key = f'{component.lower()}_window'
    if window_key in erp_cfg:
        win = erp_cfg[window_key]
    else:
        win = ERP_WINDOWS[component]
    tmin_win, tmax_win = win

    # Get ROI channels for this component
    roi_key = f'{component.lower()}_roi'
    if roi_key in erp_cfg:
        roi_chs = erp_cfg[roi_key]
    else:
        roi_chs = ERP_ROI.get(component, evoked.ch_names)

    # Filter to channels that exist in the data
    roi_chs = [ch for ch in roi_chs if ch in evoked.ch_names]
    if not roi_chs:
        roi_chs = evoked.ch_names

    # Pick ROI channels and time window
    data = evoked.copy().pick(roi_chs).get_data()  # (n_roi, n_times)
    times = evoked.times
    time_mask = (times >= tmin_win) & (times <= tmax_win)
    window_data = data[:, time_mask]
    window_times = times[time_mask]

    # Average across ROI channels for peak detection
    roi_avg = window_data.mean(axis=0)  # (n_times_in_window,)

    # Peak detection mode
    mode = ERP_PEAK_MODE.get(component, 'pos')
    if mode == 'pos':
        peak_idx = np.argmax(roi_avg)
    else:
        peak_idx = np.argmin(roi_avg)

    peak_amplitude = roi_avg[peak_idx]
    peak_latency = window_times[peak_idx]

    # Find which channel has the strongest peak
    ch_peaks = window_data[:, peak_idx]
    peak_channel_idx = (np.argmax if mode == 'pos' else np.argmin)(ch_peaks)

    # Mean amplitude across window and ROI
    mean_amplitude = window_data.mean()

    return {
        'peak_amplitude': peak_amplitude,
        'peak_latency': peak_latency,
        'peak_channel': roi_chs[peak_channel_idx],
        'mean_amplitude': mean_amplitude,
        'window_start': tmin_win,
        'window_end': tmax_win,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Component measures extraction (per-subject, per-condition)
# ──────────────────────────────────────────────────────────────────────────────

def extract_component_measures(
    all_epochs: List[mne.Epochs],
    all_trial_info: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extract ERP component measures per subject per condition.

    For each subject × condition, averages across repeats (5 trials),
    then extracts P1/N1/P2/N400 peak and mean amplitude measures.

    Parameters
    ----------
    all_epochs : list of mne.Epochs
    all_trial_info : pd.DataFrame
    config : dict
    logger : logging.Logger

    Returns
    -------
    measures : pd.DataFrame
        Columns: subject_id, condition, condition_label,
        {component}_mean_amp, {component}_peak_amp, {component}_peak_lat
        for each of P1, N1, P2, N400.
    """
    erp_cfg = config.get('erp_analysis', {})
    components = ['P1', 'N1', 'P2', 'N400']

    # Get window/ROI config per component
    comp_config = {}
    for comp in components:
        wkey = f'{comp.lower()}_window'
        rkey = f'{comp.lower()}_roi'
        comp_config[comp] = {
            'window': erp_cfg.get(wkey, ERP_WINDOWS[comp]),
            'roi': erp_cfg.get(rkey, ERP_ROI.get(comp, [])),
            'mode': ERP_PEAK_MODE[comp],
        }

    rows = []
    subjects = all_trial_info['subject_id'].unique()

    # Map each epoch to its subject index
    epoch_idx = 0
    subj_epoch_offsets = {}
    for i, epochs in enumerate(all_epochs):
        sid = all_trial_info.iloc[epoch_idx]['subject_id'] if epoch_idx < len(all_trial_info) else None
        # Find which subject this belongs to
        pass

    # Better approach: iterate through subjects
    offset = 0
    for subj_idx, epochs in enumerate(all_epochs):
        n_epochs = len(epochs)
        ti_subj = all_trial_info.iloc[offset:offset + n_epochs]
        offset += n_epochs

        subj_id = ti_subj['subject_id'].iloc[0]
        data = epochs.get_data()  # (n_epochs, n_ch, n_times)
        info = epochs.info
        times = epochs.times

        for cond in CONCENTRATIONS:
            cond_mask = ti_subj['condition'] == cond
            if cond_mask.sum() == 0:
                continue

            # Average across repeats for this subject × condition
            cond_epochs = data[cond_mask.values].mean(axis=0)  # (n_ch, n_times)
            evoked = mne.EvokedArray(cond_epochs, info, tmin=epochs.tmin,
                                     comment=f"{subj_id}_{cond}", verbose=False)

            row = {
                'subject_id': subj_id,
                'condition': cond,
                'condition_label': CONCENTRATION_LABELS.get(cond, str(cond)),
            }

            for comp in components:
                cc = comp_config[comp]
                tmin_w, tmax_w = cc['window']
                roi_chs = [ch for ch in cc['roi'] if ch in evoked.ch_names]
                if not roi_chs:
                    roi_chs = evoked.ch_names

                # Extract windowed data
                time_mask = (times >= tmin_w) & (times <= tmax_w)
                ev_data = evoked.copy().pick(roi_chs).get_data()
                win_data = ev_data[:, time_mask]
                roi_avg = win_data.mean(axis=0)

                # Peak
                mode = cc['mode']
                peak_idx = (np.argmax if mode == 'pos' else np.argmin)(roi_avg)
                peak_amp = roi_avg[peak_idx]
                peak_lat = times[time_mask][peak_idx]

                # Mean amplitude
                mean_amp = win_data.mean()

                row[f'{comp}_mean_amp'] = mean_amp
                row[f'{comp}_peak_amp'] = peak_amp
                row[f'{comp}_peak_lat'] = peak_lat

            rows.append(row)

    measures = pd.DataFrame(rows)
    logger.info(f"Extracted component measures: {measures.shape}")
    return measures


# ──────────────────────────────────────────────────────────────────────────────
# Comparison functions
# ──────────────────────────────────────────────────────────────────────────────

def compare_by_concentration(
    measures: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute summary statistics per concentration level.

    For each component measure, compute mean ± SEM across subjects
    for each concentration level.

    Returns
    -------
    summary : pd.DataFrame
        Columns: condition, component, measure, mean, sem, n_subjects.
    """
    components = ['P1', 'N1', 'P2', 'N400']
    measure_types = ['mean_amp', 'peak_amp', 'peak_lat']
    rows = []

    for cond in CONCENTRATIONS:
        cond_data = measures[measures['condition'] == cond]
        for comp in components:
            for mtype in measure_types:
                col = f'{comp}_{mtype}'
                if col in cond_data.columns:
                    vals = cond_data[col].dropna()
                    rows.append({
                        'condition': cond,
                        'condition_label': CONCENTRATION_LABELS.get(cond, str(cond)),
                        'component': comp,
                        'measure': mtype,
                        'mean': vals.mean(),
                        'sem': vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                        'n_subjects': len(vals),
                    })

    summary = pd.DataFrame(rows)
    logger.info(f"Concentration comparison: {len(summary)} rows")
    return summary


def compare_by_jar_group(
    measures: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Compute summary statistics per JAR group.

    Groups trials by JAR group regardless of concentration.

    Returns
    -------
    summary : pd.DataFrame
        Columns: jar_group, component, measure, mean, sem, n_subjects.
    """
    # Merge JAR group back onto measures
    # measures has subject_id and condition; need to look up jar_group
    # from the trial_info or re-derive it
    # Since JAR is per-subject per-condition, we can look it up
    rows = []
    components = ['P1', 'N1', 'P2', 'N400']
    measure_types = ['mean_amp', 'peak_amp', 'peak_lat']

    for _, row_data in measures.iterrows():
        subj = row_data['subject_id']
        cond = row_data['condition']
        # We need the JAR group for this subject-condition combo
        # Store it directly in measures during extraction (done below)

    # Need to add jar_group to measures first
    # This will be done in run_erp_analysis
    return pd.DataFrame()  # placeholder, filled in run_erp_analysis


def compute_difference_waves(
    all_epochs: List[mne.Epochs],
    all_trial_info: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, mne.Evoked]:
    """Compute difference waves between conditions.

    Contrasts:
    - 'High_vs_Water': condition 893 - condition 189
    - 'MedHigh_vs_Water': condition 762 - condition 189
    - 'High_vs_Medium': condition 893 - condition 605

    Returns
    -------
    diff_waves : dict
        {contrast_name: mne.Evoked}
    """
    # Concatenate all epoch data
    all_data = np.concatenate([ep.get_data() for ep in all_epochs], axis=0)
    info = all_epochs[0].info
    tmin = all_epochs[0].tmin

    contrasts = {
        'High_vs_Water': (893, 605),
        'MedHigh_vs_Water': (762, 605),
        'High_vs_Medium': (893, 189),
    }

    diff_waves = {}
    for name, (cond_a, cond_b) in contrasts.items():
        mask_a = all_trial_info['condition'] == cond_a
        mask_b = all_trial_info['condition'] == cond_b
        if mask_a.sum() > 0 and mask_b.sum() > 0:
            mean_a = all_data[mask_a.values].mean(axis=0)
            mean_b = all_data[mask_b.values].mean(axis=0)
            diff = mean_a - mean_b
            diff_waves[name] = mne.EvokedArray(
                diff, info, tmin=tmin, comment=name
            )
            logger.info(f"  Difference wave '{name}': {mask_a.sum()} vs {mask_b.sum()} trials")

    return diff_waves


# ──────────────────────────────────────────────────────────────────────────────
# Bandpower features (for ML feature extraction)
# ──────────────────────────────────────────────────────────────────────────────

def extract_bandpower_features(
    all_epochs: List[mne.Epochs],
    all_trial_info: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extract bandpower features per epoch for ML classification.

    For each epoch, computes mean PSD power per frequency band per channel.

    Returns
    -------
    features : pd.DataFrame
        Wide format: one row per epoch, columns are bandpower features.
    """
    rows = []
    offset = 0

    for epochs in all_epochs:
        n_ep = len(epochs)
        ti = all_trial_info.iloc[offset:offset + n_ep]
        offset += n_ep

        # Compute PSD per epoch (n_fft must not exceed n_times)
        n_times = epochs.get_data().shape[-1]
        n_fft = min(128, n_times)
        psd = epochs.compute_psd(method='welch', fmin=0.5, fmax=45,
                                  n_fft=n_fft, verbose=False)
        psd_data = psd.get_data()  # (n_epochs, n_channels, n_freqs)
        freqs = psd.freqs

        # Pre-compute per-band power: (n_epochs, n_channels) for each band
        band_powers = {}
        for band_name, (fmin_b, fmax_b) in FREQ_BANDS.items():
            freq_mask = (freqs >= fmin_b) & (freqs <= fmax_b)
            if freq_mask.sum() == 0:
                continue
            # Explicit two-step to avoid numpy mixed-indexing shape ambiguity
            band_powers[band_name] = psd_data[:, :, :][:, :, freq_mask].mean(axis=2)  # (n_ep, n_ch)

        for ep_idx in range(n_ep):
            row = {
                'subject_id': ti.iloc[ep_idx]['subject_id'],
                'condition': ti.iloc[ep_idx]['condition'],
                'jar_group': ti.iloc[ep_idx].get('jar_group'),
                'jar_numeric': ti.iloc[ep_idx].get('jar_numeric'),
            }
            for band_name, bp in band_powers.items():
                for ch_idx, ch_name in enumerate(epochs.ch_names):
                    row[f'bp_{band_name}_{ch_name}'] = bp[ep_idx, ch_idx]
            rows.append(row)

    features = pd.DataFrame(rows)
    logger.info(f"Bandpower features: {features.shape}")
    return features


# ──────────────────────────────────────────────────────────────────────────────
# Extended ML feature extraction (per subject × condition)
# ──────────────────────────────────────────────────────────────────────────────

def extract_extended_features(
    all_epochs: List[mne.Epochs],
    all_trial_info: pd.DataFrame,
    config: Dict[str, Any],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extract extended ML features per subject×condition (averaged over repeats).

    Features computed from the condition-averaged ERP signal:
    - Hjorth parameters (activity, mobility, complexity) × 14 channels
    - Time-domain statistics (mean, std, skew, kurtosis, ptp) × 14 channels
    - Spectral features (SEF50, SEF90, spectral centroid) × 14 channels
    - Band power ratios (theta/alpha, alpha/beta, theta/beta) × 14 channels
    - Alpha frontal asymmetry for F3/F4, F7/F8, AF3/AF4 pairs
    - Mean bandpower per band per channel (averaged over repeats)

    Returns
    -------
    features : pd.DataFrame
        One row per subject×condition. Columns: subject_id, condition,
        condition_label, jar_group + all feature columns.
    """
    rows = []
    offset = 0

    for epochs in all_epochs:
        n_ep = len(epochs)
        ti = all_trial_info.iloc[offset:offset + n_ep]
        offset += n_ep

        subj_id = ti['subject_id'].iloc[0]
        data = epochs.get_data()          # (n_ep, n_ch, n_times)
        sfreq = epochs.info['sfreq']
        times = epochs.times

        # EEG channels only (exclude ECG if present)
        ch_names = [ch for ch in epochs.ch_names
                    if not ch.upper().startswith('ECG')]
        ch_indices = [epochs.ch_names.index(ch) for ch in ch_names]

        # Pre-compute PSD per epoch per channel (for bandpower averaging)
        n_fft = min(128, data.shape[2])
        # (n_ep, n_ch, n_freqs) via welch per-epoch
        try:
            psd_obj = epochs.compute_psd(method='welch', fmin=0.5, fmax=45.0,
                                          n_fft=n_fft, verbose=False)
            psd_all = psd_obj.get_data()   # (n_ep, n_ch, n_freqs)
            psd_freqs = psd_obj.freqs
        except Exception:
            psd_all = None
            psd_freqs = None

        for cond in CONCENTRATIONS:
            cond_mask = (ti['condition'] == cond).values
            if cond_mask.sum() == 0:
                continue

            # Condition average across repeats: (n_ch, n_times)
            avg = data[cond_mask].mean(axis=0)

            # JAR group for this subject×condition
            jar_val = ti.loc[ti['condition'] == cond, 'jar_group'].iloc[0] \
                if 'jar_group' in ti.columns else None

            row = {
                'subject_id': subj_id,
                'condition': cond,
                'condition_label': CONCENTRATION_LABELS.get(cond, str(cond)),
                'jar_group': jar_val,
            }

            # ── Per-channel features ──────────────────────────────────────
            for ci, ch_name in zip(ch_indices, ch_names):
                x = avg[ci]   # (n_times,)

                # --- Hjorth parameters ---
                dx = np.diff(x)
                ddx = np.diff(dx)
                act = float(np.var(x))
                mob = float(np.sqrt(np.var(dx) / act)) if act > 1e-12 else 0.0
                cmp = float(np.sqrt(np.var(ddx) / np.var(dx)) / mob) \
                      if (np.var(dx) > 1e-12 and mob > 1e-12) else 0.0
                row[f'hjorth_act_{ch_name}'] = act
                row[f'hjorth_mob_{ch_name}'] = mob
                row[f'hjorth_cmp_{ch_name}'] = cmp

                # --- Time-domain statistics ---
                row[f'td_mean_{ch_name}']  = float(np.mean(x))
                row[f'td_std_{ch_name}']   = float(np.std(x))
                row[f'td_skew_{ch_name}']  = float(sp_skew(x))
                row[f'td_kurt_{ch_name}']  = float(sp_kurtosis(x))
                row[f'td_ptp_{ch_name}']   = float(np.ptp(x))

                # --- Spectral features ---
                freqs_w, psd_w = sp_welch(x, fs=sfreq, nperseg=n_fft, nfft=n_fft)
                valid = (freqs_w > 0) & (freqs_w <= sfreq / 2)
                fw, ps = freqs_w[valid], psd_w[valid]
                total_power = float(np.sum(ps)) if np.sum(ps) > 0 else 1.0
                cum_ps = np.cumsum(ps)

                # SEF50, SEF90
                for pct, pname in [(0.50, 'sef50'), (0.90, 'sef90')]:
                    idx = int(np.searchsorted(cum_ps, pct * cum_ps[-1]))
                    row[f'{pname}_{ch_name}'] = float(fw[min(idx, len(fw) - 1)])

                # Spectral centroid
                row[f'spec_cent_{ch_name}'] = float(np.sum(fw * ps) / total_power)

                # Band power per band (for ratios)
                bp = {}
                for bname, (bf, bt) in FREQ_BANDS.items():
                    bm = (fw >= bf) & (fw <= bt)
                    bp[bname] = float(np.mean(ps[bm])) if bm.sum() > 0 else 0.0

                # Band ratios
                for num, den in [('theta', 'alpha'), ('alpha', 'beta'),
                                  ('theta', 'beta'), ('delta', 'alpha')]:
                    d = bp.get(den, 0.0)
                    row[f'ratio_{num}_{den}_{ch_name}'] = bp.get(num, 0.0) / d \
                        if d > 1e-12 else 0.0

            # ── Alpha frontal asymmetry ──────────────────────────────────
            for left, right in [('F3', 'F4'), ('F7', 'F8'), ('AF3', 'AF4')]:
                if left in ch_names and right in ch_names:
                    xl = avg[ch_indices[ch_names.index(left)]]
                    xr = avg[ch_indices[ch_names.index(right)]]
                    _, psl = sp_welch(xl, fs=sfreq, nperseg=n_fft, nfft=n_fft)
                    fw2, psr = sp_welch(xr, fs=sfreq, nperseg=n_fft, nfft=n_fft)
                    a_mask = (fw2 >= 8) & (fw2 <= 12)
                    pow_l = float(np.mean(psl[a_mask])) if a_mask.sum() > 0 else 1e-10
                    pow_r = float(np.mean(psr[a_mask])) if a_mask.sum() > 0 else 1e-10
                    row[f'alpha_asym_{left}_{right}'] = \
                        np.log(pow_r + 1e-10) - np.log(pow_l + 1e-10)

            # ── Bandpower averaged over repeats ──────────────────────────
            if psd_all is not None and psd_freqs is not None:
                # Average PSD over the repeat trials for this condition
                psd_cond = psd_all[cond_mask].mean(axis=0)  # (n_ch, n_freqs)
                for bname, (bf, bt) in FREQ_BANDS.items():
                    bm = (psd_freqs >= bf) & (psd_freqs <= bt)
                    if bm.sum() == 0:
                        continue
                    bp_vals = psd_cond[:, bm].mean(axis=1)  # (n_ch,)
                    for ci2, ch in enumerate(epochs.ch_names):
                        row[f'bp_{bname}_{ch}'] = float(bp_vals[ci2])

            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Extended ML features: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Master entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_erp_analysis(config: Dict[str, Any],
                     logger: logging.Logger) -> Dict[str, Any]:
    """Run the full ERP analysis pipeline.

    Loads epochs from disk, computes grand averages, extracts component
    measures, computes difference waves, and saves all results.

    Parameters
    ----------
    config : dict
    logger : logging.Logger

    Returns
    -------
    results : dict
        All ERP analysis results.
    """
    from .epoching import load_all_epochs

    logger.info("=" * 60)
    logger.info("STAGE: ERP Analysis")
    logger.info("=" * 60)

    # Load epochs from disk
    all_epochs, all_trial_info = load_all_epochs(config, logger)
    if not all_epochs:
        logger.error("No epochs loaded. Run epoching stage first.")
        return {}

    # Áp dụng Woody realignment (dùng onset mới từ realign_offsets.csv)
    logger.info("Applying Woody realignment to epochs...")
    all_epochs, all_trial_info = apply_woody_realign(all_epochs, all_trial_info, logger)
    if not all_epochs:
        logger.error("No epochs remain after realignment.")
        return {}

    # Grand average
    logger.info("Computing grand averages...")
    results = compute_grand_average(all_epochs, all_trial_info, config, logger)

    # Add JAR group to measures
    # JAR is per-subject per-condition; merge from trial_info
    jar_lookup = all_trial_info.groupby(
        ['subject_id', 'condition']
    )['jar_group'].first().reset_index()
    results['measures'] = results['measures'].merge(
        jar_lookup, on=['subject_id', 'condition'], how='left'
    )

    # Concentration comparison
    logger.info("Comparing by concentration...")
    results['concentration_summary'] = compare_by_concentration(
        results['measures'], config, logger
    )

    # JAR group comparison
    logger.info("Comparing by JAR group...")
    jar_measures = results['measures'].dropna(subset=['jar_group'])
    if len(jar_measures) > 0:
        components = ['P1', 'N1', 'P2', 'N400']
        measure_types = ['mean_amp', 'peak_amp', 'peak_lat']
        jar_rows = []
        for group in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
            grp_data = jar_measures[jar_measures['jar_group'] == group]
            for comp in components:
                for mtype in measure_types:
                    col = f'{comp}_{mtype}'
                    if col in grp_data.columns:
                        vals = grp_data[col].dropna()
                        jar_rows.append({
                            'jar_group': group,
                            'component': comp,
                            'measure': mtype,
                            'mean': vals.mean(),
                            'sem': vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                            'n': len(vals),
                        })
        results['jar_summary'] = pd.DataFrame(jar_rows)
    else:
        results['jar_summary'] = pd.DataFrame()

    # Difference waves
    logger.info("Computing difference waves...")
    results['diff_waves'] = compute_difference_waves(
        all_epochs, all_trial_info, config, logger
    )

    # Bandpower features for ML
    logger.info("Extracting bandpower features...")
    results['bandpower_features'] = extract_bandpower_features(
        all_epochs, all_trial_info, config, logger
    )

    # Extended ML features (Hjorth + time-domain + spectral + bandpower per subject×condition)
    logger.info("Extracting extended ML features...")
    results['ml_features'] = extract_extended_features(
        all_epochs, all_trial_info, config, logger
    )

    # Merge ERP component measures into ml_features
    erp_cols = [c for c in results['measures'].columns
                if c not in ('jar_group', 'condition_label')]
    results['ml_features'] = results['ml_features'].merge(
        results['measures'][erp_cols],
        on=['subject_id', 'condition'], how='left'
    )
    logger.info(f"ML features merged with ERP measures: {results['ml_features'].shape}")

    # Save results
    results_dir = os.path.join(config['paths']['results_base'], 'erp')
    ensure_dir(results_dir)

    results['measures'].to_csv(
        os.path.join(results_dir, 'component_measures.csv'), index=False
    )
    results['concentration_summary'].to_csv(
        os.path.join(results_dir, 'concentration_summary.csv'), index=False
    )
    if len(results.get('jar_summary', pd.DataFrame())) > 0:
        results['jar_summary'].to_csv(
            os.path.join(results_dir, 'jar_summary.csv'), index=False
        )
    results['bandpower_features'].to_csv(
        os.path.join(results_dir, 'bandpower_features.csv'), index=False
    )
    results['ml_features'].to_csv(
        os.path.join(results_dir, 'ml_features.csv'), index=False
    )

    logger.info(f"ERP results saved to {results_dir}")
    return results
