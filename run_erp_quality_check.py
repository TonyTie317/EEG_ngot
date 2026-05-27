#!/usr/bin/env python3
"""
ERP Quality Check — phân tích chất lượng ERP từng subject, từng condition.

CẢI TIẾN: Tập trung vào condition-averaged ERP (không dùng single-trial consistency
vì gERP có SNR thấp ở mức single-trial).

Mục tiêu:
  - Với mỗi subject × condition, phát hiện condition nào có ERP pattern thật sự
    (tín hiệu) và condition nào chỉ là nhiễu (noise).
  - Dùng các chỉ số: SNR của average, component detectability (P1, N1, P2, N400),
    signal-to-noise floor ratio, waveform morphology.
  - Ghi flag để loại bỏ subject/condition kém chất lượng khỏi group-level analysis.

Usage:
    .venv/bin/python run_erp_quality_check.py
    .venv/bin/python run_erp_quality_check.py --subjects P001 P002
    .venv/bin/python run_erp_quality_check.py --verbose
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mne
from scipy.stats import pearsonr

from pipeline.config import load_config, setup_logging, ensure_dir
from pipeline.constants import (
    ALL_SUBJECTS, CONCENTRATIONS, CONCENTRATION_LABELS,
    ERP_WINDOWS, ERP_ROI, ERP_PEAK_MODE,
)

EPOCHS_BASE = 'output/epochs'
TRIAL_INFO_CSV = 'output/epochs/all_trial_info.csv'
QUALITY_OUT = 'output/results/erp/erp_quality_flags.csv'
QUALITY_REPORT = 'output/results/erp/erp_quality_report.txt'
TRIAL_LEVEL_OUT = 'output/results/erp/erp_quality_per_trial.csv'
SUBJ_SUMMARY_OUT = 'output/results/erp/erp_quality_subject_summary.csv'


# ═══════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════

def _baseline_noise(epochs, tmin=-0.5, tmax=-0.3):
    """RMS của baseline period cho mỗi epoch. Trả về (n_epochs,)."""
    data = epochs.get_data()
    mask = (epochs.times >= tmin) & (epochs.times <= tmax)
    return np.sqrt(np.mean(data[:, :, mask] ** 2, axis=(1, 2)))


def _epoch_snr(epochs, sig_win=(0.0, 1.0), base_win=(-0.5, -0.3)):
    """SNR = RMS(signal window) / RMS(baseline) — per epoch."""
    data = epochs.get_data()
    t = epochs.times
    bm = (t >= base_win[0]) & (t <= base_win[1])
    sm = (t >= sig_win[0]) & (t <= sig_win[1])
    base_rms = np.sqrt(np.mean(data[:, :, bm] ** 2, axis=(1, 2)))
    sig_rms = np.sqrt(np.mean(data[:, :, sm] ** 2, axis=(1, 2)))
    base_rms = np.clip(base_rms, 1e-12, None)
    return sig_rms / base_rms


def _detect_peak(waveform, times, window, mode='pos'):
    """Detect peak trong 1 window.

    Parameters
    ----------
    waveform : ndarray (n_times,) — signal 1 chiều (đã trung bình ROI)
    times : ndarray (n_times,)
    window : (tmin, tmax)
    mode : 'pos' | 'neg'

    Returns
    -------
    detected : bool
    peak_amp : float
    peak_lat : float
    mean_amp : float
    """
    mask = (times >= window[0]) & (times <= window[1])
    if mask.sum() == 0:
        return False, 0.0, 0.0, 0.0
    seg = waveform[mask]
    ts = times[mask]
    if mode == 'pos':
        idx = np.argmax(seg)
    else:
        idx = np.argmin(seg)
    peak_amp = seg[idx]
    peak_lat = ts[idx]
    mean_amp = float(np.mean(seg))

    noise_std = float(np.std(waveform))
    if noise_std < 1e-15:
        noise_std = 1e-12
    if mode == 'pos':
        detected = peak_amp > 0 and peak_amp > noise_std * 0.5
    else:
        detected = peak_amp < 0 and abs(peak_amp) > noise_std * 0.5
    return detected, peak_amp, peak_lat, mean_amp


def _condition_avg(epochs, trial_info, condition):
    """Lấy waveform trung bình của 1 condition: (n_ch, n_t)."""
    mask = (trial_info['condition'] == condition).values
    data = epochs.get_data()
    return data[mask].mean(axis=0)


def _condition_avg_snr(epochs, trial_info, condition):
    """Tính SNR của condition-averaged ERP.

    Công thức: SNR = variance(signal_avg) / mean(variance(trial_noise) / n_trials)
    - signal_avg: waveform trung bình qua các trial (n_ch, n_t)
    - trial_noise: trial residuals sau khi trừ average

    Trả về SNR theo từng channel, sau đó average.
    """
    mask = (trial_info['condition'] == condition).values
    if mask.sum() < 2:
        return 0.0
    data = epochs.get_data()
    cond_data = data[mask]  # (n_trials, n_ch, n_t)
    n_trials = cond_data.shape[0]

    # Average signal
    avg = cond_data.mean(axis=0)  # (n_ch, n_t)

    # Residuals: trial - average
    residuals = cond_data - avg[np.newaxis, ...]  # (n_trials, n_ch, n_t)

    # Noise variance: mean across trials of variance per timepoint
    # variance của residuals tại mỗi (ch, t): mean across trials
    noise_var = np.var(residuals, axis=0, ddof=1)  # (n_ch, n_t)

    # Signal variance across time (post-stimulus)
    t = epochs.times
    post_mask = (t >= 0.0) & (t <= 1.0)
    if post_mask.sum() == 0:
        post_mask = (t >= 0.0) & (t <= t[-1])

    signal_var = np.var(avg[:, post_mask], axis=1)  # (n_ch,) — variance across time

    # Noise: mean noise variance across time
    noise_mean_var = np.mean(noise_var[:, post_mask], axis=1)  # (n_ch,)

    # SNR per channel: signal_var / (noise_mean_var / n_trials)
    noise_mean_var = np.clip(noise_mean_var, 1e-20, None)
    snr_per_ch = signal_var / (noise_mean_var / n_trials)

    # Trả về mean SNR across channels
    return float(np.mean(snr_per_ch))


def _poststimulus_rms(data_2d, times):
    """RMS của tín hiệu sau stimulus."""
    mask = (times >= 0.0) & (times <= 1.0)
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean(data_2d[:, mask] ** 2)))


def _morphology_score(cond_avg, ch_names, times, config):
    """Đánh giá hình dạng ERP waveform — có giống ERP điển hình không.

    Kiểm tra:
    1. P1 positive trong P1 window
    2. N1 negative trong N1 window (hoặc ít nhất âm hơn P1)
    3. P2 positive trong P2 window (và lớn hơn P1)
    4. N400 negative (hoặc giảm dần sau P2)
    5. Có sự chuyển đổi cực tính giữa các component liên tiếp

    Returns
    -------
    score : float — 0.0 đến 1.0
    details : dict
    """
    erp_cfg = config.get('erp_analysis', {})
    components = ['P1', 'N1', 'P2', 'N400']
    windows = {}
    rois = {}
    modes = {}

    for comp in components:
        wkey = f'{comp.lower()}_window'
        rkey = f'{comp.lower()}_roi'
        windows[comp] = erp_cfg.get(wkey, ERP_WINDOWS[comp])
        rois[comp] = [c for c in erp_cfg.get(rkey, ERP_ROI.get(comp, [])) if c in ch_names]
        if not rois[comp]:
            rois[comp] = ch_names[:min(8, len(ch_names))]
        modes[comp] = ERP_PEAK_MODE[comp]

    # Lấy waveform trung bình ROI cho mỗi component
    roi_wfs = {}
    for comp in components:
        ch_idx = [ch_names.index(c) for c in rois[comp] if c in ch_names]
        if not ch_idx:
            ch_idx = list(range(min(8, cond_avg.shape[0])))
        roi_wfs[comp] = cond_avg[ch_idx].mean(axis=0)

    # 1. P1 positive
    mask_p1 = (times >= windows['P1'][0]) & (times <= windows['P1'][1])
    p1_pos = float(np.mean(roi_wfs['P1'][mask_p1])) if mask_p1.sum() > 0 else 0

    # 2. N1 negative (trong N1 ROI, ở N1 window)
    mask_n1 = (times >= windows['N1'][0]) & (times <= windows['N1'][1])
    n1_val = float(np.mean(roi_wfs['N1'][mask_n1])) if mask_n1.sum() > 0 else 0

    # 3. P2 positive (P2 ROI, P2 window)
    mask_p2 = (times >= windows['P2'][0]) & (times <= windows['P2'][1])
    p2_val = float(np.mean(roi_wfs['P2'][mask_p2])) if mask_p2.sum() > 0 else 0

    # 4. N400 (giảm so với P2)
    mask_n4 = (times >= windows['N400'][0]) & (times <= windows['N400'][1])
    n4_val = float(np.mean(roi_wfs['N400'][mask_n4])) if mask_n4.sum() > 0 else 0

    # Tính điểm
    checks = []

    # P1 phải dương
    checks.append(('P1_positive', p1_pos > 0 and abs(p1_pos) > 1e-8))

    # N1 nên âm hơn P1 (có inflection)
    checks.append(('N1_below_P1', n1_val < p1_pos))

    # P2 nên có biên độ đáng kể và dương
    checks.append(('P2_positive', p2_val > 0 and abs(p2_val) > 1e-8))

    # P2 nên > P1 (typical gERP pattern)
    checks.append(('P2_gt_P1', p2_val > p1_pos))

    # N400 nên âm hơn P2 (có inflection)
    checks.append(('N400_below_P2', n4_val < p2_val))

    # N400 nên âm hoặc ít nhất thấp hơn baseline
    checks.append(('N400_negative_trend', n4_val < 0 or n4_val < p2_val * 0.5))

    score = sum(1 for _, ok in checks if ok) / len(checks)

    return score, {k: v for k, v in checks}


def _noise_floor(epochs, trial_info, condition):
    """Trial-to-trial variability: RMS của variance across trials."""
    mask = (trial_info['condition'] == condition).values
    if mask.sum() < 2:
        return 0.0
    data = epochs.get_data()
    var_map = np.var(data[mask], axis=0, ddof=1)
    return float(np.sqrt(np.mean(var_map)))


# ═══════════════════════════════════════════════════════════════════════════
# Per-condition analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_condition(epochs, trial_info, condition, ch_names, config):
    """Phân tích chất lượng ERP cho 1 subject × condition.

    Returns
    -------
    row : dict — các metrics
    """
    erp_cfg = config.get('erp_analysis', {})
    cond_avg = _condition_avg(epochs, trial_info, condition)
    times = epochs.times
    data = epochs.get_data()

    cond_mask = (trial_info['condition'] == condition).values
    n_trials = int(cond_mask.sum())
    if n_trials == 0:
        return None

    # SNR của condition-averaged ERP (CẢI TIẾN — dùng variance ratio)
    avg_snr = _condition_avg_snr(epochs, trial_info, condition)

    # SNR per-trial (tham khảo)
    snr_all = _epoch_snr(epochs)
    cond_snr = snr_all[cond_mask]
    mean_trial_snr = float(np.mean(cond_snr))

    # Baseline noise
    base_noise = _baseline_noise(epochs)
    mean_base_noise = float(np.mean(base_noise[cond_mask]))

    # Signal RMS (condition average)
    signal_rms = _poststimulus_rms(cond_avg, times)

    # Noise floor
    noise_floor = _noise_floor(epochs, trial_info, condition)

    # Signal-to-noise-floor ratio
    snf_ratio = signal_rms / noise_floor if noise_floor > 1e-12 else 1.0

    # Component detectability (trên condition average)
    components = ['P1', 'N1', 'P2', 'N400']
    comp_detected = 0
    comp_details = {}

    for comp in components:
        wkey = f'{comp.lower()}_window'
        rkey = f'{comp.lower()}_roi'
        window = erp_cfg.get(wkey, ERP_WINDOWS[comp])
        roi = erp_cfg.get(rkey, ERP_ROI.get(comp, []))
        mode = ERP_PEAK_MODE[comp]

        ch_idx = [ch_names.index(c) for c in roi if c in ch_names]
        if not ch_idx:
            ch_idx = list(range(min(8, cond_avg.shape[0])))

        roi_wf = cond_avg[ch_idx].mean(axis=0)

        detected, pa, pl, ma = _detect_peak(roi_wf, times, window, mode)
        if detected:
            comp_detected += 1
        comp_details[comp] = {
            'detected': detected,
            'peak_amp': pa,
            'peak_lat': pl,
            'mean_amp': ma,
        }

    # Morphology score (hình dạng ERP có giống ERP điển hình?)
    morph_score, morph_checks = _morphology_score(cond_avg, ch_names, times, config)

    # ── Composite quality score (0–1) ──────────────────────────────────────
    avg_snr_score = min(1.0, avg_snr / 3.0)       # avg SNR=3 -> 1.0
    comp_score = comp_detected / 4.0               # 4/4 comp -> 1.0
    morph_norm = morph_score                       # 0-1
    snf_score = min(1.0, snf_ratio / 2.0)          # SNF ratio=2 -> 1.0

    quality_score = (
        0.25 * avg_snr_score +
        0.35 * comp_score +
        0.25 * morph_norm +
        0.15 * snf_score
    )

    # Flag: có ERP pattern thật sự?
    # Tiêu chí mới (phù hợp gERP):
    # - avg SNR >= 2.0 (signal variance gấp đôi noise variance)
    # - hoặc comp_detected >= 3 + morph_score >= 0.5
    # - hoặc comp_detected >= 2 + avg_snr >= 3.0
    has_pattern = (
        (avg_snr >= 2.0 and comp_detected >= 2)
        or (comp_detected >= 3 and morph_score >= 0.5)
        or (comp_detected >= 2 and avg_snr >= 3.0)
        or (comp_detected >= 4)
    )

    # 3 mức label
    if has_pattern and quality_score >= 0.5:
        label = 'GOOD'
    elif has_pattern:
        label = 'WEAK'
    else:
        label = 'BAD'

    row = {
        'subject_id': trial_info['subject_id'].iloc[0],
        'condition': condition,
        'condition_label': CONCENTRATION_LABELS.get(condition, str(condition)),
        'n_trials': n_trials,

        # SNR metrics
        'avg_snr': avg_snr,                # SNR của condition-averaged ERP (CẢI TIẾN)
        'mean_trial_snr': mean_trial_snr,  # Tham khảo: mean SNR từ single-trial

        # Noise
        'mean_baseline_noise_V': mean_base_noise,
        'signal_rms_V': signal_rms,
        'noise_floor_V': noise_floor,
        'signal_to_noise_floor_ratio': snf_ratio,

        # Component / morphology
        'n_components_detected': comp_detected,
        'morphology_score': morph_score,

        # Final
        'quality_score': quality_score,
        'has_real_pattern': has_pattern,
        'quality_label': label,
    }

    # Component details
    for comp in components:
        row[f'{comp}_detected'] = comp_details[comp]['detected']
        row[f'{comp}_peak_amp_V'] = comp_details[comp]['peak_amp']
        row[f'{comp}_peak_lat_s'] = comp_details[comp]['peak_lat']
        row[f'{comp}_mean_amp_V'] = comp_details[comp]['mean_amp']

    return row


def analyze_trials(epochs, trial_info, ch_names):
    """Phân tích chất lượng cấp trial — trả về list of dict."""
    rows = []
    data = epochs.get_data()
    times = epochs.times
    n_ep = len(epochs)

    snr_all = _epoch_snr(epochs)
    base_noise = _baseline_noise(epochs)

    for ep_i in range(n_ep):
        cond = trial_info.iloc[ep_i]['condition']
        post_rms = _poststimulus_rms(data[ep_i], times)
        rows.append({
            'subject_id': trial_info.iloc[ep_i]['subject_id'],
            'epoch_ix': int(trial_info.iloc[ep_i]['epoch_ix']),
            'condition': int(cond),
            'condition_label': CONCENTRATION_LABELS.get(int(cond), str(int(cond))),
            'repeat': int(trial_info.iloc[ep_i]['repeat']),
            'baseline_rms_V': float(base_noise[ep_i]),
            'snr_0_1s': float(snr_all[ep_i]),
            'poststimulus_rms_V': float(post_rms),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Report printing
# ═══════════════════════════════════════════════════════════════════════════

def build_report(quality_df, subj_summary):
    """Tạo báo cáo text."""
    n_good = (quality_df['quality_label'] == 'GOOD').sum()
    n_weak = (quality_df['quality_label'] == 'WEAK').sum()
    n_bad = (quality_df['quality_label'] == 'BAD').sum()
    n_total = len(quality_df)
    n_pattern = quality_df['has_real_pattern'].sum()

    lines = [
        '=' * 72,
        '  BÁO CÁO CHẤT LƯỢNG ERP — THEO TỪNG SUBJECT × CONDITION',
        '=' * 72,
        f'  Tổng số: {n_total} subject×condition',
        f'  • GOOD (pattern rõ, giữ lại):  {n_good:3d} ({n_good/n_total*100:.0f}%)',
        f'  • WEAK (pattern yếu, thận trọng): {n_weak:3d} ({n_weak/n_total*100:.0f}%)',
        f'  • BAD (nhiễu, nên loại):       {n_bad:3d} ({n_bad/n_total*100:.0f}%)',
        f'  • has_real_pattern=True:         {n_pattern:3d} ({n_pattern/n_total*100:.0f}%)',
        '',
    ]

    # BAD list
    bad = quality_df[quality_df['quality_label'] == 'BAD']
    lines.append('─' * 72)
    lines.append('  SUBJECT × CONDITION — BAD (nên loại khỏi phân tích nhóm)')
    lines.append('─' * 72)
    if len(bad):
        lines.append(f'  {"Subject":6s} {"Condition":16s} {"AvgSNR":7s} {"Comp":5s} '
                      f'{"Morph":6s} {"Score":6s} {"SNFratio":9s}')
        for _, r in bad.iterrows():
            lines.append(
                f'  {r["subject_id"]:6s} {r["condition_label"]:16s} '
                f'{r["avg_snr"]:7.1f}  {r["n_components_detected"]}/4  '
                f'{r["morphology_score"]:6.2f} {r["quality_score"]:6.3f} '
                f'{r["signal_to_noise_floor_ratio"]:9.1f}')
    else:
        lines.append('  (Không có)')

    # WEAK list
    weak = quality_df[quality_df['quality_label'] == 'WEAK']
    lines.extend([
        '',
        '─' * 72,
        '  SUBJECT × CONDITION — WEAK (cần xem xét trước khi loại)',
        '─' * 72,
    ])
    if len(weak):
        lines.append(f'  {"Subject":6s} {"Condition":16s} {"AvgSNR":7s} {"Comp":5s} '
                      f'{"Morph":6s} {"Score":6s} {"SNFratio":9s}')
        for _, r in weak.iterrows():
            lines.append(
                f'  {r["subject_id"]:6s} {r["condition_label"]:16s} '
                f'{r["avg_snr"]:7.1f}  {r["n_components_detected"]}/4  '
                f'{r["morphology_score"]:6.2f} {r["quality_score"]:6.3f} '
                f'{r["signal_to_noise_floor_ratio"]:9.1f}')
    else:
        lines.append('  (Không có)')

    # Subject summary
    lines.extend([
        '',
        '─' * 72,
        '  THỐNG KÊ THEO SUBJECT',
        '─' * 72,
        f'  {"Subj":5s} {"N":3s} {"GOOD":5s} {"WEAK":5s} {"BAD":4s} '
        f'{"Quality":7s} {"AvgSNR":7s} {"CóPatt":6s}',
        '─' * 72,
    ])
    for _, r in subj_summary.iterrows():
        lines.append(
            f'  {r["subject_id"]:5s} {int(r["n_conditions"]):3d} {int(r["n_good"]):5d} '
            f'{int(r["n_weak"]):5d} {int(r["n_bad"]):4d} '
            f'{r["mean_quality"]:7.3f} {r["mean_avg_snr"]:7.1f} '
            f'{int(r["n_with_pattern"]):2d}/{int(r["n_conditions"]):2d}')
    lines.append('─' * 72)

    # Detail: từng subject × condition
    lines.extend([
        '',
        '=' * 72,
        '  CHI TIẾT TỪNG SUBJECT × CONDITION',
        '=' * 72,
        f'  {"Subj":5s} {"Condition":10s} {"Label":8s} {"AvgSNR":6s} {"Comps":5s} '
        f'{"Morph":5s} {"Score":6s} {"P1":3s} {"N1":3s} {"P2":3s} {"N400":4s}',
        '─' * 72,
    ])
    for _, r in quality_df.iterrows():
        p1 = 'Y' if r['P1_detected'] else '.'
        n1 = 'Y' if r['N1_detected'] else '.'
        p2 = 'Y' if r['P2_detected'] else '.'
        n4 = 'Y' if r['N400_detected'] else '.'
        lines.append(
            f'  {r["subject_id"]:5s} {r["condition_label"]:10s} '
            f'{r["quality_label"]:8s} {r["avg_snr"]:6.1f}  '
            f'{r["n_components_detected"]}/4  {r["morphology_score"]:5.2f} '
            f'{r["quality_score"]:6.3f}  {p1:3s} {n1:3s} {p2:3s} {n4:4s}')

    lines.extend([
        '',
        '=' * 72,
        '  GHI CHÚ:',
        f'  • File flag:        {QUALITY_OUT}',
        f'  • File per-trial:   {TRIAL_LEVEL_OUT}',
        f'  • File subj summary: {SUBJ_SUMMARY_OUT}',
        '',
        '  Tiêu chí "has_real_pattern" = True:',
        '    1) avg_SNR ≥ 2.0 + ≥ 2/4 components detected',
        '    2) Hoặc ≥ 3/4 components + morphology ≥ 0.5',
        '    3) Hoặc ≥ 2/4 components + avg_SNR ≥ 3.0',
        '    4) Hoặc 4/4 components',
        '',
        '  avg_SNR = variance(signal_avg) / [mean(variance(residuals)) / n_trials]',
        '  Đây là SNR thực của condition-averaged ERP (không phải single-trial).',
        '',
        '  Sử dụng filter cho group analysis:',
        '    flags = pd.read_csv("output/results/erp/erp_quality_flags.csv")',
        "    ok = flags[flags['quality_label'].isin(['GOOD', 'WEAK'])]",
        '  Hoặc strict:',
        "    ok = flags[flags['has_real_pattern'] == True]",
        '=' * 72,
    ])

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ERP Quality Check per subject')
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    subjects = args.subjects if args.subjects else ALL_SUBJECTS

    if not os.path.exists(TRIAL_INFO_CSV):
        logger.error(f'Không tìm thấy {TRIAL_INFO_CSV}')
        sys.exit(1)
    all_trial_info = pd.read_csv(TRIAL_INFO_CSV)

    quality_rows = []
    trial_rows = []

    for sid in subjects:
        ep_dir = os.path.join(EPOCHS_BASE, sid)
        fif_path = os.path.join(ep_dir, 'epochs_epo.fif')
        if not os.path.exists(fif_path):
            logger.warning(f'[{sid}] Không tìm thấy epochs, bỏ qua')
            continue

        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        ti = all_trial_info[all_trial_info['subject_id'] == sid].reset_index(drop=True)
        ch_names = epochs.ch_names
        logger.info(f'[{sid}] {len(epochs)} epochs, {len(ch_names)} channels')

        # Trial-level
        trial_rows.extend(analyze_trials(epochs, ti, ch_names))

        # Condition-level
        for cond in CONCENTRATIONS:
            row = analyze_condition(epochs, ti, cond, ch_names, config)
            if row is not None:
                quality_rows.append(row)

    if not quality_rows:
        logger.error('Không có dữ liệu quality nào!')
        sys.exit(1)

    quality_df = pd.DataFrame(quality_rows)
    trial_df = pd.DataFrame(trial_rows)

    # Merge JAR
    jar_lookup = all_trial_info.groupby(
        ['subject_id', 'condition']
    )['jar_group'].first().reset_index()
    quality_df = quality_df.merge(jar_lookup, on=['subject_id', 'condition'], how='left')

    # Subject-level summary
    subj_summary = quality_df.groupby('subject_id').agg(
        n_conditions=('quality_label', 'count'),
        n_good=('quality_label', lambda x: (x == 'GOOD').sum()),
        n_weak=('quality_label', lambda x: (x == 'WEAK').sum()),
        n_bad=('quality_label', lambda x: (x == 'BAD').sum()),
        n_with_pattern=('has_real_pattern', 'sum'),
        mean_quality=('quality_score', 'mean'),
        mean_avg_snr=('avg_snr', 'mean'),
    ).reset_index()

    # Lưu
    ensure_dir(os.path.dirname(QUALITY_OUT))
    quality_df.to_csv(QUALITY_OUT, index=False)
    trial_df.to_csv(TRIAL_LEVEL_OUT, index=False)
    subj_summary.to_csv(SUBJ_SUMMARY_OUT, index=False)

    # Báo cáo
    report = build_report(quality_df, subj_summary)
    print(report)
    with open(QUALITY_REPORT, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f'Done. Report: {QUALITY_REPORT}')
    logger.info(f'Flags:    {QUALITY_OUT}')


if __name__ == '__main__':
    main()
