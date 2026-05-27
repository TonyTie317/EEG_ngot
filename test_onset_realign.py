"""
ERP Onset Re-alignment Script
==============================
Phát hiện thời điểm kích thích THỰC SỰ (t=0 thật) thay vì dùng đầu trigger.

Ba chiến lược:
  A. RMS jump     — Tìm thời điểm biên độ EEG tăng đột ngột (artifact nuốt/cơ mặt)
  B. Woody Filter — Cross-correlation để align các epoch với nhau (không cần biết artifact)
  C. Combined     — Dùng A để khởi tạo, B để tinh chỉnh

Cách dùng:
  python test_onset_realign.py                        # xem thử P001
  python test_onset_realign.py --subjects P001 P002   # nhiều subject
  python test_onset_realign.py --strategy woody       # chỉ dùng Woody Filter
  python test_onset_realign.py --apply                # lưu offset ra file để dùng lại

Output:
  output/figures/erp_visual_inspect/realign_*.png     — Đồ thị so sánh trước/sau
  output/epochs/realign_offsets.csv                   — Offset (samples) per trial
"""

import os
import sys
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

SFREQ       = 100          # Hz
SEARCH_WIN  = 200          # samples = 2s — cửa sổ tìm onset thật
MIN_OFFSET  = 0            # samples — offset tối thiểu sau trigger
MAX_OFFSET  = 200          # samples = 2s — offset tối đa sau trigger
# Epoch window sau re-align: 0.5s trước + 1s sau true onset
EPOCH_TMIN  = -0.5         # s trước true onset (sau re-align)
EPOCH_TMAX  =  1.0         # s sau true onset (sau re-align)
CONFIG_PATH = "configs/config.yaml"
OUTPUT_DIR  = "output/figures/erp_visual_inspect"
OFFSET_CSV  = "output/epochs/realign_offsets.csv"

EEG_COLS = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T3','T4','T5','T6']


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        datefmt='%H:%M:%S')
    return logging.getLogger('realign')


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_csv_path(sid, config):
    pattern = config['paths']['csv_pattern']
    return os.path.join(config['paths']['raw_data'],
                        pattern.format(subject=sid))


def load_raw_df(sid, config):
    path = get_csv_path(sid, config)
    df = pd.read_csv(path)
    # Rename legacy channel names
    rename = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    df = df.rename(columns=rename)
    return df


def detect_triggers(df):
    """Trả về list of dict: {start, end, cond, repeat, jar}."""
    vals = df['ma_mau'].to_numpy()
    triggers = []
    i = 0
    N = len(vals)
    while i < N:
        if not pd.isna(vals[i]) and vals[i] != 0:
            code = vals[i]
            j = i
            while j + 1 < N and not pd.isna(vals[j+1]) and vals[j+1] == code:
                j += 1
            seg = df.iloc[i:j+1]
            rep_vals = seg['repeat'].dropna()
            jar_vals = seg['JAR'].dropna()
            triggers.append({
                'start':  i,
                'end':    j,
                'cond':   int(code),
                'repeat': int(rep_vals.iloc[0]) if len(rep_vals) else -1,
                'jar':    float(jar_vals.iloc[0]) if len(jar_vals) else np.nan,
            })
            i = j + 1
        else:
            i += 1
    return triggers


def get_eeg_matrix(df, available_cols=None):
    """Lấy ma trận EEG (n_samples, n_ch) đã scale µV→µV (giữ nguyên)."""
    cols = [c for c in EEG_COLS if c in df.columns]
    if available_cols:
        cols = [c for c in cols if c in available_cols]
    return df[cols].to_numpy(dtype=np.float64), cols


# ─────────────────────────────────────────────────────────────────────────────
# Chiến lược A: RMS jump detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_onset_rms(eeg_window, baseline_rms=None,
                     threshold_factor=2.5, smooth_win=5):
    """
    Tìm mẫu đầu tiên trong cửa sổ mà RMS cross-channel tăng vọt.

    Parameters
    ----------
    eeg_window : (n_samples, n_ch)  — vùng tìm kiếm (MAX_OFFSET mẫu)
    baseline_rms : float | None     — RMS nền; nếu None tự tính từ 20 mẫu đầu
    threshold_factor : float        — bội số ngưỡng so với baseline
    smooth_win : int                — làm mượt RMS trước khi detect

    Returns
    -------
    offset : int  — số mẫu lệch từ đầu trigger (0 = không lệch)
    rms_curve : (n_samples,)  — đường RMS để vẽ
    """
    n = len(eeg_window)
    # RMS theo thời gian (cross-channel)
    rms = np.sqrt(np.mean(eeg_window ** 2, axis=1))

    # Làm mượt
    kernel = np.ones(smooth_win) / smooth_win
    rms_smooth = np.convolve(rms, kernel, mode='same')

    # Baseline từ 20 mẫu đầu (trước khi stimulus)
    if baseline_rms is None:
        baseline_rms = rms_smooth[:20].mean()
    baseline_std = rms_smooth[:20].std() + 1e-6

    threshold = baseline_rms + threshold_factor * baseline_std

    # Tìm lần vượt ngưỡng đầu tiên sau MIN_OFFSET
    candidates = np.where(rms_smooth[MIN_OFFSET:] > threshold)[0]
    if len(candidates) > 0:
        offset = int(candidates[0]) + MIN_OFFSET
    else:
        offset = 0  # không tìm thấy → giữ nguyên trigger

    return offset, rms_smooth


# ─────────────────────────────────────────────────────────────────────────────
# Chiến lược B: Woody Filter (cross-correlation alignment)
# ─────────────────────────────────────────────────────────────────────────────
def woody_filter(epochs_list, max_shift=None, n_iter=5, roi_ch_idx=None):
    """
    Align epochs bằng cách tìm lag tối đa hóa cross-correlation với average.

    Parameters
    ----------
    epochs_list : list of (n_ch, n_times) arrays  — các epoch raw (sau trigger)
    max_shift : int  — giới hạn shift (samples). Default = MAX_OFFSET
    n_iter : int     — số vòng lặp
    roi_ch_idx : list of int | None  — kênh dùng để align (None = tất cả)

    Returns
    -------
    offsets : (n_epochs,) int  — số mẫu shift dương = trễ so với trigger
    aligned_epochs : list of (n_ch, n_times)  — epoch đã align (cùng độ dài)
    """
    if max_shift is None:
        max_shift = MAX_OFFSET

    n_epochs = len(epochs_list)
    if roi_ch_idx is None:
        roi_ch_idx = list(range(epochs_list[0].shape[0]))

    # Chiều dài tối thiểu để align
    min_len = min(ep.shape[1] for ep in epochs_list) - max_shift
    if min_len <= 0:
        return np.zeros(n_epochs, dtype=int), epochs_list

    offsets = np.zeros(n_epochs, dtype=int)

    for iteration in range(n_iter):
        # Tính average epoch với offset hiện tại (lấy vùng cố định)
        avg = np.zeros((len(roi_ch_idx), min_len))
        for i, ep in enumerate(epochs_list):
            s = offsets[i]
            avg += ep[np.ix_(roi_ch_idx, range(s, s + min_len))]
        avg /= n_epochs
        avg_flat = avg.mean(axis=0)  # trung bình kênh → 1D

        # Tìm lag tối đa cross-corr với average
        new_offsets = np.zeros(n_epochs, dtype=int)
        for i, ep in enumerate(epochs_list):
            ep_flat = ep[roi_ch_idx, :].mean(axis=0)
            # Cross-correlation trong phạm vi max_shift
            best_lag = 0
            best_corr = -np.inf
            for lag in range(MIN_OFFSET, max_shift + 1):
                seg = ep_flat[lag: lag + min_len]
                if len(seg) < min_len:
                    break
                corr = np.corrcoef(avg_flat, seg)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_lag  = lag
            new_offsets[i] = best_lag

        delta = np.abs(new_offsets - offsets).mean()
        offsets = new_offsets
        if delta < 0.5:  # converged
            break

    # Tạo aligned epochs (cắt từ offset, cùng độ dài)
    aligned = []
    for i, ep in enumerate(epochs_list):
        s = offsets[i]
        aligned.append(ep[:, s: s + min_len])

    return offsets, aligned


def woody_filter_per_condition(epochs_list, conditions, max_shift=None,
                                n_iter=5, roi_ch_idx=None):
    """Chạy Woody Filter riêng cho từng condition.

    Thay vì align 30 trials với nhau (mix 6 nồng độ), hàm này
    group 5 trials cùng nồng độ lại rồi align trong group đó.
    Template per-condition chính xác hơn về shape ERP.

    Parameters
    ----------
    epochs_list : list of (n_ch, n_times) — epoch raw, đã shift bởi RMS
    conditions  : list of int/str — condition label cho từng trial (cùng thứ tự)
    max_shift   : int — giới hạn shift trong Woody (samples)
    n_iter      : int
    roi_ch_idx  : list of int | None

    Returns
    -------
    offsets : (n_epochs,) int — offset Woody per trial
    """
    n_epochs = len(epochs_list)
    offsets  = np.zeros(n_epochs, dtype=int)
    cond_arr = np.array(conditions)

    for cond in np.unique(cond_arr):
        idx = np.where(cond_arr == cond)[0]          # chỉ số trong epochs_list
        group_epochs = [epochs_list[i] for i in idx]

        if len(group_epochs) < 2:
            # Chỉ 1 trial, không cần align
            continue

        grp_offsets, _ = woody_filter(
            group_epochs,
            max_shift=max_shift,
            n_iter=n_iter,
            roi_ch_idx=roi_ch_idx,
        )
        for j, i in enumerate(idx):
            offsets[i] = grp_offsets[j]

    return offsets


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline chính cho 1 subject
# ─────────────────────────────────────────────────────────────────────────────
def process_subject(sid, config, strategy='combined', logger=None):
    """
    Trả về DataFrame chứa offset (samples) per trial.
    """
    if logger is None:
        logger = logging.getLogger()

    df = load_raw_df(sid, config)
    eeg, ch_names = get_eeg_matrix(df)
    triggers = detect_triggers(df)

    if not triggers:
        logger.warning(f"[{sid}] Không tìm thấy trigger")
        return None

    logger.info(f"[{sid}] {len(triggers)} triggers")

    # Chuẩn bị epoch windows (MAX_OFFSET + epoch_len mẫu)
    EPOCH_LEN = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ) + 1  # 1s trước + 1s sau true onset = 200 samples
    epochs_raw = []  # (n_ch, SEARCH_WIN + EPOCH_LEN)
    valid_trigger_idx = []

    for ti, trig in enumerate(triggers):
        s = trig['start']
        # Lấy đủ mẫu để shift + epoch
        end = s + MAX_OFFSET + EPOCH_LEN
        if end > len(eeg):
            logger.warning(f"[{sid}] Trial {ti}: không đủ mẫu, bỏ qua")
            continue
        ep = eeg[s: end, :].T  # (n_ch, n_samples)
        epochs_raw.append(ep)
        valid_trigger_idx.append(ti)

    if not epochs_raw:
        return None

    # ── Strategy A: RMS jump ─────────────────────────────────────────────
    offsets_rms = []
    rms_curves  = []
    for ep in epochs_raw:
        win = ep[:, :MAX_OFFSET].T  # (MAX_OFFSET, n_ch)
        # Dùng frontal + central + temporal + parietal channels
        # (bao phủ tất cả ERP ROI: P1→F3/F4/C3/C4, N1→F7/F8/T7/T8, P2→C3/C4/P3/P4)
        roi_idx = [ch_names.index(c) for c in ['Fp1','Fp2','F3','F4','C3','C4','T7','T8','P3','P4']
                     if c in ch_names]
        win_roi = win[:, roi_idx] if roi_idx else win
        offset, rms = detect_onset_rms(win_roi)
        offsets_rms.append(offset)
        rms_curves.append(rms)

    # ── Strategy B: Woody Filter per-condition ───────────────────────────
    # Align 5 trials cùng nồng độ với nhau (template chính xác hơn per-condition)
    roi_idx = [ch_names.index(c) for c in ['Fp1','Fp2','F3','F4','C3','C4','T7','T8','P3','P4']
                 if c in ch_names]
    trial_conditions = [triggers[ti]['cond'] for ti in valid_trigger_idx]
    offsets_woody = woody_filter_per_condition(
        epochs_raw, trial_conditions,
        max_shift=MAX_OFFSET,
        roi_ch_idx=roi_idx if roi_idx else None,
    )

    # ── Strategy C: Combined ─────────────────────────────────────────────
    # RMS làm starting point, Woody per-condition tinh chỉnh thêm
    if strategy == 'combined':
        epochs_rms_shifted = []
        for i, ep in enumerate(epochs_raw):
            s = offsets_rms[i]
            remaining = ep.shape[1] - s
            ep_shifted = ep[:, s:] if remaining > EPOCH_LEN else ep
            epochs_rms_shifted.append(ep_shifted)
        offsets_woody2 = woody_filter_per_condition(
            epochs_rms_shifted, trial_conditions,
            max_shift=MAX_OFFSET // 2,
            roi_ch_idx=roi_idx if roi_idx else None,
        )
        offsets_final = np.array(offsets_rms) + offsets_woody2
    elif strategy == 'rms':
        offsets_final = np.array(offsets_rms)
    else:  # woody
        offsets_final = offsets_woody

    # ── Build result DataFrame ────────────────────────────────────────────
    rows = []
    for i, ti in enumerate(valid_trigger_idx):
        trig = triggers[ti]
        rows.append({
            'subject_id':     sid,
            'trial_ix':       trig['trial_ix'] if 'trial_ix' in trig else ti,
            'condition':      trig['cond'],
            'repeat':         trig['repeat'],
            'trigger_sample': trig['start'],
            'offset_rms':     int(offsets_rms[i]),
            'offset_woody':   int(offsets_woody[i]),
            'offset_final':   int(offsets_final[i]),
            'new_onset':      trig['start'] + int(offsets_final[i]),
            'offset_sec':     offsets_final[i] / SFREQ,
        })

    result_df = pd.DataFrame(rows)

    logger.info(
        f"[{sid}] Offset stats (final, samples): "
        f"mean={offsets_final.mean():.1f} "
        f"std={offsets_final.std():.1f} "
        f"min={offsets_final.min()} "
        f"max={offsets_final.max()}"
    )

    return result_df, epochs_raw, offsets_rms, offsets_woody, offsets_final.tolist(), rms_curves, ch_names


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────
def plot_realign_overview(sid, result_df, epochs_raw, offsets_rms,
                           offsets_woody, offsets_final, rms_curves, ch_names,
                           output_dir, logger):
    """Vẽ 4 panel so sánh offset và waveform trước/sau align."""

    times_ms = np.arange(MAX_OFFSET + int(1.5 * SFREQ) + 1) / SFREQ * 1000

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f'Re-alignment Analysis — {sid}\n'
        f'Tìm thời điểm kích thích thật trong cửa sổ 0–{MAX_OFFSET/SFREQ:.1f}s sau trigger',
        fontsize=13, fontweight='bold'
    )

    # ── Panel 1: Histogram offset ─────────────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.hist(offsets_rms,   bins=15, alpha=0.6, label='RMS jump',    color='#2196F3')
    ax1.hist(offsets_woody, bins=15, alpha=0.6, label='Woody Filter', color='#FF9800')
    ax1.hist(offsets_final, bins=15, alpha=0.7, label='Final',        color='#E91E63')
    ax1.axvline(np.mean(offsets_final), color='black', linestyle='--',
                label=f'Mean={np.mean(offsets_final):.1f}ms')
    ax1.set_xlabel('Offset (samples, 1 sample = 10ms)')
    ax1.set_ylabel('Số trial')
    ax1.set_title('Phân phối offset')
    ax1.legend(fontsize=8)

    # ── Panel 2: Scatter RMS vs Woody ─────────────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.scatter(offsets_rms, offsets_woody, alpha=0.7, c='#4CAF50', s=50)
    lim_max = max(max(offsets_rms), max(offsets_woody)) + 5
    ax2.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4)
    ax2.set_xlabel('Offset RMS (samples)')
    ax2.set_ylabel('Offset Woody (samples)')
    ax2.set_title('RMS vs Woody — đồng thuận?')

    # ── Panel 3: Offset per trial ──────────────────────────────────────────
    ax3 = fig.add_subplot(1, 4, 3)
    trial_ids = result_df['trial_ix'].values
    ax3.plot(trial_ids, offsets_rms,   'o-', label='RMS',    alpha=0.7, markersize=4)
    ax3.plot(trial_ids, offsets_woody, 's-', label='Woody',  alpha=0.7, markersize=4)
    ax3.plot(trial_ids, offsets_final, '^-', label='Final',  linewidth=2, markersize=5)
    ax3.set_xlabel('Trial index')
    ax3.set_ylabel('Offset (samples)')
    ax3.set_title('Offset theo từng trial')
    ax3.legend(fontsize=8)

    # ── Panel 4: RMS curves của 6 trial đầu ──────────────────────────────
    ax4 = fig.add_subplot(1, 4, 4)
    colors = plt.cm.tab10(np.linspace(0, 0.9, min(6, len(rms_curves))))
    for i in range(min(6, len(rms_curves))):
        t = np.arange(len(rms_curves[i])) / SFREQ * 1000
        ax4.plot(t, rms_curves[i], color=colors[i], alpha=0.8,
                 label=f'Trial {i}')
        ax4.axvline(offsets_rms[i] / SFREQ * 1000, color=colors[i],
                    linestyle=':', linewidth=1.5)
    ax4.set_xlabel('ms sau trigger')
    ax4.set_ylabel('RMS (µV)')
    ax4.set_title('RMS curves — đường chấm = onset detected')
    ax4.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    path = os.path.join(output_dir, f'realign_{sid}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_strategy_comparison(sid, result_df, epochs_raw, offsets_rms, offsets_woody,
                              offsets_final, ch_names, output_dir, logger):
    """So sánh 3 chiến lược per-condition: 6 subplot (1/nồng độ), 3 đường chồng nhau."""
    rep_chs = ['C3', 'Cz', 'C4', 'P3']
    rep_ch  = next((c for c in rep_chs if c in ch_names), ch_names[0])
    rep_idx = ch_names.index(rep_ch)

    cond_labels  = result_df['condition'].values
    unique_conds = sorted(np.unique(cond_labels))
    display_len  = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)

    strategies = {
        'Trigger gốc': (np.zeros(len(epochs_raw), dtype=int), '#607D8B', '--'),
        'RMS jump':    (np.array(offsets_rms,    dtype=int),  '#2196F3', '-'),
        'Woody/Final': (np.array(offsets_final,  dtype=int),  '#E91E63', '-'),
    }

    ncols = 3
    nrows = int(np.ceil(len(unique_conds) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 3.5),
                              sharey=False)
    axes = axes.flatten()

    fig.suptitle(
        f'So sánh 3 chiến lược per-condition — {sid} — Kênh {rep_ch}\n'
        f'Mỗi ô = 1 nồng độ | 3 đường = 3 chiến lược | t=0 tương đối theo onset của từng chiến lược',
        fontsize=11, fontweight='bold'
    )

    for k, cond in enumerate(unique_conds):
        ax  = axes[k]
        idx = np.where(cond_labels == cond)[0]

        for strat_name, (offsets, color, ls) in strategies.items():
            cond_mean_off = offsets[idx].mean()

            sigs = []
            for i in idx:
                s     = int(offsets[i])
                avail = epochs_raw[i].shape[1] - s
                sig   = epochs_raw[i][rep_idx, s: s + display_len] if avail >= display_len \
                        else np.pad(epochs_raw[i][rep_idx, s:], (0, display_len - avail))
                sigs.append(sig)

            avg  = np.array(sigs).mean(axis=0)
            # trục t=0 = onset thật của chiến lược đó (EPOCH_TMIN làm gốc)
            t_ms = (np.arange(display_len) / SFREQ + EPOCH_TMIN) * 1000
            ax.plot(t_ms, avg, color=color, linewidth=2.0, linestyle=ls,
                    label=f'{strat_name} (+{cond_mean_off*10:.0f}ms)')

        ax.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.6)
        ax.axhline(0, color='grey',  linewidth=0.4)
        ax.set_title(f'ma_mau={cond} | {len(idx)} trials', fontsize=9)
        ax.set_xlabel('ms (t=0 = onset của chiến lược)', fontsize=8)
        ax.set_ylabel('µV', fontsize=8)
        ax.set_xlim(EPOCH_TMIN * 1000, EPOCH_TMAX * 1000)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)

    for j in range(len(unique_conds), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f'realign_compare_{sid}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_per_condition_erp(sid, result_df, epochs_raw, offsets_final,
                           ch_names, output_dir, logger):
    """
    Vẽ 6 subplot (1 per nồng độ): trước vs sau align trên cùng trục trigger.
    Mỗi subplot có đường dọc riêng = onset thật của nồng độ đó.
    """
    rep_chs = ['C3', 'Cz', 'C4', 'P3']
    rep_ch  = next((c for c in rep_chs if c in ch_names), ch_names[0])
    rep_idx = ch_names.index(rep_ch)

    display_len  = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)
    max_t        = min(ep.shape[1] for ep in epochs_raw)
    t_before_ms  = np.arange(max_t) / SFREQ * 1000

    cond_labels  = result_df['condition'].values
    unique_conds = sorted(np.unique(cond_labels))
    ncols = 3
    nrows = int(np.ceil(len(unique_conds) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5, nrows * 3.5),
                              sharey=False)
    axes = axes.flatten()

    fig.suptitle(
        f'Per-Condition ERP — {sid} — Kênh {rep_ch}\n'
        f'Xanh đứt = trước align | Đỏ liền = sau align | Đường dọc = onset thật của condition đó',
        fontsize=11, fontweight='bold'
    )

    for k, cond in enumerate(unique_conds):
        ax  = axes[k]
        idx = np.where(cond_labels == cond)[0]

        # TRƯỚC align: average 5 trials, tính từ trigger
        sigs_before = [epochs_raw[i][rep_idx, :max_t] for i in idx]
        avg_before  = np.array(sigs_before).mean(axis=0)
        ax.plot(t_before_ms, avg_before, color='#1565C0', linewidth=1.8,
                linestyle='--', alpha=0.85, label='Trước align')

        # SAU align: mỗi trial shift về onset riêng, average trên trục trigger
        cond_offsets = [offsets_final[i] for i in idx]
        cond_mean_off_samp = np.mean(cond_offsets)
        cond_mean_off_ms   = cond_mean_off_samp * 1000 / SFREQ

        sigs_after = []
        for i in idx:
            s     = int(offsets_final[i])
            avail = epochs_raw[i].shape[1] - s
            sig   = epochs_raw[i][rep_idx, s: s + display_len] if avail >= display_len \
                    else np.pad(epochs_raw[i][rep_idx, s:], (0, display_len - avail))
            sigs_after.append(sig)

        avg_after    = np.array(sigs_after).mean(axis=0)
        t_after_ms   = (np.arange(display_len) / SFREQ + cond_mean_off_samp / SFREQ) * 1000
        ax.plot(t_after_ms, avg_after, color='#C62828', linewidth=2.0,
                label=f'Sau align')

        # Đường dọc: trigger gốc (t=0) và onset thật của condition này
        ax.axvline(0,                color='#1565C0', linewidth=1.5,
                   linestyle='--', alpha=0.7, label='t=0 Trigger')
        ax.axvline(cond_mean_off_ms, color='#C62828', linewidth=1.5,
                   linestyle='--', alpha=0.7,
                   label=f'Onset ~+{cond_mean_off_ms:.0f}ms')
        ax.axhline(0, color='grey', linewidth=0.4)

        std_off = np.std(cond_offsets) * 1000 / SFREQ
        ax.set_title(
            f'ma_mau={cond} | n={len(idx)} trials\n'
            f'Onset: {cond_mean_off_ms:.0f}±{std_off:.0f}ms',
            fontsize=9
        )
        ax.set_xlabel('ms từ trigger', fontsize=8)
        ax.set_ylabel('µV', fontsize=8)
        ax.set_xlim(-200, min(max_t / SFREQ * 1000, 3500))
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)

    for j in range(len(unique_conds), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f'realign_percond_{sid}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved per-condition ERP: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_results):
    if not all_results:
        return
    df = pd.concat(all_results, ignore_index=True)
    print("\n" + "="*70)
    print("  TỔNG KẾT RE-ALIGNMENT OFFSET")
    print("="*70)
    print(f"{'Subject':<10} {'N trials':<10} {'Mean offset':<14} "
          f"{'Std offset':<12} {'Min':<8} {'Max':<8} {'Ghi chú'}")
    print("-"*70)
    for sid, grp in df.groupby('subject_id'):
        offs = grp['offset_final'].values
        note = ""
        if offs.mean() < 5:
            note = "⚠️  Offset rất nhỏ — trigger đã đúng?"
        elif offs.std() > 30:
            note = "⚠️  Jitter lớn — cần kiểm tra"
        else:
            note = "✅ Offset ổn định"
        print(f"{sid:<10} {len(grp):<10} "
              f"{offs.mean()*10:.0f}ms{'':<8} "
              f"{offs.std()*10:.0f}ms{'':<6} "
              f"{offs.min()*10:.0f}ms{'':<3} "
              f"{offs.max()*10:.0f}ms{'':<3} "
              f"{note}")
    print("="*70)
    print(f"\n💡 1 sample = 10ms (SFREQ={SFREQ}Hz)")
    print("   Nếu offset ~0 → trigger đã đúng, không cần re-align")
    print("   Nếu offset ổn định (std nhỏ) → trigger bị trễ cố định → sửa tmin")
    print("   Nếu offset biến thiên lớn (std lớn) → jitter thật → cần Woody Filter\n")


def plot_per_trial_grid(sid, result_df, epochs_raw, offsets_final,
                        ch_names, output_dir, logger):
    """Đã bỏ — nội dung trùng với plot_strategy_comparison và plot_per_condition_erp."""
    return
    rep_chs = ['C3', 'Cz', 'C4', 'P3']
    rep_ch  = next((c for c in rep_chs if c in ch_names), ch_names[0])
    rep_idx = ch_names.index(rep_ch)

    n_trials = len(epochs_raw)
    ncols    = 6
    nrows    = int(np.ceil(n_trials / ncols))

    # Độ dài hiển thị sau align: EPOCH_TMIN → EPOCH_TMAX
    display_len = int((EPOCH_TMAX - EPOCH_TMIN) * SFREQ)
    t_after_ms  = (np.arange(display_len) / SFREQ + EPOCH_TMIN) * 1000  # ms, t=0=onset

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.4),
                              sharex=False, sharey=False)
    axes = axes.flatten()

    conds   = result_df['condition'].values if 'condition' in result_df.columns else ['']*n_trials
    repeats = result_df['repeat'].values    if 'repeat'    in result_df.columns else ['']*n_trials
    jars    = result_df['jar'].values       if 'jar'       in result_df.columns else [np.nan]*n_trials

    for i, ep in enumerate(epochs_raw):
        ax  = axes[i]
        off = int(offsets_final[i])

        # Trước align: lấy cùng số mẫu display_len bắt đầu từ trigger (offset=0)
        # → t_before_ms bắt đầu từ -off/SFREQ*1000 (tính theo true onset làm gốc)
        before_sig = ep[rep_idx, :display_len] if ep.shape[1] >= display_len else \
                     np.pad(ep[rep_idx], (0, display_len - ep.shape[1]))
        t_before_ms = (np.arange(display_len) / SFREQ - off / SFREQ) * 1000

        # Sau align: bắt đầu từ offset, độ dài display_len
        avail = ep.shape[1] - off
        if avail >= display_len:
            after_sig = ep[rep_idx, off: off + display_len]
        else:
            after_sig = np.pad(ep[rep_idx, off:], (0, display_len - avail))

        ax.plot(t_before_ms, before_sig, color='#90CAF9', linewidth=0.9,
                alpha=0.85, label='Trigger gốc')
        ax.plot(t_after_ms,  after_sig,  color='#E53935', linewidth=1.1,
                label='Sau align')

        ax.axvline(0,   color='black',  linewidth=0.8, linestyle='--')  # true onset
        ax.axvline(-off / SFREQ * 1000 + 0, color='#1565C0',
                   linewidth=0.6, linestyle=':')                          # trigger gốc vị trí
        ax.axhline(0,   color='grey',   linewidth=0.4)
        ax.set_xlim(t_after_ms[0], t_after_ms[-1])

        # Y-axis scale riêng cho từng trial
        all_vals = np.concatenate([before_sig, after_sig])
        vmax = np.nanpercentile(np.abs(all_vals), 98) * 1.2
        if vmax > 0:
            ax.set_ylim(-vmax, vmax)

        jar_str = f"JAR={jars[i]:.0f}" if not np.isnan(float(jars[i])) else ""
        ax.set_title(
            f"T{i+1} | c{conds[i]} r{repeats[i]} | off={off*10}ms | {jar_str}",
            fontsize=7, pad=2
        )
        ax.tick_params(labelsize=6)
        ax.set_ylabel('µV', fontsize=6)

    # Tắt các ô thừa
    for j in range(n_trials, len(axes)):
        axes[j].set_visible(False)

    # Legend chung
    handles = [
        plt.Line2D([0], [0], color='#90CAF9', linewidth=1.5, label='Trigger gốc'),
        plt.Line2D([0], [0], color='#E53935',  linewidth=1.5, label='Sau align (Woody)'),
        plt.Line2D([0], [0], color='black', linewidth=1, linestyle='--', label='t=0 onset thật'),
    ]
    fig.legend(handles=handles, loc='lower right', fontsize=8, ncol=3)

    fig.suptitle(
        f'Per-Trial Re-alignment — {sid} — Kênh {rep_ch}\n'
        f'Xanh nhạt = trigger gốc, Đỏ = sau Woody align | đường đứt = true onset (t=0)',
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    path = os.path.join(output_dir, f'realign_pertrial_{sid}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved per-trial grid: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def get_available_subjects(config):
    epoch_base = os.path.join(config['paths']['output_base'], 'epochs')
    subjects = []
    for d in sorted(os.listdir(epoch_base)):
        fif = os.path.join(epoch_base, d, 'epochs_epo.fif')
        if not os.path.exists(fif):
            fif = os.path.join(epoch_base, d, 'epochs.fif')
        if os.path.exists(fif):
            subjects.append(d)
    return subjects


def main():
    parser = argparse.ArgumentParser(description='ERP Onset Re-alignment')
    parser.add_argument('--subjects', nargs='+', default=None)
    parser.add_argument('--max-subjects', type=int, default=3)
    parser.add_argument('--strategy', choices=['rms', 'woody', 'combined'],
                        default='combined',
                        help='Chiến lược re-align (default: combined)')
    parser.add_argument('--apply', action='store_true',
                        help='Lưu offset ra CSV để dùng lại')
    parser.add_argument('--threshold', type=float, default=2.5,
                        help='Ngưỡng RMS (bội số std, default=2.5)')
    args = parser.parse_args()

    logger = setup_logger()
    config = load_config(CONFIG_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_available_subjects(config)
        subjects = subjects[:args.max_subjects]

    logger.info(f"Strategy: {args.strategy} | Subjects: {subjects}")

    all_results = []

    for sid in subjects:
        logger.info(f"\n── Processing {sid} ──")
        try:
            out = process_subject(sid, config, strategy=args.strategy, logger=logger)
            if out is None:
                continue
            result_df, epochs_raw, offsets_rms, offsets_woody, offsets_final, rms_curves, ch_names = out

            all_results.append(result_df)

            plot_realign_overview(sid, result_df, epochs_raw,
                                   offsets_rms, offsets_woody, offsets_final,
                                   rms_curves, ch_names, OUTPUT_DIR, logger)

            plot_strategy_comparison(sid, result_df, epochs_raw, offsets_rms,
                                      offsets_woody, offsets_final,
                                      ch_names, OUTPUT_DIR, logger)

            plot_per_condition_erp(sid, result_df, epochs_raw, offsets_final,
                                    ch_names, OUTPUT_DIR, logger)

        except Exception as e:
            logger.error(f"[{sid}] Lỗi: {e}", exc_info=True)

    print_summary(all_results)

    if args.apply and all_results:
        combined = pd.concat(all_results, ignore_index=True)
        os.makedirs(os.path.dirname(OFFSET_CSV), exist_ok=True)
        combined.to_csv(OFFSET_CSV, index=False)
        logger.info(f"Saved offsets: {OFFSET_CSV}")
        print(f"\n✅ Offsets đã lưu: {OFFSET_CSV}")
        print("   Có thể dùng cột 'new_onset' thay cho 'start_sample' khi tạo epoch")

    print(f"\n📂 Ảnh tại: {os.path.abspath(OUTPUT_DIR)}/")
    print("   realign_<ID>.png           — Overview offset + grand average trước/sau")
    print("   realign_compare_<ID>.png   — So sánh 3 chiến lược song song")
    print("   realign_pertrial_<ID>.png  — Grid 30 trial riêng lẻ trước/sau align")


if __name__ == '__main__':
    main()
