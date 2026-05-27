"""
ERP Visual Inspection Script
=============================
Mục đích: Load epochs đã lưu sẵn, vẽ Grand Average Waveform để nhìn bằng mắt
kiểm tra xem các thành phần P1/N1/P2/N400 có bị lệch offset so với config không.

Chạy:
    python test_erp_visual_inspect.py
    python test_erp_visual_inspect.py --subjects P001 P002 P003
    python test_erp_visual_inspect.py --subjects P001 --channels Fz Cz Pz

Output: output/figures/erp_visual_inspect/
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
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "configs/config.yaml"
OUTPUT_DIR  = "output/figures/erp_visual_inspect"

# Màu sắc cho từng cửa sổ ERP
WINDOW_COLORS = {
    'P1':   ('#4CAF50', 0.15),   # xanh lá
    'N1':   ('#2196F3', 0.15),   # xanh dương
    'P2':   ('#FF9800', 0.15),   # cam
    'N400': ('#E91E63', 0.15),   # hồng đỏ
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger('erp_inspect')


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_subject_epochs(subject_id: str, config: dict, logger) -> tuple:
    """Load epochs + trial_info từ disk."""
    epoch_dir = os.path.join(config['paths']['output_base'], 'epochs', subject_id)
    fif_path  = os.path.join(epoch_dir, 'epochs_epo.fif')
    if not os.path.exists(fif_path):
        fif_path = os.path.join(epoch_dir, 'epochs.fif')
    csv_path  = os.path.join(epoch_dir, 'trial_info.csv')

    if not os.path.exists(fif_path):
        logger.warning(f"[{subject_id}] Không tìm thấy epochs: {fif_path}")
        return None, None

    epochs     = mne.read_epochs(fif_path, verbose=False, preload=True)
    trial_info = pd.read_csv(csv_path)
    logger.info(f"[{subject_id}] Loaded {len(epochs)} epochs, shape={epochs.get_data().shape}")
    return epochs, trial_info


def get_available_subjects(config: dict) -> list:
    """Tìm các subject đã có epochs trên disk."""
    epoch_base = os.path.join(config['paths']['output_base'], 'epochs')
    subjects = []
    for d in sorted(os.listdir(epoch_base)):
        fif = os.path.join(epoch_base, d, 'epochs_epo.fif')
        if not os.path.exists(fif):
            fif = os.path.join(epoch_base, d, 'epochs.fif')
        if os.path.exists(fif):
            subjects.append(d)
    return subjects


OFFSET_CSV = "output/epochs/realign_offsets.csv"


def apply_woody_realign(all_epochs: list, subjects: list, logger) -> list:
    """Dịch chuyển time axis của từng epoch theo Woody offset.

    Với mỗi epoch i của subject s:
      - Đọc offset_final từ realign_offsets.csv
      - Cắt epoch bắt đầu từ new_onset thay vì trigger gốc
      - Trả về list MNE Epochs mới với tmin được cập nhật

    Epoch mới: [true_onset - 0.2s  →  true_onset + 1.0s]
    (cửa sổ ERP thật, loại bỏ phần artifact đưa cốc)
    """
    if not os.path.exists(OFFSET_CSV):
        logger.warning(f"Không tìm thấy {OFFSET_CSV} — dùng trigger gốc")
        return all_epochs

    offsets_df = pd.read_csv(OFFSET_CSV)
    SFREQ = 100
    # Cửa sổ ERP sau true onset
    TMIN_AFTER = -0.2   # 200ms trước true onset (baseline)
    TMAX_AFTER =  1.0   # 1s sau true onset (bắt P1/N1/P2/N400)
    pre  = int(abs(TMIN_AFTER) * SFREQ)   # 20 mẫu baseline
    post = int(TMAX_AFTER * SFREQ)        # 100 mẫu ERP
    win_len = pre + post + 1              # 121 mẫu

    realigned = []
    for epochs, sid in zip(all_epochs, subjects):
        subj_offsets = offsets_df[offsets_df['subject_id'] == sid]
        if subj_offsets.empty:
            logger.warning(f"[{sid}] Không có offset → dùng trigger gốc")
            realigned.append((epochs, [True] * len(epochs)))
            continue

        raw_data = epochs.get_data()           # (n_ep, n_ch, n_t)
        info     = epochs.info
        tmin_orig = epochs.tmin                # -0.5s
        offset_orig = int(abs(tmin_orig) * SFREQ)  # 50 mẫu (0.5s trước trigger)

        new_data = []
        kept_mask = []  # True = epoch kept, False = dropped
        kept = 0
        for ep_i in range(len(epochs)):
            # Tìm offset của trial này theo trial_ix trong trial_info
            # Dùng thứ tự ep_i (đã match với subj_offsets nếu sort đúng)
            if ep_i < len(subj_offsets):
                final_off = int(subj_offsets.iloc[ep_i]['offset_final'])
            else:
                final_off = 0

            # Vị trí true onset trong mảng epoch
            # epoch[0] = trigger - 1s → true onset = trigger + final_off
            # → index trong epoch = offset_orig + final_off
            true_onset_idx = offset_orig + final_off

            start = true_onset_idx - pre
            end   = start + win_len

            if start < 0 or end > raw_data.shape[2]:
                # Không đủ mẫu → dùng phần có sẵn hoặc bỏ qua
                start = max(0, start)
                end   = min(raw_data.shape[2], end)
                if end - start < win_len // 2:
                    kept_mask.append(False)
                    continue
                # Pad với zeros nếu thiếu
                seg = np.zeros((raw_data.shape[1], win_len))
                seg[:, :end-start] = raw_data[ep_i, :, start:end]
            else:
                seg = raw_data[ep_i, :, start:end]

            new_data.append(seg)
            kept_mask.append(True)
            kept += 1

        if not new_data:
            logger.warning(f"[{sid}] Không có epoch nào sau re-align")
            realigned.append((epochs, [True] * len(epochs)))
            continue

        new_arr = np.stack(new_data, axis=0)  # (n_kept, n_ch, win_len)
        new_epochs = mne.EpochsArray(
            new_arr, info,
            tmin=TMIN_AFTER,
            verbose=False,
        )
        logger.info(f"[{sid}] Re-aligned: {kept}/{len(epochs)} epochs, "
                    f"window [{TMIN_AFTER*1000:.0f}ms, {TMAX_AFTER*1000:.0f}ms] vs true onset")
        realigned.append((new_epochs, kept_mask))

    return realigned


# ── Plot 1: Grand Average tất cả kênh ─────────────────────────────────────────
def plot_grand_average_all_channels(
    all_epochs: list, config: dict, output_dir: str, logger
):
    """Vẽ Grand Average của TẤT CẢ kênh trên 1 figure (butterfly plot + individual)."""
    erp_cfg = config.get('erp_analysis', {})

    # Gộp dữ liệu tất cả subject
    data_list = [ep.get_data() for ep in all_epochs]   # list of (n_ep, n_ch, n_t)
    X     = np.concatenate(data_list, axis=0)           # (N, n_ch, n_t)
    grand = X.mean(axis=0)                               # (n_ch, n_t)
    times = all_epochs[0].times
    ch_names = all_epochs[0].ch_names

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f'Grand Average ERP — {len(all_epochs)} subjects, {X.shape[0]} epochs\n'
        f'Kiểm tra xem các đỉnh P1/N1/P2/N400 có rơi vào cửa sổ màu không',
        fontsize=13, fontweight='bold'
    )

    # ── Butterfly plot (tất cả kênh chồng lên nhau) ──────────────────────
    ax = axes[0]
    ax.set_title('Butterfly Plot — Tất cả kênh', fontsize=11)
    _draw_erp_windows(ax, erp_cfg, times)
    for i, ch in enumerate(ch_names):
        ax.plot(times * 1000, grand[i] * 1e6, alpha=0.5, linewidth=0.9)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Onset')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.set_xlabel('Thời gian (ms)')
    ax.set_ylabel('Biên độ (µV)')
    ax.set_xlim(times[0] * 1000, times[-1] * 1000)
    _add_window_legend(ax, erp_cfg)

    # ── Grand Average trung bình tất cả kênh ─────────────────────────────
    ax2 = axes[1]
    ax2.set_title('Grand Average — Trung bình tất cả kênh', fontsize=11)
    _draw_erp_windows(ax2, erp_cfg, times)
    mean_signal = grand.mean(axis=0) * 1e6
    ax2.plot(times * 1000, mean_signal, color='black', linewidth=2, label='Grand Avg')
    ax2.fill_between(times * 1000, mean_signal, alpha=0.1, color='black')
    ax2.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Onset')
    ax2.axhline(0, color='grey', linewidth=0.5)
    ax2.set_xlabel('Thời gian (ms)')
    ax2.set_ylabel('Biên độ (µV)')
    ax2.set_xlim(times[0] * 1000, times[-1] * 1000)

    # Đánh dấu đỉnh thực tế
    _annotate_peaks(ax2, times, mean_signal, erp_cfg)
    _add_window_legend(ax2, erp_cfg)

    plt.tight_layout()
    path = os.path.join(output_dir, '01_grand_average_all_channels.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Plot 2: Từng kênh riêng (subplot grid) ───────────────────────────────────
def plot_per_channel_grid(
    all_epochs: list, config: dict, output_dir: str, logger
):
    """Vẽ Grand Average theo từng kênh riêng biệt trên grid."""
    erp_cfg = config.get('erp_analysis', {})

    data_list = [ep.get_data() for ep in all_epochs]
    X     = np.concatenate(data_list, axis=0)
    grand = X.mean(axis=0) * 1e6   # µV
    times = all_epochs[0].times * 1000  # ms
    ch_names = all_epochs[0].ch_names
    n_ch  = len(ch_names)

    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes = axes.flatten()

    fig.suptitle(
        'Grand Average ERP — Từng kênh riêng\n'
        'Các vùng màu = cửa sổ ERP hiện tại trong config',
        fontsize=12, fontweight='bold', y=1.01
    )

    for i, ch in enumerate(ch_names):
        ax = axes[i]
        _draw_erp_windows(ax, erp_cfg, all_epochs[0].times)
        ax.plot(times, grand[i], color='#1a1a2e', linewidth=1.5)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axhline(0, color='grey', linewidth=0.4)
        ax.set_title(ch, fontsize=10, fontweight='bold')
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('ms', fontsize=8)
        ax.set_ylabel('µV', fontsize=8)
        ax.tick_params(labelsize=7)

        # Đánh dấu đỉnh trong cửa sổ
        _annotate_peaks(ax, all_epochs[0].times, grand[i], erp_cfg, fontsize=7)

    # Ẩn các subplot thừa
    for j in range(n_ch, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, '02_per_channel_grid.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Plot 3: ROI trung bình cho từng thành phần ERP ───────────────────────────
def plot_roi_waveforms(
    all_epochs: list, config: dict, output_dir: str, logger
):
    """Vẽ Grand Average riêng cho từng ROI của mỗi thành phần ERP."""
    erp_cfg  = config.get('erp_analysis', {})
    ch_names = all_epochs[0].ch_names

    data_list = [ep.get_data() for ep in all_epochs]
    X     = np.concatenate(data_list, axis=0)
    grand = X.mean(axis=0) * 1e6
    times = all_epochs[0].times

    components = ['P1', 'N1', 'P2', 'N400']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    fig.suptitle(
        'Grand Average ERP — ROI trung bình theo từng thành phần\n'
        'Cửa sổ màu = vùng thời gian trong config. '
        'Đỉnh thực tế nên nằm BÊN TRONG cửa sổ.',
        fontsize=12, fontweight='bold'
    )

    for idx, comp in enumerate(components):
        ax = axes[idx]
        color, alpha = WINDOW_COLORS[comp]
        win_key = f'{comp.lower()}_window'
        roi_key = f'{comp.lower()}_roi'
        win = erp_cfg.get(win_key, [0, 0.5])
        roi = erp_cfg.get(roi_key, ch_names)
        roi = [ch for ch in roi if ch in ch_names]
        if not roi:
            roi = ch_names

        # Trung bình ROI
        ch_idx   = [ch_names.index(ch) for ch in roi]
        roi_data = grand[ch_idx].mean(axis=0)   # (n_times,)

        # Vẽ cửa sổ màu
        ax.axvspan(win[0] * 1000, win[1] * 1000, color=color, alpha=alpha, label='Window')
        ax.plot(times * 1000, roi_data,
                linewidth=2.2, color=color,
                label=f'{comp} ROI: {", ".join(roi)}')
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Onset')
        ax.axhline(0, color='grey', linewidth=0.5)

        # Đỉnh thực tế trong cửa sổ
        t_mask  = (times >= win[0]) & (times <= win[1])
        if t_mask.any():
            sub_sig  = roi_data[t_mask]
            sub_t    = times[t_mask]
            if comp in ('P1', 'P2'):
                peak_idx = np.argmax(sub_sig)
            else:
                peak_idx = np.argmin(sub_sig)
            peak_t   = sub_t[peak_idx] * 1000
            peak_amp = sub_sig[peak_idx]
            ax.axvline(peak_t, color='red', linewidth=1.5, linestyle=':',
                       label=f'Peak thực: {peak_t:.1f}ms ({peak_amp:.2f}µV)')
            ax.annotate(
                f'{peak_t:.0f}ms\n{peak_amp:.2f}µV',
                xy=(peak_t, peak_amp),
                xytext=(peak_t + 30, peak_amp * 0.85 if abs(peak_amp) > 0.1 else peak_amp + 0.2),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
            )

        ax.set_title(
            f'{comp} | Window: {int(win[0]*1000)}–{int(win[1]*1000)}ms | '
            f'ROI: {", ".join(roi)}',
            fontsize=10
        )
        ax.set_xlabel('Thời gian (ms)', fontsize=9)
        ax.set_ylabel('Biên độ (µV)', fontsize=9)
        ax.set_xlim(times[0] * 1000, times[-1] * 1000)
        ax.legend(fontsize=8, loc='upper right')
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, '03_roi_waveforms_per_component.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Plot 4: Theo từng condition (concentration) ───────────────────────────────
def plot_by_condition(
    all_epochs: list, all_trial_info: pd.DataFrame,
    config: dict, output_dir: str, logger
):
    """Grand Average chia theo nồng độ (condition)."""
    erp_cfg   = config.get('erp_analysis', {})
    ch_names  = all_epochs[0].ch_names
    times     = all_epochs[0].times

    data_list = [ep.get_data() for ep in all_epochs]
    X = np.concatenate(data_list, axis=0) * 1e6   # µV
    all_ti = all_trial_info.reset_index(drop=True)

    conditions = sorted(all_ti['condition'].unique())
    colors_cond = plt.cm.tab10(np.linspace(0, 0.9, len(conditions)))

    # Dùng 1 kênh đại diện (Pz nếu có, không thì kênh đầu)
    rep_chs = ['Pz', 'Cz', 'Fz', 'C3', 'C4', 'P3', 'P4']
    rep_ch  = next((ch for ch in rep_chs if ch in ch_names), ch_names[0])
    ch_idx  = ch_names.index(rep_ch)

    fig, ax = plt.subplots(figsize=(13, 6))
    _draw_erp_windows(ax, erp_cfg, times)

    for i, cond in enumerate(conditions):
        mask     = all_ti['condition'] == cond
        cond_sig = X[mask.values, ch_idx].mean(axis=0)
        label    = all_ti.loc[mask, 'condition_label'].iloc[0] if 'condition_label' in all_ti.columns else str(cond)
        ax.plot(times * 1000, cond_sig, label=label,
                color=colors_cond[i], linewidth=1.8)

    ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Onset')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_title(
        f'Grand Average theo Nồng độ — Kênh {rep_ch}\n'
        f'Kiểm tra xem sóng có khác nhau giữa các condition không',
        fontsize=12
    )
    ax.set_xlabel('Thời gian (ms)', fontsize=10)
    ax.set_ylabel('Biên độ (µV)', fontsize=10)
    ax.set_xlim(times[0] * 1000, times[-1] * 1000)
    _add_window_legend(ax, erp_cfg)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    path = os.path.join(output_dir, '04_by_condition.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Plot 5: Bảng thống kê đỉnh thực tế vs config ─────────────────────────────
def print_peak_summary(all_epochs: list, config: dict, logger):
    """In ra bảng so sánh đỉnh thực tế vs cửa sổ config."""
    erp_cfg  = config.get('erp_analysis', {})
    ch_names = all_epochs[0].ch_names
    times    = all_epochs[0].times

    data_list = [ep.get_data() for ep in all_epochs]
    X     = np.concatenate(data_list, axis=0)
    grand = X.mean(axis=0) * 1e6

    components = ['P1', 'N1', 'P2', 'N400']
    is_positive = {'P1': True, 'N1': False, 'P2': True, 'N400': False}

    print("\n" + "="*72)
    print("  KIỂM TRA OFFSET ERP — So sánh đỉnh thực tế vs cửa sổ config")
    print("="*72)
    print(f"{'Thành phần':<10} {'Window config':<20} {'Peak thực tế':<18} "
          f"{'Lệch offset':<15} {'Kênh mạnh nhất':<15} {'Nhận xét'}")
    print("-"*90)

    for comp in components:
        win_key = f'{comp.lower()}_window'
        roi_key = f'{comp.lower()}_roi'
        win  = erp_cfg.get(win_key, [0, 0.5])
        roi  = erp_cfg.get(roi_key, ch_names)
        roi  = [ch for ch in roi if ch in ch_names] or ch_names

        ch_idx   = [ch_names.index(ch) for ch in roi]
        roi_data = grand[ch_idx].mean(axis=0)

        # Tìm đỉnh BÊN TRONG window config
        t_mask = (times >= win[0]) & (times <= win[1])
        if t_mask.any():
            sub_sig = roi_data[t_mask]
            sub_t   = times[t_mask] * 1000
            if is_positive[comp]:
                in_win_peak_idx = np.argmax(sub_sig)
            else:
                in_win_peak_idx = np.argmin(sub_sig)
            in_win_peak_t   = sub_t[in_win_peak_idx]
            in_win_peak_amp = sub_sig[in_win_peak_idx]
            full_peak_t     = in_win_peak_t   # dùng peak trong window để báo cáo
        else:
            in_win_peak_t = float('nan')
            in_win_peak_amp = float('nan')
            full_peak_t = float('nan')

        # Kênh mạnh nhất
        window_data = grand[ch_idx][:, t_mask]
        if window_data.shape[1] > 0:
            if is_positive[comp]:
                best_ch_idx = np.argmax(window_data.max(axis=1))
            else:
                best_ch_idx = np.argmin(window_data.min(axis=1))
            best_ch = roi[best_ch_idx]
        else:
            best_ch = '?'

        offset_ms = full_peak_t - (win[0] + win[1]) / 2 * 1000

        status = f"✅ Peak={in_win_peak_t:.0f}ms, amp={in_win_peak_amp:.1f}µV"

        print(
            f"{comp:<10} "
            f"{int(win[0]*1000)}-{int(win[1]*1000)} ms{'':<10} "
            f"{full_peak_t:.1f} ms{'':<10} "
            f"{offset_ms:+.1f} ms{'':<7} "
            f"{best_ch:<15} "
            f"{status}"
        )

    print("="*90)
    print("\n💡 Nếu thấy 'NGOÀI window' → cần dịch cửa sổ trong config.yaml")
    print("   Ví dụ: nếu P1 thực tế đỉnh tại 180ms, sửa p1_window: [0.160, 0.220]")
    print()



# ── Plot 5: Từng subject riêng (overlay) ─────────────────────────────────────
def plot_per_subject(
    all_epochs: list, subjects: list, config: dict, output_dir: str, logger
):
    """Vẽ Grand Average của TỪNG NGƯỜI trên cùng 1 figure để so sánh.
    
    Mỗi subject = 1 đường màu riêng. Đường đen đậm = Grand Average gộp.
    Dùng kênh đại diện (Pz/Cz/C3) và trung bình tất cả kênh.
    """
    erp_cfg  = config.get('erp_analysis', {})
    ch_names = all_epochs[0].ch_names
    times    = all_epochs[0].times * 1000  # ms

    # Kênh đại diện
    rep_chs = ['Pz', 'Cz', 'Fz', 'C3', 'C4', 'P3', 'P4']
    rep_ch  = next((ch for ch in rep_chs if ch in ch_names), ch_names[0])
    ch_idx  = ch_names.index(rep_ch)

    colors_subj = plt.cm.tab20(np.linspace(0, 1, len(all_epochs)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f'ERP từng Subject riêng biệt — {len(all_epochs)} subjects\n'
        f'Đường đen đậm = Grand Average gộp | Vùng màu = cửa sổ ERP config',
        fontsize=13, fontweight='bold'
    )

    # ── Subplot 1: kênh đại diện ─────────────────────────────────────────
    ax = axes[0]
    ax.set_title(f'Kênh {rep_ch} — Từng subject', fontsize=11)
    _draw_erp_windows(ax, erp_cfg, all_epochs[0].times)

    all_subject_signals = []
    for i, (epochs, sid) in enumerate(zip(all_epochs, subjects)):
        sig = epochs.get_data()[:, ch_idx, :].mean(axis=0) * 1e6  # trung bình epochs của subject
        all_subject_signals.append(sig)
        ax.plot(times, sig, color=colors_subj[i], linewidth=1.2,
                alpha=0.75, label=sid)

    # Grand average gộp
    grand = np.array(all_subject_signals).mean(axis=0)
    ax.plot(times, grand, color='black', linewidth=2.5,
            label='Grand Avg', zorder=10)
    ax.fill_between(times, grand, alpha=0.08, color='black')

    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_xlabel('Thời gian (ms)', fontsize=10)
    ax.set_ylabel('Biên độ (µV)', fontsize=10)
    ax.set_xlim(times[0], times[-1])
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    _add_window_legend(ax, erp_cfg)

    # ── Subplot 2: trung bình tất cả kênh ────────────────────────────────
    ax2 = axes[1]
    ax2.set_title('Trung bình tất cả kênh — Từng subject', fontsize=11)
    _draw_erp_windows(ax2, erp_cfg, all_epochs[0].times)

    all_mean_signals = []
    for i, (epochs, sid) in enumerate(zip(all_epochs, subjects)):
        sig = epochs.get_data().mean(axis=(0, 1)) * 1e6  # mean over epochs & channels
        all_mean_signals.append(sig)
        ax2.plot(times, sig, color=colors_subj[i], linewidth=1.2,
                 alpha=0.75, label=sid)

    grand_mean = np.array(all_mean_signals).mean(axis=0)
    ax2.plot(times, grand_mean, color='black', linewidth=2.5,
             label='Grand Avg', zorder=10)
    ax2.fill_between(times, grand_mean, alpha=0.08, color='black')

    ax2.axvline(0, color='black', linewidth=1, linestyle='--')
    ax2.axhline(0, color='grey', linewidth=0.5)
    ax2.set_xlabel('Thời gian (ms)', fontsize=10)
    ax2.set_ylabel('Biên độ (µV)', fontsize=10)
    ax2.set_xlim(times[0], times[-1])
    ax2.legend(fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout()
    path = os.path.join(output_dir, '05_per_subject_overlay.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {path}")

    # ── Subplot grid: mỗi subject 1 ô riêng ─────────────────────────────
    n = len(all_epochs)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5),
                                sharex=True)
    axes2 = np.array(axes2).flatten()
    fig2.suptitle(
        'ERP từng Subject — Trung bình tất cả kênh\n'
        'Vùng màu = cửa sổ ERP config | Chấm màu = peak trong window',
        fontsize=12, fontweight='bold'
    )

    for i, (epochs, sid) in enumerate(zip(all_epochs, subjects)):
        ax = axes2[i]
        sig = epochs.get_data().mean(axis=(0, 1)) * 1e6
        _draw_erp_windows(ax, erp_cfg, all_epochs[0].times)
        ax.plot(times, sig, color='#1a1a2e', linewidth=1.8)
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axhline(0, color='grey', linewidth=0.4)
        ax.set_title(f'{sid} ({len(epochs)} epochs)', fontsize=10, fontweight='bold')
        ax.set_xlabel('ms', fontsize=8)
        ax.set_ylabel('µV', fontsize=8)
        ax.tick_params(labelsize=7)
        _annotate_peaks(ax, all_epochs[0].times, sig, erp_cfg, fontsize=7)

    for j in range(n, len(axes2)):
        axes2[j].set_visible(False)

    plt.tight_layout()
    path2 = os.path.join(output_dir, '05b_per_subject_grid.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved: {path2}")


# ── Helper functions ──────────────────────────────────────────────────────────
def _draw_erp_windows(ax, erp_cfg: dict, times):
    """Tô màu các cửa sổ ERP lên axes."""
    for comp, (color, alpha) in WINDOW_COLORS.items():
        win_key = f'{comp.lower()}_window'
        win = erp_cfg.get(win_key, None)
        if win:
            ax.axvspan(win[0] * 1000, win[1] * 1000,
                       color=color, alpha=alpha, zorder=0)


def _add_window_legend(ax, erp_cfg: dict):
    """Thêm legend cho các cửa sổ màu."""
    patches = []
    for comp, (color, alpha) in WINDOW_COLORS.items():
        win_key = f'{comp.lower()}_window'
        win = erp_cfg.get(win_key, None)
        if win:
            patches.append(mpatches.Patch(
                color=color, alpha=alpha + 0.3,
                label=f'{comp}: {int(win[0]*1000)}–{int(win[1]*1000)}ms'
            ))
    if patches:
        ax.legend(handles=patches, fontsize=8, loc='upper right',
                  title='ERP Windows', title_fontsize=8)


def _annotate_peaks(ax, times, signal, erp_cfg: dict, fontsize=8):
    """Đánh dấu đỉnh thực tế trong từng cửa sổ."""
    is_pos = {'P1': True, 'N1': False, 'P2': True, 'N400': False}
    for comp, (color, _) in WINDOW_COLORS.items():
        win_key = f'{comp.lower()}_window'
        win = erp_cfg.get(win_key, None)
        if not win:
            continue
        t_mask = (times >= win[0]) & (times <= win[1])
        if not t_mask.any():
            continue
        sub_sig = signal[t_mask]
        sub_t   = times[t_mask] * 1000
        if is_pos[comp]:
            idx = np.argmax(sub_sig)
        else:
            idx = np.argmin(sub_sig)
        peak_t   = sub_t[idx]
        peak_amp = sub_sig[idx]
        ax.plot(peak_t, peak_amp, 'o', color=color, markersize=6, zorder=5)
        ax.annotate(
            f'{comp}\n{peak_t:.0f}ms',
            xy=(peak_t, peak_amp),
            xytext=(peak_t + 40, peak_amp),
            fontsize=fontsize, color=color, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=color, lw=1),
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def plot_per_condition_per_subject(
    all_epochs: list, subjects: list, all_trial_info: pd.DataFrame,
    config: dict, output_dir: str, logger
):
    """Vẽ ERP đúng cách: trung bình TRONG TỪNG CONDITION (5 repeat),
    sau đó overlay các condition trên cùng axes.

    Mỗi subject = 1 figure riêng, mỗi axes = 1 kênh ROI đại diện.
    Đường màu = trung bình 5 repeat của mỗi nồng độ.
    """
    erp_cfg  = config.get('erp_analysis', {})
    ch_names = all_epochs[0].ch_names
    times    = all_epochs[0].times * 1000  # ms

    # Kênh đại diện để xem nhanh
    rep_chs = ['Pz', 'Cz', 'C3', 'C4', 'P3', 'P4', 'Fz']
    roi_chs = [ch for ch in rep_chs if ch in ch_names][:4]  # tối đa 4 kênh

    all_ti = all_trial_info.reset_index(drop=True)
    conditions = sorted(all_ti['condition'].unique())
    colors_cond = plt.cm.tab10(np.linspace(0, 0.9, len(conditions)))

    # ── Figure 1: Từng subject, overlay conditions ────────────────────────
    n = len(all_epochs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    for roi_ch in roi_chs:
        ch_idx = ch_names.index(roi_ch)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 5, nrows * 4),
                                  sharex=True, sharey=True)
        axes = np.array(axes).flatten()
        fig.suptitle(
            f'ERP theo từng Condition — Kênh {roi_ch}\n'
            f'Mỗi đường = trung bình 5 lần lặp của 1 nồng độ | '
            f'Vùng màu = cửa sổ ERP config',
            fontsize=12, fontweight='bold'
        )

        # Tính offset epoch trong all_trial_info
        offset = 0
        for i, (epochs, sid) in enumerate(zip(all_epochs, subjects)):
            ax = axes[i]
            n_ep = len(epochs)
            ti_subj = all_ti.iloc[offset:offset + n_ep].reset_index(drop=True)
            offset += n_ep

            _draw_erp_windows(ax, erp_cfg, all_epochs[0].times)

            for j, cond in enumerate(conditions):
                mask = ti_subj['condition'] == cond
                if mask.sum() == 0:
                    continue
                # Trung bình 5 repeat của condition này
                cond_data = epochs.get_data()[mask.values, ch_idx, :].mean(axis=0) * 1e6
                label_col = 'condition_label' if 'condition_label' in ti_subj.columns else 'condition'
                label = ti_subj.loc[mask, label_col].iloc[0]
                n_rep = mask.sum()
                ax.plot(times, cond_data, color=colors_cond[j],
                        linewidth=1.5, label=f'{label} (n={n_rep})')

            ax.axvline(0, color='black', linewidth=1, linestyle='--')
            ax.axhline(0, color='grey', linewidth=0.5)
            ax.set_title(f'{sid}', fontsize=10, fontweight='bold')
            ax.set_xlabel('ms', fontsize=8)
            ax.set_ylabel('µV', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc='upper right', ncol=2)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fname = f'06_per_subject_per_condition_{roi_ch}.png'
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved: {path}")

    # ── Figure 2: Grand Average per condition (gộp tất cả subjects) ──────
    fig2, axes2 = plt.subplots(1, len(roi_chs),
                                figsize=(len(roi_chs) * 5, 5),
                                sharex=True, sharey=True)
    if len(roi_chs) == 1:
        axes2 = [axes2]

    fig2.suptitle(
        'Grand Average ERP theo Condition — Gộp tất cả subjects\n'
        'Mỗi đường = trung bình (subjects × 5 repeats) của 1 nồng độ',
        fontsize=12, fontweight='bold'
    )

    # Gộp data + trial_info
    X_all = np.concatenate([ep.get_data() for ep in all_epochs], axis=0) * 1e6

    for k, roi_ch in enumerate(roi_chs):
        ch_idx = ch_names.index(roi_ch)
        ax = axes2[k]
        _draw_erp_windows(ax, erp_cfg, all_epochs[0].times)

        for j, cond in enumerate(conditions):
            mask = all_ti['condition'] == cond
            if mask.sum() == 0:
                continue
            cond_sig = X_all[mask.values, ch_idx, :].mean(axis=0)
            label_col = 'condition_label' if 'condition_label' in all_ti.columns else 'condition'
            label = all_ti.loc[mask, label_col].iloc[0]
            n_ep = mask.sum()
            ax.plot(times, cond_sig, color=colors_cond[j],
                    linewidth=2, label=f'{label} (n={n_ep})')

        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.set_title(f'Kênh {roi_ch}', fontsize=10, fontweight='bold')
        ax.set_xlabel('ms', fontsize=9)
        ax.set_ylabel('µV', fontsize=9)
        ax.set_xlim(times[0], times[-1])
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    path2 = os.path.join(output_dir, '07_grand_avg_per_condition.png')
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    logger.info(f"Saved: {path2}")


def main():
    parser = argparse.ArgumentParser(description='ERP Visual Inspection')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='Danh sách subject IDs (mặc định: tất cả có sẵn). Ví dụ: P001 P002 P003')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Giới hạn số subject (để chạy nhanh)')
    parser.add_argument('--realign', action='store_true', default=True,
                        help='Áp dụng Woody re-alignment (default: True)')
    parser.add_argument('--no-realign', dest='realign', action='store_false',
                        help='Tắt re-alignment, dùng trigger gốc')
    args = parser.parse_args()

    logger = setup_logger()
    config = load_config(CONFIG_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Xác định subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_available_subjects(config)
        if args.max_subjects:
            subjects = subjects[:args.max_subjects]

    logger.info(f"Sẽ load {len(subjects)} subjects: {subjects}")

    # Load epochs
    all_epochs    = []
    all_trial_info_parts = []
    for sid in subjects:
        ep, ti = load_subject_epochs(sid, config, logger)
        if ep is not None:
            all_epochs.append(ep)
            all_trial_info_parts.append(ti)

    if not all_epochs:
        logger.error("Không load được epoch nào. Hãy chạy epoching pipeline trước.")
        sys.exit(1)

    all_trial_info = pd.concat(all_trial_info_parts, ignore_index=True)
    logger.info(f"Tổng: {len(all_epochs)} subjects, {len(all_trial_info)} epochs")

    # ── Áp dụng Woody re-alignment ───────────────────────────────────────
    if args.realign:
        logger.info("Áp dụng Woody re-alignment (t=0 = true onset)...")
        results = apply_woody_realign(all_epochs, subjects, logger)
        # Unpack tuples (epochs, kept_mask) và sync all_trial_info
        all_epochs = [ep for ep, _ in results]
        # Xây dựng global mask để lọc all_trial_info
        global_mask = []
        for _, mask in results:
            global_mask.extend(mask)
        all_trial_info = all_trial_info[global_mask].reset_index(drop=True)
        mode_label = 'SAU Woody re-align (t=0 = true onset)'
    else:
        logger.info("Dùng trigger gốc (t=0 = trigger, CHƯA re-align)")
        mode_label = 'Trigger gốc (t=0 = trigger, chưa re-align)'
    logger.info(f"Mode: {mode_label}")

    # ── Vẽ các đồ thị ────────────────────────────────────────────────────
    logger.info("Vẽ [1/5] Grand Average Butterfly + Mean (tất cả subjects gộp)...")
    plot_grand_average_all_channels(all_epochs, config, OUTPUT_DIR, logger)

    logger.info("Vẽ [2/5] Per-channel grid (grand average gộp)...")
    plot_per_channel_grid(all_epochs, config, OUTPUT_DIR, logger)

    logger.info("Vẽ [3/5] ROI waveforms per component (grand average gộp)...")
    plot_roi_waveforms(all_epochs, config, OUTPUT_DIR, logger)

    logger.info("Vẽ [4/5] By condition...")
    plot_by_condition(all_epochs, all_trial_info, config, OUTPUT_DIR, logger)

    logger.info("Vẽ [5/6] Từng subject riêng biệt (gộp conditions)...")
    plot_per_subject(all_epochs, subjects, config, OUTPUT_DIR, logger)

    logger.info("Vẽ [6/6] Từng subject × từng condition (ĐÚNG CÁCH)...")
    plot_per_condition_per_subject(all_epochs, subjects, all_trial_info, config, OUTPUT_DIR, logger)

    # ── In bảng thống kê ────────────────────────────────────────────────
    print_peak_summary(all_epochs, config, logger)

    logger.info(f"\n✅ Xong! Xem kết quả tại: {OUTPUT_DIR}/")
    print(f"\n📂 Mode: {mode_label}")
    print(f"📂 Mở thư mục: {os.path.abspath(OUTPUT_DIR)}")
    print("   01_grand_average_all_channels.png     — Butterfly (GỘP tất cả conditions, để tham khảo)")
    print("   02_per_channel_grid.png               — Từng kênh (GỘP tất cả conditions)")
    print("   03_roi_waveforms_per_component.png    — ROI P1/N1/P2/N400 (GỘP tất cả conditions)")
    print("   04_by_condition.png                   — Overlay conditions trên kênh đại diện")
    print("   05_per_subject_overlay.png            — Từng người riêng (GỘP conditions)")
    print("   05b_per_subject_grid.png              — Grid từng người (GỘP conditions)")
    print("   06_per_subject_per_condition_*.png    — ⭐ TỪNG NGƯỜI × TỪNG CONDITION (đúng cách)")
    print("   07_grand_avg_per_condition.png        — ⭐ Grand Average theo condition (đúng cách)")


if __name__ == '__main__':
    main()
