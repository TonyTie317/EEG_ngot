"""
So sánh ERP trước và sau khi realign onset cho subject P001.
Sử dụng:
  - datadone/sub-P001_ses-S001_task-Default_run-001_eeg.csv  (dữ liệu raw)
  - output/epochs/realign_offsets.csv                         (onset mới)
"""

import os
import matplotlib
matplotlib.use("Agg")   # non-interactive, save to file

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
from mne.viz.utils import plt_show

# ─── CONFIG ──────────────────────────────────────────────────────────────────
SUBJECT      = "P001"
DATA_DIR     = "datadone"
OFFSETS_CSV  = "output/epochs/realign_offsets.csv"
TMIN, TMAX   = -0.2, 1.0   # epoch window (s) quanh onset
BASELINE     = (-0.2, 0.0)
# Điều kiện muốn vẽ ERP (giá trị cột 'condition' trong realign_offsets.csv)
CONDITIONS   = [189, 258, 453, 605, 762, 893]
COND_LABELS  = {189: "cond-189", 258: "cond-258", 453: "cond-453",
                605: "cond-605", 762: "cond-762", 893: "cond-893"}
# ─────────────────────────────────────────────────────────────────────────────


# ── 1) Đọc CSV dữ liệu ─────────────────────────────────────────────────────
csv_path = os.path.join(DATA_DIR, f"sub-{SUBJECT}_ses-S001_task-Default_run-001_eeg.csv")
df = pd.read_csv(csv_path)
print(f"[INFO] Loaded {csv_path}: {df.shape}")

non_signal_cols = {'time', 'timestamp', 'frame_idx', 'ma_mau', 'NT', 'repeat', 'JAR'}
signal_cols = [c for c in df.columns if c not in non_signal_cols and not df[c].isna().all()]
ch_types    = [('ecg' if c.upper().startswith('ECG') else 'eeg') for c in signal_cols]

# ── 2) Suy ra sfreq ────────────────────────────────────────────────────────
ts   = df['timestamp'].to_numpy()
dt   = np.median(np.diff(ts))
sfreq = int(round(1.0 / dt)) if np.isfinite(dt) and dt > 0 else 100
print(f"[INFO] sfreq = {sfreq} Hz")

# ── 3) Tạo RawArray ────────────────────────────────────────────────────────
data_mat = df[signal_cols].to_numpy(dtype=float).T * 1e-6   # µV → V
info     = mne.create_info(ch_names=signal_cols, sfreq=sfreq, ch_types=ch_types)
raw      = mne.io.RawArray(data_mat, info, verbose=False)

# Đổi tên kênh cũ → chuẩn 10-20
rename_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
raw.rename_channels({ch: rename_map.get(ch, ch) for ch in raw.ch_names})

# Montage
try:
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')
except Exception as e:
    print("[WARN] Montage:", e)

# ── 4) Tiền xử lý ─────────────────────────────────────────────────────────
raw.set_eeg_reference('average', projection=False)
raw.notch_filter(freqs=[49], verbose=False)   # sfreq=100Hz → Nyquist=50Hz
raw.filter(l_freq=0.1, h_freq=40.0, verbose=False)

# ICA
n_comp = min(15, len(signal_cols) - 1)
ica = mne.preprocessing.ICA(n_components=n_comp, random_state=97, max_iter=800,
                              method='picard', verbose=False)
ica.fit(raw, verbose=False)

try:
    eog_chs = [ch for ch in raw.ch_names if ch.lower().startswith(('fp1','fpz'))]
    if eog_chs:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_chs[0], threshold=0.5, verbose=False)
        ica.exclude.extend(eog_inds)
except Exception:
    pass

try:
    ecg_chs = [ch for ch, t in zip(raw.ch_names, ch_types) if t == 'ecg']
    if ecg_chs:
        ecg_inds, _ = ica.find_bads_ecg(raw, ch_name=ecg_chs[0], verbose=False)
        ica.exclude.extend(ecg_inds)
except Exception:
    pass

ica.exclude = sorted(set(ica.exclude))
raw_clean = ica.apply(raw.copy(), verbose=False) if ica.exclude else raw.copy()
print(f"[INFO] ICA excluded: {ica.exclude}")

# ── 5) Đọc realign_offsets ─────────────────────────────────────────────────
df_off = pd.read_csv(OFFSETS_CSV)
df_sub = df_off[df_off['subject_id'] == SUBJECT].copy()
print(f"[INFO] {len(df_sub)} trials for {SUBJECT}")


# ── 6) Hàm tạo epochs từ danh sách sample indices ─────────────────────────
def make_epochs(raw_obj, onset_samples, conditions, labels, tmin, tmax, baseline, sfreq):
    """
    onset_samples : array-like, chỉ số mẫu (int)
    conditions    : array-like, mã điều kiện (int)
    """
    # event array: [sample, 0, event_id]
    events = np.column_stack([
        np.array(onset_samples, dtype=int),
        np.zeros(len(onset_samples), dtype=int),
        np.array(conditions, dtype=int)
    ])
    unique_conds = np.unique(conditions)
    event_id = {labels.get(c, str(c)): int(c) for c in unique_conds}

    epochs = mne.Epochs(
        raw_obj, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )
    return epochs


# ── 7) Tạo epochs với onset GỐC (trigger_sample) ──────────────────────────
onset_raw  = df_sub['trigger_sample'].values
cond_codes = df_sub['condition'].values

epochs_raw = make_epochs(
    raw_clean, onset_raw, cond_codes, COND_LABELS,
    TMIN, TMAX, BASELINE, sfreq
)
print("Epochs (RAW onset) :", {k: len(epochs_raw[k]) for k in epochs_raw.event_id})

# ── 8) Tạo epochs với onset MỚI (new_onset) ───────────────────────────────
onset_new  = df_sub['new_onset'].values

epochs_new = make_epochs(
    raw_clean, onset_new, cond_codes, COND_LABELS,
    TMIN, TMAX, BASELINE, sfreq
)
print("Epochs (NEW onset) :", {k: len(epochs_new[k]) for k in epochs_new.event_id})

# ── 9) Tính ERP (Evoked) ──────────────────────────────────────────────────
evoked_raw = {k: epochs_raw[k].average() for k in epochs_raw.event_id}
evoked_new = {k: epochs_new[k].average() for k in epochs_new.event_id}

# ── 10) Vẽ so sánh ERP (raw vs new) cho từng điều kiện ───────────────────
picks_plot = [ch for ch in ['Cz', 'Pz', 'Fz', 'C3', 'C4'] if ch in raw_clean.ch_names]
if not picks_plot:
    picks_plot = raw_clean.copy().pick('eeg').ch_names[:3]

n_conds = len(evoked_raw)
fig, axes = plt.subplots(n_conds, len(picks_plot),
                          figsize=(5 * len(picks_plot), 3 * n_conds),
                          sharex=True)
fig.suptitle(f"ERP so sánh Onset Gốc vs Onset Mới — {SUBJECT}", fontsize=14)

if n_conds == 1:
    axes = np.array([axes])
if len(picks_plot) == 1:
    axes = axes[:, np.newaxis]

for row_i, cond_label in enumerate(sorted(evoked_raw.keys())):
    evk_r = evoked_raw[cond_label]
    evk_n = evoked_new.get(cond_label)

    for col_i, ch in enumerate(picks_plot):
        ax = axes[row_i, col_i]
        try:
            ch_idx = evk_r.ch_names.index(ch)
            times  = evk_r.times

            ax.plot(times, evk_r.data[ch_idx] * 1e6, label='Raw onset', color='steelblue', lw=1.5)
            if evk_n is not None and ch in evk_n.ch_names:
                ch_idx_n = evk_n.ch_names.index(ch)
                ax.plot(times, evk_n.data[ch_idx_n] * 1e6, label='New onset',
                        color='tomato', lw=1.5, linestyle='--')

            ax.axvline(0, color='k', linewidth=0.8, linestyle=':')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.set_title(f"{cond_label} @ {ch}", fontsize=9)
            ax.set_ylabel("µV")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7)
        except Exception as exc:
            ax.set_title(f"{cond_label} @ {ch}\n({exc})", fontsize=7)

axes[-1, 0].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig(f"output/figures/erp_compare_onset_{SUBJECT}.png", dpi=150)
print(f"[INFO] Saved ERP comparison figure → output/figures/erp_compare_onset_{SUBJECT}.png")
plt.close(fig)

# ── 11) Vẽ topo map so sánh cho điều kiện đầu tiên ───────────────────────
first_cond = sorted(evoked_raw.keys())[0]
try:
    fig_r = evoked_raw[first_cond].plot_topomap(
        times=[0.1, 0.2, 0.4], ch_type='eeg', show=False)
    fig_r.suptitle(f'Raw onset — {first_cond}')
    fig_r.savefig(f"output/figures/topo_raw_{first_cond}_{SUBJECT}.png", dpi=150)
    print(f"[INFO] Saved topomap raw → output/figures/topo_raw_{first_cond}_{SUBJECT}.png")
    plt.close(fig_r)

    fig_n = evoked_new[first_cond].plot_topomap(
        times=[0.1, 0.2, 0.4], ch_type='eeg', show=False)
    fig_n.suptitle(f'New onset — {first_cond}')
    fig_n.savefig(f"output/figures/topo_new_{first_cond}_{SUBJECT}.png", dpi=150)
    print(f"[INFO] Saved topomap new → output/figures/topo_new_{first_cond}_{SUBJECT}.png")
    plt.close(fig_n)
except Exception as e:
    print("[WARN] Topomap:", e)

print("[DONE] All figures saved to output/figures/")
