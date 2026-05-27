#!/usr/bin/env python3
"""
ERP Analysis per individual channel — tìm kênh EEG nào có effect significant.

Chạy:
  1. Extract component measures cho từng channel riêng lẻ (không gộp ROI)
  2. rmANOVA concentration effect cho mỗi component × channel
  3. JAR ANOVA cho mỗi component × channel
  4. Heatmap significant channels

Usage:
    .venv/bin/python run_per_channel_anova.py
    .venv/bin/python run_per_channel_anova.py --quick    # chỉ chạy 1 số channel
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mne

from pipeline.config import load_config, setup_logging
from pipeline.constants import (
    ALL_SUBJECTS, CONCENTRATIONS, CONCENTRATION_LABELS,
    ERP_WINDOWS, ERP_PEAK_MODE, EEG_CHANNELS,
)
from pipeline.stats import repeated_measures_anova
from scipy.stats import f_oneway

EPOCHS_BASE = 'output/epochs'
TRIAL_INFO_CSV = 'output/epochs/all_trial_info.csv'
OUTPUT_DIR = 'output/results/per_channel'
QUALITY_FLAGS = 'output/results/erp/erp_quality_flags.csv'

JAR_GROUPS = ['Khong_du', 'Vua_phai', 'Qua_nhieu']
COMPONENTS = ['P1', 'N1', 'P2', 'N400']
REALIGN_CSV = 'output/epochs/realign_offsets.csv'
_REALIGN_TMIN = -0.2
_REALIGN_TMAX = 1.0


def load_subject_data(subjects):
    """Load tất cả subject data, trả về list of (data, times, ch_names, trial_info)."""
    all_data = []
    all_ti = []
    ch_names = None
    times = None
    sfreq = None

    for sid in subjects:
        fif_path = os.path.join(EPOCHS_BASE, sid, 'epochs_epo.fif')
        csv_path = os.path.join(EPOCHS_BASE, sid, 'trial_info.csv')
        if not os.path.exists(fif_path):
            continue
        epochs = mne.read_epochs(fif_path, preload=True, verbose=False)
        ti = pd.read_csv(csv_path)
        data = epochs.get_data()  # (n_ep, n_ch, n_t)
        if ch_names is None:
            ch_names = epochs.ch_names
            times = epochs.times
            sfreq = epochs.info['sfreq']

        # Woody realignment nếu có
        if os.path.exists(REALIGN_CSV):
            offsets = pd.read_csv(REALIGN_CSV)
            subj_off = offsets[offsets['subject_id'] == sid]
            if len(subj_off) > 0:
                pre = int(abs(_REALIGN_TMIN) * sfreq)
                post = int(_REALIGN_TMAX * sfreq)
                win_len = pre + post + 1
                offset_orig = int(abs(epochs.tmin) * sfreq)
                new_data = []
                kept_idx = []
                for ep_i in range(len(epochs)):
                    final_off = int(subj_off.iloc[ep_i]['offset_final']) \
                        if ep_i < len(subj_off) else 0
                    true_onset = offset_orig + final_off
                    start = true_onset - pre
                    end = start + win_len
                    if start >= 0 and end <= data.shape[2]:
                        new_data.append(data[ep_i, :, start:end])
                        kept_idx.append(ep_i)
                if new_data:
                    data = np.stack(new_data, axis=0)
                    ti = ti.iloc[kept_idx].reset_index(drop=True)
                    times = np.arange(-pre, post + 1) / sfreq
                else:
                    data = data
                    ti = ti

        all_data.append(data)
        all_ti.append(ti)

    # Ghép subjects
    X = np.concatenate(all_data, axis=0)
    trial_info = pd.concat(all_ti, ignore_index=True)
    return X, times, ch_names, trial_info


def extract_channel_measures(X, times, trial_info, ch_names, ch_idx, config):
    """Extract component measures cho 1 channel."""
    ch_name = ch_names[ch_idx]
    ch_data = X[:, ch_idx, :]  # (n_epochs, n_times)
    erp_cfg = config.get('erp_analysis', {})

    measures = []
    subjects = trial_info['subject_id'].unique()

    for sid in subjects:
        subj_mask = trial_info['subject_id'] == sid
        for cond in CONCENTRATIONS:
            mask = subj_mask & (trial_info['condition'] == cond)
            if mask.sum() < 1:
                continue

            cond_data = ch_data[mask.values]  # (n_trials, n_t)
            avg = cond_data.mean(axis=0)  # (n_t,)

            row = {
                'subject_id': sid,
                'condition': cond,
                'condition_label': CONCENTRATION_LABELS.get(cond, str(cond)),
                'channel': ch_name,
            }

            for comp in COMPONENTS:
                wkey = f'{comp.lower()}_window'
                window = erp_cfg.get(wkey, ERP_WINDOWS[comp])
                mode = ERP_PEAK_MODE[comp]

                t_mask = (times >= window[0]) & (times <= window[1])
                if t_mask.sum() == 0:
                    row[f'{comp}_mean_amp'] = np.nan
                    row[f'{comp}_peak_amp'] = np.nan
                    continue

                win = avg[t_mask]
                mean_amp = win.mean()

                if mode == 'pos':
                    peak_amp = win.max()
                else:
                    peak_amp = win.min()

                row[f'{comp}_mean_amp'] = mean_amp
                row[f'{comp}_peak_amp'] = peak_amp

            # JAR
            jar_vals = trial_info.loc[mask[mask].index, 'jar_group']
            row['jar_group'] = jar_vals.iloc[0] if len(jar_vals) > 0 else None

            measures.append(row)

    return pd.DataFrame(measures)


def run_concentration_anova(measures):
    """rmANOVA concentration effect cho 1 channel × component."""
    dv_cols = [f'{comp}_{m}' for comp in COMPONENTS for m in ['mean_amp', 'peak_amp']]

    results = []
    for dv in dv_cols:
        if dv not in measures.columns:
            continue
        data = measures.dropna(subset=[dv]).copy()

        # Chỉ subjects có đủ 6 conditions
        subj_conds = data.groupby('subject_id')['condition'].nunique()
        complete = subj_conds[subj_conds == 6].index
        balanced = data[data['subject_id'].isin(complete)]

        if len(balanced) < 10 or len(complete) < 3:
            results.append({
                'dv': dv, 'comp': dv.split('_')[0], 'measure': '_'.join(dv.split('_')[1:]),
                'F': np.nan, 'p': np.nan, 'np2': np.nan, 'n_subjects': len(complete),
                'method': 'insufficient'
            })
            continue

        try:
            result = repeated_measures_anova(
                measures=balanced, dv=dv, within_factor='condition',
                subject_col='subject_id'
            )
            if result:
                results.append({
                    'dv': dv,
                    'comp': dv.split('_')[0],
                    'measure': '_'.join(dv.split('_')[1:]),
                    'F': result['F'],
                    'p': result['p_value'],
                    'np2': result['np2'],
                    'n_subjects': len(complete),
                    'method': 'rmANOVA'
                })
        except Exception as e:
            results.append({
                'dv': dv, 'comp': dv.split('_')[0], 'measure': '_'.join(dv.split('_')[1:]),
                'F': np.nan, 'p': np.nan, 'np2': np.nan,
                'n_subjects': len(complete), 'method': f'error: {e}'
            })

    return pd.DataFrame(results)


def run_jar_anova_channel(measures):
    """JAR ANOVA (1-way between) cho 1 channel × component."""
    dv_cols = [f'{comp}_{m}' for comp in COMPONENTS for m in ['mean_amp', 'peak_amp']]

    results = []
    for dv in dv_cols:
        if dv not in measures.columns:
            continue
        data = measures.dropna(subset=[dv, 'jar_group']).copy()
        if len(data) < 10:
            continue

        groups = []
        for jg in JAR_GROUPS:
            vals = data[data['jar_group'] == jg][dv].values
            if len(vals) >= 2:
                groups.append(vals)

        if len(groups) >= 2:
            try:
                f, p = f_oneway(*groups)
                results.append({
                    'dv': dv,
                    'comp': dv.split('_')[0],
                    'measure': '_'.join(dv.split('_')[1:]),
                    'F': f,
                    'p': p,
                    'n_total': len(data),
                })
            except Exception:
                pass

    return pd.DataFrame(results)


def plot_significance_heatmap(all_anova, title, filename):
    """Heatmap: -log10(p) cho mỗi channel × component."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    channels = EEG_CHANNELS
    comps = [f'{c}_mean_amp' for c in COMPONENTS]

    # Build matrix
    pvals = np.ones((len(channels), len(comps)))
    for i, ch in enumerate(channels):
        for j, dv in enumerate(comps):
            row = all_anova[(all_anova['dv'] == dv) & (all_anova['channel'] == ch)]
            if len(row) and row['p'].notna().any():
                p = row['p'].values[0]
                if not np.isnan(p) and p > 0:
                    pvals[i, j] = p

    logp = -np.log10(pvals)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(logp, cmap='YlOrRd', aspect='auto', vmin=0, vmax=3)

    # Text
    for i in range(len(channels)):
        for j in range(len(comps)):
            p = pvals[i, j]
            if p < 0.05:
                txt = f'{p:.3f}*' if p < 0.05 else ''
                color = 'white' if logp[i, j] > 1.5 else 'black'
                ax.text(j, i, f'{p:.3f}', ha='center', va='center', fontsize=7,
                        color=color, fontweight='bold')

    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(comps, rotation=30, ha='right')
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('-log10(p)', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', nargs='+', default=None,
                        help='Specific channels (default: all 16)')
    parser.add_argument('--quick', action='store_true',
                        help='Chỉ chạy P1+P2 mean_amp cho nhanh')
    args = parser.parse_args()

    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    if args.channels:
        ch_list = args.channels
    else:
        ch_list = EEG_CHANNELS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Quality filter — DISABLED (cần complete subjects cho rmANOVA)
    # Chạy không filter để có đủ subjects balanced
    keep_set = None
    logger.info('Using ALL data (no quality filter) — rmANOVA needs complete cases')

    # Load data cho ALL subjects (cần cho complete rmANOVA)
    X, times, ch_names, trial_info = load_subject_data(ALL_SUBJECTS)
    logger.info(f'Data: {X.shape[0]} epochs, {X.shape[1]} channels, {X.shape[2]} times')

    # Áp dụng quality filter
    if keep_set:
        keep_mask = trial_info.apply(
            lambda r: (r['subject_id'], int(r['condition'])) in keep_set, axis=1
        )
        X = X[keep_mask.values]
        trial_info = trial_info[keep_mask].reset_index(drop=True)
        logger.info(f'After quality filter: {len(trial_info)} trials')

    all_conc = []  # concentration ANOVA results per channel
    all_jar = []   # JAR ANOVA results per channel

    chan_names_normalized = [c.upper() for c in ch_names]

    for ch_name in ch_list:
        ch_idx = chan_names_normalized.index(ch_name.upper()) if ch_name.upper() in chan_names_normalized else None
        if ch_idx is None:
            logger.warning(f'Channel {ch_name} not found in data')
            continue

        logger.info(f'Processing {ch_name}...')

        measures = extract_channel_measures(X, times, trial_info, ch_names, ch_idx, config)
        logger.info(f'  {ch_name}: {len(measures)} subject×condition')

        # Concentration ANOVA
        conc_res = run_concentration_anova(measures)
        conc_res['channel'] = ch_name
        all_conc.append(conc_res)

        # JAR ANOVA
        jar_res = run_jar_anova_channel(measures)
        if len(jar_res):
            jar_res['channel'] = ch_name
            all_jar.append(jar_res)

    # Combine
    conc_df = pd.concat(all_conc, ignore_index=True)
    jar_df = pd.concat(all_jar, ignore_index=True) if all_jar else pd.DataFrame()

    # Thêm significance column
    for df in [conc_df, jar_df]:
        if len(df) and 'p' in df.columns:
            df['sig'] = df['p'].apply(
                lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns' if not np.isnan(p) else ''
            )

    # Save
    conc_df.to_csv(os.path.join(OUTPUT_DIR, 'concentration_anova_per_channel.csv'), index=False)
    if len(jar_df):
        jar_df.to_csv(os.path.join(OUTPUT_DIR, 'jar_anova_per_channel.csv'), index=False)

    # Print
    print('\n' + '=' * 72)
    print('  CONCENTRATION ANOVA PER CHANNEL (rmANOVA, balanced subjects)')
    print('=' * 72)
    print(f'  {"Ch":4s} {"Comp":8s} {"Measure":10s} {"F":8s} {"p":8s} {"sig":5s} {"np2":6s} {"n":4s}')
    print('  ' + '-' * 50)
    for _, r in conc_df.iterrows():
        if r['method'] == 'insufficient':
            continue
        print(f'  {r["channel"]:4s} {r["comp"]:8s} {r["measure"]:10s} '
              f'{r["F"]:8.3f} {r["p"]:8.4f} {r["sig"]:5s} '
              f'{r["np2"]:6.3f} {int(r["n_subjects"]):4d}')

    # Significant ones only
    sig_conc = conc_df[conc_df['p'] < 0.05]
    print(f'\n  --- Significant (p<0.05) ---')
    if len(sig_conc):
        for _, r in sig_conc.iterrows():
            print(f'  ✓ {r["channel"]:4s} {r["comp"]:8s} {r["measure"]:10s} '
                  f'F={r["F"]:.3f}, p={r["p"]:.4f} {r["sig"]}, np2={r["np2"]:.3f}')
    else:
        print('  (Không có channel nào significant cho concentration effect)')

    if len(jar_df):
        print('\n' + '=' * 72)
        print('  JAR ANOVA PER CHANNEL (1-way between-subjects)')
        print('=' * 72)
        print(f'  {"Ch":4s} {"Comp":8s} {"Measure":10s} {"F":8s} {"p":8s} {"sig":5s} {"n":5s}')
        print('  ' + '-' * 50)
        for _, r in jar_df.iterrows():
            print(f'  {r["channel"]:4s} {r["comp"]:8s} {r["measure"]:10s} '
                  f'{r["F"]:8.3f} {r["p"]:8.4f} {r["sig"]:5s} {int(r["n_total"]):5d}')

        sig_jar = jar_df[jar_df['p'] < 0.05]
        print(f'\n  --- Significant (p<0.05) ---')
        if len(sig_jar):
            for _, r in sig_jar.iterrows():
                print(f'  ✓ {r["channel"]:4s} {r["comp"]:8s} {r["measure"]:10s} '
                      f'F={r["F"]:.3f}, p={r["p"]:.4f} {r["sig"]}')
        else:
            print('  (Không có channel nào significant cho JAR effect)')

    # Heatmap
    if len(conc_df):
        all_conc_hm = []
        for ch_name in ch_list:
            ch_rows = conc_df[conc_df['channel'] == ch_name]
            all_conc_hm.append(ch_rows)
        conc_hm = pd.concat(all_conc_hm, ignore_index=True)

        path = plot_significance_heatmap(conc_hm,
            'Concentration Effect -log10(p) per Channel × Component',
            'concentration_heatmap.png')
        logger.info(f'Heatmap: {path}')

    logger.info(f'Done. Results in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
