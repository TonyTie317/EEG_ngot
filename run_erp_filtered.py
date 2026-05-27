#!/usr/bin/env python3
"""
Re-run ERP Analysis with quality-based filtering.

Usage:
    .venv/bin/python run_erp_filtered.py              # weak filter (default)
    .venv/bin/python run_erp_filtered.py --method strict  # chỉ giữ GOOD
    .venv/bin/python run_erp_filtered.py --compare        # chỉ so sánh
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import mne

from pipeline.config import load_config, setup_logging, ensure_dir as _mkdir
from pipeline.constants import (
    ALL_SUBJECTS, CONCENTRATIONS, CONCENTRATION_LABELS,
)
from pipeline.erp_analysis import apply_woody_realign
from pipeline.epoching import load_all_epochs

QUALITY_FLAGS = 'output/results/erp/erp_quality_flags.csv'
OUTPUT_DIR = 'output/results/erp_filtered'


def run_filtered_erp_analysis(config, logger, method='weak'):
    """Run ERP analysis with quality filter.

    Approach: subset epoch data + trial_info trước khi compute grand average.
    """
    logger.info('=' * 60)
    logger.info(f'ERP Analysis — Quality Filtered (method={method})')
    logger.info('=' * 60)

    # Load quality flags
    if not os.path.exists(QUALITY_FLAGS):
        logger.error(f'Missing {QUALITY_FLAGS}. Run run_erp_quality_check.py first.')
        return None, None
    qf = pd.read_csv(QUALITY_FLAGS)

    # Map: (subject_id, condition) → keep?
    if method == 'strict':
        keep_set = set()
        for _, row in qf[qf['quality_label'] == 'GOOD'].iterrows():
            keep_set.add((row['subject_id'], int(row['condition'])))
        logger.info(f'Strict filter: giữ {len(keep_set)} subject×condition (chỉ GOOD)')
    else:
        keep_set = set()
        for _, row in qf[qf['has_real_pattern'] == True].iterrows():
            keep_set.add((row['subject_id'], int(row['condition'])))
        logger.info(f'Weak filter: giữ {len(keep_set)} subject×condition (has_real_pattern)')

    # Load epochs (từ disk, qua load_all_epochs)
    all_epochs, all_trial_info = load_all_epochs(config, logger)
    if not all_epochs:
        logger.error('No epochs loaded')
        return None, None

    # Apply Woody realignment
    all_epochs, all_trial_info = apply_woody_realign(all_epochs, all_trial_info, logger)

    # Concatenate all epoch data (sau realignment)
    all_data_list = []
    for ep in all_epochs:
        all_data_list.append(ep.get_data())
    X = np.concatenate(all_data_list, axis=0)  # (n_total_epochs, n_ch, n_times)
    n_epochs_total = X.shape[0]
    n_channels = X.shape[1]
    n_times = X.shape[2]

    logger.info(f'Total epochs after realign: {n_epochs_total}')

    # Build mask: rows in all_trial_info to keep
    keep_mask = all_trial_info.apply(
        lambda r: (r['subject_id'], int(r['condition'])) in keep_set, axis=1
    )
    n_kept = keep_mask.sum()
    n_total = len(keep_mask)
    logger.info(f'Filter: giữ {n_kept}/{n_total} trials ({n_kept/n_total*100:.0f}%)')

    if n_kept < 5:
        logger.error('Quá ít trials sau filter')
        return None, None

    # Filter data + trial_info
    X_filt = X[keep_mask.values]
    ti_filt = all_trial_info[keep_mask].reset_index(drop=True)

    # Info từ epochs đầu tiên
    info = all_epochs[0].info
    tmin = all_epochs[0].tmin

    # ── Build results ─────────────────────────────────────────────────────────
    from pipeline.erp_analysis import extract_component_measures, compare_by_concentration
    from pipeline.erp_analysis import compute_difference_waves
    import mne

    results = {}
    logger.info(f'Filtered data: {X_filt.shape[0]} epochs, {n_channels} ch, {n_times} times')

    # Evoked: all
    results['evoked_all'] = mne.EvokedArray(
        X_filt.mean(axis=0), info.copy(), tmin=tmin, comment='All (filtered)'
    )

    # Evoked: by condition
    evoked_by_condition = {}
    for cond in CONCENTRATIONS:
        mask = ti_filt['condition'] == cond
        if mask.sum() > 0:
            evoked_by_condition[cond] = mne.EvokedArray(
                X_filt[mask.values].mean(axis=0), info.copy(), tmin=tmin,
                comment=CONCENTRATION_LABELS.get(cond, str(cond))
            )
    results['evoked_by_condition'] = evoked_by_condition

    # Evoked: by JAR group
    evoked_by_jar = {}
    for group_name in ['Khong_du', 'Vua_phai', 'Qua_nhieu']:
        mask = ti_filt['jar_group'] == group_name
        if mask.sum() > 0:
            evoked_by_jar[group_name] = mne.EvokedArray(
                X_filt[mask.values].mean(axis=0), info.copy(), tmin=tmin,
                comment=group_name
            )
    results['evoked_by_jar_group'] = evoked_by_jar

    # Component measures: tạo list EpochsArray tạm theo subject
    filtered_epochs_list = []
    for sid in ALL_SUBJECTS:
        subj_mask = ti_filt['subject_id'] == sid
        if subj_mask.sum() > 0:
            subj_data_tmp = X_filt[subj_mask.values]
            ep_tmp = mne.EpochsArray(
                subj_data_tmp, info.copy(), tmin=tmin, verbose=False
            )
            filtered_epochs_list.append(ep_tmp)

    if filtered_epochs_list:
        results['measures'] = extract_component_measures(
            filtered_epochs_list, ti_filt, config, logger
        )

        # Add JAR
        jar_lookup = ti_filt.groupby(['subject_id', 'condition'])['jar_group'].first().reset_index()
        results['measures'] = results['measures'].merge(
            jar_lookup, on=['subject_id', 'condition'], how='left'
        )

        # Concentration comparison
        results['concentration_summary'] = compare_by_concentration(
            results['measures'], config, logger
        )
    else:
        results['measures'] = pd.DataFrame()
        results['concentration_summary'] = pd.DataFrame()

    # Difference waves
    results['diff_waves'] = compute_difference_waves(
        filtered_epochs_list if filtered_epochs_list else all_epochs,
        ti_filt, config, logger
    )

    # Lưu
    suffix = f'_{method}' if method != 'weak' else ''
    _mkdir(OUTPUT_DIR)
    if len(results['measures']):
        results['measures'].to_csv(
            os.path.join(OUTPUT_DIR, f'component_measures_filtered{suffix}.csv'), index=False
        )
    if len(results['concentration_summary']):
        results['concentration_summary'].to_csv(
            os.path.join(OUTPUT_DIR, f'concentration_summary_filtered{suffix}.csv'), index=False
        )
    ti_filt.to_csv(
        os.path.join(OUTPUT_DIR, f'trial_info_filtered{suffix}.csv'), index=False
    )
    logger.info(f'Filtered results saved to {OUTPUT_DIR}')

    return results, ti_filt


def print_concentration_results(cs):
    """In kết quả concentration comparison."""
    print(f'\n{"="*60}')
    print('  CONCENTRATION COMPARISON (filtered)')
    print(f'{"="*60}')
    for comp in ['P1', 'N1', 'P2', 'N400']:
        print(f'\n  --- {comp} mean_amp ---')
        cdata = cs[(cs['component'] == comp) & (cs['measure'] == 'mean_amp')]
        for _, r in cdata.iterrows():
            print(f'  {r["condition_label"]:15s}: {r["mean"]:+.3e} ± {r["sem"]:.2e} (n={int(r["n_subjects"])})')

    # P2 peak_amp
    print(f'\n  --- P2 peak_amp ---')
    cdata = cs[(cs['component'] == 'P2') & (cs['measure'] == 'peak_amp')]
    for _, r in cdata.iterrows():
        print(f'  {r["condition_label"]:15s}: {r["mean"]:+.3e} ± {r["sem"]:.2e} (n={int(r["n_subjects"])})')


def compare_results(config, logger):
    """So sánh filtered vs unfiltered."""
    unfiltered_path = os.path.join(
        config['paths']['results_base'], 'erp', 'concentration_summary.csv'
    )
    if not os.path.exists(unfiltered_path):
        logger.warning(f'Không tìm thấy unfiltered results: {unfiltered_path}')
        return

    unfiltered = pd.read_csv(unfiltered_path)
    _mkdir(OUTPUT_DIR)

    filtered_suffixes = [('weak', ''), ('strict', '_strict')]

    for method, suffix in filtered_suffixes:
        filtered_path = os.path.join(OUTPUT_DIR, f'concentration_summary_filtered{suffix}.csv')
        if not os.path.exists(filtered_path):
            continue
        filtered = pd.read_csv(filtered_path)

        comp = 'P2'
        measure = 'mean_amp'

        lines = [
            f'{"="*70}',
            f'  SO SÁNH: Unfiltered vs Filtered ({method})',
            f'  Component: {comp}, measure: {measure}',
            f'{"="*70}',
            f'  {"Condition":15s} {"Unfiltered n":10s} {"Filtered n":10s} {"Unfilt mean":12s} {"Filt mean":12s} {"Diff":10s}',
            f'{"-"*70}',
        ]

        u = unfiltered[(unfiltered['component'] == comp) & (unfiltered['measure'] == measure)]
        f = filtered[(filtered['component'] == comp) & (filtered['measure'] == measure)]

        for (_, ur), (_, fr) in zip(u.iterrows(), f.iterrows()):
            diff = fr['mean'] - ur['mean']
            lines.append(
                f'  {ur["condition_label"]:15s} {int(ur["n_subjects"]):10d} {int(fr["n_subjects"]):10d} '
                f'{ur["mean"]:+.3e}  {fr["mean"]:+.3e}  {diff:+.2e}'
            )

        lines.append(f'{"="*70}')
        report = '\n'.join(lines)
        print(f'\n{report}')

        report_path = os.path.join(OUTPUT_DIR, f'comparison_{method}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['weak', 'strict', 'both', 'compare'],
                        default='both')
    parser.add_argument('--skip-run', action='store_true')
    args = parser.parse_args()

    config = load_config('configs/config.yaml')
    logger = setup_logging(config)

    if args.method == 'compare':
        compare_results(config, logger)
        return

    if args.skip_run:
        compare_results(config, logger)
        return

    if args.method in ('weak', 'both'):
        results, ti = run_filtered_erp_analysis(config, logger, method='weak')
        if results:
            print_concentration_results(results['concentration_summary'])

    if args.method in ('strict', 'both'):
        results2, ti2 = run_filtered_erp_analysis(config, logger, method='strict')
        if results2:
            print_concentration_results(results2['concentration_summary'])

    compare_results(config, logger)


if __name__ == '__main__':
    main()
