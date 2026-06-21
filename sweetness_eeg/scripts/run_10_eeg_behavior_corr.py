"""
Stage 10 — EEG ↔ behavior correlation.

Links frontal θ/α and gustatory γ relative power to liking, sweetness-JAR and
sweet-aftertaste. EEG and behavior are joined per subject × condition (PXXX ↔
Excel id XXX). Reports both group-level (condition means) and within-subject
(pooled subject × condition) Pearson correlations.
"""

import _bootstrap  # noqa: F401
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, spectral, behavior, viz, report
from swt.constants import COND_DISPLAY


EEG_METRICS = [('Frontal', 'theta'), ('Frontal', 'alpha'), ('Gustatory', 'gamma')]
BEH_MEASURES = ['liking', 'sweetness_jar', 'aftertaste']


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'eeg_behavior')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 10 — EEG ↔ behavior"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    bp = spectral.bandpower_table(all_epochs, cfg)
    roi_bp = spectral.roi_bandpower(bp)
    eeg = (roi_bp.groupby(['subject', 'ma_mau', 'roi', 'band'])['rel_power']
           .mean().reset_index())

    # wide EEG metrics per subject × condition
    pieces = []
    for roi, band in EEG_METRICS:
        sub = eeg[(eeg.roi == roi) & (eeg.band == band)][['subject', 'ma_mau', 'rel_power']]
        sub = sub.rename(columns={'rel_power': f'{roi}_{band}'})
        pieces.append(sub.set_index(['subject', 'ma_mau']))
    eeg_wide = pd.concat(pieces, axis=1).reset_index()

    beh = behavior.load_behavior_long(cfg['paths']['behavior_xlsx'], logger=log)
    beh_sc = behavior.subject_condition_means(beh)[
        ['subject', 'ma_mau', 'liking', 'sweetness_jar', 'aftertaste']]

    merged = eeg_wide.merge(beh_sc, on=['subject', 'ma_mau'], how='inner')
    from swt.constants import label_to_substance
    merged['substance'] = merged['ma_mau'].map(label_to_substance)
    merged.to_csv(result_path(cfg, 'eeg_behavior', 'eeg_behavior_merged.csv'), index=False)
    n_eeg_subj = eeg_wide['subject'].nunique()
    n_merged_subj = merged['subject'].nunique()
    log.info(f"Joined {n_merged_subj}/{n_eeg_subj} EEG subjects to behavior")

    eeg_cols = [f'{r}_{b}' for r, b in EEG_METRICS]

    # ── Within-subject (pooled subject × condition) correlations ─────────────
    rows = []
    for ec in eeg_cols:
        for bm in BEH_MEASURES:
            d = merged[[ec, bm]].dropna()
            if len(d) > 3:
                r, p = sps.pearsonr(d[ec], d[bm])
            else:
                r, p = np.nan, np.nan
            rows.append({'eeg_metric': ec, 'behavior': bm, 'level': 'within_subject',
                         'n': len(d), 'r': r, 'p': p})
    # ── Group-level (condition means) correlations ───────────────────────────
    cond_mean = merged.groupby('ma_mau')[eeg_cols + BEH_MEASURES].mean().reset_index()
    for ec in eeg_cols:
        for bm in BEH_MEASURES:
            d = cond_mean[[ec, bm]].dropna()
            if len(d) > 3:
                r, p = sps.pearsonr(d[ec], d[bm])
            else:
                r, p = np.nan, np.nan
            rows.append({'eeg_metric': ec, 'behavior': bm, 'level': 'group_condition',
                         'n': len(d), 'r': r, 'p': p})
    corr = pd.DataFrame(rows)
    corr.to_csv(result_path(cfg, 'eeg_behavior', 'correlations.csv'), index=False)

    # ── Scatter plots for the headline pairs ─────────────────────────────────
    headline = [('Frontal_theta', 'liking'), ('Frontal_alpha', 'liking'),
                ('Frontal_theta', 'sweetness_jar'), ('Gustatory_gamma', 'aftertaste')]
    figs = {}
    for ec, bm in headline:
        d = merged[['substance', ec, bm]].dropna()
        figs[(ec, bm)] = viz.plot_scatter_corr(
            d[ec], d[bm], os.path.join(fdir, f'{ec}_vs_{bm}.png'),
            xlabel=f'{ec} rel. power', ylabel=bm,
            title=f'{ec} vs {bm} (subject × condition)',
            hue=d['substance'].values)

    S = []
    S.append("# Stage 10 — EEG ↔ Behavior Correlation\n")
    S.append(f"Frontal θ/α and gustatory γ relative power linked to liking, sweetness "
             f"JAR and sweet aftertaste. EEG↔behavior join: PXXX ↔ Excel id XXX. "
             f"Joined **{n_merged_subj}/{n_eeg_subj}** EEG subjects.\n")
    S.append("## Correlation table\n")
    S.append(report.df_to_md(corr))
    S.append("## Headline scatter plots\n")
    for ec, bm in headline:
        S.append(report.img(figs[(ec, bm)], rdir, f'{ec} vs {bm}'))
    report.write(report_path(cfg, '10_eeg_behavior_corr.md'), S)
    log.info(f"Stage 10 done → {report_path(cfg, '10_eeg_behavior_corr.md')}")


if __name__ == '__main__':
    main()
