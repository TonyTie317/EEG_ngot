"""
Stage 05 — Inferential statistics on the spectral measures.

Two-way repeated-measures ANOVA (substance × intensity) for every band × ROI,
sucrose-vs-sucralose paired contrasts, condition-vs-water tests, and cluster-based
permutation tests on the ROI PSD spectra.
"""

import _bootstrap  # noqa: F401
import os

import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, spectral, stats, report
from swt.constants import BAND_ORDER, PAPER_ROIS


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'stats')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 05 — Statistics"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    bp = spectral.bandpower_table(all_epochs, cfg)
    roi_bp = spectral.roi_bandpower(bp)
    subj_roi = (roi_bp.groupby(['subject', 'ma_mau', 'substance', 'intensity', 'roi', 'band'])
                [['abs_power', 'rel_power']].mean().reset_index())

    # 1. ANOVA per band × ROI
    anova_tbl = stats.anova_per_band_roi(roi_bp, 'rel_power')
    anova_tbl.to_csv(result_path(cfg, 'stats', 'anova_band_roi.csv'), index=False)

    # 2. Sucrose vs sucralose contrasts (frontal/gustatory θ,α,γ)
    contrasts = []
    for roi in PAPER_ROIS:
        for band in ['theta', 'alpha', 'gamma']:
            sub = subj_roi[(subj_roi.roi == roi) & (subj_roi.band == band)].copy()
            c = stats.paired_substance_contrasts(sub.rename(columns={'rel_power': 'val'}), 'val')
            c.insert(0, 'roi', roi); c.insert(1, 'band', band)
            contrasts.append(c)
    contrast_tbl = pd.concat(contrasts, ignore_index=True)
    contrast_tbl.to_csv(result_path(cfg, 'stats', 'substance_contrasts.csv'), index=False)

    # 3. Condition vs water (frontal theta as representative)
    water_tbl = []
    for roi in PAPER_ROIS:
        for band in ['theta', 'alpha']:
            sub = subj_roi[(subj_roi.roi == roi) & (subj_roi.band == band)].copy()
            w = stats.condition_vs_water(sub.rename(columns={'rel_power': 'val'}), 'val')
            w.insert(0, 'roi', roi); w.insert(1, 'band', band)
            water_tbl.append(w)
    water_tbl = pd.concat(water_tbl, ignore_index=True)
    water_tbl.to_csv(result_path(cfg, 'stats', 'condition_vs_water.csv'), index=False)

    # 4. Cluster-based permutation tests on PSD (sucrose vs sucralose) per ROI
    cluster_results = {}
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for roi in PAPER_ROIS:
        res = stats.cluster_permutation_psd(all_epochs, cfg, roi, logger=log)
        if res is None:
            continue
        cluster_results[roi] = res
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(res['freqs'], res['T_obs'], color='k', lw=1.5)
        for mask, p in zip(res['clusters'], res['cluster_pv']):
            if p < cfg['stats'].get('alpha', 0.05):
                ax.axvspan(res['freqs'][mask][0], res['freqs'][mask][-1],
                           color='red', alpha=0.25)
        ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('cluster statistic')
        ax.set_title(f'{roi}: Sucrose vs Sucralose PSD cluster test '
                     f'(red = p<{cfg["stats"]["alpha"]})')
        p = os.path.join(fdir, f'cluster_{roi}.png')
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        cluster_results[roi]['fig'] = p

    # ── Report ───────────────────────────────────────────────────────────────
    S = []
    S.append("# Stage 05 — Statistics\n")
    S.append("Two-way repeated-measures ANOVA (substance × intensity, sweet conditions, "
             "subjects with complete cells), paired sucrose-vs-sucralose contrasts "
             "(FDR-corrected), condition-vs-water tests, and cluster permutation on the "
             "ROI PSD.\n")

    S.append("## rmANOVA p-values per band × ROI (relative power)\n")
    pcols = [c for c in anova_tbl.columns if c.startswith('p_')]
    S.append(report.df_to_md(anova_tbl[['roi', 'band'] + pcols]))

    S.append("## Sucrose vs sucralose contrasts (relative power)\n")
    S.append(report.df_to_md(contrast_tbl))

    S.append("## Sweet conditions vs water\n")
    S.append(report.df_to_md(water_tbl))

    if cluster_results:
        S.append("## Cluster-based permutation tests on PSD\n")
        for roi, res in cluster_results.items():
            sigtxt = ', '.join(f'{a:.1f}–{b:.1f} Hz (p={p:.3f})'
                               for a, b, p in res['significant']) or 'no significant clusters'
            S.append(f"**{roi}** — {sigtxt}\n")
            S.append(report.img(res['fig'], rdir, f'{roi} cluster test'))

    report.write(report_path(cfg, '05_stats.md'), S)
    log.info(f"Stage 05 done → {report_path(cfg, '05_stats.md')}")


if __name__ == '__main__':
    main()
