"""
Stage 03 — Spectral / PSD analysis (core, mirrors the reference paper).

Welch PSD over the 10 s window; absolute & relative band power for δ/θ/α/β/γ;
ROI aggregation (Frontal & Gustatory); grand-average spectra; band topographies
per condition; sucrose−sucralose difference maps; dose-response of frontal θ/α.
"""

import _bootstrap  # noqa: F401
import os

import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, spectral, viz, report
from swt.constants import (
    BAND_ORDER, CONDITIONS, COND_DISPLAY, PAPER_ROIS, SWEET_CONDITIONS,
)


def roi_band_summary(roi_bp, roi, band, value='rel_power'):
    """mean ± SEM by substance × intensity for one ROI × band (subject-level)."""
    sub = roi_bp[(roi_bp['roi'] == roi) & (roi_bp['band'] == band)]
    # subject mean per condition first
    per = sub.groupby(['subject', 'substance', 'intensity'])[value].mean().reset_index()
    rows = []
    for (s, inten), g in per.groupby(['substance', 'intensity']):
        rows.append({'substance': s, 'intensity': inten,
                     f'{value}_mean': g[value].mean(),
                     f'{value}_sem': g[value].std(ddof=1) / np.sqrt(len(g))
                     if len(g) > 1 else 0.0})
    return pd.DataFrame(rows)


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'spectral')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 03 — Spectral / PSD"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    bp = spectral.bandpower_table(all_epochs, cfg, log)
    roi_bp = spectral.roi_bandpower(bp)
    bp.to_csv(result_path(cfg, 'spectral', 'bandpower_channel_long.csv'), index=False)
    roi_bp.to_csv(result_path(cfg, 'spectral', 'bandpower_roi_long.csv'), index=False)

    figs = {}

    # 1. Grand-average PSD spectra
    freqs, grand = spectral.grand_psd_by_condition(all_epochs, cfg)
    figs['psd'] = viz.plot_psd_spectra(
        freqs, grand, os.path.join(fdir, 'grand_psd_by_condition.png'),
        'Grand-average PSD by condition (channel-averaged)')

    # 2. Topomap grid: bands × conditions (relative power)
    figs['topo_rel'] = viz.plot_topomap_grid(
        lambda c, b: spectral.topomap_band_values(bp, c, b, 'rel_power'),
        BAND_ORDER, CONDITIONS, os.path.join(fdir, 'topomap_relpower_grid.png'),
        title='Relative band power topographies (band × condition)',
        cmap='viridis', symmetric=False)

    # 3. Sucrose − Sucralose difference topographies per band (pooled sweet)
    def diff_vals(_c, band):
        suc = bp[(bp.substance == 'Sucrose') & (bp.band == band)]
        scl = bp[(bp.substance == 'Sucralose') & (bp.band == band)]
        from swt.constants import EEG_CHANNELS
        a = suc.groupby('channel')['rel_power'].mean()
        b = scl.groupby('channel')['rel_power'].mean()
        return np.array([a.get(ch, np.nan) - b.get(ch, np.nan) for ch in EEG_CHANNELS])
    figs['topo_diff'] = viz.plot_topomap_row(
        [diff_vals(None, b) for b in BAND_ORDER], BAND_ORDER,
        os.path.join(fdir, 'topomap_sucrose_minus_sucralose.png'),
        suptitle='Sucrose − Sucralose (relative power)', symmetric=True)

    # 4. ROI band-power boxes (Frontal & Gustatory) for theta & alpha
    for roi in PAPER_ROIS:
        for band in ['theta', 'alpha']:
            sub = roi_bp[(roi_bp.roi == roi) & (roi_bp.band == band)].copy()
            key = f'box_{roi}_{band}'
            figs[key] = viz.plot_condition_box(
                sub, 'rel_power', os.path.join(fdir, f'{key}.png'),
                f'{roi} {band} relative power by condition', f'{band} rel. power')

    # 5. Dose-response of frontal theta & alpha (relative power)
    dr_summaries = {}
    for band in ['theta', 'alpha']:
        s = roi_band_summary(roi_bp, 'Frontal', band, 'rel_power')
        dr_summaries[band] = s
        s.to_csv(result_path(cfg, 'spectral', f'frontal_{band}_dose.csv'), index=False)
        figs[f'dose_{band}'] = viz.plot_dose_response(
            s, 'rel_power_mean', 'rel_power_sem',
            os.path.join(fdir, f'frontal_{band}_dose.png'),
            f'Frontal {band} relative power vs intensity', f'{band} rel. power')

    # 6. Band-power summary table per paper ROI × band × substance
    rows = []
    for roi in PAPER_ROIS:
        for band in BAND_ORDER:
            sub = roi_bp[(roi_bp.roi == roi) & (roi_bp.band == band)]
            per = sub.groupby(['subject', 'substance'])['rel_power'].mean().reset_index()
            for s in ['Sucrose', 'Sucralose', 'Water']:
                v = per[per.substance == s]['rel_power']
                if len(v):
                    rows.append({'roi': roi, 'band': band, 'substance': s,
                                 'rel_power_mean': v.mean(),
                                 'rel_power_sem': v.std(ddof=1) / np.sqrt(len(v))
                                 if len(v) > 1 else 0.0})
    band_tbl = pd.DataFrame(rows)
    band_tbl.to_csv(result_path(cfg, 'spectral', 'roi_band_substance.csv'), index=False)

    # ── Report ───────────────────────────────────────────────────────────────
    S = []
    S.append("# Stage 03 — Spectral / Power Spectral Density\n")
    S.append("Welch PSD (2 s segments, 50 % overlap) over each 10 s tasting window. "
             "Band power integrated for δ (1–4), θ (4–8), α (8–13), β (13–30), "
             "γ (30–45 Hz); relative power = band / total (1–45 Hz). ROIs: "
             "**Frontal** (Fp1/Fp2/F3/F4/F7/F8, reward & hedonic) and **Gustatory** "
             "(T7/T8/C3/C4 — scalp proxy over insula/operculum).\n")

    S.append("## Grand-average spectra\n")
    S.append(report.img(figs['psd'], rdir, 'PSD by condition'))

    S.append("## Band-power topographies\n")
    S.append(report.img(figs['topo_rel'], rdir,
                        'Relative band power per band (rows) × condition (cols)'))
    S.append("### Sucrose vs sucralose contrast\n")
    S.append(report.img(figs['topo_diff'], rdir,
                        'Sucrose − Sucralose relative power (red = higher for sucrose)'))

    S.append("## Frontal & gustatory θ / α modulation\n")
    for roi in PAPER_ROIS:
        for band in ['theta', 'alpha']:
            S.append(report.img(figs[f'box_{roi}_{band}'], rdir,
                                f'{roi} {band} relative power'))

    S.append("## Dose-response of frontal θ / α (reward/hedonic hypothesis)\n")
    S.append(report.img(figs['dose_theta'], rdir, 'Frontal theta vs intensity'))
    S.append(report.img(figs['dose_alpha'], rdir, 'Frontal alpha vs intensity'))

    S.append("## Relative band power by ROI × substance\n")
    pt = band_tbl.pivot_table(index=['roi', 'band'], columns='substance',
                              values='rel_power_mean').reset_index()
    pt = pt.reindex(columns=['roi', 'band', 'Water', 'Sucrose', 'Sucralose'])
    S.append(report.df_to_md(pt))

    report.write(report_path(cfg, '03_spectral_psd.md'), S)
    log.info(f"Stage 03 done → {report_path(cfg, '03_spectral_psd.md')}")


if __name__ == '__main__':
    main()
