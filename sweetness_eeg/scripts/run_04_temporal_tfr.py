"""
Stage 04 — Temporal dynamics & time-frequency.

Time-resolved band power across the 10 s window, early-vs-late ("aftertaste")
contrasts, and Morlet TFR spectrograms per substance. Tests the prolonged
neural-activation hypothesis for the lingering sucralose aftertaste.
"""

import _bootstrap  # noqa: F401
import os

import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, spectral, stats, viz, report
from swt.constants import PAPER_ROIS


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'temporal')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 04 — Temporal / TFR"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    figs = {}

    # 1. Time-resolved band power for paper ROIs
    for roi in PAPER_ROIS:
        tr = spectral.time_resolved_bandpower(all_epochs, cfg, roi=roi)
        tr.to_csv(result_path(cfg, 'temporal', f'time_resolved_{roi}.csv'), index=False)
        for band in ['theta', 'alpha', 'gamma']:
            key = f'tr_{roi}_{band}'
            figs[key] = viz.plot_time_resolved(
                tr, band, os.path.join(fdir, f'{key}.png'),
                title=f'{roi} {band} power over the tasting window', by='substance')

    # 2. Early vs late window band power (aftertaste contrast)
    sp = cfg['spectral']
    early = spectral.window_bandpower(all_epochs, cfg, tuple(sp['early_window']), 'Gustatory')
    late = spectral.window_bandpower(all_epochs, cfg, tuple(sp['late_window']), 'Gustatory')
    early['win'] = 'early'; late['win'] = 'late'
    win_df = pd.concat([early, late], ignore_index=True)
    win_df.to_csv(result_path(cfg, 'temporal', 'early_late_gustatory.csv'), index=False)

    # late−early per subject×condition, then sucrose vs sucralose contrast per band
    persubj = (win_df.groupby(['subject', 'substance', 'intensity', 'ma_mau', 'band', 'win'])
               ['power'].mean().reset_index())
    pv = persubj.pivot_table(index=['subject', 'substance', 'intensity', 'ma_mau', 'band'],
                             columns='win', values='power').reset_index()
    pv['late_minus_early'] = pv['late'] - pv['early']
    late_contrasts = {}
    for band in ['theta', 'alpha', 'gamma']:
        sub = pv[pv['band'] == band].rename(columns={'late_minus_early': 'val'})
        late_contrasts[band] = stats.paired_substance_contrasts(sub, 'val')
        late_contrasts[band].to_csv(
            result_path(cfg, 'temporal', f'aftertaste_contrast_{band}.csv'), index=False)

    # bar of late-minus-early by substance per band
    barfig = os.path.join(fdir, 'aftertaste_late_minus_early.png')
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from swt.constants import SUBSTANCE_COLORS
    bands = ['theta', 'alpha', 'gamma']
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bands)); width = 0.38
    for k, s in enumerate(['Sucrose', 'Sucralose']):
        means, sems = [], []
        for b in bands:
            v = pv[(pv.substance == s) & (pv.band == b)]['late_minus_early']
            means.append(v.mean()); sems.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(x + (k - 0.5) * width, means, width, yerr=sems, capsize=4,
               color=SUBSTANCE_COLORS[s], label=s)
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(bands)
    ax.set_ylabel('Late − early power (V²)')
    ax.set_title('Gustatory ROI: late-window minus early-window power\n'
                 '(positive = sustained/lingering activation)')
    ax.legend(); fig.tight_layout(); fig.savefig(barfig, dpi=150); plt.close(fig)
    figs['aftertaste_bar'] = barfig

    # 3. TFR spectrograms per substance (Frontal)
    fr, tt, tfr = spectral.compute_tfr_by_substance(all_epochs, cfg, roi='Frontal')
    figs['tfr'] = viz.plot_tfr(fr, tt, tfr, os.path.join(fdir, 'tfr_frontal_by_substance.png'),
                               title='Frontal ROI time-frequency power (dB vs trial mean)')

    # ── Report ───────────────────────────────────────────────────────────────
    S = []
    S.append("# Stage 04 — Temporal Dynamics & Time-Frequency\n")
    S.append(f"Time-resolved band power (1 s windows, 0.5 s step), early "
             f"({sp['early_window']} s) vs late ({sp['late_window']} s) contrasts in the "
             "Gustatory ROI, and Morlet TFR per substance. The late-window contrast "
             "probes the *prolonged activation* expected for sucralose's lingering "
             "aftertaste.\n")

    S.append("## Time-resolved band power\n")
    for roi in PAPER_ROIS:
        for band in ['theta', 'alpha', 'gamma']:
            S.append(report.img(figs[f'tr_{roi}_{band}'], rdir, f'{roi} {band} over time'))

    S.append("## Aftertaste: late vs early window (Gustatory ROI)\n")
    S.append(report.img(figs['aftertaste_bar'], rdir, 'Late − early power by substance'))
    S.append("Sucrose vs sucralose paired contrasts on *late − early* power:\n")
    for band in ['theta', 'alpha', 'gamma']:
        S.append(f"**{band}**\n")
        S.append(report.df_to_md(late_contrasts[band]))

    S.append("## Time-frequency (Frontal ROI)\n")
    S.append(report.img(figs['tfr'], rdir, 'TFR per substance'))

    report.write(report_path(cfg, '04_temporal_tfr.md'), S)
    log.info(f"Stage 04 done → {report_path(cfg, '04_temporal_tfr.md')}")


if __name__ == '__main__':
    main()
