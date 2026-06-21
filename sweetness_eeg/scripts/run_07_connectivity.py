"""
Stage 07 — Frontal–Gustatory functional connectivity (supplementary).

Magnitude-squared coherence between the Frontal and Gustatory ROIs per frequency
band, compared between sucrose and sucralose and across intensities.
"""

import _bootstrap  # noqa: F401
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, connectivity, stats, viz, report
from swt.constants import BAND_ORDER, SUBSTANCE_COLORS


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'connectivity')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 07 — Connectivity"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    coh = connectivity.roi_pair_coherence(all_epochs, cfg, 'Frontal', 'Gustatory', log)
    coh.to_csv(result_path(cfg, 'connectivity', 'frontal_gustatory_coherence.csv'),
               index=False)

    subj = (coh.groupby(['subject', 'substance', 'intensity', 'ma_mau', 'band'])
            ['coherence'].mean().reset_index())

    # Bar: coherence per band × substance
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BAND_ORDER)); width = 0.38
    for k, s in enumerate(['Sucrose', 'Sucralose']):
        means, sems = [], []
        for b in BAND_ORDER:
            v = subj[(subj.substance == s) & (subj.band == b)]['coherence']
            means.append(v.mean()); sems.append(v.std(ddof=1)/np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(x + (k-0.5)*width, means, width, yerr=sems, capsize=4,
               color=SUBSTANCE_COLORS[s], label=s)
    ax.set_xticks(x); ax.set_xticklabels(BAND_ORDER)
    ax.set_ylabel('Frontal–Gustatory coherence')
    ax.set_title('Fronto-gustatory coherence by band & substance')
    ax.legend(); fig.tight_layout()
    barfig = os.path.join(fdir, 'coherence_by_band_substance.png')
    fig.savefig(barfig, dpi=150); plt.close(fig)

    # Contrasts
    contrasts = []
    for band in BAND_ORDER:
        sub = subj[subj.band == band].rename(columns={'coherence': 'val'})
        c = stats.paired_substance_contrasts(sub, 'val'); c.insert(0, 'band', band)
        contrasts.append(c)
    ctab = pd.concat(contrasts, ignore_index=True)
    ctab.to_csv(result_path(cfg, 'connectivity', 'coherence_contrasts.csv'), index=False)

    S = []
    S.append("# Stage 07 — Frontal–Gustatory Connectivity (supplementary)\n")
    S.append("Magnitude-squared coherence between the Frontal ROI (reward/hedonic) and "
             "the Gustatory ROI (insular/opercular scalp proxy), per band, sucrose vs "
             "sucralose.\n")
    S.append(report.img(barfig, rdir, 'Coherence by band & substance'))
    S.append("## Sucrose vs sucralose coherence contrasts (FDR-corrected)\n")
    S.append(report.df_to_md(ctab))
    report.write(report_path(cfg, '07_connectivity.md'), S)
    log.info(f"Stage 07 done → {report_path(cfg, '07_connectivity.md')}")


if __name__ == '__main__':
    main()
