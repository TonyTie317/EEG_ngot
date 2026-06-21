"""
Stage 06 — Onset-locked ERP (supplementary).

Re-epochs the preprocessed signal around cup-delivery onset (−0.5 to +2.0 s,
baseline −0.5 to −0.2 s), computes condition/substance grand-averages and
ROI waveforms + topographies. Exploratory: the continuous tasting paradigm has
no sharp sensory trigger.
"""

import _bootstrap  # noqa: F401
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from swt.config import load_config, setup_logging, fig_dir, report_path
from swt import erp, viz, report
from swt.constants import EEG_CHANNELS, ROIS, SUBSTANCE_COLORS, COND_COLORS, COND_DISPLAY, CONDITIONS


def roi_waveform(times, grand, roi, path, title):
    idx = [EEG_CHANNELS.index(c) for c in ROIS[roi] if c in EEG_CHANNELS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, data in grand.items():
        wave = data[idx].mean(axis=0) * 1e6  # µV
        color = SUBSTANCE_COLORS.get(g, COND_COLORS.get(g, 'k'))
        ax.plot(times, wave, label=COND_DISPLAY.get(g, g), color=color, lw=1.8)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('Time from onset (s)'); ax.set_ylabel('Amplitude (µV)')
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'erp')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 06 — Onset-locked ERP"); log.info("=" * 60)

    all_ep = erp.build_onset_epochs(cfg, log)
    if not all_ep:
        report.write(report_path(cfg, '06_erp.md'),
                     ["# Stage 06 — ERP\n", "No onset epochs could be built."])
        return

    times, grand_sub = erp.grand_average_by(all_ep, by='substance')
    _, grand_cond = erp.grand_average_by(all_ep, by='ma_mau')

    figs = {}
    for roi in ['Frontal', 'Gustatory', 'Central']:
        figs[f'sub_{roi}'] = roi_waveform(
            times, grand_sub, roi, os.path.join(fdir, f'erp_substance_{roi}.png'),
            f'{roi} ERP by substance (onset-locked)')
    figs['cond_Frontal'] = roi_waveform(
        times, grand_cond, 'Frontal', os.path.join(fdir, 'erp_condition_Frontal.png'),
        'Frontal ERP by condition (onset-locked)')

    # Topographies at a few post-onset latencies (sucrose − sucralose)
    lat = [0.1, 0.2, 0.35, 0.5]
    if 'Sucrose' in grand_sub and 'Sucralose' in grand_sub:
        vals, titles = [], []
        for t in lat:
            ti = int(np.argmin(np.abs(times - t)))
            vals.append((grand_sub['Sucrose'][:, ti] - grand_sub['Sucralose'][:, ti]) * 1e6)
            titles.append(f'{int(t*1000)} ms')
        figs['topo'] = viz.plot_topomap_row(
            vals, titles, os.path.join(fdir, 'erp_topo_sucrose_minus_sucralose.png'),
            suptitle='ERP Sucrose − Sucralose (µV) at post-onset latencies', symmetric=True)

    S = []
    S.append("# Stage 06 — Onset-locked ERP (supplementary)\n")
    S.append("Epochs −0.5 to +2.0 s around cup-delivery onset, baseline [−0.5, −0.2] s. "
             "Interpret cautiously: the 10 s tasting paradigm lacks a sharp sensory "
             "trigger, so ERP components are not the primary readout.\n")
    S.append("## ROI waveforms by substance\n")
    for roi in ['Frontal', 'Gustatory', 'Central']:
        S.append(report.img(figs[f'sub_{roi}'], rdir, f'{roi} ERP by substance'))
    S.append("## Frontal ERP by condition\n")
    S.append(report.img(figs['cond_Frontal'], rdir, 'Frontal ERP per condition'))
    if 'topo' in figs:
        S.append("## ERP difference topographies\n")
        S.append(report.img(figs['topo'], rdir, 'Sucrose − Sucralose ERP topographies'))

    report.write(report_path(cfg, '06_erp.md'), S)
    log.info(f"Stage 06 done → {report_path(cfg, '06_erp.md')}")


if __name__ == '__main__':
    main()
