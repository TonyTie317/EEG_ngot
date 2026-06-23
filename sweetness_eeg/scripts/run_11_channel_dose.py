"""
Stage 11 — Per-channel, within-substance dose-response of band power.

Two questions the ROI-level Stage 03/05 cannot answer:

1. **Within each substance separately** (Sucrose alone, Sucralose alone), how does
   band power change as concentration rises across the three intensities (5 / 7.5 /
   12 %)? Stage 05 only contrasted sucrose *vs* sucralose; here each substance is
   analysed on its own dose axis.
2. **Per individual channel** (all 16 electrodes) instead of pooled ROIs, so the
   spatial pattern of any dose effect is resolved electrode-by-electrode.

For every substance × band × channel we run a one-way repeated-measures ANOVA over
intensity (omnibus dose effect) and a per-subject linear slope test (direction of
the effect), FDR-corrected across channels within each band. Outputs: slope-t
topographies (significant channels circled), channel × band heatmaps, and
dose-response curves for the channels with the strongest dose effects.
"""

import _bootstrap  # noqa: F401
import os

import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, spectral, stats, viz, report
from swt.constants import BAND_ORDER, EEG_CHANNELS, INTENSITIES

SWEET_SUBSTANCES = ['Sucrose', 'Sucralose']
VALUE = 'rel_power'
ALPHA = 0.05


def channel_dose_summary(bp, substance, channel, band, value=VALUE):
    """mean ± SEM by intensity for one substance × channel × band (subject-level)."""
    sub = bp[(bp.substance == substance) & (bp.channel == channel) &
             (bp.band == band)]
    per = sub.groupby(['subject', 'intensity'])[value].mean().reset_index()
    rows = []
    for inten, g in per.groupby('intensity'):
        rows.append({'substance': substance, 'intensity': inten,
                     f'{value}_mean': g[value].mean(),
                     f'{value}_sem': g[value].std(ddof=1) / np.sqrt(len(g))
                     if len(g) > 1 else 0.0})
    return pd.DataFrame(rows)


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'channel_dose')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 11 — Per-channel dose-response"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    bp = spectral.bandpower_table(all_epochs, cfg, log)
    bp.to_csv(result_path(cfg, 'channel_dose', 'bandpower_channel_long.csv'),
              index=False)

    # ── 1. Per-channel within-substance dose statistics ───────────────────────
    dose = {s: stats.perchannel_dose_effect(bp, s, VALUE) for s in SWEET_SUBSTANCES}
    for s, tbl in dose.items():
        tbl.to_csv(result_path(cfg, 'channel_dose', f'perchannel_dose_{s.lower()}.csv'),
                   index=False)
    dose_all = pd.concat(dose.values(), ignore_index=True)
    dose_all.to_csv(result_path(cfg, 'channel_dose', 'perchannel_dose_all.csv'),
                    index=False)
    log.info(f"Per-channel dose table: {len(dose_all)} rows "
             f"({len(SWEET_SUBSTANCES)} substances × {len(BAND_ORDER)} bands × "
             f"{len(EEG_CHANNELS)} channels)")

    figs = {}

    def chan_vec(tbl, band, col):
        """Channel vector (EEG_CHANNELS order) of a column for one band."""
        m = tbl[tbl.band == band].set_index('channel')[col]
        return np.array([m.get(ch, np.nan) for ch in EEG_CHANNELS])

    # ── 2. Slope-t topomap grid: rows = bands, cols = substances ──────────────
    val_grid, mask_grid = [], []
    for band in BAND_ORDER:
        vrow, mrow = [], []
        for s in SWEET_SUBSTANCES:
            tbl = dose[s]
            vrow.append(chan_vec(tbl, band, 't_slope'))
            mrow.append(chan_vec(tbl, band, 'p_slope_fdr') < ALPHA)
        val_grid.append(vrow); mask_grid.append(mrow)
    figs['slope_grid'] = viz.plot_stat_topomap_grid(
        val_grid, mask_grid, BAND_ORDER, SWEET_SUBSTANCES,
        os.path.join(fdir, 'topomap_dose_slope_grid.png'),
        suptitle='Dose slope (t of per-subject rel-power slope vs intensity)\n'
                 'circled = FDR-significant channels',
        cbar_label='slope t')

    # ── 3. ANOVA-F topomap grid (omnibus dose effect) ─────────────────────────
    valF, maskF = [], []
    for band in BAND_ORDER:
        vrow, mrow = [], []
        for s in SWEET_SUBSTANCES:
            tbl = dose[s]
            vrow.append(chan_vec(tbl, band, 'F'))
            mrow.append(chan_vec(tbl, band, 'p_anova_fdr') < ALPHA)
        valF.append(vrow); maskF.append(mrow)
    figs['anova_grid'] = viz.plot_stat_topomap_grid(
        valF, maskF, BAND_ORDER, SWEET_SUBSTANCES,
        os.path.join(fdir, 'topomap_dose_anovaF_grid.png'),
        suptitle='Omnibus dose effect (rmANOVA F over intensity)\n'
                 'circled = FDR-significant channels',
        cbar_label='F', cmap='viridis', symmetric=False)

    # ── 4. Channel × band slope-t heatmaps (one per substance) ────────────────
    for s in SWEET_SUBSTANCES:
        tbl = dose[s]
        mat = np.array([[tbl[(tbl.channel == ch) & (tbl.band == b)]['t_slope'].iloc[0]
                         for b in BAND_ORDER] for ch in EEG_CHANNELS])
        star = np.array([[tbl[(tbl.channel == ch) & (tbl.band == b)]
                          ['p_slope_fdr'].iloc[0] < ALPHA
                          for b in BAND_ORDER] for ch in EEG_CHANNELS])
        figs[f'heat_{s}'] = viz.plot_channel_band_heatmap(
            mat, EEG_CHANNELS, BAND_ORDER,
            os.path.join(fdir, f'heatmap_dose_slope_{s.lower()}.png'),
            title=f'{s}: dose slope t per channel × band  (* = FDR p<{ALPHA})',
            cbar_label='slope t', star_mask=star)

    # ── 5. Dose-response curves for the strongest channel × band effects ──────
    ranked = (dose_all.dropna(subset=['p_slope'])
              .sort_values('p_slope').reset_index(drop=True))
    top = ranked.head(8)
    dose_figs = []
    for _, r in top.iterrows():
        ch, band = r['channel'], r['band']
        # plot BOTH substances at this channel × band for context
        summ = pd.concat([channel_dose_summary(bp, s, ch, band)
                          for s in SWEET_SUBSTANCES], ignore_index=True)
        key = f'dose_{ch}_{band}'
        path = os.path.join(fdir, f'{key}.png')
        viz.plot_channel_dose(
            summ, f'{VALUE}_mean', f'{VALUE}_sem', path,
            title=f'{ch} — {band} rel. power vs intensity '
                  f'({r["substance"]}: t={r["t_slope"]:.2f}, '
                  f'p_fdr={r["p_slope_fdr"]:.3f})',
            ylabel=f'{band} rel. power')
        dose_figs.append((ch, band, r['substance'], path))

    # ── Report ────────────────────────────────────────────────────────────────
    S = []
    S.append("# Stage 11 — Per-channel, within-substance dose-response\n")
    S.append(
        "Unlike Stage 05 (sucrose **vs** sucralose, pooled into Frontal/Gustatory "
        "ROIs), this stage analyses **each substance on its own** and resolves the "
        "**individual channels** (all 16 electrodes). For every substance × band × "
        "channel we test how relative band power changes across the three perceived "
        "intensities (~5 / 7.5 / 12 %):\n\n"
        "- **rmANOVA** over intensity — omnibus \"is there any dose effect?\" (F, p).\n"
        "- **Per-subject linear slope** of rel-power vs intensity, one-sample t-test "
        "vs 0 — the *direction* (positive ⇒ power rises with concentration).\n\n"
        "p-values are **FDR-corrected across the 16 channels within each band**. "
        "Channels surviving FDR are circled on the topographies / starred in the "
        "heatmaps.\n")

    S.append("## Dose slope topographies (direction of the effect)\n")
    S.append(report.img(figs['slope_grid'], rdir,
                        'Per-subject slope t-value of relative band power vs '
                        'intensity — rows = bands, cols = substance. Red = power '
                        'increases with concentration, blue = decreases. Circled = '
                        f'FDR p<{ALPHA}.'))

    S.append("## Omnibus dose effect (rmANOVA F)\n")
    S.append(report.img(figs['anova_grid'], rdir,
                        'rmANOVA F-statistic for the intensity main effect per '
                        f'channel (circled = FDR p<{ALPHA}).'))

    S.append("## Channel × band slope heatmaps\n")
    for s in SWEET_SUBSTANCES:
        S.append(report.img(figs[f'heat_{s}'], rdir,
                            f'{s}: dose slope t per channel × band'))

    S.append("## Strongest channel-level dose effects\n")
    S.append("Top channel × band combinations by raw slope p-value (both substances "
             "shown for context).\n")
    cols = ['substance', 'band', 'channel', 'n', 'F', 'p_anova', 'p_anova_fdr',
            'slope', 't_slope', 'p_slope', 'p_slope_fdr']
    S.append(report.df_to_md(top[cols].reset_index(drop=True)))
    for ch, band, sub, path in dose_figs:
        S.append(report.img(path, rdir, f'{ch} {band} ({sub})'))

    # significant-channel summary per substance
    S.append("## FDR-significant dose channels (summary)\n")
    sig = dose_all[dose_all['p_slope_fdr'] < ALPHA]
    if len(sig):
        S.append(report.df_to_md(
            sig[['substance', 'band', 'channel', 'n', 'slope', 't_slope',
                 'p_slope', 'p_slope_fdr']]
            .sort_values(['substance', 'band', 'p_slope_fdr'])
            .reset_index(drop=True)))
    else:
        S.append("*No channel survived FDR correction for a linear dose effect "
                 "in either substance — dose modulation of band power is weak at "
                 "the single-channel level (consistent with the weak ROI-level "
                 "contrasts in Stage 05).*\n")

    report.write(report_path(cfg, '11_channel_dose.md'), S)
    log.info(f"Stage 11 done → {report_path(cfg, '11_channel_dose.md')}")


if __name__ == '__main__':
    main()
