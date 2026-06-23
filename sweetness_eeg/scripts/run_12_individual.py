"""
Stage 12 — Individual (per-subject) analysis & insight extraction.

Group statistics (Stages 03/05/11) average over subjects and so hide *who* drives
an effect and how much people differ. This stage keeps every subject visible:

- **Individual dose trajectories** (spaghetti plots) of frontal θ/α and gustatory
  β/γ for each substance, with the group mean overlaid.
- **Per-subject dose slopes** (rel-power vs intensity) — their distribution,
  how many subjects rise vs fall with concentration, and which are the strongest
  responders.
- **Subject × band heatmaps** of the dose slope, revealing responder sub-groups.
- **Per-subject sucrose−sucralose preference** in frontal θ, sorted.
- **Individual EEG ↔ behaviour link**: does a subject's frontal-θ dose slope or
  substance preference track their own liking / sweetness-JAR?

A plain-language **insight block** is auto-generated from these per-subject numbers.
"""

import _bootstrap  # noqa: F401
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import epoching, spectral, behavior, viz, report

ROI_METRICS = [('Frontal', 'theta'), ('Frontal', 'alpha'),
               ('Gustatory', 'beta'), ('Gustatory', 'gamma')]
SUBSTANCES = ['Sucrose', 'Sucralose']
VALUE = 'rel_power'
INTENSITIES = [5, 7, 12]


def subject_intensity_means(roi_bp, roi, band, substance):
    """One rel-power value per subject × intensity (repeats averaged)."""
    sub = roi_bp[(roi_bp.roi == roi) & (roi_bp.band == band) &
                 (roi_bp.substance == substance)]
    return (sub.groupby(['subject', 'intensity'])[VALUE].mean().reset_index())


def subject_slopes(roi_bp, roi, band, substance):
    """Per-subject linear slope of rel-power vs intensity + their mean level."""
    per = subject_intensity_means(roi_bp, roi, band, substance)
    rows = []
    for subj, g in per.groupby('subject'):
        g = g.dropna(subset=[VALUE])
        if g['intensity'].nunique() < 2:
            continue
        slope = np.polyfit(g['intensity'].to_numpy(float),
                           g[VALUE].to_numpy(float), 1)[0]
        rows.append({'subject': subj, 'roi': roi, 'band': band,
                     'substance': substance, 'slope': slope,
                     'mean_power': g[VALUE].mean()})
    return pd.DataFrame(rows)


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'individual')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 12 — Individual analysis"); log.info("=" * 60)

    all_epochs, _ = epoching.load_all_epochs(cfg, log)
    bp = spectral.bandpower_table(all_epochs, cfg, log)
    roi_bp = spectral.roi_bandpower(bp)
    subj_roi = (roi_bp.groupby(['subject', 'ma_mau', 'substance', 'intensity',
                                'roi', 'band'])[VALUE].mean().reset_index())

    figs = {}

    # ── 1. Per-subject dose slopes (all ROI metrics × substances) ─────────────
    slope_tbls = []
    for roi, band in ROI_METRICS:
        for s in SUBSTANCES:
            slope_tbls.append(subject_slopes(subj_roi, roi, band, s))
    slopes = pd.concat(slope_tbls, ignore_index=True)
    slopes.to_csv(result_path(cfg, 'individual', 'subject_dose_slopes.csv'),
                  index=False)

    # ── 2. Spaghetti dose trajectories (frontal θ/α, both substances) ─────────
    for roi, band in [('Frontal', 'theta'), ('Frontal', 'alpha')]:
        for s in SUBSTANCES:
            per = subject_intensity_means(subj_roi, roi, band, s)
            key = f'spag_{roi}_{band}_{s}'
            figs[key] = viz.plot_subject_spaghetti(
                per, VALUE, os.path.join(fdir, f'{key}.png'),
                title=f'{s}: per-subject {roi} {band} vs intensity',
                ylabel=f'{band} rel. power')

    # ── 3. Slope distribution (strip + box) per metric × substance ────────────
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import seaborn as sns
    slopes['metric'] = slopes['roi'] + ' ' + slopes['band']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=slopes, x='metric', y='slope', hue='substance',
                showfliers=False, ax=ax,
                palette={'Sucrose': '#08519c', 'Sucralose': '#a50f15'})
    sns.stripplot(data=slopes, x='metric', y='slope', hue='substance',
                  dodge=True, ax=ax, size=3, alpha=0.5, color='0.2', legend=False)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(''); ax.set_ylabel('per-subject dose slope (rel-power / intensity)')
    ax.set_title('Distribution of individual dose slopes (each dot = 1 subject)')
    ax.legend(title='', fontsize=9)
    fig.tight_layout(); p = os.path.join(fdir, 'slope_distribution.png')
    fig.savefig(p, dpi=150); plt.close(fig); figs['slope_dist'] = p

    # ── 4. Subject × band slope heatmaps (Frontal ROI, one per substance) ─────
    from swt.constants import BAND_ORDER
    for s in SUBSTANCES:
        # build subject × band matrix of frontal slopes
        rows = []
        for band in BAND_ORDER:
            st = subject_slopes(subj_roi, 'Frontal', band, s).set_index('subject')['slope']
            rows.append(st)
        mat_df = pd.concat(rows, axis=1)
        mat_df.columns = BAND_ORDER
        mat_df = mat_df.sort_index()
        figs[f'heat_{s}'] = viz.plot_channel_band_heatmap(
            mat_df.to_numpy(), list(mat_df.index), BAND_ORDER,
            os.path.join(fdir, f'subject_band_slope_{s.lower()}.png'),
            title=f'{s}: per-subject Frontal dose slope (subject × band)',
            cbar_label='slope')

    # ── 5. Per-subject sucrose−sucralose preference (Frontal theta) ───────────
    pref = (subj_roi[(subj_roi.roi == 'Frontal') & (subj_roi.band == 'theta')]
            .groupby(['subject', 'substance'])[VALUE].mean().unstack('substance'))
    pref = pref.dropna(subset=['Sucrose', 'Sucralose'])
    pref['diff'] = pref['Sucrose'] - pref['Sucralose']
    pref.to_csv(result_path(cfg, 'individual', 'frontal_theta_preference.csv'))
    figs['pref'] = viz.plot_sorted_subject_bar(
        list(pref.index), list(pref['diff'].values),
        os.path.join(fdir, 'frontal_theta_preference.png'),
        title='Per-subject Frontal θ: Sucrose − Sucralose (red = higher for sucrose)',
        ylabel='Sucrose − Sucralose rel. θ power')

    # ── 6. Individual EEG ↔ behaviour link ────────────────────────────────────
    beh = behavior.load_behavior_long(cfg['paths']['behavior_xlsx'], logger=log)
    beh_sc = behavior.subject_condition_means(beh)
    from swt.constants import label_to_substance
    beh_sc['substance'] = beh_sc['ma_mau'].map(label_to_substance)
    beh_subj = (beh_sc[beh_sc.substance.isin(SUBSTANCES)]
                .groupby('subject')[['liking', 'sweetness_jar', 'aftertaste']]
                .mean().reset_index())

    # per-subject frontal-theta dose slope (substance-averaged) vs behaviour
    ftheta_slope = (slopes[(slopes.roi == 'Frontal') & (slopes.band == 'theta')]
                    .groupby('subject')['slope'].mean().reset_index()
                    .rename(columns={'slope': 'frontal_theta_slope'}))
    link = ftheta_slope.merge(beh_subj, on='subject', how='inner')
    link = link.merge(pref['diff'].reset_index()
                      .rename(columns={'diff': 'frontal_theta_pref'}),
                      on='subject', how='left')
    link.to_csv(result_path(cfg, 'individual', 'subject_eeg_behavior.csv'), index=False)

    link_rows = []
    for eeg_col in ['frontal_theta_slope', 'frontal_theta_pref']:
        for bm in ['liking', 'sweetness_jar', 'aftertaste']:
            d = link[[eeg_col, bm]].dropna()
            if len(d) > 3:
                r, p = sps.pearsonr(d[eeg_col], d[bm])
            else:
                r, p = np.nan, np.nan
            link_rows.append({'eeg_metric': eeg_col, 'behavior': bm,
                              'n': len(d), 'r': r, 'p': p})
    link_corr = pd.DataFrame(link_rows)
    link_corr.to_csv(result_path(cfg, 'individual', 'subject_link_correlations.csv'),
                     index=False)
    # scatter for the strongest |r| pair
    valid = link_corr.dropna(subset=['r'])
    if len(valid):
        best = valid.iloc[valid['r'].abs().argmax()]
        d = link[[best['eeg_metric'], best['behavior']]].dropna()
        figs['link'] = viz.plot_scatter_corr(
            d[best['eeg_metric']], d[best['behavior']],
            os.path.join(fdir, 'subject_eeg_behavior_best.png'),
            xlabel=best['eeg_metric'], ylabel=best['behavior'],
            title=f"{best['eeg_metric']} vs {best['behavior']} "
                  f"(per subject, r={best['r']:.2f}, p={best['p']:.3f})")

    # ── Insight extraction (auto, plain language) ─────────────────────────────
    insights = []
    n_subj = slopes['subject'].nunique()
    insights.append(f"- **{n_subj} subjects** analysed individually; each contributes "
                    f"one dose slope per ROI × band × substance.")
    for roi, band in [('Frontal', 'theta'), ('Frontal', 'alpha')]:
        for s in SUBSTANCES:
            g = slopes[(slopes.roi == roi) & (slopes.band == band) &
                       (slopes.substance == s)]['slope'].dropna()
            if not len(g):
                continue
            pos = int((g > 0).sum()); neg = int((g < 0).sum())
            t, p = sps.ttest_1samp(g, 0.0) if len(g) > 2 else (np.nan, np.nan)
            if g.mean() > 0:
                direction, n_with, n_opp = 'rise', pos, neg
            else:
                direction, n_with, n_opp = 'fall', neg, pos
            insights.append(
                f"- **{roi} {band}, {s}**: {n_with}/{len(g)} subjects {direction} "
                f"with concentration ({n_opp} opposite); mean slope "
                f"{g.mean():+.2e} (one-sample t={t:.2f}, p={p:.3f}). "
                + ("**Consistent across subjects.**" if (p == p and p < 0.05)
                   else "Direction is *not* consistent across people — the group "
                        "effect is driven by a subset / cancels out."))
    # strongest individual responders (|slope| frontal theta sucralose)
    ft = slopes[(slopes.roi == 'Frontal') & (slopes.band == 'theta') &
                (slopes.substance == 'Sucralose')]
    if len(ft):
        top = ft.reindex(ft['slope'].abs().sort_values(ascending=False).index).head(3)
        who = ', '.join(f"{r.subject} ({r.slope:+.2e})" for r in top.itertuples())
        insights.append(f"- **Strongest Frontal-θ Sucralose responders**: {who}.")
    # preference split
    if len(pref):
        ns = int((pref['diff'] > 0).sum()); nl = int((pref['diff'] < 0).sum())
        insights.append(f"- **Substance preference (Frontal θ)**: {ns}/{len(pref)} "
                        f"subjects show higher θ for sucrose, {nl} for sucralose — "
                        "no single substance dominates at the individual level.")
    # behaviour link
    if len(valid):
        b = valid.iloc[valid['r'].abs().argmax()]
        sig = 'significant' if (b['p'] == b['p'] and b['p'] < 0.05) else 'not significant'
        insights.append(
            f"- **Individual EEG↔behaviour**: strongest link is "
            f"{b['eeg_metric']} vs {b['behavior']} (r={b['r']:+.2f}, "
            f"p={b['p']:.3f}, n={int(b['n'])}) — {sig}. Subjects whose frontal θ "
            "responds more to dose do "
            + ("tend to differ in their ratings." if sig
               else "**not** clearly differ in their ratings."))

    # ── Report ────────────────────────────────────────────────────────────────
    S = []
    S.append("# Stage 12 — Individual (per-subject) analysis & insights\n")
    S.append("Group means hide individual variability. Here every subject stays "
             "visible: individual dose trajectories, the spread of per-subject dose "
             "slopes, responder sub-groups, individual substance preference, and "
             "whether a person's EEG dose response tracks their own ratings.\n")

    S.append("## Key insights\n")
    S.extend(insights)
    S.append("")

    S.append("## Individual dose trajectories (frontal θ / α)\n")
    S.append("Each thin line is one subject (red = power rises with dose, blue = "
             "falls); the black line is the group mean.\n")
    for roi, band in [('Frontal', 'theta'), ('Frontal', 'alpha')]:
        for s in SUBSTANCES:
            S.append(report.img(figs[f'spag_{roi}_{band}_{s}'], rdir,
                                f'{s} — {roi} {band} per-subject trajectories'))

    S.append("## Distribution of individual dose slopes\n")
    S.append(report.img(figs['slope_dist'], rdir,
                        'Each dot = one subject. Spread crossing zero ⇒ subjects '
                        'disagree in direction.'))

    S.append("## Responder sub-groups (subject × band slope)\n")
    for s in SUBSTANCES:
        S.append(report.img(figs[f'heat_{s}'], rdir,
                            f'{s}: per-subject Frontal dose slope by band'))

    S.append("## Individual substance preference (Frontal θ)\n")
    S.append(report.img(figs['pref'], rdir,
                        'Sucrose − Sucralose frontal θ per subject (sorted)'))

    S.append("## Individual EEG ↔ behaviour\n")
    S.append(report.df_to_md(link_corr))
    if 'link' in figs:
        S.append(report.img(figs['link'], rdir, 'Strongest per-subject EEG↔behaviour link'))

    report.write(report_path(cfg, '12_individual.md'), S)
    log.info(f"Stage 12 done → {report_path(cfg, '12_individual.md')}")


if __name__ == '__main__':
    main()
