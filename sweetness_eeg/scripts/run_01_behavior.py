"""
Stage 01 — Behavioral (sensory) analysis: liking, sweetness-JAR, sweet aftertaste.

Reads the curated sensory sheet, aggregates by substance × intensity, runs
dose-response and sucrose-vs-sucralose contrasts, and gives special attention to
the *lingering aftertaste* hypothesis for sucralose. Outputs figures + a report.
"""

import _bootstrap  # noqa: F401
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swt.config import load_config, setup_logging, fig_dir, result_path, report_path
from swt import behavior, viz, stats, report
from swt.constants import CONDITIONS, COND_DISPLAY, COND_COLORS, JAR_GROUPS


def jar_distribution_plot(long_df, path):
    """Stacked bar: proportion of Not_enough / Just_right / Too_much per condition."""
    order = [c for c in CONDITIONS if c in long_df['ma_mau'].unique()]
    groups = ['Not_enough', 'Just_right', 'Too_much']
    colors = {'Not_enough': '#4292c6', 'Just_right': '#41ab5d', 'Too_much': '#e6550d'}
    props = []
    for c in order:
        g = long_df[long_df['ma_mau'] == c]['jar_group'].value_counts(normalize=True)
        props.append([g.get(k, 0) for k in groups])
    props = np.array(props)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(order))
    for k, grp in enumerate(groups):
        ax.bar(range(len(order)), props[:, k], bottom=bottom, label=grp.replace('_', ' '),
               color=colors[grp])
        bottom += props[:, k]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([COND_DISPLAY.get(c, c) for c in order], rotation=30, ha='right')
    ax.set_ylabel('Proportion of ratings')
    ax.set_title('Sweetness JAR distribution by condition')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


def aftertaste_vs_sweetness_plot(summary, path):
    """Aftertaste minus sweetness (residual sweet sensation) per condition."""
    order = [c for c in CONDITIONS if c in summary['ma_mau'].values]
    s = summary.set_index('ma_mau')
    resid = [s.loc[c, 'aftertaste_mean'] - s.loc[c, 'sweetness_jar_mean'] for c in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(order)), resid, color=[COND_COLORS.get(c, '#888') for c in order])
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([COND_DISPLAY.get(c, c) for c in order], rotation=30, ha='right')
    ax.set_ylabel('Aftertaste − Sweetness (rating units)')
    ax.set_title('Residual sweet aftertaste (positive = lingers beyond in-mouth sweetness)')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    return path


def main():
    log = setup_logging()
    cfg = load_config()
    fdir = fig_dir(cfg, 'behavior')
    rdir = cfg['paths']['reports']
    log.info("=" * 60); log.info("STAGE 01 — Behavioral analysis"); log.info("=" * 60)

    long_df = behavior.load_behavior_long(cfg['paths']['behavior_xlsx'], logger=log)
    summary = behavior.condition_summary(long_df)
    subj_cond = behavior.subject_condition_means(long_df)

    long_df.to_csv(result_path(cfg, 'behavior', 'behavior_long.csv'), index=False)
    summary.to_csv(result_path(cfg, 'behavior', 'behavior_condition_summary.csv'), index=False)
    subj_cond.to_csv(result_path(cfg, 'behavior', 'behavior_subject_condition.csv'), index=False)

    # ── Figures ──────────────────────────────────────────────────────────────
    figs = {}
    figs['liking_box'] = viz.plot_condition_box(
        long_df, 'liking', os.path.join(fdir, 'liking_box.png'),
        'Hedonic liking by condition', 'Liking (1–9)')
    figs['jar_box'] = viz.plot_condition_box(
        long_df, 'sweetness_jar', os.path.join(fdir, 'sweetness_jar_box.png'),
        'Sweetness JAR by condition (3 = just right)', 'Sweetness JAR (1–5)')
    figs['after_box'] = viz.plot_condition_box(
        long_df, 'aftertaste', os.path.join(fdir, 'aftertaste_box.png'),
        'Sweet aftertaste by condition', 'Aftertaste (1–5)')
    figs['dr_liking'] = viz.plot_dose_response(
        summary, 'liking_mean', 'liking_sem', os.path.join(fdir, 'dose_liking.png'),
        'Dose-response — liking', 'Liking (1–9)')
    figs['dr_jar'] = viz.plot_dose_response(
        summary, 'sweetness_jar_mean', 'sweetness_jar_sem',
        os.path.join(fdir, 'dose_sweetness.png'), 'Dose-response — sweetness JAR',
        'Sweetness JAR (1–5)')
    figs['dr_after'] = viz.plot_dose_response(
        summary, 'aftertaste_mean', 'aftertaste_sem',
        os.path.join(fdir, 'dose_aftertaste.png'), 'Dose-response — sweet aftertaste',
        'Aftertaste (1–5)')
    figs['jar_dist'] = jar_distribution_plot(
        long_df, os.path.join(fdir, 'jar_distribution.png'))
    figs['after_resid'] = aftertaste_vs_sweetness_plot(
        summary, os.path.join(fdir, 'aftertaste_residual.png'))

    # ── Statistics ───────────────────────────────────────────────────────────
    contrasts = {}
    for dv in ['liking', 'sweetness_jar', 'aftertaste']:
        contrasts[dv] = stats.paired_substance_contrasts(subj_cond, dv)
        contrasts[dv].to_csv(
            result_path(cfg, 'behavior', f'contrast_{dv}.csv'), index=False)
    anova = {}
    for dv in ['liking', 'sweetness_jar', 'aftertaste']:
        a = stats.two_way_rm_anova(subj_cond, dv)
        if a is not None:
            anova[dv] = a
            a.to_csv(result_path(cfg, 'behavior', f'anova_{dv}.csv'), index=False)

    # ── Report ───────────────────────────────────────────────────────────────
    S = []
    S.append("# Stage 01 — Behavioral (Sensory) Analysis\n")
    S.append("Liking (9-point hedonic), sweetness Just-About-Right (JAR, 1–5, 3 = just "
             "right) and sweet aftertaste (1–5) for sucrose (S1) and sucralose (S2) at "
             "three iso-sweet intensities (~5 / 7.5 / 12 % sucrose), plus water.\n")
    S.append(f"- Participants: **{long_df['subject'].nunique()}**, "
             f"sample ratings: **{len(long_df)}**\n")

    S.append("## Condition summary (mean ± SEM)\n")
    disp = summary.copy()
    disp['condition'] = disp['ma_mau'].map(COND_DISPLAY)
    cols = ['condition', 'substance', 'intensity', 'n', 'liking_mean', 'liking_sem',
            'sweetness_jar_mean', 'sweetness_jar_sem', 'aftertaste_mean', 'aftertaste_sem']
    S.append(report.df_to_md(disp[cols].sort_values(['substance', 'intensity'])))

    S.append("## Liking, sweetness & aftertaste by condition\n")
    for k, cap in [('liking_box', 'Hedonic liking by condition'),
                   ('jar_box', 'Sweetness JAR by condition'),
                   ('after_box', 'Sweet aftertaste by condition'),
                   ('jar_dist', 'JAR category distribution')]:
        S.append(report.img(figs[k], rdir, cap))

    S.append("## Dose-response (sucrose vs sucralose)\n")
    for k, cap in [('dr_liking', 'Liking vs intensity'),
                   ('dr_jar', 'Sweetness JAR vs intensity'),
                   ('dr_after', 'Aftertaste vs intensity')]:
        S.append(report.img(figs[k], rdir, cap))

    S.append("## Lingering aftertaste (key sucralose hypothesis)\n")
    S.append("The residual = *aftertaste − in-mouth sweetness*. A positive value means "
             "sweetness persists after expectoration — the lingering aftertaste sucralose "
             "is known for.\n")
    S.append(report.img(figs['after_resid'], rdir, 'Residual sweet aftertaste'))

    S.append("## Sucrose vs sucralose contrasts (paired t-test, FDR-corrected)\n")
    for dv in ['liking', 'sweetness_jar', 'aftertaste']:
        S.append(f"**{dv}**\n")
        S.append(report.df_to_md(contrasts[dv]))

    if anova:
        S.append("## Two-way repeated-measures ANOVA (substance × intensity)\n")
        for dv, a in anova.items():
            S.append(f"**{dv}**\n")
            S.append(report.df_to_md(a))

    report.write(report_path(cfg, '01_behavior.md'), S)
    log.info(f"Stage 01 done → {report_path(cfg, '01_behavior.md')}")


if __name__ == '__main__':
    main()
