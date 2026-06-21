"""
Stage 02 — Preprocess & epoch all subjects.

Loads every datamoi recording, applies notch/band-pass/average-reference (+ optional
ICA), slices the 14 ten-second tasting windows per subject and caches MNE epochs to
``results/epochs/``. Writes a QC report (epochs kept per subject / condition).
"""

import _bootstrap  # noqa: F401

import pandas as pd

from swt.config import load_config, setup_logging, result_path, report_path
from swt import epoching, report
from swt.constants import CONDITIONS, COND_DISPLAY


def main():
    log = setup_logging()
    cfg = load_config()
    log.info("=" * 60); log.info("STAGE 02 — Preprocess & epoch"); log.info("=" * 60)

    all_epochs, all_meta = epoching.build_all_epochs(cfg, log, cache=True)
    all_meta.to_csv(result_path(cfg, 'epochs', 'all_metadata.csv'), index=False)

    # QC table: epochs per subject × condition
    pivot = (all_meta.assign(_c=1)
             .pivot_table(index='subject', columns='ma_mau', values='_c',
                          aggfunc='sum', fill_value=0))
    pivot = pivot[[c for c in CONDITIONS if c in pivot.columns]]
    pivot['total'] = pivot.sum(axis=1)
    pivot.to_csv(result_path(cfg, 'epochs', 'qc_epochs_per_condition.csv'))

    prep = cfg['preprocessing']
    S = []
    S.append("# Stage 02 — Preprocessing & Epoching\n")
    S.append("Pipeline: pick 16 EEG channels (T3/T4/T5/T6 → T7/T8/P7/P8) → average "
             f"reference → notch {prep['notch_freq']} Hz → band-pass "
             f"{prep['l_freq']}–{prep['h_freq']} Hz"
             f"{' → ICA' if prep['ica'].get('enabled') else ''}. "
             f"Each trial = 10 s tasting window (1000 samples @ 100 Hz). "
             f"Peak-to-peak rejection: {cfg['epoching'].get('reject_uv')} µV.\n")
    S.append(f"- Subjects with usable epochs: **{all_meta['subject'].nunique()}**\n"
             f"- Total epochs: **{len(all_meta)}** "
             f"(max possible = {all_meta['subject'].nunique() * 14})\n")
    S.append("## Epochs kept per subject × condition\n")
    disp = pivot.reset_index().rename(columns={c: COND_DISPLAY.get(c, c)
                                               for c in pivot.columns})
    S.append(report.df_to_md(disp, floatfmt='.0f'))
    S.append("## Epoch counts per condition (pooled)\n")
    counts = (all_meta['ma_mau'].value_counts()
              .reindex([c for c in CONDITIONS if c in all_meta['ma_mau'].unique()])
              .rename_axis('condition').reset_index(name='n_epochs'))
    counts['condition'] = counts['condition'].map(COND_DISPLAY)
    S.append(report.df_to_md(counts, floatfmt='.0f'))

    report.write(report_path(cfg, '02_preprocess.md'), S)
    log.info(f"Stage 02 done → {report_path(cfg, '02_preprocess.md')}")


if __name__ == '__main__':
    main()
