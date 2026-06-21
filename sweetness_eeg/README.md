# sweetness_eeg — Sucrose vs Sucralose EEG analysis

Self-contained pipeline for the `data/datamoi/` recordings (23 subjects, 16-ch EEG,
100 Hz, 10 s gustatory tasting trials). Reproduces the reference-paper analyses
(PSD of δ/θ/α/β/γ over frontal & gustatory regions, time + frequency domains,
topographies, sucrose-vs-sucralose contrasts, frontal θ/α modulation, prolonged
aftertaste activation) and adds ML/DL decoding, connectivity and EEG↔behaviour
correlation.

**Does not modify** the legacy `pipeline/` or `src/` code.

## Layout

```
sweetness_eeg/
  config.yaml          # paths, preprocessing, spectral, stats, ml, dl settings
  swt/                 # package: constants, loader, behavior, preprocess, epoching,
                       #          spectral, erp, connectivity, stats, ml, dl, viz, report
  scripts/             # run_01 … run_10 + run_all
  figures/  results/  reports/    # outputs (figures = PNG, reports = .md)
```

## Run

```bash
# whole pipeline (stages 01–10) + assembled REPORT_FULL.md
.venv/bin/python sweetness_eeg/scripts/run_all.py

# skip deep learning (heavy):
.venv/bin/python sweetness_eeg/scripts/run_all.py --skip 09

# a single stage:
.venv/bin/python sweetness_eeg/scripts/run_03_spectral_psd.py
```

Stage 02 caches epochs to `results/epochs/*.fif`; later stages reuse the cache.
Re-run stage 02 after changing preprocessing/epoching settings in `config.yaml`.

## Stages

| # | Script | Output report |
|---|--------|---------------|
| 01 | behavior | `reports/01_behavior.md` |
| 02 | preprocess + epoch | `reports/02_preprocess.md` |
| 03 | spectral / PSD (core) | `reports/03_spectral_psd.md` |
| 04 | temporal / TFR | `reports/04_temporal_tfr.md` |
| 05 | statistics | `reports/05_stats.md` |
| 06 | onset-locked ERP | `reports/06_erp.md` |
| 07 | connectivity | `reports/07_connectivity.md` |
| 08 | ML (LOSO) | `reports/08_ml.md` |
| 09 | DL (EEGNet/ShallowConvNet) | `reports/09_dl.md` |
| 10 | EEG ↔ behavior | `reports/10_eeg_behavior_corr.md` |
| — | `run_all.py` | `reports/REPORT_FULL.md` |
```
