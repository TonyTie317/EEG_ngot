# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG (electroencephalography) data analysis pipeline for **gustatory (taste) perception research** — specifically analyzing neural responses to sweet taste stimuli at varying concentrations. The project processes data from 28 subjects (P001–P030, P012 excluded) using a 14-channel Emotiv-style EEG headset. Each subject tasted 6 samples (5 sucrose concentrations + 1 water baseline), each repeated 5 times, with each trial lasting ~10s at 100 Hz sampling rate (raw data contains ~1100 rows = 11s per trial).

The research goal is gERP (Gustatory Event-Related Potentials) analysis: identifying when (temporal dynamics) and where (brain regions) the brain processes taste information, linking ERP components (P1 ~100ms, P2 ~350–450ms, N400) to subjective JAR (Just-About-Right) ratings.

## Codebase Organization — Two Codebases

The repository contains **two parallel codebases**:

### Active Pipeline: `pipeline/` + `configs/config.yaml`

This is the currently maintained pipeline, with `run_pipeline.py` as the entry point. All modules live in the `pipeline/` package and import domain constants from `pipeline/constants.py`. Config is loaded via `pipeline/config.py`.

```
configs/config.yaml → run_pipeline.py → pipeline/loader.py → pipeline/preprocess.py → pipeline/epoching.py → pipeline/erp_analysis.py → pipeline/stats.py → pipeline/ml.py → pipeline/dl.py → pipeline/viz.py
```

### Stage Flow

1. **Data Loading** (`pipeline/loader.py`) — Reads CSVs from `datadone/`, converts µV→V, creates MNE `RawArray` with 10-20 montage. Trial boundaries detected from `ma_mau` column. Returns structured dict with `raw`, `trials`, `trial_table`.
2. **Preprocessing** (`pipeline/preprocess.py`) — Notch filter (49 Hz), bandpass filter (0.1-45 Hz), average reference, optional ICA with auto EOG/ECG detection.
3. **Epoching** (`pipeline/epoching.py`) — Creates fixed-length epochs (configurable tmin/tmax/baseline), applies baseline correction, rejects noisy epochs. Saves `.fif` + `.npy` + `trial_info.csv` per subject. Supports Woody realignment via `realign_offsets.csv`.
4. **ERP Analysis** (`pipeline/erp_analysis.py`) — Grand-average ERP, peak detection (P1, N1, P2, N400), per-trial component measures, concentration/JAR comparisons, difference waves.
5. **Statistics** (`pipeline/stats.py`) — Repeated-measures ANOVA (pingouin or scipy fallback), pairwise t-tests, FDR correction, permutation clustering.
6. **ML Classification** (`pipeline/ml.py`) — LOSO cross-validation. Models: LogisticRegression, SVM, RandomForest, XGBoost (optional). Tasks: JAR 3-class, concentration binary.
7. **Deep Learning** (`pipeline/dl.py`) — EEGNet, ShallowConvNet, DeepConvNet via PyTorch with LOSO CV and early stopping. Guarded by `TORCH_AVAILABLE`.
8. **Visualization** (`pipeline/viz.py`) — Grand-average ERP waveforms with component highlights, topomaps, dose-response curves, confusion matrices, overfitting curves.

### Legacy Codebase: `src/`

Original modular pipeline (may be stale). Modules mirror the same stages but are not actively developed:
- `src/io.py`, `src/preprocess.py`, `src/epoching.py`, `src/repeat_split.py`
- `src/features.py`, `src/features_advanced/` — basic and advanced feature extraction (ERP, bandpower, PSD, Hjorth, entropy, STFT, TFR, wavelets, connectivity, spectral features)
- `src/stats.py`, `src/ml.py`, `src/dl.py`, `src/visualization.py`
- Uses `src/utils.py` for `load_config()`, `setup_logging()`, `get_subject_list()`
- `ml_per_feature_type.py`, `ml_kfold_per_feature_type.py` — per-feature-type ML analysis

### Domain Constants: `pipeline/constants.py`

Ground-truth constants shared across the pipeline (no config dependency):
- EEG_CHANNELS (16+2), subject list (28 subjects, P012 excluded), concentration codes (6 levels)
- JAR group mapping, ERP component windows + ROIs, frequency bands (delta/theta/alpha/beta/gamma)
- `map_jar_to_group()`, `JAR_NUMERIC`, `CONCENTRATION_LABELS`

## Data Format

**Location**: `datadone/` — flat directory with 28 BIDS-named CSVs (`sub-PXXX_ses-S001_task-Default_run-001_eeg.csv`).

**CSV columns**: `time`, `timestamp`, `frame_idx`, 14 EEG channels (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4), 2 ECG channels (ECG1, ECG2), `ma_mau` (event marker — non-zero value = trial, 6 concentration codes), `NT` (subject ID), `repeat` (repeat number, 1-5), `JAR` (Just-About-Right rating 1-5). Values in microvolts. Sampling rate: 100 Hz, ~1100 rows/trial (~11s).

**CSV path construction** (`pipeline/loader.py:get_csv_path()`): `{config.paths.raw_data}/{config.paths.csv_pattern.format(subject=subject_id)}` — flat lookup in `datadone/`, NOT a nested subject directory.

## Setup

### Python Environment

A virtual environment exists at `.venv/` (Python 3.10.12). No `requirements.txt` or `setup.py` exists. Dependencies include:

- **MNE-Python** — core EEG processing (Raw/Epochs, ICA, PSD, TFR, montages)
- **scipy** — signal processing (filters, Hilbert, coherence, Welch)
- **scikit-learn** — ML models, cross-validation, metrics
- **PyTorch** — deep learning models (EEGNet, ShallowConvNet, DeepConvNet)
- **xgboost** — gradient boosting classifier (optional, handled via try/except ImportError)
- **pingouin** — rmANOVA/ICC (optional)
- **antropy** — entropy measures (optional)
- **pywt** (PyWavelets) — DWT/CWT (optional)
- **statsmodels** — mixed effects models, FDR correction
- **matplotlib + seaborn** — visualization
- **PyYAML** — config loading

## Running the Pipeline

Use `.venv/bin/python` to run scripts (all scripts add the project root to `sys.path`).

### Full pipeline (all 8 stages):
```bash
.venv/bin/python run_pipeline.py
```

### Run stages selectively:
```bash
# Load → Preprocess → Epoch (re-run after changing tmin/tmax/baseline/reject in config)
.venv/bin/python run_epoching_only.py

# ERP Analysis from saved epochs (applies Woody realignment if offsets exist)
.venv/bin/python run_erp_analysis_only.py

# ERP insight analysis with detailed component plots and text report
.venv/bin/python run_erp_insight.py

# ML classification with rich feature engineering (ERP+bandpower+Hjorth+DWT+connectivity)
.venv/bin/python run_ml_jar3.py
```

### Test / debug scripts:
```bash
# Visual ERP inspection — load saved epochs, plot grand-average waveforms
.venv/bin/python test_erp_visual_inspect.py
.venv/bin/python test_erp_visual_inspect.py --subjects P001 P002 P003
.venv/bin/python test_erp_visual_inspect.py --subjects P001 --channels Fz Cz Pz

# Onset re-alignment — detect true stimulus onset (3 strategies: RMS, Woody, Combined)
.venv/bin/python test_onset_realign.py
.venv/bin/python test_onset_realign.py --subjects P001 P002   # specific subjects
.venv/bin/python test_onset_realign.py --strategy woody       # Woody filter only
.venv/bin/python test_onset_realign.py --apply                # save offsets for pipeline use
```

### Legacy `src/` usage:
```python
from src.utils import load_config, setup_logging, get_subject_list
config = load_config("configs/config.yaml")
logger = setup_logging(config)
subjects = get_subject_list(config)
```

### Config (`configs/config.yaml`)

Already exists with sections: `paths`, `preprocessing`, `epoching`, `erp_analysis`, `stats`, `ml`, `dl`, `visualization`, `logging`. Key settings:
- 100 Hz sampling, 16 EEG channels, 10-20 montage, average reference
- Notch 49 Hz (Vietnam power), bandpass 0.1-45 Hz
- ICA with Picard (15 components, auto EOG detection via Fp1/Fp2)
- Epoching: -0.5s to +3.0s, baseline [-0.5, -0.3]s

## Experimental Design Details

- 28 subjects, each with 5 trials per session
- 6 sample types: 5 sucrose concentrations + water baseline
- Each sample presented 5 times (repetitions)
- 1 sample type is repeated twice within the session (for test-retest reliability)
- JAR ratings: "Không đủ" (not enough), "Vừa phải" (just right), "Quá nhiều" (too much)
- Validation logic in `utils.py:validate_trials()` enforces exactly 4 unique sample types with 1 repeated

## Code Conventions

- NumPy-style docstrings on all functions
- All processing functions take a `config` dict and `logger` as parameters
- Optional dependencies (xgboost, pingouin, antropy, pywt) are handled with try/except ImportError
- Feature output uses both wide format (`features_all_trials.csv`) and long format (`features_all_trials_long.csv`)
- Subject P012 is missing from the dataset (28 subjects total, P001–P011, P013–P021, P023–P030)
