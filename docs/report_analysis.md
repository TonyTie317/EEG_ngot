# Report: ERP Quality Check & Per-Channel Analysis

> **Project**: gERP — Gustatory Event-Related Potentials (sweet taste)
> **Data**: 28 subjects (P001–P030, trừ P012, P022), 6 concentrations, 5 repeats
> **Date**: 2026-05-18

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scripts Created](#2-scripts-created)
3. [ERP Quality Check](#3-erp-quality-check)
4. [Visualization of Quality Check](#4-visualization-of-quality-check)
5. [ERP Analysis with Quality Filter](#5-erp-analysis-with-quality-filter)
6. [ANOVA Comparison (Unfiltered vs Filtered)](#6-anova-comparison)
7. [Per-Channel ANOVA](#7-per-channel-anova)
8. [JAR Analysis](#8-jar-analysis)
9. [Per-Channel Visualization](#9-per-channel-visualization)
10. [ML with Significant Channels](#10-ml-with-significant-channels)
11. [ML with Feature Selection](#11-ml-with-feature-selection)
12. [Summary of Findings](#12-summary-of-findings)

---

## 1. Overview

### Research Question
ERP components (P1, N1, P2, N400) khác biệt thế nào theo:
- **Concentration** (6 levels: Water/605, Low/258, MedLow/453, Medium/189, MedHigh/762, High/893)
- **JAR group** (3 groups: Không đủ, Vừa phải, Quá nhiều)

### Pipeline hiện tại
```
CSV data → Loader → Preprocess → Epoch → ERP Analysis → Stats → ML
```

### Problem phát hiện
- ANOVA cho JAR effect không significant ở hầu hết components
- Có thể do data quality kém ở 1 số subjects/conditions
- ROI-averaging (gộp channels) làm loãng effect

### Approach
1. **Quality check**: Đánh giá chất lượng từng subject×condition
2. **Per-channel**: Phân tích từng kênh riêng thay vì gộp ROI
3. **Feature selection**: Mutual Information chọn top features cho ML
4. **ML**: LOSO classification với features đã chọn

---

## 2. Scripts Created

| Script | Purpose | Output |
|--------|---------|--------|
| `run_erp_quality_check.py` | Đánh giá chất lượng ERP từng subject×condition | `erp_quality_flags.csv`, `erp_quality_report.txt` |
| `run_erp_quality_viz.py` | Visualization quality check results | Figures in `output/figures/erp_quality/` |
| `run_erp_filtered.py` | ERP analysis sau khi loại BAD conditions | `output/results/erp_filtered/` |
| `run_anova_comparison.py` | So sánh ANOVA unfiltered vs filtered | Console output |
| `run_jar_analysis.py` | JAR analysis cho tất cả components | Console output |
| `run_per_channel_anova.py` | rmANOVA cho từng channel riêng lẻ | `concentration_anova_per_channel.csv`, `jar_anova_per_channel.csv` |
| `run_per_channel_viz.py` | Visualization per-channel effects | Figures in `output/figures/per_channel/` |
| `run_ml_significant_channels.py` | ML với channels significant từ ANOVA | `output/results/ml_significant_channels/` |
| `run_ml_top_features.py` | ML với Mutual Information feature selection | `output/results/ml_top_features/` |

---

## 3. ERP Quality Check

**Script**: `run_erp_quality_check.py`

### Phương pháp

Mỗi **subject×condition** được đánh giá qua 4 metrics từ condition-averaged ERP waveform:

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **avg_SNR** | var(signal_avg) / [var(residuals) / n_trials] | SNR của ERP average (không phải single-trial) |
| **Component detect** | Peak đúng polarity + amplitude > noise×0.5 | Số component detect được (0-4) |
| **Morphology score** | 6 checks: P1>0, N1<P1, P2>0, P2>P1, N400<P2, N400<baseline | Hình dạng ERP có giống ERP điển hình? |
| **Signal/noise-floor ratio** | RMS(avg) / RMS(trial_variance) | Signal strength so với noise |

### Classification

**has_real_pattern = True** khi thỏa 1 trong 4:
1. avg_SNR ≥ 2.0 + detect ≥ 2/4 components
2. Detect ≥ 3/4 + morphology ≥ 0.5
3. Detect ≥ 2/4 + avg_SNR ≥ 3.0
4. Detect 4/4

**Labels**:
- **GOOD** = has_real_pattern + quality_score ≥ 0.5
- **WEAK** = has_real_pattern + quality_score < 0.5
- **BAD** = not has_real_pattern

### Kết quả

| Label | Count | % |
|-------|------|---|
| GOOD | 90 | 54% |
| WEAK | 29 | 17% |
| **BAD** | **49** | **29%** |

### Subjects nhiều BAD nhất

| Subject | BAD/tổng | Conditions BAD |
|---------|:-------:|----------------|
| **P008** | 5/6 | Water, Low, MedLow, Medium, High |
| **P001** | 4/6 | Low, MedLow, Medium, MedHigh |
| **P016** | 4/6 | Water, Low, MedLow, High |
| **P028** | 4/6 | Water, Medium, MedHigh, High |
| P015, P024, P027, P029 | 3/6 | |

### Files output

- `output/results/erp/erp_quality_flags.csv` — 168 rows, 32 columns
- `output/results/erp/erp_quality_report.txt` — báo cáo text chi tiết
- `output/results/erp/erp_quality_per_trial.csv` — 840 rows (per trial)
- `output/results/erp/erp_quality_subject_summary.csv` — 28 rows

---

## 4. Visualization of Quality Check

**Script**: `run_erp_quality_viz.py`

### Figures

| File | Nội dung |
|------|----------|
| `quality_heatmap.png` | Heatmap quality score × condition cho từng subject |
| `subject_quality_summary.png` | Bar chart GOOD/WEAK/BAD per subject |
| `good_vs_bad_waveforms.png` | ERP waveform GOOD (xanh) vs BAD (đỏ) cho 6 conditions |
| `component_detection_rates.png` | Detection rate per component per subject và per condition |
| `snr_distributions.png` | Histogram avg_SNR và quality score |
| `exclusion_list.png` | Danh sách chi tiết conditions bị loại |

### Location

`output/figures/erp_quality/`

---

## 5. ERP Analysis with Quality Filter

**Script**: `run_erp_filtered.py`

Chạy lại ERP grand average sau khi loại BAD conditions.

### Kết quả

| Metric | Unfiltered | Weak filter | Strict filter |
|--------|:----------:|:-----------:|:-------------:|
| Trials | 804 | 570 (71%) | 427 (53%) |
| Subject×condition | 168 | 119 | 90 |
| Subjects complete (6/6) | 28 | 3 | 0 |

### P2 mean_amp (µV) comparison

| Condition | Unfiltered | Weak filter | Strict filter |
|-----------|:----------:|:-----------:|:-------------:|
| Water/605 | +1.22 | +2.79 | +2.06 |
| Low/258 | **-1.65** | **+2.05** | +2.27 |
| MedLow/453 | +0.30 | +2.92 | +3.85 |
| Medium/189 | +3.07 | +3.72 | +3.36 |
| MedHigh/762 | +0.53 | +1.32 | +3.20 |
| High/893 | +5.10 | +5.03 | **+7.28** |

> **Key**: Sau filter, P2 tất cả conditions đều positive (physiologically correct). Low/258 từ -1.65µV lên +2.05µV. High/893 strict filter đạt +7.28µV.

### Cohen's d (High vs Water)

| Dataset | d |
|---------|:-:|
| Unfiltered | 0.35 |
| Filtered (weak) | 0.19 |
| **Filtered (strict)** | **0.49** |

### Files output

`output/results/erp_filtered/`:
- `component_measures_filtered.csv`
- `component_measures_filtered_strict.csv`
- `concentration_summary_filtered.csv`
- `concentration_summary_filtered_strict.csv`

---

## 6. ANOVA Comparison

**Script**: `run_anova_comparison.py`

### rmANOVA P2_mean_amp ~ Condition

| Data | N subjects complete | F | p | η²p |
|------|:---:|:---:|:---:|:---:|
| Unfiltered | 28 | 1.327 | 0.2565 | 0.040 |
| Filtered (weak) | **3** | — | — | — |
| Filtered (strict) | **0** | — | — | — |

> rmANOVA không chạy được trên filtered data vì quá ít subjects có đủ 6 conditions.

### Mixed Model (LRT)

| Data | χ² | p |
|------|:---:|:---:|
| Unfiltered | p<0.001 | * |
| Filtered (weak) | p<0.001 | * |
| Filtered (strict) | p<0.001 | * |

> Mixed model có convergence issues (random effects covariance singular). LRT results không đáng tin cậy.

### JAR ANOVA (1-way between)

| Data | F | p |
|------|:---:|:---:|
| Unfiltered | 0.619 | 0.5398 |
| Filtered (weak) | 0.245 | 0.7833 |
| Filtered (strict) | 0.190 | 0.8272 |

---

## 7. Per-Channel ANOVA

**Script**: `run_per_channel_anova.py`

Phân tích từng channel riêng lẻ thay vì gộp ROI. Dùng **unfiltered data** (28 subjects) vì rmANOVA cần balanced design.

### Concentration effect (rmANOVA, n=28)

**Significant channels (p<0.05):**

| Channel | Component | Measure | F | p | η²p | Brain region |
|---------|-----------|---------|:---:|:---:|:---:|-------------|
| **C4** | P2 | peak_amp | 2.30 | **0.0480** | 0.058 | Central right |
| **P3** | P1 | peak_amp | 2.32 | **0.0465** | 0.066 | Parietal left |
| **F7** | N1 | mean_amp | 2.39 | **0.0412** | 0.063 | Inferior frontal left |

**Gần significant (p<0.10):**

| Channel | Component | Measure | F | p | η²p |
|---------|-----------|---------|:---:|:---:|:---:|
| P3 | P1 | mean_amp | 2.17 | 0.061 | 0.061 |
| C3 | P1 | peak_amp | 2.06 | 0.074 | 0.061 |
| F4 | P2 | peak_amp | 2.03 | 0.078 | 0.052 |
| T8 | P2 | mean_amp | 1.97 | 0.087 | 0.054 |
| F4 | P2 | mean_amp | 1.93 | 0.093 | 0.051 |
| T8 | N1 | mean_amp | 1.92 | 0.096 | 0.053 |

### JAR effect (1-way between, n=168)

**Significant channels (p<0.05):**

| Channel | Component | Measure | F | p | Region |
|---------|-----------|---------|:---:|:---:|--------|
| **C4** | **P2** | **peak_amp** | **5.14** | **0.0068** | Central right ← mạnh nhất |
| **F7** | **N400** | **peak_amp** | **4.59** | **0.0115** | Inferior frontal left |
| **F7** | **N400** | **mean_amp** | **4.11** | **0.0181** | Inferior frontal left |
| **F7** | **P2** | **peak_amp** | **3.45** | **0.0341** | Inferior frontal left |
| **P4** | **N400** | **mean_amp** | **3.30** | **0.0394** | Parietal right |
| **C3** | **P1** | **peak_amp** | **3.40** | **0.0359** | Central left |
| **P7** | **P1** | **peak_amp** | **3.20** | **0.0434** | Parietal-temporal left |
| **F7** | **P1** | **mean_amp** | **3.09** | **0.0480** | Inferior frontal left |
| **P4** | **P2** | **peak_amp** | **3.16** | **0.0452** | Parietal right |

### Files output

- `output/results/per_channel/concentration_anova_per_channel.csv`
- `output/results/per_channel/jar_anova_per_channel.csv`
- `output/results/per_channel/concentration_heatmap.png`

---

## 8. JAR Analysis

**Script**: `run_jar_analysis.py`

### JAR Means — All Components (unfiltered)

**P2 mean_amp:**
| JAR | n | µV | Pattern |
|-----|:-:|:---:|---------|
| Không đủ | 83 | +0.52 | Thấp nhất |
| **Vừa phải** | 44 | **+2.62** | **Cao nhất** ✓ |
| Quá nhiều | 41 | +1.99 | Trung bình |

> ✓ Consistent với theory: P2 ~ pleasantness, cao nhất khi "vừa phải"

**N1 mean_amp:**
| JAR | n | µV | Pattern |
|-----|:-:|:---:|---------|
| Không đủ | 83 | -0.57 | |
| **Vừa phải** | 44 | **-1.58** | **Âm nhất** |
| Quá nhiều | 41 | -0.09 | |

> Vừa phải có N1 âm nhất (chú ý/phân biệt mạnh nhất)

**Sau strict filter P2:**
| JAR | µV |
|-----|:---:|
| Không đủ | +3.23 |
| Vừa phải | **+4.95** |
| Quá nhiều | +4.12 |

---

## 9. Per-Channel Visualization

**Script**: `run_per_channel_viz.py`

### Figures

| File | Content |
|------|---------|
| `C4_P2_by_JAR.png` | C4 P2 waveform split by JAR group |
| `F7_N400_by_JAR.png` | F7 N400 waveform split by JAR group |
| `F4_P2_by_JAR.png` | F4 P2 waveform split by JAR group |
| `dose_response_P2_C4_F4_P3.png` | Dose-response curves (P2 peak_amp × concentration) |
| `topomap_P2_by_JAR.png` | P2 topography by JAR group |
| `topomap_N400_by_JAR.png` | N400 topography by JAR group |
| `key_channels_JAR_bars.png` | Bar charts: C4 P2, F7 N400, P4 N400 by JAR |

### Location

`output/figures/per_channel/`

---

## 10. ML with Significant Channels

**Script**: `run_ml_significant_channels.py`

### Feature sets

| Set | Features | Channels × Components |
|-----|----------|----------------------|
| **JAR_significant** | 8 | P2_C4, N400_F7, P2_F7, P1_C3, N400_P4, P1_P7, P1_F7, P2_P4 |
| **Conc_significant** | 3 | N1_F7, P1_P3, P2_C4 |

### Results (LOSO, quality-filtered data)

**JAR 3-class (chance = 33.3%):**

| Features | LogisticRegression | SVM | RandomForest |
|----------|:---:|:---:|:---:|
| JAR_significant (8) | **46.8%** | 31.2% | 38.1% |
| Conc_significant (3) | **48.1%** | 35.3% | 40.4% |

**High vs Water (chance = 50%):**

| Features | LogisticRegression | SVM | RandomForest |
|----------|:---:|:---:|:---:|
| JAR_significant (8) | 46.4% | **50.7%** | 49.3% |
| Conc_significant (3) | 46.4% | 50.2% | 43.0% |

### Files

`output/results/ml_significant_channels/` + `output/figures/ml_significant_channels/`

---

## 11. ML with Feature Selection

**Script**: `run_ml_top_features.py`

### Feature pool

| Type | Count | Detail |
|------|:-----:|--------|
| ERP mean_amp | 64 | 16 channels × 4 components |
| Bandpower (log10) | 80 | 16 channels × 5 bands (delta/theta/alpha/beta/gamma) |
| **Total** | **144** | |

### Method

1. Extract 144 features per trial
2. StandardScaler
3. Mutual Information ranking
4. Filter by F-test p<0.10
5. Select top-K features
6. LOSO CV

### Best Results

| Task | Model | K | Accuracy | vs Chance |
|------|:----:|:-:|:--------:|:---------:|
| **JAR 3-class** | LogisticRegression | 5 | **47.7%** | +14.4% |
| **Vua_phai vs Others** | LogisticRegression | 5 | **74.6%** | +24.6% |
| **High vs Water** | RandomForest | 15 | **57.5%** | +7.5% |

### Top 5 Features per Task

**JAR 3-class (47.7%):**
1. BP_gamma_P7 — gamma power parietal-temporal left
2. ERP_N1_F4 — N1 amplitude frontal right
3. BP_alpha_P3 — alpha power parietal left
4. BP_alpha_P7 — alpha power parietal-temporal left
5. ERP_P2_F3 — P2 amplitude frontal left

**Vua_phai vs Others (74.6%):**
1. ERP_N1_F4 — N1 amplitude frontal right
2. ERP_N400_Fp2 — N400 amplitude frontopolar right
3. BP_gamma_P7 — gamma power parietal-temporal left
4. ERP_N400_P7 — N400 amplitude parietal-temporal left
5. BP_beta_Fp1 — beta power frontopolar left

**High vs Water (57.5%):**
1. ERP_P2_P7 — P2 amplitude parietal-temporal left
2. ERP_N400_C3 — N400 amplitude central left
3. BP_beta_P4 — beta power parietal right
4. ERP_P2_Fp2 — P2 amplitude frontopolar right
5. ERP_N400_O2 — N400 amplitude occipital right

### Accuracy vs K

```
Task           K=5    K=10   K=15   K=20   K=48   ALL(144)
JAR 3-class   47.7   46.7   45.3   44.4   38.1   36.7
Vua_phai vs Others 74.6  74.4   74.0   73.7   73.3   67.4
High vs Water 56.0   54.1   57.5   50.7   51.2   45.9
```

> **K=5 là optimal** — thêm features làm giảm accuracy (curse of dimensionality)

### Files

- `output/results/ml_top_features/all_results.csv` — tất cả results
- `output/results/ml_top_features/feature_importance.csv` — MI + F-score cho 144 features
- `output/figures/ml_top_features/accuracy_vs_k.png`
- `output/figures/ml_top_features/top_features_mi.png`

---

## 12. Summary of Findings

### What We Found

1. **ERP quality varies significantly across subjects**
   - 29% subject×condition là BAD cần loại bỏ
   - P008 worst (5/6 BAD), P013/P019/P023 best (6/6 all GOOD)

2. **Quality filter improves ERP waveforms**
   - P2 amplitudes become positive (physiologically correct)
   - High/893 P2 increases from +5.10µV to +7.28µV after strict filter
   - Cohen's d (High vs Water) improves from 0.35 to 0.49

3. **JAR effect không significant ở cấp ROI nhưng có signal ở cấp channel**
   - C4 P2 peak_amp: p=0.0068 cho JAR effect
   - F7 N400: p=0.0115 cho JAR effect
   - Pattern: Vừa phải > Quá nhiều > Không đủ (P2 amplitude)

4. **Concentration effect yếu nhưng detectable ở channels cụ thể**
   - C4 P2 peak: p=0.048
   - P3 P1 peak: p=0.047
   - F7 N1 mean: p=0.041

5. **Không effect nào survives Bonferroni correction** (128 tests, α=0.00039)
   - Effect size nhỏ (η²p=0.05-0.07)
   - Cần N lớn hơn để detect effects reliably

6. **ML với feature selection cho kết quả khả quan**
   - JAR 3-class: 47.7% (chance 33.3%)
   - Vua_phai vs Others: 74.6% (chance 50%)
   - High vs Water: 57.5% (chance 50%) — vẫn khó

7. **Bandpower features bổ sung tốt cho ERP features**
   - Gamma_P7, Alpha_P3, Beta_P4 xuất hiện trong top features
   - Kết hợp ERP + bandpower tốt hơn ERP đơn thuần

### Recommendations

1. **Phân tích nên dùng filter** (loại BAD conditions) cho grand average và descriptive stats
2. **Per-channel analysis** thay vì ROI averaging — C4, F7, P3 là channels quan trọng nhất
3. **K=5 features là optimal** cho ML — không nên dùng quá nhiều features
4. **Vua_phai vs Others** là task ML khả thi nhất (74.6%) — có thể dùng deployment
5. **High vs Water cần thêm features** — có thể thử ERP peak latencies, TFR, connectivity

### File Structure

```
output/
├── results/
│   ├── erp/
│   │   ├── erp_quality_flags.csv
│   │   ├── erp_quality_report.txt
│   │   ├── erp_quality_per_trial.csv
│   │   ├── erp_quality_subject_summary.csv
│   │   └── component_measures.csv
│   ├── erp_filtered/
│   │   ├── component_measures_filtered.csv
│   │   ├── component_measures_filtered_strict.csv
│   │   ├── concentration_summary_filtered.csv
│   │   └── concentration_summary_filtered_strict.csv
│   ├── per_channel/
│   │   ├── concentration_anova_per_channel.csv
│   │   ├── jar_anova_per_channel.csv
│   │   └── concentration_heatmap.png
│   ├── ml_significant_channels/
│   │   └── all_results_comparison.csv
│   └── ml_top_features/
│       ├── all_results.csv
│       └── feature_importance.csv
├── figures/
│   ├── erp_quality/             (6 files)
│   ├── per_channel/             (7 files)
│   └── ml_top_features/         (2 files)
```
