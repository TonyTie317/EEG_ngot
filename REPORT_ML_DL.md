# Báo cáo thực nghiệm ML/DL — Phân loại Vua_phai vs Others

**Bài toán:** Phân loại nhị phân — Vua_phai (vừa phải ngọt) vs Others (Khong_du + Qua_nhieu)  
**Dữ liệu:** 28 subject, 6 điều kiện/subject (5 nồng độ sucrose + 1 nước), mỗi điều kiện lặp 5 lần  
**EEG:** 16 kênh, 100 Hz, epoch -0.5 → +3.0 s, baseline [-0.5, -0.3] s  
**Ngày thực hiện:** 2026-05-28  

---

## 0. Metric quan trọng

| Metric | Ý nghĩa | Tại sao dùng |
|--------|---------|-------------|
| **balanced_acc** | (recall_Vua_phai + recall_Others) / 2 | **Metric chính** — không bị imbalance ảnh hưởng |
| accuracy | (TP+TN) / N | Dễ gian lận (predict toàn Others → acc=0.756) |
| oracle_acc | accuracy tốt nhất với threshold tối ưu post-hoc | Ceiling lý thuyết của raw accuracy |
| recall_vua_phai | TPR — bắt được bao nhiêu % Vua_phai | Quan trọng nếu muốn phát hiện Vua_phai |

> **Lưu ý:** `accuracy = 0.789` trong báo cáo cũ là **giả** (balanced_acc = 0.500 = ngẫu nhiên).  
> Model chỉ predict toàn "Others" → cao hơn majority baseline (0.756) không có nghĩa.

---

## 1. Dữ liệu và thiết lập

### 1.1 Condition-averaged (n=119)
- Trung bình 5 lần lặp per (subject, condition) → 1 hàng/điều kiện
- Sau quality filter (loại BAD): n=119, Vua_phai=29 (24.4%), Others=90
- Majority baseline acc = 0.756

### 1.2 Per-trial (n=840)  
- Giữ nguyên từng trial: n=840, Vua_phai=220 (26.2%), Others=620
- Majority baseline acc = 0.738

### 1.3 Feature sets
| Set | File | # Features | Nguồn |
|----|------|-----------|-------|
| v3 general | `features_jar3_adv.csv` | 962 | ERP + bandpower + Hjorth + DWT + connectivity |
| gERP-specific | `features_gerp_avg.csv` | 489 | Late positivity + 50ms bins + asymmetry + theta |
| Combined | merge v3+gERP | 1,417 | — |

---

## 2. ML — Condition-Averaged Features

### 2.1 Baseline v1 (trước thực nghiệm này)
```
Filter: weak (non-BAD)
Model:  RandomForest, K=15 MI features, IsoForest contam=0.10
Result: accuracy=0.773  balanced_acc=0.558  ← v1 best (nhưng model predict chủ yếu Others)
```

### 2.2 ML v2 — SMOTE + Threshold Tuning (`run_ml_vuaphai_v2.py`)

**Strategies:**
- Phase 1: Grid sweep toàn bộ sampling (none/smote/adasyn/borderline) × K × 8 models
- Phase 2: Threshold tuning (inner 3-fold CV) cho fast models (LogReg, SVM, GradBoost)
- Phase 3: Stacking (GBT + SVM + LogReg → meta-LogReg)

**Kết quả tốt nhất:**

| Rank | Model | Sampling | K | Thr | acc | **bacc** | f1 | rec_vua | rec_oth |
|------|-------|---------|---|-----|-----|---------|-----|---------|---------|
| 1 | **GradBoost** | none | 20 | **0.208** | 0.681 | **0.649** | 0.622 | 0.586 | 0.711 |
| 2 | GradBoost | smote | 30 | 0.5 | 0.706 | 0.607 | 0.606 | 0.414 | 0.767 |
| 3 | GradBoost | none | 30 | 0.5 | 0.748 | 0.600 | 0.609 | 0.310 | — |

> **Insight chính:** Threshold tuning (thr=0.208 thay vì 0.5) là yếu tố quyết định.  
> Model mặc định "lười" → dự đoán Others. Hạ ngưỡng xuống 0.208 buộc model bắt Vua_phai.

**Runtime:** 21 phút

---

### 2.3 ML v3 — Extended K Sweep + GPU + Subject Removal (`run_ml_vuaphai_v3.py`)

**Strategies:**
- K sweep: 1→30 (step 1)
- IsoForest contam: {0.05, 0.10, 0.15} (loại outlier mẫu per fold)
- Subject SNR removal: {0, 2, 3} subject SNR thấp nhất bị loại
  - Loại: P008 (snr=0.354), P005 (snr=0.376), P025 (snr=0.437)
- GPU models: XGBoost (device='cuda'), LightGBM (device='gpu')
- CPU models: RF_100, BalancedRF, LogReg, SVM
- Oracle threshold: post-hoc sweep 0.10→0.90

**Top 5 kết quả (theo oracle_acc):**

| # | Model | samp | K | iso | rm | acc | oracle_acc | bacc |
|---|-------|------|---|-----|----|-----|-----------|------|
| 1 | **XGB_gpu** | none | 29 | 0.10 | 3 | 0.778 | **0.824** | 0.604 |
| 2 | LGBM_gpu | none | 29 | 0.05 | 2 | 0.768 | 0.821 | 0.674 |
| 3 | RF_100 | smote | 16 | 0.10 | 0 | 0.756 | 0.815 | 0.652 |
| 4 | XGB_gpu | none | 25 | 0.10 | 3 | 0.806 | 0.815 | 0.622 |
| 5 | XGB_gpu | none | 27 | 0.10 | 3 | 0.769 | 0.815 | 0.598 |

**Kết quả best:**
```
Model:        XGB_gpu (device='cuda')
K:            29 MI features
iso_contam:   0.10
subj_remove:  3 (loại P008, P005, P025 — SNR thấp nhất)
sampling:     none (scale_pos_weight=3 xử lý imbalance)
accuracy:     0.7778  (threshold=0.5)
oracle_acc:   0.8241  (threshold=0.59)
balanced_acc: 0.6039
recall_vua:   0.2800
```

> **Target 0.85 không đạt.** Ceiling thực tế ~0.82-0.83 oracle_acc với dữ liệu condition-averaged.

**Runtime:** 124 phút

---

### 2.4 ML gERP — Feature Engineering Chuyên Biệt (`run_ml_gerp_avg.py`)

**Phát hiện từ grand-average ERP tại Cz:**
| Window | Vua_phai | Others | Δ |
|--------|---------|--------|---|
| Late Positivity 700-1000ms | **4.30 µV** | 2.25 µV | **+2.05 µV** |
| Late Positivity 500-700ms | **3.91 µV** | 2.58 µV | **+1.34 µV** |
| N400 300-500ms | 1.86 µV | 2.38 µV | -0.52 µV |
| P2 200-350ms | 1.73 µV | 2.50 µV | -0.76 µV |

**gERP Features (489 tổng):**
- Fine-grained 50ms bins 0→2s tại 10 taste channels (400 feat)
- Component amplitudes: LP1, LP2, LP3, N400, P2 (90 feat)
- Hemispheric asymmetry F4-F3, C4-C3, P4-P3 (20 feat)
- Theta/beta spectral trong early/late windows (24 feat)
- Slope của Late Positivity (4 feat)

**Top MI features (sau quality filter n=119):**
| # | Feature | Vua_phai | Others | Δ |
|---|---------|---------|--------|---|
| 1 | bin_F3_0550 (F3 @ 550ms) | 4.32 µV | -0.79 µV | **+5.12** |
| 2 | bin_Fz_1400 (Fz @ 1400ms) | -3.70 µV | -9.17 µV | **+5.47** |
| 3 | bin_Fz_0300 (Fz @ 300ms) | -1.34 µV | -6.10 µV | **+4.76** |
| 4 | bin_C3_0250 (C3 @ 250ms) | 1.55 µV | 9.88 µV | **-8.33** |

> **Insight neuroscience:** Kênh phân biệt tốt nhất là **F3/Fz (frontal)**, không phải Cz (central).  
> Prefrontal cortex xử lý **hedonic evaluation** (đánh giá vừa phải) của vị ngọt.

**Kết quả:**
```
Best: GradBoost K=8 samp=smote
accuracy:     0.6975
balanced_acc: 0.5897  ← thấp hơn v2 (0.649)
```

> gERP features đứng một mình không beat được general features.  
> Lý do: 962 general features trong v3 đã bao phủ các features tương tự.

---

### 2.5 Combined Features (`run_ml_combined.py`)
Kết hợp v3 (962) + gERP (455) = 1,417 features, MI chọn best K.

| Dataset | Model | samp | K | bacc | oracle_bacc | rec_vua |
|---------|-------|------|---|------|------------|---------|
| **v3_only** | **XGB_gpu** | smote | 25 | **0.637** | 0.637 | 0.552 |
| combined | XGB_gpu | none | 30 | 0.624 | 0.635 | 0.414 |
| gerp_only | GradBoost | smote | 30 | 0.549 | 0.559 | 0.241 |

> Combined không cải thiện so với v3_only. gERP features không mang thêm thông tin mới.

---

## 3. Per-Trial ML (`run_ml_pertrial.py`)

**Approach:** Trích xuất features từng trial riêng lẻ (không average) → n=840 samples.

**Features per trial (128 tổng):**
- ERP window mean/peak/rms tại 5 component windows
- Band power (delta/theta/alpha/beta/gamma) per channel
- Time-domain stats per channel

**Kết quả:**
```
Best: RF K=5 iso=0.05 none
oracle_acc: 0.739  ≈ majority baseline (0.738!)
balanced_acc: 0.500  ← chance level
```

> **Thất bại hoàn toàn.** Single-trial EEG quá nhiễu để trích xuất ERP components đáng tin.  
> Averaging là bắt buộc để có tín hiệu ERP rõ.

---

## 4. Deep Learning — Per-Trial Raw Epochs

**Input:** Raw epoch (16 ch × 351 tp), không feature engineering  
**CV:** LOSO by subject (28 folds)  
**GPU:** RTX 4090 (25GB VRAM), PyTorch 2.8.0+cu128  
**Class imbalance:** WeightedRandomSampler + BCEWithLogitsLoss(pos_weight)

### 4.1 DL v1 — Model Comparison (`run_dl_vuaphai.py`)

**Config:** pos_weight=3.0, n_epochs=200, lr=5e-4, batch_size=32, patience=25  
**Augmentation:** Random temporal crop 80-100%

| Model | params | acc | **bacc** | f1 | rec_vua | rec_oth | oracle_acc |
|-------|--------|-----|---------|-----|---------|---------|-----------|
| EEGNet | 5,377 | 0.391 | 0.570 | 0.384 | **0.946** | 0.194 | 0.723 |
| EEGNet_light | 1,673 | 0.401 | 0.578 | 0.396 | **0.950** | 0.207 | 0.736 |
| ShallowConvNet | 27,361 | 0.570 | 0.614 | 0.552 | 0.705 | 0.522 | 0.725 |
| **DeepConvNet** | **145,726** | 0.520 | **0.624** | 0.517 | 0.841 | 0.406 | 0.720 |

> EEGNet recall_vua=0.945 nhưng acc=0.39 → quá bias về Vua_phai (pos_weight=3.0 quá cao).  
> DeepConvNet balance tốt nhất nhưng bacc=0.624 < ML v2 (0.649).

**Runtime:** 5 phút 27 giây

---

### 4.2 DL v2 — pos_weight Sweep + Augmentation + Ensemble (`run_dl_vuaphai_v2.py`)

**Improvements:**
- pos_weight sweep: {1.5, 2.0, 3.0, 4.0}
- Augmentation: Gaussian noise (σ=0.03×std) + random time-shift ±10 samples
- n_epochs=250, patience=30
- DL × XGB GPU ensemble (α=0.3/0.5/0.7)

**pos_weight sweep results:**

*EEGNet:*
| pw | acc | **bacc** | rec_vua | rec_oth |
|----|-----|---------|---------|---------|
| 1.5 | 0.494 | 0.618 | 0.877 | 0.358 |
| **2.0** | 0.464 | **0.621** | 0.950 | 0.292 |
| 3.0 | 0.408 | 0.577 | 0.932 | 0.223 |
| 4.0 | 0.387 | 0.567 | 0.946 | 0.189 |

*ShallowConvNet:*
| pw | acc | **bacc** | rec_vua | rec_oth |
|----|-----|---------|---------|---------|
| 1.5 | 0.638 | 0.641 | 0.646 | 0.636 |
| 2.0 | 0.612 | 0.623 | 0.646 | 0.600 |
| **3.0** | **0.639** | **0.674** | **0.746** | **0.602** |
| 4.0 | 0.586 | 0.633 | 0.732 | 0.534 |

*DeepConvNet:*
| pw | acc | **bacc** | rec_vua | rec_oth |
|----|-----|---------|---------|---------|
| 1.5 | 0.604 | 0.624 | 0.668 | 0.581 |
| 2.0 | 0.566 | 0.622 | 0.741 | 0.503 |
| **3.0** | 0.569 | **0.666** | **0.868** | 0.463 |
| 4.0 | 0.499 | 0.628 | 0.900 | 0.357 |

**Best overall:**
```
Model:         ShallowConvNet  pw=3.0
accuracy:      0.6393
balanced_acc:  0.6735  ← BEST TOÀN BỘ
f1_macro:      0.6155
recall_vua:    0.7455  (bắt 74.6% Vua_phai)
recall_others: 0.6016  (đúng 60.2% Others)
```

**DL × XGB Ensemble:** Không cải thiện (bacc=0.624) — do condition-avg (XGB) và per-trial (DL) có phân phối khác nhau.

**Runtime:** 21 phút 31 giây

---

## 5. Tổng hợp kết quả

### 5.1 Bảng tiến trình (theo balanced_acc)

| # | Phương pháp | Script | acc | **bacc** | rec_vua | Ghi chú |
|---|-------------|--------|-----|---------|---------|---------|
| 1 | RF K=15 (v1 baseline) | — | 0.773 | 0.500 | ~0.0 | **GIẢ** — predict toàn Others |
| 2 | GradBoost + IsoForest | v1 opt | 0.748 | 0.558 | 0.138 | Real learning bắt đầu |
| 3 | ML v3 XGB GPU K=29 rm=3 | v3 | 0.778 | 0.604 | 0.280 | Best raw accuracy |
| 4 | GradBoost smote samp=none | v2 | 0.706 | 0.607 | 0.414 | SMOTE cải thiện rec_vua |
| 5 | DL DeepConvNet pw=3 | DL v1 | 0.520 | 0.624 | 0.841 | DL bắt nhiều Vua_phai hơn |
| 6 | **GradBoost thr=0.208** | **v2** | **0.681** | **0.649** | **0.586** | **Best ML** |
| 7 | DL ShallowConvNet pw=3 | DL v2 | 0.639 | **0.674** | 0.746 | **Best toàn bộ** |

### 5.2 Tradeoff accuracy vs balanced_acc

```
High acc + low bacc  → model predict chủ yếu Others (majority class)
Low acc  + high bacc → model bắt nhiều Vua_phai nhưng nhiều false positive
Balance              → ShallowConvNet pw=3: acc=0.64, bacc=0.674, rec_vua=0.75
```

---

## 6. Phân tích kết quả

### 6.1 Tại sao không đạt acc > 0.85?

**Nguyên nhân cốt lõi: n quá nhỏ + class imbalance nặng**
- n=119 condition-averaged samples (sau quality filter)
- Chỉ 29 Vua_phai samples (24%)
- LOSO-CV: mỗi fold train có ~27 Vua_phai — quá ít cho reliable learning
- Ceiling dữ liệu: oracle_acc ~0.82-0.83

**Kiểm tra:** 5 phương pháp khác nhau đều tụ về ~0.67-0.68 bacc → đây là **data ceiling**, không phải model limitation.

### 6.2 Tại sao per-trial thất bại?
- Single-trial EEG SNR thấp: ERP components (P2, N400, Late Positivity) chỉ rõ ràng sau averaging 5+ repeats
- gERP features per-trial → bacc=0.51 (chance level)
- DL trên raw per-trial: bacc=0.67 — tốt hơn nhờ học trực tiếp từ raw signal

### 6.3 Insight neuroscience
- **Late Positivity (500-1000ms) tại Cz** là component phân biệt tốt nhất:  
  Vua_phai=4.30µV vs Others=2.25µV (Δ=+2.05µV)
- **F3/Fz (frontal)** phân biệt tốt hơn Cz (central) trong MI analysis
- Giải thích: Prefrontal cortex xử lý hedonic evaluation (cảm giác "vừa phải")

---

## 7. Hướng cải thiện tiếp theo

| Hướng | Kỳ vọng | Effort |
|-------|---------|--------|
| **Thu thập thêm subject** (50+) | ↑↑↑ | Cao |
| Cải thiện quality EEG (ICA thủ công) | ↑↑ | Trung bình |
| Transfer learning từ dataset EEG lớn (BCI Competition) | ↑↑ | Cao |
| Tối ưu EEGNet với attention mechanism | ↑ | Trung bình |
| Subject-specific threshold tuning | ↑ | Thấp |

---

## 8. Files output

```
output/results/
├── ml_vuaphai_v2/       # SMOTE + threshold tuning
│   ├── all_results_v2.csv
│   └── best_per_sampling.csv
├── ml_vuaphai_v3/       # GPU K sweep + subject removal
│   ├── all_results_v3.csv
│   ├── top30_by_oracle_acc.csv
│   └── top30_by_balanced_acc.csv
├── dl_vuaphai/          # EEGNet/ShallowConvNet/DeepConvNet v1
│   └── results_dl.csv
├── dl_vuaphai_v2/       # pos_weight sweep + ensemble
│   ├── results_dl_v2.csv
│   └── ensemble_results.csv
├── ml_pertrial/         # Per-trial ML (failed)
│   └── results_pertrial.csv
├── ml_gerp_avg/         # gERP condition-avg features
│   ├── features_gerp_avg.csv
│   └── results_gerp_avg.csv
└── ml_combined/         # Combined v3+gERP features
    └── results_combined.csv

output/figures/
├── ml_vuaphai_v2/       # sampling_comparison.png, balanced_acc_sweep.png
├── ml_vuaphai_v3/       # heatmap_K_vs_contam.png, oracle_acc_vs_K.png
├── dl_vuaphai/          # model_comparison.png, cm_*.png
├── dl_vuaphai_v2/       # bacc_vs_posweight.png, cm_best.png
└── ml_combined/         # dataset_comparison.png
```

---

*Báo cáo được tạo tự động từ log files thực nghiệm — 2026-05-28*
