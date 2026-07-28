# ST-GCN Training Guide: Improved Implementation

This directory contains improved ST-GCN models for AU-based emotion recognition, inspired by the original ST-GCN paper (Yan et al., AAAI 2018).

## 📁 Files Overview

### Core Files
- `model_stgcn.py` - Original SimpleSTGCN (lightweight baseline)
- `model_stgcn_improved.py` - **NEW**: 3 improved variants (Simple/Hybrid/Full)
- `train_stgcn.py` - Original training script
- `train_stgcn_improved.py` - **NEW**: Enhanced training with LR scheduling, early stopping
- `dataset_stgcn.py` - Dataset loader for AU sequences

### Analysis & Visualization
- `benchmark_models.py` - **NEW**: Compare speed, memory, parameters
- `visualize_edge_importance.py` - **NEW**: Visualize learned AU connections
- `MODEL_COMPARISON.md` - **NEW**: Detailed comparison documentation

### Data Processing
- `b2_video_normalization.py` - Extract AU features from videos
- `make_split.py` - Create train/val split

---

## 🚀 Quick Start

### 1. Process Videos to AU Features
```bash
python b2_video_normalization.py \
    --input ../Labrecorder/Data_chuan \
    --output outputb2 \
    --target-fps 60 \
    --write-preview
```

### 2. Create Train/Val Split
```bash
python make_split.py
```
This creates `split.csv` with train/val assignments.

### 3. Train Model (Choose One)

**Option A: Original SimpleSTGCN (Fast, Small)**
```bash
python train_stgcn.py
```

**Option B: Improved Models (Recommended)**
```bash
# Edit train_stgcn_improved.py, set MODEL_TYPE to:
# - 'simple': Same as original (200K params)
# - 'hybrid': Balanced model (400K params) ← RECOMMENDED
# - 'full': Full ST-GCN (800K params)

python train_stgcn_improved.py
```

### 4. Benchmark Models
```bash
python benchmark_models.py
```

### 5. Visualize Learned Connections
```bash
python visualize_edge_importance.py \
    --model stgcn_hybrid_best.pt \
    --model-type hybrid \
    --output-dir edge_viz
```

---

## 📊 Model Comparison

| Model | Params | Layers | BatchNorm | Edge Learning | Best For |
|-------|--------|--------|-----------|---------------|----------|
| **SimpleSTGCN** | 200K | 2 | ❌ | ❌ | Small data (<500) |
| **HybridSTGCN** | 400K | 3 | ✅ | ✅ | **Medium data (500-2K)** |
| **ImprovedSTGCN** | 800K | 5 | ✅ | ✅ | Large data (>2K) |

### Key Improvements in New Models

1. **Input Normalization**: BatchNorm on all node features
2. **Proper GCN**: Symmetric normalized adjacency `D^(-1/2) A D^(-1/2)`
3. **Residual Connections**: With dimension matching
4. **Edge Importance Weighting**: Learnable per-layer edge weights
5. **Training Enhancements**:
   - Learning rate scheduling (ReduceLROnPlateau)
   - Gradient clipping
   - Early stopping
   - Better regularization (dropout)

---

## 📈 Expected Results

### Your Dataset (~50 samples)

| Model | Train Acc | Val Acc | Training Time | Overfitting Risk |
|-------|-----------|---------|---------------|------------------|
| SimpleSTGCN | ~80% | ~60% | Fast (2-3 min/epoch) | Low |
| HybridSTGCN | ~90% | **~65-70%** | Medium (5-7 min/epoch) | Medium |
| ImprovedSTGCN | ~95% | ~60-65% | Slow (10-15 min/epoch) | High |

**Recommendation**: Start with **HybridSTGCN** + high dropout (0.5)

---

## 🔍 Detailed Comparison with Original ST-GCN

### Original ST-GCN (GitHub - for skeleton action recognition)
```
Input: (N, C, T, V, M) - batch, channels, time, vertices, persons
Architecture:
  ├─ 10 ST-GCN layers
  ├─ Channels: 64→64→64→64→128→128→128→256→256→256
  ├─ Temporal kernel: 9
  ├─ Spatial: Multi-partition adjacency [K, V, V]
  ├─ Edge importance: Learnable per layer
  └─ Parameters: ~3.5M

Design for: Large datasets (NTU-RGB+D: 56K samples)
```

### Your SimpleSTGCN (Current baseline)
```
Input: (B, N, T, F) - batch, nodes, time, features
Architecture:
  ├─ 2 simple layers (spatial + temporal)
  ├─ Hidden: 128 (fixed)
  ├─ Temporal kernel: 5, 3
  ├─ Spatial: Simple message passing A @ X
  ├─ No BatchNorm, no residual, no edge learning
  └─ Parameters: ~200K

Design for: Very small datasets (<1K samples)
```

### ImprovedSTGCN (New - balanced for AU data)
```
Input: (B, N, T, F)
Architecture:
  ├─ 5 ST-GCN layers (balanced depth)
  ├─ Channels: 64→64→128→128→256
  ├─ Temporal kernel: 9 (same as original)
  ├─ Spatial: Normalized adjacency D^(-1/2) A D^(-1/2)
  ├─ Edge importance: Learnable [N, N] per layer
  ├─ BatchNorm + Residual + Dropout
  └─ Parameters: ~800K

Design for: Medium AU datasets (1K-5K samples)
```

---

## 🛠️ Configuration Guide

### Training Hyperparameters

```python
# In train_stgcn_improved.py

# For small dataset (<500 samples)
EPOCHS = 50
BATCH_SIZE = 4
LR = 5e-4
WEIGHT_DECAY = 1e-3  # High regularization
MODEL_TYPE = 'simple'
# In model config: dropout=0.5

# For medium dataset (500-2000 samples)
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_TYPE = 'hybrid'  # ← RECOMMENDED
# In model config: dropout=0.3

# For large dataset (>2000 samples)
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
MODEL_TYPE = 'full'
# In model config: dropout=0.2
```

### Model Parameters

```python
# Create custom model
from model_stgcn_improved import HybridSTGCN

model = HybridSTGCN(
    in_channels=10,      # AU feature dim
    num_classes=4,       # ngot, man, chua, dang
    num_nodes=15,        # Number of AU nodes
    hid=128,             # Hidden size (128 or 256)
    dropout=0.3          # Dropout rate (0.2-0.5)
)
```

---

## 📖 Understanding the Architecture

### What is ST-GCN?

**Spatial-Temporal Graph Convolutional Network**:
- **Spatial**: Learn relationships between AU nodes (which AUs affect each other)
- **Temporal**: Learn temporal dynamics (how AU activations change over time)
- **Graph**: Face represented as graph where nodes = AU regions, edges = connections

### Why is it good for AU-based emotion recognition?

1. **Structured representation**: Face has inherent structure (brows, eyes, lips, etc.)
2. **Relationship modeling**: AUs don't work alone (e.g., brow + eye for surprise)
3. **Temporal modeling**: Emotions unfold over time (not just single frames)
4. **Learnable edges**: Model learns which AU connections matter (may differ from anatomy)

### Key Components

#### 1. Spatial GCN
```
Message passing over graph:
  H' = σ(D^(-1/2) A D^(-1/2) H W)
  
Where:
  H: Node features [N, F]
  A: Adjacency matrix [N, N]
  D: Degree matrix
  W: Learnable weights
  σ: Activation (ReLU)
```

#### 2. Temporal Conv
```
1D convolution over time for each node:
  H_t = Conv1D(kernel=9)(H)
  
Captures temporal patterns like:
  - Gradual onset (surprise)
  - Quick onset (disgust)
  - Sustained (neutral)
```

#### 3. Edge Importance Weighting
```
Instead of fixed adjacency A, learn:
  A_effective = A ⊙ W_importance
  
Where W_importance is learned per layer.
Allows model to discover important connections.
```

---

## 🔬 Advanced Analysis

### 1. Visualize Learned Edge Weights

After training, see which AU connections the model thinks are important:

```bash
python visualize_edge_importance.py \
    --model stgcn_hybrid_best.pt \
    --model-type hybrid \
    --output-dir edge_viz
```

Output:
- `edge_viz/layer_1_edge_importance.png` - Layer 1 connections
- `edge_viz/layer_2_edge_importance.png` - Layer 2 connections
- `edge_viz/edge_evolution.png` - How edges change across layers
- Text analysis of top connections

### 2. Feature Importance

Which AU features (cx, cy, area, etc.) matter most?

```python
# Add after training in train_stgcn_improved.py

# Get first layer weights
first_layer = list(model.st_gcn_networks[0].gcn.parameters())[0]
feature_importance = first_layer.abs().mean(dim=0)  # [F]

import matplotlib.pyplot as plt
plt.bar(range(10), feature_importance.detach().cpu().numpy())
plt.xticks(range(10), FEATURE_NAMES, rotation=45)
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
```

### 3. Temporal Receptive Field

How far back in time does the model look?

```
SimpleSTGCN: 2 layers × kernel=5 → receptive field ~10 frames
HybridSTGCN: 3 layers × kernel=7,5,3 → receptive field ~15 frames  
ImprovedSTGCN: 5 layers × kernel=9 → receptive field ~45 frames

At 60 FPS:
  - SimpleSTGCN: ~0.17 seconds
  - HybridSTGCN: ~0.25 seconds
  - ImprovedSTGCN: ~0.75 seconds
```

---

## 🐛 Troubleshooting

### Low Accuracy (~25%, random guessing)

**Possible causes:**
1. **Data issue**: Check if labels are correct in `split.csv`
2. **Normalization**: Features may not be normalized
3. **Learning rate**: Try 1e-4 or 5e-4 instead of 1e-3
4. **Model too complex**: Try SimpleSTGCN on small data

**Debug steps:**
```python
# Check data distribution
import pandas as pd
df = pd.read_csv('split.csv')
print(df['label'].value_counts())

# Check if model learns anything
# After 5 epochs, train_acc should be > 30%
# If stuck at 25%, something is wrong
```

### CUDA Out of Memory

**Solutions:**
```python
# 1. Reduce batch size
BATCH_SIZE = 4  # or 2

# 2. Reduce T_fix
T_fix = 200  # instead of median

# 3. Use smaller model
MODEL_TYPE = 'simple'

# 4. Mixed precision training
# Add to train loop:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    logits = model(x, adj)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Overfitting

**Signs:**
- Train acc >> Val acc (e.g., 95% vs 60%)
- Val loss increases while train loss decreases

**Solutions:**
```python
# 1. Increase dropout
dropout=0.5  # or 0.7

# 2. Increase weight decay
WEIGHT_DECAY = 1e-3

# 3. Data augmentation (in dataset_stgcn.py)
# Add random temporal jitter, noise, etc.

# 4. Use simpler model
MODEL_TYPE = 'simple'

# 5. Early stopping (already in improved training)
EARLY_STOP_PATIENCE = 10  # Lower patience
```

---

## 📚 References

### Papers
1. **ST-GCN** (Original):
   ```
   Yan, S., Xiong, Y., & Lin, D. (2018).
   Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.
   AAAI 2018.
   ```

2. **Graph Convolutional Networks**:
   ```
   Kipf, T. N., & Welling, M. (2017).
   Semi-Supervised Classification with Graph Convolutional Networks.
   ICLR 2017.
   ```

3. **Facial Action Units**:
   ```
   Ekman, P., & Friesen, W. V. (1978).
   Facial action coding system: A technique for the measurement of facial movement.
   ```

### GitHub Repos
- Original ST-GCN: https://github.com/yysijie/st-gcn
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric

---

## 💡 Tips & Best Practices

### 1. Start Simple, Then Scale Up
```
Day 1: Train SimpleSTGCN → Baseline (~60% acc)
Day 2: Train HybridSTGCN → Improvement? (~70% acc)
Day 3: Train ImprovedSTGCN → Further improvement? (~75% acc)
Day 4: Tune best model (hyperparameters, augmentation)
```

### 2. Monitor Training Carefully
- **Healthy training**: Train and val loss both decrease
- **Overfitting**: Train loss ↓, val loss ↑
- **Underfitting**: Both losses high and flat
- **Just right**: Small gap between train/val, both converge

### 3. Ensemble for Production
```python
# Train 3 models with different seeds
for seed in [42, 123, 999]:
    torch.manual_seed(seed)
    model = HybridSTGCN(...)
    train(model, ...)
    torch.save(model, f'model_seed{seed}.pt')

# Inference: Average predictions
models = [load_model(f'model_seed{s}.pt') for s in [42,123,999]]
preds = [m(x, A) for m in models]
final_pred = torch.stack(preds).mean(0).argmax(1)
```

### 4. Data Augmentation Ideas
```python
# Temporal augmentation
- Random speed: Sample frames at different rates
- Random crop: Use subsequences instead of full video
- Reverse: Flip time direction (if emotion is symmetric)

# Feature augmentation
- Gaussian noise: Add noise to features
- Dropout: Random masking of AU nodes
- Mixup: Interpolate between samples
```

---

## 🎯 Next Steps

1. **More data**: Collect more samples (most important!)
2. **Transfer learning**: Pre-train on larger emotion dataset
3. **Attention mechanisms**: Add attention to ST-GCN layers
4. **Multi-modal**: Combine with audio, text, physiological signals
5. **Deployment**: Convert to ONNX for faster inference

---

## 📞 Support

If you encounter issues:
1. Check MODEL_COMPARISON.md for detailed explanations
2. Run benchmark_models.py to verify setup
3. Check data shapes and distributions
4. Try SimpleSTGCN first as baseline

Good luck! 🚀
