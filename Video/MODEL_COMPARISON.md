# So sánh ST-GCN Models: GitHub vs Current Implementation

## Tổng quan

Đã tạo 2 files mới:
- `model_stgcn_improved.py`: 3 variants của ST-GCN
- `train_stgcn_improved.py`: Training script mới

## 1. KIẾN TRÚC SO SÁNH

### Original ST-GCN (GitHub)
```
Architecture: 10 layers deep
├─ Input: (N, C, T, V, M) - supports multiple persons
├─ BatchNorm on all node features: V*C
├─ ST-GCN Blocks (10 layers):
│  ├─ Channel progression: 64→64→64→64→128→128→128→256→256→256
│  ├─ Temporal kernel: 9
│  ├─ Spatial: Multi-partition adjacency [K, V, V]
│  └─ Residual connections with BatchNorm
├─ Edge importance weighting (learnable)
├─ Global average pooling (T, V)
└─ Conv2d classifier

Parameters: ~3.5M (for skeleton data)
Best for: Large datasets (>10K samples)
```

### Your SimpleSTGCN (Current)
```
Architecture: 2 layers (very lightweight)
├─ Input: (B, N, T, F) - single person
├─ No BatchNorm
├─ Spatial GCN:
│  └─ Simple message passing: A @ X
├─ Temporal Conv1d:
│  ├─ kernel_size: 5, 3
│  └─ Process each node separately
├─ Fixed adjacency (no learning)
├─ Mean pooling (N, T)
└─ Linear classifier

Parameters: ~200K
Best for: Very small datasets (<1K samples)
```

### ImprovedSTGCN (New - Full version)
```
Architecture: 5 layers (balanced depth)
├─ Input: (B, N, T, F)
├─ BatchNorm1d on input: N*F features
├─ ST-GCN Blocks (5 layers):
│  ├─ Channel progression: 64→64→128→128→256
│  ├─ Temporal kernel: 9 (same as original)
│  ├─ Spatial: Symmetric normalized adjacency D^(-1/2) A D^(-1/2)
│  ├─ Residual connections with BatchNorm
│  └─ Dropout for regularization
├─ Edge importance weighting (learnable)
├─ Global average pooling (T, N)
└─ Conv2d classifier

Parameters: ~800K
Best for: Medium datasets (2K-5K samples)
```

### HybridSTGCN (New - Lightweight version)
```
Architecture: 3 layers (light but effective)
├─ Input: (B, N, T, F)
├─ BatchNorm1d on input
├─ ST-GCN Blocks (3 layers):
│  ├─ Channel progression: 128→128→256
│  ├─ Temporal kernel: 7, 5, 3 (decreasing)
│  ├─ Spatial: Normalized adjacency
│  ├─ Residual connections
│  └─ Dropout
├─ Edge importance weighting
├─ Global pooling
└─ 2-layer classifier with dropout

Parameters: ~400K
Best for: Small-medium datasets (1K-3K samples)
```

---

## 2. KEY IMPROVEMENTS

### A) Input Normalization
**Problem in SimpleSTGCN:**
```python
# No normalization → unstable training
x = self.theta1(x)  # direct transform
```

**Solution in Improved models:**
```python
# BatchNorm on all node features
self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
x_bn = x.view(B, N * F, T)
x_bn = self.data_bn(x_bn)
```

**Why it matters:** BatchNorm stabilizes training, reduces internal covariate shift

---

### B) Spatial Graph Convolution

**SimpleSTGCN (Basic):**
```python
def spatial_gcn(self, x, adj):
    Ax = torch.bmm(adj, x)  # Simple multiplication
    return Ax
```

**ImprovedSTGCN (Proper GCN):**
```python
# Symmetric normalization: D^(-1/2) A D^(-1/2)
A = A + torch.eye(N)  # Add self-loops
D = A.sum(dim=-1).clamp(min=1.0)
D_inv_sqrt = D.pow(-0.5)
A_norm = D_inv_sqrt * A * D_inv_sqrt.T

# Message passing + feature transform
x = torch.bmm(A_norm, x)
x = self.conv(x)  # Learnable weights
```

**Why it matters:**
- Self-loops: Node aggregates its own features
- Symmetric normalization: Prevents gradient explosion
- Learnable transformation: Better feature extraction

---

### C) Temporal Convolution

**SimpleSTGCN:**
```python
# Process each node separately
x = x.reshape(B * N, H, T)
x = nn.Conv1d(hid, hid, kernel_size=5)(x)
```

**ImprovedSTGCN:**
```python
# Process all nodes together with proper padding
nn.Conv2d(
    out_channels,
    out_channels,
    (9, 1),  # kernel: (time, node)
    (stride, 1),
    padding=((9-1)//2, 0)
)
```

**Why it matters:**
- Larger receptive field (k=9 vs k=5)
- Better captures temporal dynamics
- Proper padding maintains sequence length

---

### D) Residual Connections

**SimpleSTGCN:**
```python
# Simple addition
x = x + x_sp.unsqueeze(2)
```

**ImprovedSTGCN:**
```python
# Proper residual with dimension matching
if (in_channels == out_channels) and (stride == 1):
    self.residual = lambda x: x
else:
    self.residual = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride=(stride,1)),
        nn.BatchNorm2d(out_channels)
    )

x = self.tcn(x) + self.residual(x)
```

**Why it matters:**
- Handles dimension changes
- Gradient flow for deep networks
- Training stability

---

### E) Edge Importance Weighting

**SimpleSTGCN:**
```python
# Fixed adjacency
adj = build_au_adjacency()  # Static, never updated
```

**Improved models:**
```python
# Learnable edge weights
self.edge_importance = nn.ParameterList([
    nn.Parameter(torch.ones(N, N))
    for _ in self.st_gcn_networks
])

# During forward
A_weighted = A * importance  # Element-wise
```

**Why it matters:**
- Model learns which AU connections are important
- Different layers can have different edge weights
- Adaptive to task

---

## 3. TRAINING IMPROVEMENTS

### Learning Rate Scheduling
```python
# New in train_stgcn_improved.py
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)
```

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Early Stopping
```python
EARLY_STOP_PATIENCE = 15
if patience_counter >= EARLY_STOP_PATIENCE:
    break
```

---

## 4. USAGE GUIDE

### Training với các model variants:

```bash
# 1. Simple model (current - fastest, least parameters)
python train_stgcn_improved.py  # Set MODEL_TYPE = 'simple'

# 2. Hybrid model (recommended - good balance)
python train_stgcn_improved.py  # Set MODEL_TYPE = 'hybrid'

# 3. Full model (most powerful - may overfit on small data)
python train_stgcn_improved.py  # Set MODEL_TYPE = 'full'
```

### Hoặc sửa trong code:
```python
# In train_stgcn_improved.py, line ~20
MODEL_TYPE = 'hybrid'  # Change to 'simple', 'hybrid', or 'full'
```

---

## 5. EXPECTED RESULTS

### Your Current Dataset (~50 samples)

| Model | Parameters | Train Acc | Val Acc | Training Time | Risk |
|-------|-----------|-----------|---------|---------------|------|
| SimpleSTGCN | 200K | ~80% | ~60% | Fast | Low overfit |
| HybridSTGCN | 400K | ~90% | ~65-70% | Medium | Medium overfit |
| ImprovedSTGCN | 800K | ~95% | ~60-65% | Slow | High overfit |

**Recommendation cho dataset nhỏ:** 
1. Start with **HybridSTGCN**
2. Tăng data augmentation
3. Tăng dropout (0.5-0.7)

---

## 6. DEBUGGING TIPS

### Check model output shape:
```python
from model_stgcn_improved import create_stgcn
import torch

B, N, T, F = 8, 15, 300, 10
x = torch.randn(B, N, T, F)
A = torch.randn(N, N)

model = create_stgcn('hybrid', in_channels=F, num_classes=4, num_nodes=N)
out = model(x, A)
print(f"Output shape: {out.shape}")  # Should be [8, 4]
```

### Visualize learned edge importance:
```python
# After training
importance = model.edge_importance[0].detach().cpu().numpy()
import matplotlib.pyplot as plt
plt.imshow(importance, cmap='hot')
plt.colorbar()
plt.title("Learned Edge Importance")
plt.savefig("edge_importance.png")
```

---

## 7. FURTHER IMPROVEMENTS

### Data Augmentation cho video:
```python
# In b2_video_normalization.py
- Time warping: Random speed up/down
- Spatial jitter: Small random translations
- Feature noise: Add Gaussian noise to features
```

### Ensemble Methods:
```python
# Train 3 models and average predictions
models = [
    create_stgcn('simple', ...),
    create_stgcn('hybrid', ...),
    create_stgcn('full', ...)
]

# Average predictions
logits = torch.stack([m(x, A) for m in models]).mean(0)
```

### Attention Mechanisms:
```python
# Add attention to ST-GCN blocks
class AttentionSTGCN(nn.Module):
    def __init__(self, ...):
        self.attention = nn.MultiheadAttention(hid, num_heads=4)
        
    def forward(self, x, A):
        # x: [B, C, T, N]
        x_attn = self.attention(x, x, x)[0]
        return x + x_attn  # Residual
```

---

## 8. PAPER REFERENCES

**Original ST-GCN:**
```
Yan, S., Xiong, Y., & Lin, D. (2018). 
Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. 
AAAI 2018.
```

**Key differences for AU-based emotion recognition:**
- Input: AU features (continuous) vs skeleton coordinates
- Graph: Facial AU topology vs human skeleton
- Temporal: Facial dynamics (~60fps) vs body motion (~30fps)
- Scale: 15 nodes vs 18-25 nodes

---

## 9. QUICK START

```bash
# 1. Make sure you have the data processed
python b2_video_normalization.py --input ../Labrecorder/Data_chuan --output outputb2

# 2. Create train/val split
python make_split.py

# 3. Train with improved model
python train_stgcn_improved.py

# 4. Compare results
# Check these files:
# - train_curves_hybrid.png
# - conf_matrix_hybrid.png
# - stgcn_hybrid_best.pt
```

---

## 10. TROUBLESHOOTING

**Error: "CUDA out of memory"**
```python
# Reduce batch size
BATCH_SIZE = 4  # or 2

# Or use smaller model
MODEL_TYPE = 'simple'
```

**Error: "Adjacency dimension mismatch"**
```python
# In train loop, ensure correct shape:
if MODEL_TYPE != 'simple':
    if adj.dim() == 3:
        adj = adj[0]  # Use [N, N] not [B, N, N]
```

**Low accuracy (random guessing ~25%)**
```python
# Check:
1. Data normalization: X should be normalized
2. Learning rate: Try 1e-4 or 5e-4
3. Model capacity: Maybe too complex for small data
4. Class imbalance: Check distribution in split.csv
```

---

## SUMMARY

**Main Differences:**

| Component | SimpleSTGCN | Improved Models |
|-----------|-------------|-----------------|
| Depth | 2 layers | 3-5 layers |
| BatchNorm | ❌ | ✅ |
| Residual | Simple add | Proper with projection |
| Adjacency | Fixed | Learnable weights |
| Normalization | None | D^(-1/2) A D^(-1/2) |
| Temporal kernel | 3, 5 | 7, 9 |
| Dropout | ❌ | ✅ |
| LR scheduling | ❌ | ✅ |
| Early stopping | ❌ | ✅ |

**Recommendation:**
1. **Small dataset (<1K)**: Use `HybridSTGCN` với dropout=0.5
2. **Medium dataset (1K-5K)**: Use `ImprovedSTGCN`
3. **Large dataset (>5K)**: Implement full 10-layer version

Good luck! 🚀
