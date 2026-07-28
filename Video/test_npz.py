#!/usr/bin/env python3
import numpy as np
import json

npz_path = "outputb2/P001/P001_213.npz"
print(f"Loading {npz_path}...")

data = np.load(npz_path, allow_pickle=True)

print("\n=== Files in NPZ ===")
print(data.files)

print("\n=== Shapes ===")
for key in data.files:
    print(f"{key}: {data[key].shape if hasattr(data[key], 'shape') else type(data[key])}")

if "graph_seq" in data.files:
    X = data["graph_seq"]
    print(f"\ngraph_seq shape: {X.shape}")
    print(f"Min/Max values: {X.min():.4f} / {X.max():.4f}")

if "adj" in data.files:
    A = data["adj"]
    print(f"\nadj shape: {A.shape}")
    print(f"Non-zero edges: {(A > 0).sum()}")

if "meta" in data.files:
    meta_raw = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
    if isinstance(meta_raw, (str, bytes)):
        meta = json.loads(meta_raw)
    else:
        meta = dict(meta_raw)
    print(f"\n=== Meta ===")
    for k, v in meta.items():
        if isinstance(v, (list, dict)) and len(str(v)) > 100:
            print(f"{k}: {type(v)} (length={len(v)})")
        else:
            print(f"{k}: {v}")

if "preview" in data.files:
    preview = data["preview"]
    print(f"\n=== Preview ===")
    print(f"Shape: {preview.shape}")
    print(f"Dtype: {preview.dtype}")
else:
    print("\n⚠️  No 'preview' found in NPZ")

print("\n✓ File loaded successfully!")
