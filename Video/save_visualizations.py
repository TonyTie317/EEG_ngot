#!/usr/bin/env python3
"""Save visualizations to PNG files instead of showing"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import json
from pathlib import Path

def visualize_npz(npz_path, output_dir="visualizations"):
    """Create and save visualizations for NPZ file"""
    print(f"Loading {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    
    X = data["graph_seq"]  # [T, N, F]
    A = data["adj"]        # [N, N]
    meta_raw = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
    if isinstance(meta_raw, (str, bytes)):
        meta = json.loads(meta_raw)
    else:
        meta = dict(meta_raw)
    
    T, N, F = X.shape
    print(f"Data: T={T} frames, N={N} nodes, F={F} features")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    base_name = Path(npz_path).stem
    
    # 1. Adjacency matrix heatmap
    print("Creating adjacency heatmap...")
    plt.figure(figsize=(10, 8))
    im = plt.imshow(A, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Edge Weight')
    plt.title(f'Adjacency Matrix - {base_name}')
    plt.xlabel('Node j')
    plt.ylabel('Node i')
    
    # Add node labels if available
    if 'au_nodes' in meta:
        labels = meta['au_nodes']
        plt.xticks(range(N), labels, rotation=45, ha='right', fontsize=8)
        plt.yticks(range(N), labels, fontsize=8)
    
    plt.tight_layout()
    adj_file = output_path / f"{base_name}_adjacency.png"
    plt.savefig(adj_file, dpi=150)
    print(f"✓ Saved: {adj_file}")
    plt.close()
    
    # 2. Feature statistics over time
    print("Creating feature statistics...")
    feat_names = meta.get('feature_names', [f'f{i}' for i in range(F)])
    
    # Plot first 6 features for first 3 nodes
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for feat_idx in range(min(6, F)):
        ax = axes[feat_idx]
        for node_idx in range(min(3, N)):
            ax.plot(X[:, node_idx, feat_idx], label=f'Node {node_idx}', alpha=0.7)
        ax.set_title(f'Feature: {feat_names[feat_idx]}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Feature Time Series - {base_name}', fontsize=14)
    plt.tight_layout()
    feat_file = output_path / f"{base_name}_features.png"
    plt.savefig(feat_file, dpi=150)
    print(f"✓ Saved: {feat_file}")
    plt.close()
    
    # 3. Node positions snapshot (cx, cy at multiple timepoints)
    print("Creating node position snapshots...")
    feat_names = meta.get('feature_names', [])
    cx_idx = feat_names.index('cx') if 'cx' in feat_names else 0
    cy_idx = feat_names.index('cy') if 'cy' in feat_names else 1
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    timepoints = [0, T//5, 2*T//5, 3*T//5, 4*T//5, T-1]
    for i, t in enumerate(timepoints):
        ax = axes[i]
        cx = X[t, :, cx_idx]
        cy = X[t, :, cy_idx]
        
        # Plot nodes
        ax.scatter(cx, cy, s=100, alpha=0.8, c=range(N), cmap='tab20')
        
        # Add labels
        if 'au_nodes' in meta:
            for j, label in enumerate(meta['au_nodes']):
                ax.annotate(label, (cx[j], cy[j]), fontsize=7, 
                           xytext=(3, 3), textcoords='offset points')
        
        # Draw edges
        for ni in range(N):
            for nj in range(ni+1, N):
                if A[ni, nj] > 0 or A[nj, ni] > 0:
                    ax.plot([cx[ni], cx[nj]], [cy[ni], cy[nj]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_title(f'Frame {t}')
        ax.set_xlabel('cx')
        ax.set_ylabel('cy')
        ax.invert_yaxis()  # Flip Y to match image coordinates
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Node Positions Over Time - {base_name}', fontsize=14)
    plt.tight_layout()
    pos_file = output_path / f"{base_name}_positions.png"
    plt.savefig(pos_file, dpi=150)
    print(f"✓ Saved: {pos_file}")
    plt.close()
    
    print(f"\n✓ All visualizations saved to {output_path}/")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python save_visualizations.py <npz_file> [output_dir]")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations"
    
    visualize_npz(npz_file, output_dir)
