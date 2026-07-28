#!/usr/bin/env python3
"""Create GIF animation showing all frames of the graph"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
from PIL import Image
import io

def create_gif_animation(npz_path, output_dir="animations", fps=10, step=1):
    """Create GIF animation of graph over time
    
    Args:
        npz_path: Path to NPZ file
        output_dir: Output directory for GIF
        fps: Frames per second in output GIF
        step: Frame step (1=all frames, 2=every other frame, etc.)
    """
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
    print(f"Creating animation with step={step} (will generate {T//step} frames)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    base_name = Path(npz_path).stem
    
    # Get features
    feat_names = meta.get('feature_names', [])
    cx_idx = feat_names.index('cx') if 'cx' in feat_names else 0
    cy_idx = feat_names.index('cy') if 'cy' in feat_names else 1
    area_idx = feat_names.index('area') if 'area' in feat_names else -1
    
    # Get node labels
    labels = meta.get('au_nodes', [f'Node{i}' for i in range(N)])
    
    # Prepare frames
    frames = []
    frame_indices = range(0, T, step)
    
    for i, t in enumerate(frame_indices):
        if i % 50 == 0:
            print(f"  Rendering frame {i+1}/{len(frame_indices)}...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cx = X[t, :, cx_idx]
        cy = X[t, :, cy_idx]
        
        # Size based on area if available
        if area_idx != -1:
            sizes = X[t, :, area_idx]
            sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-6)
            sizes = 50 + 200 * sizes
        else:
            sizes = 100
        
        # Draw edges first (in background)
        for ni in range(N):
            for nj in range(ni+1, N):
                if A[ni, nj] > 0 or A[nj, ni] > 0:
                    ax.plot([cx[ni], cx[nj]], [cy[ni], cy[nj]], 
                           'gray', alpha=0.4, linewidth=1.5, zorder=1)
        
        # Draw nodes
        scatter = ax.scatter(cx, cy, s=sizes, alpha=0.8, 
                           c=range(N), cmap='tab20', 
                           edgecolors='black', linewidth=1.5, zorder=2)
        
        # Add labels
        for j, label in enumerate(labels):
            ax.annotate(label, (cx[j], cy[j]), 
                       fontsize=8, fontweight='bold',
                       ha='center', va='center',
                       color='white', zorder=3)
        
        ax.set_xlim(cx.min() - 0.1, cx.max() + 0.1)
        ax.set_ylim(cy.min() - 0.1, cy.max() + 0.1)
        ax.invert_yaxis()
        ax.set_xlabel('cx', fontsize=12)
        ax.set_ylabel('cy', fontsize=12)
        ax.set_title(f'{base_name} - Frame {t}/{T-1} ({t/meta.get("fps", 30):.2f}s)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Convert to image
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close()
        buf.close()
    
    # Save as GIF
    gif_path = output_path / f"{base_name}_animation.gif"
    print(f"\nSaving GIF with {len(frames)} frames at {fps} fps...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000//fps,  # milliseconds per frame
        loop=0,
        optimize=True
    )
    
    print(f"✓ Saved: {gif_path} ({gif_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Duration: {len(frames)/fps:.1f} seconds")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python create_animation.py <npz_file> [fps] [step]")
        print("  fps: frames per second (default=10)")
        print("  step: frame step, 1=all frames, 2=every other (default=2)")
        print("\nExample:")
        print("  python create_animation.py outputb2/P001/P001_213.npz 15 3")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    fps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    create_gif_animation(npz_file, fps=fps, step=step)
