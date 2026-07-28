#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization tool for NPZ output from b2_video_normalization.py
Displays temporal features of AU nodes over time
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from matplotlib.gridspec import GridSpec

def load_npz_data(npz_path: Path):
    """Load and parse NPZ file"""
    data = np.load(npz_path, allow_pickle=True)
    
    graph_seq = data['graph_seq']  # [T, N, F]
    adj = data['adj']  # [N, N]
    meta = json.loads(data['meta'].item())
    
    return graph_seq, adj, meta

def visualize_au_features(graph_seq, meta, save_path=None):
    """
    Visualize all AU node features over time
    graph_seq: [T, N, F] - T frames, N nodes, F features
    """
    T, N, F = graph_seq.shape
    au_nodes = meta['au_nodes']
    feature_names = meta['feature_names']
    fps = meta['fps']
    
    # Create time axis in seconds
    time = np.arange(T) / fps
    
    # Create figure with subplots for different feature groups
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Position features (cx, cy)
    ax1 = fig.add_subplot(gs[0, :])
    for n in range(N):
        ax1.plot(time, graph_seq[:, n, 0], label=au_nodes[n], alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (cx)')
    ax1.set_title(f'AU Nodes - X Position Over Time\n{meta["subject_id"]}_{meta["ma_mau"]} | FPS: {fps:.2f}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Y Position
    ax2 = fig.add_subplot(gs[1, 0])
    for n in range(N):
        ax2.plot(time, graph_seq[:, n, 1], label=au_nodes[n], alpha=0.7, linewidth=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (cy)')
    ax2.set_title('Y Position Over Time')
    ax2.grid(True, alpha=0.3)
    
    # 3. Z Position (depth)
    ax3 = fig.add_subplot(gs[1, 1])
    for n in range(N):
        ax3.plot(time, graph_seq[:, n, 2], label=au_nodes[n], alpha=0.7, linewidth=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (cz)')
    ax3.set_title('Z Position (Depth) Over Time')
    ax3.grid(True, alpha=0.3)
    
    # 4. Visibility
    ax4 = fig.add_subplot(gs[2, 0])
    for n in range(N):
        ax4.plot(time, graph_seq[:, n, 3], label=au_nodes[n], alpha=0.7, linewidth=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Visibility')
    ax4.set_title('Visibility Over Time')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3)
    
    # 5. Area
    ax5 = fig.add_subplot(gs[2, 1])
    for n in range(N):
        ax5.plot(time, graph_seq[:, n, 4], label=au_nodes[n], alpha=0.7, linewidth=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Area')
    ax5.set_title('Convex Hull Area Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 6. Aspect Ratio
    ax6 = fig.add_subplot(gs[3, 0])
    for n in range(N):
        ax6.plot(time, graph_seq[:, n, 5], label=au_nodes[n], alpha=0.7, linewidth=0.8)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Aspect Ratio')
    ax6.set_title('Aspect Ratio Over Time')
    ax6.grid(True, alpha=0.3)
    
    # 7. Temporal features (deltas)
    ax7 = fig.add_subplot(gs[3, 1])
    # Show mean absolute delta for each node
    mean_delta_cx = np.mean(np.abs(graph_seq[:, :, 6]), axis=0)
    mean_delta_cy = np.mean(np.abs(graph_seq[:, :, 7]), axis=0)
    x_pos = np.arange(N)
    width = 0.35
    ax7.bar(x_pos - width/2, mean_delta_cx, width, label='|Δcx|', alpha=0.8)
    ax7.bar(x_pos + width/2, mean_delta_cy, width, label='|Δcy|', alpha=0.8)
    ax7.set_xlabel('AU Node')
    ax7.set_ylabel('Mean Absolute Delta')
    ax7.set_title('Mean Movement Magnitude per Node')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([au[:10] for au in au_nodes], rotation=45, ha='right', fontsize=8)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✔ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_adjacency_matrix(adj, au_nodes, save_path=None):
    """Visualize the adjacency matrix"""
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(adj, cmap='Blues', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(au_nodes)))
    ax.set_yticks(np.arange(len(au_nodes)))
    ax.set_xticklabels(au_nodes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(au_nodes, fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Connection', rotation=270, labelpad=15)
    
    # Add title
    ax.set_title('AU Nodes Adjacency Matrix\n(Graph Connectivity)', fontsize=12, pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(len(au_nodes))-.5, minor=True)
    ax.set_yticks(np.arange(len(au_nodes))-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✔ Saved adjacency matrix to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_selected_nodes(graph_seq, meta, selected_nodes, save_path=None):
    """
    Visualize specific AU nodes with all their features
    """
    T, N, F = graph_seq.shape
    au_nodes = meta['au_nodes']
    feature_names = meta['feature_names']
    fps = meta['fps']
    time = np.arange(T) / fps
    
    # Get indices for selected nodes
    node_indices = [au_nodes.index(node) for node in selected_nodes if node in au_nodes]
    
    if not node_indices:
        print("Warning: None of the selected nodes found in AU_NODES")
        return
    
    # Create subplots for each feature
    fig, axes = plt.subplots(5, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for f_idx, f_name in enumerate(feature_names):
        ax = axes[f_idx]
        for n_idx in node_indices:
            ax.plot(time, graph_seq[:, n_idx, f_idx], 
                   label=au_nodes[n_idx], alpha=0.8, linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f_name)
        ax.set_title(f'Feature: {f_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Selected AU Nodes Features\n{meta["subject_id"]}_{meta["ma_mau"]}', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✔ Saved selected nodes visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_statistics(graph_seq, meta):
    """Print basic statistics about the data"""
    T, N, F = graph_seq.shape
    au_nodes = meta['au_nodes']
    feature_names = meta['feature_names']
    
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Subject ID: {meta['subject_id']}")
    print(f"Sample Code: {meta['ma_mau']}")
    print(f"Video Path: {meta['video_path']}")
    print(f"FPS: {meta['fps']:.2f}")
    print(f"Total Frames: {T}")
    print(f"Duration: {T/meta['fps']:.2f} seconds")
    print(f"Number of AU Nodes: {N}")
    print(f"Number of Features: {F}")
    print(f"\nAU Nodes: {', '.join(au_nodes)}")
    print(f"\nFeatures: {', '.join(feature_names)}")
    
    # Calculate some statistics
    print("\n" + "-"*60)
    print("FEATURE STATISTICS (mean across all nodes and frames)")
    print("-"*60)
    for f_idx, f_name in enumerate(feature_names):
        mean_val = np.mean(graph_seq[:, :, f_idx])
        std_val = np.std(graph_seq[:, :, f_idx])
        min_val = np.min(graph_seq[:, :, f_idx])
        max_val = np.max(graph_seq[:, :, f_idx])
        print(f"{f_name:12s}: mean={mean_val:8.4f}, std={std_val:8.4f}, "
              f"min={min_val:8.4f}, max={max_val:8.4f}")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Visualize NPZ output from face normalization")
    parser.add_argument("--input", type=str, required=True, 
                       help="Path to NPZ file to visualize")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save visualizations (if not specified, will display)")
    parser.add_argument("--selected-nodes", type=str, nargs="+", default=None,
                       help="Specific AU nodes to visualize in detail (e.g., 'upper_lip' 'lower_lip')")
    parser.add_argument("--show-stats", action="store_true",
                       help="Print detailed statistics")
    
    args = parser.parse_args()
    
    # Load data
    npz_path = Path(args.input)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    print(f"Loading data from: {npz_path}")
    graph_seq, adj, meta = load_npz_data(npz_path)
    
    # Print statistics if requested
    if args.show_stats:
        print_statistics(graph_seq, meta)
    
    # Prepare output paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = npz_path.stem
        features_path = output_dir / f"{base_name}_features.png"
        adj_path = output_dir / f"{base_name}_adjacency.png"
    else:
        features_path = None
        adj_path = None
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. All features
    visualize_au_features(graph_seq, meta, features_path)
    
    # 2. Adjacency matrix
    visualize_adjacency_matrix(adj, meta['au_nodes'], adj_path)
    
    # 3. Selected nodes (if specified)
    if args.selected_nodes:
        if args.output_dir:
            selected_path = output_dir / f"{base_name}_selected_nodes.png"
        else:
            selected_path = None
        visualize_selected_nodes(graph_seq, meta, args.selected_nodes, selected_path)
    
    print("\n✔ Visualization complete!")

if __name__ == "__main__":
    main()




# # Xem tất cả features + in thống kê
# python3 visualize_npz.py --input outputb2/P001/P001_213.npz --output-dir outputb2/P001/visualizations --show-stats

# # Xem chi tiết các node cụ thể
# python3 visualize_npz.py --input outputb2/P001/P001_213.npz --output-dir outputb2/P001/visualizations --selected-nodes upper_lip lower_lip eye_left_upper eye_right_upper

# # Hiển thị trực tiếp (không lưu file)
# python3 visualize_npz.py --input outputb2/P001/P001_213.npz --show-stats